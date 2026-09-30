// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Adaptive nprobes for the shared-scan batch IVF search ([`super::ANNIvfBatchExec`]).
//!
//! The single-query path ([`super::ANNIvfSubIndexExec`]) probes adaptively:
//!
//! 1. every delta ranks the partitions and [`AutoProbePolicy`] raises the
//!    initial budget (`minimum_nprobes`) from the centroid distances;
//! 2. the *early* search probes that budget in every delta;
//! 3. once all deltas finish, a query that found fewer than `k` rows either
//!    returns the unseen prefilter rows directly (when fewer than `k` rows can
//!    match at all) or runs a *late* search that probes further partitions in
//!    rank order, one at a time, until it has found `min(k, rows allowed by the
//!    prefilter)` rows or reaches `maximum_nprobes`.
//!
//! This module reproduces those rules for a batch of queries while sharing the
//! partition loads across queries. The early search is one shared scan per
//! delta. The late search runs in rounds: each still-active query contributes
//! its next few partitions (a *wave*), every distinct partition in the round is
//! loaded once, and each query then consumes its wave in rank order and stops
//! at the same partition the sequential single-query search would stop at. Rows
//! from partitions after that cut are discarded so results stay identical;
//! the wave width doubles per round to bound both round count and wasted work.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt32Array, cast::AsArray};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_core::ROW_ID;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_index::metrics::MetricsCollector;
use lance_index::prefilter::PreFilter;
use lance_index::vector::{DIST_COL, Query, VectorIndex};
use lance_select::RowAddrMask;
use roaring::RoaringBitmap;

use super::adaptive_probe::AutoProbePolicy;
use super::{AnnIndexMetrics, restrict_to_segment};
use crate::index::prefilter::DatasetPreFilter;

/// One query's probe plan against one index delta.
pub(super) struct QueryProbePlan {
    /// Partition ids ranked by centroid distance (at most `maximum_nprobes`).
    partitions: Arc<UInt32Array>,
    /// Query-to-centroid distance for each entry of `partitions`.
    q_c_dists: Arc<Float32Array>,
    /// Number of leading partitions probed by the early search.
    early_end: usize,
    /// Exclusive end of the late-search range `[early_end, late_end)`. Equal to
    /// `early_end` when late search cannot expand the probe set.
    late_end: usize,
}

impl QueryProbePlan {
    pub(super) fn early_partitions(&self) -> (Arc<UInt32Array>, Arc<Float32Array>) {
        (
            Arc::new(self.partitions.slice(0, self.early_end)),
            Arc::new(self.q_c_dists.slice(0, self.early_end)),
        )
    }

    fn has_late_range(&self) -> bool {
        self.late_end > self.early_end
    }
}

/// A searched index delta: the opened index, the batch query normalized for
/// it, one probe plan per query (in query order), and its segment ownership.
pub(super) struct BatchDelta {
    pub(super) index: Arc<dyn VectorIndex>,
    pub(super) query: Query,
    pub(super) plans: Vec<QueryProbePlan>,
    /// The shared prefilter restricted to the fragments this segment owns.
    pub(super) pre_filter: Arc<dyn PreFilter>,
    /// Rows this segment owns, when a per-segment restriction is in effect.
    pub(super) seg_mask: Option<Arc<RowAddrMask>>,
}

impl BatchDelta {
    /// Whether this delta joins query `query_index`'s late search. Like the
    /// single-query path, a segment whose fragments all moved to newer deltas can
    /// produce no row, so it is skipped rather than probed to `maximum_nprobes`.
    fn joins_late_search(&self, query_index: usize) -> bool {
        self.plans[query_index].has_late_range()
            && !self
                .seg_mask
                .as_ref()
                .is_some_and(|mask| mask.max_len() == Some(0))
    }
}

/// Rank every query of a batch against the IVF centroids of `index` and apply
/// the same initial-budget policy the single-query path applies per delta.
///
/// `query.key` holds `query_count` vectors of width `dim`, already normalized
/// for `index`. Ranking and the policy are pure CPU work that the batch width
/// multiplies, so the whole loop runs as one `spawn_cpu` job.
pub(super) async fn plan_batch_probes(
    index: Arc<dyn VectorIndex>,
    query: Query,
    query_count: usize,
    dim: usize,
    vector_type: DataType,
) -> DataFusionResult<Vec<QueryProbePlan>> {
    spawn_cpu(move || -> DataFusionResult<Vec<QueryProbePlan>> {
        let metric = index.metric_type();
        let mut plans = Vec::with_capacity(query_count);
        for query_index in 0..query_count {
            let mut single_query = query.clone();
            single_query.key = query.key.slice(query_index * dim, dim);
            // The scanner rejects `nprobes(0)` for the batch path (the single-query
            // path probes nothing for it), so the budget is always positive here.
            debug_assert!(
                single_query.minimum_nprobes > 0,
                "batch node reached with minimum_nprobes == 0; the scanner gate should have fallen back"
            );
            let (partitions, q_c_dists) = index.find_partitions(&single_query).map_err(|e| {
                DataFusionError::Execution(format!("Failed to find partitions: {e}"))
            })?;
            let policy = AutoProbePolicy::from_env(&single_query, index.as_ref(), &vector_type)?;
            policy.apply(&mut single_query, q_c_dists.values(), metric);

            // Same clamps as the single-query early and late searches.
            let available = partitions.len();
            let early_end = single_query.minimum_nprobes.min(available);
            let late_end = single_query
                .maximum_nprobes
                .unwrap_or(available)
                .min(available)
                .max(early_end);
            plans.push(QueryProbePlan {
                partitions: Arc::new(partitions),
                q_c_dists: Arc::new(q_c_dists),
                early_end,
                late_end,
            });
        }
        Ok(plans)
    })
    .await
}

/// Append a `search_partitions_batch` result batch to a query's candidates and
/// return how many rows it held.
pub(super) fn append_candidates(
    batch: &RecordBatch,
    candidates: &mut Vec<(f32, u64)>,
) -> DataFusionResult<usize> {
    let dists = batch
        .column_by_name(DIST_COL)
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "batch partition search result missing '{DIST_COL}' column"
            ))
        })?
        .as_primitive::<Float32Type>();
    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "batch partition search result missing '{ROW_ID}' column"
            ))
        })?
        .as_primitive::<UInt64Type>();
    candidates.extend(
        dists
            .values()
            .iter()
            .copied()
            .zip(row_ids.values().iter().copied()),
    );
    Ok(batch.num_rows())
}

/// Run `search_partitions_batch`, check it returned one batch per query, and
/// drop rows the segment does not own (see [`super::restrict_to_segment`]).
pub(super) async fn search_batch(
    index: Arc<dyn VectorIndex>,
    query: Query,
    partitions: Vec<Arc<UInt32Array>>,
    q_c_dists: Vec<Arc<Float32Array>>,
    pre_filter: Arc<dyn PreFilter>,
    seg_mask: Option<&RowAddrMask>,
    metrics: &AnnIndexMetrics,
) -> DataFusionResult<Vec<RecordBatch>> {
    let query_count = partitions.len();
    let distinct_partitions: RoaringBitmap = partitions
        .iter()
        .flat_map(|parts| parts.values().iter().copied())
        .collect();
    // The batch search loads each distinct partition once for every query that
    // probes it, so the union is the honest "partitions searched" count.
    metrics
        .partitions_searched
        .add(distinct_partitions.len() as usize);
    let index_metrics: Arc<dyn MetricsCollector> = Arc::new(metrics.index_metrics.clone());
    let per_query = index
        .search_partitions_batch(query, partitions, q_c_dists, pre_filter, index_metrics)
        .await?;
    // A mismatch would silently drop or misattribute results.
    if per_query.len() != query_count {
        return Err(DataFusionError::Internal(format!(
            "batch partition search returned {} result batches for {query_count} queries",
            per_query.len()
        )));
    }
    per_query
        .into_iter()
        .map(|batch| restrict_to_segment(batch, seg_mask))
        .collect()
}

/// Upper bound on (query, partition) probes issued in one late-search round.
const MAX_LATE_PROBES_PER_ROUND: usize = 1024;

/// Adaptive completion after the early search of every delta.
///
/// `candidates[i]` holds query `i`'s early-search rows from all deltas and is
/// extended in place with its late-search rows (or the unseen prefilter rows at
/// `+inf` distance when fewer than `k` rows can match). Every round advances all
/// deltas together and they share each query's found count, so a query stops as
/// soon as its budget is met.
pub(super) async fn late_search(
    deltas: &[BatchDelta],
    dim: usize,
    k: usize,
    pre_filter: Arc<DatasetPreFilter>,
    candidates: &mut [Vec<(f32, u64)>],
    metrics: &AnnIndexMetrics,
) -> DataFusionResult<()> {
    let query_count = candidates.len();
    let needs_late: Vec<bool> = (0..query_count)
        .map(|query_index| {
            deltas
                .iter()
                .any(|delta| delta.joins_late_search(query_index))
        })
        .collect();
    if !needs_late.iter().any(|&needs| needs) {
        return Ok(());
    }

    // The early search already waited on the prefilter; this is a no-op that
    // makes the dependency explicit before reading the mask.
    pre_filter.wait_for_ready().await?;
    let prefilter_mask = pre_filter.mask();
    let max_results = prefilter_mask.max_len().map(|len| len as usize);
    let budget = max_results.unwrap_or(usize::MAX).min(k);

    // Rows found so far, counted like the single-query `initial_ids` (capped at k).
    let mut found: Vec<usize> = candidates.iter().map(|c| c.len().min(k)).collect();
    let mut is_active: Vec<bool> = Vec::with_capacity(query_count);
    for query_index in 0..query_count {
        if !needs_late[query_index] || found[query_index] >= k {
            is_active.push(false);
            continue;
        }
        if let Some(max_results) = max_results
            && found[query_index] < max_results
            && max_results <= k
            && let Some(allowed) = prefilter_mask.iter_addrs()
        {
            // Fewer than k rows can match the prefilter: return every allowed row
            // the early search did not reach instead of probing further. As in the
            // single-query path, a restricted delta emits only the rows it owns and
            // an unrestricted delta emits every allowed row.
            let joining: Vec<&BatchDelta> = deltas
                .iter()
                .filter(|delta| delta.joins_late_search(query_index))
                .collect();
            let is_unrestricted = joining.iter().any(|delta| delta.seg_mask.is_none());
            let mut seen: HashSet<u64> = candidates[query_index]
                .iter()
                .map(|&(_, row_id)| row_id)
                .collect();
            let unseen: Vec<(f32, u64)> = allowed
                .map(u64::from)
                .filter(|&addr| {
                    (is_unrestricted
                        || joining.iter().any(|delta| {
                            delta
                                .seg_mask
                                .as_ref()
                                .is_some_and(|mask| mask.selected(addr))
                        }))
                        && seen.insert(addr)
                })
                .map(|addr| (f32::INFINITY, addr))
                .collect();
            candidates[query_index].extend(unseen);
            is_active.push(false);
            continue;
        }
        is_active.push(found[query_index] < budget);
    }

    let max_wave = get_num_compute_intensive_cpus().max(1);
    // Each virtual query copies its query vector, so bound the per-round total to
    // keep key memory and per-query setup independent of batch size × CPU count.
    let max_round_probes = MAX_LATE_PROBES_PER_ROUND.max(max_wave);
    // Per (delta, query) progress. Every round advances all deltas together, like
    // the single-query path whose per-delta late searches run concurrently.
    let mut cursor: Vec<Vec<usize>> = deltas
        .iter()
        .map(|delta| delta.plans.iter().map(|plan| plan.early_end).collect())
        .collect();
    let mut wave: Vec<Vec<usize>> = vec![vec![1; query_count]; deltas.len()];
    // Rotates the starting query so a capped round does not starve later queries.
    let mut rotation = 0;
    loop {
        // round[delta] = (query, width) pairs probed in that delta this round.
        let mut round: Vec<Vec<(usize, usize)>> = vec![Vec::new(); deltas.len()];
        let mut round_probes = 0;
        let mut last_admitted = None;
        let mut is_pending = false;
        for step in 0..query_count {
            let query_index = (rotation + step) % query_count;
            if !is_active[query_index] || found[query_index] >= budget {
                continue;
            }
            let widths: Vec<usize> = (0..deltas.len())
                .map(|d| {
                    if !deltas[d].joins_late_search(query_index) {
                        return 0;
                    }
                    let remaining = deltas[d].plans[query_index]
                        .late_end
                        .saturating_sub(cursor[d][query_index]);
                    wave[d][query_index].min(remaining)
                })
                .collect();
            let query_probes: usize = widths.iter().sum();
            if query_probes == 0 {
                continue;
            }
            if round_probes >= max_round_probes {
                is_pending = true;
                break;
            }
            round_probes += query_probes;
            last_admitted = Some(query_index);
            for (d, &width) in widths.iter().enumerate() {
                if width > 0 {
                    round[d].push((query_index, width));
                }
            }
        }
        let Some(last_admitted) = last_admitted else {
            break;
        };
        if is_pending {
            rotation = (last_admitted + 1) % query_count;
        }

        let searches = deltas
            .iter()
            .zip(&round)
            .enumerate()
            .filter(|(_, (_, probes))| !probes.is_empty())
            .map(|(d, (delta, probes))| {
                // One virtual query per (query, partition) so each partition's rows
                // stay separable and the per-query cut can match the sequential search.
                let probe_count = probes.iter().map(|&(_, width)| width).sum();
                let mut keys: Vec<ArrayRef> = Vec::with_capacity(probe_count);
                let mut partitions = Vec::with_capacity(probe_count);
                let mut q_c_dists = Vec::with_capacity(probe_count);
                for &(query_index, width) in probes {
                    let plan = &delta.plans[query_index];
                    let start = cursor[d][query_index];
                    let key = delta.query.key.slice(query_index * dim, dim);
                    for offset in start..start + width {
                        keys.push(key.clone());
                        partitions.push(Arc::new(plan.partitions.slice(offset, 1)));
                        q_c_dists.push(Arc::new(plan.q_c_dists.slice(offset, 1)));
                    }
                }
                let pre_filter = delta.pre_filter.clone();
                let seg_mask = delta.seg_mask.as_deref();
                async move {
                    let key_refs: Vec<&dyn Array> = keys.iter().map(|key| key.as_ref()).collect();
                    let mut round_query = delta.query.clone();
                    round_query.key = arrow_select::concat::concat(&key_refs)?;
                    let results = search_batch(
                        delta.index.clone(),
                        round_query,
                        partitions,
                        q_c_dists,
                        pre_filter,
                        seg_mask,
                        metrics,
                    )
                    .await?;
                    Ok::<_, DataFusionError>((d, results))
                }
            });
        let delta_results = futures::future::try_join_all(searches).await?;

        // Per query, the probe results of each delta in rank order.
        let mut per_query: Vec<Vec<std::vec::IntoIter<RecordBatch>>> =
            vec![Vec::new(); query_count];
        for (d, results) in delta_results {
            let mut results = results.into_iter();
            for &(query_index, width) in &round[d] {
                let batches: Vec<RecordBatch> = results.by_ref().take(width).collect();
                if batches.len() != width {
                    return Err(DataFusionError::Internal(
                        "late batch search returned fewer results than probes".to_string(),
                    ));
                }
                per_query[query_index].push(batches.into_iter());
                grow_wave(&mut wave[d][query_index], max_wave);
            }
        }
        // Consume each query's partitions round-robin across deltas, stopping at the
        // budget exactly like the sequential search checks it before each partition.
        for (query_index, mut streams) in per_query.into_iter().enumerate() {
            let delta_ids: Vec<usize> = (0..deltas.len())
                .filter(|&d| round[d].iter().any(|&(q, _)| q == query_index))
                .collect();
            let mut is_progress = true;
            while is_progress && found[query_index] < budget {
                is_progress = false;
                for (stream, &d) in streams.iter_mut().zip(&delta_ids) {
                    if found[query_index] >= budget {
                        break;
                    }
                    if let Some(batch) = stream.next() {
                        found[query_index] +=
                            append_candidates(&batch, &mut candidates[query_index])?;
                        cursor[d][query_index] += 1;
                        is_progress = true;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Doubles a late-search wave width, capped at `max_wave`.
fn grow_wave(wave: &mut usize, max_wave: usize) {
    *wave = (*wave * 2).min(max_wave);
}
