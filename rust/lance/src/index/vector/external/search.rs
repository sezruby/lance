// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `search()` implementation for [`super::ExternalIvfPqIndex`].
//!
//! Pipeline per query:
//!
//! 1. **IVF probe**: `IvfModel::find_partitions(query, nprobes)` returns the
//!    closest `nprobes` partition IDs.
//! 2. **PQ candidate scoring**: for each probed partition, read the on-disk PQ
//!    codes + row IDs, score each row's code against the query, accumulate
//!    `(rid, pq_distance)` into a max-heap of size `k * refine_factor`.
//! 3. **Decode rids → (file_id, row_index)**: `(rid >> 32, rid & 0xFFFF_FFFF)`.
//! 4. **Refinement read**: group candidates by file, fetch each file's actual
//!    vectors via the page-index-aware parquet reader, compute exact distances.
//! 5. **Apply `RowFilter`** before re-ranking; rows the filter rejects are
//!    dropped.
//! 6. **Top-K trim**: sort by exact distance, take K, return as
//!    `Vec<SearchResult>` keyed on `(file_path, row_index)`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use arrow::compute::concat;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, UInt64Array, UInt8Array};
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::pq::storage::transpose;
use lance_io::traits::Reader;
use lance_linalg::distance::MetricType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::RowSelection;

use super::open::OpenedExternalIndex;
use super::parquet_source::{ParquetMetaCache, ParquetVectorSource};
use super::types::{ParquetFileSpec, RowFilter, SearchResult};

/// Single-query nearest-neighbor search. Thin wrapper over [`search_batch`] —
/// every interesting bit of work (probe, PQ scoring, per-file refinement) is
/// already factored out for the batch path. Kept as a public entry point
/// because callers that only have one query at a time shouldn't be forced to
/// allocate `Vec<Vec<f32>>` and then unwrap `Vec<Vec<SearchResult>>`.
pub async fn search(
    opened: &OpenedExternalIndex,
    query: &[f32],
    k: usize,
    nprobes: usize,
    refine_factor: usize,
    filter: Option<&dyn RowFilter>,
) -> Result<Vec<SearchResult>> {
    let queries = std::slice::from_ref(&query);
    let mut batch = search_batch(opened, queries, k, nprobes, refine_factor, filter).await?;
    Ok(batch.pop().unwrap_or_default())
}

/// Batched nearest-neighbor search. Same probe + PQ-score work per query as
/// [`search`], but a single unioned per-file refinement read covers every
/// query's candidates at once. Designed for offline join workloads where many
/// query vectors land on the same Spark task and share the same source parquet
/// files.
///
/// Wall-clock benefit comes from collapsing N per-query refinement reads (each
/// pulling ~50 MB of parquet pages from cloud storage at numL=100) into one
/// read per file per task. Per-query cost falls from `(probe + pq + refine_io)`
/// to `(probe + pq + refine_io / N)`.
///
/// Result vector is in 1:1 correspondence with the input `queries` slice.
pub async fn search_batch(
    opened: &OpenedExternalIndex,
    queries: &[&[f32]],
    k: usize,
    nprobes: usize,
    refine_factor: usize,
    filter: Option<&dyn RowFilter>,
) -> Result<Vec<Vec<SearchResult>>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }

    let batch_seq = QUERY_SEQ.fetch_add(1, Ordering::Relaxed);
    let t_start = Instant::now();
    let dim = opened.ivf.dimension();
    for (qi, q) in queries.iter().enumerate() {
        if q.len() != dim {
            return Err(Error::invalid_input(format!(
                "query[{qi}] dim {} != index dim {dim}",
                q.len()
            )));
        }
    }
    let mt: MetricType = opened.metric;
    let centroids = opened
        .ivf
        .centroids_array()
        .ok_or_else(|| Error::index("opened index has no centroids"))?
        .clone();
    let centroid_values = centroids
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("centroids must be Float32")
        .clone();
    let candidate_count = (k * refine_factor.max(1)).max(k);
    let num_queries = queries.len();

    // 1. Probe + PQ score every query. Each query owns its own top
    //    `candidate_count` heap by approximate distance.
    let t_probe_pq = Instant::now();
    // (file_id, row_in_file, file_path) per refine candidate, per query.
    let mut to_refine_by_query: Vec<Vec<(u32, u64, String)>> = vec![Vec::new(); num_queries];
    for (qi, &query) in queries.iter().enumerate() {
        let query_array = Float32Array::from(query.to_vec());
        let (part_ids, _) =
            opened
                .ivf
                .find_partitions(&query_array, nprobes.max(1), opened.metric)?;

        // Residuals: for L2/Cosine the index stores residual-encoded PQ codes
        // (subtract centroid before scoring); Dot uses the raw query.
        let residual_for_partition = |part_id: u32| -> Vec<f32> {
            let start = (part_id as usize) * dim;
            if matches!(mt, MetricType::L2 | MetricType::Cosine) {
                (0..dim)
                    .map(|d| query[d] - centroid_values.value(start + d))
                    .collect()
            } else {
                query.to_vec()
            }
        };

        let mut top_heap: BinaryHeap<(OrderedF32, u64)> =
            BinaryHeap::with_capacity(candidate_count);
        // Read all probed partitions CONCURRENTLY. Each read_partition is an object-store
        // get_range; awaiting them serially cost ~nprobes round trips per query (~0.15 s at
        // nprobes=16 over abfss). The PQ scoring below stays serial (it's CPU-only).
        let part_reads = part_ids.values().iter().filter_map(|&part_id| {
            let part_range = opened.ivf.row_range(part_id as usize);
            if part_range.is_empty() {
                None
            } else {
                let reader = opened.index_file_reader.clone();
                let pq = &opened.pq;
                Some(async move {
                    let (codes, rids) = read_partition(&reader, pq, part_range).await?;
                    Ok::<(u32, Vec<u8>, Vec<u64>), Error>((part_id, codes, rids))
                })
            }
        });
        let partitions = futures::future::try_join_all(part_reads).await?;

        for (part_id, pq_codes, row_ids) in partitions {
            if row_ids.is_empty() {
                continue;
            }

            let num_sub_vectors = opened.pq.num_sub_vectors;
            let pq_array = UInt8Array::from(pq_codes);
            let transposed = transpose(&pq_array, row_ids.len(), num_sub_vectors);
            let residual_arr = Float32Array::from(residual_for_partition(part_id));
            let dists = opened.pq.compute_distances(&residual_arr, &transposed)?;
            let dist_values = dists.values();

            for (i, &rid) in row_ids.iter().enumerate() {
                let dist = dist_values[i];
                if top_heap.len() < candidate_count {
                    top_heap.push((OrderedF32(dist), rid));
                } else if let Some(top) = top_heap.peek() {
                    if dist < top.0.0 {
                        top_heap.pop();
                        top_heap.push((OrderedF32(dist), rid));
                    }
                }
            }
        }

        // Decode rids → (file_id, row_in_file, file_path) and apply the
        // pre-refinement RowFilter so we don't pay parquet I/O on dropped rows.
        let mut to_refine: Vec<(u32, u64, String)> = Vec::with_capacity(top_heap.len());
        for (_, rid) in top_heap {
            let file_id = (rid >> 32) as u32;
            let row_in_file = rid & 0xFFFF_FFFF;
            let file_path = opened
                .manifest
                .file_path(file_id)
                .ok_or_else(|| {
                    Error::index(format!(
                        "candidate rid {rid:#x} encodes file_id={file_id} but manifest has only {} files",
                        opened.manifest.files.len()
                    ))
                })?
                .to_string();
            if let Some(f) = filter {
                if !f.keep(&file_path, row_in_file) {
                    continue;
                }
            }
            to_refine.push((file_id, row_in_file, file_path));
        }
        to_refine_by_query[qi] = to_refine;
    }
    let probe_pq_ms = t_probe_pq.elapsed().as_millis();

    // 2. Union candidate (file_path, row_in_file) pairs across all queries
    //    so we issue ONE refinement read per file regardless of how much
    //    overlap there is. Rows fetched per file are deduped; the per-query
    //    re-rank below looks up each query's candidates by row_in_file.
    let t_union = Instant::now();
    let mut union_by_file: HashMap<String, std::collections::HashSet<u64>> = HashMap::new();
    let mut total_candidate_pairs = 0usize;
    for refs in &to_refine_by_query {
        for (_, row, path) in refs {
            union_by_file.entry(path.clone()).or_default().insert(*row);
            total_candidate_pairs += 1;
        }
    }
    let union_ms = t_union.elapsed().as_millis();

    // 3+4. Refine candidates → per-query top-K. Two paths:
    //   - SQ8 rerank store present: read co-located int8 codes by contiguous
    //     byte-range and rerank with integer `l2_u8` — no source-parquet read.
    //   - otherwise: the parquet refinement path (read originals from the source
    //     vector column via the page-index-aware reader).
    let t_refine = Instant::now();
    let mut total_refine_open_ms: u128 = 0;
    let mut total_refine_io_ms: u128 = 0;
    let mut total_refine_rows: usize = 0;

    let out: Vec<Vec<SearchResult>> = if let Some(rerank) = opened.rerank.as_ref() {
        // Map every candidate (file_id, row_in_file) → global ordinal and fetch
        // its SQ8 code once. `union_by_file` already deduped per file; flatten to
        // ordinals across all files.
        let mut ordinals: Vec<u64> = Vec::new();
        for (file_path, row_set) in &union_by_file {
            let file_id = opened.manifest.file_id(file_path).ok_or_else(|| {
                Error::index(format!("refine: file '{file_path}' not in manifest"))
            })?;
            let base = opened.manifest.global_base(file_id).ok_or_else(|| {
                Error::index(format!("refine: no global_base for file_id {file_id}"))
            })?;
            for &row in row_set {
                ordinals.push(base + row);
            }
        }
        total_refine_rows = ordinals.len();
        let t_io = Instant::now();
        let codes = rerank.fetch_codes(&ordinals).await?;
        total_refine_io_ms = t_io.elapsed().as_millis();

        let t_topk = Instant::now();
        let mut out = Vec::with_capacity(num_queries);
        for (qi, refs) in to_refine_by_query.iter().enumerate() {
            let query_repr = rerank.encode_query(queries[qi])?;
            let mut scored: Vec<(f32, &str, u64)> = Vec::with_capacity(refs.len());
            for (file_id, row_in_file, file_path) in refs {
                let base = opened.manifest.global_base(*file_id).expect("global_base");
                let ord = base + *row_in_file;
                let row_code = codes
                    .get(&ord)
                    .ok_or_else(|| Error::index(format!("refine: ordinal {ord} not fetched")))?;
                let dist = rerank.distance(&query_repr, row_code);
                scored.push((dist, file_path.as_str(), *row_in_file));
            }
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            out.push(
                scored
                    .into_iter()
                    .take(k)
                    .map(|(dist, path, row_in_file)| SearchResult {
                        file_path: path.to_string(),
                        row_index: row_in_file,
                        distance: dist,
                    })
                    .collect(),
            );
        }
        let topk_ms = t_topk.elapsed().as_millis();
        let refine_ms = t_refine.elapsed().as_millis();
        let total_ms = t_start.elapsed().as_millis();
        let store_kind = opened
            .manifest
            .rerank
            .as_ref()
            .map(|m| m.kind.as_str())
            .unwrap_or("rerank");
        log::info!(
            "extidx_search_batch seq={batch_seq} store={store_kind} n_queries={num_queries} \
             total={total_ms}ms probe_pq={probe_pq_ms}ms union={union_ms}ms refine={refine_ms}ms \
             topk={topk_ms}ms refine_io={total_refine_io_ms}ms refine_rows={total_refine_rows} \
             candidate_pairs={total_candidate_pairs}",
        );
        out
    } else {
        // file_path → (sorted_unique_row_indices, fetched FixedSizeListArray).
        // The array's row order matches the sorted_unique_row_indices vector.
        let mut fetched_by_file: HashMap<String, (Vec<u64>, FixedSizeListArray)> =
            HashMap::with_capacity(union_by_file.len());
        for (file_path, row_set) in union_by_file {
            let mut sorted: Vec<u64> = row_set.into_iter().collect();
            sorted.sort_unstable();
            total_refine_rows += sorted.len();
            let (fetched, open_ms, io_ms) = read_vectors_by_row_index(
                &opened.parquet_meta_cache,
                &file_path,
                &opened.manifest.vector_column,
                &sorted,
            )
            .await?;
            total_refine_open_ms += open_ms;
            total_refine_io_ms += io_ms;
            fetched_by_file.insert(file_path, (sorted, fetched));
        }
        let refine_files = fetched_by_file.len();
        let refine_ms = t_refine.elapsed().as_millis();

        let t_topk = Instant::now();
        let mut out: Vec<Vec<SearchResult>> = Vec::with_capacity(num_queries);
        for (qi, refs) in to_refine_by_query.iter().enumerate() {
            let query = queries[qi];
            let mut scored: Vec<(f32, &str, u64)> = Vec::with_capacity(refs.len());
            for (_, row_in_file, file_path) in refs {
                let (sorted, fetched) = fetched_by_file
                    .get(file_path)
                    .expect("file_path missing from fetched union");
                let pos = sorted.binary_search(row_in_file).map_err(|_| {
                    Error::index(format!(
                        "row_in_file {row_in_file} missing from refined union for {file_path}"
                    ))
                })?;
                let dim = fetched.value_length() as usize;
                let values = fetched
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Float32Array vectors");
                let mut dist = 0.0f32;
                for d in 0..dim {
                    let v = values.value(pos * dim + d);
                    let q = query[d];
                    let diff = v - q;
                    dist += diff * diff;
                }
                let _ = mt; // L2 / Cosine approximation — see search() docs.
                scored.push((dist, file_path.as_str(), *row_in_file));
            }
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            out.push(
                scored
                    .into_iter()
                    .take(k)
                    .map(|(dist, path, row_in_file)| SearchResult {
                        file_path: path.to_string(),
                        row_index: row_in_file,
                        distance: dist,
                    })
                    .collect(),
            );
        }
        let topk_ms = t_topk.elapsed().as_millis();
        let total_ms = t_start.elapsed().as_millis();
        log::info!(
            "extidx_search_batch seq={batch_seq} store=parquet n_queries={num_queries} \
             total={total_ms}ms probe_pq={probe_pq_ms}ms union={union_ms}ms refine={refine_ms}ms \
             topk={topk_ms}ms refine_open={total_refine_open_ms}ms refine_io={total_refine_io_ms}ms \
             refine_files={refine_files} refine_rows={total_refine_rows} \
             candidate_pairs={total_candidate_pairs}",
        );
        out
    };

    Ok(out)
}

/// Exact brute-force nearest-neighbor search — the no-index baseline.
///
/// Streams EVERY vector from the source parquet (via [`ParquetVectorSource::iter_batches`])
/// and computes the exact distance of each query to each row, maintaining a size-`k` top
/// heap per query. No IVF probe, no PQ, no refine — this is the "just scan the parquet"
/// reference against which the index's latency/recall is judged. Cost is O(|R| × |queries| ×
/// dim); it reads the full corpus once regardless of `k`.
///
/// Only the vector column is read (projected), so I/O is the vector bytes, not the wide row.
/// L2 / Cosine only (Cosine assumes normalized vectors, matching the index paths).
pub async fn search_flat(
    opened: &OpenedExternalIndex,
    queries: &[&[f32]],
    k: usize,
) -> Result<Vec<Vec<SearchResult>>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let dim = opened.ivf.dimension();
    for (qi, q) in queries.iter().enumerate() {
        if q.len() != dim {
            return Err(Error::invalid_input(format!(
                "query[{qi}] dim {} != index dim {dim}",
                q.len()
            )));
        }
    }
    let t_start = Instant::now();

    // Rebuild a source over the manifest's files (global file_id = position, same order the
    // index was built with) so decoded rids match the index's (file_id, row_in_file) space.
    let specs: Vec<ParquetFileSpec> = opened
        .manifest
        .files
        .iter()
        .map(|e| ParquetFileSpec::of(e.file_path.clone()))
        .collect();
    let source =
        ParquetVectorSource::try_new(specs, &opened.manifest.vector_column).await?;

    // Per query: a max-heap of (distance, rid) capped at k (largest distance on top, so we
    // pop the worst when a closer row arrives).
    let mut heaps: Vec<BinaryHeap<(OrderedF32, u64)>> =
        (0..queries.len()).map(|_| BinaryHeap::with_capacity(k + 1)).collect();

    let mut stream = source.iter_batches().await?;
    let mut scanned: u64 = 0;
    while let Some(batch) = stream.try_next().await? {
        let vec_col = batch
            .column_by_name(&opened.manifest.vector_column)
            .ok_or_else(|| Error::index("flat scan batch missing vector column"))?;
        let fsl = vec_col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| Error::index("flat scan vector column is not FixedSizeList"))?;
        let values = fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| Error::index("flat scan vectors are not Float32"))?;
        let rid_col = batch
            .column_by_name(super::parquet_source::RID_COLUMN_NAME)
            .ok_or_else(|| Error::index("flat scan batch missing rid column"))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::index("flat scan rid column is not UInt64"))?;

        let n = fsl.len();
        for row in 0..n {
            let base = row * dim;
            let rid = rid_col.value(row);
            for (qi, &query) in queries.iter().enumerate() {
                let mut dist = 0.0f32;
                for d in 0..dim {
                    let diff = values.value(base + d) - query[d];
                    dist += diff * diff;
                }
                let heap = &mut heaps[qi];
                if heap.len() < k {
                    heap.push((OrderedF32(dist), rid));
                } else if let Some(top) = heap.peek() {
                    if dist < top.0.0 {
                        heap.pop();
                        heap.push((OrderedF32(dist), rid));
                    }
                }
            }
        }
        scanned += n as u64;
    }

    let out: Vec<Vec<SearchResult>> = heaps
        .into_iter()
        .map(|heap| {
            let mut v: Vec<(OrderedF32, u64)> = heap.into_vec();
            v.sort_by(|a, b| a.0.0.partial_cmp(&b.0.0).unwrap());
            v.into_iter()
                .map(|(d, rid)| {
                    let file_id = (rid >> 32) as u32;
                    let row_in_file = rid & 0xFFFF_FFFF;
                    let file_path = opened
                        .manifest
                        .file_path(file_id)
                        .unwrap_or("")
                        .to_string();
                    SearchResult {
                        file_path,
                        row_index: row_in_file,
                        distance: d.0,
                    }
                })
                .collect()
        })
        .collect();

    log::info!(
        "extidx_search_flat n_queries={} scanned_rows={scanned} total={}ms",
        queries.len(),
        t_start.elapsed().as_millis()
    );
    Ok(out)
}

/// Per-task monotonic counter assigning a sequence number to each `search()`
/// call so log lines are ordered and joinable.
static QUERY_SEQ: AtomicU64 = AtomicU64::new(0);

/// Read one partition's PQ codes + row IDs from the index file.
///
/// Layout (matching `write_pq_partitions`):
///
///   `[partition_offset .. partition_offset + len * num_sub_vectors]`  PQ codes (u8)
///   `[partition_offset + len * num_sub_vectors .. + len * 8]`         row IDs (u64)
async fn read_partition(
    reader: &Arc<dyn Reader>,
    pq: &ProductQuantizer,
    range: Range<usize>,
) -> Result<(Vec<u8>, Vec<u64>)> {
    let len = range.end - range.start;
    let pq_bytes = len * pq.num_sub_vectors;
    let row_id_bytes = len * 8;

    // PQ codes and row IDs are stored CONTIGUOUSLY for a partition, so read the whole span in
    // ONE get_range and split in memory — halves the object-store round trips per partition
    // (was two serial reads). Matters most on high-latency stores where each read is an RTT.
    let byte_start = range.start;
    let byte_end = range.start + pq_bytes + row_id_bytes;
    let data = reader.get_range(byte_start..byte_end).await?;
    let pq_codes: Vec<u8> = data[..pq_bytes].to_vec();

    let rid_data = &data[pq_bytes..pq_bytes + row_id_bytes];
    let mut row_ids: Vec<u64> = Vec::with_capacity(len);
    for i in 0..len {
        let chunk: [u8; 8] = rid_data[i * 8..(i + 1) * 8]
            .try_into()
            .map_err(|_| Error::io("partition row_id chunk truncated".to_string()))?;
        row_ids.push(u64::from_le_bytes(chunk));
    }
    Ok((pq_codes, row_ids))
}

/// Page-index-aware random fetch from one parquet file. Returns rows in
/// caller-input order. The same primitive [`super::fetch_rows`] uses for
/// post-topK materialization, but specialized to a single file (since
/// refinement is already grouped by file at the call site).
async fn read_vectors_by_row_index(
    cache: &ParquetMetaCache,
    path: &str,
    column: &str,
    row_indices: &[u64],
) -> Result<(FixedSizeListArray, u128, u128)> {
    let t_open = Instant::now();
    let builder = super::parquet_source::open_parquet_cached(cache, path).await?;
    let open_ms = t_open.elapsed().as_millis();
    let total_rows: u64 = builder.metadata().file_metadata().num_rows() as u64;
    let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);

    // Dedup + sort since RowSelection requires strictly increasing ranges.
    let mut sorted: Vec<u64> = row_indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let ranges: Vec<Range<usize>> = sorted
        .iter()
        .map(|&r| (r as usize)..(r as usize + 1))
        .collect();
    let selection = RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows as usize);

    let stream = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| {
            Error::invalid_input(format!("failed to build parquet reader for {path}: {e}"))
        })?;

    let t_io = Instant::now();
    let batches: Vec<arrow_array::RecordBatch> = stream.try_collect().await.map_err(|e| {
        Error::invalid_input(format!("error reading parquet batches from {path}: {e}"))
    })?;
    let io_ms = t_io.elapsed().as_millis();

    // Detect dim from the first non-empty batch's column. coerce_to_fsl handles both
    // FixedSizeList and List<Float32>.
    let dim_from_batch: usize = batches
        .iter()
        .flat_map(|b| b.column_by_name(column).map(|c| c.clone()))
        .find(|c| c.len() > 0)
        .and_then(|col| {
            if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
                Some(fsl.value_length() as usize)
            } else if let Some(la) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
                Some(la.value_length(0) as usize)
            } else {
                None
            }
        })
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "vector column '{column}' yielded no rows in {path}"
            ))
        })?;

    // Coerce each batch's column to FSL, then concat.
    let mut fsl_chunks: Vec<FixedSizeListArray> = Vec::new();
    for b in &batches {
        let col = b
            .column_by_name(column)
            .ok_or_else(|| {
                Error::invalid_input(format!("vector column '{column}' missing in {path}"))
            })?
            .clone();
        fsl_chunks.push(super::parquet_source::coerce_to_fsl(&col, dim_from_batch)?);
    }
    let array_refs: Vec<&dyn Array> = fsl_chunks.iter().map(|a| a as &dyn Array).collect();
    let concatenated = concat(&array_refs)
        .map_err(|e| Error::invalid_input(format!("failed to concat batches from {path}: {e}")))?;
    let fsl = concatenated
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| Error::invalid_input(format!("vector column '{column}' missing in {path}")))?
        .clone();

    // Reorder to caller's input order.
    let dim = fsl.value_length() as usize;
    let result_values = fsl
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32 values");
    let mut reordered = Vec::with_capacity(row_indices.len() * dim);
    for &rid in row_indices {
        let pos = sorted
            .binary_search(&rid)
            .map_err(|_| Error::index(format!("rid {rid} missing from parquet result")))?;
        for d in 0..dim {
            reordered.push(result_values.value(pos * dim + d));
        }
    }
    let flat = Float32Array::from(reordered);
    let _ = ArrayRef::from(Arc::new(flat.clone()) as ArrayRef);
    use lance_arrow::FixedSizeListArrayExt;
    let fsl = FixedSizeListArray::try_new_from_values(flat, dim as i32)
        .map_err(|e| Error::index(format!("failed to rebuild FSL: {e}")))?;
    Ok((fsl, open_ms, io_ms))
}

// Wrap f32 to make it Ord-eligible for the BinaryHeap. Heap is a max-heap so we
// also flip the order via Reverse pattern at insert time? Actually we use Reverse
// at heap construction to make it min-by-distance for keep semantics; but the
// snippet above pushes (OrderedF32(dist), rid) directly. Because BinaryHeap is a
// max-heap, peek() returns the largest distance — exactly what we want as the
// "weakest candidate to be kicked out when a better one arrives."
#[derive(Copy, Clone, PartialEq, PartialOrd)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// `Reverse` is unused but referenced in the doc — keep it imported via use::reexport
// silencing.
const _: fn() = || {
    let _: Reverse<u32> = Reverse(0);
};
