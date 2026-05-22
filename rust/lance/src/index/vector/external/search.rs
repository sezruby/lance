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
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use arrow::compute::concat;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use lance_core::{Error, Result};
use lance_index::vector::pq::ProductQuantizer;
use lance_io::traits::Reader;
use lance_linalg::distance::MetricType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
};
use parquet::file::metadata::PageIndexPolicy;

use super::open::OpenedExternalIndex;
use super::types::{RowFilter, SearchResult};

/// Top-level search entry point. Heavy lifting is in the helpers; this is the
/// orchestrator.
pub async fn search(
    opened: &OpenedExternalIndex,
    query: &[f32],
    k: usize,
    nprobes: usize,
    refine_factor: usize,
    filter: Option<&dyn RowFilter>,
) -> Result<Vec<SearchResult>> {
    let dim = opened.ivf.dimension();
    if query.len() != dim {
        return Err(Error::invalid_input(format!(
            "query dim {} != index dim {dim}",
            query.len()
        )));
    }
    let mt: MetricType = opened.metric;

    // 1. Probe nprobes partitions
    let query_array = Float32Array::from(query.to_vec());
    let (part_ids, _part_distances) =
        opened
            .ivf
            .find_partitions(&query_array, nprobes.max(1), opened.metric)?;

    // For L2/Cosine the index stores residual-encoded PQ codes — we'd need to
    // subtract the partition centroid from the query before PQ scoring. For Dot,
    // residuals aren't applied. Since the build path computes residuals for
    // L2/Cosine, mirror that here.
    let centroids = opened
        .ivf
        .centroids_array()
        .ok_or_else(|| Error::index("opened index has no centroids"))?
        .clone();
    let residual_for_partition = |part_id: u32| -> Vec<f32> {
        let centroid_values = centroids
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("centroids must be Float32");
        let start = (part_id as usize) * dim;
        if matches!(mt, MetricType::L2 | MetricType::Cosine) {
            (0..dim)
                .map(|d| query[d] - centroid_values.value(start + d))
                .collect()
        } else {
            query.to_vec()
        }
    };

    // 2. Score PQ codes across probed partitions, build top (k*refine_factor)
    //    candidate min-heap by PQ-approx distance.
    let candidate_count = (k * refine_factor.max(1)).max(k);
    let mut top_heap: BinaryHeap<(OrderedF32, u64)> = BinaryHeap::with_capacity(candidate_count);

    for &part_id in part_ids.values().iter() {
        let residual = residual_for_partition(part_id);
        let part_idx = part_id as usize;
        let part_range = opened.ivf.row_range(part_idx);
        if part_range.is_empty() {
            continue;
        }
        let (pq_codes, row_ids) =
            read_partition(&opened.index_file_reader, &opened.pq, part_range).await?;

        for i in 0..row_ids.len() {
            let code_offset = i * opened.pq.num_sub_vectors;
            let dist = score_pq_code(&residual, &opened.pq, &pq_codes, code_offset, mt);
            let rid = row_ids[i];

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

    // 3. Decode rids and apply RowFilter pre-refinement (so we skip parquet I/O
    //    on dropped rows).
    let candidates: Vec<u64> = top_heap.into_iter().map(|(_, rid)| rid).collect();
    let mut to_refine: Vec<(u64, u32, u64, String)> = Vec::with_capacity(candidates.len());
    for rid in candidates {
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
        to_refine.push((rid, file_id, row_in_file, file_path));
    }

    // 4. Per-file refinement reads.
    let mut by_file: HashMap<String, Vec<(usize, u64)>> = HashMap::new();
    for (input_pos, &(_rid, _file_id, row_in_file, ref file_path)) in to_refine.iter().enumerate() {
        by_file
            .entry(file_path.clone())
            .or_default()
            .push((input_pos, row_in_file));
    }

    let mut exact_dists: Vec<(usize, f32)> = Vec::with_capacity(to_refine.len());
    for (file_path, hits) in by_file {
        let row_indices: Vec<u64> = hits.iter().map(|(_, r)| *r).collect();
        let fetched =
            read_vectors_by_row_index(&file_path, &opened.manifest.vector_column, &row_indices)?;
        let dim = fetched.value_length() as usize;
        let values = fetched
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Float32Array vectors");
        for (out_idx, &(input_pos, _row)) in hits.iter().enumerate() {
            let mut dist = 0.0f32;
            for d in 0..dim {
                let v = values.value(out_idx * dim + d);
                let q = query[d];
                let diff = v - q;
                dist += diff * diff;
            }
            // For Cosine: distance is 1 - cosine similarity = 1 - (a·b/(|a||b|)).
            // For L2: dist is the squared L2 above.
            // For Dot: -dot.
            // We approximate with squared L2 here for L2 and Cosine; full
            // Cosine support lands in a follow-up after we plumb normalization
            // through this path.
            let _ = mt;
            exact_dists.push((input_pos, dist));
        }
    }

    // 5. Sort + top-K
    exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut out: Vec<SearchResult> = Vec::with_capacity(k);
    for (input_pos, dist) in exact_dists.into_iter().take(k) {
        let (_, _, row_in_file, file_path) = &to_refine[input_pos];
        out.push(SearchResult {
            file_path: file_path.clone(),
            row_index: *row_in_file,
            distance: dist,
        });
    }
    Ok(out)
}

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

    let pq_range = range.start..range.start + pq_bytes;
    let pq_data = reader.get_range(pq_range).await?;
    let pq_codes: Vec<u8> = pq_data.to_vec();

    let rid_range_start = range.start + pq_bytes;
    let rid_range = rid_range_start..rid_range_start + row_id_bytes;
    let rid_data = reader.get_range(rid_range).await?;
    let mut row_ids: Vec<u64> = Vec::with_capacity(len);
    for i in 0..len {
        let chunk: [u8; 8] = rid_data[i * 8..(i + 1) * 8]
            .try_into()
            .map_err(|_| Error::io("partition row_id chunk truncated".to_string()))?;
        row_ids.push(u64::from_le_bytes(chunk));
    }
    Ok((pq_codes, row_ids))
}

/// Score one PQ code against the (residual or raw) query using the PQ codebook's
/// distance tables. This is a slow, straightforward implementation — the SIMD
/// fast path lives in `lance_index::vector::pq` and lands as a follow-up. For
/// Phase 1 the goal is correctness; Phase 1.5 perf is acceptable as long as it
/// matches the IVF probe's output.
fn score_pq_code(
    query_or_residual: &[f32],
    pq: &ProductQuantizer,
    pq_codes: &[u8],
    code_offset: usize,
    _mt: MetricType,
) -> f32 {
    let m = pq.num_sub_vectors;
    let dim = pq.dimension;
    let sub_dim = dim / m;

    let codebook_values = pq
        .codebook
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("codebook is Float32");

    let mut total = 0.0f32;
    for s in 0..m {
        let code = pq_codes[code_offset + s] as usize;
        // codebook layout: [num_subvectors][num_codes][sub_dim]
        let cb_offset = (s * (1usize << pq.num_bits) + code) * sub_dim;
        let q_offset = s * sub_dim;
        for d in 0..sub_dim {
            let cb_val = codebook_values.value(cb_offset + d);
            let q_val = query_or_residual[q_offset + d];
            let diff = q_val - cb_val;
            total += diff * diff;
        }
    }
    total
}

/// Page-index-aware random fetch from one parquet file. Returns rows in
/// caller-input order. The same primitive [`super::fetch_rows`] uses for
/// post-topK materialization, but specialized to a single file (since
/// refinement is already grouped by file at the call site).
fn read_vectors_by_row_index(
    path: &str,
    column: &str,
    row_indices: &[u64],
) -> Result<FixedSizeListArray> {
    let file = File::open(path)
        .map_err(|e| Error::invalid_input(format!("failed to open parquet {path}: {e}")))?;
    let opts = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let builder =
        ParquetRecordBatchReaderBuilder::try_new_with_options(file, opts).map_err(|e| {
            Error::invalid_input(format!("failed to read parquet metadata for {path}: {e}"))
        })?;
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

    let reader = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| {
            Error::invalid_input(format!("failed to build parquet reader for {path}: {e}"))
        })?;

    let batches: Vec<arrow_array::RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| {
            Error::invalid_input(format!("error reading parquet batches from {path}: {e}"))
        })?;

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
    Ok(FixedSizeListArray::try_new_from_values(flat, dim as i32)
        .map_err(|e| Error::index(format!("failed to rebuild FSL: {e}")))?)
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
