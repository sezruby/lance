// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `fetch_rows()` implementation for [`super::ExternalIvfPqIndex`].
//!
//! Post-topK materialization primitive. The caller supplies a list of
//! `(file_path, row_index)` keys and a projection column list; Lance batches by
//! file, issues one page-index-aware parquet read per file, reassembles the
//! result in caller-input order, and returns a `RecordBatch` with one row per
//! input key.
//!
//! Why this matters for lance-spark: today the join's materialize stage writes
//! all R columns into a temp Lance file because there's no way to fetch them on
//! demand later. With `fetch_rows`, the join only fetches the projection columns
//! for surviving top-K rows — a hard win on materialize I/O for large R.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance_core::{Error, Result};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::RowSelection;

use super::open::OpenedExternalIndex;
use super::parquet_source::{ParquetMetaCache, open_parquet_async, open_parquet_cached};
use super::types::ParquetRowKey;

/// Fetch rows by `(file_path, row_index)` from the registered parquet files.
///
/// `projection` may include any column from the parquet schema, not just the
/// vector column the index was built over.
///
/// Result rows are in caller-input order — duplicates in the input produce
/// duplicate result rows. Empty input returns an empty batch with the projected
/// schema.
pub async fn fetch_rows(
    opened: &OpenedExternalIndex,
    row_keys: &[ParquetRowKey],
    projection: &[&str],
) -> Result<RecordBatch> {
    let first_file_path = opened
        .manifest
        .files
        .first()
        .map(|f| f.file_path.as_str())
        .unwrap_or("");
    fetch_rows_impl(
        &opened.parquet_meta_cache,
        |p| opened.manifest.file_id(p).is_some(),
        first_file_path,
        row_keys,
        projection,
    )
    .await
}

/// Storage-agnostic core of [`fetch_rows`]. Reused by the external scalar (BTree)
/// index, which carries the same `(file_path, row_index)` identity model but its
/// own manifest type. `is_registered` validates that a fetched path belongs to the
/// index's registered file set; `first_file_path` supplies the schema source for
/// the empty-input case.
pub(crate) async fn fetch_rows_impl(
    parquet_meta_cache: &ParquetMetaCache,
    is_registered: impl Fn(&str) -> bool,
    first_file_path: &str,
    row_keys: &[ParquetRowKey],
    projection: &[&str],
) -> Result<RecordBatch> {
    if projection.is_empty() {
        return Err(Error::invalid_input(
            "fetch_rows: projection must contain at least one column",
        ));
    }

    // Group input by file_path while remembering each input's position so we
    // can reorder later.
    let mut by_file: HashMap<String, Vec<(usize, u64)>> = HashMap::new();
    for (input_pos, key) in row_keys.iter().enumerate() {
        by_file
            .entry(key.file_path.clone())
            .or_default()
            .push((input_pos, key.row_index));
    }

    // Validate every file appears in the manifest. We don't strictly need it
    // for correctness here (Lance will just open the parquet path) but it
    // catches typos early and matches the contract that fetched files belong
    // to the registered set.
    for path in by_file.keys() {
        if !is_registered(path) {
            return Err(Error::invalid_input(format!(
                "fetch_rows: file '{path}' not registered with this index"
            )));
        }
    }

    // Pull the schema from the first file's read so we can build an empty batch
    // of the projected schema if the input is empty.
    if row_keys.is_empty() {
        let schema = projected_schema_from_file(first_file_path, projection).await?;
        return Ok(RecordBatch::new_empty(schema));
    }

    // Per-file read. Inputs within a file go to one parquet read regardless of
    // duplicates / order.
    let mut per_file_results: Vec<(Vec<usize>, RecordBatch)> = Vec::with_capacity(by_file.len());
    let mut shared_schema: Option<SchemaRef> = None;
    for (file_path, hits) in by_file {
        let row_indices: Vec<u64> = hits.iter().map(|(_, r)| *r).collect();
        let batch = read_rows_from_file(
            parquet_meta_cache,
            &file_path,
            projection,
            &row_indices,
        )
        .await?;
        if shared_schema.is_none() {
            shared_schema = Some(batch.schema());
        }
        per_file_results.push((hits.iter().map(|(p, _)| *p).collect(), batch));
    }
    let schema = shared_schema.expect("non-empty input means at least one batch");

    // Reorder: build one row of `RecordBatch` per input position, then concat.
    // We use `take`-style indexing per column. For typical sizes (top-K * Q)
    // this is fast.
    let total_rows = row_keys.len();
    let mut column_collectors: Vec<Vec<ArrayRef>> =
        vec![Vec::with_capacity(total_rows); schema.fields().len()];

    // Build a position-aware view of each per-file batch:
    //   position_in_input → (file_batch_index, row_in_batch)
    let mut by_input_position: Vec<Option<(usize, usize)>> = vec![None; total_rows];
    for (file_batch_index, (input_positions, _batch)) in per_file_results.iter().enumerate() {
        // input_positions[i] is the input position for row i in this batch.
        // Store dedup-aware positions: a duplicate input rid in the same file
        // shares the same row in the read result (we deduped in read_rows_from_file
        // before issuing the parquet read).
        // Re-derive the row-in-batch via the dedup map produced inside
        // read_rows_from_file? Easier: re-run a small reorder here.
        // Build the dedup ordering the same way read_rows_from_file did.
        let row_indices_in_file: Vec<u64> = input_positions
            .iter()
            .map(|&p| row_keys[p].row_index)
            .collect();
        let mut sorted = row_indices_in_file.clone();
        sorted.sort_unstable();
        sorted.dedup();
        for (input_pos, &row_index) in input_positions.iter().zip(row_indices_in_file.iter()) {
            let row_in_batch = sorted
                .binary_search(&row_index)
                .expect("row missing from per-file batch index");
            by_input_position[*input_pos] = Some((file_batch_index, row_in_batch));
        }
    }

    // Slice each input row's columns into the collectors.
    for input_pos in 0..total_rows {
        let (batch_idx, row_in_batch) = by_input_position[input_pos].ok_or_else(|| {
            Error::index(format!("fetch_rows: input position {input_pos} unmapped"))
        })?;
        let batch = &per_file_results[batch_idx].1;
        for (col_idx, col) in batch.columns().iter().enumerate() {
            column_collectors[col_idx].push(col.slice(row_in_batch, 1));
        }
    }

    // Concat each column's collected slices into one array.
    let mut final_columns: Vec<ArrayRef> = Vec::with_capacity(column_collectors.len());
    for col_chunks in column_collectors {
        let refs: Vec<&dyn Array> = col_chunks.iter().map(|a| a.as_ref()).collect();
        let concatenated = arrow::compute::concat(&refs)
            .map_err(|e| Error::index(format!("fetch_rows: column concat failed: {e}")))?;
        final_columns.push(concatenated);
    }
    let out = RecordBatch::try_new(schema, final_columns)
        .map_err(|e| Error::index(format!("fetch_rows: failed to build output batch: {e}")))?;
    Ok(out)
}

/// Read the requested `row_indices` of `projection` columns from one parquet
/// file. Returns a single `RecordBatch` with rows in **deduped sorted order**.
/// Caller is responsible for reordering to its input order.
async fn read_rows_from_file(
    cache: &ParquetMetaCache,
    path: &str,
    projection: &[&str],
    row_indices: &[u64],
) -> Result<RecordBatch> {
    let builder = open_parquet_cached(cache, path).await?;
    let total_rows: u64 = builder.metadata().file_metadata().num_rows() as u64;

    let mask = ProjectionMask::columns(builder.parquet_schema(), projection.iter().copied());

    let mut sorted: Vec<u64> = row_indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if let Some(&last) = sorted.last() {
        if last >= total_rows {
            return Err(Error::invalid_input(format!(
                "fetch_rows: row_index {last} out of range for {path} ({total_rows} rows)"
            )));
        }
    }
    let ranges: Vec<Range<usize>> = sorted
        .iter()
        .map(|&r| (r as usize)..(r as usize + 1))
        .collect();
    let selection = RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows as usize);

    let stream = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| Error::invalid_input(format!("fetch_rows: build reader {path}: {e}")))?;

    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| Error::invalid_input(format!("fetch_rows: read {path}: {e}")))?;
    if batches.is_empty() {
        // Build an empty batch with the projected schema.
        let schema = projected_schema_from_file(path, projection).await?;
        return Ok(RecordBatch::new_empty(schema));
    }
    let schema = batches[0].schema();
    concat_batches(&schema, &batches)
        .map_err(|e| Error::index(format!("fetch_rows: concat batches from {path}: {e}")))
}

async fn projected_schema_from_file(path: &str, projection: &[&str]) -> Result<SchemaRef> {
    let builder = open_parquet_async(path).await?;
    let arrow_schema = builder.schema();
    let fields: Vec<Field> = projection
        .iter()
        .map(|name| {
            arrow_schema.field_with_name(name).cloned().map_err(|e| {
                Error::invalid_input(format!(
                    "fetch_rows: projected column '{name}' missing in {path}: {e}"
                ))
            })
        })
        .collect::<Result<_>>()?;
    Ok(Arc::new(Schema::new(fields)))
}
