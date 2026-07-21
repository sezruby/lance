// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Parquet scalar-key source for the external BTree build path.
//!
//! Reads the key column from `Vec<ParquetFileSpec>`, attaches the encoded rid
//! `(file_id_u32 << 32) | row_index_u32` as a `_rowid` UInt64 column, renames the
//! key column to `"value"` (`VALUE_COLUMN_NAME`), and produces a single
//! `SendableRecordBatchStream` of `(value, _rowid)` **sorted ascending by value** —
//! exactly the shape `train_btree_index` consumes.
//!
//! The dataset-backed scalar-index path gets this sorted `(value, _rowid)` stream
//! from `scan.order_by(col asc).with_row_id().project_with_transform`. With no
//! dataset here, we replicate the two pieces explicitly: the projection/rename +
//! rid attach below, and a DataFusion [`SortExec`] over the unioned in-memory
//! batches (the same sort the `train_btree_index` test precedents use).

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::expressions::col;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use futures::TryStreamExt;
use lance_core::{Error, ROW_ID, Result};
use lance_datafusion::exec::OneShotExec;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use parquet::arrow::ProjectionMask;

use super::types::ParquetFileSpec;
use crate::index::vector::external::parquet_source::open_parquet_async;

/// Read the parquet footer's row count for `path`.
pub(super) async fn read_num_rows(path: &str) -> Result<u64> {
    let builder = open_parquet_async(path).await?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}

/// Read `key_column` from every file in manifest order, attach the encoded rid as
/// a `_rowid` column, and rename the key column to `"value"`. Returns the shared
/// `(value, _rowid)` schema plus every file's batches.
///
/// The key column's Arrow type is inferred from the first file's footer and every
/// batch is built against that shared schema, so a file whose key column has a
/// different type surfaces as a `RecordBatch::try_new` error rather than silently
/// corrupting the index.
async fn collect_value_rowid_batches(
    files: &[ParquetFileSpec],
    key_column: &str,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let first_builder = open_parquet_async(&files[0].file_path).await?;
    let value_type = first_builder
        .schema()
        .field_with_name(key_column)
        .map_err(|e| {
            Error::invalid_input(format!(
                "key column '{key_column}' not found in {}: {e}",
                files[0].file_path
            ))
        })?
        .data_type()
        .clone();

    // Widen any integer key column to Int64 so the stored index value type matches
    // the Int64 query keys callers send: the JVM `searchLongKeys` path marshals
    // Int/Short/Byte/Long all to Int64, and a BTree built over e.g. Int32 would
    // otherwise mismatch the Int64 `IsIn` query. Utf8 and other types are kept as-is.
    let widen_to_int64 = matches!(
        value_type,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    );
    let out_value_type = if widen_to_int64 {
        DataType::Int64
    } else {
        value_type
    };

    // value is field 0 (train_btree_index reads schema().field(0) for the value
    // type) and named VALUE_COLUMN_NAME; _rowid is the row identity column.
    let out_schema = Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, out_value_type, true),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));

    let mut batches: Vec<RecordBatch> = Vec::new();
    for (local_id, spec) in files.iter().enumerate() {
        let file_id = local_id as u64;
        let builder = open_parquet_async(&spec.file_path).await?;
        let mask = ProjectionMask::columns(builder.parquet_schema(), [key_column]);
        let mut stream = builder.with_projection(mask).build().map_err(|e| {
            Error::invalid_input(format!(
                "failed to build parquet reader for {}: {e}",
                spec.file_path
            ))
        })?;

        let mut row_in_file: u64 = 0;
        while let Some(batch) = stream.try_next().await.map_err(|e| {
            Error::invalid_input(format!(
                "error reading parquet batch from {}: {e}",
                spec.file_path
            ))
        })? {
            let n = batch.num_rows();
            let mut rids: Vec<u64> = Vec::with_capacity(n);
            for i in 0..n {
                rids.push((file_id << 32) | (row_in_file + i as u64));
            }
            let rid_array = Arc::new(UInt64Array::from(rids)) as ArrayRef;
            let raw_value_col = batch
                .column_by_name(key_column)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "key column '{key_column}' missing from batch in {}",
                        spec.file_path
                    ))
                })?
                .clone();
            let value_col = if widen_to_int64 {
                arrow_cast::cast(&raw_value_col, &DataType::Int64).map_err(|e| {
                    Error::invalid_input(format!(
                        "failed to widen key column '{key_column}' to Int64 in {}: {e}",
                        spec.file_path
                    ))
                })?
            } else {
                raw_value_col
            };
            let new_batch = RecordBatch::try_new(out_schema.clone(), vec![value_col, rid_array])
                .map_err(|e| {
                    Error::invalid_input(format!(
                        "failed to build (value, _rowid) batch for {}: {e}",
                        spec.file_path
                    ))
                })?;
            batches.push(new_batch);
            row_in_file += n as u64;
        }
    }
    Ok((out_schema, batches))
}

/// Build the sorted `(value, _rowid)` stream that feeds `train_btree_index`.
///
/// Unions every file's `(value, _rowid)` batches and sorts them ascending by
/// `value` with a DataFusion [`SortExec`] (over a single [`OneShotExec`]
/// partition). The trained BTree therefore sees globally value-sorted keys — the
/// invariant `train_btree_index` relies on to chunk pages by ascending value.
pub(super) async fn sorted_value_rowid_stream(
    files: &[ParquetFileSpec],
    key_column: &str,
) -> Result<SendableRecordBatchStream> {
    let (out_schema, batches) = collect_value_rowid_batches(files, key_column).await?;

    let input_stream = RecordBatchStreamAdapter::new(
        out_schema.clone(),
        futures::stream::iter(batches.into_iter().map(Ok::<RecordBatch, DataFusionError>)),
    );
    let source: Arc<dyn ExecutionPlan> = Arc::new(OneShotExec::new(Box::pin(input_stream)));

    let sort_expr = PhysicalSortExpr::new_default(
        col(VALUE_COLUMN_NAME, out_schema.as_ref())
            .map_err(|e| Error::io(format!("failed to resolve '{VALUE_COLUMN_NAME}' column: {e}")))?,
    );
    let sort = Arc::new(SortExec::new([sort_expr].into(), source));
    sort.execute(0, Arc::new(TaskContext::default()))
        .map_err(|e| Error::io(format!("sort exec for btree training failed: {e}")))
}
