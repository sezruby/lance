// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Parquet vector source for the external IVF-PQ build path.
//!
//! Reads vectors from `Vec<ParquetFileSpec>` and produces:
//!
//! - [`ParquetVectorSource::sample`] — a `FixedSizeListArray` of training vectors
//!   for kmeans + PQ codebook learning
//! - [`ParquetVectorSource::iter_batches`] — a `RecordBatchStream` of `(vec, _rowid)`
//!   batches feeding `IvfTransformer` + `shuffle_dataset`. `_rowid` is the encoded
//!   `(file_id_u32 << 32) | row_index_u32` rid the RFC pins.

use std::fs::File;
use std::sync::Arc;

use arrow::compute::concat;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::types::ParquetFileSpec;

/// The output column name for caller-controlled rids in batches yielded by
/// [`ParquetVectorSource::iter_batches`]. Matches the name `shuffle_dataset` and
/// `write_pq_partitions` already use internally for Lance's row identity.
pub const RID_COLUMN_NAME: &str = "_rowid";

/// Read vectors from a list of parquet files, attach the encoded rid, and feed
/// them into the IVF-PQ build pipeline.
pub struct ParquetVectorSource {
    files: Vec<ParquetFileSpec>,
    vector_column: String,
    /// Vector dimension, discovered when the first file's schema is read. Cached
    /// so `dim()` and `iter_batches()` agree.
    dim: usize,
}

impl ParquetVectorSource {
    /// Construct a source over `files` reading `vector_column`. Reads each file's
    /// footer to validate the column type and infer the dimension.
    pub fn try_new(files: Vec<ParquetFileSpec>, vector_column: &str) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::invalid_input(
                "ExternalIvfPqIndex requires at least one parquet file",
            ));
        }

        let first_dim = read_vector_dim(&files[0].file_path, vector_column)?;
        Ok(Self {
            files,
            vector_column: vector_column.to_string(),
            dim: first_dim,
        })
    }

    /// Vector dimension shared across all registered files.
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total row count, summed across files. Reads each footer.
    pub fn num_rows(&self) -> Result<u64> {
        let mut total = 0u64;
        for spec in &self.files {
            if spec.num_rows > 0 {
                total += spec.num_rows;
            } else {
                total += read_num_rows(&spec.file_path)?;
            }
        }
        Ok(total)
    }

    /// Sample up to `n` vectors uniformly across the file list for training.
    ///
    /// Strategy: round-robin read from each file's start until `n` is hit. For PQ
    /// training this is sufficient — kmeans + codebook quality is set by vector
    /// distribution, not by random row selection. A future iteration can add
    /// reservoir sampling if recall numbers say it's needed.
    pub async fn sample(&self, n: usize) -> Result<FixedSizeListArray> {
        let per_file = n.div_ceil(self.files.len()).max(1);
        let mut accumulated: Vec<ArrayRef> = Vec::new();
        let mut total = 0usize;

        for spec in &self.files {
            if total >= n {
                break;
            }
            let want = (n - total).min(per_file);
            let fsl = read_first_n_vectors(&spec.file_path, &self.vector_column, want)?;
            total += fsl.len();
            accumulated.push(Arc::new(fsl));
            if total >= n {
                break;
            }
        }

        let array_refs: Vec<&dyn Array> = accumulated.iter().map(|a| a.as_ref()).collect();
        let concatenated = concat(&array_refs).map_err(|e| {
            Error::invalid_input(format!("failed to concatenate sample batches: {e}"))
        })?;
        let fsl = concatenated
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| Error::invalid_input("concatenated sample is not a FixedSizeListArray"))?
            .clone();
        Ok(fsl)
    }

    /// Stream batches of `(vec, _rowid)` across all registered files. `_rowid` is
    /// `(file_id_u32 << 32) | row_index_u32`. Feeds `IvfTransformer` +
    /// `shuffle_dataset` during the build.
    ///
    /// Implementation note: parquet-rs uses blocking reads, so we materialize
    /// per-file batches eagerly into a `Vec` and stream them. Memory cost is
    /// bounded by the largest single parquet file's row groups since we don't
    /// hold all files in memory at once — but we do hold one file's worth at a
    /// time. A truly streaming variant (spawn_blocking ladder) lands later if
    /// large-file memory becomes a concern.
    pub fn iter_batches(&self) -> Result<impl RecordBatchStream + Unpin + 'static> {
        let out_schema = Arc::new(Schema::new(vec![
            Field::new(
                self.vector_column.clone(),
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim as i32,
                ),
                false,
            ),
            Field::new(RID_COLUMN_NAME, DataType::UInt64, false),
        ]));

        let batches = collect_rid_annotated_batches(
            self.files.clone(),
            self.vector_column.clone(),
            out_schema.clone(),
        )?;

        let stream =
            futures::stream::iter(batches.into_iter().map(|b| Ok::<RecordBatch, Error>(b)));
        Ok(RecordBatchStreamAdapter::new(out_schema, stream))
    }
}

// ---- helpers ------------------------------------------------------------------------

fn parquet_open_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("failed to open parquet file {path}: {source}"))
}

fn parquet_meta_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!(
        "failed to read parquet metadata for {path}: {source}"
    ))
}

fn parquet_reader_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!(
        "failed to build parquet reader for {path}: {source}"
    ))
}

fn parquet_batch_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("error reading parquet batch from {path}: {source}"))
}

fn read_vector_dim(path: &str, column: &str) -> Result<usize> {
    let file = File::open(path).map_err(|e| parquet_open_err(path, e))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| parquet_meta_err(path, e))?;
    let arrow_schema = builder.schema().clone();
    let field = arrow_schema.field_with_name(column).map_err(|e| {
        Error::invalid_input(format!("vector column '{column}' not found in {path}: {e}"))
    })?;
    match field.data_type() {
        DataType::FixedSizeList(_, n) => Ok(*n as usize),
        DataType::List(_) | DataType::LargeList(_) => {
            // Spark and most JVM parquet writers emit `List<Float32>` even when every row
            // has the same length — Arrow's fixed-size-list metadata round-trips poorly
            // through parquet. Probe the first batch's first row to infer the dimension;
            // the read path enforces every subsequent row to match.
            let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);
            let reader = builder
                .with_projection(mask)
                .with_batch_size(1)
                .build()
                .map_err(|e| parquet_reader_err(path, e))?;
            for batch_result in reader {
                let batch = batch_result.map_err(|e| parquet_batch_err(path, e))?;
                if batch.num_rows() == 0 {
                    continue;
                }
                let col = batch.column_by_name(column).ok_or_else(|| {
                    Error::invalid_input(format!("column '{column}' missing in {path}"))
                })?;
                if let Some(la) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
                    let first_len = la.value_length(0);
                    return Ok(first_len as usize);
                }
                if let Some(la) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
                    let first_len = la.value_length(0);
                    return Ok(first_len as usize);
                }
                break;
            }
            Err(Error::invalid_input(format!(
                "vector column '{column}' in {path} is List<Float32> but file contains no rows"
            )))
        }
        other => Err(Error::invalid_input(format!(
            "vector column '{column}' in {path} must be FixedSizeList<Float32> or List<Float32>, got {other:?}"
        ))),
    }
}

fn read_num_rows(path: &str) -> Result<u64> {
    let file = File::open(path).map_err(|e| parquet_open_err(path, e))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| parquet_meta_err(path, e))?;
    Ok(builder.metadata().file_metadata().num_rows() as u64)
}

/// Coerce a column array (FixedSizeList or List) to FixedSizeListArray, validating that
/// every row has length `expected_dim`. Spark and most JVM parquet writers emit List, so
/// this is the common-case path.
pub(crate) fn coerce_to_fsl(col: &ArrayRef, expected_dim: usize) -> Result<FixedSizeListArray> {
    if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        if fsl.value_length() as usize != expected_dim {
            return Err(Error::invalid_input(format!(
                "vector column dim {} != expected {}",
                fsl.value_length(),
                expected_dim
            )));
        }
        return Ok(fsl.clone());
    }
    if let Some(la) = col.as_any().downcast_ref::<ListArray>() {
        let values = la.values();
        // Validate: every offset increment must equal expected_dim; no nulls; values are Float32.
        let offsets = la.offsets();
        for i in 0..la.len() {
            let len = (offsets[i + 1] - offsets[i]) as usize;
            if len != expected_dim {
                return Err(Error::invalid_input(format!(
                    "row {i} of List<Float32> column has length {len}, expected {expected_dim}"
                )));
            }
        }
        let f32_values = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "List vector column has non-Float32 inner type {:?}",
                    values.data_type()
                ))
            })?
            .clone();
        return FixedSizeListArray::try_new_from_values(f32_values, expected_dim as i32)
            .map_err(|e| Error::invalid_input(format!("List → FSL conversion failed: {e}")));
    }
    Err(Error::invalid_input(format!(
        "vector column has unsupported type {:?}; expected FixedSizeList<Float32> or List<Float32>",
        col.data_type()
    )))
}

fn read_first_n_vectors(path: &str, column: &str, n: usize) -> Result<FixedSizeListArray> {
    let dim = read_vector_dim(path, column)?;
    let file = File::open(path).map_err(|e| parquet_open_err(path, e))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| parquet_meta_err(path, e))?;
    let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(n.max(1024))
        .build()
        .map_err(|e| parquet_reader_err(path, e))?;

    let mut collected = 0usize;
    let mut fsl_chunks: Vec<FixedSizeListArray> = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|e| parquet_batch_err(path, e))?;
        let take_n = (n - collected).min(batch.num_rows());
        let col = batch
            .column_by_name(column)
            .ok_or_else(|| {
                Error::invalid_input(format!("column '{column}' missing from batch in {path}"))
            })?
            .slice(0, take_n);
        let fsl = coerce_to_fsl(&col, dim)?;
        fsl_chunks.push(fsl);
        collected += take_n;
        if collected >= n {
            break;
        }
    }

    let array_refs: Vec<&dyn Array> = fsl_chunks.iter().map(|a| a as &dyn Array).collect();
    let concatenated = concat(&array_refs)
        .map_err(|e| Error::invalid_input(format!("failed to concat batches from {path}: {e}")))?;
    let fsl = concatenated
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "vector column '{column}' did not yield FixedSizeListArray in {path}"
            ))
        })?
        .clone();
    Ok(fsl)
}

fn collect_rid_annotated_batches(
    files: Vec<ParquetFileSpec>,
    column: String,
    out_schema: SchemaRef,
) -> Result<Vec<RecordBatch>> {
    // Coerce on the way in: List<Float32> rows from Spark/JVM writers get reshaped to
    // FixedSizeListArray so the downstream IvfTransformer / shuffle path sees the schema
    // it expects (out_schema declares FixedSizeList).
    let dim_field = out_schema.field_with_name(&column).map_err(|e| {
        Error::invalid_input(format!("column '{column}' missing in out_schema: {e}"))
    })?;
    let dim = match dim_field.data_type() {
        DataType::FixedSizeList(_, n) => *n as usize,
        other => {
            return Err(Error::invalid_input(format!(
                "out_schema vector column '{column}' must be FixedSizeList, got {other:?}"
            )));
        }
    };

    let mut out: Vec<RecordBatch> = Vec::new();
    for (file_id, spec) in files.iter().enumerate() {
        let file = File::open(&spec.file_path).map_err(|e| parquet_open_err(&spec.file_path, e))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| parquet_meta_err(&spec.file_path, e))?;
        let mask = ProjectionMask::columns(builder.parquet_schema(), [column.as_str()]);
        let reader = builder
            .with_projection(mask)
            .build()
            .map_err(|e| parquet_reader_err(&spec.file_path, e))?;

        let mut row_in_file: u64 = 0;
        for batch_result in reader {
            let batch = batch_result.map_err(|e| parquet_batch_err(&spec.file_path, e))?;
            let n = batch.num_rows();
            let mut rids: Vec<u64> = Vec::with_capacity(n);
            for i in 0..n {
                rids.push(((file_id as u64) << 32) | (row_in_file + i as u64));
            }
            let rid_array = Arc::new(UInt64Array::from(rids)) as ArrayRef;
            let raw_vec_col = batch
                .column_by_name(&column)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "column '{}' missing from batch in {}",
                        column, spec.file_path
                    ))
                })?
                .clone();
            let vec_fsl = coerce_to_fsl(&raw_vec_col, dim)?;
            let vec_array: ArrayRef = Arc::new(vec_fsl);
            let new_batch = RecordBatch::try_new(out_schema.clone(), vec![vec_array, rid_array])
                .map_err(|e| {
                    Error::invalid_input(format!("failed to build (vec,_rowid) batch: {e}"))
                })?;
            out.push(new_batch);
            row_in_file += n as u64;
        }
    }
    Ok(out)
}

// Keep Float32Array referenced for tests; harmless on non-test builds.
#[allow(dead_code)]
const _: fn() = || {
    let _: Option<Float32Array> = None;
};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_array::RecordBatch;
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn make_parquet(path: &PathBuf, num_rows: usize, dim: usize, seed: f32) {
        let values: Vec<f32> = (0..num_rows * dim).map(|i| seed + i as f32).collect();
        let flat = Float32Array::from(values);
        let fsl = FixedSizeListArray::try_new_from_values(flat, dim as i32).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            fsl.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl) as ArrayRef]).unwrap();

        let file = std::fs::File::create(path).unwrap();
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn source_dim_and_num_rows() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("a.parquet");
        make_parquet(&p, 100, 8, 0.0);
        let spec = ParquetFileSpec::of(p.to_str().unwrap());
        let src = ParquetVectorSource::try_new(vec![spec], "vec").unwrap();
        assert_eq!(src.dim(), 8);
        assert_eq!(src.num_rows().unwrap(), 100);
    }

    #[tokio::test]
    async fn sample_returns_requested_count() {
        let tmp = TempDir::new().unwrap();
        let mut files = Vec::new();
        for i in 0..3 {
            let p = tmp.path().join(format!("part-{i}.parquet"));
            make_parquet(&p, 50, 4, i as f32 * 1000.0);
            files.push(ParquetFileSpec::of(p.to_str().unwrap()));
        }
        let src = ParquetVectorSource::try_new(files, "vec").unwrap();
        let sample = src.sample(60).await.unwrap();
        assert_eq!(sample.len(), 60);
        assert_eq!(sample.value_length(), 4);
    }

    #[tokio::test]
    async fn iter_batches_assigns_correct_rids() {
        let tmp = TempDir::new().unwrap();
        let mut files = Vec::new();
        for i in 0..2 {
            let p = tmp.path().join(format!("part-{i}.parquet"));
            make_parquet(&p, 3, 2, i as f32 * 100.0);
            files.push(ParquetFileSpec::of(p.to_str().unwrap()));
        }
        let src = ParquetVectorSource::try_new(files, "vec").unwrap();
        let stream = src.iter_batches().unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let mut all_rids: Vec<u64> = Vec::new();
        for batch in &batches {
            let rid_arr = batch
                .column_by_name(RID_COLUMN_NAME)
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..rid_arr.len() {
                all_rids.push(rid_arr.value(i));
            }
        }

        assert_eq!(
            all_rids,
            vec![0, 1, 2, (1u64 << 32), (1u64 << 32) | 1, (1u64 << 32) | 2]
        );
    }
}
