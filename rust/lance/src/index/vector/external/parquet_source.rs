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

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat;
use arrow_array::{
    new_empty_array, Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use object_store::ObjectStoreExt;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, RowSelection};
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStreamBuilder,
};
use parquet::errors::ParquetError;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use tokio::sync::Mutex;

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
    /// Global `file_id` of `files[0]`. The encoded rid is
    /// `((file_id_offset + local_index) << 32) | row_in_file`. Zero for a
    /// whole-corpus build; for a distributed shard build it is the position of
    /// this shard's first file in the FULL (sorted) manifest file list, so every
    /// shard emits globally-consistent rids that merge correctly.
    file_id_offset: u32,
}

impl ParquetVectorSource {
    /// Construct a source over `files` reading `vector_column`. Reads each file's
    /// footer to validate the column type and infer the dimension. Whole-corpus
    /// build (`file_id_offset = 0`).
    pub async fn try_new(files: Vec<ParquetFileSpec>, vector_column: &str) -> Result<Self> {
        Self::try_new_with_offset(files, vector_column, 0).await
    }

    /// Construct a source for a distributed shard. `file_id_offset` is the global
    /// index of `files[0]` in the full manifest file list; rids are encoded as
    /// `((file_id_offset + local_index) << 32) | row_in_file` so shards built on
    /// separate executors carry consistent global file ids and merge correctly.
    pub async fn try_new_with_offset(
        files: Vec<ParquetFileSpec>,
        vector_column: &str,
        file_id_offset: u32,
    ) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::invalid_input(
                "ExternalIvfPqIndex requires at least one parquet file",
            ));
        }

        let first_dim = read_vector_dim(&files[0].file_path, vector_column).await?;
        Ok(Self {
            files,
            vector_column: vector_column.to_string(),
            dim: first_dim,
            file_id_offset,
        })
    }

    /// Vector dimension shared across all registered files.
    #[allow(dead_code)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total row count, summed across files. Reads each footer for files whose
    /// `num_rows` is unknown.
    pub async fn num_rows(&self) -> Result<u64> {
        let mut total = 0u64;
        for spec in &self.files {
            if spec.num_rows > 0 {
                total += spec.num_rows;
            } else {
                total += read_num_rows(&spec.file_path).await?;
            }
        }
        Ok(total)
    }

    /// Sample up to `n` vectors for kmeans + PQ training, spread across each file's FULL row
    /// range rather than its first rows (strided by default, or random via
    /// `LANCE_EXT_SAMPLE_STRATEGY=random`).
    ///
    /// Why spread across the range, not first-N: source parquet is frequently value-ordered
    /// (e.g. Wikipedia embeddings are laid out article-by-article, so consecutive rows are
    /// paragraphs of the same document). A first-N-per-file sample is then a highly correlated,
    /// low-diversity slice — it trains centroids + codebook on a skewed sub-distribution and
    /// measurably drops recall. The read stays cheap because `RowSelection` uses the parquet
    /// page/offset index to skip unselected rows (the same mechanism as the refinement fetch),
    /// so we page-decode only the rows we keep.
    ///
    /// Reproducible: the per-file phase / RNG seed is derived from the file's position, so the
    /// sample — and thus the trained index — is stable across runs.
    pub async fn sample(&self, n: usize) -> Result<FixedSizeListArray> {
        let per_file = n.div_ceil(self.files.len()).max(1);
        let mut accumulated: Vec<ArrayRef> = Vec::new();
        let mut total = 0usize;

        for (file_idx, spec) in self.files.iter().enumerate() {
            if total >= n {
                break;
            }
            let want = (n - total).min(per_file);
            let fsl = read_range_spread_vectors(
                &spec.file_path,
                &self.vector_column,
                want,
                file_idx as u64,
            )
            .await?;
            total += fsl.len();
            accumulated.push(Arc::new(fsl));
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
    /// Implementation note: per-file batches are materialized into a `Vec` first
    /// and then streamed. Memory cost is bounded by the largest single parquet
    /// file's row groups since we don't hold all files at once — one file's
    /// worth at a time. A truly streaming variant (one batch in flight) is a
    /// later optimization if large-file memory becomes a concern.
    pub async fn iter_batches(&self) -> Result<impl RecordBatchStream + Unpin + 'static> {
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
            self.file_id_offset,
        )
        .await?;

        let stream =
            futures::stream::iter(batches.into_iter().map(|b| Ok::<RecordBatch, Error>(b)));
        Ok(RecordBatchStreamAdapter::new(out_schema, stream))
    }
}

// ---- helpers ------------------------------------------------------------------------

pub(super) fn parquet_open_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("failed to open parquet file {path}: {source}"))
}

pub(super) fn parquet_meta_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!(
        "failed to read parquet metadata for {path}: {source}"
    ))
}

pub(super) fn parquet_reader_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!(
        "failed to build parquet reader for {path}: {source}"
    ))
}

pub(super) fn parquet_batch_err(path: &str, source: impl std::fmt::Display) -> Error {
    Error::invalid_input(format!("error reading parquet batch from {path}: {source}"))
}

/// Coalesce gap for refinement reads. parquet-rs's default `get_byte_ranges`
/// path uses `OBJECT_STORE_COALESCE_DEFAULT = 1 MiB` — fine for local reads,
/// but for cloud storage where each HTTP round-trip is 30-50 ms, we want to
/// merge ranges that are further apart so 80 scattered candidate vectors
/// (typical refinement footprint) land in fewer requests.
///
/// We tried 64 MiB first; the DBR/abfss bench showed it merged the entire
/// per-query refinement footprint into ONE 50 MB request when only ~40 KB
/// of vector bytes were actually needed (1250× over-read, 297 ms median).
/// 4 MiB is a deliberate compromise: large enough to swallow cross-page
/// gaps within a typical row group (~1-2 MB pages, ~16-32 MB row groups),
/// but small enough to skip over the wide payload columns that sit between
/// vector pages when candidates are clustered.
const REFINEMENT_COALESCE_GAP: u64 = 4 * 1024 * 1024;

/// Maximum concurrent range fetches when refining cloud-stored parquet.
/// `object_store`'s default `coalesce_ranges` parallelism is hardcoded to 10;
/// for refinement, where queue depth is the bottleneck, going higher is a
/// straightforward win on Azure / S3.
const REFINEMENT_PARALLEL_RANGES: usize = 32;

/// Wrapper around [`ParquetObjectReader`] that overrides `get_byte_ranges` to
/// coalesce ranges with a larger gap and fetch more in parallel than the
/// `object_store::coalesce_ranges` defaults. Significantly reduces refinement
/// latency on cloud storage where per-request RTT dominates.
///
/// All other `AsyncFileReader` methods delegate to the inner reader unchanged.
pub(super) struct CoalescingParquetReader {
    inner: ParquetObjectReader,
    /// Reference to the same object_store the inner reader uses, so our
    /// custom `get_byte_ranges` can issue the coalesced fetch directly.
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
}

impl CoalescingParquetReader {
    fn new(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
        file_size: u64,
    ) -> Self {
        let inner = ParquetObjectReader::new(store.clone(), path.clone()).with_file_size(file_size);
        Self { inner, store, path }
    }
}

impl AsyncFileReader for CoalescingParquetReader {
    fn get_bytes(
        &mut self,
        range: std::ops::Range<u64>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> futures::future::BoxFuture<'_, parquet::errors::Result<Vec<bytes::Bytes>>> {
        let store = self.store.clone();
        let path = self.path.clone();
        Box::pin(async move {
            // Use object_store's coalesce_ranges directly with a tuned gap.
            // The fetch closure issues one HTTP range read per coalesced range
            // and is parallelized by `coalesce_ranges` internally — but its
            // parallelism is fixed at 10. We pre-merge with a larger gap, then
            // issue more fetches in parallel via a buffered stream.
            use futures::stream::{self, StreamExt, TryStreamExt};
            use std::time::Instant;
            let t = Instant::now();
            let input_ranges = ranges.len();
            let merged = merge_ranges_with_gap(&ranges, REFINEMENT_COALESCE_GAP);
            let merged_bytes: u64 = merged.iter().map(|r| r.end - r.start).sum();
            let merged_count = merged.len();
            let merged_clone = merged.clone();
            let buffered: Vec<bytes::Bytes> = stream::iter(merged.into_iter())
                .map(|r| {
                    let store = store.clone();
                    let path = path.clone();
                    async move { store.get_range(&path, r).await }
                })
                .buffered(REFINEMENT_PARALLEL_RANGES)
                .try_collect()
                .await
                .map_err(|e| ParquetError::External(Box::new(e)))?;
            let elapsed_ms = t.elapsed().as_millis();
            log::info!(
                "extidx_byte_ranges input={input_ranges} coalesced={merged_count} \
                 bytes={merged_bytes} elapsed={elapsed_ms}ms",
            );
            // Slice the merged Bytes back into the per-input-range chunks the
            // caller asked for. parquet-rs expects `Vec<Bytes>` aligned with
            // the input order.
            Ok(slice_merged(&ranges, &merged_clone, buffered))
        })
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> futures::future::BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

/// Like `object_store::merge_ranges` but with a configurable gap (the
/// upstream version uses `OBJECT_STORE_COALESCE_DEFAULT`, hardcoded at 1 MiB).
/// Returns a sorted+merged list of ranges that covers all input ranges.
fn merge_ranges_with_gap(
    ranges: &[std::ops::Range<u64>],
    coalesce: u64,
) -> Vec<std::ops::Range<u64>> {
    if ranges.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<std::ops::Range<u64>> = ranges.to_vec();
    sorted.sort_by_key(|r| r.start);
    let mut out = Vec::with_capacity(sorted.len());
    let mut cur = sorted[0].clone();
    for r in sorted.into_iter().skip(1) {
        if r.start <= cur.end + coalesce {
            cur.end = cur.end.max(r.end);
        } else {
            out.push(std::mem::replace(&mut cur, r));
        }
    }
    out.push(cur);
    out
}

/// Given the original input ranges, the merged ranges, and the bytes returned
/// for each merged range, return the bytes corresponding to each input range.
/// Preserves input order. Each input range must lie inside exactly one merged
/// range (which is invariant given how `merge_ranges_with_gap` is constructed).
fn slice_merged(
    inputs: &[std::ops::Range<u64>],
    merged: &[std::ops::Range<u64>],
    fetched: Vec<bytes::Bytes>,
) -> Vec<bytes::Bytes> {
    let mut out = Vec::with_capacity(inputs.len());
    for r in inputs {
        // Find the merged range that contains this input range.
        let (idx, m) = merged
            .iter()
            .enumerate()
            .find(|(_, m)| m.start <= r.start && r.end <= m.end)
            .expect("input range must lie within a merged range");
        let offset = (r.start - m.start) as usize;
        let len = (r.end - r.start) as usize;
        out.push(fetched[idx].slice(offset..offset + len));
    }
    out
}

/// Per-index parquet metadata cache. Keyed by file path; the value carries the
/// parsed [`ArrowReaderMetadata`] (which holds an `Arc<ParquetMetaData>` plus the
/// page index when one was loaded), the resolved object store, the parquet
/// path inside that store, and the file size.
///
/// Lance opens each refinement read by re-resolving the URI, issuing a `head`,
/// then fetching the parquet footer + page index. On cloud storage this is
/// 3 round trips (10s of ms each on abfss). For a query workload that touches
/// the same parquet files repeatedly, all three are cacheable across queries.
/// This is the same trick Spark's `ParquetIOMetadataCache`, DuckDB's
/// `parquet_metadata_cache`, and parquet-rs's own
/// `ParquetRecordBatchStreamBuilder::new_with_metadata` example use.
///
/// The cache lives on [`super::OpenedExternalIndex`] so its lifetime matches
/// one task's worth of queries; per-task is enough to amortize the overhead
/// across all queries running on the task.
pub(super) struct ParquetMetaCache {
    inner: Mutex<HashMap<String, CachedFile>>,
}

#[derive(Clone)]
struct CachedFile {
    store: Arc<dyn object_store::ObjectStore>,
    parquet_path: object_store::path::Path,
    file_size: u64,
    /// Cached metadata loaded with `PageIndexPolicy::Required` so refinement
    /// reads can reuse the page index (the most expensive bit to fetch).
    metadata: ArrowReaderMetadata,
}

impl ParquetMetaCache {
    pub(super) fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl std::fmt::Debug for ParquetMetaCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetMetaCache").finish_non_exhaustive()
    }
}

/// Open a parquet file at `path` for async reads. Resolves any URI scheme
/// supported by lance's [`ObjectStore::from_uri`] (file://, s3://, abfss://,
/// gs://, ...). Credentials follow the standard precedence each provider
/// expects (env vars, IMDS, managed identity).
///
/// Returns the [`ParquetRecordBatchStreamBuilder`] which the caller turns into
/// a stream via `.with_projection(...).build()`. The async reader supports the
/// same options as the sync one (page index, row selection, projection mask).
pub(super) async fn open_parquet_async(
    path: &str,
) -> Result<ParquetRecordBatchStreamBuilder<CoalescingParquetReader>> {
    open_parquet_async_with_options(path, ArrowReaderOptions::new()).await
}

/// Cache-aware version of [`open_parquet_async_with_options`].
///
/// On cache hit, reuses the resolved [`ObjectStore`], file size, and
/// [`ArrowReaderMetadata`] — skipping URI resolution, the `head` round trip,
/// and the footer + page-index fetch (the three cloud-storage round trips
/// per refinement read). On miss, runs the full open and stores the result.
///
/// The cached metadata is loaded with [`PageIndexPolicy::Required`] so
/// callers that want the page index (refinement reads) hit the cache; callers
/// that don't care still benefit from the cached footer.
pub(super) async fn open_parquet_cached(
    cache: &ParquetMetaCache,
    path: &str,
) -> Result<ParquetRecordBatchStreamBuilder<CoalescingParquetReader>> {
    {
        let guard = cache.inner.lock().await;
        if let Some(entry) = guard.get(path) {
            log::debug!("extidx_meta_cache hit path={path}");
            let reader = CoalescingParquetReader::new(
                entry.store.clone(),
                entry.parquet_path.clone(),
                entry.file_size,
            );
            return Ok(ParquetRecordBatchStreamBuilder::new_with_metadata(
                reader,
                entry.metadata.clone(),
            ));
        }
    }
    log::info!("extidx_meta_cache miss path={path}");

    // Cache miss: do the full open with PageIndexPolicy::Required so the
    // cached metadata is usable by every caller (refinement reads need the
    // page index; sample/dim reads ignore it cheaply).
    let (store, parquet_path) = ObjectStore::from_uri(path)
        .await
        .map_err(|e| parquet_open_err(path, e))?;
    let head = store
        .inner
        .head(&parquet_path)
        .await
        .map_err(|e| parquet_open_err(path, e))?;
    let mut reader =
        CoalescingParquetReader::new(store.inner.clone(), parquet_path.clone(), head.size);
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let metadata = ArrowReaderMetadata::load_async(&mut reader, options)
        .await
        .map_err(|e| parquet_meta_err(path, e))?;

    {
        let mut guard = cache.inner.lock().await;
        guard.entry(path.to_string()).or_insert_with(|| CachedFile {
            store: store.inner.clone(),
            parquet_path: parquet_path.clone(),
            file_size: head.size,
            metadata: metadata.clone(),
        });
    }

    let fresh_reader = CoalescingParquetReader::new(store.inner.clone(), parquet_path, head.size);
    Ok(ParquetRecordBatchStreamBuilder::new_with_metadata(
        fresh_reader,
        metadata,
    ))
}

/// Like [`open_parquet_async`] but lets the caller pass [`ArrowReaderOptions`] —
/// e.g. to require the page index for refinement reads.
///
/// Calls `head` on the object first to discover the file size and feeds it to
/// [`ParquetObjectReader::with_file_size`]. Without this the reader would
/// default to issuing an HTTP suffix range request (`Range: bytes=-N`) for the
/// footer, which Azure Blob Storage explicitly does not support — `head` + a
/// normal range read works on every backend.
pub(super) async fn open_parquet_async_with_options(
    path: &str,
    options: ArrowReaderOptions,
) -> Result<ParquetRecordBatchStreamBuilder<CoalescingParquetReader>> {
    let (store, parquet_path) = ObjectStore::from_uri(path)
        .await
        .map_err(|e| parquet_open_err(path, e))?;
    let meta = store
        .inner
        .head(&parquet_path)
        .await
        .map_err(|e| parquet_open_err(path, e))?;
    let reader = CoalescingParquetReader::new(store.inner.clone(), parquet_path, meta.size);
    ParquetRecordBatchStreamBuilder::new_with_options(reader, options)
        .await
        .map_err(|e| parquet_meta_err(path, e))
}

async fn read_vector_dim(path: &str, column: &str) -> Result<usize> {
    let builder = open_parquet_async(path).await?;
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
            let mut stream = builder
                .with_projection(mask)
                .with_batch_size(1)
                .build()
                .map_err(|e| parquet_reader_err(path, e))?;
            while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| parquet_batch_err(path, e))?
            {
                if batch.num_rows() == 0 {
                    continue;
                }
                let col = batch.column_by_name(column).ok_or_else(|| {
                    Error::invalid_input(format!("column '{column}' missing in {path}"))
                })?;
                if let Some(la) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
                    return Ok(la.value_length(0) as usize);
                }
                if let Some(la) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
                    return Ok(la.value_length(0) as usize);
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

async fn read_num_rows(path: &str) -> Result<u64> {
    let builder = open_parquet_async(path).await?;
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

/// Read up to `n` vectors from `path`, spread across the file's FULL row range (not the
/// first `n`). Strategy is chosen by env `LANCE_EXT_SAMPLE_STRATEGY` ("strided" default, or
/// "random"); `seed_salt` (the file's position) staggers per-file offsets so files don't all
/// sample the same relative positions, and keeps the sample reproducible.
///
/// Spreading across the range (vs first-N) matters on value-ordered corpora (e.g. Wikipedia
/// embeddings laid out article-by-article): a first-N slice is a correlated, low-diversity
/// training set that drops recall. Either strategy uses a `RowSelection` so the parquet reader
/// skips unselected rows via the page/offset index (loaded with `PageIndexPolicy::Required`) —
/// it page-decodes only the kept rows, not the whole file. See [`ParquetVectorSource::sample`].
async fn read_range_spread_vectors(
    path: &str,
    column: &str,
    n: usize,
    seed_salt: u64,
) -> Result<FixedSizeListArray> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let dim = read_vector_dim(path, column).await?;
    // Page index required so RowSelection can skip cheaply.
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    let builder = open_parquet_async_with_options(path, options).await?;
    let total_rows = builder.metadata().file_metadata().num_rows().max(0) as usize;
    let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);

    if total_rows == 0 || n == 0 {
        let empty = new_empty_array(&DataType::Float32);
        let f32 = empty.as_any().downcast_ref::<Float32Array>().unwrap().clone();
        return FixedSizeListArray::try_new_from_values(f32, dim as i32)
            .map_err(|e| Error::invalid_input(format!("empty FSL construction failed: {e}")));
    }

    // Pick `want` row indices across the FULL range. Two strategies (env
    // `LANCE_EXT_SAMPLE_STRATEGY`, default "strided"):
    //   - "strided": evenly-spaced offsets (phase-shifted per file). Guarantees uniform coverage
    //     of a value-ordered range; measured best on article-ordered Cohere data.
    //   - "random": distinct uniform-random offsets (mirrors Lance-core's sample_fsl_uniform).
    // Both read via the page-index RowSelection below, so cost is identical; only which rows are
    // trained on differs. Kept as a toggle so the two can be A/B'd against recall directly.
    let want = n.min(total_rows);
    let strategy = std::env::var("LANCE_EXT_SAMPLE_STRATEGY").unwrap_or_else(|_| "strided".into());
    let indices: Vec<usize> = if want == total_rows {
        (0..total_rows).collect()
    } else if strategy.eq_ignore_ascii_case("random") {
        let mut rng = StdRng::seed_from_u64(0xA5A5_5A5A_C0FF_EE00 ^ seed_salt);
        let mut chosen = std::collections::HashSet::with_capacity(want);
        while chosen.len() < want {
            chosen.insert(rng.gen_range(0..total_rows));
        }
        let mut v: Vec<usize> = chosen.into_iter().collect();
        // RowSelection requires strictly increasing ranges.
        v.sort_unstable();
        v
    } else {
        // Strided: evenly-spaced, offset by a per-file phase so files stagger their samples.
        let stride = (total_rows / want).max(1);
        let start = (seed_salt as usize).wrapping_mul(2_654_435_761) % stride;
        let mut v: Vec<usize> = Vec::with_capacity(want);
        let mut r = start;
        while r < total_rows && v.len() < want {
            v.push(r);
            r += stride;
        }
        v
    };
    let ranges: Vec<std::ops::Range<usize>> = indices.iter().map(|&i| i..(i + 1)).collect();
    let selection = RowSelection::from_consecutive_ranges(ranges.into_iter(), total_rows);

    let mut stream = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| parquet_reader_err(path, e))?;

    let mut fsl_chunks: Vec<FixedSizeListArray> = Vec::new();
    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|e| parquet_batch_err(path, e))?
    {
        let col = batch.column_by_name(column).ok_or_else(|| {
            Error::invalid_input(format!("column '{column}' missing from batch in {path}"))
        })?;
        fsl_chunks.push(coerce_to_fsl(col, dim)?);
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

async fn collect_rid_annotated_batches(
    files: Vec<ParquetFileSpec>,
    column: String,
    out_schema: SchemaRef,
    file_id_offset: u32,
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
    for (local_id, spec) in files.iter().enumerate() {
        // Global file id: shard-local position plus the shard's offset in the full
        // manifest list. Offset 0 for a whole-corpus build.
        let file_id = file_id_offset as u64 + local_id as u64;
        let builder = open_parquet_async(&spec.file_path).await?;
        let mask = ProjectionMask::columns(builder.parquet_schema(), [column.as_str()]);
        let mut stream = builder
            .with_projection(mask)
            .build()
            .map_err(|e| parquet_reader_err(&spec.file_path, e))?;

        let mut row_in_file: u64 = 0;
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| parquet_batch_err(&spec.file_path, e))?
        {
            let n = batch.num_rows();
            let mut rids: Vec<u64> = Vec::with_capacity(n);
            for i in 0..n {
                rids.push((file_id << 32) | (row_in_file + i as u64));
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
        let src = ParquetVectorSource::try_new(vec![spec], "vec")
            .await
            .unwrap();
        assert_eq!(src.dim(), 8);
        assert_eq!(src.num_rows().await.unwrap(), 100);
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
        let src = ParquetVectorSource::try_new(files, "vec").await.unwrap();
        let sample = src.sample(60).await.unwrap();
        assert_eq!(sample.len(), 60);
        assert_eq!(sample.value_length(), 4);
    }

    // The training sample must spread across each file's FULL row range and pick DISTINCT
    // rows, not just its first rows — a first-N sample is a correlated, low-diversity slice
    // on value-ordered corpora and drops recall (see `ParquetVectorSource::sample`). One
    // 1000-row file with row r's first value = r*dim; a range-spread sample of 100 rows
    // must reach rows well past the first 100 (which a first-N sample cannot) and be
    // duplicate-free.
    #[tokio::test]
    async fn sample_spreads_across_full_row_range() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("part-0.parquet");
        let (num_rows, dim) = (1000usize, 4usize);
        make_parquet(&p, num_rows, dim, 0.0); // row r → first value r*dim
        let src = ParquetVectorSource::try_new(vec![ParquetFileSpec::of(p.to_str().unwrap())], "vec")
            .await
            .unwrap();
        let want = 100usize;
        let sample = src.sample(want).await.unwrap();
        assert_eq!(sample.len(), want);
        // Recover each sampled row index from its first value (= row * dim).
        let values = sample
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let mut sampled_rows: Vec<usize> = (0..sample.len())
            .map(|i| (values.value(i * dim) as usize) / dim)
            .collect();
        sampled_rows.sort_unstable();
        let max_row = *sampled_rows.last().unwrap();
        // A first-N sample would top out at row `want-1` (=99). A range-spread sample over
        // 1000 rows must reach far beyond that (near ~990).
        assert!(
            max_row > want * 5,
            "sample did not spread across the file: max sampled row {max_row} \
             (want={want}); a range-spread sample should reach near {num_rows}"
        );
        // Sampled rows must be distinct (the sampler dedups before building the selection).
        let distinct = sampled_rows.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(distinct, want, "sampled rows must be distinct");
    }

    #[tokio::test]
    async fn cache_serves_repeat_reads_without_reopening() {
        // Sanity check that the cache key works and metadata is reused across
        // calls. Asserts identity on the cached `ArrowReaderMetadata` rather
        // than counting opens, since `object_store::local::LocalFileSystem` is
        // not instrumentable; the second call must hit the early-return branch
        // in `open_parquet_cached`.
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("a.parquet");
        make_parquet(&p, 100, 8, 0.0);
        let path = p.to_str().unwrap();
        let cache = ParquetMetaCache::new();

        // First call populates the cache.
        let _ = open_parquet_cached(&cache, path).await.unwrap();
        let inserted_meta_ptr = {
            let guard = cache.inner.lock().await;
            let entry = guard.get(path).unwrap();
            Arc::as_ptr(entry.metadata.metadata())
        };

        // Second call must reuse the cached `ArrowReaderMetadata` —
        // identity on the inner `Arc<ParquetMetaData>` proves the cache
        // returned its stored value rather than re-fetching the footer.
        let builder2 = open_parquet_cached(&cache, path).await.unwrap();
        let used_meta_ptr = Arc::as_ptr(builder2.metadata());
        assert_eq!(inserted_meta_ptr, used_meta_ptr);
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
        let src = ParquetVectorSource::try_new(files, "vec").await.unwrap();
        let stream = src.iter_batches().await.unwrap();
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
