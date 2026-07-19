// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for [`lance::index::vector::external::ExternalIvfPqIndex`] — the
//! IVF-PQ vector index built directly over source parquet files (no rewrite into Lance).
//!
//! Mirrors the JNI surface (`java/lance-jni/src/external_index.rs`) so Python callers
//! (e.g. Ray tasks/actors) can build, open, and query the index without a JVM. Search
//! results are returned as lists of `(file_path, row_index, distance)` tuples;
//! `fetch_rows` returns a pyarrow `RecordBatch`.

use std::sync::Arc;

use arrow::pyarrow::ToPyArrow;
use lance::index::vector::external::{
    build_shard_to_parquet, merge_shards_to_index, train_broadcast_payload, BroadcastPayload,
    ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetFileSpec, ParquetRowKey, RerankStore,
    SearchResult,
};
use lance_linalg::distance::MetricType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::error::PythonErrorExt;
use crate::rt;

fn parse_metric(s: &str) -> PyResult<MetricType> {
    match s.to_ascii_lowercase().as_str() {
        "l2" => Ok(MetricType::L2),
        "cosine" => Ok(MetricType::Cosine),
        "dot" => Ok(MetricType::Dot),
        other => Err(PyValueError::new_err(format!(
            "unsupported metric '{other}'; expected one of l2, cosine, dot"
        ))),
    }
}

fn parse_rerank_store(s: &str) -> PyResult<RerankStore> {
    match s.to_ascii_lowercase().as_str() {
        "none" | "" => Ok(RerankStore::None),
        "sq8" => Ok(RerankStore::Sq8),
        "flat" => Ok(RerankStore::Flat),
        other => Err(PyValueError::new_err(format!(
            "unsupported rerank store '{other}'; expected one of none, sq8, flat"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_params(
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits_per_sub_vector: usize,
    metric: &str,
    max_iters: usize,
    sample_rate: usize,
    seed: u64,
    rerank_store: &str,
) -> PyResult<ExternalIvfPqIndexParams> {
    Ok(ExternalIvfPqIndexParams::builder()
        .num_partitions(num_partitions)
        .num_sub_vectors(num_sub_vectors)
        .num_bits_per_sub_vector(num_bits_per_sub_vector)
        .metric(parse_metric(metric)?)
        .max_iters(max_iters)
        .sample_rate(sample_rate)
        .seed(seed)
        .rerank_store(parse_rerank_store(rerank_store)?)
        .build())
}

/// One search hit: `(file_path, row_index, distance)`.
fn results_to_py(results: Vec<SearchResult>) -> Vec<(String, u64, f32)> {
    results
        .into_iter()
        .map(|r| (r.file_path, r.row_index, r.distance))
        .collect()
}

/// An opened external IVF-PQ index over parquet files. Usable from any Python process
/// (no JVM) — the path for Ray-based build and serving.
#[pyclass(name = "ExternalIvfPqIndex")]
pub struct PyExternalIvfPqIndex {
    inner: Arc<ExternalIvfPqIndex>,
}

#[pymethods]
impl PyExternalIvfPqIndex {
    /// Build an index over `file_paths` (order significant — fixes each file's global id),
    /// writing it under `output_uri/<uuid>`. Returns the opened index.
    ///
    /// Single-node build (whole-corpus scan on this process). For a cluster build, fan
    /// `build_shard` across workers and `merge_shards` on the driver (see module docs).
    #[staticmethod]
    #[pyo3(signature = (
        file_paths, vector_column, output_uri,
        num_partitions = 256, num_sub_vectors = 16, num_bits_per_sub_vector = 8,
        metric = "l2", max_iters = 50, sample_rate = 256, seed = 0, rerank_store = "none"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        file_paths: Vec<String>,
        vector_column: String,
        output_uri: String,
        num_partitions: usize,
        num_sub_vectors: usize,
        num_bits_per_sub_vector: usize,
        metric: &str,
        max_iters: usize,
        sample_rate: usize,
        seed: u64,
        rerank_store: &str,
    ) -> PyResult<Self> {
        let params = build_params(
            num_partitions,
            num_sub_vectors,
            num_bits_per_sub_vector,
            metric,
            max_iters,
            sample_rate,
            seed,
            rerank_store,
        )?;
        let files: Vec<ParquetFileSpec> =
            file_paths.into_iter().map(ParquetFileSpec::of).collect();
        let output_uri_ret = output_uri.clone();
        let uuid = rt()
            .block_on(None, async move {
                ExternalIvfPqIndex::build(files, &vector_column, &output_uri, params).await
            })?
            .infer_error()?;
        let uri = format!("{output_uri_ret}/{uuid}");
        Self::open(uri)
    }

    /// Open an existing index by URI (the `output_uri/<uuid>` directory).
    #[staticmethod]
    fn open(uri: String) -> PyResult<Self> {
        let inner = rt()
            .block_on(None, async move { ExternalIvfPqIndex::open(&uri).await })?
            .infer_error()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of registered parquet files.
    #[getter]
    fn num_files(&self) -> usize {
        self.inner.num_files()
    }

    /// Number of IVF partitions.
    #[getter]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    /// Vector column the index was built over.
    #[getter]
    fn vector_column(&self) -> String {
        self.inner.vector_column().to_string()
    }

    /// Single-query approximate search. Returns up to `k` `(file_path, row_index, distance)`
    /// tuples, best first.
    #[pyo3(signature = (query, k = 10, nprobes = 16, refine_factor = 8))]
    fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        nprobes: usize,
        refine_factor: usize,
    ) -> PyResult<Vec<(String, u64, f32)>> {
        let inner = self.inner.clone();
        let results = rt()
            .block_on(None, async move {
                inner.search(&query, k, nprobes, refine_factor, None).await
            })?
            .infer_error()?;
        Ok(results_to_py(results))
    }

    /// Batched approximate search — one shared per-file refinement read across all queries.
    /// Returns a list (per query, input order) of `(file_path, row_index, distance)` lists.
    #[pyo3(signature = (queries, k = 10, nprobes = 16, refine_factor = 8))]
    fn search_batch(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
        nprobes: usize,
        refine_factor: usize,
    ) -> PyResult<Vec<Vec<(String, u64, f32)>>> {
        let inner = self.inner.clone();
        let batch = rt()
            .block_on(None, async move {
                let refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
                inner.search_batch(&refs, k, nprobes, refine_factor, None).await
            })?
            .infer_error()?;
        Ok(batch.into_iter().map(results_to_py).collect())
    }

    /// Exact brute-force search (no index): scans every source-parquet vector. The no-index
    /// baseline; O(|R|) per call. Returns per-query `(file_path, row_index, distance)` lists.
    #[pyo3(signature = (queries, k = 10))]
    fn search_flat(
        &self,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<(String, u64, f32)>>> {
        let inner = self.inner.clone();
        let batch = rt()
            .block_on(None, async move {
                let refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
                inner.search_flat(&refs, k).await
            })?
            .infer_error()?;
        Ok(batch.into_iter().map(results_to_py).collect())
    }

    /// Materialize `projection` columns for the given `(file_path, row_index)` keys via
    /// page-index-aware parquet random access. Returns a pyarrow RecordBatch.
    fn fetch_rows<'py>(
        &self,
        py: Python<'py>,
        keys: Vec<(String, u64)>,
        projection: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let batch = rt()
            .block_on(None, async move {
                let row_keys: Vec<ParquetRowKey> = keys
                    .into_iter()
                    .map(|(file_path, row_index)| ParquetRowKey {
                        file_path,
                        row_index,
                    })
                    .collect();
                let proj: Vec<&str> = projection.iter().map(|s| s.as_str()).collect();
                inner.fetch_rows(&row_keys, &proj).await
            })?
            .infer_error()?;
        batch.to_pyarrow(py)
    }
}

// ---- Distributed build (Ray / any orchestrator) --------------------------------------------
//
// Three module functions mirroring the JNI distributed surface. The orchestrator (e.g. a Ray
// driver) does: train_broadcast on the head → ship the returned bytes to workers → each worker
// build_shard over its file range → driver merge_shards. Only the ~KB payload + sidecar row
// counts cross between nodes; each worker reads/encodes its own file shard, so peak memory is
// per-shard, not whole-corpus (this is the fix for large-|R| builds that OOM single-node).

/// Driver phase 1: train IVF centroids + PQ codebook on a sample of the whole corpus, returning
/// a broadcast-ready byte payload (quantizers + SQ8 bounds). Cheap — reads a sample, not |R|.
#[pyfunction]
#[pyo3(signature = (
    file_paths, vector_column,
    num_partitions = 256, num_sub_vectors = 16, num_bits_per_sub_vector = 8,
    metric = "l2", max_iters = 50, sample_rate = 256, seed = 0, rerank_store = "none"
))]
#[allow(clippy::too_many_arguments)]
pub fn external_train_broadcast<'py>(
    py: Python<'py>,
    file_paths: Vec<String>,
    vector_column: String,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits_per_sub_vector: usize,
    metric: &str,
    max_iters: usize,
    sample_rate: usize,
    seed: u64,
    rerank_store: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let params = build_params(
        num_partitions, num_sub_vectors, num_bits_per_sub_vector,
        metric, max_iters, sample_rate, seed, rerank_store,
    )?;
    let files: Vec<ParquetFileSpec> = file_paths.into_iter().map(ParquetFileSpec::of).collect();
    let bytes = rt()
        .block_on(None, async move {
            train_broadcast_payload(files, &vector_column, &params).await
        })?
        .infer_error()?
        .to_bytes();
    Ok(PyBytes::new(py, &bytes))
}

/// Executor phase 2: encode this worker's file shard into an index shard parquet at
/// `shard_uri` (+ a rerank sidecar shard when a store is set), using the broadcast payload.
/// `file_id_offset` = the global index of this shard's first file so rids stay consistent.
/// Returns the sidecar shard's row count (0 when no rerank store).
#[pyfunction]
#[pyo3(signature = (
    payload, file_paths, vector_column, file_id_offset, shard_uri,
    num_partitions = 256, num_sub_vectors = 16, num_bits_per_sub_vector = 8,
    metric = "l2", max_iters = 50, sample_rate = 256, seed = 0, rerank_store = "none"
))]
#[allow(clippy::too_many_arguments)]
pub fn external_build_shard(
    payload: Vec<u8>,
    file_paths: Vec<String>,
    vector_column: String,
    file_id_offset: u32,
    shard_uri: String,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits_per_sub_vector: usize,
    metric: &str,
    max_iters: usize,
    sample_rate: usize,
    seed: u64,
    rerank_store: &str,
) -> PyResult<u64> {
    let params = build_params(
        num_partitions, num_sub_vectors, num_bits_per_sub_vector,
        metric, max_iters, sample_rate, seed, rerank_store,
    )?;
    let files: Vec<ParquetFileSpec> = file_paths.into_iter().map(ParquetFileSpec::of).collect();
    let res = rt()
        .block_on(None, async move {
            let bp = BroadcastPayload::from_bytes(&payload)?;
            build_shard_to_parquet(&bp, files, &vector_column, file_id_offset, &params, &shard_uri)
                .await
        })?
        .infer_error()?;
    Ok(res.sidecar.map(|s| s.rows).unwrap_or(0))
}

/// Driver phase 3: heap-merge the shard parquets into `<index_dir_uri>/<uuid>/index.idx` +
/// manifest, leaving an openable index directory. `file_paths` is the full sorted file list
/// (position = global file_id). `sidecar_shards` are `(uri, rows)` in global file order (empty
/// when no rerank store). Only PQ codes cross back — cheap relative to the encode. Returns the
/// resulting index directory URI.
#[pyfunction]
#[pyo3(signature = (
    payload, shard_uris, file_paths, vector_column, index_dir_uri, sidecar_shards,
    num_partitions = 256, num_sub_vectors = 16, num_bits_per_sub_vector = 8,
    metric = "l2", max_iters = 50, sample_rate = 256, seed = 0, rerank_store = "none"
))]
#[allow(clippy::too_many_arguments)]
pub fn external_merge_shards(
    payload: Vec<u8>,
    shard_uris: Vec<String>,
    file_paths: Vec<String>,
    vector_column: String,
    index_dir_uri: String,
    sidecar_shards: Vec<(String, u64)>,
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits_per_sub_vector: usize,
    metric: &str,
    max_iters: usize,
    sample_rate: usize,
    seed: u64,
    rerank_store: &str,
) -> PyResult<()> {
    let params = build_params(
        num_partitions, num_sub_vectors, num_bits_per_sub_vector,
        metric, max_iters, sample_rate, seed, rerank_store,
    )?;
    rt()
        .block_on(None, async move {
            let bp = BroadcastPayload::from_bytes(&payload)?;
            merge_shards_to_index(
                &bp, &shard_uris, &file_paths, &vector_column, &index_dir_uri, &params,
                &sidecar_shards,
            )
            .await
        })?
        .infer_error()
}
