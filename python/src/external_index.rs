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
    ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetFileSpec, ParquetRowKey, RerankStore,
    SearchResult,
};
use lance_linalg::distance::MetricType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
