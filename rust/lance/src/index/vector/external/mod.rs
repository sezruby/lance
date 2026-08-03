// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! External vector index — IVF-PQ over caller-supplied parquet files.
//!
//! Lance builds and queries a vector index over parquet data without copying it into
//! a Lance dataset. The caller registers a list of parquet files; Lance encodes each
//! row's identity as `(file_id_u32 << 32) | row_index_u32` internally, and surfaces
//! search results as `(file_path, row_index, distance)` so callers never have to
//! decode the rid themselves.
//!
//! Read [`ExternalIvfPqIndex`] for the entry point and [`crate::index::vector`] for
//! the equivalent dataset-backed index.
//!
//! # Why
//!
//! For engines (Spark, Trino, Daft, ...) that already store data in parquet/Delta/
//! Iceberg, building a Lance dataset just to get a Lance vector index doubles the
//! storage. The external index lets the source format stay the source of truth and
//! confines Lance to the index file alone.
//!
//! # Status
//!
//! Phase 1 of the "External Vector Index" RFC. Surface area:
//!
//! - [`ExternalIvfPqIndex`] — handle: build / open / search / fetch_rows
//! - [`ParquetFileSpec`] — describes one parquet file in the registry
//! - [`SearchResult`] — `{file_path, row_index, distance}` returned by `search`
//! - [`ParquetRowKey`] — `(file_path, row_index)` accepted by `fetch_rows`
//! - [`RowFilter`] — extensibility hook for Delta deletion vectors / Iceberg
//!   position deletes / ad-hoc skip predicates
//! - [`ExternalIvfPqIndexParams`] — kmeans / PQ / metric configuration

mod build;
pub(crate) mod distributed;
pub(crate) mod fetch;
pub(crate) mod manifest;
pub(crate) mod open;
pub mod params;
pub(crate) mod parquet_source;
mod rerank;
mod search;
pub mod types;

pub use params::{ExternalIvfPqIndexParams, RerankStore};
pub use types::{ParquetFileSpec, ParquetRowKey, RowFilter, SearchResult};
// Distributed-build entry points (driver train + broadcast, executor shard build,
// driver merge). Exposed for the JNI/Spark orchestration layer.
pub use distributed::{
    BroadcastPayload, ShardResult, SidecarShard, assemble_broadcast_payload_from_centroids,
    assemble_payload_resident, build_shard_to_parquet, compute_partial_stats_in_memory,
    compute_partial_stats_resident, free_resident_samples, load_driver_sample,
    merge_shards_to_index, sample_shard, sample_shard_to_parquet,
    select_initial_centroids_resident, train_broadcast_payload,
    train_broadcast_payload_from_sample,
};

use arrow_array::RecordBatch;
use lance_core::Result;

/// IVF-PQ index over caller-registered parquet files.
///
/// Lance owns the parquet reader (page-index-aware random access via
/// `PageIndexPolicy::Required`) and the refinement step. Callers see
/// `(file_path, row_index, distance)` results.
///
/// # Example
///
/// Sketch — full impl lands in subsequent phases. Today this only constructs.
///
/// ```ignore
/// # use lance::index::vector::external::*;
/// # async fn run(files: Vec<ParquetFileSpec>) -> lance_core::Result<()> {
/// let params = ExternalIvfPqIndexParams::builder()
///     .num_partitions(256)
///     .num_sub_vectors(16)
///     .build();
/// ExternalIvfPqIndex::build(files, "vec", "/tmp/idx", params).await?;
///
/// let idx = ExternalIvfPqIndex::open("/tmp/idx").await?;
/// let hits = idx.search(&[0.1; 128], 10, 16, 8, None).await?;
/// for hit in &hits {
///     println!("{} @ row {} = {}", hit.file_path, hit.row_index, hit.distance);
/// }
/// # Ok(()) }
/// ```
pub struct ExternalIvfPqIndex {
    /// All deserialized index state (manifest + IVF model + PQ codebooks +
    /// object_store + index_dir).
    inner: open::OpenedExternalIndex,
}

impl ExternalIvfPqIndex {
    /// Build an external IVF-PQ index over the given parquet files.
    ///
    /// Reads sample vectors for kmeans + PQ training, encodes residuals, writes a
    /// single index file at `output_uri`. Synchronous: returns when the file is on
    /// disk and durable.
    ///
    /// `vector_column` must be a non-null `FixedSizeList<Float>` column in every
    /// file's schema; this is validated against each parquet footer.
    ///
    /// `file_id` is implicit in `files`'s position. Reordering invalidates the
    /// index.
    pub async fn build(
        files: Vec<ParquetFileSpec>,
        vector_column: &str,
        output_uri: &str,
        params: ExternalIvfPqIndexParams,
    ) -> Result<uuid::Uuid> {
        build::build_index(files, vector_column, output_uri, params).await
    }

    /// Open an external IVF-PQ index by URI.
    ///
    /// Cheap: reads the manifest + index header. Per-file parquet readers are
    /// constructed lazily on first `search` / `fetch_rows`.
    pub async fn open(uri: &str) -> Result<Self> {
        let inner = open::open_index(uri).await?;
        Ok(Self { inner })
    }

    /// Number of registered parquet files.
    pub fn num_files(&self) -> usize {
        self.inner.manifest.files.len()
    }

    /// Look up the file path for a given `file_id` (the high 32 bits of the rid).
    /// `None` if the id is out of range.
    pub fn file_path(&self, file_id: u32) -> Option<&str> {
        self.inner.manifest.file_path(file_id)
    }

    /// Vector column the index was built over.
    pub fn vector_column(&self) -> &str {
        &self.inner.manifest.vector_column
    }

    /// Number of IVF partitions in the index.
    pub fn num_partitions(&self) -> usize {
        self.inner.ivf.num_partitions()
    }

    /// Run an approximate nearest-neighbor query over the index.
    ///
    /// Probes `nprobes` IVF partitions, fetches `k * refine_factor` PQ-approx
    /// candidates, refines them by reading their actual vectors from the source
    /// parquet via page-index-aware random access, and returns the top-`k` after
    /// exact distance recompute.
    ///
    /// `filter` (if `Some`) is consulted during refinement; rows it rejects are
    /// dropped before re-ranking. See [`RowFilter`] for typical use cases like
    /// Delta deletion vectors.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobes: usize,
        refine_factor: usize,
        filter: Option<&dyn RowFilter>,
    ) -> Result<Vec<SearchResult>> {
        search::search(&self.inner, query, k, nprobes, refine_factor, filter).await
    }

    /// Batched variant of [`search`](Self::search). Runs `queries.len()` queries
    /// in one call, sharing a single per-file refinement read across the whole
    /// batch. Designed for offline join workloads where many query vectors are
    /// available up-front and would otherwise repeat the same parquet fetches.
    ///
    /// Returns `Vec<Vec<SearchResult>>` in 1:1 correspondence with `queries`.
    pub async fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        nprobes: usize,
        refine_factor: usize,
        filter: Option<&dyn RowFilter>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        search::search_batch(&self.inner, queries, k, nprobes, refine_factor, filter).await
    }

    /// Exact brute-force search (no index) — scans every source-parquet vector and returns the
    /// exact top-`k` per query. The no-index baseline for latency/recall comparison; O(|R|).
    pub async fn search_flat(
        &self,
        queries: &[&[f32]],
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        search::search_flat(&self.inner, queries, k).await
    }

    /// Random-access fetch by `(file_path, row_index)` keys.
    ///
    /// Lance batches by file internally and issues one page-index-aware parquet
    /// read per file, then reassembles the result in caller-input order. The
    /// result has one row per input key.
    ///
    /// `projection` may include columns that are not part of the index — they're
    /// read from the parquet schema directly. The killer feature: post-topK
    /// materialization fetches only the projection columns for the surviving rows.
    pub async fn fetch_rows(
        &self,
        row_keys: &[ParquetRowKey],
        projection: &[&str],
    ) -> Result<RecordBatch> {
        fetch::fetch_rows(&self.inner, row_keys, projection).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::MetricType;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    /// Smoke test: open() on a missing path errors out cleanly. Real round-trip
    /// is exercised by `build_then_open_round_trip` below.
    #[tokio::test]
    async fn open_missing_errors() {
        let result = ExternalIvfPqIndex::open("/tmp/this-path-does-not-exist-9f3a8").await;
        assert!(result.is_err(), "open() on missing path should error");
    }

    fn write_random_parquet(path: &PathBuf, num_rows: usize, dim: usize, seed: u64) {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = (0..num_rows * dim)
            .map(|_| rng.random_range(-1.0f32..1.0))
            .collect();
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

    fn brute_force_topk(
        vectors: &FixedSizeListArray,
        query: &[f32],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let values = vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let dim = vectors.value_length() as usize;
        let n = vectors.len();
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let mut s = 0.0f32;
                for (d, q) in query.iter().enumerate() {
                    let v = values.value(i * dim + d);
                    let diff = v - q;
                    s += diff * diff;
                }
                (i, s)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).collect()
    }

    /// End-to-end: build → open → search → assert recall against brute force
    /// ground truth. Toy scale; meant to validate the pipeline runs and rids round-trip.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_open_search_recall_above_threshold() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const NUM_VECTORS: usize = 1024;
        const DIM: usize = 8;
        const K: usize = 10;
        const NUM_PARTITIONS: usize = 4;
        const NUM_SUB_VECTORS: usize = 2;

        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("vec.parquet");
        write_random_parquet(&p, NUM_VECTORS, DIM, 42);

        // Read back the vectors so we can compute ground truth in-memory.
        let all_vectors = {
            let file = std::fs::File::open(&p).unwrap();
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap();
            let reader = builder.build().unwrap();
            let mut batches: Vec<RecordBatch> = Vec::new();
            for r in reader {
                batches.push(r.unwrap());
            }
            let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(0).as_ref()).collect();
            let cat = arrow::compute::concat(&arrays).unwrap();
            cat.as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .clone()
        };

        let tmp_out = TempDir::new().unwrap();
        let output_uri = tmp_out.path().to_str().unwrap();
        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(NUM_PARTITIONS)
            .num_sub_vectors(NUM_SUB_VECTORS)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(10)
            .sample_rate(64)
            .build();
        let files = vec![ParquetFileSpec::of(p.to_str().unwrap())];
        let uuid = ExternalIvfPqIndex::build(files, "vec", output_uri, params)
            .await
            .expect("build_index");
        let idx = ExternalIvfPqIndex::open(tmp_out.path().join(uuid.to_string()).to_str().unwrap())
            .await
            .expect("open");

        // Run a few queries; check that the top-1 from search matches the
        // brute-force top-1 for at least most queries (recall@1 ≥ 0.5 at this
        // toy scale is the bar — PQ at dim=8 / 2 sub-vectors loses fidelity).
        let mut rng = StdRng::seed_from_u64(99);
        let mut hits = 0;
        let total_queries = 16;
        for _ in 0..total_queries {
            let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let truth = brute_force_topk(&all_vectors, &query, K);
            let truth_set: std::collections::HashSet<u64> =
                truth.iter().map(|(i, _)| *i as u64).collect();

            let results = idx
                .search(
                    &query, K, /* nprobes = */ 4, /* refine_factor = */ 4, None,
                )
                .await
                .expect("search");
            assert!(!results.is_empty(), "search returned empty");
            let result_set: std::collections::HashSet<u64> =
                results.iter().map(|r| r.row_index).collect();
            let intersection = result_set.intersection(&truth_set).count();
            if intersection >= K / 2 {
                hits += 1;
            }
        }
        assert!(
            hits >= total_queries / 2,
            "recall too low: {hits}/{total_queries} queries had ≥ K/2 correct"
        );
    }

    /// search_batch must produce identical (file_path, row_index, distance)
    /// triples to running `search()` per query — this is the correctness
    /// invariant that lets us collapse N per-query refinement reads into one.
    #[tokio::test(flavor = "multi_thread")]
    async fn search_batch_matches_per_query_search() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const NUM_VECTORS: usize = 1024;
        const DIM: usize = 8;
        const K: usize = 10;
        const NUM_PARTITIONS: usize = 4;
        const NUM_SUB_VECTORS: usize = 2;

        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("vec.parquet");
        write_random_parquet(&p, NUM_VECTORS, DIM, 7);

        let tmp_out = TempDir::new().unwrap();
        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(NUM_PARTITIONS)
            .num_sub_vectors(NUM_SUB_VECTORS)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(10)
            .sample_rate(64)
            .build();
        let files = vec![ParquetFileSpec::of(p.to_str().unwrap())];
        let uuid =
            ExternalIvfPqIndex::build(files, "vec", tmp_out.path().to_str().unwrap(), params)
                .await
                .expect("build_index");
        let idx = ExternalIvfPqIndex::open(tmp_out.path().join(uuid.to_string()).to_str().unwrap())
            .await
            .expect("open");

        let mut rng = StdRng::seed_from_u64(123);
        let queries: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect())
            .collect();

        // Per-query reference.
        let mut per_query_results: Vec<Vec<SearchResult>> = Vec::with_capacity(queries.len());
        for q in &queries {
            let r = idx
                .search(q, K, /* nprobes */ 4, /* refine_factor */ 4, None)
                .await
                .unwrap();
            per_query_results.push(r);
        }

        // Batched.
        let q_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let batched = idx.search_batch(&q_refs, K, 4, 4, None).await.unwrap();

        assert_eq!(batched.len(), per_query_results.len());
        for (i, (a, b)) in batched.iter().zip(per_query_results.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "result count differs at query {i}");
            for (j, (ar, br)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(ar.file_path, br.file_path, "file_path differs at q{i}/{j}");
                assert_eq!(ar.row_index, br.row_index, "row_index differs at q{i}/{j}");
                let diff = (ar.distance - br.distance).abs();
                assert!(diff < 1e-4, "distance differs at q{i}/{j}: {diff}");
            }
        }
    }

    /// fetch_rows() returns parquet projection cols for arbitrary (file_path,
    /// row_index) keys, in caller-input order, including duplicates and
    /// non-vector columns.
    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_rows_returns_projection_in_input_order() {
        use arrow_array::{Int64Array, StringArray, UInt64Array};

        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("payload.parquet");

        // Build a parquet file with a vector column + an integer payload column
        // + a string payload column. fetch_rows should be able to project any
        // subset. PQ training needs ≥ 256 rows so size accordingly.
        let n = 320usize;
        let dim = 4usize;
        let values: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let flat = Float32Array::from(values);
        let fsl = FixedSizeListArray::try_new_from_values(flat, dim as i32).unwrap();
        let ids = Int64Array::from((0..n as i64).map(|i| i * 10).collect::<Vec<_>>());
        let names = StringArray::from((0..n).map(|i| format!("name-{i}")).collect::<Vec<_>>());

        let schema = Arc::new(Schema::new(vec![
            Field::new("vec", fsl.data_type().clone(), false),
            Field::new("id", arrow_schema::DataType::Int64, false),
            Field::new("name", arrow_schema::DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(fsl) as ArrayRef,
                Arc::new(ids) as ArrayRef,
                Arc::new(names) as ArrayRef,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(&p).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(
            file,
            schema,
            Some(parquet::file::properties::WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Build + open
        let tmp_out = TempDir::new().unwrap();
        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(2)
            .num_sub_vectors(2)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(5)
            // sample_rate * num_partitions must yield ≥ 256 for PQ training.
            .sample_rate(150)
            .build();
        let uuid = ExternalIvfPqIndex::build(
            vec![ParquetFileSpec::of(p.to_str().unwrap())],
            "vec",
            tmp_out.path().to_str().unwrap(),
            params,
        )
        .await
        .unwrap();
        let idx = ExternalIvfPqIndex::open(tmp_out.path().join(uuid.to_string()).to_str().unwrap())
            .await
            .unwrap();

        // Fetch rows in non-sorted, with-duplicate order.
        let path = p.to_str().unwrap().to_string();
        let keys = vec![
            ParquetRowKey::of(&path, 5),
            ParquetRowKey::of(&path, 0),
            ParquetRowKey::of(&path, 5), // duplicate
            ParquetRowKey::of(&path, 30),
        ];
        let result = idx
            .fetch_rows(&keys, &["id", "name"])
            .await
            .expect("fetch_rows");
        assert_eq!(result.num_rows(), 4);
        assert_eq!(result.num_columns(), 2);

        let id_col = result
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // expected ids: [50, 0, 50, 300]
        assert_eq!(
            (0..id_col.len())
                .map(|i| id_col.value(i))
                .collect::<Vec<_>>(),
            vec![50, 0, 50, 300]
        );

        let name_col = result
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            (0..name_col.len())
                .map(|i| name_col.value(i).to_string())
                .collect::<Vec<_>>(),
            vec!["name-5", "name-0", "name-5", "name-30"]
        );

        // Empty input: returns empty batch with projected schema.
        let empty = idx.fetch_rows(&[], &["id"]).await.unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_eq!(empty.schema().fields().len(), 1);
        assert_eq!(empty.schema().field(0).name(), "id");

        // Unknown file: validation error before any I/O.
        let bad = idx
            .fetch_rows(&[ParquetRowKey::of("/nope/no.parquet", 0)], &["id"])
            .await;
        assert!(bad.is_err());

        // Silence the unused variable warnings.
        let _ = UInt64Array::from(vec![0u64]);
    }

    /// build() writes a non-empty index file + manifest. open() round-trips them.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_then_open_round_trip() {
        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("vec.parquet");
        write_random_parquet(&p, 256, 4, 42);

        let tmp_out = TempDir::new().unwrap();
        let output_uri = tmp_out.path().to_str().unwrap();

        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(4)
            .num_sub_vectors(2)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(5)
            .sample_rate(64)
            .build();

        let files = vec![ParquetFileSpec::of(p.to_str().unwrap())];
        let uuid = ExternalIvfPqIndex::build(files, "vec", output_uri, params)
            .await
            .expect("build_index");

        let idx_dir = tmp_out.path().join(uuid.to_string());
        let idx_file = idx_dir.join("index.idx");
        let manifest_file = idx_dir.join("manifest.json");
        for f in [&idx_file, &manifest_file] {
            let meta = std::fs::metadata(f)
                .unwrap_or_else(|e| panic!("expected file at {}: {e}", f.display()));
            assert!(meta.len() > 0, "{} is empty", f.display());
        }

        let opened_uri = idx_dir.to_str().unwrap();
        let idx = ExternalIvfPqIndex::open(opened_uri)
            .await
            .expect("open() must succeed after build()");
        assert_eq!(idx.num_files(), 1);
        assert_eq!(idx.num_partitions(), 4);
        assert_eq!(idx.vector_column(), "vec");
        assert_eq!(idx.file_path(0), Some(p.to_str().unwrap()));
        assert_eq!(idx.file_path(1), None);
    }

    /// With an SQ8 rerank store, build() writes the `rerank.sq8` sidecar, records
    /// its meta in the manifest, and search() reranks against the int8 codes
    /// (no source-parquet refine read) — recall should stay at least as good as
    /// the parquet-refine path, since SQ8 is a near-exact stand-in for the f32
    /// originals that path re-reads.
    #[tokio::test(flavor = "multi_thread")]
    async fn sq8_store_builds_and_reranks_with_recall() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const NUM_VECTORS: usize = 1024;
        const DIM: usize = 16;
        const K: usize = 10;
        const NUM_PARTITIONS: usize = 4;
        const NUM_SUB_VECTORS: usize = 4;

        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("vec.parquet");
        write_random_parquet(&p, NUM_VECTORS, DIM, 2024);

        let all_vectors = {
            let file = std::fs::File::open(&p).unwrap();
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap();
            let reader = builder.build().unwrap();
            let batches: Vec<RecordBatch> = reader.map(|r| r.unwrap()).collect();
            let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(0).as_ref()).collect();
            arrow::compute::concat(&arrays)
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .clone()
        };

        let make_params = |store: params::RerankStore| {
            ExternalIvfPqIndexParams::builder()
                .num_partitions(NUM_PARTITIONS)
                .num_sub_vectors(NUM_SUB_VECTORS)
                .num_bits_per_sub_vector(8)
                .metric(MetricType::L2)
                .max_iters(10)
                .sample_rate(64)
                .rerank_store(store)
                .build()
        };
        let spec = || vec![ParquetFileSpec::of(p.to_str().unwrap())];

        // SQ8 index.
        let tmp_sq8 = TempDir::new().unwrap();
        let uuid_sq8 = ExternalIvfPqIndex::build(
            spec(),
            "vec",
            tmp_sq8.path().to_str().unwrap(),
            make_params(params::RerankStore::Sq8),
        )
        .await
        .expect("build sq8");

        // The sidecar exists and is the right size: total_rows * dim bytes.
        let sq8_dir = tmp_sq8.path().join(uuid_sq8.to_string());
        let sidecar = sq8_dir.join("rerank.sq8");
        let meta = std::fs::metadata(&sidecar).expect("rerank.sq8 must exist");
        assert_eq!(meta.len(), (NUM_VECTORS * DIM) as u64);

        let idx_sq8 = ExternalIvfPqIndex::open(sq8_dir.to_str().unwrap())
            .await
            .expect("open sq8");

        // Parquet-refine index over the same data, as a recall reference.
        let tmp_pq = TempDir::new().unwrap();
        let uuid_pq = ExternalIvfPqIndex::build(
            spec(),
            "vec",
            tmp_pq.path().to_str().unwrap(),
            make_params(params::RerankStore::None),
        )
        .await
        .expect("build none");
        let idx_pq =
            ExternalIvfPqIndex::open(tmp_pq.path().join(uuid_pq.to_string()).to_str().unwrap())
                .await
                .expect("open none");

        let mut rng = StdRng::seed_from_u64(7);
        let total_queries = 24;
        let mut sq8_hits = 0;
        let mut pq_hits = 0;
        for _ in 0..total_queries {
            let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let truth: std::collections::HashSet<u64> = brute_force_topk(&all_vectors, &query, K)
                .iter()
                .map(|(i, _)| *i as u64)
                .collect();

            let r_sq8 = idx_sq8
                .search(&query, K, 4, 4, None)
                .await
                .expect("sq8 search");
            let r_pq = idx_pq
                .search(&query, K, 4, 4, None)
                .await
                .expect("pq search");
            assert!(!r_sq8.is_empty());

            let s_sq8: std::collections::HashSet<u64> = r_sq8.iter().map(|r| r.row_index).collect();
            let s_pq: std::collections::HashSet<u64> = r_pq.iter().map(|r| r.row_index).collect();
            if s_sq8.intersection(&truth).count() >= K / 2 {
                sq8_hits += 1;
            }
            if s_pq.intersection(&truth).count() >= K / 2 {
                pq_hits += 1;
            }
        }
        // SQ8 recall must clear the bar and not trail the parquet-refine path by
        // more than one query (SQ8 ≈ exact f32 rerank for ordering).
        assert!(
            sq8_hits >= total_queries / 2,
            "sq8 recall too low: {sq8_hits}/{total_queries}"
        );
        assert!(
            sq8_hits + 1 >= pq_hits,
            "sq8 recall ({sq8_hits}) trails parquet refine ({pq_hits}) by >1"
        );
    }

    /// The Flat (full-precision f32) rerank store: build writes a `rerank.flat`
    /// sidecar of exactly `total_rows * dim * 4` bytes, and its exact-distance
    /// refine gives recall at least as good as SQ8 over the same data (exact ≥
    /// quantized). Same layout/plumbing as SQ8 — this guards the f32 branch.
    #[tokio::test(flavor = "multi_thread")]
    async fn flat_store_builds_and_reranks_at_least_as_well_as_sq8() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const NUM_VECTORS: usize = 1024;
        const DIM: usize = 16;
        const K: usize = 10;

        let tmp_data = TempDir::new().unwrap();
        let p = tmp_data.path().join("vec.parquet");
        write_random_parquet(&p, NUM_VECTORS, DIM, 2024);

        let all_vectors = {
            let file = std::fs::File::open(&p).unwrap();
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap();
            let batches: Vec<RecordBatch> = builder.build().unwrap().map(|r| r.unwrap()).collect();
            let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(0).as_ref()).collect();
            arrow::compute::concat(&arrays)
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .clone()
        };

        let make_params = |store: params::RerankStore| {
            ExternalIvfPqIndexParams::builder()
                .num_partitions(4)
                .num_sub_vectors(4)
                .num_bits_per_sub_vector(8)
                .metric(MetricType::L2)
                .max_iters(10)
                .sample_rate(64)
                .rerank_store(store)
                .build()
        };
        let spec = || vec![ParquetFileSpec::of(p.to_str().unwrap())];

        let tmp_flat = TempDir::new().unwrap();
        let uuid_flat = ExternalIvfPqIndex::build(
            spec(),
            "vec",
            tmp_flat.path().to_str().unwrap(),
            make_params(params::RerankStore::Flat),
        )
        .await
        .expect("build flat");

        // Sidecar is full-precision: total_rows * dim * 4 bytes.
        let flat_dir = tmp_flat.path().join(uuid_flat.to_string());
        let meta = std::fs::metadata(flat_dir.join("rerank.flat")).expect("rerank.flat must exist");
        assert_eq!(meta.len(), (NUM_VECTORS * DIM * 4) as u64);

        let idx_flat = ExternalIvfPqIndex::open(flat_dir.to_str().unwrap())
            .await
            .expect("open flat");

        let tmp_sq8 = TempDir::new().unwrap();
        let uuid_sq8 = ExternalIvfPqIndex::build(
            spec(),
            "vec",
            tmp_sq8.path().to_str().unwrap(),
            make_params(params::RerankStore::Sq8),
        )
        .await
        .expect("build sq8");
        let idx_sq8 =
            ExternalIvfPqIndex::open(tmp_sq8.path().join(uuid_sq8.to_string()).to_str().unwrap())
                .await
                .expect("open sq8");

        let mut rng = StdRng::seed_from_u64(7);
        let total_queries = 24;
        let mut flat_hits = 0;
        let mut sq8_hits = 0;
        for _ in 0..total_queries {
            let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let truth: std::collections::HashSet<u64> = brute_force_topk(&all_vectors, &query, K)
                .iter()
                .map(|(i, _)| *i as u64)
                .collect();
            let r_flat = idx_flat
                .search(&query, K, 4, 4, None)
                .await
                .expect("flat search");
            let r_sq8 = idx_sq8
                .search(&query, K, 4, 4, None)
                .await
                .expect("sq8 search");
            assert!(!r_flat.is_empty());
            let s_flat: std::collections::HashSet<u64> =
                r_flat.iter().map(|r| r.row_index).collect();
            let s_sq8: std::collections::HashSet<u64> = r_sq8.iter().map(|r| r.row_index).collect();
            if s_flat.intersection(&truth).count() >= K / 2 {
                flat_hits += 1;
            }
            if s_sq8.intersection(&truth).count() >= K / 2 {
                sq8_hits += 1;
            }
        }
        assert!(
            flat_hits >= total_queries / 2,
            "flat recall too low: {flat_hits}/{total_queries}"
        );
        // Exact refine should not trail quantized refine (allow 1 query of slack for
        // ties at this toy scale).
        assert!(
            flat_hits + 1 >= sq8_hits,
            "flat recall ({flat_hits}) trails sq8 ({sq8_hits}) by >1 — exact should match or beat"
        );
    }

    /// Multi-file SQ8: the sidecar is row-major by *global* ordinal
    /// (`global_base(file_id) + row_in_file`), so rerank must map candidate rids
    /// from different files to the right byte range. Two files force a non-zero
    /// base on the second file.
    #[tokio::test(flavor = "multi_thread")]
    async fn sq8_store_multi_file_global_ordinal() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const N0: usize = 512;
        const N1: usize = 640;
        const DIM: usize = 16;
        const K: usize = 10;

        let tmp_data = TempDir::new().unwrap();
        let p0 = tmp_data.path().join("a.parquet");
        let p1 = tmp_data.path().join("b.parquet");
        write_random_parquet(&p0, N0, DIM, 11);
        write_random_parquet(&p1, N1, DIM, 22);

        let tmp_out = TempDir::new().unwrap();
        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(4)
            .num_sub_vectors(4)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(10)
            .sample_rate(64)
            .rerank_store(params::RerankStore::Sq8)
            .build();
        let files = vec![
            ParquetFileSpec::of(p0.to_str().unwrap()),
            ParquetFileSpec::of(p1.to_str().unwrap()),
        ];
        let uuid =
            ExternalIvfPqIndex::build(files, "vec", tmp_out.path().to_str().unwrap(), params)
                .await
                .expect("build");

        // Sidecar size covers both files' rows.
        let dir = tmp_out.path().join(uuid.to_string());
        let sz = std::fs::metadata(dir.join("rerank.sq8")).unwrap().len();
        assert_eq!(sz, ((N0 + N1) * DIM) as u64);

        let idx = ExternalIvfPqIndex::open(dir.to_str().unwrap())
            .await
            .expect("open");

        // Queries return valid hits keyed to the correct file; row_index must be
        // in-range for whichever file it names (proves ordinal→(file,row) is
        // consistent end to end).
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..16 {
            let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let results = idx.search(&query, K, 4, 4, None).await.expect("search");
            assert!(!results.is_empty());
            for r in &results {
                let limit = if r.file_path == p0.to_str().unwrap() {
                    N0 as u64
                } else if r.file_path == p1.to_str().unwrap() {
                    N1 as u64
                } else {
                    panic!("unexpected file_path {}", r.file_path);
                };
                assert!(r.row_index < limit, "row_index {} >= {limit}", r.row_index);
            }
        }
    }

    /// The distributed build (train once, shard the assign+encode across N groups,
    /// merge) must produce an index IDENTICAL to building the same rows in one pass.
    /// This is the local correctness oracle for the Spark distributed build: it
    /// shares one trained (ivf, pq) across a 1-shard and a 4-shard assembly (kmeans
    /// is nondeterministic, so both MUST reuse the same quantizers) and asserts
    /// byte-identical search results — proving that (a) per-shard partition
    /// assignment + merge reconstructs the whole-corpus index, and (b) the global
    /// file_id_offset keeps rids consistent so results map to the right (file, row).
    #[tokio::test(flavor = "multi_thread")]
    async fn distributed_shard_build_matches_single_pass() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        const DIM: usize = 16;
        const K: usize = 10;
        const NUM_FILES: usize = 4;
        const PER_FILE: usize = 300;

        let tmp_data = TempDir::new().unwrap();
        let files: Vec<ParquetFileSpec> = (0..NUM_FILES)
            .map(|i| {
                let p = tmp_data.path().join(format!("f{i}.parquet"));
                write_random_parquet(&p, PER_FILE, DIM, 100 + i as u64);
                ParquetFileSpec::of(p.to_str().unwrap())
            })
            .collect();

        let params = ExternalIvfPqIndexParams::builder()
            .num_partitions(4)
            .num_sub_vectors(4)
            .num_bits_per_sub_vector(8)
            .metric(MetricType::L2)
            .max_iters(10)
            .sample_rate(80)
            .rerank_store(params::RerankStore::Sq8)
            .build();

        // Train ONCE, share across both builds (kmeans is nondeterministic, so two
        // independent trainings would diverge — the merge invariant is only defined
        // for a fixed codebook).
        let src = super::build::_test_source(files.clone(), "vec").await;
        let trained = super::build::train_quantizers(&src, &params).await.unwrap();

        let out1 = TempDir::new().unwrap();
        let uuid1 = super::build::build_index_from_shards_local(
            files.clone(),
            "vec",
            out1.path().to_str().unwrap(),
            params.clone(),
            /* num_shards */ 1,
            Some(trained.clone()),
        )
        .await
        .expect("1-shard build");

        let out4 = TempDir::new().unwrap();
        let uuid4 = super::build::build_index_from_shards_local(
            files.clone(),
            "vec",
            out4.path().to_str().unwrap(),
            params.clone(),
            /* num_shards */ 4,
            Some(trained.clone()),
        )
        .await
        .expect("4-shard build");

        let idx1 = ExternalIvfPqIndex::open(out1.path().join(uuid1.to_string()).to_str().unwrap())
            .await
            .expect("open 1-shard");
        let idx4 = ExternalIvfPqIndex::open(out4.path().join(uuid4.to_string()).to_str().unwrap())
            .await
            .expect("open 4-shard");

        // Same queries must return identical (file_path, row_index, distance) top-K.
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..32 {
            let q: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let r1 = idx1.search(&q, K, 4, 4, None).await.expect("search 1");
            let r4 = idx4.search(&q, K, 4, 4, None).await.expect("search 4");
            assert_eq!(r1.len(), r4.len(), "result count differs");
            for (a, b) in r1.iter().zip(r4.iter()) {
                assert_eq!(a.file_path, b.file_path, "file_path differs");
                assert_eq!(a.row_index, b.row_index, "row_index differs");
                assert!((a.distance - b.distance).abs() < 1e-4, "distance differs");
            }
        }
    }
}
