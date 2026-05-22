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
mod fetch;
pub(crate) mod manifest;
pub(crate) mod open;
pub mod params;
pub(crate) mod parquet_source;
mod search;
pub mod types;

pub use params::ExternalIvfPqIndexParams;
pub use types::{ParquetFileSpec, ParquetRowKey, RowFilter, SearchResult};

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
                for d in 0..dim {
                    let v = values.value(i * dim + d);
                    let q = query[d];
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
}
