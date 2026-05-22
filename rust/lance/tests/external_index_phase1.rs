// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Phase 1 integration tests: drive only the public `ExternalIvfPqIndex` API
//! (build / open / search / fetch_rows + RowFilter). These mirror the two
//! existing PoC tests (`external_index_poc.rs`, `external_index_parquet_poc.rs`)
//! but go through the new public surface end-to-end.
//!
//! Run with:
//!     cargo test -p lance --test external_index_phase1 -- --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use arrow::compute::concat;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lance::index::vector::external::{
    ExternalIvfPqIndex, ExternalIvfPqIndexParams, ParquetFileSpec, ParquetRowKey, RowFilter,
    SearchResult,
};
use lance_arrow::FixedSizeListArrayExt;
use lance_linalg::distance::MetricType;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use rand::{Rng, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

const DIM: usize = 8;
const NUM_VECTORS_PER_FILE: usize = 320; // PQ training requires ≥ 256 sampled rows
const NUM_FILES: usize = 3;
const TOTAL_VECTORS: usize = NUM_VECTORS_PER_FILE * NUM_FILES;
const NUM_PARTITIONS: usize = 4;
const NUM_SUB_VECTORS: usize = 2;
const TOP_K: usize = 10;
const REFINE_FACTOR: usize = 8;

fn write_parquet_with_payload(
    path: &PathBuf,
    num_rows: usize,
    dim: usize,
    seed: u64,
    id_offset: i64,
) -> FixedSizeListArray {
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Vec<f32> = (0..num_rows * dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let flat = Float32Array::from(values);
    let fsl = FixedSizeListArray::try_new_from_values(flat, dim as i32).unwrap();

    let ids = Int64Array::from(
        (0..num_rows as i64)
            .map(|i| id_offset + i)
            .collect::<Vec<_>>(),
    );
    let names = StringArray::from(
        (0..num_rows)
            .map(|i| format!("file{seed}-row{i}"))
            .collect::<Vec<_>>(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("vec", fsl.data_type().clone(), false),
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(fsl.clone()) as ArrayRef,
            Arc::new(ids) as ArrayRef,
            Arc::new(names) as ArrayRef,
        ],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(64)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    fsl
}

fn read_back_vectors(path: &PathBuf) -> FixedSizeListArray {
    let file = std::fs::File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.build().unwrap();
    let mut batches: Vec<RecordBatch> = Vec::new();
    for r in reader {
        batches.push(r.unwrap());
    }
    let arrays: Vec<&dyn Array> = batches
        .iter()
        .map(|b| b.column_by_name("vec").unwrap().as_ref())
        .collect();
    let cat = concat(&arrays).unwrap();
    cat.as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap()
        .clone()
}

fn brute_force_topk_global(
    per_file_vectors: &[FixedSizeListArray],
    query: &[f32],
    k: usize,
) -> Vec<(usize, usize, f32)> {
    let dim = per_file_vectors[0].value_length() as usize;
    let mut all: Vec<(usize, usize, f32)> = Vec::new();
    for (file_id, vectors) in per_file_vectors.iter().enumerate() {
        let values = vectors
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let n = vectors.len();
        for i in 0..n {
            let mut s = 0.0f32;
            for d in 0..dim {
                let v = values.value(i * dim + d);
                let q = query[d];
                let diff = v - q;
                s += diff * diff;
            }
            all.push((file_id, i, s));
        }
    }
    all.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    all.into_iter().take(k).collect()
}

/// End-to-end Phase 1 integration:
///
/// 1. Build an external IVF-PQ index over 3 parquet files.
/// 2. Open it.
/// 3. Run a few search queries; confirm SearchResult points at registered files
///    and at least some queries hit the brute-force top-K.
/// 4. Use fetch_rows() to materialize id/name columns for the survivors and
///    confirm the values match the source parquet.
/// 5. Plug a RowFilter that drops half the corpus; confirm survivors don't
///    include filtered rows.
#[tokio::test(flavor = "multi_thread")]
async fn phase1_e2e_build_open_search_fetch_filter() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ---- Stage 1: write parquet files ------------------------------------------
    let tmp_data = TempDir::new().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut per_file_vectors: Vec<FixedSizeListArray> = Vec::new();
    for f in 0..NUM_FILES {
        let p = tmp_data.path().join(format!("part-{f}.parquet"));
        write_parquet_with_payload(
            &p,
            NUM_VECTORS_PER_FILE,
            DIM,
            42 + f as u64,
            (f as i64) * 1_000_000,
        );
        per_file_vectors.push(read_back_vectors(&p));
        paths.push(p);
    }

    // ---- Stage 2: build + open ------------------------------------------------
    let tmp_out = TempDir::new().unwrap();
    let params = ExternalIvfPqIndexParams::builder()
        .num_partitions(NUM_PARTITIONS)
        .num_sub_vectors(NUM_SUB_VECTORS)
        .num_bits_per_sub_vector(8)
        .metric(MetricType::L2)
        .max_iters(10)
        .sample_rate(64)
        .build();
    let files: Vec<ParquetFileSpec> = paths
        .iter()
        .map(|p| ParquetFileSpec::of(p.to_str().unwrap()))
        .collect();
    let uuid = ExternalIvfPqIndex::build(files, "vec", tmp_out.path().to_str().unwrap(), params)
        .await
        .expect("build");
    let idx_dir = tmp_out.path().join(uuid.to_string());
    let idx = ExternalIvfPqIndex::open(idx_dir.to_str().unwrap())
        .await
        .expect("open");

    assert_eq!(idx.num_files(), NUM_FILES);
    assert_eq!(idx.num_partitions(), NUM_PARTITIONS);
    assert_eq!(idx.vector_column(), "vec");

    // ---- Stage 3: search recall ------------------------------------------------
    let mut rng = StdRng::seed_from_u64(7);
    let mut hit_queries = 0;
    let total_queries = 16;
    for _ in 0..total_queries {
        let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let truth = brute_force_topk_global(&per_file_vectors, &query, TOP_K);
        let truth_set: std::collections::HashSet<(usize, u64)> =
            truth.iter().map(|(f, r, _)| (*f, *r as u64)).collect();

        let results: Vec<SearchResult> = idx
            .search(
                &query,
                TOP_K,
                /* nprobes = */ NUM_PARTITIONS,
                REFINE_FACTOR,
                None,
            )
            .await
            .expect("search");

        // SearchResult must point at one of the registered files.
        for r in &results {
            assert!(
                paths.iter().any(|p| p.to_str().unwrap() == r.file_path),
                "search returned unknown file: {}",
                r.file_path
            );
        }

        let result_set: std::collections::HashSet<(usize, u64)> = results
            .iter()
            .map(|r| {
                let file_id = paths
                    .iter()
                    .position(|p| p.to_str().unwrap() == r.file_path)
                    .unwrap();
                (file_id, r.row_index)
            })
            .collect();
        if result_set.intersection(&truth_set).count() >= TOP_K / 2 {
            hit_queries += 1;
        }
    }
    assert!(
        hit_queries >= total_queries / 2,
        "recall too low: only {hit_queries}/{total_queries} queries had ≥ K/2 correct"
    );

    // ---- Stage 4: fetch_rows materializes payload columns ---------------------
    let query: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let results = idx
        .search(&query, TOP_K, NUM_PARTITIONS, REFINE_FACTOR, None)
        .await
        .unwrap();
    let row_keys: Vec<ParquetRowKey> = results
        .iter()
        .map(|r| ParquetRowKey::of(&r.file_path, r.row_index))
        .collect();
    let payload = idx
        .fetch_rows(&row_keys, &["id", "name"])
        .await
        .expect("fetch_rows");
    assert_eq!(payload.num_rows(), results.len());

    // Verify id values match what we wrote per file: row r in file f → id = f*1M + r
    let id_col = payload
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for (i, r) in results.iter().enumerate() {
        let file_id = paths
            .iter()
            .position(|p| p.to_str().unwrap() == r.file_path)
            .unwrap();
        let expected = (file_id as i64) * 1_000_000 + r.row_index as i64;
        assert_eq!(
            id_col.value(i),
            expected,
            "fetched payload mismatch at result {i}: rid=({}, {}), got id={}",
            r.file_path,
            r.row_index,
            id_col.value(i)
        );
    }

    // ---- Stage 5: RowFilter — drop file 0 entirely; confirm survivors exclude it
    struct DropFile(String);
    impl RowFilter for DropFile {
        fn keep(&self, file_path: &str, _row_index: u64) -> bool {
            file_path != self.0
        }
    }
    let filter = DropFile(paths[0].to_str().unwrap().to_string());
    let filtered = idx
        .search(&query, TOP_K, NUM_PARTITIONS, REFINE_FACTOR, Some(&filter))
        .await
        .expect("filtered search");
    for r in &filtered {
        assert_ne!(
            r.file_path,
            paths[0].to_str().unwrap(),
            "filtered search returned a result from the dropped file"
        );
    }

    println!(
        "Phase 1 e2e ✓ — {} files × {} rows; recall hits {}/{}; fetched payloads agree; \
         RowFilter drops {} survivors.",
        NUM_FILES,
        NUM_VECTORS_PER_FILE,
        hit_queries,
        total_queries,
        results.len() - filtered.len()
    );
    let _ = TOTAL_VECTORS;
}
