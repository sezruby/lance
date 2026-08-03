// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#![allow(clippy::print_stdout)]

//! Phase 1 integration tests for the external scalar (BTree) index: drive only
//! the public `ExternalBtreeIndex` API (build / open / search_keys / fetch_rows +
//! RowFilter). Mirrors `external_index_phase1.rs` for the vector index.
//!
//! Run with:
//!     cargo test -p lance --test external_scalar_index_phase1 -- --nocapture

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::scalar::ScalarValue;
use lance::index::scalar_external::{
    ExternalBtreeIndex, ExternalBtreeIndexParams, ParquetFileSpec, ParquetRowKey, RowFilter,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const ROWS_PER_FILE: usize = 100;
const NUM_FILES: usize = 3;

/// file `f`, local row `r` → id = f*100 + r, name = "f{f}-r{r}".
fn write_scalar_parquet(path: &PathBuf, file_idx: usize, num_rows: usize) {
    let base = (file_idx * 100) as i64;
    let ids = Int64Array::from((0..num_rows as i64).map(|r| base + r).collect::<Vec<_>>());
    let names = StringArray::from(
        (0..num_rows)
            .map(|r| format!("f{file_idx}-r{r}"))
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids) as ArrayRef, Arc::new(names) as ArrayRef],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    // Small pages so the sorted-key BTree spans multiple pages across files.
    let props = WriterProperties::builder()
        .set_data_page_row_count_limit(32)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// End-to-end Phase 1: build → open → search_keys → RowFilter → fetch_rows over a
/// BTree indexing a globally-unique `id` column across 3 parquet files.
#[tokio::test(flavor = "multi_thread")]
async fn phase1_e2e_build_open_search_filter_fetch() {
    let _ = env_logger::builder().is_test(true).try_init();

    // ---- Stage 1: write parquet files ----------------------------------------
    let tmp_data = TempDir::new().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    for f in 0..NUM_FILES {
        let p = tmp_data.path().join(format!("part-{f}.parquet"));
        write_scalar_parquet(&p, f, ROWS_PER_FILE);
        paths.push(p);
    }

    // ---- Stage 2: build + open -----------------------------------------------
    let tmp_out = TempDir::new().unwrap();
    let files: Vec<ParquetFileSpec> = paths
        .iter()
        .map(|p| ParquetFileSpec::of(p.to_str().unwrap()))
        .collect();
    let params = ExternalBtreeIndexParams::default().with_batch_size(64);
    let uuid = ExternalBtreeIndex::build(files, "id", tmp_out.path().to_str().unwrap(), params)
        .await
        .expect("build");
    let idx_dir = tmp_out.path().join(uuid.to_string());
    let idx = ExternalBtreeIndex::open(idx_dir.to_str().unwrap())
        .await
        .expect("open");

    assert_eq!(idx.num_files(), NUM_FILES);
    assert_eq!(idx.key_column(), "id");
    assert_eq!(idx.file_path(0), Some(paths[0].to_str().unwrap()));
    assert_eq!(idx.file_path(NUM_FILES as u32), None);

    // Helper: resolve a result's (file_id, row) from its file_path.
    let file_id_of = |file_path: &str| -> usize {
        paths
            .iter()
            .position(|p| p.to_str().unwrap() == file_path)
            .unwrap_or_else(|| panic!("search returned unknown file: {file_path}"))
    };

    // ---- Stage 3: search_keys spanning multiple files (+ an absent key) -------
    // id 5   -> (file0, row 5)
    // id 100 -> (file1, row 0)
    // id 150 -> (file1, row 50)
    // id 250 -> (file2, row 50)
    // id 999 -> absent
    let keys: Vec<ScalarValue> = [5i64, 100, 150, 250, 999]
        .into_iter()
        .map(|v| ScalarValue::Int64(Some(v)))
        .collect();
    let results = idx.search_keys(&keys, None).await.expect("search_keys");

    let got: HashSet<(usize, u64)> = results
        .iter()
        .map(|r| (file_id_of(&r.file_path), r.row_index))
        .collect();
    let expected: HashSet<(usize, u64)> = [(0usize, 5u64), (1, 0), (1, 50), (2, 50)]
        .into_iter()
        .collect();
    assert_eq!(got, expected, "search_keys returned wrong (file, row) set");
    // distance is meaningless for a scalar lookup and is fixed at 0.0.
    assert!(results.iter().all(|r| r.distance == 0.0));

    // ---- Stage 4: RowFilter drops one matched row -----------------------------
    // Drop exactly (file1, row 50) == id 150; the other three must survive.
    struct DropOne {
        file_path: String,
        row_index: u64,
    }
    impl RowFilter for DropOne {
        fn keep(&self, file_path: &str, row_index: u64) -> bool {
            !(file_path == self.file_path && row_index == self.row_index)
        }
    }
    let filter = DropOne {
        file_path: paths[1].to_str().unwrap().to_string(),
        row_index: 50,
    };
    let filtered = idx
        .search_keys(&keys, Some(&filter))
        .await
        .expect("filtered search_keys");
    let got_filtered: HashSet<(usize, u64)> = filtered
        .iter()
        .map(|r| (file_id_of(&r.file_path), r.row_index))
        .collect();
    let expected_filtered: HashSet<(usize, u64)> =
        [(0usize, 5u64), (1, 0), (2, 50)].into_iter().collect();
    assert_eq!(
        got_filtered, expected_filtered,
        "RowFilter should have dropped exactly (file1, row 50)"
    );
    assert!(
        !got_filtered.contains(&(1, 50)),
        "dropped row leaked into filtered results"
    );

    // ---- Stage 5: fetch_rows materializes payload for the survivors -----------
    let row_keys: Vec<ParquetRowKey> = filtered
        .iter()
        .map(|r| ParquetRowKey::of(&r.file_path, r.row_index))
        .collect();
    let payload = idx
        .fetch_rows(&row_keys, &["id", "name"])
        .await
        .expect("fetch_rows");
    assert_eq!(payload.num_rows(), filtered.len());

    let id_col = payload
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let name_col = payload
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // fetch_rows preserves caller-input order, so row i corresponds to filtered[i].
    for (i, r) in filtered.iter().enumerate() {
        let fid = file_id_of(&r.file_path);
        let expected_id = (fid as i64) * 100 + r.row_index as i64;
        let expected_name = format!("f{fid}-r{}", r.row_index);
        assert_eq!(
            id_col.value(i),
            expected_id,
            "fetched id mismatch at survivor {i} ({}, {})",
            r.file_path,
            r.row_index
        );
        assert_eq!(
            name_col.value(i),
            expected_name,
            "fetched name mismatch at survivor {i}"
        );
    }

    // Absent-only query returns an empty set.
    let absent = idx
        .search_keys(&[ScalarValue::Int64(Some(100_000))], None)
        .await
        .expect("absent search_keys");
    assert!(absent.is_empty(), "absent key must return no rows");

    println!(
        "external scalar (btree) phase1 e2e OK: {} files x {} rows; \
         search_keys set exact; RowFilter drops 1; fetch_rows payload agrees.",
        NUM_FILES, ROWS_PER_FILE
    );
}
