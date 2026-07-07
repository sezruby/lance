// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//
//! Benchmark validating the fragment-scoped merge optimization.
//!
//! A distributed merge splits the source across N tasks and each task runs an
//! uncommitted `merge_insert` against the target. Today every task scans the
//! WHOLE target, so the aggregate target I/O is `O(N * target)`. With
//! `try_target_fragments`, each task scans only its own slice of the target's
//! fragments, so the aggregate target I/O is `O(target)` — the target is read
//! once in total instead of N times.
//!
//! This bench measures ONE full round (all N tasks) of a matched-only merge in
//! two modes, on a single machine so the difference isolates the target-scan
//! cost (no Spark / cluster noise):
//!
//! - `full_scan_per_task`: each task merges the whole source against the whole
//!   target (the current behavior). Cost grows with N.
//! - `fragment_scoped_per_task`: each task merges the whole source against only
//!   its assigned fragment slice (the optimization). Cost is ~constant in N.
//!
//! Both modes are matched-only (`WhenNotMatched::DoNothing`), which is the shape
//! `try_target_fragments` supports and the shape a pure-update upsert uses. The
//! per-task transactions are NOT committed here — we measure the scan+merge
//! work, which is where the `O(N * target)` blowup lives.
//!
//! Run with `cargo bench --bench fragment_scoped_merge`.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{Criterion, criterion_group, criterion_main};
use futures::future::try_join_all;
use lance::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_core::utils::tempfile::TempStrDir;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

// A wide-ish target split into many fragments so the per-task scan cost is
// meaningful and the N-way fan-out is visible. ROWS_PER_FRAG * NUM_FRAGS rows.
const ROWS_PER_FRAG: u64 = 5_000;
const NUM_FRAGS: u64 = 64;
// Number of distributed tasks the source is split across (== target scans in
// the full-scan mode; == fragment slices in the scoped mode).
const NUM_TASKS: usize = 8;
// Source rows PER TASK (matched updates of existing keys). The whole source in
// scoped mode is handed to each task, but we keep it modest so the target scan
// — not the source hash-build — dominates, which is the cost we are attacking.
const SOURCE_ROWS_PER_TASK: usize = 2_000;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn make_batch(start_id: i64, n: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start_id..start_id + n as i64);
    let vals = Int64Array::from_iter_values((start_id..start_id + n as i64).map(|v| v * 10));
    RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(vals)]).unwrap()
}

fn make_batches(total_rows: u64) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    let mut remaining = total_rows;
    let mut next_start = 0i64;
    while remaining > 0 {
        let n = remaining.min(ROWS_PER_FRAG) as usize;
        out.push(make_batch(next_start, n));
        next_start += n as i64;
        remaining -= n as u64;
    }
    out
}

async fn build_base(path: &str) -> Dataset {
    let total = ROWS_PER_FRAG * NUM_FRAGS;
    let params = WriteParams {
        max_rows_per_file: ROWS_PER_FRAG as usize,
        max_rows_per_group: ROWS_PER_FRAG as usize,
        mode: WriteMode::Create,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(make_batches(total).into_iter().map(Ok), schema());
    Dataset::write(reader, path, Some(params)).await.unwrap();
    Dataset::open(path).await.unwrap()
}

/// A task's source: `SOURCE_ROWS_PER_TASK` matched updates of existing keys,
/// starting at `base`. New value = id * 10 + 1 so updates are observable.
fn task_source(base: i64) -> RecordBatch {
    let ids = Int64Array::from_iter_values(base..base + SOURCE_ROWS_PER_TASK as i64);
    let vals = Int64Array::from_iter_values(
        (base..base + SOURCE_ROWS_PER_TASK as i64).map(|v| v * 10 + 1),
    );
    RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(vals)]).unwrap()
}

fn source_reader(
    base: i64,
) -> RecordBatchIterator<std::vec::IntoIter<arrow::error::Result<RecordBatch>>> {
    RecordBatchIterator::new(vec![Ok(task_source(base))].into_iter(), schema())
}

/// One task's uncommitted matched-only merge scanning the WHOLE target.
async fn full_scan_task(ds: Arc<Dataset>, base: i64) {
    let mut builder = MergeInsertBuilder::try_new(ds, vec!["id".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing);
    let job = builder.try_build().unwrap();
    let _ = job.execute_uncommitted(source_reader(base)).await.unwrap();
}

/// One task's uncommitted matched-only merge scanning ONLY `fragment_ids`.
async fn fragment_scoped_task(ds: Arc<Dataset>, base: i64, fragment_ids: Vec<u32>) {
    let mut builder = MergeInsertBuilder::try_new(ds, vec!["id".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_target_fragments(fragment_ids);
    let job = builder.try_build().unwrap();
    let _ = job.execute_uncommitted(source_reader(base)).await.unwrap();
}

/// Assign the target's fragments round-robin to NUM_TASKS slices.
fn fragment_slices(ds: &Dataset) -> Vec<Vec<u32>> {
    let mut slices = vec![Vec::new(); NUM_TASKS];
    for (i, frag) in ds.get_fragments().iter().enumerate() {
        slices[i % NUM_TASKS].push(frag.id() as u32);
    }
    slices
}

fn bench_fragment_scoped_merge(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = TempStrDir::default();
    let path = dir.as_str().to_string();
    let ds = Arc::new(rt.block_on(build_base(&path)));
    let slices = fragment_slices(&ds);
    // Each task updates a distinct key range so, across a round, the tasks
    // cover different rows (mirrors a key-partitioned source).
    let task_bases: Vec<i64> = (0..NUM_TASKS)
        .map(|t| (t as i64) * SOURCE_ROWS_PER_TASK as i64)
        .collect();

    // Baseline: every task scans the whole target → O(NUM_TASKS * target).
    c.bench_function("fragment_scoped_merge/full_scan_per_task", |b| {
        b.iter(|| {
            rt.block_on(async {
                let futs = task_bases
                    .iter()
                    .map(|&base| full_scan_task(ds.clone(), base));
                try_join_all(futs.map(tokio::spawn)).await.unwrap();
            })
        })
    });

    // Optimization: each task scans only its fragment slice → O(target) total.
    c.bench_function("fragment_scoped_merge/fragment_scoped_per_task", |b| {
        b.iter(|| {
            rt.block_on(async {
                let futs = task_bases
                    .iter()
                    .zip(slices.iter())
                    .map(|(&base, slice)| fragment_scoped_task(ds.clone(), base, slice.clone()));
                try_join_all(futs.map(tokio::spawn)).await.unwrap();
            })
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fragment_scoped_merge);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_fragment_scoped_merge);

criterion_main!(benches);
