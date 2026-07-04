// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//
//! Benchmark validating the fragment-scoped delete optimization.
//!
//! A distributed / parallel delete splits the target's fragments across N tasks
//! and each task runs an uncommitted delete of a shared predicate. Without
//! scoping, every task would scan the WHOLE target to find matching rows, so the
//! aggregate target I/O is `O(N * target)` — the pattern that OOMs and stalls a
//! large distributed delete. With `DeleteBuilder::with_target_fragments`, each
//! task scans only its own slice of the target's fragments, so the aggregate
//! target I/O is `O(target)` — the target is read once in total instead of N
//! times.
//!
//! This bench measures ONE full round (all N tasks) of the same predicate delete
//! in two modes, on a single machine so the difference isolates the target-scan
//! cost (no Spark / cluster noise):
//!
//! - `full_scan_per_task`: each task deletes over the whole target (the naive
//!   distributed shape). Aggregate scan cost grows with N.
//! - `fragment_scoped_per_task`: each task deletes only within its assigned
//!   fragment slice (the optimization). Aggregate scan cost is ~constant in N.
//!
//! The per-task transactions are NOT committed here — we measure the scan +
//! deletion-application work, which is where the `O(N * target)` blowup lives.
//!
//! Caveat: a single-node delete already parallelizes its own scan across cores
//! (DataFusion `target_partitions = num_cpus`), so the honest single-node story
//! is the aggregate work fanned out across N tasks plus the distributed
//! `O(N * target) -> O(target)` collapse, not a raw per-scan speedup. Framing the
//! bench as "aggregate work across N tasks" matches the distributed model this
//! optimization targets.
//!
//! Run with `cargo bench --bench fragment_scoped_delete`.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use futures::future::try_join_all;
use lance::dataset::write::DeleteBuilder;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_core::utils::tempfile::TempStrDir;
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};

// A wide-ish target split into many fragments so the per-task scan cost is
// meaningful and the N-way fan-out is visible. ROWS_PER_FRAG * NUM_FRAGS rows.
const ROWS_PER_FRAG: u64 = 5_000;
const NUM_FRAGS: u64 = 64;
// Number of distributed tasks the fragments are split across (== whole-target
// scans in the full-scan mode; == fragment slices in the scoped mode).
const NUM_TASKS: usize = 8;

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

// A predicate that matches ~10% of rows in every fragment, so both modes do
// real deletion work and every fragment contributes.
const PREDICATE: &str = "id % 10 = 0";

/// One task's uncommitted delete scanning the WHOLE target.
async fn full_scan_task(ds: Arc<Dataset>) {
    let _ = DeleteBuilder::new(ds, PREDICATE)
        .execute_uncommitted()
        .await
        .unwrap();
}

/// One task's uncommitted delete scanning ONLY `fragment_ids`.
async fn fragment_scoped_task(ds: Arc<Dataset>, fragment_ids: Vec<u32>) {
    let _ = DeleteBuilder::new(ds, PREDICATE)
        .with_target_fragments(fragment_ids)
        .execute_uncommitted()
        .await
        .unwrap();
}

/// Assign the target's fragments round-robin to NUM_TASKS disjoint slices.
fn fragment_slices(ds: &Dataset) -> Vec<Vec<u32>> {
    let mut slices = vec![Vec::new(); NUM_TASKS];
    for (i, frag) in ds.get_fragments().iter().enumerate() {
        slices[i % NUM_TASKS].push(frag.id() as u32);
    }
    slices
}

/// Build a fresh target in its own temp dir. Returned as the (untimed) per-
/// iteration setup so each timed round starts from a clean store: an
/// uncommitted delete still WRITES per-fragment deletion files to storage, and
/// reusing one dataset would let those files accumulate across iterations and
/// bias the full-scan mode (which writes NUM_TASKS× as many) more than the
/// scoped mode. The TempStrDir is returned alongside the dataset so it stays
/// alive during the timed round and is dropped (cleaning the files) after it.
fn fresh_target(rt: &tokio::runtime::Runtime) -> (TempStrDir, Arc<Dataset>, Vec<Vec<u32>>) {
    let dir = TempStrDir::default();
    let ds = Arc::new(rt.block_on(build_base(dir.as_str())));
    let slices = fragment_slices(&ds);
    (dir, ds, slices)
}

fn bench_fragment_scoped_delete(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Baseline: every task scans the whole target → O(NUM_TASKS * target).
    c.bench_function("fragment_scoped_delete/full_scan_per_task", |b| {
        b.iter_batched(
            || fresh_target(&rt),
            |(_dir, ds, _slices)| {
                rt.block_on(async {
                    let futs = (0..NUM_TASKS).map(|_| full_scan_task(ds.clone()));
                    try_join_all(futs.map(tokio::spawn)).await.unwrap();
                })
            },
            BatchSize::PerIteration,
        )
    });

    // Optimization: each task scans only its fragment slice → O(target) total.
    c.bench_function("fragment_scoped_delete/fragment_scoped_per_task", |b| {
        b.iter_batched(
            || fresh_target(&rt),
            |(_dir, ds, slices)| {
                rt.block_on(async {
                    let futs = slices
                        .iter()
                        .map(|slice| fragment_scoped_task(ds.clone(), slice.clone()));
                    try_join_all(futs.map(tokio::spawn)).await.unwrap();
                })
            },
            BatchSize::PerIteration,
        )
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_fragment_scoped_delete);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_fragment_scoped_delete);

criterion_main!(benches);
