#!/usr/bin/env python3
"""Distributed lance merge_insert over Ray — proof of the key-partition + combine-commit kernel.

Pattern (the whole distributed merge, no Spark):
  1. driver hash-partitions the merge SOURCE by key into N shards
  2. each Ray task runs MergeInsertBuilder.execute_uncommitted on its shard against the
     target (writes new data fragments, does NOT commit) → returns a Transaction
  3. driver calls target.commit_merge_transactions([...]) → combine (union per-fragment
     deletion vectors) + one atomic commit

This validates the pieces we added to the Python binding (commit_merge_transactions) and
that the source-key partitioning invariant holds end-to-end.

Run locally:
    python distributed_merge.py --target-rows 5_000_000 --source-frac 0.2 --workers 8 --uri /tmp/merge_target.lance

On Anyscale (object store): pass an abfss:// --uri and --storage-options.
"""

import argparse
import time

import pyarrow as pa
import pyarrow.compute as pc
import ray

import lance


def gen_target(uri, n_rows, storage_options, max_rows_per_file):
    """Write a fresh target dataset: id (int64 key) + value (int64). Many fragments."""
    ids = pa.array(range(n_rows), type=pa.int64())
    vals = pc.multiply(ids, pa.scalar(10, pa.int64()))
    tbl = pa.table({"id": ids, "value": vals})
    lance.write_dataset(
        tbl,
        uri,
        mode="overwrite",
        storage_options=storage_options,
        max_rows_per_file=max_rows_per_file,
    )
    return lance.dataset(uri, storage_options=storage_options)


def make_source(target_uri, source_frac, storage_options, seed=42):
    """Source = a fraction of target ids with value bumped (+1). Pure updates."""
    ds = lance.dataset(target_uri, storage_options=storage_options)
    n = ds.count_rows()
    # deterministic sample: every k-th id, so keys are unique and cover the id range
    k = max(1, int(1 / source_frac))
    ids = pa.array(range(0, n, k), type=pa.int64())
    vals = pc.add(pc.multiply(ids, pa.scalar(10, pa.int64())), pa.scalar(1, pa.int64()))
    return pa.table({"id": ids, "value": vals})


def partition_by_key(source_tbl, n_parts):
    """Hash-partition the source by `id` into n_parts disjoint shards (Arrow tables).

    Invariant for combine_merge_transactions: every occurrence of a key lands in exactly
    one shard, so a target row is modified by at most one transaction.
    """
    ids = source_tbl.column("id").to_numpy()
    buckets = ids % n_parts
    shards = []
    for p in range(n_parts):
        mask = pa.array(buckets == p)
        shards.append(source_tbl.filter(mask))
    return shards


@ray.remote
def merge_shard(target_uri, storage_options, shard):
    """Run an uncommitted matched-update merge of one source shard against the target.

    Returns the uncommitted Transaction (a picklable dataclass) + stats. The Transaction
    is combined + committed on the driver via commit_merge_transactions.
    """
    ds = lance.dataset(target_uri, storage_options=storage_options)
    builder = ds.merge_insert(on="id").when_matched_update_all()
    txn, stats = builder.execute_uncommitted(shard)
    return txn, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--target-rows", type=int, default=5_000_000)
    ap.add_argument("--source-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rows-per-file", type=int, default=100_000)
    ap.add_argument("--storage-option", action="append", default=[],
                    help="key=value, repeatable")
    ap.add_argument("--skip-gen", action="store_true", help="target already exists")
    args = ap.parse_args()

    storage_options = dict(kv.split("=", 1) for kv in args.storage_option) or None

    ray.init(address="auto", ignore_reinit_error=True)
    print("ray resources:", ray.cluster_resources())

    if not args.skip_gen:
        t0 = time.time()
        gen_target(args.uri, args.target_rows, storage_options, args.max_rows_per_file)
        print(f"[gen] target {args.target_rows} rows in {time.time()-t0:.1f}s")

    target = lance.dataset(args.uri, storage_options=storage_options)
    base_rows = target.count_rows()
    base_frags = len(target.get_fragments())
    print(f"target: {base_rows} rows, {base_frags} fragments")

    source = make_source(args.uri, args.source_frac, storage_options)
    print(f"source: {source.num_rows} rows ({args.source_frac:.2%} of target)")

    shards = partition_by_key(source, args.workers)
    print(f"partitioned source into {len(shards)} shards; "
          f"sizes={[s.num_rows for s in shards]}")

    # --- distributed uncommitted merges ---
    t0 = time.time()
    futures = [
        merge_shard.remote(args.uri, storage_options, s)
        for s in shards if s.num_rows > 0
    ]
    results = ray.get(futures)
    merge_s = time.time() - t0
    txns = [txn for txn, _stats in results]
    total_updated = sum(st.get("num_updated_rows", 0) for _t, st in results)
    print(f"[merge] {len(txns)} uncommitted txns in {merge_s:.1f}s; "
          f"updated≈{total_updated}")

    # --- combine + commit on the driver ---
    t0 = time.time()
    merged = target.commit_merge_transactions(txns)
    commit_s = time.time() - t0
    print(f"[commit] combined + committed in {commit_s:.1f}s")

    # --- verify ---
    final = lance.dataset(args.uri, storage_options=storage_options)
    final_rows = final.count_rows()
    final_frags = len(final.get_fragments())
    # updated rows should now have value = id*10 + 1; sample-check a few source ids
    sample_ids = source.column("id").to_pylist()[:5]
    tbl = final.to_table(filter=f"id in ({','.join(map(str, sample_ids))})").to_pydict()
    ok_rows = final_rows == base_rows
    print("=== RESULT ===")
    print(f"rows: {final_rows} (expected {base_rows}) {'OK' if ok_rows else 'MISMATCH'}")
    print(f"fragments: {base_frags} -> {final_frags}")
    print(f"merge={merge_s:.1f}s commit={commit_s:.1f}s")
    print(f"sample updated rows: {tbl}")


if __name__ == "__main__":
    main()
