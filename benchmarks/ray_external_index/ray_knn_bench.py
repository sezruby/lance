#!/usr/bin/env python3
"""
Ray benchmark for the external Lance IVF-PQ index over parquet (PyO3 bindings).

Measures two things on a Ray cluster, using lance.lance.ExternalIvfPqIndex:

  1. BUILD: build the index over the source parquet (single-node build on one Ray
     worker — the PyO3 surface exposes whole-corpus build; distributed shard/merge
     bindings are a follow-up). Reports build wall-clock.
  2. SERVING: N Ray actors each open the built index once; a driver fans a query
     stream across them and measures aggregate QPS + per-query latency (median/p95).
     This is the "distributed online serving" number Spark's batch-join shape can't
     give — many concurrent single-queries across a cluster.

Env:
  RAY_KNN_PARQUET_DIR   parquet dir (abfss/s3) that is the corpus R. Required.
  RAY_KNN_INDEX_URI     where to build/reuse the index (abfss/s3 dir). Required.
  RAY_KNN_VEC_COL       vector column (default "emb").
  RAY_KNN_ACTORS        serving actors (default 8).
  RAY_KNN_QUERIES       total queries to fan across actors (default 2000).
  RAY_KNN_K / NPROBES / REFINE   search params (default 10 / 16 / 8).
  RAY_KNN_STORE         rerank store none|sq8|flat (default sq8).
  RAY_KNN_REUSE         "true" to open an existing index at RAY_KNN_INDEX_URI.
  AZURE_STORAGE_ACCOUNT_NAME / AZURE_STORAGE_ACCOUNT_KEY   object-store creds.
"""
import os
import time
import ray
import numpy as np

# CRITICAL (Anyscale-on-Azure): the cluster injects Azure workload-identity env
# (AZURE_TENANT_ID / CLIENT_ID / FEDERATED_TOKEN_FILE / AUTHORITY_HOST /
# STORAGE_ACCOUNT_URL). lance-io + the Azure SDK read ALL Azure env globally and rank
# workload-identity ABOVE the account key, so reads of our storage account 403 with
# AuthorizationPermissionMismatch. Strip those here (in the driver AND re-stripped in
# each actor/task) so AZURE_STORAGE_ACCOUNT_NAME/KEY win. Done before any Azure client
# is constructed.
_AMBIENT_AZURE = [
    "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_AUTHORITY_HOST", "AZURE_STORAGE_ACCOUNT_URL", "AZURE_CLIENT_SECRET",
]


def _strip_ambient_azure():
    for k in _AMBIENT_AZURE:
        os.environ.pop(k, None)


_strip_ambient_azure()


def _azure_fs():
    """pyarrow AzureFileSystem authed with the account KEY (not ambient identity)."""
    import pyarrow.fs as pafs
    return pafs.AzureFileSystem(
        account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
    )


def _container_path(abfss_uri):
    """abfss://container@acct.dfs.core.windows.net/path -> 'container/path' for pyarrow fs."""
    from urllib.parse import urlparse
    p = urlparse(abfss_uri)
    container = p.netloc.split("@", 1)[0]
    return f"{container}{p.path}"


def _list_parquet(abfss_dir):
    import pyarrow.fs as pafs
    fs = _azure_fs()
    base = _container_path(abfss_dir)
    sel = pafs.FileSelector(base, recursive=False)
    files = [f.path for f in fs.get_file_info(sel) if f.path.endswith(".parquet")]
    # lance wants the full abfss URIs (it has its own object store); rebuild them.
    from urllib.parse import urlparse
    p = urlparse(abfss_dir)
    scheme, netloc = p.scheme, p.netloc
    return sorted(f"{scheme}://{netloc}/{f.split('/', 1)[1]}" for f in files)


def _sample_queries(abfss_dir, vec_col, n):
    """First n non-null vectors from the corpus (deterministic query set), via account-key fs.

    Returns plain Python lists of floats (NOT numpy arrays): Ray pickles query args across
    the driver→actor boundary, and numpy's pickle format is version-coupled — a numpy
    mismatch between driver and actor env raises 'No module named numpy._core.numeric' on
    unpickle. Plain lists sidestep that entirely (and search() takes lists anyway)."""
    import pyarrow.dataset as ds
    fs = _azure_fs()
    base = _container_path(abfss_dir)
    dataset = ds.dataset(base, filesystem=fs, format="parquet")
    tbl = dataset.head(n, columns=[vec_col])
    return [[float(x) for x in v] for v in tbl.column(vec_col).to_pylist()]


@ray.remote
class IndexActor:
    """Holds one open index handle; serves single queries. One per Ray worker slot."""

    def __init__(self, index_uri, vec_col, acct, key):
        import os as _os
        for _k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_FEDERATED_TOKEN_FILE",
                   "AZURE_AUTHORITY_HOST", "AZURE_STORAGE_ACCOUNT_URL", "AZURE_CLIENT_SECRET"):
            _os.environ.pop(_k, None)
        _os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = acct
        _os.environ["AZURE_STORAGE_ACCOUNT_KEY"] = key
        from lance.lance import ExternalIvfPqIndex
        self.idx = ExternalIvfPqIndex.open(index_uri)
        self.vec_col = vec_col

    def warmup(self, q, k, nprobes, refine):
        self.idx.search(list(q), k, nprobes, refine)
        return True

    def search_many(self, queries, k, nprobes, refine):
        """Run a batch of single-query searches, return per-query latencies (ms)."""
        lats = []
        for q in queries:
            t0 = time.perf_counter()
            self.idx.search(list(q), k, nprobes, refine)
            lats.append((time.perf_counter() - t0) * 1000.0)
        return lats


def main():
    parquet_dir = os.environ["RAY_KNN_PARQUET_DIR"]
    index_uri = os.environ["RAY_KNN_INDEX_URI"]
    vec_col = os.environ.get("RAY_KNN_VEC_COL", "emb")
    n_actors = int(os.environ.get("RAY_KNN_ACTORS", "8"))
    n_queries = int(os.environ.get("RAY_KNN_QUERIES", "2000"))
    k = int(os.environ.get("RAY_KNN_K", "10"))
    nprobes = int(os.environ.get("RAY_KNN_NPROBES", "16"))
    refine = int(os.environ.get("RAY_KNN_REFINE", "8"))
    store = os.environ.get("RAY_KNN_STORE", "sq8")
    reuse = os.environ.get("RAY_KNN_REUSE", "false").lower() == "true"

    # The entrypoint pip-installs the wheel on the HEAD only; Ray tasks/actors run on
    # WORKER nodes that don't have it. Ship the wheel via runtime_env pip so every node
    # installs pylance before running our code. Glob the wheel shipped in the job dir.
    import glob
    wheels = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylance-*.whl"))
    runtime_env = {"pip": wheels} if wheels else None
    if runtime_env:
        print(f"runtime_env pip wheel: {os.path.basename(wheels[0])}")
    ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    print(f"ray resources: {ray.cluster_resources()}")

    files = _list_parquet(parquet_dir)
    # Optional file cap: the single-node build holds the whole encoded corpus in RAM
    # (shard_partition_streams collects all batches), so cap files to fit the build node's
    # memory. Per-query SERVING latency is corpus-size-independent, so a subset still gives
    # representative serving numbers. RAY_KNN_MAX_FILES=0 = all.
    max_files = int(os.environ.get("RAY_KNN_MAX_FILES", "0"))
    if max_files > 0:
        files = files[:max_files]
    print(f"corpus: {len(files)} parquet files under {parquet_dir}"
          + (f" (capped to {max_files})" if max_files else ""))

    # ---- BUILD (single-node, on a Ray task) ----
    if reuse:
        built_uri = index_uri
        print(f"reusing existing index at {built_uri}")
    else:
        acct = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
        key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

        @ray.remote(num_cpus=8)
        def build_index(files, vec_col, index_uri, store, acct, key):
            import os as _os
            for _k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_FEDERATED_TOKEN_FILE",
                       "AZURE_AUTHORITY_HOST", "AZURE_STORAGE_ACCOUNT_URL", "AZURE_CLIENT_SECRET"):
                _os.environ.pop(_k, None)
            _os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = acct
            _os.environ["AZURE_STORAGE_ACCOUNT_KEY"] = key
            from lance.lance import ExternalIvfPqIndex
            import time as _t
            t0 = _t.time()
            idx = ExternalIvfPqIndex.build(
                file_paths=files, vector_column=vec_col, output_uri=index_uri,
                num_partitions=256, num_sub_vectors=16, num_bits_per_sub_vector=8,
                metric="l2", rerank_store=store)
            return _t.time() - t0, idx.num_files, idx.num_partitions
        build_ms, nf, npart = ray.get(
            build_index.remote(files, vec_col, index_uri, store, acct, key))
        print(f"BUILD: {build_ms:.1f}s  files={nf} partitions={npart}  (single-node on one worker)")
        built_uri = index_uri  # actors resolve the uuid dir below

    # The build writes index_uri/<uuid>; discover it for the actors.
    # (fs listing of the index dir; pick the single uuid subdir)
    built_index_dir = _resolve_index_dir(built_uri)
    print(f"index dir: {built_index_dir}")

    # ---- SERVING (distributed across actors) ----
    queries = _sample_queries(parquet_dir, vec_col, n_queries)
    print(f"sampled {len(queries)} queries")

    acct = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    actors = [IndexActor.remote(built_index_dir, vec_col, acct, key) for _ in range(n_actors)]
    ray.get([a.warmup.remote(queries[0], k, nprobes, refine) for a in actors])

    # Shard queries across actors, run concurrently, time the whole fan-out.
    shards = [queries[i::n_actors] for i in range(n_actors)]
    t0 = time.perf_counter()
    lat_lists = ray.get([
        actors[i].search_many.remote(shards[i], k, nprobes, refine)
        for i in range(n_actors)
    ])
    wall = time.perf_counter() - t0

    all_lat = sorted(l for sub in lat_lists for l in sub)
    total = len(all_lat)
    qps = total / wall
    p50 = all_lat[total // 2]
    p95 = all_lat[min(total - 1, int(total * 0.95))]
    print("=" * 80)
    print(f"SERVING: actors={n_actors} queries={total} k={k} nprobes={nprobes} refine={refine} store={store}")
    print(f"  aggregate QPS = {qps:.1f}   wall = {wall:.2f}s")
    print(f"  per-query latency: median={p50:.1f}ms  p95={p95:.1f}ms  mean={sum(all_lat)/total:.1f}ms")
    print("=" * 80)


def _resolve_index_dir(index_uri):
    """build() wrote index_uri/<uuid>; return the full abfss URI of that single uuid subdir,
    listing via the account-key fs (not URI auto-auth, which hits ambient identity)."""
    import pyarrow.fs as pafs
    from urllib.parse import urlparse
    fs = _azure_fs()
    base = _container_path(index_uri)
    sel = pafs.FileSelector(base, recursive=False)
    subdirs = [f.path for f in fs.get_file_info(sel) if f.type == pafs.FileType.Directory]
    if not subdirs:
        return index_uri
    leaf = subdirs[-1].rstrip("/").rsplit("/", 1)[-1]
    p = urlparse(index_uri)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}/{leaf}"


if __name__ == "__main__":
    main()
