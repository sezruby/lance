# Ray serving benchmark — external IVF-PQ index over parquet

Measures distributed online-serving throughput/latency of the external Lance IVF-PQ
index (`lance.lance.ExternalIvfPqIndex`, the PyO3 binding) on a Ray cluster: N actors
each open one index handle and a driver fans a query stream across them.

## Files
- `ray_knn_bench.py` — the benchmark. Builds (or reuses) the index over a parquet corpus,
  then fans queries across `RAY_KNN_ACTORS` actors and reports aggregate QPS + p50/p95.
- `entrypoint.sh` — installs the shipped pylance wheel, runs the bench.
- `job.yaml` — Anyscale job config template (scrub placeholders before use).

## Run (Anyscale job)
1. Build the linux wheel: `docker run --platform linux/amd64 -v <lance-repo>:/io -w /io/python
   ghcr.io/pyo3/maturin:latest build --release --manylinux 2_28` (needs protoc >= 3.15 in the
   image). Drop the `.whl` next to these files.
2. Fill `job.yaml` placeholders (compute config, storage URIs, `AZURE_STORAGE_ACCOUNT_*`).
3. `anyscale job submit -f job.yaml`.

## Env knobs
`RAY_KNN_{PARQUET_DIR,INDEX_URI,VEC_COL,ACTORS,QUERIES,K,NPROBES,REFINE,STORE,REUSE,MAX_FILES}`.

## Notes
- Queries are passed as plain Python lists (not numpy) across the Ray boundary to avoid
  numpy-pickle version coupling between driver and actor envs.
- On Azure-hosted Anyscale, the ambient workload-identity env is stripped so the
  account key wins (else object-store reads 403 with AuthorizationPermissionMismatch).
- `RAY_KNN_MAX_FILES` caps the corpus: the single-node build holds the encoded corpus in
  RAM, so cap to fit the build node. (Distributed build is the follow-up for large |R|.)
