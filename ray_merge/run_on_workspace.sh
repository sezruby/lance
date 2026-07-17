#!/usr/bin/env bash
# Run INSIDE the Anyscale workspace terminal (esong-lance-merge) to build the custom
# pylance wheel (with commit_merge_transactions) from this branch and run the distributed
# merge. The workspace is a Linux env with the Rust toolchain, so this avoids the broken
# local Mac build and any cross-compile.
#
# The binding depends on the OneAdobe rust/lance crate (combine_merge_transactions), so we
# build the WHOLE python extension from the repo (maturin builds rust/lance transitively).
set -euo pipefail

REPO_URL="https://github.com/sezruby/lance.git"
BRANCH="feat/py-combine-merge"
WORKDIR="${WORKDIR:-/mnt/cluster_storage/lance-ray}"

echo "=== 1. clone/pull the branch ==="
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch origin "$BRANCH" && git -C "$WORKDIR" checkout "$BRANCH" && git -C "$WORKDIR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR/python"

echo "=== 2. build + install the custom pylance wheel (Linux, native toolchain) ==="
# Anyscale images ship maturin + rust; if not: pip install maturin, rustup default stable.
pip install -U maturin >/dev/null 2>&1 || true
# Build release wheel of the local extension (compiles rust/lance transitively — slow first time).
maturin build --release --out /tmp/pylance_wheels
pip install --force-reinstall /tmp/pylance_wheels/pylance-*.whl

echo "=== 3. verify the new binding is present ==="
python -c "import lance; d=lance.__file__; import inspect; \
  assert hasattr(lance.dataset.LanceDataset, 'commit_merge_transactions'), 'binding missing'; \
  print('commit_merge_transactions present OK', d)"

echo "=== 4. run the distributed merge (Ray auto-connects to the workspace cluster) ==="
cd "$WORKDIR"
python ray_merge/distributed_merge.py \
  --uri /mnt/cluster_storage/merge_target.lance \
  --target-rows "${TARGET_ROWS:-5000000}" \
  --source-frac "${SOURCE_FRAC:-0.2}" \
  --workers "${WORKERS:-8}" \
  --max-rows-per-file 100000
