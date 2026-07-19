#!/usr/bin/env bash
set -euo pipefail
echo "=== node $(hostname) ==="
PY=~/anaconda3/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
$PY -m pip install --force-reinstall ./pylance-*.whl 2>&1 | tail -2
$PY -c "from lance.lance import ExternalIvfPqIndex; print('import OK')"
$PY ray_knn_bench.py
