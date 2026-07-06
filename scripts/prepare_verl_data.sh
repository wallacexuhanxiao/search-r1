#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/search-r1-bm25}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}

cd "$ROOT/repo"
"$PYTHON" data/prepare_verl_parquet.py \
  --splits-dir "$ROOT/data/splits" \
  --output-dir "$ROOT/data/verl"
