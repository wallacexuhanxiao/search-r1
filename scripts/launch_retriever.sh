#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
PYTHON=${PYTHON:-/root/vllm_env/bin/python}

INDEX_DIR="${1:-$SEARCH_R1_DATA/indexes/wiki2018-bm25}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8008}"

"$PYTHON" -m search_r1_bm25.retrieval.server --index-dir "$INDEX_DIR" --host "$HOST" --port "$PORT"
