#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

PASSAGES="${1:-$SEARCH_R1_DATA/passages/wiki2018_passages.jsonl}"
INDEX_DIR="${2:-$SEARCH_R1_DATA/indexes/wiki2018-bm25}"

python -m search_r1_bm25.retrieval.build_bm25_index \
  --passages "$PASSAGES" \
  --index-dir "$INDEX_DIR" \
  --threads "${THREADS:-8}"
