#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATASET="${1:?usage: scripts/evaluate.sh /path/to/eval.jsonl}"
python -m search_r1_bm25.evaluation.retrieval_metrics --dataset "$DATASET" --endpoint "${ENDPOINT:-http://127.0.0.1:8008/search}"
