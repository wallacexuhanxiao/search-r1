#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

CONFIG="${1:-$SEARCH_R1_REPO/configs/search_r1_qwen3b_bm25.yaml}"

echo "verl-agent training entrypoint placeholder"
echo "Config: $CONFIG"
echo "Wire this to the installed verl-agent CLI once the exact package/API is confirmed."
