#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

PASSAGES="${PASSAGES:-$SEARCH_R1_DATA/passages/wiki2018_passages.jsonl}"
INDEX_DIR="${INDEX_DIR:-$SEARCH_R1_DATA/indexes/wiki2018-bm25}"
WIKI_LOG="${WIKI_LOG:-$SEARCH_R1_HOME/logs/prepare_wiki_dpr_w100.log}"
INDEX_LOG="${INDEX_LOG:-$SEARCH_R1_HOME/logs/build_bm25_index.log}"
THREADS="${THREADS:-1}"
JAVA_HEAP="${JAVA_HEAP:-1536m}"
MEMORY_BUFFER_MB="${MEMORY_BUFFER_MB:-256}"

mkdir -p "$(dirname "$INDEX_LOG")" "$(dirname "$INDEX_DIR")"

echo "[index-watcher] started $(date)" | tee -a "$INDEX_LOG"
while tmux has-session -t searchr1_wiki 2>/dev/null; do
  lines="$(wc -l < "$PASSAGES" 2>/dev/null || echo 0)"
  size="$(du -sh "$PASSAGES" 2>/dev/null | cut -f1 || echo 0)"
  echo "[index-watcher] waiting wiki $(date) lines=$lines size=$size" | tee -a "$INDEX_LOG"
  sleep 60
done

echo "[index-watcher] wiki session ended $(date)" | tee -a "$INDEX_LOG"
tail -20 "$WIKI_LOG" | tee -a "$INDEX_LOG" || true

if ! grep -q "done" "$WIKI_LOG"; then
  echo "[index-watcher] wiki did not finish cleanly; refusing to index" | tee -a "$INDEX_LOG"
  exit 1
fi

echo "[index-watcher] final size=$(du -sh "$PASSAGES" | cut -f1)" | tee -a "$INDEX_LOG"
rm -rf "$INDEX_DIR"

python -m search_r1_bm25.retrieval.build_bm25_index \
  --passages "$PASSAGES" \
  --index-dir "$INDEX_DIR" \
  --threads "$THREADS" \
  --java-heap "$JAVA_HEAP" \
  --memory-buffer-mb "$MEMORY_BUFFER_MB" 2>&1 | tee -a "$INDEX_LOG"

echo "[index-watcher] index complete $(date)" | tee -a "$INDEX_LOG"

python -m search_r1_bm25.retrieval.server \
  --index-dir "$INDEX_DIR" \
  --host 127.0.0.1 \
  --port 8008 > "$SEARCH_R1_HOME/logs/retriever_smoke_server.log" 2>&1 &
server_pid="$!"
sleep 8
python -m search_r1_bm25.retrieval.client \
  "author of The Old Man and the Sea" \
  --top-k 3 2>&1 | tee "$SEARCH_R1_HOME/logs/retriever_smoke.log"
kill "$server_pid" || true

echo "[index-watcher] smoke complete $(date)" | tee -a "$INDEX_LOG"
