#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/search-r1-bm25}
PYTHON=${PYTHON:-/root/vllm_env/bin/python}
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

echo "[auto-train] started $(date)"
echo "[auto-train] waiting for GPU and expanded memory cgroup"

while true; do
  gpu_ok=0
  mem_ok=0
  if nvidia-smi >/dev/null 2>&1; then
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
      gpu_ok=1
    fi
  fi
  mem_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)"
  if [ "$mem_max" = "max" ] || [ "${mem_max:-0}" -ge 50000000000 ] 2>/dev/null; then
    mem_ok=1
  fi
  if [ "$gpu_ok" = 1 ] && [ "$mem_ok" = 1 ]; then
    break
  fi
  echo "[auto-train] $(date) gpu_ok=$gpu_ok mem_max=$mem_max"
  sleep 30
done

echo "[auto-train] resources ready $(date)"
cd "$ROOT/repo"
source scripts/env.sh

tmux kill-session -t searchr1_retriever 2>/dev/null || true
# free port 8000 if an old uvicorn/java worker survived tmux shutdown
pkill -9 -f "search_r1_bm25.retrieval.server.*--port 8000" 2>/dev/null || true
sleep 2
tmux new-session -d -s searchr1_retriever "cd '$ROOT/repo' && source scripts/env.sh && PORT=8000 bash scripts/launch_retriever.sh 2>&1 | tee '$LOG_DIR/retriever_8000.log'"

echo "[auto-train] waiting for BM25 retriever"
for _ in $(seq 1 60); do
  if curl -fsS -m 20 -X POST http://127.0.0.1:8000/retrieve \
    -H "Content-Type: application/json" \
    -d '{"queries":["author of The Old Man and the Sea"],"topk":3,"return_scores":true}' \
    > "$LOG_DIR/retriever_8000_smoke.json"; then
    break
  fi
  sleep 5
done

echo "[auto-train] running 1-step training smoke $(date)"
bash scripts/train_verl_search_grpo.sh \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  data.val_batch_size=4 \
  2>&1 | tee "$LOG_DIR/train_smoke.log"

echo "[auto-train] smoke passed; starting formal 300-step training $(date)"
exec bash scripts/train_verl_search_grpo.sh 2>&1 | tee "$LOG_DIR/train_300.log"
