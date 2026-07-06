#!/usr/bin/env bash
set -euo pipefail

PROJECT_HOME="${PROJECT_HOME:-/root/autodl-tmp/search-r1-bm25}"
ENV_NAME="${ENV_NAME:-base}"

mkdir -p "$PROJECT_HOME"/{repo,data,models,cache,hf,checkpoints,logs,wheels,tmp}

source /root/miniconda3/etc/profile.d/conda.sh
if [ "$ENV_NAME" != "base" ] && ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

export TMPDIR="$PROJECT_HOME/tmp"
export PIP_CACHE_DIR="$PROJECT_HOME/cache/pip"
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple
python -m pip install -U pip setuptools wheel

python -m pip install -e "$PROJECT_HOME/repo[dev]"
python -m pip install pyjnius==1.7.0 faiss-cpu onnxruntime openai tiktoken Cython scikit-learn scipy sentencepiece flask ir_datasets
python -m pip install --no-deps pyserini==0.24.0

python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
except Exception as exc:
    print("torch not checked/installed by this script:", repr(exc))
PY
