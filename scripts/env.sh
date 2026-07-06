#!/usr/bin/env bash
set -euo pipefail

export SEARCH_R1_HOME="${SEARCH_R1_HOME:-/root/autodl-tmp/search-r1-bm25}"
export SEARCH_R1_REPO="${SEARCH_R1_REPO:-$SEARCH_R1_HOME/repo}"
export SEARCH_R1_DATA="${SEARCH_R1_DATA:-$SEARCH_R1_HOME/data}"
export SEARCH_R1_MODELS="${SEARCH_R1_MODELS:-$SEARCH_R1_HOME/models}"
export SEARCH_R1_CACHE="${SEARCH_R1_CACHE:-$SEARCH_R1_HOME/cache}"

export TMPDIR="$SEARCH_R1_HOME/tmp"
export PIP_CACHE_DIR="$SEARCH_R1_CACHE/pip"
export HF_HOME="$SEARCH_R1_HOME/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-not-used-for-bm25}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$SEARCH_R1_DATA" "$SEARCH_R1_MODELS"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${SEARCH_R1_CONDA_ENV:-base}"
