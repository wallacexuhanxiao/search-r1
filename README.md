# Search-R1 BM25

This repo implements the fixed BM25-only Search-R1 scaffold:

- Qwen/Qwen2.5-3B-Instruct
- LoRA-GRPO through `verl-agent`
- local Wikipedia BM25 retrieval through Pyserini/Lucene
- at most two model-generated searches
- final-answer-only 0/1 reward

## AutoDL Layout

Runtime environment used in the smoke test and 50-step run:

```bash
/root/vllm_env/bin/python
```

Data, model cache, code, logs, checkpoints:

```bash
/root/autodl-tmp/search-r1-bm25
```

## Setup

```bash
cd /root/autodl-tmp/search-r1-bm25/repo
bash scripts/install_java.sh
PYTHON=/root/vllm_env/bin/python python -m pip install -e ".[dev]"
PYTHON=/root/vllm_env/bin/python python -m pip install pyserini fastapi uvicorn peft hydra-core tensordict codetiming wandb
```

Do not reinstall torch, CUDA, or vLLM when the image already provides them. The
successful run used torch 2.11, CUDA 12.9/13 runtime support, and vLLM 0.20.0
from the prebuilt image environment.

If using the current `verl-agent` checkout with vLLM 0.20.0, apply:

```bash
cd /root/autodl-tmp/search-r1-bm25/third_party/verl-agent
git apply /root/autodl-tmp/search-r1-bm25/repo/patches/verl-agent-vllm020-torch211.patch
```

## Build BM25

```bash
source scripts/env.sh
python -m search_r1_bm25.data_prepare \
  --input /root/autodl-tmp/search-r1-bm25/data/wiki/articles.jsonl \
  --output /root/autodl-tmp/search-r1-bm25/data/passages/wiki2018_passages.jsonl

bash scripts/build_index.sh
bash scripts/launch_retriever.sh
```

Smoke test:

```bash
python -m search_r1_bm25.retrieval.client "author of The Old Man and the Sea"
```

## Train

Start BM25 first, then run the smoke-and-train helper:

```bash
tmux new-session -d -s searchr1_retriever "cd /root/autodl-tmp/search-r1-bm25/repo && bash scripts/launch_retriever.sh"
tmux new-session -d -s searchr1_train_300 "cd /root/autodl-tmp/search-r1-bm25/repo && bash scripts/wait_gpu_smoke_train.sh"
```

The main training script now defaults to the v2 stability settings:

- `Qwen2.5-3B-Instruct`
- LoRA rank 16 / alpha 32
- vLLM rollout
- GRPO group size 4 through `env.rollout.n=4`
- `data.max_prompt_length=2048`, `data.max_response_length=384`
- `actor_rollout_ref.actor.optim.lr=5e-6`
- `env.max_steps=3`, `env.history_length=3` to better enforce at most two searches before answer/termination
- `actor_rollout_ref.actor.use_invalid_action_penalty=False` for final-answer-only reward
- `save_freq=50`, `test_freq=50`, `total_training_steps=300`

The first successful v1 run saved a resumable checkpoint at:

```bash
/root/autodl-tmp/search-r1-bm25/third_party/verl-agent/checkpoints/search_r1_bm25/qwen2_5_3b_lora_grpo_bm25/global_step_50
```

## Core Rules

The parser accepts one action per model continuation:

- `<search>query</search>`
- `<answer>final answer</answer>`

Invalid trajectories receive reward 0. `<information>...</information>` is environment text and must be masked out by the rollout/training integration.

## Notes from the v1 50-step run

The initial v1 configuration used `lr=1e-5`, `max_prompt_length=1536`, `max_response_length=512`, `env.max_steps=4`, and `test_freq=25`. At step 50, validation scores dropped while tool-call counts increased. The v2 settings reduce policy-update aggressiveness, reduce response length, increase prompt budget, and cap the environment loop more tightly.
