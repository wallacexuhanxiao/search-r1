#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/search-r1-bm25}
PYTHON=${PYTHON:-/root/vllm_env/bin/python}
ENGINE=${ENGINE:-vllm}

export PATH="/root/miniconda3/bin:$PATH"
export HF_HOME=${HF_HOME:-$ROOT/hf_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$ROOT/hf_home/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$ROOT/hf_home/datasets}
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen2.5-3B-Instruct}
TRAIN_DATA=${TRAIN_DATA:-$ROOT/data/verl/train.parquet}
VAL_DATA=${VAL_DATA:-$ROOT/data/verl/validation.parquet}
RETRIEVE_URL=${RETRIEVE_URL:-http://127.0.0.1:8000/retrieve}

cd "$ROOT/third_party/verl-agent"

"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=4 \
  data.val_batch_size=64 \
  data.max_prompt_length=2048 \
  data.max_response_length=384 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=False \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  env.env_name=search \
  env.seed=42 \
  env.max_steps=3 \
  env.rollout.n=4 \
  env.history_length=3 \
  env.search.search_url="$RETRIEVE_URL" \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=search_r1_bm25 \
  trainer.experiment_name=qwen2_5_3b_lora_grpo_bm25 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_training_steps=300 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  "$@"