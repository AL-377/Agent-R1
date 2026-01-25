#!/bin/bash

# Fix Ray issues
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Disable dashboard/metrics to avoid dashboard_agent crash on some envs
export RAY_DISABLE_DASHBOARD=1
export RAY_ENABLE_METRICS_COLLECTION=0
# Fix worker registration issues
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_BACKEND_LOG_LEVEL=error
export RAY_WORKER_REGISTER_TIMEOUT_S=300

# Clean up any existing Ray sessions before starting
ray stop --force 2>/dev/null || true
sleep 2

if [ -z "$BASE_MODEL" ]; then
  LOCAL_QWEN_MODEL="/home/tiger/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
  if [ -d "$LOCAL_QWEN_MODEL" ]; then
    BASE_MODEL="$LOCAL_QWEN_MODEL"
  else
    BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
  fi
fi
export PROJECT_NAME='carecall-memory'
export EXPERIMENT_NAME=grpo-carecall-memory-qwen2.5-3b
export SWANLAB_API_KEY=IQLCKdLdPp6ZTpRFBqRgM

# Local model
export MEMORY_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
export MEMORY_EMBEDDING_LOCAL_ONLY=1
export HF_HOME=/home/tiger/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m agent_r1.src.main_agent \
    data.train_files=['examples/dataset/carecall/carecall_train_v1.parquet'] \
    data.val_files=['examples/dataset/carecall/carecall_train_v1.parquet'] \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=8192 \
    data.use_default_tool_template=False \
    data.reward_fn_key='memory/medical_dialogue' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    tool.max_turns=1 \
    tool.tools=['memory_insert','memory_update','memory_delete','memory_wait'] \
    tool.env=memory \
    tool.max_tool_response_length=512 \
    ray_init.num_cpus=32 $@  # Limit

