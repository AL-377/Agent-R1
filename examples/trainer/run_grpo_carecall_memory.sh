#!/bin/bash
# Training script for CareCall Memory Model using GRPO with Qwen 2.5 3B

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'  # Qwen 2.5 3B model
export PROJECT_NAME='carecall-memory'
export EXPERIMENT_NAME=grpo-carecall-memory-qwen2.5-3b

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
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
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
    tool.max_tool_response_length=512 $@

