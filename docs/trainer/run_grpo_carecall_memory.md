### `run_grpo_carecall_memory.sh` 参数说明

本文件解释 `examples/trainer/run_grpo_carecall_memory.sh` 中各部分配置的含义，方便查阅和二次修改。

---

## 1. Ray 环境变量与清理

脚本开头设置了一组 Ray 相关的环境变量，用来减少噪声日志、避免 dashboard 问题，并放宽 worker 注册超时：

```bash
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_DASHBOARD=1
export RAY_ENABLE_METRICS_COLLECTION=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_BACKEND_LOG_LEVEL=error
export RAY_WORKER_REGISTER_TIMEOUT_S=300
```

- **RAY_DISABLE_IMPORT_WARNING**: 关闭 Ray 的 import 警告，减少日志噪声。  
- **RAY_DEDUP_LOGS**: 不去重日志，方便调试时查看完整输出。  
- **RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO**: 禁止 Ray 自动覆盖某些加速相关环境变量。  
- **RAY_DISABLE_DASHBOARD / RAY_ENABLE_METRICS_COLLECTION**: 关闭 Dashboard 和 Metrics，减少额外进程与开销。  
- **RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE**: 允许 object store 使用较慢磁盘，降低对存储速度的要求。  
- **RAY_BACKEND_LOG_LEVEL**: Ray 后端仅输出 error 级别日志。  
- **RAY_WORKER_REGISTER_TIMEOUT_S**: 将 worker 注册超时时间设置为 300 秒，适应大模型初始化较慢的情况。

在真正启动训练前，会先清理可能存在的旧 Ray 集群：

```bash
ray stop --force 2>/dev/null || true
sleep 2
```

- **ray stop --force**: 强制停止已有 Ray 集群，避免“脏环境”干扰当前实验。  
- **sleep 2**: 稍作等待，保证旧进程完全退出。

---

## 2. 模型路径与实验信息

脚本会优先尝试使用本地缓存好的 Qwen2.5-3B 模型，否则回退到 HuggingFace Hub：

```bash
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
export SWANLAB_API_KEY=...
```

- **BASE_MODEL**:  
  - 若环境变量 `BASE_MODEL` 未设置，则：  
    - 如果本地目录 `LOCAL_QWEN_MODEL` 存在，使用该本地路径；  
    - 否则使用 HuggingFace 模型名 `Qwen/Qwen2.5-3B-Instruct`。  
  - 最终会作为 `actor_rollout_ref.model.path`，决定策略 / rollout / ref 使用的 HF 权重路径。
- **PROJECT_NAME**: 训练日志中的项目名，对应 `trainer.project_name`，用于 SwanLab/WandB 等追踪系统。  
- **EXPERIMENT_NAME**: 当前实验名称，对应 `trainer.experiment_name`，用来区分不同实验。  
- **SWANLAB_API_KEY**: SwanLab 日志后端所需的 API Key。

---

## 3. 记忆检索与 HF 缓存环境

脚本为记忆检索和 HuggingFace 缓存设置了一些全局环境变量：

```bash
export MEMORY_EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
export MEMORY_EMBEDDING_LOCAL_ONLY=1
export HF_HOME=/home/tiger/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

- **MEMORY_EMBEDDING_MODEL**:  
  - 被 `MemoryEnv` / `MemoryManager` 读取，用作 patient memory RAG 的向量检索模型（例如 `BAAI/bge-large-en-v1.5`）。
- **MEMORY_EMBEDDING_LOCAL_ONLY**:  
  - 项目内部约定，表示检索模型只从本地加载，不访问远程（具体逻辑在 `MemoryManager` 中实现）。
- **HF_HOME**: HuggingFace 缓存的根目录，模型和 tokenizer 均从此路径读取。  
- **HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE**: 强制 Transformers / HF Hub 以离线模式运行，只从本地缓存加载，不访问网络。

---

## 4. 训练配置（Hydra 覆盖项）

核心启动命令如下，其中每一项都是对 `agent_trainer.yaml` 的覆盖：

```bash
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
    ray_init.num_cpus=32 $@
```

下面按模块解释各字段的含义。

### 4.1 数据相关 `data.*`

- **`data.train_files` / `data.val_files`**:  
  - 训练和验证数据的 parquet 文件列表，传给 `ToolRLDataset` 加载样本。  
  - 本脚本中两者使用同一份文件 `examples/dataset/carecall/carecall_train_v1.parquet`。

- **`data.train_batch_size=64`**:  
  - driver 视角的逻辑 batch 大小；  
  - 实际训练 batch 大小为 `train_batch_size × rollout.n_repeat`。

- **`data.max_prompt_length=8192`**:  
  - prompt 最大 token 长度，`ToolRLDataset` 在构造 `input_ids` / `raw_prompt_ids` 时会按此值裁剪。  
  - `ToolGenerationManager` 也会根据这个长度限制输入。

- **`data.max_response_length=8192`**:  
  - 整个 response（包括多轮工具交互结果）允许的最大 token 数。

- **`data.max_response_length_single_turn=8192`**:  
  - 单轮生成的最大 response 长度，传到底层 rollout（vLLM）作为 `response_length` 限制。

- **`data.use_default_tool_template=False`**:  
  - 若为 True，`ToolRLDataset` 会在 chat template 中自动注入工具 schema；  
  - 此处为 False，说明工具提示由训练数据本身控制，而非默认模板。

- **`data.reward_fn_key='memory/medical_dialogue'`**:  
  - 传给 `AgentRewardManager(reward_fn_key=...)`；  
  - `AgentRewardManager` 内部会访问 `data_item.non_tensor_batch[reward_fn_key]` 以决定用哪种打分逻辑。  
  - 当前值为 `'memory/medical_dialogue'`（实际运算中会按这个 key 去取非张量数据）。

### 4.2 策略模型与 FSDP `actor_rollout_ref.*`

- **`actor_rollout_ref.model.path=$BASE_MODEL`**:  
  - 策略、rollout 和 reference policy 使用的 HF 模型路径（Qwen2.5-3B 指令版），由前面的 `BASE_MODEL` 逻辑决定本地/远程。

- **`actor_rollout_ref.actor.optim.lr=1e-6`**:  
  - 策略网络的学习率。

- **`actor_rollout_ref.model.use_remove_padding=True`**:  
  - 开启“移除 padding”优化，在 FSDP / sequence parallel 模式下可提升效率，并满足部分并行配置的约束。

- **`actor_rollout_ref.actor.ppo_mini_batch_size=32`**:  
  - PPO 内部更新策略时的 mini-batch 大小。

- **`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`**:  
  - 每张 GPU 上的 micro batch size，用于控制单次前向的样本数，避免显存爆炸。

- **`actor_rollout_ref.model.enable_gradient_checkpointing=True`**:  
  - 启用梯度检查点，以减少显存占用。

- **`actor_rollout_ref.actor.fsdp_config.param_offload=False` / `optimizer_offload=False`**:  
  - 不将参数和优化器状态 offload 到 CPU/磁盘，简化部署，代价是占用更多显存。

- **`actor_rollout_ref.actor.use_kl_loss=True`**:  
  - 将 KL loss（策略 vs reference policy）纳入损失函数，用于稳定训练。

- **`actor_rollout_ref.actor.kl_loss_coef=0.001`**:  
  - KL loss 的权重系数。

### 4.3 rollout / vLLM / reference policy

- **`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`**:  
  - 计算 log_prob 时，每张 GPU 上的 micro batch size，防止一次性处理过多样本导致 OOM。

- **`actor_rollout_ref.rollout.tensor_model_parallel_size=1`**:  
  - vLLM 的 tensor parallel 大小为 1，不进行 tensor 维度切分（仅 data parallel）。

- **`actor_rollout_ref.rollout.name=vllm`**:  
  - 指定 rollout 引擎为 vLLM，用于推理生成和 log_prob 计算。

- **`actor_rollout_ref.rollout.stop_token_ids=[]`**:  
  - 不额外指定停止 token id，依赖 tokenizer 内置 eos 来停止生成。

- **`actor_rollout_ref.rollout.gpu_memory_utilization=0.6`**:  
  - vLLM 估算用于模型和 KV cache 的显存上限约为 60%，留下部分余量防止溢出。

- **`actor_rollout_ref.rollout.n_repeat=5`**:  
  - GRPO 中每个 prompt 采样的 response 数量，利于进行 group-based advantage 估计。

- **`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`**:  
  - reference policy 计算 log_prob 时，每卡 micro batch size。

- **`actor_rollout_ref.ref.fsdp_config.param_offload=True`**:  
  - 允许 reference policy 的参数 offload 出 GPU，以节省显存，因为 ref 仅用于 KL 约束。

### 4.4 RL 算法 `algorithm.*`

- **`algorithm.adv_estimator=grpo`**:  
  - 使用 GRPO 优势估计方式（基于 outcome reward 的 group 对比优势），对应 `compute_grpo_outcome_advantage`。

- **`algorithm.kl_ctrl.kl_coef=0.001`**:  
  - KL controller 的系数，在 fixed 模式下即为固定的 KL 权重。

- **`algorithm.use_kl_in_reward=False`**:  
  - 不在 reward 值中加 KL 惩罚，仅通过 KL loss 对策略进行约束。

### 4.5 训练控制与日志 `trainer.*`

- **`trainer.logger=['console','swanlab']`**:  
  - 使用控制台和 SwanLab 双重日志后端。

- **`trainer.project_name=$PROJECT_NAME` / `trainer.experiment_name=$EXPERIMENT_NAME`**:  
  - 在日志系统中的项目名和实验名，来自脚本顶部的环境变量。

- **`trainer.n_gpus_per_node=8` / `trainer.nnodes=1`**:  
  - 每个节点使用 8 块 GPU，单节点训练，总资源为 8 GPU。

- **`trainer.save_freq=-1`**:  
  - 不按 step 周期性保存 checkpoint，只依赖其他逻辑（本脚本中不会自动保存）。

- **`trainer.test_freq=10`**:  
  - 每 10 个 global step 进行一次验证（调用 `_validate()`）。

- **`trainer.total_epochs=10`**:  
  - 总训练轮数，结合 dataloader 长度确定 `total_training_steps`。

- **`trainer.val_before_train=True`**:  
  - 在正式训练前先跑一次验证，记录初始指标。

- **`trainer.log_val_generations=0`**:  
  - 不上传验证集上的生成样本，仅记录度量指标。

### 4.6 工具环境 `tool.*`

- **`tool.max_turns=1`**:  
  - `ToolGenerationManager` 中每个样本最多进行 1 轮 “模型生成 + 工具调用 + 拼接” 的循环。

- **`tool.tools=['memory_insert','memory_update','memory_delete','memory_wait']`**:  
  - 启用的工具列表，由 `_default_tool(name)` 解析为：
    - `MemoryInsertTool`、`MemoryUpdateTool`、`MemoryDeleteTool`、`MemoryWaitTool`。

- **`tool.env=memory`**:  
  - 工具环境类型为 `memory`，对应 `MemoryEnv`，负责：  
  - 解析 `<tool_call>...</tool_call>`；调用 `MemoryManager` 执行插入/更新/删除/等待；  
  - 构造 `<tool_response>...</tool_response>` 注入下一轮 prompt。

- **`tool.max_tool_response_length=512`**:  
  - 单次工具响应文本的最大长度，超出部分会被截断，避免 tool 输出过长污染后续生成。

### 4.7 Ray 初始化 `ray_init.*` 与额外参数

- **`ray_init.num_cpus=32`**:  
  - 传给 `ray.init(num_cpus=...)`，限制 Ray 可使用的 CPU 数量为 32，避免占满整机。

- **`$@`**:  
  - 将脚本外部追加的参数继续转发给 Hydra，允许在命令行进一步覆盖配置，例如：
  - `bash examples/trainer/run_grpo_carecall_memory.sh trainer.n_gpus_per_node=4 algorithm.adv_estimator=gae`

---

以上即为 `run_grpo_carecall_memory.sh` 中各项参数的说明，可在修改脚本或排查训练问题时作为参考。


