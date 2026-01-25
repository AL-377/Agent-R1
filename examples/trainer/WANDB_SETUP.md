# 训练日志工具配置说明

## 支持的日志工具

训练框架支持以下日志记录工具（可以同时使用多个）：

1. **console** - 控制台输出（默认，无需配置）
2. **wandb** - Weights & Biases（最常用）
3. **swanlab** - SwanLab（国产替代方案）

## 训练曲线查看位置

根据配置的日志工具，训练曲线可以在以下位置查看：
- **WandB**: https://wandb.ai/
- **SwanLab**: https://swanlab.cn/
- **Console**: 终端输出

### 当前脚本配置

在 `run_grpo_carecall_memory.sh` 中已经配置了：

```bash
trainer.logger=['console','wandb'] \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME
```

其中：
- `PROJECT_NAME='carecall-memory'` - WandB项目名称
- `EXPERIMENT_NAME=grpo-carecall-memory-qwen2.5-3b` - 实验名称

## WandB 配置步骤

### 1. 安装 WandB

```bash
pip install wandb
```

### 2. 登录 WandB

有两种方式：

#### 方式1：使用 API Key（推荐）

1. 访问 [https://wandb.ai/](https://wandb.ai/) 注册/登录账号
2. 进入 Settings -> API keys，复制你的 API key
3. 设置环境变量：

```bash
export WANDB_API_KEY="your_api_key_here"
```

或者在运行训练脚本前设置：

```bash
export WANDB_API_KEY="your_api_key_here"
bash examples/trainer/run_grpo_carecall_memory.sh
```

#### 方式2：使用命令行登录

```bash
wandb login
```

然后输入你的 API key。

### 3. 查看训练曲线

训练开始后，访问：
- **WandB 网页**: https://wandb.ai/your-username/carecall-memory
- 或者直接访问项目页面，实验名称是 `grpo-carecall-memory-qwen2.5-3b`

### 4. 本地运行模式（离线模式）

如果无法连接 WandB 服务器，可以使用离线模式：

```bash
export WANDB_MODE=offline
bash examples/trainer/run_grpo_carecall_memory.sh
```

离线数据会保存在 `wandb/` 目录下，之后可以同步：

```bash
wandb sync wandb/offline-run-xxxxx
```

## 记录的指标

训练过程中会记录以下指标到 WandB：

### 训练指标
- `actor/policy_loss` - 策略损失
- `actor/kl_loss` - KL散度损失
- `actor/entropy_loss` - 熵损失
- `actor/total_loss` - 总损失
- `actor/learning_rate` - 学习率
- `actor/grad_norm` - 梯度范数

### 数据指标
- `critic/rewards/mean` - 平均奖励
- `critic/rewards/max` - 最大奖励
- `critic/rewards/min` - 最小奖励
- `critic/advantages/mean` - 平均优势
- `critic/returns/mean` - 平均回报
- `turns/mean` - 平均工具调用轮数

### 验证指标
- `val/reward/mean` - 验证集平均奖励
- `val/score/mean` - 验证集平均分数

### 其他指标
- `train/epoch` - 当前epoch
- `train/step` - 当前训练步数
- `train/samples_per_second` - 训练速度

## SwanLab 配置（可选）

SwanLab 是一个国产的机器学习实验管理平台，可以作为 WandB 的替代方案。

### 1. 安装 SwanLab

```bash
pip install swanlab
```

### 2. 配置脚本

修改训练脚本，将 `wandb` 替换为 `swanlab`：

```bash
trainer.logger=['console','swanlab'] \
trainer.project_name=$PROJECT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME
```

### 3. 登录 SwanLab

```bash
swanlab login
```

或者设置环境变量：

```bash
export SWANLAB_API_KEY="your_api_key_here"
```

### 4. 查看训练曲线

访问：https://swanlab.cn/your-username/carecall-memory

## 同时使用多个日志工具

可以同时配置多个日志工具：

```bash
trainer.logger=['console','wandb','swanlab'] \
```

这样训练指标会同时记录到控制台、WandB 和 SwanLab。

## 禁用 WandB

如果不想使用 WandB，可以修改脚本：

```bash
trainer.logger=['console'] \  # 只使用console输出
# 或者
trainer.logger=['console','swanlab'] \  # 使用console和swanlab
```

或者设置环境变量：

```bash
export WANDB_DISABLED=true
bash examples/trainer/run_grpo_carecall_memory.sh
```

## 常见问题

### 1. WandB 连接失败

如果遇到连接问题，检查：
- 网络连接是否正常
- API key 是否正确
- 防火墙是否阻止了连接

可以使用离线模式：`export WANDB_MODE=offline`

### 2. 找不到实验

确保：
- 项目名称正确：`carecall-memory`
- 实验名称正确：`grpo-carecall-memory-qwen2.5-3b`
- 已登录正确的 WandB 账号

### 3. 权限问题

确保你的 WandB 账号有权限访问该项目。如果是团队项目，需要相应的权限。

## 示例

完整的训练启动命令（包含 WandB 配置）：

```bash
# 设置 WandB API Key
export WANDB_API_KEY="your_api_key_here"

# 运行训练
bash examples/trainer/run_grpo_carecall_memory.sh
```

训练开始后，在终端会看到类似输出：

```
wandb: Currently logged in as: your-username
wandb: Tracking run with wandb version x.x.x
wandb: Run data is saved locally in wandb/run-xxxxx
wandb: Syncing run grpo-carecall-memory-qwen2.5-3b to https://wandb.ai/...
```

然后在 WandB 网页上就可以看到实时的训练曲线了！

