# Medical Dialogue Memory Model - 代码设计文档

## 1. 概述 (Overview)

本文档描述了基于Agent-R1框架的医疗对话记忆模型（Memory Model）的完整设计和实现。该模型旨在通过强化学习训练一个能够智能管理患者记忆信息的Agent，支持多层次的记忆存储和检索。

This document describes the complete design and implementation of a Medical Dialogue Memory Model based on the Agent-R1 framework. The model aims to train an Agent capable of intelligently managing patient memory information through reinforcement learning, supporting multi-layer memory storage and retrieval.

## 2. 系统架构 (System Architecture)

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Agent (LLM)                          │
│  Input: Dialogue Context + Memory State                 │
│  Output: Thought + Actions (insert/update/delete/wait)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Memory Environment                     │
│  - Manages tool execution                               │
│  - Provides memory state to agent                       │
│  - Handles RAG retrieval                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Memory Tools (Actions)                     │
│  - memory_insert: Add new memory                        │
│  - memory_update: Modify existing memory                │
│  - memory_delete: Remove memory                         │
│  - memory_wait: No operation                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Memory Manager (Core)                      │
│  - Multi-layer memory storage                           │
│  - RAG retrieval interface                              │
│  - FAISS-based vector search                            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 MemoryManager (`agent_r1/tool/memory_manager.py`)

**职责 (Responsibilities):**
- 管理多层记忆存储（工作记忆层、长期记忆层）
- 提供RAG检索接口
- 使用FAISS进行向量相似度搜索
- 支持记忆的增删改查操作

**关键功能 (Key Features):**
- `insert(layer, content, metadata)`: 插入新记忆
- `update(layer, memory_id, new_content)`: 更新记忆
- `delete(layer, memory_id)`: 删除记忆
- `retrieve(query, layers, top_k)`: RAG检索
- `get_memory_state()`: 获取当前记忆状态

**记忆层次 (Memory Layers):**
1. **working**: 工作记忆层 - 当前对话的关键信息
2. **identity**: 患者身份与基础信息层
3. **history**: 历史诊疗层
4. **experience**: 临床经验与案例层

#### 2.2.2 Memory Tools (`agent_r1/tool/tools/memory_tools.py`)

**工具定义 (Tool Definitions):**

1. **MemoryInsertTool**
   - 功能: 向指定记忆层插入新记忆条目
   - 参数: `layer`, `content`, `metadata` (可选)
   - 返回: 记忆ID和操作结果

2. **MemoryUpdateTool**
   - 功能: 更新现有记忆条目
   - 参数: `layer`, `memory_id`, `content`, `metadata` (可选)
   - 返回: 操作成功/失败状态

3. **MemoryDeleteTool**
   - 功能: 删除记忆条目
   - 参数: `layer`, `memory_id`
   - 返回: 操作成功/失败状态

4. **MemoryWaitTool**
   - 功能: 无操作（等待）
   - 参数: 无
   - 返回: 确认信息

#### 2.2.3 MemoryEnv (`agent_r1/tool/envs/memory.py`)

**职责 (Responsibilities):**
- 管理工具执行流程
- 在每个step向Agent提供记忆状态
- 处理工具调用的解析和响应格式化
- 支持批量工具执行

**关键方法 (Key Methods):**
- `step(raw_response)`: 执行单步记忆操作
- `batch_step(raw_responses)`: 批量执行
- `get_memory_state()`: 获取当前记忆状态
- `retrieve_memories(query, layers, top_k)`: RAG检索

#### 2.2.4 Verify Function (`agent_r1/src/reward_score/memory.py`)

**验证功能 (Verification Functions):**

1. **format_score**: 验证输出格式正确性
   - 检查是否有正确的结构标签
   - 验证思考过程（Thought）的存在
   - 验证工具调用格式

2. **operation_score**: 验证操作正确性
   - 与期望操作对比（如果有）
   - 基于对话上下文验证操作合理性
   - 验证操作参数的有效性（如memory_id是否存在）

3. **thought_score**: 验证思考质量
   - 检查思考内容的长度和相关性
   - 验证思考是否与操作相关

**评分函数 (Scoring Functions):**
- `compute_score(solution_str, ground_truth, extra_info)`: 综合评分
- `compute_score_format(solution_str)`: 格式评分
- `compute_score_operations(solution_str, ground_truth, extra_info)`: 操作评分

## 3. 数据流 (Data Flow)

### 3.1 训练数据格式

每个训练样本应包含：
```json
{
  "prompt": "当前对话上下文和记忆状态的提示",
  "data_source": "memory/medical_dialogue",
  "extra_info": {
    "dialogue_context": "当前对话内容",
    "memory_state": {
      "working": [...],
      "identity": [...],
      "history": [...],
      "experience": [...]
    },
    "expected_operations": [
      {
        "action": "memory_insert",
        "arguments": {...}
      }
    ]
  },
  "ground_truth": "期望的最终状态或验证信息"
}
```

### 3.2 Agent输入格式

在每个训练step，Agent接收：
```
Current Dialogue Context:
[对话内容]

Current Memory State:
[格式化的记忆状态]

[系统提示和操作说明]
```

### 3.3 Agent输出格式

Agent应输出：
```
<think>
[关于记忆操作的思考过程]
</think>

<tool_call>
{
  "name": "memory_insert",
  "arguments": {
    "layer": "working",
    "content": "..."
  }
}
</tool_call>
```

## 4. 训练流程 (Training Pipeline)

### 4.1 数据预处理

使用 `examples/data_preprocess/memory.py` 处理原始对话数据：

```bash
python examples/data_preprocess/memory.py \
    --input data/raw_dialogues.jsonl \
    --output data/memory/train.parquet
```

### 4.2 训练启动

#### PPO训练
```bash
bash examples/trainer/run_ppo_memory.sh
```

#### GRPO训练
```bash
bash examples/trainer/run_grpo_memory.sh
```

### 4.3 配置说明

主要配置参数：
- `data.reward_fn_key`: 设置为 `'memory/medical_dialogue'`
- `tool.tools`: `['memory_insert','memory_update','memory_delete','memory_wait']`
- `tool.env`: `'memory'`
- `data.max_prompt_length`: 建议4096（包含对话和记忆状态）
- `data.max_response_length`: 建议2048（包含思考和工具调用）

## 5. RAG检索接口 (RAG Retrieval Interface)

### 5.1 检索流程

1. **查询编码**: 使用FlagEmbedding模型将查询文本编码为向量
2. **向量搜索**: 在FAISS索引中搜索相似记忆
3. **结果排序**: 按相似度分数排序
4. **返回Top-K**: 返回最相关的K个记忆条目

### 5.2 使用示例

```python
from agent_r1.tool.memory_manager import MemoryManager

# 初始化
manager = MemoryManager()

# 插入记忆
memory_id = manager.insert(
    layer="working",
    content="患者主诉头痛3天",
    metadata={"symptom": "headache", "duration": "3 days"}
)

# 检索记忆
results = manager.retrieve(
    query="患者有什么症状？",
    layers=["working", "history"],
    top_k=5
)
```

## 6. 验证机制 (Verification Mechanism)

### 6.1 验证维度

1. **格式验证 (Format Verification)**
   - 检查输出结构完整性
   - 验证标签正确性
   - 验证JSON格式

2. **操作验证 (Operation Verification)**
   - 验证操作类型合理性
   - 验证参数有效性
   - 与期望操作对比

3. **思考验证 (Thought Verification)**
   - 验证思考内容质量
   - 验证思考与操作的相关性

### 6.2 评分机制

综合评分公式：
```
overall_score = 0.2 * format_score + 0.6 * operation_score + 0.2 * thought_score
```

## 7. 扩展性 (Extensibility)

### 7.1 添加新的记忆层

在 `MemoryLayer` enum中添加新层：
```python
class MemoryLayer(Enum):
    WORKING = "working"
    IDENTITY = "identity"
    HISTORY = "history"
    EXPERIENCE = "experience"
    NEW_LAYER = "new_layer"  # 新层
```

### 7.2 自定义验证规则

在 `verify_operations_correctness` 函数中添加自定义验证逻辑。

### 7.3 自定义嵌入模型

在初始化 `MemoryManager` 时指定不同的嵌入模型：
```python
manager = MemoryManager(
    embedding_model="your-model-name",
    index_type="IVF4096,Flat"
)
```

## 8. 文件结构 (File Structure)

```
agent_r1/
├── tool/
│   ├── memory_manager.py          # 核心记忆管理模块
│   ├── tools/
│   │   └── memory_tools.py         # 记忆操作工具
│   └── envs/
│       └── memory.py                # 记忆环境
├── src/
│   └── reward_score/
│       └── memory.py                # 验证函数
examples/
├── data_preprocess/
│   └── memory.py                    # 数据预处理脚本
└── trainer/
    ├── run_ppo_memory.sh            # PPO训练脚本
    └── run_grpo_memory.sh           # GRPO训练脚本
```

## 9. 使用示例 (Usage Examples)

### 9.1 基本使用

```python
from agent_r1.tool.memory_manager import MemoryManager
from agent_r1.tool.envs.memory import MemoryEnv
from agent_r1.tool.tools.memory_tools import MemoryInsertTool

# 初始化记忆管理器
manager = MemoryManager()

# 创建环境
tools = [MemoryInsertTool(memory_manager=manager)]
env = MemoryEnv(tools=tools, max_tool_response_length=512)

# 获取记忆状态
memory_state = env.get_memory_state()

# 执行操作
response = "<tool_call>{\"name\":\"memory_insert\",\"arguments\":{\"layer\":\"working\",\"content\":\"test\"}}</tool_call>"
tool_response, success, active = env.step(response)
```

### 9.2 训练数据准备

```python
from examples.data_preprocess.memory import process_memory_sample

sample = process_memory_sample(
    dialogue_context="医生：您有什么症状？\n患者：我头痛3天了。",
    memory_state={
        "working": [],
        "identity": [{"id": "id1", "content": "患者：张三，45岁"}]
    },
    expected_operations=[
        {
            "action": "memory_insert",
            "arguments": {
                "layer": "working",
                "content": "患者主诉头痛3天"
            }
        }
    ]
)
```

## 10. 注意事项 (Important Notes)

1. **内存管理**: FAISS索引会占用内存，注意控制记忆条目数量
2. **嵌入模型**: 首次加载嵌入模型需要时间，建议使用GPU加速
3. **索引类型**: 对于大规模记忆，建议使用IVF索引而非Flat索引
4. **批量操作**: 支持批量工具执行以提高训练效率
5. **验证数据**: 确保验证函数能够访问到必要的上下文信息（通过extra_info）

## 11. 未来改进 (Future Improvements)

1. **增量索引更新**: 优化索引更新机制，避免全量重建
2. **记忆压缩**: 实现记忆压缩和去重机制
3. **多模态支持**: 支持图像、表格等多模态记忆
4. **记忆关联**: 实现记忆之间的关联和推理
5. **自适应评分**: 根据任务特点自适应调整评分权重

## 12. 参考文献 (References)

- Agent-R1 Framework: https://github.com/0russwest0/Agent-R1
- FAISS: https://github.com/facebookresearch/faiss
- FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding

