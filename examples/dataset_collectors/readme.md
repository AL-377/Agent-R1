# 医疗对话记忆评估数据集 — Examiner Agent 处理系统

## 一、系统概览

本系统将五个开源中文医疗对话数据集统一处理为 **"同一患者多次问诊"** 的格式，与 CareCall 第一版数据集完全兼容。核心流程：

```
原始数据集
  ↓  load_raw_data()        — 加载原始数据
  ↓  normalize_dialogue()   — 归一化为 {role, content} 格式
  ↓  grouping_key_fn()      — 按分组键聚类
  ↓  multi_session_synth    — 合成同一患者多次就诊 + 插入"---诊疗分割线---"
  ↓  process_single_dialogue()
  ↓    ├─ Stage 1: LLM提取to_memory → 生成memory操作 → 构建oracle_memory_base
  ↓    └─ Stage 2: Examiner出题（Level 1-4）→ 生成memory_query
  ↓
输出: patient_{id}.json  （与CareCall格式一致）
```

### 输出格式

每个患者文件包含：

```json
{
  "patient_id": "xxx",
  "metadata": {
    "scenario": "chronic_followup",
    "scenario_name": "慢性病随访管理",
    "session_count": 3
  },
  "messages": [
    {"user":      {"content": "...", "to_memory": [...], "oracle_memory_base": {...}, "memory_query": {...}}},
    {"assistant": {"content": "...", "to_memory": [...], "oracle_memory_base": {...}}},
    {"assistant": {"content": "---诊疗分割线---", "to_memory": [null], "oracle_memory_base": {...}}},
    {"user":      {"content": "第二次就诊...", ...}}
  ]
}
```

### 五大临床场景覆盖

每个合成出的患者会被分配到以下场景之一（由 `clinical_scenarios.py` 根据科室/疾病自动匹配）：


| 场景      | 对应记忆层次    | 时间跨度   | 核心挑战           |
| ------- | --------- | ------ | -------------- |
| 慢性病随访管理 | 身份记忆+诊疗过程 | 6-12个月 | 用药调整追踪、指标趋势记忆  |
| 复杂疾病诊疗  | 诊疗过程+临床经验 | 3-6个月  | 检查结果记忆、诊断推理链   |
| 多疾病共病管理 | 身份记忆+冲突检测 | 6-12个月 | 药物冲突检测、跨科室信息整合 |
| 术后康复管理  | 过程记忆+实时交互 | 1-3个月  | 术后指标追踪、康复方案连续性 |
| 罕见病鉴别诊断 | 医学知识+逻辑推理 | 1-2个月  | 鉴别诊断记忆、排除法推理   |


### 四级评测能力（Examiner Agent Level 1-4）

每个 Level 使用不同的 **上下文窗口**、**记忆采样策略** 和 **出题 prompt**，以匹配不同难度对记忆时间跨度的要求。

| Level   | 考察能力 | 采样逻辑 | 上下文窗口 | 时间要求 | 示例 |
| ------- | ---- | --- | --- | --- | --- |
| Level 1 | 状态追踪 | working 层，偏好最近插入的记忆 | 最近6轮 | 无（可以刚说的） | "患者今天反馈头晕持续了多久？" |
| Level 2 | 精准检索 | identity/history 层，偏好旧记忆 | 最近4轮 | 信息≥6轮前记录 | "患者首次就诊时提到的过敏药物是什么？" |
| Level 3 | 长程理解 | history + working 跨层，一旧一新 | 最近8轮 | 旧信息≥10轮前 | "对比首次就诊和本次的血压，是否有改善？" |
| Level 4 | 冲突检查 | working + identity/history 跨层 | 最近6轮 | 信息≥4轮前 | "患者报告过青霉素过敏，但本次处方含阿莫西林？" |

**关键设计原则：**

1. **基于诊疗记录事实，而非医学知识**：所有题目的答案必须来自对话中明确提到过的事实，禁止出需要外部医学知识才能回答的题目
2. **半衰期（Half-Life）机制**：每个记忆项记录其被创建时的 `turn_idx`，采样时计算"信息年龄"（当前轮次 - 创建轮次），Level 2/3/4 要求信息足够"久远"后才会被触发
3. **跨层采样**：Level 3/4 要求采样的记忆点来自不同的记忆层（如一个来自 history，一个来自 working），确保题目需要跨时间/跨层关联
4. **加权级别选择**：当多个 Level 都满足条件时，优先出高 Level 题（权重 L1:1.0, L2:1.5, L3:2.0, L4:2.5）

输出中每道题包含 `memory_age_turns` 字段，记录答案来源记忆点与出题时刻的距离（轮次差）。


---

## 二、各数据集处理逻辑详解

---

### 1. MedDialog-CN（`meddialog_cn_examiner.py`）

**数据概况**：UCSD/haodf.com，340万对话，1130万轮次，涵盖172个科室

**记忆聚焦**：身份与病史记忆

**原始数据形态**：

- 每条记录是一次 **独立的** 单次问诊对话
- 不同记录之间的患者无关联
- 格式：`{"utterances": ["患者：...", "医生：..."], "department": "内分泌科"}`

**多次问诊合成方式**：

```
原始: [对话A(内分泌科), 对话B(内分泌科), 对话C(内分泌科), 对话D(心血管), ...]
                    ↓ grouping_key_fn: 按科室分组
分组: {内分泌科: [A, B, C, ...], 心血管: [D, ...], ...}
                    ↓ multi_session_synthesizer: 每2~5条合并为一个患者
合成: 患者_001 = [对话A ---分割线--- 对话B ---分割线--- 对话C]  (场景:慢性病随访)
      患者_002 = [对话D ---分割线--- 对话E]                    (场景:复杂疾病诊疗)
```

**分组键**：`metadata["department"]`（科室名），无科室时从对话首句提取科室关键词

**合成策略说明**：同一科室的对话在医学主题上是相关的（如内分泌科的对话大多涉及糖尿病/甲亢等），将其合并为同一患者的多次随访在医学逻辑上是合理的。

---

### 2. CMtMedQA（`cmtmedqa_examiner.py`）

**数据概况**：仲景团队，7万条真实医患多轮对话

**记忆聚焦**：主动问诊与状态追踪

**原始数据形态**：

- 大部分是 **单轮Q&A**（一个问题 + 一个回答），不是多轮对话
- 格式：`{"question": "我最近头痛...", "answer": "根据您的描述...建议...注意..."}`

**多次问诊合成方式（两步）**：

```
步骤1 — 长回答拆分为伪多轮:
  原始: question="头痛怎么办"  answer="1.建议做CT 2.注意休息 3.可以服用..."
  拆分: [patient:"头痛怎么办", doctor:"建议做CT", patient:"注意休息", doctor:"可以服用..."]
  方法: _split_answer_into_turns() 按编号(1.2.3.)、段落、换行拆分

步骤2 — 按科室/分类分组合成多次问诊:
  原始: [Q&A_A(心血管), Q&A_B(心血管), Q&A_C(心血管), ...]
                    ↓ grouping_key_fn: 按department > category > title关键词
  合成: 患者_001 = [拆分后的A ---分割线--- 拆分后的B ---分割线--- 拆分后的C]
```

**分组键**：`department` > `category` > `title` > 对话首句疾病关键词

**合成策略说明**：先将单条长回答拆分为多轮对话（模拟医生问诊过程），再将同一科室/分类的多条Q&A合并为同一患者在不同时间点的问诊。这样一条原始Q&A变成一次"就诊"，多条同类Q&A合并为一个患者的"多次就诊"。

---

### 3. IMCS-21 / KaMed（`imcs21_examiner.py`）

**数据概况**：复旦大学，6万多个会话，具备实体识别、对话行为分类等细粒度标注

**记忆聚焦**：临床路径记忆

**原始数据形态**：

- 每条记录是一次 **完整的多轮医患对话**，并附带诊断(diagnosis)、报告(report)、实体标注(entities)
- 不同记录之间的患者无关联
- 格式：`{"dialogue": [{"role":"D", "content":"..."}], "diagnosis": "上呼吸道感染", "report": "..."}`

**多次问诊合成方式**：

```
原始: [对话A(诊断:糖尿病), 对话B(诊断:糖尿病), 对话C(诊断:糖尿病), ...]
                    ↓ grouping_key_fn: 按诊断分组
分组: {糖尿病: [A, B, C, ...], 上呼吸道感染: [D, E, ...], ...}
                    ↓ multi_session_synthesizer: 每2~4条合并为一个患者
合成: 患者_001 = [对话A ---分割线--- 对话B ---分割线--- 对话C]
```

**分组键**：`metadata["diagnosis"]` > `disease` > `department`

**合成策略说明**：同一诊断的对话代表同一类疾病的不同就诊过程，合并为同一患者多次就诊。由于IMCS-21有丰富的实体标注和诊疗报告，这些信息被保留在metadata中，增强了记忆提取的上下文。

**特殊格式支持**：

- dict-of-dicts 格式（`{dialogue_id: {dialogue_data}}`）
- `"D"`/`"P"` 单字母角色标识
- 纯文本对话字符串解析（`_parse_dialogue_string()`）

---

### 4. Huatuo-26M（`huatuo26m_examiner.py`）

**数据概况**：华佗团队，2600万个问答对，涵盖在线咨询、百科、知识库等多源

**记忆聚焦**：医学知识记忆（经验层/知识层）

**原始数据形态**：

- **纯单轮Q&A对**，连多轮对话都不是
- 格式：`{"question": "糖尿病能吃什么水果", "answer": "...", "source": "encyclopedia"}`

**多次问诊合成方式（三步，最复杂）**：

```
步骤1 — 按类别聚合为伪会话（load_raw_data阶段）:
  原始: [QA_1(糖尿病), QA_2(糖尿病), QA_3(糖尿病), QA_4(糖尿病), QA_5(糖尿病), ...]
                    ↓ _group_qa_pairs(): 每group_size=5条合并为一个伪会话
  伪会话: session_A = {qa_pairs: [QA_1~5], category: "糖尿病"}
          session_B = {qa_pairs: [QA_6~10], category: "糖尿病"}

步骤2 — normalize_dialogue: 每个伪会话转为多轮对话:
  session_A → [patient:Q1, doctor:A1, patient:Q2, doctor:A2, ..., patient:Q5, doctor:A5]

步骤3 — multi_session_synthesizer: 按category将多个伪会话合并为同一患者多次就诊:
  合成: 患者_001 = [session_A ---分割线--- session_B ---分割线--- session_C]
```

**分组键**：`metadata["category"]` / `source` / `type`

**合成策略说明**：

- 第一步（`_group_qa_pairs`）：同类别的Q&A对按`group_size`条一组，组成一次"伪问诊"
- 第二步（`normalize_dialogue`）：将伪问诊展开为交替的patient→doctor轮次
- 第三步（`_synthesize_multi_session`）：将多个同类别伪问诊合并为同一患者的多次就诊

这是所有数据集中合成路径最长的——从最原子的单轮Q&A开始，经过三层合成，最终变成多次就诊。

---

### 5. LCMDC（`lcmdc_examiner.py`）

**数据概况**：DUTIR-BioNLP，分诊43万 + 诊断20万 + 咨询47万，三个子集

**记忆聚焦**：长程诊疗闭环

**原始数据形态**：

- **唯一天然具备多会话结构的数据集**
- 三个子集（triage分诊 / diagnosis诊断 / consultation问诊），部分记录共享 `patient_id`
- 同一 `patient_id` 可能在分诊、诊断、问诊三个阶段都有记录

**多次问诊合成方式（两层策略）**：

```
第一层 — 天然串联（_chain_subsets）:
  triage子集:       {patient_id: "P001", dialogue: [...], triage_result: "内科"}
  diagnosis子集:    {patient_id: "P001", dialogue: [...], diagnosis: "2型糖尿病"}
  consultation子集: {patient_id: "P001", dialogue: [...], prescription: "二甲双胍"}
                    ↓ 按patient_id匹配，按 triage→diagnosis→consultation 排序
  患者P001 = [分诊对话 ---分割线--- 诊断对话 ---分割线--- 问诊对话]

第二层 — 退化分组（grouping_key_fn + multi_session_synthesizer）:
  无法匹配patient_id的单独记录 → 按department分组 → 合成为多次就诊
  已串联的患者 → 分配唯一key → 不再被进一步合并
```

**分组键**：

- 已串联记录：`_chained_{dialogue_id}`（唯一键，防止二次合并）
- 未串联记录：`department` > `diagnosis`

**合成策略说明**：

- **优先使用真实的跨子集关联**：如果同一`patient_id`同时出现在分诊、诊断、问诊子集中，就直接串联——这是真正的"同一患者多阶段诊疗"
- **兜底使用分组合成**：对于无法匹配到跨子集链的单独记录，退化为按科室分组
- **子集排序**：固定顺序 triage(0) → diagnosis(1) → consultation(2)，模拟真实诊疗流程

---

## 三、数据集对比总结


| 维度                      | MedDialog-CN   | CMtMedQA | IMCS-21 | Huatuo-26M | LCMDC         |
| ----------------------- | -------------- | -------- | ------- | ---------- | ------------- |
| **原始形态**                | 独立多轮对话         | 单轮Q&A    | 带标注多轮对话 | 纯单轮Q&A     | 三子集(部分共享ID)   |
| **天然多会话**               | ❌              | ❌        | ❌       | ❌          | ⚠️ 部分         |
| **合成步骤数**               | 1步             | 2步       | 1步      | 3步         | 2层            |
| **分组键**                 | 科室             | 科室/分类    | 诊断      | 类别/来源      | patient_id+科室 |
| **合成后sessions/patient** | 2~5            | 2~4      | 2~4     | 2~4        | 2~4(+天然链)     |
| **独特处理**                | description作首句 | 长回答拆分    | 实体标注保留  | 三层合成       | 跨子集串联         |
| **最终是否多次问诊**            | ✅              | ✅        | ✅       | ✅          | ✅             |


---

## 四、使用方式

### 处理全部数据集

```bash
python examples/dataset_collectors/run_all.py \
    --input_dir /path/to/raw_datasets \
    --output_dir /path/to/output \
    --download
```

### 处理单个数据集

```bash
python examples/dataset_collectors/meddialog_cn_examiner.py \
    --input_path /path/to/meddialog_cn/train.json \
    --output_path /path/to/output/meddialog_cn_examiner.json \
    --max_dialogues 100
```

### 禁用多会话合成

```bash
python examples/dataset_collectors/run_all.py \
    --input_dir /path/to/raw \
    --output_dir /path/to/out \
    --no_multi_session
```

### 自定义每患者会话数

```bash
python examples/dataset_collectors/run_all.py \
    --input_dir /path/to/raw \
    --output_dir /path/to/out \
    --sessions_min 3 --sessions_max 6
```

### 启用LLM生成患者档案

```bash
python examples/dataset_collectors/run_all.py \
    --input_dir /path/to/raw \
    --output_dir /path/to/out \
    --use_llm_profile
```

---

## 五、文件结构

```
examples/dataset_collectors/
├── __init__.py                    # 统一导出
├── clinical_scenarios.py          # 5大临床场景配置
├── multi_session_synthesizer.py   # 多会话合成器（分组→场景分配→合并→profile生成）
├── base_examiner.py               # 基类（LightMemoryManager + 两阶段pipeline + run()）
├── meddialog_cn_examiner.py       # MedDialog-CN: 按科室分组
├── cmtmedqa_examiner.py           # CMtMedQA: 长回答拆分 + 按分类分组
├── imcs21_examiner.py             # IMCS-21: 按诊断分组
├── huatuo26m_examiner.py          # Huatuo-26M: 三步合成（聚合→展开→分组）
├── lcmdc_examiner.py              # LCMDC: 跨子集串联 + 退化分组
├── download_datasets.py           # HuggingFace数据集下载器
├── run_all.py                     # 统一运行入口
└── README.md                      # 本文档
```

