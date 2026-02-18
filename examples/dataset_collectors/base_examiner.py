"""
Base Examiner Agent Module
==========================

Shared logic for all dataset examiner agents. Implements the two-stage pipeline:
  Stage 1 (Memory Preprocess): LLM extracts to_memory → generates memory operations → executes via MemoryManager → builds oracle_memory_base
  Stage 2 (Examiner Agent):    Samples memory points → determines difficulty/type → LLM generates question/answer → annotates source

Multi-Session Synthesis:
  All datasets are converted to "same-patient multi-session" format using the
  MultiSessionSynthesizer. Each examiner subclass provides a `grouping_key_fn()`
  and optional `sessions_range` to control how single-session dialogues are
  merged into multi-session patient records separated by "---诊疗分割线---".

Usage:
    Subclass `BaseExaminerAgent` and implement:
      - load_raw_data(input_path) -> list of raw records
      - normalize_dialogue(raw_record) -> (dialogue_id, messages, metadata)
      - grouping_key_fn(normalised_record) -> str  (for multi-session grouping)
"""

import json
import os
import sys
import re
import uuid
import time
import random
import hashlib
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

# ── project imports ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Agent-R1/
sys.path.insert(0, str(PROJECT_ROOT))

from agent_r1.utils.llm import query_llm
from examples.dataset_collectors.multi_session_synthesizer import (
    synthesize_multi_session_patients,
    merge_sessions_into_messages,
    CONSULTATION_SEPARATOR,
)
from examples.dataset_collectors.clinical_scenarios import (
    CLINICAL_SCENARIOS,
    SCENARIO_KEYS,
    pick_scenario_for_record,
)

# ── Constants ────────────────────────────────────────────────────────

# Default LLM model for examiner tasks
DEFAULT_MODEL = "gpt-4o-2024-11-20"

# Memory layer definitions
MEMORY_LAYERS = ["working", "identity", "history", "experience"]

MEMORY_LAYER_DESCRIPTIONS = {
    "working":    "当前对话关键信息（临时，会话特定）。存储当前问诊中的症状、检查结果、诊断结论等。",
    "identity":   "患者身份和基础信息（永久特征）。存储姓名、年龄、性别、职业等稳定个人信息。",
    "history":    "历史诊断和医疗记录（过去医疗事件）。存储既往诊断、病史、既往治疗、慢性病等。",
    "experience": "临床经验和案例（通用医学知识）。存储临床观察、治疗模式、通用医学知识等。",
}

MEMORY_LAYER_DESCRIPTIONS_EN = {
    "working":    "Current dialogue key information (temporary, session-specific). Store current symptoms, examination results, diagnosis conclusions.",
    "identity":   "Patient identity and basic information (permanent characteristics). Store name, age, gender, occupation.",
    "history":    "Historical diagnosis and medical records (past medical events). Store past diagnoses, medical history, chronic conditions.",
    "experience": "Clinical experience and case (general medical knowledge). Store clinical observations, treatment patterns, medical knowledge.",
}

# Examiner difficulty/type mapping
LEVEL_CONFIG = {
    "Level 1": {
        "type": "State Tracking",
        "type_cn": "状态追踪",
        "description": "Track current dialogue state from working memory.",
        "sampling": {"primary": ["working"], "count": 1},
        "context_window": 6,       # only recent turns — current session
        "recency_bias": "recent",  # prefer memory items from current session
        "min_memory_age_turns": 0, # can test something just said
        "prompt_guidance": (
            "Generate a factual question about something the PATIENT or DOCTOR explicitly "
            "stated in the CURRENT session. The answer must come directly from what was said "
            "in the dialogue, NOT from external medical knowledge. "
            "Example: 'What symptom did the patient report today?' or "
            "'What medication did the doctor prescribe in this visit?'"
        ),
        "prompt_guidance_cn": (
            "根据当前会话中患者或医生明确陈述的事实生成问题。"
            "答案必须直接来源于对话中说过的话，禁止依赖外部医学知识。\n"
            "题目只能问'是什么/有哪些'，不能在问题中把答案写出来。\n"
            "好: '患者今天主诉的症状是什么？'\n"
            "好: '医生在本次就诊中建议做哪些检查？'\n"
            "坏: '患者今天主诉腰疼持续且轻微，医生建议了什么？'（题干已泄露症状细节）\n"
            "坏: '前列腺增生有哪些危害？'（医学知识题，不是记忆题）"
        ),
    },
    "Level 2": {
        "type": "Accurate Retrieval",
        "type_cn": "精确检索",
        "description": "Retrieve specific factual information from long-term memory.",
        "sampling": {"primary": ["identity", "history"], "count": 1},
        "context_window": 4,        # minimal recent context (just enough to frame the question)
        "recency_bias": "old",      # prefer memory items stored earlier in the dialogue
        "min_memory_age_turns": 6,   # information must be at least 6 turns old
        "prompt_guidance": (
            "Generate a question that requires precise recall of a specific FACT recorded "
            "earlier in this patient's medical history or identity. The fact must have been "
            "explicitly mentioned in a PREVIOUS session or earlier in the dialogue. "
            "Do NOT ask about medical knowledge — only ask about THIS patient's documented data. "
            "Example: 'What was the patient's fasting blood sugar reported three months ago?' or "
            "'Which allergy did the patient mention in the first visit?'"
        ),
        "prompt_guidance_cn": (
            "根据患者档案或既往诊疗中明确记录的具体事实出题。"
            "这个事实必须是之前的会话或对话早期明确提到过的，而不是当前刚说的。\n"
            "禁止出医学知识题，只能问这位患者的具体记录数据。\n"
            "题目要问一个需要回忆才能回答的具体事实，不能在题干中透露该事实。\n"
            "好: '患者首次就诊时报告对哪种药物过敏？'\n"
            "好: '上次检查的空腹血糖值是多少？'\n"
            "坏: '患者首次就诊时报告对青霉素过敏，这是否需要注意？'（题干已透露过敏药物）\n"
            "坏: '患者的性别和年龄分别是什么？'（过于通用，无诊疗特异性）"
        ),
    },
    "Level 3": {
        "type": "Long-range Understanding",
        "type_cn": "长程理解",
        "description": "Cross-temporal association between history and working memory.",
        "sampling": {"primary": ["history", "working"], "count": 2, "require_cross_layer": True},
        "context_window": 8,         # moderate context
        "recency_bias": "cross",     # one old + one recent item
        "min_memory_age_turns": 10,  # the history item should be distant
        "prompt_guidance": (
            "Generate a question that requires COMPARING or CONNECTING a fact from an EARLIER "
            "session with a fact from the CURRENT session. The question must reference concrete "
            "data points from the dialogue record (e.g., lab values, symptoms, medications) "
            "across two different time points. "
            "Do NOT ask generic medical questions. "
            "Example: 'Comparing the blood pressure from the first visit to today, has it improved?' or "
            "'The patient was prescribed Drug A last time but now reports symptom X — is there a connection?'"
        ),
        "prompt_guidance_cn": (
            "生成一个需要将之前某次就诊的事实与当前就诊的事实进行对比或关联的问题。\n"
            "题目应指明需要对比的两个时间点，但不能把具体数据值/内容写在题干里。\n"
            "好: '对比患者首次就诊和本次复查的血压数据，变化趋势如何？'（需要回忆两次的具体数值）\n"
            "好: '患者之前的用药方案和本次有什么变化？'（需要回忆两次的具体药物）\n"
            "坏: '患者上次血压140/90，本次130/85，是否有改善？'（两次数值都写出来了）\n"
            "坏: '长期高血压有什么后果？'（知识题）"
        ),
    },
    "Level 4": {
        "type": "Conflict & Safety Check",
        "type_cn": "冲突与安全检查",
        "description": "Detect conflicts or safety issues between memory layers.",
        "sampling": {"primary": ["working", "identity", "history"], "count": 2, "require_cross_layer": True},
        "context_window": 6,
        "recency_bias": "cross",    # one from identity/history, one from working
        "min_memory_age_turns": 4,
        "prompt_guidance": (
            "Generate a question about a REAL conflict or safety issue that can be detected "
            "by cross-referencing facts from THIS patient's record. Both facts must have been "
            "explicitly stated in the dialogue. "
            "Do NOT invent hypothetical scenarios — the conflict must exist in the actual data. "
            "Example: 'The patient reported penicillin allergy in visit 1, but was prescribed "
            "amoxicillin today — is this safe?' or "
            "'The patient's previous diagnosis was X but the current symptoms suggest Y — "
            "is there a contradiction?'"
        ),
        "prompt_guidance_cn": (
            "生成一个关于该患者记录中真实存在的矛盾或安全问题的问题。\n"
            "题目应提示存在潜在冲突需要检查，但不能直接写出冲突的具体内容。\n"
            "好: '患者曾报告的药物过敏史与本次处方是否存在冲突？'（需要回忆过敏药物和处方内容）\n"
            "好: '患者既往诊断与当前症状是否存在矛盾？请说明。'（需要回忆两边的具体内容）\n"
            "坏: '患者对青霉素过敏，但本次开了阿莫西林，是否安全？'（冲突双方都写出来了）\n"
            "坏: '如果患者自行使用α受体阻滞剂可能有什么风险？'（假设性+知识题）"
        ),
    },
}

# Trigger probability settings
TRIGGER_BASE_PROB = 0.30        # base trigger probability
TRIGGER_BOOST_PROB = 0.70       # trigger probability when to_memory is non-empty


# ── Helper dataclass (lightweight, no FAISS dependency) ──────────────

@dataclass
class LightMemoryItem:
    """Lightweight memory item (no embedding required)."""
    id: str
    layer: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        if "turn_idx" in self.metadata:
            d["turn_idx"] = self.metadata["turn_idx"]
        return d


class LightMemoryManager:
    """
    Lightweight memory manager that does NOT require FAISS / FlagEmbedding.
    Used during data preprocessing to track oracle_memory_base state.
    For actual retrieval tasks, use the full MemoryManager.
    """

    def __init__(self):
        self.memories: Dict[str, List[LightMemoryItem]] = {
            layer: [] for layer in MEMORY_LAYERS
        }
        self._current_turn_idx: int = 0

    def set_turn_idx(self, turn_idx: int):
        self._current_turn_idx = turn_idx

    # ── core operations ──────────────────────────────────────────────

    def insert(self, layer: str, content: str, metadata: Dict[str, Any] = None) -> str:
        assert layer in MEMORY_LAYERS, f"Invalid layer: {layer}"
        mem_id = str(uuid.uuid4())
        meta = metadata or {}
        meta.setdefault("turn_idx", self._current_turn_idx)
        item = LightMemoryItem(
            id=mem_id, layer=layer, content=content,
            timestamp=time.time(), metadata=meta,
        )
        self.memories[layer].append(item)
        return mem_id

    def update(self, layer: str, memory_id: str, new_content: str, metadata: Dict[str, Any] = None) -> bool:
        assert layer in MEMORY_LAYERS, f"Invalid layer: {layer}"
        for item in self.memories[layer]:
            if item.id == memory_id:
                item.content = new_content
                if metadata:
                    item.metadata.update(metadata)
                return True
        return False

    def delete(self, layer: str, memory_id: str) -> bool:
        assert layer in MEMORY_LAYERS, f"Invalid layer: {layer}"
        for i, item in enumerate(self.memories[layer]):
            if item.id == memory_id:
                self.memories[layer].pop(i)
                return True
        return False

    def reset_working_memory(self):
        self.memories["working"] = []

    def get_memory_state(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            layer: [item.to_dict() for item in items]
            for layer, items in self.memories.items()
        }

    def get_memory_summary_text(self) -> str:
        parts = []
        for layer in MEMORY_LAYERS:
            items = self.memories[layer]
            if items:
                parts.append(f"\n{layer.upper()} MEMORY ({len(items)} items):")
                for item in items:
                    parts.append(f"  - [{item.id}] {item.content}")
        return "\n".join(parts) if parts else "No memories stored yet."

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.memories.values())

    def total_items(self) -> int:
        return sum(len(v) for v in self.memories.values())


# ── LLM prompt builders ─────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Heuristic: if >30 % CJK characters → 'zh', else 'en'."""
    if not text:
        return "en"
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return "zh" if cjk / max(len(text), 1) > 0.15 else "en"


def build_extract_to_memory_prompt(
    current_content: str,
    role: str,
    dialogue_history: List[str],
    language: str = "zh",
) -> str:
    """Build prompt for Stage 1 Step 1: extract to_memory items from a message."""
    history_text = "\n".join(dialogue_history[-10:]) if dialogue_history else "(无)"

    if language == "zh":
        return f"""你是一个医疗对话信息提取专家。请从当前消息中提取需要记忆的关键医疗信息，并分配到正确的记忆层。

## 对话历史（最近几轮）
{history_text}

## 当前消息（{role}）
{current_content}

## 记忆层分配规则（必须严格遵守）

### identity（患者身份档案，跨就诊永久保留）
必须放入identity的信息：
- 性别、年龄、姓名
- 过敏史（如药物过敏）
- 基础疾病/慢性病（如高血压、糖尿病）
- 体质特征（如体重、BMI）
- 家族病史
- 生活习惯（如吸烟史、饮酒量）

### history（诊疗历史记录，跨就诊永久保留）
必须放入history的信息：
- 既往诊断结果（如"确诊为前列腺增生"）
- 曾用药物和处方
- 检查报告和化验结果（如血糖值、影像报告）
- 手术记录
- 医生给出的确定性建议（如"建议每年复查PSA"）

### working（当前会话临时信息，切换会话后清空）
仅放入working的信息：
- 本次就诊的主诉（如"今天头疼"）
- 当前症状描述（如"疼痛程度轻微"）
- 本次对话中的临时讨论内容
- 尚未确定的推测和问题

### experience（临床经验，通常不用）
- 仅当对话包含通用临床经验总结时才使用

## 关键原则
1. 同一条信息可能需要拆分到不同层：比如"患者张三，男，45岁，主诉头疼3天"应拆为 identity + working
2. 诊断结果、用药方案、检查报告属于 history，不是 working
3. 如果不确定，偏向 identity/history 而非 working（宁可多存不可漏存）

## 输出
JSON数组，每个元素格式为 "层名: 内容"。无信息则输出 [null]。

示例:
["identity: 患者男，45岁，有青霉素过敏史", "history: 半年前确诊2型糖尿病，目前服用二甲双胍", "working: 本次主诉近一周头痛加重"]

请直接输出JSON数组。"""
    else:
        return f"""You are a medical dialogue information extraction expert. Extract key medical information and assign each item to the CORRECT memory layer.

## Dialogue History (recent turns)
{history_text}

## Current Message ({role})
{current_content}

## Memory Layer Assignment Rules (MUST follow strictly)

### identity (patient profile, persists across visits)
MUST go into identity:
- Gender, age, name
- Allergies (e.g., drug allergies)
- Chronic conditions (e.g., hypertension, diabetes)
- Physical characteristics (e.g., weight, BMI)
- Family medical history
- Lifestyle habits (e.g., smoking, alcohol use)

### history (medical records, persists across visits)
MUST go into history:
- Past diagnoses (e.g., "diagnosed with BPH")
- Previous medications and prescriptions
- Lab results and examination reports (e.g., blood sugar levels, imaging)
- Surgical records
- Definitive medical advice from doctor (e.g., "annual PSA screening recommended")

### working (current session only, cleared between visits)
ONLY put in working:
- Current visit's chief complaint (e.g., "headache today")
- Current symptom descriptions (e.g., "mild pain")
- Temporary discussion points in this conversation
- Unconfirmed speculation and questions

### experience (clinical knowledge, rarely used)
- Only for general clinical experience summaries

## Key Principles
1. One message may need items in MULTIPLE layers: "John, male, 45, reports headache for 3 days" = identity + working
2. Diagnoses, prescriptions, and lab results belong in history, NOT working
3. When in doubt, prefer identity/history over working (better to persist than to lose)

## Output
JSON array, each element: "layer: content". Output [null] if nothing to memorize.

Example:
["identity: Male, 45yo, penicillin allergy", "history: Diagnosed with T2DM 6 months ago, on metformin", "working: Chief complaint: worsening headache for 1 week"]

Output only the JSON array."""


def build_memory_operations_prompt(
    to_memory_items: List[str],
    memory_state: Dict[str, List[Dict[str, Any]]],
    language: str = "zh",
) -> str:
    """Build prompt for Stage 1 Step 2: generate memory function calls."""

    state_text_parts = []
    for layer in MEMORY_LAYERS:
        items = memory_state.get(layer, [])
        if items:
            state_text_parts.append(f"\n{layer.upper()} MEMORY:")
            for it in items:
                state_text_parts.append(f'  - [id={it["id"]}] {it["content"]}')
    state_text = "\n".join(state_text_parts) if state_text_parts else "（空）"

    items_text = "\n".join(f"  - {item}" for item in to_memory_items if item)

    if language == "zh":
        return f"""你是一个医疗记忆管理专家。请根据待存储信息和当前记忆状态，生成记忆操作。

## 待存储信息
{items_text}

## 当前记忆状态
{state_text}

## 可用操作
1. memory_insert(layer, content) - 插入新记忆
2. memory_update(layer, memory_id, content) - 更新已有记忆
3. memory_delete(layer, memory_id) - 删除记忆
4. memory_wait() - 无需操作

## 输出要求
输出JSON数组，每个元素是一个操作对象。示例:
[
  {{"action": "memory_insert", "layer": "working", "content": "患者主诉头痛"}},
  {{"action": "memory_update", "layer": "identity", "memory_id": "xxx", "content": "患者张三，45岁"}},
  {{"action": "memory_wait"}}
]

注意：
- 如果信息已存在且内容相同，使用 memory_wait
- 如果信息已存在但需要更新，使用 memory_update 并提供正确的 memory_id
- 新信息使用 memory_insert
- 请直接输出JSON数组"""
    else:
        return f"""You are a medical memory management expert. Generate memory operations based on the information to store and current memory state.

## Information to Store
{items_text}

## Current Memory State
{state_text}

## Available Operations
1. memory_insert(layer, content) - Insert new memory
2. memory_update(layer, memory_id, content) - Update existing memory
3. memory_delete(layer, memory_id) - Delete memory
4. memory_wait() - No operation needed

## Output
Output a JSON array of operation objects. Example:
[
  {{"action": "memory_insert", "layer": "working", "content": "Patient reports headache"}},
  {{"action": "memory_update", "layer": "identity", "memory_id": "xxx", "content": "Patient John, age 45"}},
  {{"action": "memory_wait"}}
]

Notes:
- If info already exists and is the same, use memory_wait
- If info exists but needs updating, use memory_update with correct memory_id
- New info → memory_insert
- Output only the JSON array."""


def build_examiner_prompt(
    dialogue_context: str,
    target_memory_items: List[Dict[str, Any]],
    level: str,
    language: str = "zh",
) -> str:
    """Build prompt for Stage 2: Examiner Agent question/answer generation."""
    cfg = LEVEL_CONFIG[level]
    guidance = cfg["prompt_guidance_cn"] if language == "zh" else cfg["prompt_guidance"]
    type_name = cfg["type_cn"] if language == "zh" else cfg["type"]

    items_parts = []
    for it in target_memory_items:
        layer_tag = it.get("layer", "?")
        content_val = it.get("content", "")
        turn = it.get("turn_idx", it.get("metadata", {}).get("turn_idx", ""))
        if turn != "" and level in ("Level 3", "Level 4"):
            items_parts.append(f'  - [{layer_tag}, 记录于第{turn}轮] {content_val}')
        else:
            items_parts.append(f'  - [{layer_tag}] {content_val}')
    items_text = "\n".join(items_parts)

    if language == "zh":
        return f"""你是一个医疗诊疗记忆评测出题专家（Examiner Agent）。

## 你的任务
根据对话上下文和目标记忆点，出一道"记忆测试题"。
这道题测试的是：模型是否记住了对话中提到过的某个具体事实。

## 最重要的规则：题目禁止泄露答案

题目只能提问，不能在题干中复述、引用、暗示或包含答案的内容。
如果题目读完就能猜到答案，这道题就是失败的。

### 反面示例（绝对禁止）

❌ 问："患者主诉腰疼持续存在且疼痛感轻微，医生建议适当休息、调整坐姿。这是否与患者要求做检查存在矛盾？"
→ 题干已经把"医生建议了什么"写出来了，看题就知道答案。

❌ 问："患者在本次咨询中提到前列腺增生，结合上次医生建议的改善生活习惯（规律作息、避免辛辣刺激食物、适当运动），这些建议是否有帮助？"
→ 括号里直接写出了上次的建议内容，信息全部泄露。

❌ 问："患者的性别和年龄分别是什么？"
→ 虽然不泄露答案，但这类问题过于简单和重复，缺乏诊疗特异性。

### 正面示例（应该这样出题）

✓ 问："医生在上次就诊时给出了哪些生活方式调整建议？"
→ 只问"有哪些建议"，不透露具体内容，需要回忆才能回答。

✓ 问："患者第一次就诊时报告的主要症状是什么？"
→ 指向一个具体时间点的事实，不泄露症状内容。

✓ 问："上次复查时医生开的处方药名称是什么？"
→ 问具体事实，不在题干中透露药名。

✓ 问："患者之前报告过对哪种药物过敏？本次处方是否需要注意？"
→ Level 4 冲突检测，只提示"有过敏史"但不说具体药物。

## 对话上下文
{dialogue_context}

## 目标记忆点（内部参考，用于确定正确答案——不要在题干中复述这些内容！）
{items_text}

## 难度等级: {level}
## 题型: {type_name}

## 出题指导
{guidance}

## 输出JSON
{{
  "question": "一个不包含答案的提问",
  "answer": "从目标记忆点中直接得到的简洁答案"
}}

最终检查清单（输出前逐条确认）:
1. 把question和answer放在一起读——如果不看answer，只看question，能否猜到答案？如果能，必须重新出题
2. question中是否出现了answer的关键词或同义表述？如果是，必须删除
3. 答案是否来自对话中明确提到的事实？（禁止医学知识推断）
4. 题目是否有诊疗特异性？（禁止"患者性别和年龄是什么"这类通用题）

请直接输出JSON对象。"""
    else:
        return f"""You are a medical record memory evaluation expert (Examiner Agent).

## Your Task
Generate a memory test question based on the dialogue context and target memory points.
The question tests whether the model REMEMBERS a specific fact from the dialogue.

## CRITICAL RULE: The question must NOT leak the answer

The question can only ASK — it must not restate, quote, hint at, or contain the answer.
If someone can guess the answer just by reading the question, the question has FAILED.

### Bad Examples (FORBIDDEN)

❌ Q: "The patient reported persistent mild back pain, and the doctor suggested rest, posture adjustment, and exercise. Does this conflict with the patient wanting tests?"
→ The question already states what the doctor suggested — answer is in the question.

❌ Q: "Combining the doctor's previous advice on lifestyle changes (regular sleep, avoid spicy food, moderate exercise), are these helpful for the current symptoms?"
→ The parenthetical directly reveals the advice content.

### Good Examples (DO THIS)

✓ Q: "What lifestyle adjustment advice did the doctor give during the last visit?"
→ Asks WHAT the advice was without revealing it. Requires memory to answer.

✓ Q: "What was the main symptom the patient reported in the first consultation?"
→ Points to a specific time point without leaking the symptom.

✓ Q: "What medication was the patient previously reported to be allergic to? Should the current prescription be reconsidered?"
→ Level 4 conflict detection — mentions allergy exists but not the specific drug.

## Dialogue Context
{dialogue_context}

## Target Memory Points (internal reference for determining correct answer — DO NOT restate in question!)
{items_text}

## Difficulty: {level}
## Type: {type_name}

## Guidance
{guidance}

## Output JSON
{{
  "question": "a question that does NOT contain the answer",
  "answer": "concise answer derived directly from target memory points"
}}

Pre-output checklist (verify each before responding):
1. Read question + answer together — can you guess the answer from the question alone? If yes, REWRITE
2. Does the question contain keywords or paraphrases from the answer? If yes, REMOVE them
3. Is the answer a fact explicitly stated in the dialogue? (no medical knowledge inference)
4. Is the question clinically specific? (no generic "what is the patient's age?" questions)

Output only the JSON object."""


# ── Answer leakage detection ─────────────────────────────────────────

def _check_answer_leakage(question: str, answer: str, target_items: List[Dict[str, Any]]) -> bool:
    """
    Return True if the question likely leaks the answer content.
    Checks: (1) long answer substrings in question, (2) key entity overlap,
    (3) generic identity-only questions.
    """
    q = question.strip()
    a = answer.strip()
    if not q or not a:
        return True

    # Check 1: Long substrings of the answer appear verbatim in question
    # Split answer into segments, check if any segment >= 6 chars appears in Q
    a_segments = re.split(r'[，。、；：！？\s,\.;:!?\n]+', a)
    leaked_chars = 0
    total_chars = sum(len(s) for s in a_segments if len(s) >= 4)
    for seg in a_segments:
        seg = seg.strip()
        if len(seg) >= 4 and seg in q:
            leaked_chars += len(seg)
    if total_chars > 0 and leaked_chars / total_chars > 0.4:
        return True

    # Check 2: Generic identity-only questions
    generic_patterns = [
        r'^患者的?(性别|年龄|姓名).{0,6}(是什么|分别是|多少)',
        r'^(What|what).*(age|gender|sex|name).*\?$',
    ]
    for pat in generic_patterns:
        if re.search(pat, q):
            return True

    # Check 3: Target memory content substantially quoted in question
    for it in target_items:
        mem_content = it.get("content", "")
        if len(mem_content) >= 8:
            # Check if > 60% of the memory content appears in the question
            mem_segs = re.split(r'[，。、；：！？\s,\.;:!?\n]+', mem_content)
            mem_total = sum(len(s) for s in mem_segs if len(s) >= 4)
            mem_leaked = sum(len(s) for s in mem_segs if len(s) >= 4 and s in q)
            if mem_total > 0 and mem_leaked / mem_total > 0.5:
                return True

    return False


# ── Layer reclassification keywords ──────────────────────────────────

_IDENTITY_KEYWORDS_ZH = [
    "岁", "男", "女", "性别", "年龄", "过敏", "体重", "身高", "BMI",
    "吸烟", "饮酒", "家族", "慢性", "基础疾病", "既往", "姓名",
]
_HISTORY_KEYWORDS_ZH = [
    "诊断", "确诊", "处方", "用药", "服用", "检查结果", "化验", "报告",
    "手术", "血糖", "血压", "CT", "MRI", "B超", "复查",
    "治疗方案", "PSA", "指标",
]


def _classify_content_layer(content: str, language: str = "zh") -> Optional[str]:
    """
    Rule-based layer classification for a memory content string.
    Returns 'identity', 'history', or None (leave as working).
    """
    if not content:
        return None

    if language == "zh":
        identity_score = sum(1 for kw in _IDENTITY_KEYWORDS_ZH if kw in content)
        history_score = sum(1 for kw in _HISTORY_KEYWORDS_ZH if kw in content)
    else:
        content_lower = content.lower()
        identity_kw = ["age", "male", "female", "gender", "allergy", "allergic",
                        "weight", "height", "bmi", "smoking", "alcohol", "family",
                        "chronic", "name"]
        history_kw = ["diagnos", "prescri", "medication", "lab result", "report",
                       "surgery", "blood sugar", "blood pressure", "ct ", "mri",
                       "ultrasound", "follow-up", "recommend", "treatment plan"]
        identity_score = sum(1 for kw in identity_kw if kw in content_lower)
        history_score = sum(1 for kw in history_kw if kw in content_lower)

    if identity_score > 0 and identity_score >= history_score:
        return "identity"
    if history_score > 0:
        return "history"
    return None


# ── JSON parsing helpers ─────────────────────────────────────────────

def safe_parse_json(text: str) -> Any:
    """Try to parse JSON from LLM response, tolerant of markdown fences."""
    if not text:
        return None
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object in text
        for pattern in [r'\[.*\]', r'\{.*\}']:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    continue
        return None


# ── Core examiner logic ──────────────────────────────────────────────

def _reclassify_layer(item_str: str, language: str = "zh") -> str:
    """
    Post-process a 'layer: content' string. If the LLM tagged something as
    'working' but its content clearly belongs to identity/history, reclassify it.
    """
    if not item_str or ":" not in item_str:
        return item_str
    layer, content = item_str.split(":", 1)
    layer = layer.strip().lower()
    content = content.strip()

    if layer != "working":
        return item_str  # already non-working, trust the LLM

    better = _classify_content_layer(content, language)
    if better and better != "working":
        return f"{better}: {content}"
    return item_str


def extract_to_memory(
    content: str,
    role: str,
    dialogue_history: List[str],
    model: str = DEFAULT_MODEL,
    language: str = "zh",
) -> List[Optional[str]]:
    """Stage 1 Step 1: Use LLM to extract to_memory items, with layer reclassification."""
    prompt = build_extract_to_memory_prompt(content, role, dialogue_history, language)
    try:
        result = query_llm(model_name=model, messages=prompt, temperature=0.3)
        parsed = safe_parse_json(result.get("response", ""))
        if isinstance(parsed, list):
            return [_reclassify_layer(item, language) if isinstance(item, str) else item
                    for item in parsed]
        return [None]
    except Exception as e:
        print(f"  [extract_to_memory] LLM error: {e}")
        return [None]


def generate_and_execute_memory_ops(
    to_memory_items: List[Optional[str]],
    memory_mgr: LightMemoryManager,
    model: str = DEFAULT_MODEL,
    language: str = "zh",
) -> None:
    """Stage 1 Steps 2–3: Generate memory operations via LLM and execute them."""
    valid_items = [item for item in to_memory_items if item is not None and item.strip()]
    if not valid_items:
        return

    prompt = build_memory_operations_prompt(
        valid_items, memory_mgr.get_memory_state(), language
    )
    try:
        # print("LLM Prompt: ", prompt)
        result = query_llm(model_name=model, messages=prompt, temperature=0.3)
        # print("LLM PResult: ", result)
        ops = safe_parse_json(result.get("response", ""))
        if not isinstance(ops, list):
            # Fallback: directly insert all items
            for item_str in valid_items:
                parts = item_str.split(":", 1)
                if len(parts) == 2:
                    layer = parts[0].strip().lower()
                    content = parts[1].strip()
                    if layer in MEMORY_LAYERS:
                        memory_mgr.insert(layer, content)
            return

        for op in ops:
            if not isinstance(op, dict):
                continue
            action = op.get("action", "")
            if action == "memory_insert":
                layer = op.get("layer", "working")
                content = op.get("content", "")
                if layer in MEMORY_LAYERS and content:
                    memory_mgr.insert(layer, content)
            elif action == "memory_update":
                layer = op.get("layer", "working")
                mem_id = op.get("memory_id", "")
                content = op.get("content", "")
                if layer in MEMORY_LAYERS and mem_id and content:
                    memory_mgr.update(layer, mem_id, content)
            elif action == "memory_delete":
                layer = op.get("layer", "working")
                mem_id = op.get("memory_id", "")
                if layer in MEMORY_LAYERS and mem_id:
                    memory_mgr.delete(layer, mem_id)
            # memory_wait → do nothing

    except Exception as e:
        print(f"  [generate_and_execute_memory_ops] LLM error: {e}")
        # Fallback: direct insert
        for item_str in valid_items:
            parts = item_str.split(":", 1)
            if len(parts) == 2:
                layer = parts[0].strip().lower()
                content = parts[1].strip()
                if layer in MEMORY_LAYERS:
                    memory_mgr.insert(layer, content)


def sample_memory_for_level(
    oracle_memory_base: Dict[str, List[Dict[str, Any]]],
    level: str,
    current_turn_idx: int = 0,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Stage 2 Step 2: Sample memory points with temporal awareness.

    Sampling strategies by recency_bias:
      - "recent":  prefer items with higher turn_idx (recent items)
      - "old":     prefer items with lower turn_idx (distant items)
      - "cross":   pick one old + one recent, preferring different layers

    Items are filtered by min_memory_age_turns: a memory item created at
    turn T is only eligible for "old" sampling if (current_turn_idx - T)
    >= min_memory_age_turns. This implements the "half-life" concept where
    the information point was planted earlier and the question is triggered
    at the current turn.
    """
    cfg = LEVEL_CONFIG.get(level)
    if cfg is None:
        return None, []

    primary_layers = cfg["sampling"]["primary"]
    count = cfg["sampling"]["count"]
    recency_bias = cfg.get("recency_bias", "recent")
    min_age = cfg.get("min_memory_age_turns", 0)
    require_cross = cfg.get("sampling", {}).get("require_cross_layer", False)

    # Collect all candidate items with their layer tag
    all_candidates = []
    for layer in primary_layers:
        for item in oracle_memory_base.get(layer, []):
            enriched = {**item, "layer": layer}
            enriched.setdefault("turn_idx", 0)
            all_candidates.append(enriched)

    if len(all_candidates) < count:
        return None, []

    if recency_bias == "recent":
        # Sort by turn_idx descending, prefer recent
        all_candidates.sort(key=lambda x: x.get("turn_idx", 0), reverse=True)
        # Weighted sampling: recent items get higher weight
        weights = [max(1, c.get("turn_idx", 0) + 1) for c in all_candidates]
        selected = _weighted_sample(all_candidates, weights, count)

    elif recency_bias == "old":
        # Filter: only items old enough
        old_candidates = [
            c for c in all_candidates
            if (current_turn_idx - c.get("turn_idx", 0)) >= min_age
        ]
        if not old_candidates:
            old_candidates = all_candidates  # fallback
        # Weighted sampling: older items get higher weight
        weights = [max(1, current_turn_idx - c.get("turn_idx", 0) + 1) for c in old_candidates]
        selected = _weighted_sample(old_candidates, weights, count)

    elif recency_bias == "cross":
        # Pick one old item and one recent item, preferring different layers
        old_candidates = [
            c for c in all_candidates
            if (current_turn_idx - c.get("turn_idx", 0)) >= min_age
        ]
        recent_candidates = [
            c for c in all_candidates
            if (current_turn_idx - c.get("turn_idx", 0)) < max(min_age, 4)
        ]
        if not old_candidates:
            old_candidates = all_candidates
        if not recent_candidates:
            recent_candidates = all_candidates

        old_pick = random.choice(old_candidates)
        # For cross-layer: try to pick from a different layer
        if require_cross:
            cross_recent = [c for c in recent_candidates if c["layer"] != old_pick["layer"]]
            recent_pool = cross_recent if cross_recent else recent_candidates
        else:
            recent_pool = recent_candidates
        recent_pick = random.choice(recent_pool)

        selected = [old_pick, recent_pick][:count]
    else:
        selected = random.sample(all_candidates, count)

    return level, selected


def _weighted_sample(items: list, weights: list, k: int) -> list:
    """Weighted sampling without replacement."""
    if len(items) <= k:
        return list(items)
    total = sum(weights)
    if total == 0:
        return random.sample(items, k)
    result = []
    remaining = list(zip(items, weights))
    for _ in range(k):
        r = random.uniform(0, sum(w for _, w in remaining))
        cumulative = 0
        for idx, (item, w) in enumerate(remaining):
            cumulative += w
            if cumulative >= r:
                result.append(item)
                remaining.pop(idx)
                break
    return result


def decide_trigger_and_level(
    to_memory_items: List[Optional[str]],
    oracle_memory_base: Dict[str, List[Dict[str, Any]]],
    current_turn_idx: int = 0,
) -> Optional[str]:
    """
    Stage 2 Step 1 + 3: Decide whether to trigger and pick a level.

    Level selection is weighted by what's available:
      - Level 1 can fire anytime (only needs working memory)
      - Level 2 needs identity/history items that are old enough
      - Level 3/4 need both old and recent items across layers
      - Higher levels are preferred when enough dialogue history exists

    Returns a level string or None.
    """
    has_new_info = any(item is not None and item.strip() for item in to_memory_items)
    prob = TRIGGER_BOOST_PROB if has_new_info else TRIGGER_BASE_PROB

    if random.random() > prob:
        return None

    # Build eligible levels with weights (prefer higher levels when possible)
    level_weights = {"Level 1": 1.0, "Level 2": 1.5, "Level 3": 2.0, "Level 4": 2.5}
    eligible = []
    for level in LEVEL_CONFIG:
        _, items = sample_memory_for_level(oracle_memory_base, level, current_turn_idx)
        if items:
            eligible.append((level, level_weights.get(level, 1.0)))

    if not eligible:
        return None

    # Weighted random selection among eligible levels
    total_w = sum(w for _, w in eligible)
    r = random.uniform(0, total_w)
    cumulative = 0
    for level, w in eligible:
        cumulative += w
        if cumulative >= r:
            return level
    return eligible[-1][0]


def generate_memory_query(
    dialogue_context: str,
    oracle_memory_base: Dict[str, List[Dict[str, Any]]],
    to_memory_items: List[Optional[str]],
    model: str = DEFAULT_MODEL,
    language: str = "zh",
    current_turn_idx: int = 0,
    full_dialogue_history: List[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Stage 2: Full examiner pipeline — trigger → sample → generate Q/A → annotate source.

    Uses level-specific context windows: Level 1 sees recent turns, Level 3/4
    see a wider window that includes the session where the target info was planted.
    """
    level = decide_trigger_and_level(
        to_memory_items, oracle_memory_base, current_turn_idx
    )
    if level is None:
        return None

    _, target_items = sample_memory_for_level(
        oracle_memory_base, level, current_turn_idx
    )
    if not target_items:
        return None

    # Build level-specific dialogue context
    cfg = LEVEL_CONFIG[level]
    ctx_window = cfg.get("context_window", 10)

    if full_dialogue_history is not None:
        level_context = "\n".join(full_dialogue_history[-ctx_window:])
    else:
        level_context = dialogue_context

    MAX_RETRIES = 2
    prompt = build_examiner_prompt(level_context, target_items, level, language)

    for attempt in range(MAX_RETRIES + 1):
        try:
            temp = 0.7 + attempt * 0.15  # increase randomness on retry
            result = query_llm(model_name=model, messages=prompt, temperature=min(temp, 1.0))
            parsed = safe_parse_json(result.get("response", ""))
            if not isinstance(parsed, dict) or "question" not in parsed or "answer" not in parsed:
                continue

            q_text = parsed["question"]
            a_text = parsed["answer"]

            if _check_answer_leakage(q_text, a_text, target_items):
                if attempt < MAX_RETRIES:
                    continue
                # Last attempt still leaked — skip this question entirely
                return None

            # Build source annotation with temporal distance (half-life info)
            source_by_layer: Dict[str, List[str]] = {}
            ages = []
            for it in target_items:
                layer_name = it.get("layer", "working")
                source_by_layer.setdefault(layer_name, []).append(it["id"])
                item_turn = it.get("turn_idx", it.get("metadata", {}).get("turn_idx", 0))
                ages.append(current_turn_idx - item_turn)
            source = [{"layer": l, "ids": ids} for l, ids in source_by_layer.items()]

            cfg_out = LEVEL_CONFIG[level]
            return {
                "difficulty": level,
                "type": cfg_out["type"],
                "question": q_text,
                "answer": a_text,
                "source": source,
                "memory_age_turns": ages,
            }
        except Exception as e:
            print(f"  [generate_memory_query] LLM error (attempt {attempt+1}): {e}")
            if attempt == MAX_RETRIES:
                return None
    return None


# ── Cache helpers ────────────────────────────────────────────────────

def get_cache_dir(output_path: str) -> str:
    base = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(os.path.dirname(output_path), f".{base}_cache")


def save_cache(cache_dir: str, patient_id: str, data: Dict[str, Any]):
    os.makedirs(cache_dir, exist_ok=True)
    safe_id = hashlib.md5(str(patient_id).encode()).hexdigest()[:16]
    tmp_path = os.path.join(cache_dir, f"patient_{safe_id}.tmp.json")
    final_path = os.path.join(cache_dir, f"patient_{safe_id}.json")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


def load_cache(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    cached = {}
    if not os.path.isdir(cache_dir):
        return cached
    for fname in os.listdir(cache_dir):
        if fname.endswith('.json') and not fname.endswith('.tmp.json'):
            fpath = os.path.join(cache_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pid = data.get("patient_id", fname)
                cached[str(pid)] = data
            except Exception:
                pass
    return cached


# ── Main processing function (per-dialogue) ─────────────────────────

def process_single_dialogue(
    dialogue_id: str,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    language: str = "zh",
    consultation_separator: str = "---诊疗分割线---",
) -> Dict[str, Any]:
    """
    Process a single dialogue through the full two-stage pipeline.

    Args:
        dialogue_id: unique identifier
        messages: list of {"role": "doctor"/"patient", "content": "..."}
                  (normalized format)
        model: LLM model name
        language: "zh" or "en"
        consultation_separator: separator between consultations (if any)

    Returns:
        Patient-level dict matching the CareCall output format.
    """
    memory_mgr = LightMemoryManager()
    output_messages = []
    dialogue_history: List[str] = []

    total_msgs = len(messages)
    session_idx = 0

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        memory_mgr.set_turn_idx(msg_idx)

        # Check for consultation separator
        if consultation_separator and consultation_separator in content:
            session_idx += 1
            memory_mgr.reset_working_memory()
            output_messages.append({
                "assistant" if role in ("doctor", "assistant") else "user": {
                    "content": content,
                    "to_memory": [None],
                    "oracle_memory_base": memory_mgr.get_memory_state(),
                }
            })
            dialogue_history.append(f"{'Doctor' if role in ('doctor', 'assistant') else 'Patient'}: {content}")
            pct = (msg_idx + 1) * 100 // total_msgs
            print(f"\r    [{pct:3d}%] msg {msg_idx+1}/{total_msgs} "
                  f"| session {session_idx+1} | ── separator ──",
                  end="", flush=True)
            continue

        out_role = "assistant" if role in ("doctor", "assistant") else "user"
        role_label = "doctor" if out_role == "assistant" else "patient"

        pct = (msg_idx + 1) * 100 // total_msgs
        snippet = content[:30].replace('\n', ' ')
        print(f"\r    [{pct:3d}%] msg {msg_idx+1}/{total_msgs} "
              f"| session {session_idx+1} | {role_label}: {snippet}...",
              end="", flush=True)

        # ── Stage 1: Memory Preprocess ──
        to_memory = extract_to_memory(content, role, dialogue_history, model, language)
        generate_and_execute_memory_ops(to_memory, memory_mgr, model, language)
        oracle_memory_base = memory_mgr.get_memory_state()

        # ── Stage 2: Examiner Agent (level-specific context) ──
        current_line = f"{'Doctor' if out_role == 'assistant' else 'Patient'}: {content}"
        dialogue_history.append(current_line)

        memory_query = generate_memory_query(
            dialogue_context="\n".join(dialogue_history[-10:]),
            oracle_memory_base=oracle_memory_base,
            to_memory_items=to_memory,
            model=model,
            language=language,
            current_turn_idx=msg_idx,
            full_dialogue_history=dialogue_history,
        )

        # Build output entry
        entry: Dict[str, Any] = {
            "content": content,
            "to_memory": to_memory,
            "oracle_memory_base": oracle_memory_base,
        }
        if memory_query is not None:
            entry["memory_query"] = memory_query

        output_messages.append({out_role: entry})

    print()  # newline after progress bar
    return {
        "patient_id": str(dialogue_id),
        "messages": output_messages,
    }


# ── Base class for dataset-specific examiner agents ──────────────────

class BaseExaminerAgent:
    """
    Abstract base class. Subclass and implement:
      - load_raw_data(input_path) -> list of raw records
      - normalize_dialogue(raw_record) -> (dialogue_id, messages, metadata)
      - grouping_key_fn(normalised_record) -> str  (for multi-session grouping)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str = "zh",
        consultation_separator: str = "---诊疗分割线---",
        max_workers: int = 4,
        # ── Multi-session synthesis settings ──
        enable_multi_session: bool = True,
        sessions_range: Tuple[int, int] = (2, 5),
        use_llm_profile: bool = False,
    ):
        self.model = model
        self.language = language
        self.consultation_separator = consultation_separator
        self.max_workers = max_workers
        self.enable_multi_session = enable_multi_session
        self.sessions_range = sessions_range
        self.use_llm_profile = use_llm_profile

    # ── abstract methods ─────────────────────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """Load raw dataset and return list of records."""
        raise NotImplementedError

    def normalize_dialogue(self, raw_record: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Convert a raw record into normalized format.
        Returns:
            (dialogue_id, messages, extra_metadata)
        where messages = [{"role": "doctor"/"patient", "content": "..."}]
        """
        raise NotImplementedError

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Return a grouping key for multi-session synthesis.
        Records with the same key will be candidates for merging into
        the same patient's multi-session record.

        Default: group by metadata['department'] or 'default'.
        Subclasses should override for dataset-specific grouping.
        """
        meta = normalised_record.get("metadata", {})
        return meta.get("department", meta.get("category", "default"))

    # ── multi-session synthesis ───────────────────────────────────────

    def _normalize_all(self, raw_data: List[Any]) -> List[Dict[str, Any]]:
        """Normalize all raw records into a flat list of standardised dicts."""
        normalised = []
        for idx, record in enumerate(raw_data):
            try:
                dialogue_id, messages, meta = self.normalize_dialogue(record)
                normalised.append({
                    "dialogue_id": str(dialogue_id),
                    "messages": messages,
                    "metadata": meta or {},
                })
            except Exception as e:
                print(f"  [SKIP] Record {idx}: normalize error: {e}")
        return normalised

    def _synthesize_multi_session(
        self, normalised: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group normalised single-session dialogues into multi-session patient
        records using the MultiSessionSynthesizer.

        Returns list of {"patient_id", "scenario", "metadata", "messages"}.
        """
        return synthesize_multi_session_patients(
            normalised_dialogues=normalised,
            key_fn=self.grouping_key_fn,
            sessions_range=self.sessions_range,
            use_llm_profile=self.use_llm_profile,
            model=self.model,
            language=self.language,
        )

    # ── main entry ───────────────────────────────────────────────────

    def run(
        self,
        input_path: str,
        output_path: str,
        max_dialogues: Optional[int] = None,
    ):
        """
        Full pipeline: load → normalize → (multi-session synthesis) → process → save.
        """
        print(f"\n{'='*60}")
        print(f"[{self.__class__.__name__}] Starting processing")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Model:  {self.model}")
        print(f"  Multi-session: {self.enable_multi_session}")
        if self.enable_multi_session:
            print(f"  Sessions/patient: {self.sessions_range}")
        print(f"{'='*60}\n")

        # Load cache
        cache_dir = get_cache_dir(output_path)
        cached = load_cache(cache_dir)
        print(f"Loaded {len(cached)} cached dialogues from {cache_dir}")

        # Load raw data
        raw_data = self.load_raw_data(input_path)
        if max_dialogues:
            raw_data = raw_data[:max_dialogues]
        print(f"Loaded {len(raw_data)} raw records")

        # Normalize all
        normalised = self._normalize_all(raw_data)
        print(f"Normalised {len(normalised)} dialogues")

        # ── Multi-session synthesis (or pass-through) ──
        if self.enable_multi_session:
            patients = self._synthesize_multi_session(normalised)
            print(f"Synthesized {len(patients)} multi-session patients")
        else:
            # No synthesis: each normalised dialogue becomes one "patient"
            patients = []
            for rec in normalised:
                patients.append({
                    "patient_id": rec["dialogue_id"],
                    "scenario": "single_session",
                    "metadata": rec.get("metadata", {}),
                    "messages": rec["messages"],
                })

        # Process each patient through the two-stage pipeline
        results = []
        total_patients = len(patients)
        cached_count = 0
        pipeline_start = time.time()

        print(f"\n{'─'*60}")
        print(f"Processing {total_patients} patients through 2-stage pipeline")
        print(f"{'─'*60}")

        for idx, patient in enumerate(patients):
            pid = patient["patient_id"]
            if str(pid) in cached:
                results.append(cached[str(pid)])
                cached_count += 1
                continue

            n_msgs = len(patient.get("messages", []))
            scenario = patient.get("scenario", "?")
            elapsed = time.time() - pipeline_start
            eta_str = ""
            processed_so_far = idx - cached_count
            if processed_so_far > 0:
                avg_time = elapsed / processed_so_far
                remaining = (total_patients - idx) * avg_time
                eta_min = remaining / 60
                eta_str = f" | ETA {eta_min:.1f}min"

            print(f"\n  ▶ Patient {idx+1}/{total_patients} [{pid}]"
                  f" | {n_msgs} msgs | scenario={scenario}{eta_str}")

            try:
                patient_start = time.time()
                patient_data = process_single_dialogue(
                    dialogue_id=pid,
                    messages=patient["messages"],
                    model=self.model,
                    language=self.language,
                    consultation_separator=self.consultation_separator,
                )
                patient_data["metadata"] = patient.get("metadata", {})
                patient_data["metadata"]["scenario"] = patient.get("scenario", "")

                results.append(patient_data)
                save_cache(cache_dir, pid, patient_data)
                patient_elapsed = time.time() - patient_start
                out_msgs = len(patient_data.get("messages", []))
                print(f"    ✓ Done: {out_msgs} output messages in {patient_elapsed:.1f}s")

            except Exception as e:
                print(f"    ✗ Error processing patient {pid}: {e}")
                traceback.print_exc()
                continue

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'─'*60}")
        print(f"Pipeline complete: {len(results)}/{total_patients} patients "
              f"({cached_count} from cache) in {total_elapsed:.1f}s")
        print(f"{'─'*60}")

        # Save final output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved {len(results)} patients to {output_path}")
