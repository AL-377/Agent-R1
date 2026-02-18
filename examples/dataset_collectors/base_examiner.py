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
        "prompt_guidance": (
            "Generate a question that tests whether the model can track the CURRENT state "
            "of the dialogue. Focus on information just mentioned or recently updated in the "
            "working memory. The question should be answerable from a single working memory item."
        ),
        "prompt_guidance_cn": (
            "生成一个测试模型是否能追踪当前对话状态的问题。聚焦于工作记忆中刚提到或最近更新的信息。"
            "问题应该能从单个工作记忆条目中回答。"
        ),
    },
    "Level 2": {
        "type": "Accurate Retrieval",
        "type_cn": "精确检索",
        "description": "Retrieve specific factual information from long-term memory.",
        "sampling": {"primary": ["identity", "history", "experience"], "count": 1},
        "prompt_guidance": (
            "Generate a question that tests accurate retrieval of a specific fact from "
            "long-term memory (identity, history, or experience). The question should require "
            "precise recall of stored information."
        ),
        "prompt_guidance_cn": (
            "生成一个测试从长期记忆（身份、病史或经验）中精确检索特定事实的问题。"
            "问题应该需要精确回忆已存储的信息。"
        ),
    },
    "Level 3": {
        "type": "Long-range Understanding",
        "type_cn": "长程理解",
        "description": "Cross-temporal association between history and working memory.",
        "sampling": {"primary": ["history", "working"], "count": 2},
        "prompt_guidance": (
            "Generate a question that requires connecting information across different time "
            "points — combining historical medical records with current dialogue state. "
            "The question should test long-range reasoning ability."
        ),
        "prompt_guidance_cn": (
            "生成一个需要跨时间点关联信息的问题——将历史医疗记录与当前对话状态结合。"
            "问题应该测试长程推理能力。"
        ),
    },
    "Level 4": {
        "type": "Conflict & Safety Check",
        "type_cn": "冲突与安全检查",
        "description": "Detect conflicts or safety issues between memory layers.",
        "sampling": {"primary": ["working", "identity", "history"], "count": 2},
        "prompt_guidance": (
            "Generate a question about potential conflicts or safety issues. For example: "
            "drug allergy conflicts, contradictory diagnoses, unsafe treatment combinations. "
            "The question should test the model's ability to detect dangerous inconsistencies."
        ),
        "prompt_guidance_cn": (
            "生成一个关于潜在冲突或安全问题的问题。例如：药物过敏冲突、矛盾诊断、不安全的治疗组合。"
            "问题应该测试模型检测危险不一致性的能力。"
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
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


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

    # ── core operations ──────────────────────────────────────────────

    def insert(self, layer: str, content: str, metadata: Dict[str, Any] = None) -> str:
        assert layer in MEMORY_LAYERS, f"Invalid layer: {layer}"
        mem_id = str(uuid.uuid4())
        item = LightMemoryItem(
            id=mem_id, layer=layer, content=content,
            timestamp=time.time(), metadata=metadata or {},
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
        return f"""你是一个医疗对话信息提取专家。请从当前消息中提取需要记忆的关键医疗信息。

## 对话历史（最近几轮）
{history_text}

## 当前消息（{role}）
{current_content}

## 记忆层说明
- working: 当前对话关键信息（临时，会话特定）
- identity: 患者身份和基础信息（永久特征）
- history: 历史诊断和医疗记录（过去医疗事件）
- experience: 临床经验和案例（通用医学知识）

## 输出要求
请输出一个JSON数组，每个元素是一个字符串，格式为 "层名: 内容"。
如果当前消息没有需要记忆的信息，输出 [null]。

示例输出:
["identity: 患者姓名张三，男，45岁", "history: 患者有2型糖尿病史5年", "working: 患者主诉头痛3天"]

或无信息时:
[null]

请直接输出JSON数组，不要添加其他内容。"""
    else:
        return f"""You are a medical dialogue information extraction expert. Extract key medical information from the current message that should be memorized.

## Dialogue History (recent turns)
{history_text}

## Current Message ({role})
{current_content}

## Memory Layers
- working: Current dialogue key information (temporary, session-specific)
- identity: Patient identity and basic information (permanent characteristics)
- history: Historical diagnosis and medical records (past medical events)
- experience: Clinical experience and case (general medical knowledge)

## Output Requirements
Output a JSON array where each element is a string in format "layer: content".
If no information needs to be memorized, output [null].

Example:
["identity: Patient name John, male, age 45", "history: Patient has type 2 diabetes for 5 years", "working: Patient reports headache for 3 days"]

Or when no info:
[null]

Output only the JSON array, nothing else."""


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

    items_text = "\n".join(
        f'  - [{it.get("layer", "?")}] {it.get("content", "")}'
        for it in target_memory_items
    )

    if language == "zh":
        return f"""你是一个医疗记忆评测出题专家（Examiner Agent）。请根据以下信息生成一个评测问题和标准答案。

## 对话上下文
{dialogue_context}

## 目标记忆点（答案来源）
{items_text}

## 难度等级: {level}
## 题型: {type_name}

## 出题指导
{guidance}

## 输出要求
请输出一个JSON对象，包含以下字段:
{{
  "question": "评测问题",
  "answer": "标准答案"
}}

要求:
1. 问题必须能从目标记忆点中找到答案
2. 答案必须简洁、准确
3. 问题应符合{level}（{type_name}）的难度要求
4. 使用中文出题

请直接输出JSON对象。"""
    else:
        return f"""You are a medical memory evaluation expert (Examiner Agent). Generate an evaluation question and standard answer based on the following information.

## Dialogue Context
{dialogue_context}

## Target Memory Points (answer source)
{items_text}

## Difficulty: {level}
## Type: {type_name}

## Guidance
{guidance}

## Output
Output a JSON object:
{{
  "question": "evaluation question",
  "answer": "standard answer"
}}

Requirements:
1. The question must be answerable from the target memory points
2. The answer must be concise and accurate
3. The question should match {level} ({type_name}) difficulty
4. Use the same language as the dialogue

Output only the JSON object."""


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

def extract_to_memory(
    content: str,
    role: str,
    dialogue_history: List[str],
    model: str = DEFAULT_MODEL,
    language: str = "zh",
) -> List[Optional[str]]:
    """Stage 1 Step 1: Use LLM to extract to_memory items."""
    prompt = build_extract_to_memory_prompt(content, role, dialogue_history, language)
    try:
        result = query_llm(model_name=model, messages=prompt, temperature=0.3)
        parsed = safe_parse_json(result.get("response", ""))
        if isinstance(parsed, list):
            return parsed
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
        result = query_llm(model_name=model, messages=prompt, temperature=0.3)
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
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Stage 2 Step 2: Sample memory points according to level strategy.
    Returns (selected_level, list_of_target_items) or (None, []) if impossible.
    """
    cfg = LEVEL_CONFIG.get(level)
    if cfg is None:
        return None, []

    primary_layers = cfg["sampling"]["primary"]
    count = cfg["sampling"]["count"]

    # Collect candidate items
    candidates = []
    for layer in primary_layers:
        for item in oracle_memory_base.get(layer, []):
            candidates.append({**item, "layer": layer})

    if len(candidates) < count:
        return None, []

    selected = random.sample(candidates, count)
    return level, selected


def decide_trigger_and_level(
    to_memory_items: List[Optional[str]],
    oracle_memory_base: Dict[str, List[Dict[str, Any]]],
) -> Optional[str]:
    """
    Stage 2 Step 1 + 3: Decide whether to trigger and pick a level.
    Returns a level string or None.
    """
    has_new_info = any(item is not None and item.strip() for item in to_memory_items)
    prob = TRIGGER_BOOST_PROB if has_new_info else TRIGGER_BASE_PROB

    if random.random() > prob:
        return None

    # Try levels in random order; pick the first one that has enough memory items
    levels = list(LEVEL_CONFIG.keys())
    random.shuffle(levels)
    for level in levels:
        _, items = sample_memory_for_level(oracle_memory_base, level)
        if items:
            return level
    return None


def generate_memory_query(
    dialogue_context: str,
    oracle_memory_base: Dict[str, List[Dict[str, Any]]],
    to_memory_items: List[Optional[str]],
    model: str = DEFAULT_MODEL,
    language: str = "zh",
) -> Optional[Dict[str, Any]]:
    """
    Stage 2: Full examiner pipeline — trigger → sample → generate Q/A → annotate source.
    Returns memory_query dict or None.
    """
    level = decide_trigger_and_level(to_memory_items, oracle_memory_base)
    if level is None:
        return None

    _, target_items = sample_memory_for_level(oracle_memory_base, level)
    if not target_items:
        return None

    prompt = build_examiner_prompt(dialogue_context, target_items, level, language)
    try:
        result = query_llm(model_name=model, messages=prompt, temperature=0.7)
        parsed = safe_parse_json(result.get("response", ""))
        if not isinstance(parsed, dict) or "question" not in parsed or "answer" not in parsed:
            return None

        # Build source annotation
        source_by_layer: Dict[str, List[str]] = {}
        for it in target_items:
            layer = it.get("layer", "working")
            source_by_layer.setdefault(layer, []).append(it["id"])
        source = [{"layer": l, "ids": ids} for l, ids in source_by_layer.items()]

        cfg = LEVEL_CONFIG[level]
        return {
            "difficulty": level,
            "type": cfg["type"],
            "question": parsed["question"],
            "answer": parsed["answer"],
            "source": source,
        }
    except Exception as e:
        print(f"  [generate_memory_query] LLM error: {e}")
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

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Check for consultation separator
        if consultation_separator and consultation_separator in content:
            memory_mgr.reset_working_memory()
            output_messages.append({
                "assistant" if role in ("doctor", "assistant") else "user": {
                    "content": content,
                    "to_memory": [None],
                    "oracle_memory_base": memory_mgr.get_memory_state(),
                }
            })
            dialogue_history.append(f"{'Doctor' if role in ('doctor', 'assistant') else 'Patient'}: {content}")
            continue

        # Map role to output key
        out_role = "assistant" if role in ("doctor", "assistant") else "user"

        # ── Stage 1: Memory Preprocess ──
        to_memory = extract_to_memory(content, role, dialogue_history, model, language)
        generate_and_execute_memory_ops(to_memory, memory_mgr, model, language)
        oracle_memory_base = memory_mgr.get_memory_state()

        # ── Stage 2: Examiner Agent ──
        # Build dialogue context for examiner
        ctx_lines = list(dialogue_history[-10:])
        ctx_lines.append(f"{'Doctor' if out_role == 'assistant' else 'Patient'}: {content}")
        dialogue_context = "\n".join(ctx_lines)

        memory_query = generate_memory_query(
            dialogue_context, oracle_memory_base, to_memory, model, language
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

        # Update dialogue history
        dialogue_history.append(f"{'Doctor' if out_role == 'assistant' else 'Patient'}: {content}")

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
        for idx, patient in enumerate(patients):
            pid = patient["patient_id"]
            if str(pid) in cached:
                results.append(cached[str(pid)])
                continue

            try:
                print(f"  Processing patient {pid} ({idx+1}/{len(patients)}, "
                      f"scenario={patient.get('scenario', '?')}) ...")
                patient_data = process_single_dialogue(
                    dialogue_id=pid,
                    messages=patient["messages"],
                    model=self.model,
                    language=self.language,
                    consultation_separator=self.consultation_separator,
                )
                # Attach metadata (scenario, session info, etc.)
                patient_data["metadata"] = patient.get("metadata", {})
                patient_data["metadata"]["scenario"] = patient.get("scenario", "")

                results.append(patient_data)
                save_cache(cache_dir, pid, patient_data)
                print(f"    ✓ Done ({len(patient_data.get('messages', []))} messages)")

            except Exception as e:
                print(f"    ✗ Error processing patient {pid}: {e}")
                traceback.print_exc()
                continue

        # Save final output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved {len(results)} patients to {output_path}")
