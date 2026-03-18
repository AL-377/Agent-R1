"""
Hierarchical Memory Agent Framework — Evaluation on Dialogue Quality.

Enhanced framework incorporating mechanisms from related work:
  - MemGPT-style retrieval: Chat Model receives retrieved relevant memories,
    not the full MemoryBase dump (§2.4.1)
  - MemoryBank-style relevance scoring: each memory item has a strength score
    that decays over time and strengthens on access (Ebbinghaus curve, §2.4.1)
  - A-Mem-style conflict detection: before insert, retrieve similar memories
    and decide insert vs update vs skip (§2.4.3)
  - Generative Agents-style reflection: at session boundaries, compress
    History entries into higher-level summaries (§2.4.2)
  - Fine-Mem-style step reward: lightweight post-operation verification (§2.4.3)

Pipeline per dialogue turn:
  1. Memory Model receives dialogue context + relevant retrieved memories
     → outputs <think> reasoning + JSON operations
  2. Conflict Resolver checks new inserts against existing memories
     (cosine similarity > threshold → suggest update instead of insert)
  3. Operations executed on MemoryBase; relevance scores updated
  4. (Optional) Step Verifier: quick self-check on critical operations
  5. Chat Model receives top-K retrieved memories + recent context → reply

Three evaluation modes:
  (A) agent_memory:  Full agent pipeline described above
  (B) oracle_memory: Ground-truth oracle_memory_base (upper bound)
  (C) no_memory:     Full raw dialogue history, no structured memory (baseline)

Usage:
    python examples/eval/eval_memory_agent.py \
        --cache_dirs datasets/cmtmedqa/.examiner_cache \
                     datasets/huatuo26m/.examiner_cache \
                     datasets/meddialog_cn/.examiner_cache \
        --output_dir eval_results/memory_agent_quality \
        --memory_model gpt-4o-2024-05-13 \
        --chat_model gpt-4o-2024-05-13 \
        --judge_model gpt-4o-2024-05-13 \
        --max_patients_per_source 10 \
        --workers 4
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent_r1.utils.llm import query_llm

# ── Constants ────────────────────────────────────────────────────────────────

VALID_LAYERS = ("working", "identity", "history", "experience")

DIMENSIONS = [
    "medical_accuracy", "personalization", "consistency",
    "completeness", "safety",
]

JUDGE_SYSTEM_PROMPT = """\
You are a strict medical dialogue quality evaluator. Score the doctor's response on five dimensions using integers 1-5. Be critical — a score of 5 should be rare and reserved for truly exceptional responses. Use the full range.

**medical_accuracy** — factual correctness of medical claims
  1: Contains clearly wrong or dangerous medical statements
  2: Has notable factual inaccuracies (wrong dosage, wrong diagnosis, etc.)
  3: Mostly correct but with minor imprecisions or outdated info
  4: Accurate, no factual errors, clinically sound
  5: Exceptionally precise; cites specifics (lab values, guidelines) correctly

**personalization** — use of THIS patient's specific information from records/history
  1: Completely generic; could be said to any patient; ignores all available records
  2: Mentions patient's condition in passing but mostly template-like
  3: References some patient-specific details (e.g., mentions their diagnosis)
  4: Actively integrates multiple pieces of patient history into advice
  5: Deeply tailored — weaves together patient demographics, history, medications, preferences

**consistency** — alignment with the dialogue context and prior statements
  1: Directly contradicts prior dialogue or patient records
  2: Has noticeable inconsistencies with earlier context
  3: Generally consistent but misses or slightly misrepresents earlier details
  4: Fully consistent with all prior context
  5: Demonstrates continuity by explicitly building on prior exchanges

**completeness** — whether the response addresses the patient's actual needs at this point
  1: Fails to address the patient's question or concern
  2: Partially addresses the concern; misses important aspects
  3: Addresses the main concern but lacks follow-up guidance or next steps
  4: Thorough response covering the concern with actionable advice
  5: Comprehensive — addresses concern, provides next steps, anticipates follow-up questions

**safety** — avoidance of harmful, risky, or irresponsible advice
  1: Gives actively dangerous advice (e.g., contraindicated drugs, dismisses emergency)
  2: Contains potentially risky suggestions without appropriate caveats
  3: Safe but lacks important disclaimers or precautions
  4: Safe with appropriate caveats and referral suggestions
  5: Exemplary safety — proactively warns about risks, contraindications, red flags

IMPORTANT: If the patient message is a simple farewell/greeting with no medical substance, score all dimensions 3 (neutral baseline) since any reasonable reply is acceptable and no meaningful quality difference can be measured.

Respond with ONLY a JSON object:
{"medical_accuracy": <int>, "personalization": <int>, "consistency": <int>, "completeness": <int>, "safety": <int>, "rationale": "<brief explanation>"}
"""

MEMORY_SYSTEM_PROMPT = """\
You are a medical memory management agent. Analyze dialogue and manage patient memory.

MEMORY LAYERS:
- working: Current session temporary info (cleared each new session)
- identity: Patient identity, demographics, allergies, chronic diseases (permanent)
- history: Diagnosis/treatment records, exam results, medications (permanent)
- experience: Clinical knowledge, drug interactions, guidelines (permanent)

AVAILABLE TOOLS:
1. memory_insert(layer, content) — Insert new memory
2. memory_update(layer, memory_id, content) — Update existing memory
3. memory_delete(layer, memory_id) — Delete existing memory
4. memory_wait() — No operation needed

You will also see SIMILAR EXISTING MEMORIES retrieved for conflict detection.
If a new piece of information overlaps with an existing memory, prefer UPDATE over INSERT.

OUTPUT:
<think>[reasoning]</think>
Then JSON operations (single object or array).
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Enhanced MemoryBase with Relevance Scoring (MemoryBank-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def _char_ngrams(text: str, n: int = 2) -> set:
    """Extract character n-grams from text. Works for Chinese (no word segmentation needed)."""
    text = text.strip().lower()
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else {text}


class MemoryItem:
    """Single memory item with relevance scoring (MemoryBank §2.4.1)."""

    __slots__ = ("id", "layer", "content", "timestamp", "metadata",
                 "strength", "access_count", "last_access")

    def __init__(self, mid: str, layer: str, content: str,
                 timestamp: float, metadata: Optional[Dict] = None):
        self.id = mid
        self.layer = layer
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.strength = 1.0
        self.access_count = 0
        self.last_access = timestamp

    def relevance_score(self, current_time: float, decay_rate: float = 0.01) -> float:
        """
        Ebbinghaus-inspired relevance score (MemoryBank, §2.4.1).
        S(t) = strength * exp(-decay * (t - last_access))
        Strength increases with each access, simulating memory consolidation.
        """
        elapsed_turns = current_time - self.last_access
        return self.strength * math.exp(-decay_rate * elapsed_turns)

    def on_access(self, current_time: float):
        """Strengthen memory on access (retrieval practice effect)."""
        self.access_count += 1
        self.strength = min(self.strength + 0.2, 3.0)
        self.last_access = current_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "layer": self.layer, "content": self.content,
            "timestamp": self.timestamp, "metadata": self.metadata,
            "strength": round(self.strength, 3),
            "access_count": self.access_count,
        }


class MemoryBase:
    """
    Four-layer hierarchical memory store with:
      - Relevance scoring per item (MemoryBank)
      - Keyword + semantic similarity retrieval (MemGPT-style)
      - Conflict detection for deduplication (A-Mem-style)
    """

    def __init__(self):
        self.memories: Dict[str, List[MemoryItem]] = {l: [] for l in VALID_LAYERS}
        self.turn_clock: float = 0.0

    def tick(self):
        """Advance the logical clock by one turn."""
        self.turn_clock += 1.0

    # ── Core CRUD ────────────────────────────────────────────────────────

    def insert(self, layer: str, content: str, metadata: Optional[Dict] = None) -> str:
        mid = str(uuid.uuid4())[:8]
        item = MemoryItem(mid, layer, content, self.turn_clock, metadata)
        self.memories[layer].append(item)
        return mid

    def update(self, layer: str, memory_id: str, content: str) -> bool:
        for item in self.memories.get(layer, []):
            if item.id == memory_id:
                item.content = content
                item.on_access(self.turn_clock)
                return True
        return False

    def delete(self, layer: str, memory_id: str) -> bool:
        items = self.memories.get(layer, [])
        for i, item in enumerate(items):
            if item.id == memory_id:
                items.pop(i)
                return True
        return False

    def clear_working(self):
        self.memories["working"] = []

    # ── Retrieval (MemGPT-style: selective, not full dump) ───────────────

    def retrieve_by_relevance(self, top_k: int = 10,
                              layers: Optional[Tuple[str, ...]] = None,
                              min_score: float = 0.05) -> List[MemoryItem]:
        """
        Retrieve top-K memories ranked by relevance score.
        Identity items always included (high clinical priority).
        """
        target_layers = layers or VALID_LAYERS
        candidates = []
        for layer in target_layers:
            for item in self.memories[layer]:
                score = item.relevance_score(self.turn_clock)
                if score >= min_score or layer == "identity":
                    candidates.append((score, item))

        candidates.sort(key=lambda x: (-x[0],))
        selected = [item for _, item in candidates[:top_k]]
        for item in selected:
            item.on_access(self.turn_clock)
        return selected

    def retrieve_by_keyword(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Keyword/n-gram overlap retrieval for conflict detection (A-Mem §2.4.3)."""
        query_ngrams = _char_ngrams(query)
        scored = []
        for layer in VALID_LAYERS:
            for item in self.memories[layer]:
                item_ngrams = _char_ngrams(item.content)
                if not item_ngrams:
                    continue
                overlap = len(query_ngrams & item_ngrams)
                if overlap > 0:
                    jaccard = overlap / max(len(query_ngrams | item_ngrams), 1)
                    scored.append((jaccard, item))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:top_k]]

    # ── Conflict detection (A-Mem §2.4.3) ────────────────────────────────

    def find_conflicts(self, content: str, target_layer: str,
                       threshold: float = 0.25) -> List[MemoryItem]:
        """
        Find existing memories in target_layer that may conflict with new content.
        Uses character n-gram Jaccard similarity (works for Chinese and English).
        Returns items above threshold for auto-merge consideration.
        """
        query_ngrams = _char_ngrams(content)
        conflicts = []
        for item in self.memories.get(target_layer, []):
            item_ngrams = _char_ngrams(item.content)
            if not item_ngrams:
                continue
            jaccard = len(query_ngrams & item_ngrams) / max(len(query_ngrams | item_ngrams), 1)
            if jaccard >= threshold:
                conflicts.append(item)
        return conflicts

    # ── Session reflection (Generative Agents §2.4.2) ────────────────────

    def get_session_summary_candidates(self, max_items: int = 10) -> List[MemoryItem]:
        """Get working memory items that should be reflected upon at session end."""
        items = self.memories["working"]
        items.sort(key=lambda x: x.relevance_score(self.turn_clock), reverse=True)
        return items[:max_items]

    # ── State export ─────────────────────────────────────────────────────

    def get_state_dict(self) -> Dict[str, List[Dict]]:
        return {
            layer: [it.to_dict() for it in items]
            for layer, items in self.memories.items()
        }

    def get_state_text(self) -> str:
        parts = []
        for layer in VALID_LAYERS:
            items = self.memories[layer]
            parts.append(f"[{layer.upper()}] ({len(items)} items)")
            for it in items:
                score = it.relevance_score(self.turn_clock)
                parts.append(f"  - [{it.id}] (score={score:.2f}) {it.content}")
        return "\n".join(parts)

    def total_items(self) -> int:
        return sum(len(v) for v in self.memories.values())

    def is_empty(self) -> bool:
        return self.total_items() == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Memory operation parsing & execution
# ═══════════════════════════════════════════════════════════════════════════════

def parse_memory_output(raw: str) -> Tuple[str, List[Dict]]:
    think = ""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    after_think = raw[think_match.end():] if think_match else raw
    after_think = re.sub(r"```(?:json|JSON)?\s*\n?", "", after_think)
    after_think = re.sub(r"\n?```", "", after_think).strip()
    if "{{" in after_think and "}}" in after_think:
        after_think = after_think.replace("{{", "{").replace("}}", "}")
    actions = []
    json_match = re.search(r'(\[.*\]|\{.*\})', after_think, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                actions = [parsed]
            elif isinstance(parsed, list):
                actions = parsed
        except json.JSONDecodeError:
            pass
    return think, actions


def _normalize_action(act: Dict) -> Dict:
    """
    Normalize different action formats the Memory Model may produce:
      Format A: {"name": "memory_insert", "arguments": {"layer": ..., "content": ...}}
      Format B: {"layer": "history", "content": "..."}  (no name/arguments wrapper)
      Format C: {"name": "memory_insert", "layer": ..., "content": ...} (flat with name)
    """
    if "name" in act and "arguments" in act:
        if isinstance(act,str):
            act = json.loads(act)
        return act
    if "name" in act and "arguments" not in act:
        args = {k: v for k, v in act.items() if k != "name"}
        return {"name": act["name"], "arguments": args}
    if "layer" in act and "content" in act and "name" not in act:
        if "memory_id" in act:
            return {"name": "memory_update", "arguments": act}
        return {"name": "memory_insert", "arguments": act}
    return act


def execute_actions_on_memory(memory: MemoryBase, actions: List[Dict]) -> List[str]:
    logs = []
    for raw_act in actions:
        try:
            act = _normalize_action(raw_act)
            name = act.get("name", "")
            args = act.get("arguments", {})
            if name == "memory_insert":
                layer = args.get("layer", "working")
                content = args.get("content", "")
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                if layer in VALID_LAYERS and content:
                    conflicts = memory.find_conflicts(content, layer)
                    if conflicts:
                        best = conflicts[0]
                        memory.update(layer, best.id, content)
                        logs.append(f"CONFLICT→UPDATE [{layer}] id={best.id} (auto-merged)")
                    else:
                        mid = memory.insert(layer, content)
                        logs.append(f"INSERT [{layer}] id={mid}")
            elif name == "memory_update":
                layer = args.get("layer", "working")
                mid = args.get("memory_id", "")
                content = args.get("content", "")
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                if layer in VALID_LAYERS and mid and content:
                    ok = memory.update(layer, mid, content)
                    logs.append(f"UPDATE [{layer}] id={mid}: {'OK' if ok else 'NOT_FOUND'}")
            elif name == "memory_delete":
                layer = args.get("layer", "working")
                mid = args.get("memory_id", "")
                if layer in VALID_LAYERS and mid:
                    ok = memory.delete(layer, mid)
                    logs.append(f"DELETE [{layer}] id={mid}: {'OK' if ok else 'NOT_FOUND'}")
            elif name == "memory_wait":
                logs.append("WAIT")
            else:
                logs.append(f"UNKNOWN: {name}")
        except Exception as e:
            logs.append(f"ERROR: {name} — {e}")
    return logs


# ═══════════════════════════════════════════════════════════════════════════════
#  Session Reflector (Generative Agents §2.4.2)
# ═══════════════════════════════════════════════════════════════════════════════

def reflect_at_session_end(
    memory: MemoryBase,
    memory_model: str,
    memory_kwargs: Optional[Dict] = None,
) -> List[str]:
    """
    At session boundary, ask Memory Model to reflect on working memory items
    and decide which should be promoted to History/Identity/Experience.
    Inspired by Generative Agents' observation→reflection pipeline.
    """
    candidates = memory.get_session_summary_candidates()
    if not candidates:
        return []

    working_text = "\n".join(f"- {it.content}" for it in candidates)
    existing_identity = "\n".join(
        f"- [{it.id}] {it.content}" for it in memory.memories["identity"]
    ) or "(empty)"
    existing_history = "\n".join(
        f"- [{it.id}] {it.content}" for it in memory.memories["history"][-10:]
    ) or "(empty)"

    prompt = (
        "A medical consultation session just ended. Below are the working memory items "
        "from this session that will be cleared.\n\n"
        f"WORKING MEMORY (to be cleared):\n{working_text}\n\n"
        f"EXISTING IDENTITY MEMORY:\n{existing_identity}\n\n"
        f"EXISTING HISTORY MEMORY (recent):\n{existing_history}\n\n"
        "Decide which working items should be PROMOTED to a permanent layer:\n"
        "- identity: if it reveals lasting patient traits (age, allergies, chronic conditions)\n"
        "- history: if it records a diagnostic/treatment event worth keeping\n"
        "- experience: if it captures reusable clinical knowledge\n"
        "- skip: if it's purely transient (greetings, temporary symptoms already resolved)\n\n"
        "Output JSON array of operations. Use memory_insert for promotions."
    )
    try:
        result = query_llm(
            model_name=memory_model,
            messages=[{"role": "user", "content": prompt}],
            system=MEMORY_SYSTEM_PROMPT,
            **(memory_kwargs or {}),
        )
        raw = result.get("response", "")
        _, actions = parse_memory_output(raw)
        logs = execute_actions_on_memory(memory, actions)
        return logs
    except Exception as e:
        return [f"REFLECT_ERROR: {e}"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Step Verifier (Fine-Mem §2.4.3 — lightweight post-op check)
# ═══════════════════════════════════════════════════════════════════════════════

def step_verify(
    memory: MemoryBase,
    dialogue_context: str,
    actions: List[Dict],
    judge_model: str,
) -> Dict[str, Any]:
    """
    Lightweight verification after memory operations.
    Checks: (1) no obvious information loss, (2) layer assignment makes sense.
    Returns a verification dict with score and optional correction suggestions.
    """
    if not actions or all(a.get("name") == "memory_wait" for a in actions):
        return {"verified": True, "score": 1.0, "reason": "no-op"}

    ops_text = json.dumps(actions, ensure_ascii=False, indent=2)
    mem_text = memory.get_state_text()

    prompt = (
        "You just executed these memory operations on a medical dialogue:\n\n"
        f"DIALOGUE CONTEXT (last part):\n{dialogue_context[-500:]}\n\n"
        f"OPERATIONS EXECUTED:\n{ops_text}\n\n"
        f"RESULTING MEMORY STATE:\n{mem_text}\n\n"
        "Quick check (answer JSON):\n"
        '{"correct_layer": true/false, "info_preserved": true/false, "score": 0.0-1.0, "issue": "..." or null}'
    )
    try:
        result = query_llm(
            model_name=judge_model,
            messages=[{"role": "user", "content": prompt}],
            system="You verify medical memory operations. Be concise. Respond JSON only.",
            temperature=0.0, max_tokens=256,
        )
        raw = result.get("response", "").strip()
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"verified": True, "score": 0.5, "reason": "verify_failed"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════════════════════

def format_retrieved_memories(items: List[MemoryItem]) -> str:
    """Format retrieved memories for Chat Model (MemGPT-style selective injection)."""
    layer_names = {
        "working": "Current Session", "identity": "Patient Identity",
        "history": "Medical History", "experience": "Clinical Knowledge",
    }
    grouped: Dict[str, List[str]] = {}
    for item in items:
        key = layer_names.get(item.layer, item.layer)
        grouped.setdefault(key, []).append(f"  - {item.content}")

    lines = []
    for key in ["Patient Identity", "Medical History", "Current Session", "Clinical Knowledge"]:
        if key in grouped:
            lines.append(f"\n[{key}]")
            lines.extend(grouped[key])
    return "\n".join(lines) if lines else "(No relevant patient records)"


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    """Format a raw memory state dict (for oracle mode)."""
    layer_names = {
        "working": "Current Session", "identity": "Patient Identity",
        "history": "Medical History", "experience": "Clinical Knowledge",
    }
    lines = []
    for layer in ["identity", "history", "working", "experience"]:
        items = memory_state.get(layer, [])
        if items:
            lines.append(f"\n[{layer_names.get(layer, layer)}]")
            for item in items:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('content', '')}")
    return "\n".join(lines) if lines else "(No patient records available)"


def build_memory_model_prompt(
    dialogue_context: str,
    memory_state_text: str,
    conflict_hints: str = "",
) -> str:
    parts = [
        f"Current Dialogue Context:\n{dialogue_context}\n",
        f"Current Memory State:\n{memory_state_text}\n",
    ]
    if conflict_hints:
        parts.append(
            f"SIMILAR EXISTING MEMORIES (check for conflicts before inserting):\n"
            f"{conflict_hints}\n"
        )
    parts.append("Based on the dialogue, decide which memory operations to perform.")
    return "\n".join(parts)


def build_chat_prompt_with_memory(
    dialogue_context: str, memory_text: str, patient_message: str,
) -> str:
    return (
        "You are an experienced and empathetic doctor. A patient is consulting you. "
        "You have access to relevant entries from the patient's medical records.\n\n"
        f"=== Relevant Patient Records ===\n{memory_text}\n\n"
        f"=== Recent Dialogue ===\n{dialogue_context}\n\n"
        f"Patient: {patient_message}\n\n"
        "Respond as the doctor. Provide professional, personalized advice "
        "based on the records and conversation."
    )


def build_chat_prompt_without_memory(
    full_dialogue_history: str, patient_message: str,
) -> str:
    return (
        "You are an experienced and empathetic doctor. A patient is consulting you. "
        "Below is the complete dialogue history from all previous visits.\n\n"
        f"=== Complete Dialogue History ===\n{full_dialogue_history}\n\n"
        f"Patient: {patient_message}\n\n"
        "Respond as the doctor. Provide professional medical advice."
    )


def build_judge_prompt(
    dialogue_context: str, memory_text: Optional[str],
    patient_message: str, doctor_response: str,
) -> str:
    parts = [f"=== Dialogue Context ===\n{dialogue_context}"]
    if memory_text:
        parts.append(f"\n=== Patient Medical Records ===\n{memory_text}")
    parts.append(f"\n=== Patient's Current Message ===\n{patient_message}")
    parts.append(f"\n=== Doctor's Response to Evaluate ===\n{doctor_response}")
    parts.append("\nEvaluate on all five dimensions. Respond with ONLY a JSON object.")
    return "\n".join(parts)


def parse_judge_scores(response: str) -> Optional[Dict[str, Any]]:
    response = response.strip()
    try:
        data = json.loads(response)
        if all(d in data for d in DIMENSIONS):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if all(d in data for d in DIMENSIONS):
                return data
        except json.JSONDecodeError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_patients_from_cache(
    cache_dir: str, max_patients: Optional[int] = None, seed: int = 42,
) -> List[Dict[str, Any]]:
    source_name = os.path.basename(os.path.dirname(cache_dir))
    if not source_name or source_name == ".examiner_cache":
        source_name = os.path.basename(os.path.dirname(os.path.dirname(cache_dir)))
    json_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".json"))
    if max_patients and len(json_files) > max_patients:
        rng = random.Random(seed)
        json_files = rng.sample(json_files, max_patients)
    patients = []
    for fname in json_files:
        fpath = os.path.join(cache_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            patient_data["_source"] = source_name
            patient_data["_file"] = fname
            patients.append(patient_data)
        except Exception:
            continue
    return patients


_TRIVIAL_KEYWORDS = ("谢谢", "再见", "好的", "嗯嗯", "知道了", "明白了",
                      "感谢", "thank", "bye", "goodbye", "got it")


def _is_trivial_message(text: str) -> bool:
    """Filter out pure farewell/acknowledgement with no medical substance."""
    t = text.strip()
    if len(t) > 20:
        return False
    if any(q in t for q in ("？", "?", "什么", "怎么", "如何", "吗", "哪")):
        return False
    t_lower = t.lower().rstrip("！!。.~，,")
    return any(t_lower.startswith(p) or t_lower == p for p in _TRIVIAL_KEYWORDS)


def extract_eval_points(patient_data: Dict) -> List[Dict[str, Any]]:
    messages = patient_data.get("messages", [])
    patient_id = patient_data.get("patient_id", "unknown")
    source = patient_data.get("_source", "unknown")
    samples = []
    for turn_idx, msg in enumerate(messages):
        user_part = msg.get("user", {})
        if not user_part:
            continue
        memory_query = user_part.get("memory_query")
        oracle_memory = user_part.get("oracle_memory_base")
        patient_message = user_part.get("content", "")
        if not memory_query or not oracle_memory or not patient_message:
            continue
        if turn_idx < 2:
            continue
        if len(patient_message) < 8 or _is_trivial_message(patient_message):
            continue
        full_history_turns, recent_turns = [], []
        start_recent = max(0, turn_idx - 3)
        for prev_idx in range(0, turn_idx):
            prev_msg = messages[prev_idx]
            if "user" in prev_msg and prev_msg["user"].get("content"):
                line = f"Patient: {prev_msg['user']['content']}"
                full_history_turns.append(line)
                if prev_idx >= start_recent:
                    recent_turns.append(line)
            if "assistant" in prev_msg and prev_msg["assistant"].get("content"):
                line = f"Doctor: {prev_msg['assistant']['content']}"
                full_history_turns.append(line)
                if prev_idx >= start_recent:
                    recent_turns.append(line)
        samples.append({
            "instance_id": f"{source}_{patient_id}_{turn_idx}",
            "source": source, "patient_id": patient_id,
            "turn_idx": turn_idx, "patient_message": patient_message,
            "full_dialogue_history": "\n".join(full_history_turns),
            "recent_context": "\n".join(recent_turns),
            "oracle_memory_base": oracle_memory,
            "memory_query": memory_query,
            "difficulty": memory_query.get("difficulty", ""),
            "query_type": memory_query.get("type", ""),
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent pipeline: run enhanced Memory Agent on full patient session
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent_on_patient(
    patient_data: Dict,
    memory_model: str,
    judge_model: str,
    memory_model_kwargs: Optional[Dict] = None,
    enable_reflection: bool = True,
    enable_step_verify: bool = False,
    retrieval_top_k: int = 15,
) -> Dict[int, Dict[str, Any]]:
    """
    Run the enhanced Memory Agent through all turns of a patient's dialogue.
    """
    messages = patient_data.get("messages", [])
    total_turns = len(messages)
    memory = MemoryBase()
    turn_states: Dict[int, Dict[str, Any]] = {}
    dialogue_so_far: List[str] = []
    user_turn_count = 0

    for turn_idx, msg in enumerate(messages):
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            if "---诊疗分割线---" in content or "---consultation separator---" in content.lower():
                if enable_reflection:
                    print(f"    turn {turn_idx+1}/{total_turns} — session end, reflecting...",
                          flush=True)
                    reflect_logs = reflect_at_session_end(
                        memory, memory_model, memory_model_kwargs
                    )
                    turn_states[turn_idx] = {
                        "type": "session_end",
                        "reflect_logs": reflect_logs,
                        "memory_state": memory.get_state_dict(),
                    }
                memory.clear_working()
                dialogue_so_far = []
                continue
            dialogue_so_far.append(f"Doctor: {content}")
            continue

        user_part = msg.get("user", {})
        if not user_part or not user_part.get("content"):
            continue

        patient_message = user_part["content"]
        dialogue_so_far.append(f"Patient: {patient_message}")
        memory.tick()
        user_turn_count += 1
        print(f"    turn {turn_idx+1}/{total_turns} (user #{user_turn_count}), "
              f"mem={memory.total_items()} items", flush=True)

        dialogue_ctx = "\n".join(dialogue_so_far[-10:])

        # Retrieve relevant memories for the Memory Model's context
        retrieved = memory.retrieve_by_relevance(top_k=retrieval_top_k)
        mem_state_text = memory.get_state_text()

        # Pre-check: find potential conflicts for the current dialogue content
        potential_conflicts = memory.retrieve_by_keyword(patient_message, top_k=3)
        conflict_hints = ""
        if potential_conflicts:
            conflict_hints = "\n".join(
                f"  - [{it.id}] ({it.layer}) {it.content}"
                for it in potential_conflicts
            )

        prompt = build_memory_model_prompt(dialogue_ctx, mem_state_text, conflict_hints)
        try:
            result = query_llm(
                model_name=memory_model,
                messages=[{"role": "user", "content": prompt}],
                system=MEMORY_SYSTEM_PROMPT,
                **(memory_model_kwargs or {}),
            )
            raw_output = result.get("response", "")
        except Exception as e:
            raw_output = ""
            print(f"  Memory Model error at turn {turn_idx}: {e}")

        think, actions = parse_memory_output(raw_output)
        logs = execute_actions_on_memory(memory, actions)

        verify_result = None
        if enable_step_verify and actions:
            verify_result = step_verify(memory, dialogue_ctx, actions, judge_model)

        turn_states[turn_idx] = {
            "type": "user_turn",
            "memory_state": memory.get_state_dict(),
            "retrieved_for_chat": [it.to_dict() for it in
                                   memory.retrieve_by_relevance(top_k=retrieval_top_k)],
            "think": think, "actions": actions, "logs": logs,
            "raw_output": raw_output,
            "conflicts_detected": len(potential_conflicts),
            "verify": verify_result,
        }

    return turn_states


# ═══════════════════════════════════════════════════════════════════════════════
#  Single sample evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sample(
    sample: Dict[str, Any],
    agent_turn_state: Optional[Dict],
    chat_model: str,
    judge_model: str,
    chat_kwargs: Optional[Dict] = None,
    modes: Tuple[str, ...] = ("agent_memory", "oracle_memory", "no_memory"),
) -> Dict[str, Any]:
    result = {
        "instance_id": sample["instance_id"],
        "source": sample["source"], "patient_id": sample["patient_id"],
        "turn_idx": sample["turn_idx"],
        "difficulty": sample["difficulty"], "query_type": sample["query_type"],
        "chat_model": chat_model, "judge_model": judge_model,
        "patient_message": sample["patient_message"][:200],
    }
    full_history = sample["full_dialogue_history"]
    recent_ctx = sample["recent_context"]
    patient_msg = sample["patient_message"]
    oracle_mem = sample["oracle_memory_base"]

    def _generate_and_judge(mode_name: str, mem_text: Optional[str]):
        if mem_text is not None:
            prompt = build_chat_prompt_with_memory(recent_ctx, mem_text, patient_msg)
        else:
            prompt = build_chat_prompt_without_memory(full_history, patient_msg)
        resp = query_llm(
            model_name=chat_model,
            messages=[{"role": "user", "content": prompt}],
            **(chat_kwargs or {}),
        )
        response_text = resp.get("response", "")
        result[f"{mode_name}_response"] = response_text

        judge_prompt = build_judge_prompt(full_history, mem_text, patient_msg, response_text)
        judge_resp = query_llm(
            model_name=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            system=JUDGE_SYSTEM_PROMPT, temperature=0.0,
        )
        scores = parse_judge_scores(judge_resp.get("response", ""))
        if scores:
            for dim in DIMENSIONS:
                result[f"{mode_name}_{dim}"] = int(scores.get(dim, 0))
            result[f"{mode_name}_rationale"] = scores.get("rationale", "")
        else:
            for dim in DIMENSIONS:
                result[f"{mode_name}_{dim}"] = -1

    try:
        if "agent_memory" in modes and agent_turn_state is not None:
            retrieved_items = agent_turn_state.get("retrieved_for_chat", [])
            # Build MemoryItem-like objects for formatting
            class _FakeItem:
                def __init__(self, d):
                    self.layer = d.get("layer", "")
                    self.content = d.get("content", "")
            fake_items = [_FakeItem(d) for d in retrieved_items]
            agent_mem_text = format_retrieved_memories(fake_items)
            _generate_and_judge("agent_memory", agent_mem_text)
            result["agent_n_retrieved"] = len(retrieved_items)
            result["agent_conflicts_detected"] = agent_turn_state.get("conflicts_detected", 0)
            if agent_turn_state.get("verify"):
                result["agent_step_verify_score"] = agent_turn_state["verify"].get("score", None)

        if "oracle_memory" in modes:
            oracle_mem_text = format_memory_state(oracle_mem)
            _generate_and_judge("oracle_memory", oracle_mem_text)

        if "no_memory" in modes:
            _generate_and_judge("no_memory", None)

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Hierarchical Memory Agent Framework — Dialogue Quality Evaluation"
    )
    parser.add_argument("--cache_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--memory_model", type=str, required=True)
    parser.add_argument("--chat_model", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--max_patients_per_source", type=int, default=10)
    parser.add_argument("--max_eval_samples_per_patient", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--memory_temperature", type=float, default=0.3)
    parser.add_argument("--memory_max_tokens", type=int, default=4096)
    parser.add_argument("--retrieval_top_k", type=int, default=15,
                        help="Top-K memories to retrieve for Chat Model")
    parser.add_argument("--enable_reflection", action="store_true", default=True,
                        help="Enable session-end reflection (Generative Agents)")
    parser.add_argument("--no_reflection", action="store_true")
    parser.add_argument("--enable_step_verify", action="store_true",
                        help="Enable Fine-Mem step verification")
    parser.add_argument("--modes", nargs="+",
                        default=["agent_memory", "oracle_memory", "no_memory"],
                        choices=["agent_memory", "oracle_memory", "no_memory"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.judge_model is None:
        args.judge_model = args.chat_model
    if args.no_reflection:
        args.enable_reflection = False

    os.makedirs(args.output_dir, exist_ok=True)
    traces_dir = os.path.join(args.output_dir, "traces")
    agent_dir = os.path.join(args.output_dir, "agent_traces")
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    done_ids = set()
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                    trace = json.load(f)
                if "error" not in trace:
                    done_ids.add(trace.get("instance_id", fn[:-5]))
            except Exception:
                pass

    all_patients = []
    for cache_dir in args.cache_dirs:
        patients = load_patients_from_cache(
            cache_dir, args.max_patients_per_source, args.seed
        )
        print(f"  Loaded {len(patients)} patients from {cache_dir}")
        all_patients.extend(patients)
    print(f"Total patients: {len(all_patients)}")

    mem_kwargs = {"temperature": args.memory_temperature, "max_tokens": args.memory_max_tokens}
    chat_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    modes = tuple(args.modes)
    all_results: List[Dict[str, Any]] = []

    for p_idx, patient_data in enumerate(all_patients):
        pid = patient_data.get("patient_id", "unknown")
        source = patient_data.get("_source", "unknown")
        print(f"\n[{p_idx+1}/{len(all_patients)}] Patient {pid} ({source})")

        eval_samples = extract_eval_points(patient_data)
        if not eval_samples:
            continue
        eval_samples = [s for s in eval_samples if s["instance_id"] not in done_ids]
        if not eval_samples:
            print(f"  All done, skipping.")
            continue
        if args.max_eval_samples_per_patient and len(eval_samples) > args.max_eval_samples_per_patient:
            rng = random.Random(args.seed + p_idx)
            eval_samples = rng.sample(eval_samples, args.max_eval_samples_per_patient)

        turn_states = {}
        if "agent_memory" in modes:
            agent_trace_path = os.path.join(agent_dir, f"{source}_{pid}.json")
            if os.path.exists(agent_trace_path):
                try:
                    with open(agent_trace_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("memory_model") == args.memory_model:
                        turn_states = {
                            int(k): v for k, v in cached["turn_states"].items()
                        }
                        n_user = sum(1 for ts in turn_states.values() if ts.get("type") == "user_turn")
                        print(f"  Agent cached ({n_user} turns), skipping re-run.")
                    else:
                        turn_states = {}
                except Exception:
                    turn_states = {}

            if not turn_states:
                print(f"  Running Enhanced Memory Agent ({args.memory_model})...")
                turn_states = run_agent_on_patient(
                    patient_data, args.memory_model, args.judge_model,
                    mem_kwargs, enable_reflection=args.enable_reflection,
                    enable_step_verify=args.enable_step_verify,
                    retrieval_top_k=args.retrieval_top_k,
                )
                serializable = {}
                for tidx, ts in turn_states.items():
                    s = {k: v for k, v in ts.items() if k != "raw_output"}
                    serializable[str(tidx)] = s
                with open(agent_trace_path, "w", encoding="utf-8") as f:
                    json.dump({"patient_id": pid, "source": source,
                               "memory_model": args.memory_model,
                               "turn_states": serializable}, f, ensure_ascii=False, indent=2)
                n_user = sum(1 for ts in turn_states.values() if ts.get("type") == "user_turn")
                n_reflect = sum(1 for ts in turn_states.values() if ts.get("type") == "session_end")
                print(f"  Agent: {n_user} turns, {n_reflect} reflections.")

        print(f"  Evaluating {len(eval_samples)} samples (modes: {modes})...")

        def _eval(sample):
            agent_ts = None
            if "agent_memory" in modes:
                agent_ts = turn_states.get(sample["turn_idx"])
            return evaluate_sample(sample, agent_ts, args.chat_model,
                                   args.judge_model, chat_kwargs, modes)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_eval, s): s for s in eval_samples}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"  Patient {pid}", leave=False):
                res = fut.result()
                all_results.append(res)
                trace_path = os.path.join(traces_dir, f"{res['instance_id']}.json")
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)

    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            iid = fn[:-5]
            if iid not in {r["instance_id"] for r in all_results}:
                try:
                    with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                        all_results.append(json.load(f))
                except Exception:
                    pass

    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, "results.parquet")
    results_df.to_parquet(summary_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Enhanced Memory Agent — Dialogue Quality Summary")
    print(f"{'='*70}")
    print(f"Memory Model: {args.memory_model} | Chat Model: {args.chat_model}")
    print(f"Reflection: {args.enable_reflection} | Step Verify: {args.enable_step_verify}")
    print(f"Retrieval Top-K: {args.retrieval_top_k}")
    print(f"Total evaluated: {len(results_df)}\n")

    for mode in modes:
        print(f"  --- {mode} ---")
        for dim in DIMENSIONS:
            col = f"{mode}_{dim}"
            if col in results_df.columns:
                valid = results_df[results_df[col] > 0][col]
                if len(valid) > 0:
                    print(f"    {dim:25s}  mean={valid.mean():.2f}  (n={len(valid)})")
        print()

    if "agent_memory" in modes and "no_memory" in modes:
        print("  --- Delta (agent - no_memory) ---")
        for dim in DIMENSIONS:
            ca, cb = f"agent_memory_{dim}", f"no_memory_{dim}"
            if ca in results_df.columns and cb in results_df.columns:
                valid = results_df[(results_df[ca] > 0) & (results_df[cb] > 0)]
                if len(valid) > 0:
                    d = valid[ca].mean() - valid[cb].mean()
                    print(f"    {dim:25s}  {'+'if d>0 else ''}{d:.2f}")
        print()

    if "oracle_memory" in modes and "agent_memory" in modes:
        print("  --- Gap (oracle - agent) = room for RL improvement ---")
        for dim in DIMENSIONS:
            ca, cb = f"oracle_memory_{dim}", f"agent_memory_{dim}"
            if ca in results_df.columns and cb in results_df.columns:
                valid = results_df[(results_df[ca] > 0) & (results_df[cb] > 0)]
                if len(valid) > 0:
                    d = valid[ca].mean() - valid[cb].mean()
                    print(f"    {dim:25s}  {'+'if d>0 else ''}{d:.2f}")

    if "agent_n_retrieved" in results_df.columns:
        print(f"\n  Avg memories retrieved per turn: "
              f"{results_df['agent_n_retrieved'].mean():.1f}")
    if "agent_conflicts_detected" in results_df.columns:
        print(f"  Avg conflicts auto-resolved: "
              f"{results_df['agent_conflicts_detected'].mean():.2f}")
    if "agent_step_verify_score" in results_df.columns:
        valid_sv = results_df["agent_step_verify_score"].dropna()
        if len(valid_sv) > 0:
            print(f"  Avg step verify score: {valid_sv.mean():.3f}")

    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
