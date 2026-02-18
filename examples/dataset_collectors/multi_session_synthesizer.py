"""
Multi-Session Synthesizer
==========================

Converts a collection of *independent* single-session dialogues into
*multi-session patient records* — i.e. dialogues that look like the same
patient visiting the hospital multiple times, separated by "---诊疗分割线---".

Design:
  1. **Group**: Cluster raw dialogues by disease / department / topic so that
     dialogues in the same group are medically related.
  2. **Assign scenario**: Pick one of the 5 clinical scenarios for each group.
  3. **Synthesize patient profile**: Use LLM to generate a coherent patient
     identity that ties the sessions together.
  4. **Merge**: Concatenate sessions with "---诊疗分割线---" separator, optionally
     inserting a brief LLM-generated bridging sentence (e.g. "两周后复诊").

This module is used by all five dataset-specific examiners.
"""

import os
import sys
import json
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Agent-R1/
sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset_collectors.clinical_scenarios import (
    CLINICAL_SCENARIOS,
    SCENARIO_KEYS,
    pick_scenario_for_record,
    get_scenario_distribution_prompt,
)

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o-2024-11-20"
CONSULTATION_SEPARATOR = "---诊疗分割线---"

# How many single-session dialogues to merge into one multi-session record
DEFAULT_SESSIONS_PER_PATIENT = (2, 5)  # (min, max)


# ── Grouping helpers ──────────────────────────────────────────────────

def group_dialogues_by_key(
    dialogues: List[Dict[str, Any]],
    key_fn,
    sessions_range: Tuple[int, int] = DEFAULT_SESSIONS_PER_PATIENT,
) -> List[Dict[str, Any]]:
    """
    Group a flat list of normalised dialogues into multi-session patient records.

    Args:
        dialogues: list of {"dialogue_id": str, "messages": [...], "metadata": {...}}
        key_fn: callable(record) -> str  — returns a grouping key (e.g. department)
        sessions_range: (min_sessions, max_sessions) per patient

    Returns:
        list of multi-session patient dicts, each containing:
          {
            "patient_id": str,
            "scenario": str,         # clinical scenario key
            "sessions": [             # list of original dialogues
              {"dialogue_id": ..., "messages": [...], "metadata": {...}},
              ...
            ],
          }
    """
    # 1. Bucket by key
    buckets: Dict[str, List[Dict]] = {}
    for dlg in dialogues:
        k = key_fn(dlg)
        buckets.setdefault(k, []).append(dlg)

    # 2. Within each bucket, chunk into patient groups
    patients = []
    for group_key, group_items in buckets.items():
        random.shuffle(group_items)
        min_s, max_s = sessions_range
        i = 0
        while i < len(group_items):
            n = random.randint(min_s, max_s)
            chunk = group_items[i : i + n]
            if len(chunk) < min_s and patients:
                # Too few left — append to previous patient in same group
                patients[-1]["sessions"].extend(chunk)
            elif len(chunk) >= 1:
                # Determine scenario
                first_meta = chunk[0].get("metadata", {})
                scenario = pick_scenario_for_record(
                    department=first_meta.get("department", ""),
                    disease=first_meta.get("diagnosis", first_meta.get("disease", "")),
                    text_hint=group_key,
                )
                pid = hashlib.md5(
                    f"{group_key}_{i}_{random.random()}".encode()
                ).hexdigest()[:12]
                patients.append({
                    "patient_id": pid,
                    "scenario": scenario,
                    "group_key": group_key,
                    "sessions": chunk,
                })
            i += n

    random.shuffle(patients)
    return patients


def merge_sessions_into_messages(
    sessions: List[Dict[str, Any]],
    separator: str = CONSULTATION_SEPARATOR,
) -> List[Dict[str, str]]:
    """
    Merge multiple session dialogues into a single message list,
    inserting "---诊疗分割线---" between sessions.

    Args:
        sessions: list of {"messages": [{"role": ..., "content": ...}], ...}
        separator: text to insert between sessions

    Returns:
        Flat list of {"role": ..., "content": ...} with separator entries.
    """
    merged: List[Dict[str, str]] = []
    for idx, session in enumerate(sessions):
        if idx > 0:
            merged.append({"role": "assistant", "content": separator})
        msgs = session.get("messages", [])
        merged.extend(msgs)
    return merged


# ── LLM-based patient profile generation (optional enrichment) ────────

def build_patient_profile_prompt(
    scenario_key: str,
    session_summaries: List[str],
    language: str = "zh",
) -> str:
    """
    Build a prompt that asks the LLM to generate a coherent patient identity
    tying multiple sessions together under a clinical scenario.
    """
    scenario_desc = get_scenario_distribution_prompt(scenario_key, language)
    summaries_text = "\n".join(
        f"  第{i+1}次就诊摘要: {s}" for i, s in enumerate(session_summaries)
    )

    if language == "zh":
        return f"""你是一个医疗数据合成专家。请根据以下临床场景和多次就诊摘要，生成一个统一的患者基本信息描述，
使得这些就诊记录看起来是同一个患者在不同时间点的多次就诊。

## 临床场景
{scenario_desc}

## 各次就诊摘要
{summaries_text}

## 输出要求
请输出一个JSON对象，包含以下字段:
{{
  "patient_name": "患者姓名（虚构）",
  "age": "年龄",
  "gender": "性别",
  "chief_complaint": "主诉（概括性描述）",
  "medical_history": "既往病史摘要",
  "identity_summary": "一句话患者身份描述，用于插入到对话开头"
}}

请直接输出JSON对象。"""
    else:
        return f"""You are a medical data synthesis expert. Generate a coherent patient profile
that ties these visit summaries together under the given clinical scenario.

## Clinical Scenario
{scenario_desc}

## Visit Summaries
{summaries_text}

## Output
Output a JSON object:
{{
  "patient_name": "patient name (fictional)",
  "age": "age",
  "gender": "gender",
  "chief_complaint": "chief complaint summary",
  "medical_history": "past medical history summary",
  "identity_summary": "one-sentence patient identity for dialogue header"
}}

Output only the JSON object."""


def build_bridge_sentence_prompt(
    prev_session_summary: str,
    next_session_summary: str,
    scenario_key: str,
    language: str = "zh",
) -> str:
    """Build prompt for generating a brief bridging sentence between sessions."""
    scenario_name = CLINICAL_SCENARIOS[scenario_key]["name_cn" if language == "zh" else "name_en"]

    if language == "zh":
        return f"""请为以下两次就诊之间生成一句简短的过渡描述（如"两周后，患者因...再次就诊"）。

临床场景: {scenario_name}
上次就诊: {prev_session_summary}
本次就诊: {next_session_summary}

请直接输出一句话，不要添加其他内容。"""
    else:
        return f"""Generate a brief bridging sentence between two visits (e.g. "Two weeks later, the patient returned for...").

Scenario: {scenario_name}
Previous visit: {prev_session_summary}
Current visit: {next_session_summary}

Output only the bridging sentence."""


def summarize_session(messages: List[Dict[str, str]], max_chars: int = 200) -> str:
    """Create a brief text summary of a session's messages for prompting."""
    texts = []
    for msg in messages[:6]:  # First 6 turns
        role = msg.get("role", "?")
        content = msg.get("content", "")[:80]
        texts.append(f"{role}: {content}")
    summary = " | ".join(texts)
    return summary[:max_chars]


# ── High-level API ────────────────────────────────────────────────────

def synthesize_multi_session_patients(
    normalised_dialogues: List[Dict[str, Any]],
    key_fn,
    sessions_range: Tuple[int, int] = DEFAULT_SESSIONS_PER_PATIENT,
    use_llm_profile: bool = False,
    model: str = DEFAULT_MODEL,
    language: str = "zh",
) -> List[Dict[str, Any]]:
    """
    Top-level function: takes a flat list of normalised single-session dialogues
    and returns a list of multi-session patient records ready for the examiner
    pipeline.

    Each returned record has:
      {
        "patient_id": str,
        "scenario": str,
        "metadata": {...},
        "messages": [...]   # merged with ---诊疗分割线--- separators
      }

    Args:
        normalised_dialogues: list of
            {"dialogue_id": str, "messages": [{"role":..,"content":..}], "metadata": {}}
        key_fn: grouping key function
        sessions_range: (min, max) sessions per patient
        use_llm_profile: whether to call LLM to generate patient profiles
        model: LLM model name
        language: "zh" or "en"
    """
    # Step 1: Group
    patient_groups = group_dialogues_by_key(
        normalised_dialogues, key_fn, sessions_range
    )
    print(f"  [MultiSessionSynth] Grouped {len(normalised_dialogues)} dialogues "
          f"into {len(patient_groups)} multi-session patients")

    # Step 2: Merge sessions
    results = []
    for pg in patient_groups:
        sessions = pg["sessions"]
        scenario = pg["scenario"]
        patient_id = pg["patient_id"]

        # Merge messages with separator
        merged_messages = merge_sessions_into_messages(sessions)

        # Collect metadata
        meta = {
            "scenario": scenario,
            "scenario_name": CLINICAL_SCENARIOS[scenario]["name_cn"],
            "session_count": len(sessions),
            "group_key": pg.get("group_key", ""),
        }
        # Merge metadata from individual sessions
        for i, sess in enumerate(sessions):
            sess_meta = sess.get("metadata", {})
            for k, v in sess_meta.items():
                meta[f"session_{i}_{k}"] = v

        # Optional: LLM patient profile
        if use_llm_profile and len(sessions) >= 2:
            try:
                from agent_r1.utils.llm import query_llm
                from examples.dataset_collectors.base_examiner import safe_parse_json

                summaries = [summarize_session(s.get("messages", [])) for s in sessions]
                prompt = build_patient_profile_prompt(scenario, summaries, language)
                resp = query_llm(model_name=model, messages=prompt, temperature=0.7)
                profile = safe_parse_json(resp.get("response", ""))
                if isinstance(profile, dict):
                    meta["patient_profile"] = profile
                    # Insert identity message at the beginning
                    identity_text = profile.get("identity_summary", "")
                    if identity_text:
                        merged_messages.insert(0, {
                            "role": "patient",
                            "content": identity_text,
                        })
            except Exception as e:
                print(f"    [profile] LLM error for patient {patient_id}: {e}")

        results.append({
            "patient_id": patient_id,
            "scenario": scenario,
            "metadata": meta,
            "messages": merged_messages,
        })

    return results

