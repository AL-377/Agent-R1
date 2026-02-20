"""
Shared utilities for converting examiner-processed patient data into
swalm RL training format (parquet).

Handles both CareCall-style (bare-string separators in messages list) and
new-dataset-style (dict separators with {"assistant": {"content": "---诊疗分割线---"}}).

Supports three context/prompt splits:
  - pure_summary:   context = all messages from start to current position;
                     prompt = simple summarizer instruction
  - layers_summary: context = min-coverage of source memory items (earliest→latest turn_idx);
                     prompt = original memory-management instruction
  - no_summary:     context = all messages from start to current position;
                     prompt = directly ask the memory_query question (no memory ops)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple


SEPARATOR_MARKER = "---诊疗分割线---"

SPLIT_PURE_SUMMARY = "pure_summary"
SPLIT_LAYERS_SUMMARY = "layers_summary"
SPLIT_NO_SUMMARY = "no_summary"
ALL_SPLITS = [SPLIT_PURE_SUMMARY, SPLIT_LAYERS_SUMMARY, SPLIT_NO_SUMMARY]


# ---------------------------------------------------------------------------
# 1. Split consultations
# ---------------------------------------------------------------------------

def split_consultations(messages: List[Any]) -> List[List[Dict[str, Any]]]:
    consultations: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for msg in messages:
        if _is_separator(msg):
            if current:
                consultations.append(current)
                current = []
            continue
        if isinstance(msg, dict):
            current.append(msg)

    if current:
        consultations.append(current)
    return consultations


def _is_separator(msg: Any) -> bool:
    if isinstance(msg, str):
        return SEPARATOR_MARKER in msg
    if isinstance(msg, dict):
        for _role, body in msg.items():
            if isinstance(body, dict):
                if SEPARATOR_MARKER in body.get("content", ""):
                    return True
            elif isinstance(body, str) and SEPARATOR_MARKER in body:
                return True
    return False


# ---------------------------------------------------------------------------
# 2. Flatten messages (skip separators) with global index tracking
# ---------------------------------------------------------------------------

def flatten_messages(messages: List[Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """Return (global_idx, msg_dict) pairs, skipping separators."""
    result: List[Tuple[int, Dict[str, Any]]] = []
    for i, msg in enumerate(messages):
        if _is_separator(msg):
            continue
        if isinstance(msg, dict):
            result.append((i, msg))
    return result


# ---------------------------------------------------------------------------
# 3. Memory-ID helpers
# ---------------------------------------------------------------------------

def extract_memory_ids_from_source(source: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    ids: List[Tuple[str, str]] = []
    for src in source:
        layer = src.get("layer", "")
        for mem_id in src.get("ids", []):
            ids.append((layer, mem_id))
    return ids


def get_first_time_memory_ids(
    consultation: List[Dict[str, Any]],
    query_position: int,
    source_memory_ids: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    seen = set()
    for i in range(query_position):
        oracle = _get_oracle_memory(consultation[i])
        if oracle:
            for layer, items in oracle.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "id" in item:
                            seen.add((layer, item["id"]))
    return [(l, m) for l, m in source_memory_ids if (l, m) not in seen]


def create_previous_memory(
    oracle_memory_base: Dict[str, Any],
    ids_to_remove: List[Tuple[str, str]],
) -> Dict[str, Any]:
    remove_set = set(ids_to_remove)
    prev: Dict[str, Any] = {}
    for layer, items in oracle_memory_base.items():
        prev[layer] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    if (layer, item["id"]) not in remove_set:
                        prev[layer].append(item.copy())
    return prev


# ---------------------------------------------------------------------------
# 4. Context builders for each split
# ---------------------------------------------------------------------------

def format_dialogue_range(messages_flat: List[Tuple[int, Dict[str, Any]]],
                          start_flat: int, end_flat: int) -> str:
    """Format dialogue from flat index start_flat to end_flat (inclusive)."""
    lines: List[str] = []
    for fi in range(start_flat, end_flat + 1):
        _gidx, msg = messages_flat[fi]
        if "assistant" in msg:
            lines.append(f"Assistant: {msg['assistant'].get('content', '')}")
        elif "user" in msg:
            lines.append(f"User: {msg['user'].get('content', '')}")
    return "\n".join(lines)


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    lines: List[str] = []
    for layer, items in memory_state.items():
        if items:
            lines.append(f"\n{layer.upper()} MEMORY:")
            for item in items:
                lines.append(f"  - [{item.get('id', 'unknown')}] {item.get('content', '')}")
    return "\n".join(lines) if lines else "No memories stored yet."


def _find_source_turn_range(
    oracle_memory_base: Dict[str, Any],
    source_memory_ids: List[Tuple[str, str]],
) -> Tuple[Optional[int], Optional[int]]:
    """Find min and max turn_idx among source memory items."""
    source_set = set(source_memory_ids)
    turn_indices: List[int] = []
    for layer, items in oracle_memory_base.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if (layer, mid) not in source_set:
                continue
            ti = item.get("turn_idx")
            if ti is None:
                ti = (item.get("metadata") or {}).get("turn_idx")
            if ti is not None:
                turn_indices.append(int(ti))
    if not turn_indices:
        return None, None
    return min(turn_indices), max(turn_indices)


def _global_idx_to_flat(messages_flat: List[Tuple[int, Dict[str, Any]]],
                        global_idx: int) -> int:
    """Find the flat index corresponding to a global message index.
    Returns the closest flat index <= global_idx."""
    best = 0
    for fi, (gi, _msg) in enumerate(messages_flat):
        if gi <= global_idx:
            best = fi
        else:
            break
    return best


# ---------------------------------------------------------------------------
# 5. Prompt builders per split
# ---------------------------------------------------------------------------

def build_prompt_original(dialogue_context: str, previous_memory: Dict[str, Any]) -> str:
    memory_text = format_memory_state(previous_memory)
    return (
        "You are an intelligent medical memory management assistant specialized "
        "in healthcare dialogue systems. Your role is to analyze medical consultation "
        "dialogues and manage patient memory information across different layers, "
        "ensuring accurate and organized storage of medical data for effective patient care.\n"
        "---\n"
        f"Current Dialogue Context:\n{dialogue_context}\n\n"
        f"Current Memory State:\n{memory_text}\n\n"
        "Based on the dialogue context and current memory state, decide what memory "
        "operations (if any) are needed.\n\n"
        "MEMORY LAYERS:\n"
        "- working: Current dialogue key information (temporary, session-specific). "
        "Store patient basic info, current symptoms, examination results, diagnosis "
        "conclusions that are relevant to the current consultation session.\n"
        "- identity: Patient identity and basic information layer (permanent patient "
        "characteristics). Store patient name, age, gender, occupation, and other "
        "stable personal information.\n"
        "- history: Historical diagnosis and medical history layer (past medical events). "
        "Store past diagnoses, medical history, previous treatments, chronic conditions, "
        "and historical medical events.\n"
        "- experience: Clinical experience and case layer (general medical knowledge). "
        "Store clinical observations, treatment patterns, and general medical knowledge "
        "that can be referenced for future cases.\n\n"
        "AVAILABLE MEMORY TOOLS:\n"
        "1. memory_insert(layer, content, metadata=None)\n"
        "2. memory_update(layer, memory_id, content, metadata=None)\n"
        "3. memory_delete(layer, memory_id)\n"
        "4. memory_wait()\n\n"
        "OUTPUT FORMAT:\n"
        "<think>\n"
        "[Your thinking about what memory operations are needed]\n"
        "</think>\n\n"
        "After your reasoning, output the memory operation(s) as JSON. You can output:\n"
        '- A single operation: {{"name": "memory_insert", "arguments": '
        '{{"layer": "working", "content": "..."}}}}\n'
        '- Multiple operations as a list: [{{"name": "memory_insert", "arguments": '
        '{{"layer": "working", "content": "..."}}}}, {{"name": "memory_update", '
        '"arguments": {{"layer": "identity", "memory_id": "...", "content": "..."}}}}]\n\n'
        "IMPORTANT GUIDELINES:\n"
        "- For memory_update and memory_delete, you MUST provide a valid memory_id "
        "from the current memory state\n"
        "- The JSON must be valid (single object or array of objects)\n"
        '- The layer must be one of: "working", "identity", "history", or "experience"\n'
        '- If no operation is needed, output: {{"name": "memory_wait", "arguments": {{}}}}\n'
    )


def build_prompt_pure_summary(dialogue_context: str) -> str:
    return (
        "You are a summarizer, you should take down necessary information "
        "from this conversations context.\n"
        "---\n"
        f"Conversation Context:\n{dialogue_context}\n"
    )


def build_prompt_no_summary(dialogue_context: str, memory_query: Dict[str, Any]) -> str:
    question = memory_query.get("question", "")
    return (
        f"Conversation Context:\n{dialogue_context}\n\n"
        f"Question: {question}\n"
    )


# ---------------------------------------------------------------------------
# 6. Core conversion: patient JSON → training samples (all 3 splits)
# ---------------------------------------------------------------------------

def process_patient_data(
    patient_data: Dict[str, Any],
    *,
    data_source_tag: str,
    splits: Optional[List[str]] = None,
    skip_instance_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if splits is None:
        splits = ALL_SPLITS

    patient_id = patient_data.get("patient_id", "unknown")
    messages = patient_data.get("messages", [])
    metadata = patient_data.get("metadata", {})

    consultations = split_consultations(messages)
    messages_flat = flatten_messages(messages)

    # Build a mapping: (consult_idx, local_msg_idx) → flat_idx in messages_flat
    consult_flat_offsets: List[int] = []
    fi = 0
    for consultation in consultations:
        consult_flat_offsets.append(fi)
        fi += len(consultation)

    samples: List[Dict[str, Any]] = []

    for consult_idx, consultation in enumerate(consultations):
        flat_offset = consult_flat_offsets[consult_idx]

        for msg_idx, msg in enumerate(consultation):
            memory_query = _get_memory_query(msg)
            oracle_memory_base = _get_oracle_memory(msg)

            if memory_query is None or oracle_memory_base is None:
                continue

            source = memory_query.get("source", [])
            source_memory_ids = extract_memory_ids_from_source(source)
            first_time_ids = get_first_time_memory_ids(
                consultation, msg_idx, source_memory_ids
            )
            if not first_time_ids:
                continue

            short_key = f"{patient_id}_{consult_idx}_{msg_idx}"
            if skip_instance_ids and short_key in skip_instance_ids:
                continue

            previous_memory = create_previous_memory(oracle_memory_base, first_time_ids)
            cur_flat = flat_offset + msg_idx

            for split in splits:
                if split == SPLIT_PURE_SUMMARY:
                    ctx = format_dialogue_range(messages_flat, 0, cur_flat)
                    prompt_text = build_prompt_pure_summary(ctx)

                elif split == SPLIT_LAYERS_SUMMARY:
                    min_ti, max_ti = _find_source_turn_range(
                        oracle_memory_base, source_memory_ids
                    )
                    if min_ti is not None and max_ti is not None:
                        start_fi = _global_idx_to_flat(messages_flat, min_ti)
                        end_fi = _global_idx_to_flat(messages_flat, max_ti)
                        end_fi = max(end_fi, cur_flat)
                    else:
                        start_fi = flat_offset
                        end_fi = cur_flat
                    ctx = format_dialogue_range(messages_flat, start_fi, end_fi)
                    prompt_text = build_prompt_original(ctx, previous_memory)

                elif split == SPLIT_NO_SUMMARY:
                    ctx = format_dialogue_range(messages_flat, 0, cur_flat)
                    prompt_text = build_prompt_no_summary(ctx, memory_query)

                else:
                    continue

                instance_id = f"{patient_id}_{consult_idx}_{msg_idx}_{uuid.uuid4()}"

                extra_info: Dict[str, Any] = {
                    "agent_class": "swalm.core.agent.medmem_agent::MedMemAgent",
                    "agent_run_params": {"max_iterations": 1},
                    "dataset_id": instance_id,
                    "instance_id": instance_id,
                    "task_spec_class": "swalm.core.task.medmem.task::MedMemTaskSpec",
                    "task_type": "swalm.core.task.medmem.task::run_medmem_task",
                    "custom_data": {
                        "split": split,
                        "patient_id": patient_id,
                        "consultation_idx": consult_idx,
                        "message_idx": msg_idx,
                        "dialogue_context": ctx,
                        "memory_state": json.dumps(previous_memory, ensure_ascii=False),
                        "oracle_memory_base": json.dumps(oracle_memory_base, ensure_ascii=False),
                        "memory_query": memory_query,
                        "supposed_new_memory_things": [
                            {"layer": layer, "id": mem_id}
                            for layer, mem_id in first_time_ids
                        ],
                    },
                    "prompt": [{"role": "user", "content": prompt_text}],
                }

                if metadata:
                    extra_info["custom_data"]["patient_metadata"] = metadata

                sample = {
                    "data_source": data_source_tag,
                    "ability": "swalm_env",
                    "reward_model": {
                        "style": "swalm_agent_verifier",
                        "ground_truth": memory_query.get("answer", ""),
                    },
                    "extra_info": extra_info,
                    "prompt": [],
                }
                samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# 7. Dataset-level runner
# ---------------------------------------------------------------------------

def process_dataset(
    input_dir: str,
    output_file: str,
    *,
    data_source_tag: str,
    splits: Optional[List[str]] = None,
    max_patients: Optional[int] = None,
    skip_instance_ids: Optional[set] = None,
):
    import pandas as pd

    if splits is None:
        splits = ALL_SPLITS

    patient_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".json") and f.startswith("patient_")
    )
    if max_patients:
        patient_files = patient_files[:max_patients]

    print(f"[{data_source_tag}] Processing {len(patient_files)} patient files "
          f"from {input_dir} (splits={splits})")

    all_samples: List[Dict[str, Any]] = []
    for pf in patient_files:
        path = os.path.join(input_dir, pf)
        try:
            with open(path, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            samples = process_patient_data(
                patient_data,
                data_source_tag=data_source_tag,
                splits=splits,
                skip_instance_ids=skip_instance_ids,
            )
            all_samples.extend(samples)
            print(f"  {pf}: {len(samples)} samples")
        except Exception as exc:
            print(f"  ERROR {pf}: {exc}")

    if all_samples:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df = pd.DataFrame(all_samples)
        df.to_parquet(output_file, index=False, engine="pyarrow", row_group_size=4)
        print(f"[{data_source_tag}] Saved {len(all_samples)} samples → {output_file}")
    else:
        print(f"[{data_source_tag}] No samples generated!")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_oracle_memory(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for role in ("user", "assistant"):
        body = msg.get(role)
        if isinstance(body, dict) and "oracle_memory_base" in body:
            return body["oracle_memory_base"]
    return None


def _get_memory_query(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for role in ("user", "assistant"):
        body = msg.get(role)
        if isinstance(body, dict) and body.get("memory_query"):
            return body["memory_query"]
    return None
