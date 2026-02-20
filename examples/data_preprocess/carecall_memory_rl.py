"""
Convert CareCall data into swalm + alpha-seed RL format for memory tasks.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import uuid
import tqdm
import glob

def filter_wait_yes_samples(dir:str) -> list:
    res_files = glob.glob(os.path.join(dir, "*.json"))
    false_positives = []
    for res_file in tqdm.tqdm(res_files):
        score = json.load(open(res_file, 'r'))["eval_result"]["score"]
        if score == 0:
            continue
        if "memory_wait" in json.load(open(res_file, 'r'))["eval_result"]["raw_result"]["operations"]:
            false_positives.append(os.path.basename(res_file).split(".")[0])

    return false_positives

false_negatives = filter_wait_yes_samples("/opt/tiger/swalm_agent/debug/run_medmem_task/traces_gpt4o")
print(f"Len False negatives: {len(false_negatives)}")

def split_consultations(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    consultations = []
    current_consultation = []
    for msg in messages:
        is_separator = False
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            if content.strip() == "---诊疗分割线---" or "---诊疗分割线---" in content:
                is_separator = True
        elif "user" in msg:
            content = msg["user"].get("content", "")
            if content.strip() == "---诊疗分割线---" or "---诊疗分割线---" in content:
                is_separator = True

        if is_separator:
            if current_consultation:
                consultations.append(current_consultation)
                current_consultation = []
        else:
            current_consultation.append(msg)

    if current_consultation:
        consultations.append(current_consultation)
    return consultations


def extract_memory_ids_from_source(source: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    memory_ids = []
    for src in source:
        layer = src.get("layer", "")
        ids = src.get("ids", [])
        for mem_id in ids:
            memory_ids.append((layer, mem_id))
    return memory_ids


def get_first_time_memory_ids(
    consultation: List[Dict[str, Any]],
    query_position: int,
    source_memory_ids: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    seen_ids = set()
    for i in range(query_position):
        msg = consultation[i]
        oracle_memory = None
        if "assistant" in msg:
            oracle_memory = msg["assistant"].get("oracle_memory_base")
        elif "user" in msg:
            oracle_memory = msg["user"].get("oracle_memory_base")
        if oracle_memory:
            for layer, items in oracle_memory.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "id" in item:
                            seen_ids.add((layer, item["id"]))

    first_time_ids = []
    for layer, mem_id in source_memory_ids:
        if (layer, mem_id) not in seen_ids:
            first_time_ids.append((layer, mem_id))
    return first_time_ids


def create_previous_memory(
    oracle_memory_base: Dict[str, Any], ids_to_remove: List[Tuple[str, str]]
) -> Dict[str, Any]:
    previous_memory = {}
    ids_to_remove_set = set(ids_to_remove)
    for layer, items in oracle_memory_base.items():
        previous_memory[layer] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    mem_id = item.get("id")
                    if (layer, mem_id) not in ids_to_remove_set:
                        previous_memory[layer].append(item.copy())
    return previous_memory


def format_dialogue_context(consultation: List[Dict[str, Any]], query_position: int) -> str:
    dialogue_lines = []
    for i in range(query_position + 1):
        msg = consultation[i]
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            dialogue_lines.append(f"Assistant: {content}")
        elif "user" in msg:
            content = msg["user"].get("content", "")
            dialogue_lines.append(f"User: {content}")
    return "\n".join(dialogue_lines)


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    lines = []
    for layer, items in memory_state.items():
        if items:
            lines.append(f"\n{layer.upper()} MEMORY:")
            for item in items:
                item_id = item.get("id", "unknown")
                content = item.get("content", "")
                lines.append(f"  - [{item_id}] {content}")
    return "\n".join(lines) if lines else "No memories stored yet."


def build_prompt(dialogue_context: str, previous_memory: Dict[str, Any]) -> str:
    memory_text = format_memory_state(previous_memory)
    return f"""You are an intelligent medical memory management assistant specialized in healthcare dialogue systems. Your role is to analyze medical consultation dialogues and manage patient memory information across different layers, ensuring accurate and organized storage of medical data for effective patient care.
---
Current Dialogue Context:
{dialogue_context}

Current Memory State:
{memory_text}

Based on the dialogue context and current memory state, decide what memory operations (if any) are needed.

MEMORY LAYERS:
- working: Current dialogue key information (temporary, session-specific). Store patient basic info, current symptoms, examination results, diagnosis conclusions that are relevant to the current consultation session.
- identity: Patient identity and basic information layer (permanent patient characteristics). Store patient name, age, gender, occupation, and other stable personal information.
- history: Historical diagnosis and medical history layer (past medical events). Store past diagnoses, medical history, previous treatments, chronic conditions, and historical medical events.
- experience: Clinical experience and case layer (general medical knowledge). Store clinical observations, treatment patterns, and general medical knowledge that can be referenced for future cases.

AVAILABLE MEMORY TOOLS:
1. memory_insert(layer, content, metadata=None)
2. memory_update(layer, memory_id, content, metadata=None)
3. memory_delete(layer, memory_id)
4. memory_wait()

OUTPUT FORMAT:
<think>
[Your thinking about what memory operations are needed]
</think>

After your reasoning, output the memory operation(s) as JSON. You can output:
- A single operation: {{"name": "memory_insert", "arguments": {{"layer": "working", "content": "..."}}}}
- Multiple operations as a list: [{{"name": "memory_insert", "arguments": {{"layer": "working", "content": "..."}}}}, {{"name": "memory_update", "arguments": {{"layer": "identity", "memory_id": "...", "content": "..."}}}}]

IMPORTANT GUIDELINES:
- For memory_update and memory_delete, you MUST provide a valid memory_id from the current memory state
- The JSON must be valid (single object or array of objects)
- The layer must be one of: "working", "identity", "history", or "experience"
- If no operation is needed, output: {{"name": "memory_wait", "arguments": {{}}}}
"""


def process_patient_data(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    patient_id = patient_data.get("patient_id", "unknown")
    messages = patient_data.get("messages", [])
    consultations = split_consultations(messages)

    training_samples = []
    for consult_idx, consultation in enumerate(consultations):
        for msg_idx, msg in enumerate(consultation):
            memory_query = None
            oracle_memory_base = None
            if "user" in msg:
                memory_query = msg["user"].get("memory_query")
                oracle_memory_base = msg["user"].get("oracle_memory_base")
            elif "assistant" in msg:
                memory_query = msg["assistant"].get("memory_query")
                oracle_memory_base = msg["assistant"].get("oracle_memory_base")

            if memory_query is None or oracle_memory_base is None:
                continue

            source = memory_query.get("source", [])
            source_memory_ids = extract_memory_ids_from_source(source)
            first_time_ids = get_first_time_memory_ids(consultation, msg_idx, source_memory_ids)
            if not first_time_ids:
                continue

            previous_memory = create_previous_memory(oracle_memory_base, first_time_ids)
            dialogue_context = format_dialogue_context(consultation, msg_idx)
            prompt = build_prompt(dialogue_context, previous_memory)

            uuid_str = str(uuid.uuid4())
            if f"{patient_id}_{consult_idx}_{msg_idx}" in false_negatives:
                continue
            instance_id = f"{patient_id}_{consult_idx}_{msg_idx}_{uuid_str}"
            extra_info = {
                "agent_class": "swalm.core.agent.medmem_agent::MedMemAgent",
                "agent_run_params": {"max_iterations": 1},
                "dataset_id": instance_id,
                "instance_id": instance_id,
                "task_spec_class": "swalm.core.task.medmem.task::MedMemTaskSpec",
                "task_type": "swalm.core.task.medmem.task::run_medmem_task",
                "custom_data": {
                    "patient_id": patient_id,
                    "consultation_idx": consult_idx,
                    "message_idx": msg_idx,
                    "dialogue_context": dialogue_context,
                    "memory_state": json.dumps(previous_memory, ensure_ascii=False),
                    "oracle_memory_base": json.dumps(oracle_memory_base, ensure_ascii=False),
                    "memory_query": memory_query,
                    "supposed_new_memory_things": [
                        {"layer": layer, "id": mem_id} for layer, mem_id in first_time_ids
                    ],
                },
                "prompt": [{"role": "user", "content": prompt}]
            }

            sample = {
                "data_source": "memory/medical_dialogue",
                "ability": "swalm_env",
                "reward_model": {
                    "style": "swalm_agent_verifier",
                    "ground_truth": memory_query.get("answer", ""),
                },
                "extra_info": extra_info,
                "prompt":[] # 看extra_info的就行
            }
            training_samples.append(sample)

    return training_samples


def process_dataset(input_dir: str, output_file: str, max_patients: Optional[int] = None):
    patient_files = [
        f for f in os.listdir(input_dir) if f.endswith(".json") and f.startswith("patient_")
    ]
    patient_files.sort()
    if max_patients:
        patient_files = patient_files[:max_patients]

    all_samples = []
    for patient_file in patient_files:
        file_path = os.path.join(input_dir, patient_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            samples = process_patient_data(patient_data)
            all_samples.extend(samples)
        except Exception as exc:
            print(f"Error processing {patient_file}: {exc}")
            continue

    if all_samples:
        df = pd.DataFrame(all_samples)
        df.to_parquet(output_file, index=False, engine="pyarrow", row_group_size=4)
        print(f"Saved {len(all_samples)} samples to {output_file}")
    else:
        print("No samples generated.")


def main():
    parser = argparse.ArgumentParser(description="Process CareCall Memory RL data for swalm.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing patient JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--max_patients", type=int, default=None, help="Maximum number of patients to process")
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output, args.max_patients)


if __name__ == "__main__":
    main()

