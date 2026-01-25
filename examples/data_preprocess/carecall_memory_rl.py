"""
Data preprocessing script for CareCall Memory RL training data

Processes carecall-examiner data into RL training format:
1. Split consultations by "---诊疗分割线---"
2. Find memory_query positions
3. Extract previous_memory by removing first-time memory IDs
4. Create training samples with dialogue context and memory state
"""

import json
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd


def split_consultations(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split messages into separate consultations by "---诊疗分割线---"
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        List of consultation message lists
    """
    consultations = []
    current_consultation = []
    
    for msg in messages:
        # Check if this is a separator message
        # The separator appears as a standalone message with "---诊疗分割线---" as content
        is_separator = False
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            # Check if content is exactly or contains the separator
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
            # Don't add separator message to consultation
        else:
            current_consultation.append(msg)
    
    # Add the last consultation
    if current_consultation:
        consultations.append(current_consultation)
    
    return consultations


def extract_memory_ids_from_source(source: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Extract memory IDs from memory_query source
    
    Args:
        source: List of source dictionaries with layer and ids
    
    Returns:
        List of (layer, memory_id) tuples
    """
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
    source_memory_ids: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Find memory IDs that appear for the first time in this consultation before query_position
    
    Args:
        consultation: Current consultation messages
        query_position: Position of memory_query in consultation
        source_memory_ids: Memory IDs from memory_query source
    
    Returns:
        List of (layer, memory_id) that are first-time in this consultation
    """
    # Track all memory IDs seen before query_position
    seen_ids = set()
    
    # Check all messages before query_position
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
    
    # Find source IDs that are NOT in seen_ids (first-time)
    first_time_ids = []
    for layer, mem_id in source_memory_ids:
        if (layer, mem_id) not in seen_ids:
            first_time_ids.append((layer, mem_id))
    
    return first_time_ids


def create_previous_memory(
    oracle_memory_base: Dict[str, Any],
    ids_to_remove: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Create previous_memory by removing specified memory IDs from oracle_memory_base
    
    Args:
        oracle_memory_base: Original memory base
        ids_to_remove: List of (layer, memory_id) to remove
    
    Returns:
        Previous memory state with specified IDs removed
    """
    previous_memory = {}
    
    # Create a set for quick lookup
    ids_to_remove_set = set(ids_to_remove)
    
    for layer, items in oracle_memory_base.items():
        previous_memory[layer] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    mem_id = item.get("id")
                    if (layer, mem_id) not in ids_to_remove_set:
                        # Deep copy the item
                        previous_memory[layer].append(item.copy())
    
    return previous_memory


def format_dialogue_context(
    consultation: List[Dict[str, Any]],
    query_position: int
) -> str:
    """
    Format dialogue context up to query_position (not including the message with memory_query)
    
    Args:
        consultation: Current consultation messages
        query_position: Position of memory_query
    
    Returns:
        Formatted dialogue context string
    """
    dialogue_lines = []
    
    for i in range(query_position+1):
        msg = consultation[i]
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            dialogue_lines.append(f"Assistant: {content}")
        elif "user" in msg:
            content = msg["user"].get("content", "")
            dialogue_lines.append(f"User: {content}")
    
    return "\n".join(dialogue_lines)


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    """
    Format memory state as readable text
    
    Args:
        memory_state: Memory state dictionary
    
    Returns:
        Formatted text string
    """
    lines = []
    for layer, items in memory_state.items():
        if len(items) > 0:
            lines.append(f"\n{layer.upper()} MEMORY:")
            for item in items:
                item_id = item.get("id", "unknown")
                content = item.get("content", "")
                lines.append(f"  - [{item_id[:8]}] {content}")
    
    return "\n".join(lines) if lines else "No memories stored yet."


def process_patient_data(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single patient's data into training samples
    
    Args:
        patient_data: Patient data dictionary with patient_id and messages
    
    Returns:
        List of training samples
    """
    patient_id = patient_data.get("patient_id", "unknown")
    messages = patient_data.get("messages", [])
    
    # Split into consultations
    consultations = split_consultations(messages)
    
    training_samples = []
    
    for consult_idx, consultation in enumerate(consultations):
        # Track memory IDs seen in this consultation
        seen_memory_ids = set()
        
        # Process each message in the consultation
        for msg_idx, msg in enumerate(consultation):
            # Check if this message has a memory_query
            memory_query = None
            oracle_memory_base = None
            
            if "user" in msg:
                memory_query = msg["user"].get("memory_query")
                oracle_memory_base = msg["user"].get("oracle_memory_base")
            elif "assistant" in msg:
                memory_query = msg["assistant"].get("memory_query")
                oracle_memory_base = msg["assistant"].get("oracle_memory_base")
            
            if memory_query is None or oracle_memory_base is None:
                # Update seen memory IDs
                if oracle_memory_base:
                    for layer, items in oracle_memory_base.items():
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict) and "id" in item:
                                    seen_memory_ids.add((layer, item["id"]))
                continue
            
            # Extract source memory IDs
            source = memory_query.get("source", [])
            source_memory_ids = extract_memory_ids_from_source(source)
            
            # Find first-time memory IDs in this consultation
            first_time_ids = get_first_time_memory_ids(
                consultation, msg_idx, source_memory_ids
            )
            
            # Skip if no first-time IDs to remove
            if not first_time_ids:
                # Still update seen memory IDs
                for layer, items in oracle_memory_base.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and "id" in item:
                                seen_memory_ids.add((layer, item["id"]))
                continue
            
            # Create previous_memory by removing first-time IDs
            previous_memory = create_previous_memory(oracle_memory_base, first_time_ids)
            
            # Format dialogue context
            dialogue_context = format_dialogue_context(consultation, msg_idx)
            
            # Format memory state
            memory_text = format_memory_state(previous_memory)
            
            # Create prompt
            prompt = f"""Current Dialogue Context:
{dialogue_context}

Current Memory State:
{memory_text}

Based on the dialogue context and current memory state, decide what memory operations (if any) are needed.
You can:
1. Insert new memory entries (memory_insert)
2. Update existing memory entries (memory_update)
3. Delete memory entries (memory_delete)
4. Wait if no operation is needed (memory_wait)

Think about what information should be stored or updated, then perform the appropriate memory operations.
Your response should start with <think> for your thinking process, followed by the memory operations."""

            # Create training sample
            sample = {
                "prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "data_source": "memory/medical_dialogue",
                "ability": "memory_management",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": memory_query.get("answer", "")
                },
                "extra_info": {
                    "patient_id": patient_id,
                    "consultation_idx": consult_idx,
                    "message_idx": msg_idx,
                    "dialogue_context": dialogue_context,
                    "memory_state": json.dumps(previous_memory, ensure_ascii=False),
                    "oracle_memory_base": json.dumps(oracle_memory_base, ensure_ascii=False),
                    "memory_query": memory_query,
                    "supposed_new_memory_things": [
                        {"layer": layer, "id": mem_id}
                        for layer, mem_id in first_time_ids
                    ],
                    "first_time_memory_ids": first_time_ids
                }
            }
            
            training_samples.append(sample)
            
            # Update seen memory IDs
            for layer, items in oracle_memory_base.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "id" in item:
                            seen_memory_ids.add((layer, item["id"]))
    
    return training_samples


def process_dataset(
    input_dir: str,
    output_file: str,
    max_patients: Optional[int] = None
):
    """
    Process entire dataset from input directory
    
    Args:
        input_dir: Directory containing patient JSON files
        output_file: Output parquet file path
        max_patients: Maximum number of patients to process (None = all)
    """
    # Get all patient JSON files
    patient_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.json') and f.startswith('patient_')
    ]
    patient_files.sort()
    
    if max_patients:
        patient_files = patient_files[:max_patients]
    
    print(f"Processing {len(patient_files)} patient files...")
    
    all_samples = []
    
    for patient_file in patient_files:
        file_path = os.path.join(input_dir, patient_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            samples = process_patient_data(patient_data)
            all_samples.extend(samples)
            
            print(f"Processed {patient_file}: {len(samples)} samples")
        except Exception as e:
            print(f"Error processing {patient_file}: {e}")
            continue
    
    # Save to parquet
    if all_samples:
        df = pd.DataFrame(all_samples)
        df.to_parquet(output_file, index=False, engine="pyarrow",row_group_size=4)
        print(f"\nTotal: {len(all_samples)} training samples saved to {output_file}")
    else:
        print("No samples generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Process CareCall Memory RL training data"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing patient JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for testing)"
    )
    
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output, args.max_patients)


if __name__ == "__main__":
    main()

