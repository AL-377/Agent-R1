"""
Data preprocessing script for Medical Dialogue Memory Model training data

This script processes medical dialogue data into the format required for training.
Each sample should contain:
- dialogue_context: Current dialogue turn
- memory_state: Current global memory state
- expected_operations: Expected memory operations (optional, for supervised learning)
- ground_truth: Expected final memory state or operation verification
"""

import json
import pandas as pd
from typing import Dict, List, Any
import argparse


def process_memory_sample(
    dialogue_context: str,
    memory_state: Dict[str, Any],
    expected_operations: List[Dict[str, Any]] = None,
    ground_truth: str = None
) -> Dict[str, Any]:
    """
    Process a single memory training sample
    
    Args:
        dialogue_context: Current dialogue context
        memory_state: Current memory state (dict with layer -> list of items)
        expected_operations: Expected memory operations (optional)
        ground_truth: Ground truth for verification (optional)
    
    Returns:
        Processed sample dictionary
    """
    # Format memory state as text
    memory_text = format_memory_state(memory_state)
    
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

Think about what information should be stored or updated, then perform the appropriate memory operations."""

    # Create sample
    sample = {
        "prompt": prompt,
        "data_source": "memory/medical_dialogue",
        "extra_info": {
            "dialogue_context": dialogue_context,
            "memory_state": memory_state,
            "expected_operations": expected_operations or []
        }
    }
    
    if ground_truth:
        sample["ground_truth"] = ground_truth
    
    return sample


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


def load_dialogue_data(input_file: str) -> List[Dict[str, Any]]:
    """
    Load dialogue data from input file
    
    Args:
        input_file: Path to input JSON/JSONL file
    
    Returns:
        List of dialogue samples
    """
    samples = []
    
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            else:
                samples = [data]
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    return samples


def process_dataset(input_file: str, output_file: str):
    """
    Process entire dataset
    
    Args:
        input_file: Input dialogue data file
        output_file: Output parquet file path
    """
    # Load input data
    raw_samples = load_dialogue_data(input_file)
    
    # Process samples
    processed_samples = []
    for raw_sample in raw_samples:
        dialogue_context = raw_sample.get("dialogue_context", "")
        memory_state = raw_sample.get("memory_state", {})
        expected_operations = raw_sample.get("expected_operations")
        ground_truth = raw_sample.get("ground_truth")
        
        processed_sample = process_memory_sample(
            dialogue_context=dialogue_context,
            memory_state=memory_state,
            expected_operations=expected_operations,
            ground_truth=ground_truth
        )
        processed_samples.append(processed_sample)
    
    # Save to parquet
    df = pd.DataFrame(processed_samples)
    df.to_parquet(output_file, index=False)
    print(f"Processed {len(processed_samples)} samples and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process memory model training data")
    parser.add_argument("--input", type=str, required=True, help="Input dialogue data file (JSON/JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

