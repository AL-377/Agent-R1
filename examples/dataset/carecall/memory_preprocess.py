"""
Memory evaluation dataset preprocessing script

Processes medical dialogue data into memory evaluation format.
For each patient, maintains an oracle_memory_base and processes dialogues sequentially.
"""

import json
import re
import copy
import concurrent.futures
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import sys
import os
from tqdm import tqdm

from agent_r1.utils.llm import query_llm_inhouse,open_proxy,close_proxy
from agent_r1.tool.memory_manager import MemoryManager


def extract_patient_id(guid: str) -> str:
    """
    Extract patient ID from guid.
    Format: "interactive-{patient_id}-S{session_num}"
    
    Args:
        guid: Guid string like "interactive-0-S1"
    
    Returns:
        Patient ID string (e.g., "0")
    """
    match = re.match(r"interactive-(\d+)-", guid)
    if match:
        return match.group(1)
    return None


def group_dialogues_by_patient(data: List[List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Group dialogues by patient ID.
    
    Args:
        data: Raw data from JSON file
    
    Returns:
        Dictionary mapping patient_id to list of dialogue sessions
    """
    patient_dialogues = defaultdict(list)
    
    for session_list in data:
        for session in session_list:
            guid = session.get("guid", "")
            patient_id = extract_patient_id(guid)
            if patient_id is not None:
                patient_dialogues[patient_id].append(session)
    
    # Sort sessions by guid to maintain order
    for patient_id in patient_dialogues:
        patient_dialogues[patient_id].sort(key=lambda x: x.get("guid", ""))
    
    return patient_dialogues


def extract_to_memory(message_content: str, dialogue_history: List[Dict], 
                     model_name: str = "gpt-4o-2024-11-20") -> List[str]:
    """
    Use LLM to extract valuable information that should be memorized.
    
    Args:
        message_content: Current message content
        dialogue_history: Previous dialogue messages for context
        model_name: LLM model name
    
    Returns:
        List of memory items with format "layer: content" (can be empty)
    """
    # Format dialogue history
    history_text = ""
    for msg in dialogue_history[-5:]:  # Use last 5 messages for context
        role = msg.get("role", "")
        text = msg.get("text", "")
        # Map system to assistant for clarity
        display_role = "assistant" if role == "system" else role
        history_text += f"{display_role}: {text}\n"
    
    prompt = f"""You are analyzing a medical dialogue to extract valuable information that should be memorized.

Dialogue History:
{history_text}

Current Message:
{message_content}

Based on the dialogue, identify valuable information that should be stored in memory. Consider:
- Patient identity and basic info (name, age, gender, etc.)
- Current symptoms and complaints
- Medical history and diagnoses
- Treatment plans and medications
- Clinical observations

Memory layers:
- working: Current dialogue key info (temporary, session-specific)
- identity: Patient identity and basic info (permanent patient characteristics)
- history: Historical diagnosis and medical history (past medical events)
- experience: Clinical experience and cases (general medical knowledge)

Extract information items that should be memorized. For each item, specify which layer it belongs to.
Format: Return a JSON list of strings, where each string has format "layer: content".
Example: ["identity: Patient name is John, age 45", "history: Patient has diabetes diagnosed 5 years ago", "working: Current complaint is headache"]

Important:
- Only extract information that is explicitly mentioned or clearly implied
- If no valuable information should be memorized, return an empty list []
- Each item should start with one of: working:, identity:, history:, or experience:

Return only the JSON list, no other text."""

    try:
        response = query_llm_inhouse(
            model_name=model_name,
            messages=prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        result_text = response.get("response", "").strip()
        # if with think, extract content after <think>**</think>
        if result_text.startswith("<think>"):
            think_end = result_text.find("</think>")
            if think_end != -1:
                result_text = result_text[think_end+len("</think>"):].strip()
        # Try to parse JSON
        # Remove markdown code blocks if present
        result_text = re.sub(r"```json\s*", "", result_text)
        result_text = re.sub(r"```\s*", "", result_text)
        result_text = result_text.strip()
        print(result_text)
        # Try to extract JSON array
        if result_text.startswith("["):
            memory_items = json.loads(result_text)
            if isinstance(memory_items, list):
                # Validate and filter items
                valid_items = []
                for item in memory_items:
                    item_str = str(item).strip()
                    # Check if item has valid layer prefix
                    if any(item_str.startswith(f"{layer}:") for layer in ["working", "identity", "history", "experience"]):
                        valid_items.append(item_str)
                return valid_items
        
        return []
    except Exception as e:
        print(f"Error extracting to_memory: {e}")
        return []


def generate_memory_function_calls(to_memory: List[str], current_memory_state: Dict[str, Any],
                                   model_name: str = "gpt-4o-2024-11-20") -> List[Dict[str, Any]]:
    """
    Use LLM to generate function calls for memory operations.
    
    Args:
        to_memory: List of memory items with format "layer: content"
        current_memory_state: Current oracle_memory_base state
        model_name: LLM model name
    
    Returns:
        List of function call dictionaries
    """
    if not to_memory:
        return []
    
    # Format current memory state
    memory_state_text = format_memory_state(current_memory_state)
    
    # Format to_memory items
    to_memory_text = "\n".join([f"- {item}" for item in to_memory])
    
    prompt = f"""You need to store the following information into the memory system:

Information to store (format: "layer: content"):
{to_memory_text}

Current Memory State:
{memory_state_text}

Available memory tools:
1. memory_insert(layer, content, metadata=None) - Insert new memory
2. memory_update(layer, memory_id, content, metadata=None) - Update existing memory
3. memory_delete(layer, memory_id) - Delete memory
4. memory_wait() - No operation needed

For each information item (which already has the layer specified):
- Extract the layer and content from the "layer: content" format
- If it's new information, use memory_insert with the extracted layer and content
- If it updates existing information, use memory_update (you need to find the memory_id from current state)
- If it's redundant or should be removed, use memory_delete
- If no operation is needed, use memory_wait. **It is used when the information is already in the memory and no update is needed.**

Important:
- The layer is already specified in each item (working, identity, history, or experience)
- Extract the layer and content from each item
- For memory_update, you must provide a valid memory_id from the current memory state

Return a JSON list of function calls. Each function call should be a dictionary with:
- "name": tool name (e.g., "memory_insert")
- "arguments": dictionary of arguments

Example:
[
  {{"name": "memory_insert", "arguments": {{"layer": "identity", "content": "Patient name is John, age 45"}}}},
  {{"name": "memory_insert", "arguments": {{"layer": "history", "content": "Patient has diabetes diagnosed 5 years ago"}}}}
]

Return only the JSON list, no other text."""

    try:
        response = query_llm_inhouse(
            model_name=model_name,
            messages=prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        result_text = response.get("response", "").strip()
        
        # if with think, extract content after <think>**</think>
        if result_text.startswith("<think>"):
            think_end = result_text.find("</think>")
            if think_end != -1:
                result_text = result_text[think_end+len("</think>"):].strip()
        # Remove markdown code blocks if present
        result_text = re.sub(r"```json\s*", "", result_text)
        result_text = re.sub(r"```\s*", "", result_text)
        result_text = result_text.strip()
        
        print(result_text)
        # Try to parse JSON
        if result_text.startswith("["):
            function_calls = json.loads(result_text)
            if isinstance(function_calls, list):
                # Validate function calls
                valid_calls = []
                for call in function_calls:
                    if isinstance(call, dict) and "name" in call and "arguments" in call:
                        # Ensure layer is valid
                        args = call.get("arguments", {})
                        layer = args.get("layer")
                        if layer in ["working", "identity", "history", "experience"]:
                            valid_calls.append(call)
                return valid_calls
        
        return []
    except Exception as e:
        print(f"Error generating function calls: {e}")
        return []


def execute_memory_operations(function_calls: List[Dict[str, Any]], 
                             memory_manager: MemoryManager) -> bool:
    """
    Execute memory operations using MemoryManager.
    
    Args:
        function_calls: List of function call dictionaries
        memory_manager: MemoryManager instance
    
    Returns:
        True if any operation was successful, False otherwise
    """
    if not function_calls:
        return False
    
    any_success = False
    
    for func_call in function_calls:
        name = func_call.get("name", "")
        args = func_call.get("arguments", {})
        
        try:
            if name == "memory_insert":
                layer = args.get("layer")
                content = args.get("content")
                metadata = args.get("metadata")
                if layer and content:
                    memory_manager.insert(layer=layer, content=content, metadata=metadata)
                    any_success = True
            
            elif name == "memory_update":
                layer = args.get("layer")
                memory_id = args.get("memory_id")
                content = args.get("content")
                metadata = args.get("metadata")
                if layer and memory_id and content:
                    success = memory_manager.update(
                        layer=layer, 
                        memory_id=memory_id, 
                        new_content=content, 
                        metadata=metadata
                    )
                    if success:
                        any_success = True
            
            elif name == "memory_delete":
                layer = args.get("layer")
                memory_id = args.get("memory_id")
                if layer and memory_id:
                    success = memory_manager.delete(layer=layer, memory_id=memory_id)
                    if success:
                        any_success = True
            
            elif name == "memory_wait":
                # No operation
                pass
        
        except Exception as e:
            print(f"Error executing {name}: {e}")
    
    return any_success


def generate_memory_query(to_memory: List[str], memory_changes: bool,
                          model_name: str = "gpt-4o-2024-11-20") -> Optional[Dict[str, str]]:
    """
    Generate a memory query (question and answer) to test memory completeness.
    
    Args:
        to_memory: List of memory items that were stored
        memory_changes: Whether memory was actually changed
        model_name: LLM model name
    
    Returns:
        Dictionary with "question" and "answer" keys, or None
    """
    if not memory_changes or not to_memory:
        return None
    
    # Format to_memory items
    to_memory_text = "\n".join([f"- {item}" for item in to_memory])
    
    prompt = f"""Based on the following information that was just stored in memory, generate a question and answer pair to test if the memory system can recall this information.

Information stored:
{to_memory_text}

Generate a question that tests whether the memory system can recall the key information that was just stored.
Also provide the correct answer to this question.

Return a JSON object with:
- "question": A question about the stored information
- "answer": The correct answer based on the stored information

Example:
{{"question": "What is the patient's name and age?", "answer": "The patient's name is John and age is 45"}}

Return only the JSON object, no other text."""

    try:
        response = query_llm_inhouse(
            model_name=model_name,
            messages=prompt,
            temperature=0.7,
            max_tokens=16384
        )
        
        result_text = response.get("response", "").strip()
        
        # Remove markdown code blocks if present
        result_text = re.sub(r"```json\s*", "", result_text)
        result_text = re.sub(r"```\s*", "", result_text)
        result_text = result_text.strip()
        
        # Try to parse JSON
        if result_text.startswith("{"):
            query_dict = json.loads(result_text)
            if "question" in query_dict and "answer" in query_dict:
                return {
                    "question": str(query_dict["question"]),
                    "answer": str(query_dict["answer"])
                }
        
        return None
    except Exception as e:
        print(f"Error generating memory query: {e}")
        return None


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    """
    Format memory state as readable text.
    
    Args:
        memory_state: Memory state dictionary from MemoryManager.get_memory_state()
    
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


def process_patient_dialogues(patient_id: str, sessions: List[Dict], 
                             model_name: str = "gpt-4o-2024-11-20") -> Optional[Dict]:
    """
    Process all dialogues for a single patient.
    
    Args:
        patient_id: Patient ID
        sessions: List of dialogue sessions for this patient
        model_name: LLM model name
    
    Returns:
        Dict with "patient_id" and "messages" keys, or None if processing failed
    """
    print(f"Processing patient {patient_id}")
    try:
        # Initialize memory manager for this patient
        try:
            memory_manager = MemoryManager(device="cuda")
        except Exception as e:
            print(f"Warning: Failed to initialize MemoryManager with embeddings: {e}")
            print("Attempting to use MemoryManager without embeddings...")
            # Try to create a minimal memory manager
            # For now, raise the error - user should install required dependencies
            raise RuntimeError(
                f"MemoryManager initialization failed. Please ensure FlagEmbedding and faiss are installed. "
                f"Error: {e}"
            )
        
        # Store processed messages
        processed_messages = []
        
        # Track dialogue history
        dialogue_history = []
        
        # Process each session
        for session_idx, session in enumerate(sessions):
            # Reset working memory at the start of each session
            memory_manager.reset_working_memory()
            
            # Get dialogue from session
            dialogue = session.get("dialogue", [])
            
            # Add session separator (except for first session)
            if session_idx > 0:
                processed_messages.append("---诊疗分割线---")
            
            # Process each message in the dialogue
            for msg in dialogue:
                role = msg.get("role", "")
                text = msg.get("text", "")
                
                # Map roles: system -> assistant, user -> user
                # Skip if role is not user or system
                if role not in ["user", "system"]:
                    dialogue_history.append(msg)
                    continue
                
                # Map system to assistant for output
                output_role = "assistant" if role == "system" else "user"
                
                # Get current memory state before processing
                memory_state_before = copy.deepcopy(memory_manager.get_memory_state())
                
                # Extract to_memory
                to_memory = extract_to_memory(text, dialogue_history, model_name)
                print(f"Patient {patient_id} to_memory: {to_memory}")
                # Generate function calls if to_memory is not empty
                function_calls = []
                if to_memory:
                    function_calls = generate_memory_function_calls(
                        to_memory, memory_state_before, model_name
                    )
                    print(f"Patient {patient_id} function_calls: {function_calls}")
                # Execute function calls
                memory_changed = False
                if function_calls:
                    memory_changed = execute_memory_operations(function_calls, memory_manager)
                
                # Get memory state after processing
                memory_state_after = memory_manager.get_memory_state()
                
                # Generate memory_query if memory changed
                memory_query = None
                if memory_changed:
                    memory_query = generate_memory_query(to_memory, memory_changed, model_name)
                
                # Format memory state for output
                oracle_memory_base = {
                    "working": memory_state_after.get("working", []),
                    "identity": memory_state_after.get("identity", []),
                    "history": memory_state_after.get("history", []),
                    "experience": memory_state_after.get("experience", [])
                }
                
                # Create processed message in target format
                # Format: {role: {content, to_memory, oracle_memory_base, memory_query?}}
                processed_msg = {
                    output_role: {
                        "content": text,
                        "to_memory": to_memory if to_memory else [None],
                        "oracle_memory_base": oracle_memory_base
                    }
                }
                
                # Add memory_query if available
                if memory_query:
                    processed_msg[output_role]["memory_query"] = memory_query
                
                # Add to processed messages
                processed_messages.append(processed_msg)
                
                # Update dialogue history
                dialogue_history.append(msg)
        
        print(f"Completed processing patient {patient_id}")
        return {"patient_id": patient_id, "messages": processed_messages}
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return None


def process_dataset(input_file: str, output_file: str, 
                   model_name: str = "gpt-4o-2024-11-20",
                   max_patients: Optional[int] = None,
                   workers: int = 4) -> None:
    """
    Process the entire dataset.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        model_name: LLM model name to use
        max_patients: Maximum number of patients to process (None for all)
        workers: Number of concurrent workers to use
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Grouping dialogues by patient...")
    patient_dialogues = group_dialogues_by_patient(data)
    
    print(f"Found {len(patient_dialogues)} patients")
    
    if max_patients:
        patient_ids = sorted(patient_dialogues.keys())[:max_patients]
        print(f"Processing first {len(patient_ids)} patients...")
    else:
        patient_ids = sorted(patient_dialogues.keys())
    
    all_processed_data = []
    
    # Process patients in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Prepare tasks
        future_to_patient = {
            executor.submit(process_patient_dialogues, patient_id, patient_dialogues[patient_id], model_name): 
            patient_id for patient_id in patient_ids
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patient_ids)):
            patient_id = future_to_patient[future]
            try:
                result = future.result()
                if result:
                    all_processed_data.append(result)
                    print(f"  Processed {len(result['messages'])} messages for patient {patient_id}")
            except Exception as e:
                print(f"  Error processing patient {patient_id}: {e}")
    
    # Sort results by patient_id to maintain order
    all_processed_data.sort(key=lambda x: x['patient_id'])
    
    print(f"\nSaving processed data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Processed {len(all_processed_data)} patients")


def main():
    import argparse
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any ProcessPoolExecutor
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set, which is fine
        pass
    
    parser = argparse.ArgumentParser(
        description="Process medical dialogue data into memory evaluation format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="examples/dataset/carecall-memory_en_auto_translated.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples/dataset/memory_eval_dataset.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="LLM model name to use"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (for testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers to use"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        max_patients=args.max_patients,
        workers=args.workers
    )


if __name__ == "__main__":
    open_proxy()
    main()