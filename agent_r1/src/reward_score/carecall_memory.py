"""
CareCall Memory Model Verification and Scoring Functions

Implements the verification workflow:
1. Execute model actions on previous_memory to generate after_memory_base
2. Use chat model (GPT-4o) to answer memory_query based on after_memory_base
3. Use judge model (DeepSeek R1) to compare answers
4. Check coverage of supposed_new_memory_things
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
import copy

# Note: We implement memory operations directly without MemoryManager
# to preserve original memory IDs for verification


def extract_memory_operations(solution_str: str) -> List[Dict[str, Any]]:
    """
    Extract memory operations from solution string
    
    Args:
        solution_str: Solution string containing tool calls
    
    Returns:
        List of memory operation dictionaries
    """
    operations = []
    
    # Extract tool calls
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_calls = tool_call_pattern.findall(solution_str)
    
    for tool_call_str in tool_calls:
        try:
            tool_call = json.loads(tool_call_str)
            if "name" in tool_call and tool_call["name"].startswith("memory_"):
                operations.append({
                    "action": tool_call["name"],
                    "arguments": tool_call.get("arguments", {})
                })
        except json.JSONDecodeError:
            continue
    
    return operations


def execute_memory_operations(
    previous_memory: Dict[str, Any],
    operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute memory operations on previous_memory to generate after_memory_base
    
    Args:
        previous_memory: Previous memory state
        operations: List of memory operations to execute
    
    Returns:
        after_memory_base: Memory state after operations
    """
    # Deep copy previous_memory to avoid modifying original
    after_memory_base = copy.deepcopy(previous_memory)
    
    # Track ID mapping for newly inserted items (since MemoryManager generates new IDs)
    # For verification, we'll work directly with the memory structure
    
    # Execute operations directly on the memory structure
    for op in operations:
        action = op.get("action")
        args = op.get("arguments", {})
        
        try:
            if action == "memory_insert":
                layer = args.get("layer")
                content = args.get("content", "").strip()
                metadata = args.get("metadata", {})
                
                if layer and content:
                    import uuid
                    import time
                    new_item = {
                        "id": str(uuid.uuid4()),
                        "content": content,
                        "timestamp": time.time(),
                        "metadata": metadata or {}
                    }
                    if layer not in after_memory_base:
                        after_memory_base[layer] = []
                    after_memory_base[layer].append(new_item)
            
            elif action == "memory_update":
                layer = args.get("layer")
                memory_id = args.get("memory_id")
                new_content = args.get("content", "").strip()
                metadata = args.get("metadata")
                
                if layer and memory_id and new_content and layer in after_memory_base:
                    for item in after_memory_base[layer]:
                        if isinstance(item, dict) and item.get("id") == memory_id:
                            item["content"] = new_content
                            if metadata:
                                item["metadata"].update(metadata)
                            break
            
            elif action == "memory_delete":
                layer = args.get("layer")
                memory_id = args.get("memory_id")
                
                if layer and memory_id and layer in after_memory_base:
                    after_memory_base[layer] = [
                        item for item in after_memory_base[layer]
                        if not (isinstance(item, dict) and item.get("id") == memory_id)
                    ]
            
            elif action == "memory_wait":
                # No operation
                pass
        except Exception as e:
            print(f"Error executing {action}: {e}")
            continue
    
    return after_memory_base


def check_memory_coverage(
    after_memory_base: Dict[str, Any],
    supposed_new_memory_things: List[Dict[str, str]]
) -> float:
    """
    Check what percentage of supposed_new_memory_things is covered in after_memory_base
    
    Args:
        after_memory_base: Memory state after operations
        supposed_new_memory_things: List of {layer, id} that should be present
    
    Returns:
        Coverage percentage (0.0 to 1.0)
    """
    if not supposed_new_memory_things:
        return 1.0
    
    covered = 0
    for item in supposed_new_memory_things:
        layer = item.get("layer")
        mem_id = item.get("id")
        
        if layer in after_memory_base:
            memory_items = after_memory_base[layer]
            if isinstance(memory_items, list):
                for mem_item in memory_items:
                    if isinstance(mem_item, dict) and mem_item.get("id") == mem_id:
                        covered += 1
                        break
    
    return covered / len(supposed_new_memory_things) if supposed_new_memory_things else 1.0


def verify_with_chat_model(
    memory_query: Dict[str, Any],
    after_memory_base: Dict[str, Any],
    chat_model_func: Optional[callable] = None,
    chat_model_name: str = "gpt-4o-2024-11-20"
) -> str:
    """
    Use chat model to answer memory_query question based on after_memory_base
    
    Args:
        memory_query: Memory query dictionary with question and answer
        after_memory_base: Memory state after operations
        chat_model_func: Function to call chat model (if None, uses query_llm from utils)
        chat_model_name: Model name for chat (default: "gpt-4o-2024-11-20")
    
    Returns:
        Chat model's answer
    """
    question = memory_query.get("question", "")
    
    # Format memory state for prompt
    memory_text = format_memory_state_for_prompt(after_memory_base)
    
    prompt = f"""Based on the following patient memory information, answer the question.

Memory Information:
{memory_text}

Question: {question}

Answer:"""
    
    if chat_model_func is not None:
        # Use provided function
        try:
            answer = chat_model_func(prompt)
            return answer
        except Exception as e:
            print(f"Error calling chat model function: {e}")
            return "[Error calling chat model]"
    
    # Use query_llm from utils
    try:
        from agent_r1.utils.llm import query_llm_inhouse
        
        result = query_llm_inhouse(
            model_name=chat_model_name,
            messages=prompt,
            system="You are a medical assistant. Answer questions based on the provided patient memory information.",
            temperature=0.0,
            max_tokens=500
        )
        return result.get('response', '[No response from chat model]').strip()
    except Exception as e:
        print(f"Error calling chat model via query_llm: {e}")
        return "[Error calling chat model]"


def judge_answer_consistency(
    chat_answer: str,
    ground_truth_answer: str,
    judge_model_func: Optional[callable] = None,
    judge_model_name: str = "DeepSeek-R1"
) -> float:
    """
    Use judge model to determine if chat_answer and ground_truth_answer are consistent
    
    Args:
        chat_answer: Answer from chat model
        ground_truth_answer: Ground truth answer from memory_query
        judge_model_func: Function to call judge model (if None, uses query_llm from utils)
        judge_model_name: Model name for judge (default: "DeepSeek-R1")
    
    Returns:
        Consistency score (0.0 to 1.0)
    """
    if judge_model_func is not None:
        # Use provided function
        try:
            judge_prompt = f"""Compare these two answers and determine if they are consistent (meaning the same thing).

Answer 1: {chat_answer}
Answer 2: {ground_truth_answer}

Are these answers consistent? Respond with only "Yes" or "No"."""
            
            response = judge_model_func(judge_prompt)
            response_lower = response.lower().strip()
            
            if "yes" in response_lower:
                return 1.0
            elif "no" in response_lower:
                return 0.0
            else:
                # Try to extract a score
                try:
                    score = float(response)
                    return max(0.0, min(1.0, score))
                except:
                    return 0.5
        except Exception as e:
            print(f"Error calling judge model function: {e}")
            return 0.0
    
    # Use query_llm from utils
    try:
        from agent_r1.utils.llm import query_llm_inhouse
        
        judge_prompt = f"""Compare these two answers and determine if they are consistent (meaning the same thing).

Answer 1: {chat_answer}
Answer 2: {ground_truth_answer}

Are these answers consistent? Respond with only "Yes" or "No"."""
        
        result = query_llm_inhouse(
            model_name=judge_model_name,
            messages=judge_prompt,
            system="You are a judge that determines if two answers are consistent (meaning the same thing). Respond with only 'Yes' or 'No'.",
            temperature=0.0,
            max_tokens=50
        )
        
        response = result.get('response', 'No').strip()
        response_lower = response.lower().strip()
        
        if "yes" in response_lower:
            return 1.0
        elif "no" in response_lower:
            return 0.0
        else:
            # Try to extract a score
            try:
                score = float(response)
                return max(0.0, min(1.0, score))
            except:
                return 0.5
    except Exception as e:
        print(f"Error calling judge model via query_llm: {e}")
        # Fallback to simple comparison
        chat_lower = chat_answer.lower().strip()
        gt_lower = ground_truth_answer.lower().strip()
        
        # Exact match
        if chat_lower == gt_lower:
            return 1.0
        
        # Check if ground truth is contained in chat answer
        if gt_lower in chat_lower:
            return 0.8
        
        # Check for key words overlap
        gt_words = set(gt_lower.split())
        chat_words = set(chat_lower.split())
        if len(gt_words) > 0:
            overlap = len(gt_words & chat_words) / len(gt_words)
            return overlap * 0.6  # Max 0.6 for word overlap
        
        return 0.0


def format_memory_state_for_prompt(memory_state: Dict[str, Any]) -> str:
    """
    Format memory state for chat model prompt
    
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
                content = item.get("content", "")
                lines.append(f"  - {content}")
    
    return "\n".join(lines) if lines else "No memories stored yet."


def compute_score(
    solution_str: str,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    chat_model_func: Optional[callable] = None,
    judge_model_func: Optional[callable] = None,
    chat_model_name: str = "gpt-4o-2024-11-20",
    judge_model_name: str = "DeepSeek-R1"
) -> float:
    """
    Main scoring function for CareCall Memory Model
    
    Args:
        solution_str: Solution string containing memory operations
        ground_truth: Ground truth answer (from memory_query)
        extra_info: Additional information containing:
            - previous_memory: Previous memory state
            - memory_query: Memory query dictionary
            - supposed_new_memory_things: List of memory items that should be present
        chat_model_func: Function to call chat model (if None, uses query_llm)
        judge_model_func: Function to call judge model (if None, uses query_llm)
        chat_model_name: Model name for chat (default: "gpt-4o-2024-11-20")
        judge_model_name: Model name for judge (default: "DeepSeek-R1")
    
    Returns:
        Score (0.0 to 1.0)
    """
    print(f"Input solution_str: {solution_str}")
    if solution_str is None or extra_info is None:
        return 0.0
    
    # Extract memory operations
    operations = extract_memory_operations(solution_str)
    
    # Get memory_state and other info
    previous_memory = json.loads(extra_info.get("memory_state"))
    memory_query = extra_info.get("memory_query")
    supposed_new_memory_things = extra_info.get("supposed_new_memory_things", [])
    
    if previous_memory is None or memory_query is None:
        return 0.0
    
    # Execute operations to get after_memory_base
    try:
        after_memory_base = execute_memory_operations(previous_memory, operations)
    except Exception as e:
        print(f"Error executing memory operations: {e}")
        return 0.0
    
    # Check memory coverage TODO: 目前先不检查coverage
    # coverage_score = check_memory_coverage(after_memory_base, supposed_new_memory_things)
    
    # Verify with chat model and judge
    chat_answer = verify_with_chat_model(
        memory_query, 
        after_memory_base, 
        chat_model_func,
        chat_model_name
    )
    print(f"chat_answer: {chat_answer}")

    consistency_score = judge_answer_consistency(
        chat_answer,
        ground_truth or memory_query.get("answer", ""),
        judge_model_func,
        judge_model_name
    )
    print(f"consistency_score: {consistency_score}")
    
    # Combined score: 50% consistency, 50% coverage
    # final_score = 0.5 * consistency_score + 0.5 * coverage_score
    final_score = consistency_score

    print(f"Output final_score: {final_score}")
    return final_score


def compute_score_format(solution_str: str) -> float:
    """
    Format-only scoring function
    
    Args:
        solution_str: Solution string
    
    Returns:
        Format score (0.0 to 1.0)
    """
    operations = extract_memory_operations(solution_str)
    
    if not operations:
        return 0.3  # Some score for having output
    
    # Check format validity
    valid_ops = 0
    for op in operations:
        action = op.get("action")
        args = op.get("arguments", {})
        
        if action in ["memory_insert", "memory_update", "memory_delete", "memory_wait"]:
            if action == "memory_wait" or "arguments" in op:
                valid_ops += 1
    
    format_score = valid_ops / len(operations) if operations else 0.0
    
    # Check for thought/reasoning
    thought_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    has_thought = bool(thought_pattern.search(solution_str))
    
    if has_thought:
        format_score = min(1.0, format_score + 0.2)
    
    return format_score


def compute_score_operations(
    solution_str: str,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None
) -> float:
    """
    Operations-only scoring function (without chat/judge models)
    
    Args:
        solution_str: Solution string
        ground_truth: Ground truth
        extra_info: Additional information
    
    Returns:
        Operations score (0.0 to 1.0)
    """
    if solution_str is None or extra_info is None:
        return 0.0
    
    operations = extract_memory_operations(solution_str)
    previous_memory = json.loads(extra_info.get("memory_state"))
    supposed_new_memory_things = extra_info.get("supposed_new_memory_things", [])
    
    if previous_memory is None:
        return 0.0
    
    try:
        after_memory_base = execute_memory_operations(previous_memory, operations)
        coverage_score = check_memory_coverage(after_memory_base, supposed_new_memory_things)
        return coverage_score
    except Exception as e:
        print(f"Error in compute_score_operations: {e}")
        return 0.0

