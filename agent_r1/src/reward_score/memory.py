"""
Memory Model Verification Functions
Validates memory operations correctness based on dialogue context and expected memory state
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple


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


def extract_thought(solution_str: str) -> Optional[str]:
    """
    Extract thought/reasoning from solution string
    
    Args:
        solution_str: Solution string
    
    Returns:
        Thought string or None
    """
    # Try to extract from <think> or <think> tags
    patterns = [
        r'<think>(.*?)</think>',
        r'<think>(.*?)</think>',
        r'Thought:(.*?)(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


def verify_memory_operations(
    solution_str: str,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Verify memory operations correctness
    
    Args:
        solution_str: Solution string containing memory operations
        ground_truth: Ground truth (expected memory state or operations)
        extra_info: Additional information containing:
            - expected_operations: List of expected memory operations
            - dialogue_context: Current dialogue context
            - memory_state: Current memory state
    
    Returns:
        Dictionary with verification results:
        - score: Overall verification score (0.0 to 1.0)
        - format_score: Format correctness score
        - operation_score: Operation correctness score
        - thought_score: Thought quality score
        - details: Detailed verification information
    """
    if solution_str is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "operation_score": 0.0,
            "thought_score": 0.0,
            "details": "Solution string is None"
        }
    
    # Extract components
    operations = extract_memory_operations(solution_str)
    thought = extract_thought(solution_str)
    
    # Format verification
    format_score = verify_format(solution_str, operations)
    
    # Operation verification
    operation_score = 0.0
    operation_details = {}
    
    if extra_info is not None:
        expected_operations = extra_info.get("expected_operations", [])
        dialogue_context = extra_info.get("dialogue_context", "")
        memory_state = json.loads(extra_info.get("memory_state", "{}"))
        
        operation_score, operation_details = verify_operations_correctness(
            operations=operations,
            expected_operations=expected_operations,
            dialogue_context=dialogue_context,
            memory_state=memory_state
        )
    
    # Thought quality verification
    thought_score = verify_thought_quality(thought, operations, extra_info)
    
    # Overall score (weighted combination)
    overall_score = (
        0.2 * format_score +
        0.6 * operation_score +
        0.2 * thought_score
    )
    
    return {
        "score": overall_score,
        "format_score": format_score,
        "operation_score": operation_score,
        "thought_score": thought_score,
        "details": {
            "operations": operations,
            "operation_details": operation_details,
            "thought": thought
        }
    }


def verify_format(solution_str: str, operations: List[Dict[str, Any]]) -> float:
    """
    Verify format correctness of solution
    
    Args:
        solution_str: Solution string
        operations: Extracted operations
    
    Returns:
        Format score (0.0 to 1.0)
    """
    score = 0.0
    
    # Check for proper structure
    if "<|im_start|>assistant" in solution_str and "<|im_end|>" in solution_str:
        score += 0.3
    
    # Check for thought/reasoning
    thought = extract_thought(solution_str)
    if thought and len(thought.strip()) > 10:
        score += 0.3
    
    # Check for proper tool call format
    if len(operations) > 0:
        valid_operations = sum(
            1 for op in operations
            if op.get("action") in ["memory_insert", "memory_update", "memory_delete", "memory_wait"]
            and "arguments" in op
        )
        if valid_operations == len(operations):
            score += 0.4
    else:
        # No operations is also valid (wait action)
        score += 0.4
    
    return min(score, 1.0)


def verify_operations_correctness(
    operations: List[Dict[str, Any]],
    expected_operations: List[Dict[str, Any]] = None,
    dialogue_context: str = "",
    memory_state: Dict[str, Any] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Verify correctness of memory operations
    
    Args:
        operations: Actual operations performed
        expected_operations: Expected operations (if available)
        dialogue_context: Current dialogue context
        memory_state: Current memory state
    
    Returns:
        Tuple of (score, details)
    """
    if expected_operations is not None and len(expected_operations) > 0:
        # Compare with expected operations
        return verify_against_expected(operations, expected_operations)
    else:
        # Verify based on dialogue context and memory state
        return verify_against_context(operations, dialogue_context, memory_state)


def verify_against_expected(
    operations: List[Dict[str, Any]],
    expected_operations: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Verify operations against expected operations
    
    Args:
        operations: Actual operations
        expected_operations: Expected operations
    
    Returns:
        Tuple of (score, details)
    """
    if len(expected_operations) == 0 and len(operations) == 0:
        return 1.0, {"match": "Both empty (wait action)"}
    
    if len(operations) == 0:
        # Check if wait was expected
        if any(op.get("action") == "memory_wait" for op in expected_operations):
            return 1.0, {"match": "Wait action as expected"}
        else:
            return 0.0, {"match": "No operations but expected operations"}
    
    # Match operations
    matched = 0
    total = max(len(operations), len(expected_operations))
    details = []
    
    # Simple matching: check if action types match
    actual_actions = [op.get("action") for op in operations]
    expected_actions = [op.get("action") for op in expected_operations]
    
    # Count matches
    for expected_action in expected_actions:
        if expected_action in actual_actions:
            matched += 1
            actual_actions.remove(expected_action)
    
    score = matched / total if total > 0 else 0.0
    
    details.append({
        "matched": matched,
        "total": total,
        "actual_actions": [op.get("action") for op in operations],
        "expected_actions": expected_actions
    })
    
    return score, {"match_details": details}


def verify_against_context(
    operations: List[Dict[str, Any]],
    dialogue_context: str,
    memory_state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Verify operations against dialogue context and memory state
    
    Args:
        operations: Actual operations
        dialogue_context: Dialogue context
        memory_state: Current memory state
    
    Returns:
        Tuple of (score, details)
    """
    if len(operations) == 0:
        # Wait action - check if it's reasonable
        # If dialogue has important information, wait might be wrong
        if dialogue_context and len(dialogue_context) > 50:
            # Check for key medical terms
            key_terms = ["症状", "诊断", "检查", "治疗", "病史", "患者"]
            has_key_info = any(term in dialogue_context for term in key_terms)
            if has_key_info:
                return 0.3, {"reason": "Important information in dialogue but no operation"}
            else:
                return 0.8, {"reason": "Wait action reasonable"}
        else:
            return 0.5, {"reason": "Wait action with minimal context"}
    
    # Verify operation validity
    valid_operations = 0
    details = []
    
    for op in operations:
        action = op.get("action")
        args = op.get("arguments", {})
        
        if action == "memory_insert":
            # Check if content is meaningful
            content = args.get("content", "")
            layer = args.get("layer", "")
            if content and len(content.strip()) > 5 and layer:
                valid_operations += 1
                details.append({"op": "insert", "valid": True})
            else:
                details.append({"op": "insert", "valid": False, "reason": "Invalid content or layer"})
        
        elif action == "memory_update":
            # Check if memory_id exists in memory_state
            memory_id = args.get("memory_id", "")
            layer = args.get("layer", "")
            content = args.get("content", "")
            
            if memory_id and layer and content:
                # Check if memory exists
                if memory_state and layer in memory_state:
                    memory_ids = [item.get("id") for item in memory_state.get(layer, [])]
                    if memory_id in memory_ids:
                        valid_operations += 1
                        details.append({"op": "update", "valid": True})
                    else:
                        details.append({"op": "update", "valid": False, "reason": "Memory ID not found"})
                else:
                    details.append({"op": "update", "valid": False, "reason": "Layer not in memory state"})
            else:
                details.append({"op": "update", "valid": False, "reason": "Missing required parameters"})
        
        elif action == "memory_delete":
            # Check if memory_id exists
            memory_id = args.get("memory_id", "")
            layer = args.get("layer", "")
            
            if memory_id and layer:
                if memory_state and layer in memory_state:
                    memory_ids = [item.get("id") for item in memory_state.get(layer, [])]
                    if memory_id in memory_ids:
                        valid_operations += 1
                        details.append({"op": "delete", "valid": True})
                    else:
                        details.append({"op": "delete", "valid": False, "reason": "Memory ID not found"})
                else:
                    details.append({"op": "delete", "valid": False, "reason": "Layer not in memory state"})
            else:
                details.append({"op": "delete", "valid": False, "reason": "Missing required parameters"})
        
        elif action == "memory_wait":
            valid_operations += 1
            details.append({"op": "wait", "valid": True})
    
    total_operations = len(operations)
    score = valid_operations / total_operations if total_operations > 0 else 0.0
    
    return score, {"operation_details": details, "valid_operations": valid_operations, "total_operations": total_operations}


def verify_thought_quality(
    thought: Optional[str],
    operations: List[Dict[str, Any]],
    extra_info: Dict[str, Any] = None
) -> float:
    """
    Verify thought/reasoning quality
    
    Args:
        thought: Extracted thought string
        operations: Memory operations
        extra_info: Additional information
    
    Returns:
        Thought quality score (0.0 to 1.0)
    """
    if thought is None or len(thought.strip()) < 10:
        return 0.2  # Low score for missing or very short thought
    
    score = 0.5  # Base score for having thought
    
    # Check thought length (reasonable length)
    if 20 <= len(thought) <= 500:
        score += 0.2
    
    # Check for relevant keywords
    relevant_keywords = ["记忆", "信息", "患者", "症状", "诊断", "更新", "添加", "删除"]
    found_keywords = sum(1 for keyword in relevant_keywords if keyword in thought)
    if found_keywords > 0:
        score += 0.2
    
    # Check if thought relates to operations
    if len(operations) > 0:
        # Thought should mention operations or reasoning
        operation_mentions = ["插入", "更新", "删除", "等待", "insert", "update", "delete", "wait"]
        if any(mention in thought for mention in operation_mentions):
            score += 0.1
    
    return min(score, 1.0)


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
    Main scoring function for memory model verification
    
    Args:
        solution_str: Solution string
        ground_truth: Ground truth
        extra_info: Additional information
        chat_model_func: Optional function to call chat model (for CareCall workflow)
        judge_model_func: Optional function to call judge model (for CareCall workflow)
        chat_model_name: Model name for chat (default: "gpt-4o-2024-11-20")
        judge_model_name: Model name for judge (default: "DeepSeek-R1")
    
    Returns:
        Score (0.0 to 1.0)
    """
    # If we have memory_query, use CareCall workflow (with or without custom functions)
    if extra_info:
        memory_query = extra_info.get("memory_query")
        previous_memory = json.loads(extra_info.get("memory_state"))
        supposed_new_memory_things = extra_info.get("supposed_new_memory_things", [])
        
        if memory_query and previous_memory:
            try:
                from . import carecall_memory
                return carecall_memory.compute_score(
                    solution_str,
                    ground_truth,
                    extra_info,
                    chat_model_func,
                    judge_model_func,
                    chat_model_name,
                    judge_model_name
                )
            except Exception as e:
                print(f"Error using CareCall memory scoring: {e}, falling back to standard scoring")
    
    # Standard scoring
    result = verify_memory_operations(solution_str, ground_truth, extra_info)
    return result["score"]


def compute_score_format(solution_str: str) -> float:
    """
    Format-only scoring function
    
    Args:
        solution_str: Solution string
    
    Returns:
        Format score (0.0 to 1.0)
    """
    operations = extract_memory_operations(solution_str)
    return verify_format(solution_str, operations)


def compute_score_operations(
    solution_str: str,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None
) -> float:
    """
    Operations-only scoring function
    
    Args:
        solution_str: Solution string
        ground_truth: Ground truth
        extra_info: Additional information
    
    Returns:
        Operations score (0.0 to 1.0)
    TODO: 修复
    """
    operations = extract_memory_operations(solution_str)
    
    if extra_info is not None:
        expected_operations = extra_info.get("expected_operations", [])
        dialogue_context = extra_info.get("dialogue_context", "")
        memory_state = json.loads(extra_info.get("memory_state", "{}"))
        
        score, _ = verify_operations_correctness(
            operations=operations,
            expected_operations=expected_operations,
            dialogue_context=dialogue_context,
            memory_state=memory_state
        )
        return score
    
    return 0.5  # Default score if no context provided

