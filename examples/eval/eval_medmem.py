"""
Evaluation script for MedMem pure_summary, layers_summary, and no_summary splits.

For each sample:
  1. Call the target model with the sample's prompt to get a response.

  2. (pure_summary) Use judge to answer memory_query based on model's summary,
     then judge consistency with ground truth. If correct, compute compression ratio.

  3. (layers_summary) Parse memory operations from model output, execute them on
     previous_memory, use judge to answer memory_query based on updated memory,
     then judge consistency with ground truth.

  4. (no_summary) Directly judge model's answer against ground truth.

Results are saved per-instance as JSON and aggregated into a DataFrame parquet.

Usage:
    python examples/eval/eval_medmem.py \
        --input datasets/cmtmedqa/cmtmedqa_pure_summary.parquet \
        --output_dir eval_results/cmtmedqa_pure_summary \
        --model gpt-4o-2024-11-20 \
        --judge_model gpt-4o-2024-11-20 \
        --max_samples 10

    python examples/eval/eval_medmem.py \
        --input datasets/cmtmedqa/cmtmedqa_layers_summary.parquet \
        --output_dir eval_results/cmtmedqa_layers_summary \
        --model gpt-4o-2024-11-20

    python examples/eval/eval_medmem.py \
        --input datasets/cmtmedqa/cmtmedqa_no_summary.parquet \
        --output_dir eval_results/cmtmedqa_no_summary \
        --model gpt-4o-2024-11-20
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent_r1.utils.llm import query_llm


ALLOWED_MEMORY_LAYERS = {"working", "identity", "history", "experience"}


# ---------------------------------------------------------------------------
# Memory operation parsing & execution (mirrors task.py logic)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences like ```json ... ``` from text."""
    import re
    text = re.sub(r"```(?:json|JSON)?\s*\n?", "", text)
    text = re.sub(r"\n?```", "", text)
    return text.strip()


def extract_memory_operations(solution_str: str) -> List[Dict[str, Any]]:
    """Parse memory operations from model output: <think>...</think> then JSON."""
    ops: List[Dict[str, Any]] = []
    if not solution_str:
        return ops

    end_tag = "</think>"
    tag_pos = solution_str.find(end_tag)
    json_part = (
        solution_str[tag_pos + len(end_tag):].strip() if tag_pos != -1
        else solution_str.strip()
    )
    json_part = _strip_code_fences(json_part)
    if not json_part:
        return ops

    # Models sometimes echo double-braces from prompt templates
    if "{{" in json_part and "}}" in json_part:
        json_part = json_part.replace("{{", "{").replace("}}", "}")

    def _collect(parsed):
        if isinstance(parsed, dict) and str(parsed.get("name", "")).startswith("memory_"):
            ops.append({
                "action": parsed["name"],
                "arguments": parsed.get("arguments", {}),
            })
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and str(item.get("name", "")).startswith("memory_"):
                    ops.append({
                        "action": item["name"],
                        "arguments": item.get("arguments", {}),
                    })

    try:
        _collect(json.loads(json_part))
    except json.JSONDecodeError:
        for i, ch in enumerate(json_part):
            if ch in ("{", "["):
                try:
                    _collect(json.loads(json_part[i:]))
                except json.JSONDecodeError:
                    pass
                break
    return ops


def validate_memory_operations(operations: List[Dict[str, Any]]) -> bool:
    if not operations:
        return False
    if all(op.get("action") == "memory_wait" for op in operations):
        return True
    for op in operations:
        action = op.get("action")
        args = op.get("arguments", {})
        if action == "memory_insert":
            if args.get("layer") not in ALLOWED_MEMORY_LAYERS or not args.get("content"):
                return False
        elif action == "memory_update":
            if (args.get("layer") not in ALLOWED_MEMORY_LAYERS
                    or not args.get("memory_id") or not args.get("content")):
                return False
        elif action == "memory_delete":
            if args.get("layer") not in ALLOWED_MEMORY_LAYERS or not args.get("memory_id"):
                return False
        elif action == "memory_wait":
            continue
        else:
            return False
    return True


def execute_memory_operations(
    previous_memory: Dict[str, Any],
    operations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mem = json.loads(json.dumps(previous_memory, ensure_ascii=False))
    for op in operations:
        action = op.get("action")
        args = op.get("arguments", {})
        try:
            if action == "memory_insert":
                layer = args.get("layer")
                content = str(args.get("content", "")).strip()
                if layer in ALLOWED_MEMORY_LAYERS and content:
                    mem.setdefault(layer, []).append({
                        "id": str(uuid.uuid4()),
                        "content": content,
                        "timestamp": time.time(),
                        "metadata": args.get("metadata") or {},
                    })
            elif action == "memory_update":
                layer, mid = args.get("layer"), args.get("memory_id")
                new_content = str(args.get("content", "")).strip()
                if layer in ALLOWED_MEMORY_LAYERS and mid and new_content:
                    for item in mem.get(layer, []):
                        if isinstance(item, dict) and item.get("id") == mid:
                            item["content"] = new_content
                            break
            elif action == "memory_delete":
                layer, mid = args.get("layer"), args.get("memory_id")
                if layer in ALLOWED_MEMORY_LAYERS and mid:
                    mem[layer] = [
                        it for it in mem.get(layer, [])
                        if not (isinstance(it, dict) and it.get("id") == mid)
                    ]
        except Exception:
            continue
    return mem


def answer_query_with_memory(
    judge_model: str,
    memory_state: Dict[str, Any],
    memory_query: Dict[str, Any],
) -> str:
    """Use judge model to answer memory_query based on a memory state dict."""
    question = memory_query.get("question", "")
    lines: List[str] = []
    for layer, items in memory_state.items():
        if items:
            lines.append(f"\n{layer.upper()} MEMORY:")
            for item in items:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('content', '')}")
    memory_text = "\n".join(lines) if lines else "No memories stored yet."
    prompt = (
        "Based on the following patient memory information, answer the question.\n\n"
        f"Memory Information:\n{memory_text}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    result = query_llm(
        model_name=judge_model,
        messages=[{"role": "user", "content": prompt}],
        system="You are a medical assistant. Answer questions based on the "
               "provided patient memory information.",
        temperature=0.0,
    )
    return result.get("response", "")


# ---------------------------------------------------------------------------
# LLM wrappers
# ---------------------------------------------------------------------------

def call_model(model_name: str, prompt_text: str, **kwargs) -> str:
    result = query_llm(
        model_name=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        **kwargs,
    )
    return result.get("response", "")


def call_judge(judge_model: str, answer_a: str, answer_b: str) -> float:
    """Return 1.0 if consistent, 0.0 if not, -2 if unclear."""
    prompt = (
        "Compare these two answers and determine if they are consistent "
        "(meaning the same thing).\n\n"
        f"Answer 1: {answer_a}\n"
        f"Answer 2: {answer_b}\n\n"
        'Are these answers consistent? Respond with only "Yes" or "No".'
    )
    result = query_llm(
        model_name=judge_model,
        messages=[{"role": "user", "content": prompt}],
        system="You are a judge that determines if two answers are consistent. "
               "Respond with only 'Yes' or 'No'.",
        temperature=0.0,
    )
    resp = result.get("response", "").strip().lower()
    if "yes" in resp:
        return 1.0
    if "no" in resp:
        return 0.0
    return -2


def answer_query_with_summary(
    judge_model: str,
    summary: str,
    memory_query: Dict[str, Any],
) -> str:
    """Use the judge model to answer memory_query based on a summary."""
    question = memory_query.get("question", "")
    prompt = (
        "Based on the following summary of a medical consultation, "
        "answer the question.\n\n"
        f"Summary:\n{summary}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    result = query_llm(
        model_name=judge_model,
        messages=[{"role": "user", "content": prompt}],
        system="You are a medical assistant. Answer questions based on the "
               "provided summary information.",
        temperature=0.0,
    )
    return result.get("response", "")


# ---------------------------------------------------------------------------
# Supposed-new content extraction
# ---------------------------------------------------------------------------

def get_supposed_new_content(
    supposed_new: List[Dict[str, str]],
    oracle_memory_base: Dict[str, Any],
) -> str:
    """Concatenate content of supposed_new_memory_things from oracle_memory_base."""
    contents: List[str] = []
    for item in supposed_new:
        layer = item.get("layer", "")
        mid = item.get("id", "")
        for mem in oracle_memory_base.get(layer, []):
            if isinstance(mem, dict) and mem.get("id") == mid:
                contents.append(mem.get("content", ""))
                break
    return "\n".join(contents)


# ---------------------------------------------------------------------------
# Single instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    row: Dict[str, Any],
    model_name: str,
    judge_model: str,
    model_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Evaluate a single sample. Returns a result dict."""
    extra_info = row["extra_info"]
    custom_data = extra_info["custom_data"]
    split = custom_data.get("split", "unknown")
    instance_id = extra_info["instance_id"]
    prompt_text = extra_info["prompt"][0]["content"]
    ground_truth = row["reward_model"]["ground_truth"]
    memory_query = custom_data["memory_query"]
    oracle_memory_base_str = custom_data.get("oracle_memory_base", "{}")
    oracle_memory_base = (
        json.loads(oracle_memory_base_str)
        if isinstance(oracle_memory_base_str, str)
        else oracle_memory_base_str
    )
    supposed_new = custom_data.get("supposed_new_memory_things", [])

    patient_metadata = custom_data.get("patient_metadata", {})
    scenario = patient_metadata.get("scenario", "")
    scenario_name = patient_metadata.get("scenario_name", "")
    query_type = memory_query.get("type", "")
    difficulty = memory_query.get("difficulty", "")

    result = {
        "instance_id": instance_id,
        "split": split,
        "model": model_name,
        "judge_model": judge_model,
        "scenario": scenario,
        "scenario_name": scenario_name,
        "query_type": query_type,
        "difficulty": difficulty,
        "question": memory_query.get("question", ""),
        "ground_truth": ground_truth,
        "patient_id": custom_data.get("patient_id", ""),
        "consultation_idx": custom_data.get("consultation_idx", -1),
        "message_idx": custom_data.get("message_idx", -1),
    }

    try:
        model_output = call_model(model_name, prompt_text, **(model_kwargs or {}))
        result["model_output"] = model_output

        if split == "pure_summary":
            chat_answer = answer_query_with_summary(
                judge_model, model_output, memory_query
            )
            result["chat_answer"] = chat_answer
            score = call_judge(judge_model, chat_answer, ground_truth)
            result["correct"] = score

            if score == 1.0:
                snt_content = get_supposed_new_content(supposed_new, oracle_memory_base)
                if snt_content:
                    result["compression_ratio"] = len(model_output) / max(len(snt_content), 1)
                else:
                    result["compression_ratio"] = None
                result["summary_length"] = len(model_output)
                result["snt_content_length"] = len(snt_content) if snt_content else 0
            else:
                result["compression_ratio"] = None
                result["summary_length"] = len(model_output)
                result["snt_content_length"] = None

        elif split == "layers_summary":
            memory_state_str = custom_data.get("memory_state", "{}")
            previous_memory = (
                json.loads(memory_state_str)
                if isinstance(memory_state_str, str)
                else memory_state_str
            )
            operations = extract_memory_operations(model_output)
            format_ok = validate_memory_operations(operations)
            result["format_valid"] = format_ok
            result["num_operations"] = len(operations)

            if format_ok:
                after_memory = execute_memory_operations(previous_memory, operations)
                chat_answer = answer_query_with_memory(
                    judge_model, after_memory, memory_query
                )
                result["chat_answer"] = chat_answer
                score = call_judge(judge_model, chat_answer, ground_truth)
                result["correct"] = score

                if score == 1.0:
                    snt_content = get_supposed_new_content(supposed_new, oracle_memory_base)
                    if snt_content:
                        result["compression_ratio"] = len(model_output) / max(len(snt_content), 1)
                    else:
                        result["compression_ratio"] = None
                    result["summary_length"] = len(model_output)
                    result["snt_content_length"] = len(snt_content) if snt_content else 0
                else:
                    result["compression_ratio"] = None
                    result["summary_length"] = len(model_output)
                    result["snt_content_length"] = None
            else:
                result["correct"] = 0.0
                result["compression_ratio"] = None
                result["summary_length"] = len(model_output)
                result["snt_content_length"] = None

        elif split == "no_summary":
            score = call_judge(judge_model, model_output, ground_truth)
            result["correct"] = score
            result["compression_ratio"] = None
            result["summary_length"] = None
            result["snt_content_length"] = None

        else:
            result["correct"] = -2
            result["compression_ratio"] = None
            result["summary_length"] = None
            result["snt_content_length"] = None

    except Exception as exc:
        result["model_output"] = ""
        result["correct"] = -2
        result["compression_ratio"] = None
        result["summary_length"] = None
        result["snt_content_length"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedMem pure_summary / layers_summary / no_summary")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str, required=True, help="Target model name for generation")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Judge model name (defaults to --model)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.judge_model is None:
        args.judge_model = args.model

    os.makedirs(args.output_dir, exist_ok=True)
    traces_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    df = pd.read_parquet(args.input)
    records = df.to_dict(orient="records")

    if args.max_samples:
        records = records[:args.max_samples]

    # Skip already-evaluated instances (only those without errors)
    done_ids = set()
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                    trace = json.load(f)
                if "error" not in trace:
                    done_ids.add(fn[:-5])
            except Exception:
                pass
    records = [r for r in records if r["extra_info"]["instance_id"] not in done_ids]

    print(f"Evaluating {len(records)} samples with model={args.model}, "
          f"judge={args.judge_model}, workers={args.workers}")

    model_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    all_results: List[Dict[str, Any]] = []

    def _run(row):
        return evaluate_instance(row, args.model, args.judge_model, model_kwargs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, r): r for r in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = fut.result()
            all_results.append(result)
            trace_path = os.path.join(traces_dir, f"{result['instance_id']}.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Also load previously done results
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            iid = fn[:-5]
            if iid not in {r["instance_id"] for r in all_results}:
                with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))

    # Save aggregated DataFrame
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, "results.parquet")
    results_df.to_parquet(summary_path, index=False)

    # Print quick summary
    valid = results_df[results_df["correct"] >= 0]
    if len(valid) > 0:
        acc = valid["correct"].mean()
        print(f"\nAccuracy: {acc:.4f} ({int(valid['correct'].sum())}/{len(valid)})")
        if "compression_ratio" in valid.columns:
            cr = valid["compression_ratio"].dropna()
            if len(cr) > 0:
                print(f"Avg compression ratio (correct only): {cr.mean():.2f}")
    else:
        print("\nNo valid results.")

    print(f"Results saved to {summary_path}")
    print(f"Traces saved to {traces_dir}/")


if __name__ == "__main__":
    main()
