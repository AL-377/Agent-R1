"""
Evaluation script for MedMem pure_summary and no_summary splits.

For each sample:
  1. Call the target model with the sample's prompt to get a response.
  2. (pure_summary only) Use a judge model to answer the memory_query
     based on the model's summary, then judge consistency with ground truth.
  3. (no_summary only) Directly judge the model's answer against ground truth.
  4. (pure_summary + correct) Compute compression ratio:
       len(summary) / len(content(supposed_new_memory_things))

Results are saved per-instance as JSON and aggregated into a DataFrame parquet.

Usage:
    python examples/eval/eval_medmem.py \
        --input datasets/cmtmedqa/cmtmedqa_pure_summary.parquet \
        --output_dir eval_results/cmtmedqa_pure_summary \
        --model gpt-4o-2024-11-20 \
        --judge_model gpt-4o-2024-11-20 \
        --max_samples 10

    python examples/eval/eval_medmem.py \
        --input datasets/cmtmedqa/cmtmedqa_no_summary.parquet \
        --output_dir eval_results/cmtmedqa_no_summary \
        --model gpt-4o-2024-11-20 \
        --judge_model gpt-4o-2024-11-20
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent_r1.utils.llm import query_llm


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
    parser = argparse.ArgumentParser(description="Evaluate MedMem pure_summary / no_summary")
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

    # Skip already-evaluated instances
    done_ids = set()
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            done_ids.add(fn[:-5])
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
