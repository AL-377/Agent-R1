"""
Evaluate the impact of memory enhancement on medical dialogue quality.

This script compares Chat Model responses under two conditions:
  (A) with_memory:  Chat Model receives the structured memory state (oracle)
                    plus recent dialogue context, simulating the full prototype
                    system workflow.
  (B) no_memory:    Chat Model receives the FULL raw dialogue history from
                    turn 0 to the current turn (no structured memory), simulating
                    a vanilla long-context setting. This mirrors the "no_summary"
                    split design in chap3 benchmark.

For each dialogue sample, the Chat Model generates a doctor response, then an
LLM Judge scores the response on five clinically-relevant dimensions (1-5):
  1. Medical Accuracy   – factual correctness of medical advice
  2. Personalization     – whether the response uses patient-specific information
  3. Consistency         – alignment with prior dialogue and medical history
  4. Completeness        – thoroughness and actionability of the response
  5. Safety              – absence of potentially harmful or contraindicated advice

Results are saved per-instance as JSON and aggregated into a parquet DataFrame.

Usage:
    python examples/eval/eval_dialogue_quality.py \
        --cache_dirs datasets/cmtmedqa/.examiner_cache \
                     datasets/huatuo26m/.examiner_cache \
                     datasets/meddialog_cn/.examiner_cache \
        --output_dir eval_results/dialogue_quality \
        --chat_model gpt-4o-2024-11-20 \
        --judge_model gpt-4o-2024-11-20 \
        --max_samples_per_source 50 \
        --workers 4
"""

import argparse
import json
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent_r1.utils.llm import query_llm

DIMENSIONS = [
    "medical_accuracy",
    "personalization",
    "consistency",
    "completeness",
    "safety",
]

DIMENSION_DESCRIPTIONS_EN = {
    "medical_accuracy": "Medical Accuracy: Is the medical advice factually correct and evidence-based?",
    "personalization": "Personalization: Does the response leverage patient-specific information (history, allergies, demographics)?",
    "consistency": "Consistency: Is the response consistent with prior dialogue context and established medical facts?",
    "completeness": "Completeness: Is the response thorough, covering necessary aspects and providing actionable guidance?",
    "safety": "Safety: Does the response avoid potentially harmful, contraindicated, or dangerous suggestions?",
}

# Scoring rubric passed to the judge
JUDGE_SYSTEM_PROMPT = """\
You are an expert medical dialogue quality evaluator. You will be given:
1. A patient-doctor dialogue context
2. A doctor's response to evaluate
3. (Optionally) the patient's known medical memory/history

Score the doctor's response on each of the following five dimensions using a 1-5 scale:

Dimensions:
- medical_accuracy (1-5): Is the medical advice factually correct and evidence-based?
  1=major errors, 2=some inaccuracies, 3=mostly correct, 4=accurate, 5=exemplary
- personalization (1-5): Does the response use patient-specific information?
  1=completely generic, 2=minimal personalization, 3=some reference to patient info, 4=well-personalized, 5=deeply personalized
- consistency (1-5): Is the response consistent with prior context and medical history?
  1=contradicts history, 2=ignores important context, 3=partially consistent, 4=consistent, 5=perfectly coherent
- completeness (1-5): Is the response thorough and actionable?
  1=extremely incomplete, 2=missing key info, 3=adequate, 4=thorough, 5=comprehensive and actionable
- safety (1-5): Does the response avoid harmful suggestions?
  1=dangerous advice, 2=concerning omissions, 3=acceptable, 4=safe, 5=exemplary safety awareness

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{"medical_accuracy": <int>, "personalization": <int>, "consistency": <int>, "completeness": <int>, "safety": <int>, "rationale": "<brief explanation>"}
"""


def format_memory_state(memory_state: Dict[str, Any]) -> str:
    """Format a memory state dict into readable text for the Chat Model."""
    lines = []
    layer_names = {
        "working": "Current Session Info",
        "identity": "Patient Identity & Demographics",
        "history": "Medical History",
        "experience": "Clinical Knowledge",
    }
    for layer in ["identity", "history", "working", "experience"]:
        items = memory_state.get(layer, [])
        if items:
            lines.append(f"\n[{layer_names.get(layer, layer)}]")
            for item in items:
                if isinstance(item, dict):
                    lines.append(f"  - {item.get('content', '')}")
    return "\n".join(lines) if lines else ""


def build_chat_prompt_with_memory(
    dialogue_context: str,
    memory_text: str,
    patient_message: str,
) -> str:
    """Build the Chat Model prompt with memory context."""
    return (
        "You are an experienced and empathetic doctor. A patient is consulting you. "
        "You have access to the patient's medical records and memory from previous visits.\n\n"
        f"=== Patient Medical Records ===\n{memory_text}\n\n"
        f"=== Recent Dialogue ===\n{dialogue_context}\n\n"
        f"Patient: {patient_message}\n\n"
        "Please respond as the doctor. Provide professional, personalized medical advice "
        "based on both the current conversation and the patient's medical records."
    )


def build_chat_prompt_without_memory(
    full_dialogue_history: str,
    patient_message: str,
) -> str:
    """Build the Chat Model prompt with full raw dialogue history but no structured memory."""
    return (
        "You are an experienced and empathetic doctor. A patient is consulting you. "
        "Below is the complete dialogue history from all previous visits and the current session.\n\n"
        f"=== Complete Dialogue History ===\n{full_dialogue_history}\n\n"
        f"Patient: {patient_message}\n\n"
        "Please respond as the doctor. Provide professional medical advice "
        "based on the dialogue history above."
    )


def build_judge_prompt(
    dialogue_context: str,
    memory_text: Optional[str],
    patient_message: str,
    doctor_response: str,
) -> str:
    """Build the prompt for the Judge model to score the response."""
    parts = [f"=== Dialogue Context ===\n{dialogue_context}"]
    if memory_text:
        parts.append(f"\n=== Patient Medical Records ===\n{memory_text}")
    parts.append(f"\n=== Patient's Current Message ===\n{patient_message}")
    parts.append(f"\n=== Doctor's Response to Evaluate ===\n{doctor_response}")
    parts.append(
        "\nPlease evaluate the doctor's response on all five dimensions. "
        "Respond with ONLY a JSON object."
    )
    return "\n".join(parts)


def parse_judge_scores(response: str) -> Optional[Dict[str, Any]]:
    """Extract the five dimension scores from judge output."""
    import re
    response = response.strip()
    # Try direct JSON parse
    try:
        data = json.loads(response)
        if all(d in data for d in DIMENSIONS):
            return data
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if all(d in data for d in DIMENSIONS):
                return data
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Data loading — build evaluation samples from .examiner_cache
# ---------------------------------------------------------------------------

def load_samples_from_cache(
    cache_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load evaluation samples from examiner cache JSON files.

    For each patient file, we select dialogue turns that have memory_query
    (i.e. turns where memory was evaluated) and at least 2 prior turns of
    context. Each sample contains:
      - full_dialogue_history: ALL turns from turn 0 to current (for no_memory)
      - recent_context: last few turns (for with_memory prompt)
      - oracle_memory_base: structured memory state (for with_memory condition)
    """
    samples = []
    source_name = os.path.basename(os.path.dirname(cache_dir))
    if not source_name or source_name == ".examiner_cache":
        source_name = os.path.basename(os.path.dirname(os.path.dirname(cache_dir)))

    json_files = sorted(
        f for f in os.listdir(cache_dir) if f.endswith(".json")
    )
    for fname in json_files:
        fpath = os.path.join(cache_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
        except Exception:
            continue

        patient_id = patient_data.get("patient_id", fname.replace(".json", ""))
        messages = patient_data.get("messages", [])

        for turn_idx, msg in enumerate(messages):
            user_part = msg.get("user", {})
            if not user_part:
                continue

            memory_query = user_part.get("memory_query")
            oracle_memory = user_part.get("oracle_memory_base")
            patient_message = user_part.get("content", "")

            if not memory_query or not oracle_memory or not patient_message:
                continue
            if turn_idx < 2:
                continue

            # Build FULL dialogue history: turn 0 to turn (turn_idx - 1)
            # This gives the no_memory condition the complete raw conversation
            full_history_turns = []
            for prev_idx in range(0, turn_idx):
                prev_msg = messages[prev_idx]
                if "user" in prev_msg and prev_msg["user"].get("content"):
                    full_history_turns.append(f"Patient: {prev_msg['user']['content']}")
                if "assistant" in prev_msg and prev_msg["assistant"].get("content"):
                    full_history_turns.append(f"Doctor: {prev_msg['assistant']['content']}")
            full_dialogue_history = "\n".join(full_history_turns)

            # Build recent context (last few turns, for with_memory prompt)
            recent_turns = []
            start = max(0, turn_idx - 3)
            for prev_idx in range(start, turn_idx):
                prev_msg = messages[prev_idx]
                if "user" in prev_msg and prev_msg["user"].get("content"):
                    recent_turns.append(f"Patient: {prev_msg['user']['content']}")
                if "assistant" in prev_msg and prev_msg["assistant"].get("content"):
                    recent_turns.append(f"Doctor: {prev_msg['assistant']['content']}")
            recent_context = "\n".join(recent_turns)

            gt_response = ""
            if "assistant" in msg and msg["assistant"].get("content"):
                gt_response = msg["assistant"]["content"]

            difficulty = memory_query.get("difficulty", "")
            query_type = memory_query.get("type", "")

            samples.append({
                "instance_id": f"{source_name}_{patient_id}_{turn_idx}",
                "source": source_name,
                "patient_id": patient_id,
                "turn_idx": turn_idx,
                "patient_message": patient_message,
                "full_dialogue_history": full_dialogue_history,
                "recent_context": recent_context,
                "oracle_memory_base": oracle_memory,
                "memory_query": memory_query,
                "gt_response": gt_response,
                "difficulty": difficulty,
                "query_type": query_type,
            })

    if max_samples and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)

    return samples


# ---------------------------------------------------------------------------
# Single instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    sample: Dict[str, Any],
    chat_model: str,
    judge_model: str,
    chat_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Evaluate a single sample under both with_memory and no_memory conditions."""
    result = {
        "instance_id": sample["instance_id"],
        "source": sample["source"],
        "patient_id": sample["patient_id"],
        "turn_idx": sample["turn_idx"],
        "difficulty": sample["difficulty"],
        "query_type": sample["query_type"],
        "chat_model": chat_model,
        "judge_model": judge_model,
        "patient_message": sample["patient_message"][:200],
    }

    full_dialogue_history = sample["full_dialogue_history"]
    recent_context = sample["recent_context"]
    patient_message = sample["patient_message"]
    oracle_memory = sample["oracle_memory_base"]
    memory_text = format_memory_state(oracle_memory)

    try:
        # --- Condition A: with_memory ---
        # Chat Model gets structured memory state + recent dialogue context
        prompt_with = build_chat_prompt_with_memory(
            recent_context, memory_text, patient_message
        )
        resp_with = query_llm(
            model_name=chat_model,
            messages=[{"role": "user", "content": prompt_with}],
            **(chat_kwargs or {}),
        )
        response_with_memory = resp_with.get("response", "")
        result["response_with_memory"] = response_with_memory

        # --- Condition B: no_memory ---
        # Chat Model gets the FULL raw dialogue history (all turns), no structured memory
        prompt_without = build_chat_prompt_without_memory(
            full_dialogue_history, patient_message
        )
        resp_without = query_llm(
            model_name=chat_model,
            messages=[{"role": "user", "content": prompt_without}],
            **(chat_kwargs or {}),
        )
        response_no_memory = resp_without.get("response", "")
        result["response_no_memory"] = response_no_memory

        # --- Judge: score condition A ---
        # Judge sees full history + memory (ground truth context for fair scoring)
        judge_prompt_a = build_judge_prompt(
            full_dialogue_history, memory_text, patient_message, response_with_memory
        )
        judge_resp_a = query_llm(
            model_name=judge_model,
            messages=[{"role": "user", "content": judge_prompt_a}],
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        scores_a = parse_judge_scores(judge_resp_a.get("response", ""))
        if scores_a:
            for dim in DIMENSIONS:
                result[f"with_memory_{dim}"] = int(scores_a.get(dim, 0))
            result["with_memory_rationale"] = scores_a.get("rationale", "")
        else:
            for dim in DIMENSIONS:
                result[f"with_memory_{dim}"] = -1
            result["with_memory_parse_error"] = judge_resp_a.get("response", "")[:300]

        # --- Judge: score condition B ---
        # Judge sees the same full history (fair ground truth), but no memory
        judge_prompt_b = build_judge_prompt(
            full_dialogue_history, None, patient_message, response_no_memory
        )
        judge_resp_b = query_llm(
            model_name=judge_model,
            messages=[{"role": "user", "content": judge_prompt_b}],
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        scores_b = parse_judge_scores(judge_resp_b.get("response", ""))
        if scores_b:
            for dim in DIMENSIONS:
                result[f"no_memory_{dim}"] = int(scores_b.get(dim, 0))
            result["no_memory_rationale"] = scores_b.get("rationale", "")
        else:
            for dim in DIMENSIONS:
                result[f"no_memory_{dim}"] = -1
            result["no_memory_parse_error"] = judge_resp_b.get("response", "")[:300]

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dialogue quality with/without memory enhancement"
    )
    parser.add_argument(
        "--cache_dirs", nargs="+", required=True,
        help="Paths to .examiner_cache directories",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--chat_model", type=str, required=True,
        help="Chat model to generate doctor responses",
    )
    parser.add_argument(
        "--judge_model", type=str, default=None,
        help="Judge model for scoring (defaults to --chat_model)",
    )
    parser.add_argument("--max_samples_per_source", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.judge_model is None:
        args.judge_model = args.chat_model

    os.makedirs(args.output_dir, exist_ok=True)
    traces_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    all_samples = []
    for cache_dir in args.cache_dirs:
        samples = load_samples_from_cache(
            cache_dir, args.max_samples_per_source, args.seed
        )
        print(f"  Loaded {len(samples)} samples from {cache_dir}")
        all_samples.extend(samples)
    print(f"Total samples: {len(all_samples)}")

    # Skip already-evaluated instances
    done_ids = set()
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                    trace = json.load(f)
                if "error" not in trace:
                    done_ids.add(trace.get("instance_id", fn[:-5]))
            except Exception:
                pass
    all_samples = [s for s in all_samples if s["instance_id"] not in done_ids]
    print(f"Remaining after skipping done: {len(all_samples)}")

    chat_kwargs = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    all_results: List[Dict[str, Any]] = []

    def _run(sample):
        return evaluate_instance(sample, args.chat_model, args.judge_model, chat_kwargs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, s): s for s in all_samples}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = fut.result()
            all_results.append(result)
            trace_path = os.path.join(traces_dir, f"{result['instance_id']}.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    # Load all trace results (including previously done)
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            iid = fn[:-5]
            if iid not in {r["instance_id"] for r in all_results}:
                try:
                    with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                        all_results.append(json.load(f))
                except Exception:
                    pass

    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, "results.parquet")
    results_df.to_parquet(summary_path, index=False)

    # Quick summary
    print(f"\n{'='*60}")
    print(f"  Dialogue Quality Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total evaluated: {len(results_df)}")

    for dim in DIMENSIONS:
        col_with = f"with_memory_{dim}"
        col_no = f"no_memory_{dim}"
        if col_with in results_df.columns and col_no in results_df.columns:
            valid_with = results_df[results_df[col_with] > 0][col_with]
            valid_no = results_df[results_df[col_no] > 0][col_no]
            if len(valid_with) > 0 and len(valid_no) > 0:
                delta = valid_with.mean() - valid_no.mean()
                sign = "+" if delta > 0 else ""
                print(
                    f"  {dim:25s}  "
                    f"with_memory={valid_with.mean():.2f}  "
                    f"no_memory={valid_no.mean():.2f}  "
                    f"delta={sign}{delta:.2f}"
                )

    print(f"\nResults saved to {summary_path}")
    print(f"Traces saved to {traces_dir}/")


if __name__ == "__main__":
    main()
