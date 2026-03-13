"""
Evaluate MED-MEM memory accuracy using the Enhanced Memory Agent Framework.

Uses the same enhanced MemoryBase from eval_memory_agent.py with:
  - Relevance scoring (MemoryBank-inspired Ebbinghaus decay)
  - Conflict detection & auto-merge (A-Mem-inspired)
  - Session-end reflection (Generative Agents-inspired)

Pipeline per patient:
  1. Initialize empty MemoryBase
  2. For each message:
     a. Session separator → reflect on working memory, then clear
     b. User turn → Memory Model decides operations → execute with conflict detection
     c. Assistant turn → append to history
  3. At each user turn with memory_query:
     a. Judge answers query from agent-maintained memory
     b. Compare with ground truth

Usage:
    python examples/eval/eval_medmem_agent.py \
        --cache_dirs datasets/cmtmedqa/.examiner_cache \
                     datasets/huatuo26m/.examiner_cache \
                     datasets/meddialog_cn/.examiner_cache \
        --output_dir eval_results/medmem_agent \
        --memory_model gpt-4o-2024-05-13 \
        --judge_model gpt-4o-2024-05-13 \
        --max_patients_per_source 50 \
        --workers 4
"""

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from agent_r1.utils.llm import query_llm
from examples.eval.eval_memory_agent import (
    MemoryBase, VALID_LAYERS, MEMORY_SYSTEM_PROMPT,
    parse_memory_output, execute_actions_on_memory,
    reflect_at_session_end, build_memory_model_prompt,
)


# ── Judge helpers ────────────────────────────────────────────────────────────

def answer_query_with_memory(
    judge_model: str, memory_state: Dict[str, Any], memory_query: Dict[str, Any],
) -> str:
    question = memory_query.get("question", "")
    lines = []
    for layer in VALID_LAYERS:
        items = memory_state.get(layer, [])
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


def judge_consistency(judge_model: str, answer_a: str, answer_b: str) -> float:
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
        system="You are a judge. Respond with only 'Yes' or 'No'.",
        temperature=0.0,
    )
    resp = result.get("response", "").strip().lower()
    if "yes" in resp:
        return 1.0
    if "no" in resp:
        return 0.0
    return -2


# ── Data loading ─────────────────────────────────────────────────────────────

def load_patients_from_cache(
    cache_dir: str, max_patients: Optional[int] = None, seed: int = 42,
) -> List[Dict[str, Any]]:
    source_name = os.path.basename(os.path.dirname(cache_dir))
    if not source_name or source_name == ".examiner_cache":
        source_name = os.path.basename(os.path.dirname(os.path.dirname(cache_dir)))

    json_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".json"))
    if max_patients and len(json_files) > max_patients:
        rng = random.Random(seed)
        json_files = rng.sample(json_files, max_patients)

    patients = []
    for fname in json_files:
        fpath = os.path.join(cache_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            patient_data["_source"] = source_name
            patient_data["_file"] = fname
            patients.append(patient_data)
        except Exception:
            continue
    return patients


# ── Process one patient ──────────────────────────────────────────────────────

def process_patient(
    patient_data: Dict[str, Any],
    memory_model: str,
    judge_model: str,
    memory_kwargs: Optional[Dict] = None,
    also_eval_oracle: bool = True,
    enable_reflection: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the Enhanced Memory Agent through a patient's full dialogue history.
    Uses MemoryBase with relevance scoring, conflict detection, and reflection.
    """
    messages = patient_data.get("messages", [])
    patient_id = patient_data.get("patient_id", "unknown")
    source = patient_data.get("_source", "unknown")
    metadata = patient_data.get("metadata", {})

    memory = MemoryBase()
    dialogue_so_far: List[str] = []
    results = []

    for turn_idx, msg in enumerate(messages):
        if "assistant" in msg:
            content = msg["assistant"].get("content", "")
            if "---诊疗分割线---" in content or "---consultation separator---" in content.lower():
                if enable_reflection:
                    reflect_at_session_end(memory, memory_model, memory_kwargs)
                memory.clear_working()
                dialogue_so_far = []
                continue
            dialogue_so_far.append(f"Doctor: {content}")
            continue

        user_part = msg.get("user", {})
        if not user_part or not user_part.get("content"):
            continue

        patient_message = user_part["content"]
        dialogue_so_far.append(f"Patient: {patient_message}")
        memory.tick()

        dialogue_ctx = "\n".join(dialogue_so_far[-10:])
        mem_state_text = memory.get_state_text()

        potential_conflicts = memory.retrieve_by_keyword(patient_message, top_k=3)
        conflict_hints = ""
        if potential_conflicts:
            conflict_hints = "\n".join(
                f"  - [{it.id}] ({it.layer}) {it.content}"
                for it in potential_conflicts
            )

        prompt = build_memory_model_prompt(dialogue_ctx, mem_state_text, conflict_hints)

        raw_output = ""
        try:
            result = query_llm(
                model_name=memory_model,
                messages=[{"role": "user", "content": prompt}],
                system=MEMORY_SYSTEM_PROMPT,
                **(memory_kwargs or {}),
            )
            raw_output = result.get("response", "")
        except Exception as e:
            print(f"  Memory Model error at turn {turn_idx}: {e}")

        think, actions = parse_memory_output(raw_output)
        logs = execute_actions_on_memory(memory, actions)

        # If this turn has a memory_query, evaluate
        memory_query = user_part.get("memory_query")
        oracle_memory = user_part.get("oracle_memory_base")

        if memory_query and oracle_memory:
            ground_truth = memory_query.get("answer", "")
            difficulty = memory_query.get("difficulty", "")
            query_type = memory_query.get("type", "")
            question = memory_query.get("question", "")

            instance_id = f"{source}_{patient_id}_{turn_idx}"
            row = {
                "instance_id": instance_id,
                "source": source,
                "patient_id": patient_id,
                "turn_idx": turn_idx,
                "difficulty": difficulty,
                "query_type": query_type,
                "question": question,
                "ground_truth": ground_truth,
                "memory_model": memory_model,
                "judge_model": judge_model,
                "scenario": metadata.get("scenario", ""),
                "scenario_name": metadata.get("scenario_name", ""),
            }

            # Evaluate agent memory (retrieve top-K by relevance, not full dump)
            try:
                retrieved = memory.retrieve_by_relevance(top_k=15)
                retrieved_state: Dict[str, list] = {l: [] for l in VALID_LAYERS}
                for it in retrieved:
                    retrieved_state[it.layer].append({"content": it.content})
                agent_answer = answer_query_with_memory(
                    judge_model, retrieved_state, memory_query
                )
                agent_score = judge_consistency(judge_model, agent_answer, ground_truth)
                row["agent_answer"] = agent_answer
                row["agent_correct"] = agent_score
                row["agent_memory_state"] = json.dumps(
                    memory.get_state_dict(), ensure_ascii=False
                )
                row["agent_actions"] = json.dumps(actions, ensure_ascii=False)
                row["agent_think"] = think
                row["agent_n_retrieved"] = len(retrieved)
                row["agent_conflicts_detected"] = len(potential_conflicts)
            except Exception as e:
                row["agent_correct"] = -2
                row["agent_error"] = str(e)

            # Evaluate oracle memory
            if also_eval_oracle:
                try:
                    oracle_answer = answer_query_with_memory(
                        judge_model, oracle_memory, memory_query
                    )
                    oracle_score = judge_consistency(judge_model, oracle_answer, ground_truth)
                    row["oracle_answer"] = oracle_answer
                    row["oracle_correct"] = oracle_score
                except Exception as e:
                    row["oracle_correct"] = -2
                    row["oracle_error"] = str(e)

            results.append(row)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MED-MEM memory accuracy using Memory Agent Framework"
    )
    parser.add_argument("--cache_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--memory_model", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--max_patients_per_source", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallelism at patient level (memory agent is sequential per patient)")
    parser.add_argument("--memory_temperature", type=float, default=0.3)
    parser.add_argument("--memory_max_tokens", type=int, default=4096)
    parser.add_argument("--no_oracle", action="store_true",
                        help="Skip oracle memory evaluation")
    parser.add_argument("--no_reflection", action="store_true",
                        help="Disable session-end reflection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.judge_model is None:
        args.judge_model = args.memory_model

    os.makedirs(args.output_dir, exist_ok=True)
    traces_dir = os.path.join(args.output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    # Load done patient IDs
    done_patients = set()
    done_file = os.path.join(args.output_dir, "done_patients.txt")
    if os.path.exists(done_file):
        with open(done_file, "r") as f:
            done_patients = set(line.strip() for line in f if line.strip())

    # Load patients
    all_patients = []
    for cache_dir in args.cache_dirs:
        patients = load_patients_from_cache(
            cache_dir, args.max_patients_per_source, args.seed
        )
        print(f"  Loaded {len(patients)} patients from {cache_dir}")
        all_patients.extend(patients)

    # Filter done
    remaining = [
        p for p in all_patients
        if f"{p.get('_source', '')}_{p.get('patient_id', '')}" not in done_patients
    ]
    print(f"Total patients: {len(all_patients)}, remaining: {len(remaining)}")

    mem_kwargs = {"temperature": args.memory_temperature, "max_tokens": args.memory_max_tokens}
    all_results: List[Dict[str, Any]] = []

    def _process(patient):
        return process_patient(
            patient, args.memory_model, args.judge_model,
            mem_kwargs, also_eval_oracle=not args.no_oracle,
            enable_reflection=not args.no_reflection,
        )

    if args.workers <= 1:
        for p_idx, patient in enumerate(remaining):
            pid = patient.get("patient_id", "unknown")
            source = patient.get("_source", "unknown")
            print(f"\n[{p_idx+1}/{len(remaining)}] Patient {pid} ({source})")
            results = _process(patient)
            all_results.extend(results)

            for r in results:
                trace_path = os.path.join(traces_dir, f"{r['instance_id']}.json")
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(r, f, ensure_ascii=False, indent=2)

            with open(done_file, "a") as f:
                f.write(f"{source}_{pid}\n")

            # Periodic status
            valid = [r for r in all_results if r.get("agent_correct", -2) >= 0]
            if valid:
                acc = sum(r["agent_correct"] for r in valid) / len(valid)
                print(f"  → {len(results)} queries | running agent acc: {acc:.3f} ({len(valid)} samples)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, p): p for p in remaining}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Patients"):
                patient = futures[fut]
                pid = patient.get("patient_id", "unknown")
                source = patient.get("_source", "unknown")
                results = fut.result()
                all_results.extend(results)
                for r in results:
                    trace_path = os.path.join(traces_dir, f"{r['instance_id']}.json")
                    with open(trace_path, "w", encoding="utf-8") as f:
                        json.dump(r, f, ensure_ascii=False, indent=2)
                with open(done_file, "a") as f:
                    f.write(f"{source}_{pid}\n")

    # Load existing traces
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            iid = fn[:-5]
            if iid not in {r["instance_id"] for r in all_results}:
                try:
                    with open(os.path.join(traces_dir, fn), "r", encoding="utf-8") as f:
                        all_results.append(json.load(f))
                except Exception:
                    pass

    if not all_results:
        print("No results to analyze.")
        return

    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(args.output_dir, "results.parquet")
    results_df.to_parquet(summary_path, index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Enhanced MED-MEM Agent Memory Accuracy Evaluation")
    print(f"{'='*70}")
    print(f"Memory Model: {args.memory_model}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Total samples: {len(results_df)}")
    if "agent_n_retrieved" in results_df.columns:
        print(f"Avg memories retrieved per query: "
              f"{results_df['agent_n_retrieved'].mean():.1f}")
    if "agent_conflicts_detected" in results_df.columns:
        print(f"Avg conflicts auto-resolved: "
              f"{results_df['agent_conflicts_detected'].mean():.2f}")

    # Agent accuracy
    valid_agent = results_df[results_df["agent_correct"] >= 0]
    if len(valid_agent) > 0:
        agent_acc = valid_agent["agent_correct"].mean()
        print(f"\nAgent Memory Accuracy: {agent_acc:.4f} "
              f"({int(valid_agent['agent_correct'].sum())}/{len(valid_agent)})")

        # By source
        print("\n  By source:")
        for src in sorted(valid_agent["source"].unique()):
            sub = valid_agent[valid_agent["source"] == src]
            acc = sub["agent_correct"].mean()
            print(f"    {src:30s}  {acc:.4f} ({int(sub['agent_correct'].sum())}/{len(sub)})")

        # By difficulty
        print("\n  By difficulty:")
        for diff in sorted(valid_agent["difficulty"].unique()):
            sub = valid_agent[valid_agent["difficulty"] == diff]
            acc = sub["agent_correct"].mean()
            print(f"    {diff:30s}  {acc:.4f} ({int(sub['agent_correct'].sum())}/{len(sub)})")

    # Oracle accuracy
    if "oracle_correct" in results_df.columns:
        valid_oracle = results_df[results_df["oracle_correct"] >= 0]
        if len(valid_oracle) > 0:
            oracle_acc = valid_oracle["oracle_correct"].mean()
            print(f"\nOracle Memory Accuracy: {oracle_acc:.4f} "
                  f"({int(valid_oracle['oracle_correct'].sum())}/{len(valid_oracle)})")

            print("\n  By source:")
            for src in sorted(valid_oracle["source"].unique()):
                sub = valid_oracle[valid_oracle["source"] == src]
                acc = sub["oracle_correct"].mean()
                print(f"    {src:30s}  {acc:.4f}")

    # Gap analysis
    if "oracle_correct" in results_df.columns:
        both_valid = results_df[
            (results_df["agent_correct"] >= 0) & (results_df["oracle_correct"] >= 0)
        ]
        if len(both_valid) > 0:
            gap = both_valid["oracle_correct"].mean() - both_valid["agent_correct"].mean()
            print(f"\nGap (oracle - agent): {gap:+.4f}")
            print("  This gap represents room for improvement via RL training.")

    print(f"\nResults saved to {summary_path}")
    print(f"Traces saved to {traces_dir}/")


if __name__ == "__main__":
    main()
