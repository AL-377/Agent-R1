"""
Unified Run Script for All Dataset Examiner Agents
====================================================

Downloads (optional) and processes all five medical dialogue datasets
through the examiner agent pipeline with multi-session synthesis.

Usage:
    # Process all datasets (multi-session enabled by default)
    python run_all.py --input_dir /path/to/raw_datasets --output_dir /path/to/output

    # Process specific dataset
    python run_all.py --input_dir /path/to/raw --output_dir /path/to/out --dataset meddialog_cn

    # Download first, then process
    python run_all.py --input_dir /path/to/raw --output_dir /path/to/out --download

    # Disable multi-session synthesis
    python run_all.py --input_dir /path/to/raw --output_dir /path/to/out --no_multi_session

    # Test with small sample
    python run_all.py --input_dir /path/to/raw --output_dir /path/to/out --max_dialogues 10
"""

import os
import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset_collectors.meddialog_cn_examiner import MedDialogCNExaminer
from examples.dataset_collectors.cmtmedqa_examiner import CMtMedQAExaminer
from examples.dataset_collectors.imcs21_examiner import IMCS21Examiner
from examples.dataset_collectors.huatuo26m_examiner import Huatuo26MExaminer
from examples.dataset_collectors.lcmdc_examiner import LCMDCExaminer
from examples.dataset_collectors.base_examiner import DEFAULT_MODEL


# ── Dataset registry ─────────────────────────────────────────────────

DATASET_CONFIGS = {
    "meddialog_cn": {
        "name": "MedDialog-CN",
        "class": MedDialogCNExaminer,
        "input_subdir": "meddialog_cn",
        "output_file": "meddialog_cn_examiner.json",
        "focus": "身份与病史记忆",
        "sessions_range": (2, 5),
        "extra_kwargs": {},
    },
    "cmtmedqa": {
        "name": "CMtMedQA",
        "class": CMtMedQAExaminer,
        "input_subdir": "cmtmedqa",
        "output_file": "cmtmedqa_examiner.json",
        "focus": "主动问诊与状态追踪",
        "sessions_range": (2, 4),
        "extra_kwargs": {},
    },
    "imcs21": {
        "name": "IMCS-21 / KaMed",
        "class": IMCS21Examiner,
        "input_subdir": "imcs21",
        "output_file": "imcs21_examiner.json",
        "focus": "临床路径记忆",
        "sessions_range": (2, 5),
        "extra_kwargs": {},
    },
    "huatuo26m": {
        "name": "Huatuo-26M",
        "class": Huatuo26MExaminer,
        "input_subdir": "huatuo26m",
        "output_file": "huatuo26m_examiner.json",
        "focus": "医学知识记忆",
        "sessions_range": (3, 6),
        "extra_kwargs": {},
    },
    "lcmdc": {
        "name": "LCMDC",
        "class": LCMDCExaminer,
        "input_subdir": "lcmdc",
        "output_file": "lcmdc_examiner.json",
        "focus": "长程诊疗闭环",
        "sessions_range": (2, 4),
        "extra_kwargs": {},
    },
}


def download_datasets(input_dir: str, datasets: list, max_samples: int = None):
    """Download datasets using the download helper."""
    from examples.dataset_collectors.download_datasets import download_dataset
    for ds_key in datasets:
        try:
            download_dataset(ds_key, input_dir, max_samples)
        except Exception as e:
            print(f"WARNING: Failed to download {ds_key}: {e}")
            print(f"  Please download manually and place in: {os.path.join(input_dir, ds_key)}")


def process_dataset(
    ds_key: str,
    input_dir: str,
    output_dir: str,
    model: str,
    max_dialogues: int = None,
    workers: int = 4,
    enable_multi_session: bool = True,
    sessions_range: tuple = None,
    use_llm_profile: bool = False,
):
    """Process a single dataset through its examiner agent."""
    config = DATASET_CONFIGS[ds_key]

    input_path = os.path.join(input_dir, config["input_subdir"])
    output_path = os.path.join(output_dir, config["output_file"])

    # Check if input exists
    if not os.path.exists(input_path):
        # Try common file patterns
        for ext in [".json", ".jsonl"]:
            alt_path = input_path + ext
            if os.path.exists(alt_path):
                input_path = alt_path
                break
        else:
            print(f"\n⚠ Input not found: {input_path}")
            print(f"  Please download {config['name']} first, or use --download flag")
            return False

    # Use dataset-specific sessions_range if not overridden
    if sessions_range is None:
        sessions_range = config.get("sessions_range", (2, 5))

    print(f"\n{'='*60}")
    print(f"Processing: {config['name']}")
    print(f"  Focus: {config['focus']}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Multi-session: {enable_multi_session}")
    if enable_multi_session:
        print(f"  Sessions/patient: {sessions_range}")
    print(f"{'='*60}")

    start_time = time.time()

    agent_cls = config["class"]
    extra_kwargs = config.get("extra_kwargs", {})

    agent = agent_cls(
        model=model,
        language="zh",
        max_workers=workers,
        enable_multi_session=enable_multi_session,
        sessions_range=sessions_range,
        use_llm_profile=use_llm_profile,
        **extra_kwargs,
    )
    agent.run(input_path, output_path, max_dialogues)

    elapsed = time.time() - start_time
    print(f"\n✓ {config['name']} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified runner for all medical dataset examiner agents"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing raw datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for examiner output")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=list(DATASET_CONFIGS.keys()) + ["all"],
                        help="Which dataset to process (default: all)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_dialogues", type=int, default=None,
                        help="Max dialogues per dataset (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets before processing")
    parser.add_argument("--download_max_samples", type=int, default=None,
                        help="Max samples per split when downloading")
    # ── Multi-session synthesis options ──
    parser.add_argument("--no_multi_session", action="store_true",
                        help="Disable multi-session synthesis (each dialogue = one patient)")
    parser.add_argument("--sessions_min", type=int, default=None,
                        help="Override min sessions per patient (default: per-dataset config)")
    parser.add_argument("--sessions_max", type=int, default=None,
                        help="Override max sessions per patient (default: per-dataset config)")
    parser.add_argument("--use_llm_profile", action="store_true",
                        help="Use LLM to generate patient profiles (slower, richer)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "all":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    # Sessions range override
    sessions_range_override = None
    if args.sessions_min is not None and args.sessions_max is not None:
        sessions_range_override = (args.sessions_min, args.sessions_max)

    # Download if requested
    if args.download:
        print("\n" + "="*60)
        print("STEP 1: Downloading datasets")
        print("="*60)
        download_datasets(args.input_dir, datasets, args.download_max_samples)

    # Process
    print("\n" + "="*60)
    step = "STEP 2" if args.download else "STEP 1"
    print(f"{step}: Processing datasets through Examiner Agents")
    print(f"  Multi-session synthesis: {'DISABLED' if args.no_multi_session else 'ENABLED'}")
    print("="*60)

    results = {}
    total_start = time.time()

    for ds_key in datasets:
        success = process_dataset(
            ds_key=ds_key,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_dialogues=args.max_dialogues,
            workers=args.workers,
            enable_multi_session=not args.no_multi_session,
            sessions_range=sessions_range_override,
            use_llm_profile=args.use_llm_profile,
        )
        results[ds_key] = success

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ds_key, success in results.items():
        status = "✓" if success else "✗"
        name = DATASET_CONFIGS[ds_key]["name"]
        focus = DATASET_CONFIGS[ds_key]["focus"]
        print(f"  {status} {name:<20s} ({focus})")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

