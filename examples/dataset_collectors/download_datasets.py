"""
Dataset Download Helper

Downloads the five medical dialogue datasets from HuggingFace / GitHub.
Usage:
    python download_datasets.py --output_dir /path/to/raw_datasets [--dataset DATASET_NAME]
"""

import os
import sys
import json
import argparse
from pathlib import Path

DATASET_REGISTRY = {
    "meddialog_cn": {
        "name": "MedDialog-CN",
        "hf_repo": "medical_dialog",
        "hf_subset": "processed.zh",
        "description": "3.4M Chinese medical dialogues from haodf.com, 11.3M turns, 172 departments",
        "focus": "Identity & Medical History Memory"
    },
    "cmtmedqa": {
        "name": "CMtMedQA",
        "hf_repo": "Suprit/CMtMedQA",
        "hf_subset": None,
        "description": "70K real doctor-patient multi-turn dialogues",
        "focus": "Active Inquiry & State Tracking"
    },
    "imcs21": {
        "name": "IMCS-21",
        "hf_repo": "DUTIR-BioNLP/IMCS-V2-MRG",
        "hf_subset": None,
        "description": "60K+ sessions with fine-grained annotations from Fudan University",
        "focus": "Clinical Pathway Memory"
    },
    "huatuo26m": {
        "name": "Huatuo-26M",
        "hf_repo": "FreedomIntelligence/Huatuo26M-Lite",
        "hf_subset": None,
        "description": "26M QA pairs covering online consultation, encyclopedia, knowledge base",
        "focus": "Medical Knowledge Memory"
    },
    "lcmdc": {
        "name": "LCMDC",
        "hf_repo": "DUTIR-BioNLP/Dialogue-Triage",
        "hf_subset": None,
        "description": "430K triage + 200K diagnosis + 470K consultation",
        "focus": "Long-term Treatment Closed Loop"
    }
}


def download_dataset(dataset_key: str, output_dir: str, max_samples: int = None):
    """
    Download a single dataset using HuggingFace datasets library.
    Falls back to manual download if needed.
    """
    info = DATASET_REGISTRY[dataset_key]
    save_dir = os.path.join(output_dir, dataset_key)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {info['name']}")
    print(f"  HF Repo: {info['hf_repo']}")
    print(f"  Description: {info['description']}")
    print(f"  Focus: {info['focus']}")
    print(f"  Save to: {save_dir}")
    print(f"{'='*60}")

    try:
        from datasets import load_dataset

        if dataset_key == "meddialog_cn":
            ds = load_dataset(info["hf_repo"], info["hf_subset"], trust_remote_code=True)
        elif dataset_key == "cmtmedqa":
            ds = load_dataset(info["hf_repo"], trust_remote_code=True)
        elif dataset_key == "imcs21":
            ds = load_dataset(info["hf_repo"], trust_remote_code=True)
        elif dataset_key == "huatuo26m":
            ds = load_dataset(info["hf_repo"], trust_remote_code=True)
        elif dataset_key == "lcmdc":
            ds = load_dataset(info["hf_repo"], trust_remote_code=True)
        else:
            print(f"Unknown dataset: {dataset_key}")
            return

        # Save dataset
        for split_name in ds:
            split_data = ds[split_name]
            if max_samples and len(split_data) > max_samples:
                split_data = split_data.select(range(max_samples))

            output_path = os.path.join(save_dir, f"{split_name}.json")
            split_data.to_json(output_path, force_ascii=False)
            print(f"  Saved {split_name}: {len(split_data)} samples -> {output_path}")

        print(f"✓ {info['name']} downloaded successfully")

    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Failed to download {info['name']}: {e}")
        print(f"  You may need to manually download from: https://huggingface.co/datasets/{info['hf_repo']}")
        print(f"  And place the data files in: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download medical dialogue datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to save downloaded datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASET_REGISTRY.keys()) + ["all"],
                        help="Which dataset to download (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples per split (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset is None or args.dataset == "all":
        datasets_to_download = list(DATASET_REGISTRY.keys())
    else:
        datasets_to_download = [args.dataset]

    for dataset_key in datasets_to_download:
        download_dataset(dataset_key, args.output_dir, args.max_samples)

    print(f"\n{'='*60}")
    print("All downloads complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

