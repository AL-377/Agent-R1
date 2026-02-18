"""
Dataset Download Helper

Downloads the five medical dialogue datasets from HuggingFace / GitHub.

Compatible with datasets library v3.x and v4.x (trust_remote_code removed in v4).
Uses multiple fallback strategies:
  1. huggingface_hub direct file download (most reliable)
  2. HuggingFace load_dataset with parquet fallback
  3. Google Drive download (for MedDialog-CN)
  4. GitHub clone (for IMCS-21)

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
        "hf_repo": "UCSD26/medical_dialog",
        "hf_subset": "processed.zh",
        "description": "3.4M Chinese medical dialogues from haodf.com, 11.3M turns, 172 departments",
        "focus": "Identity & Medical History Memory",
        "gdrive_files": {
            "train": "https://drive.google.com/uc?export=download&id=1AaDJoHaiHAwEZwtskRH8oL1UP4FRgmgx",
            "validation": "https://drive.google.com/uc?export=download&id=1TvfZCmQqP1kURIfEinOcj5VOPelTuGwI",
            "test": "https://drive.google.com/uc?export=download&id=1pmmG95Yl6mMXRXDDSRb9-bYTxOE7ank5",
        },
    },
    "cmtmedqa": {
        "name": "CMtMedQA",
        "hf_repo": "Suprit/CMtMedQA",
        "hf_subset": None,
        "description": "70K real doctor-patient multi-turn dialogues",
        "focus": "Active Inquiry & State Tracking",
        "hf_direct_files": [
            {"filename": "CMtMedQA.json", "save_as": "train.json"},
        ],
    },
    "imcs21": {
        "name": "IMCS-21",
        "hf_repo": None,
        "github_repo": "lemuria-wchen/imcs21",
        "github_data_path": "dataset",
        "description": "4,116 annotated medical consultation records covering 10 pediatric diseases",
        "focus": "Clinical Pathway Memory",
    },
    "huatuo26m": {
        "name": "Huatuo-26M",
        "hf_repo": "FreedomIntelligence/Huatuo26M-Lite",
        "hf_subset": None,
        "description": "178K refined QA pairs (Lite version), covering online consultation, encyclopedia",
        "focus": "Medical Knowledge Memory",
        "hf_direct_files": [
            {"filename": "format_data.jsonl", "save_as": "train.jsonl"},
        ],
    },
    "lcmdc": {
        "name": "LCMDC",
        "hf_repo": None,
        "github_repo": None,
        "description": "430K triage + 200K diagnosis + 470K consultation (from 120ask.com)",
        "focus": "Long-term Treatment Closed Loop",
        "manual_url": "https://arxiv.org/abs/2410.03521",
        "manual_note": (
            "LCMDC is not publicly hosted on HuggingFace or GitHub.\n"
            "  Paper: https://arxiv.org/abs/2410.03521\n"
            "  Data source: https://www.120ask.com/ (Quick Doctor)\n"
            "  Contact the paper authors (Xi'an Jiaotong University) for data access.\n"
            "  Alternative: skip LCMDC with --dataset flag, e.g.:\n"
            "    python run_all.py --dataset meddialog_cn --input_dir ... --output_dir ..."
        ),
    },
}


# ── HuggingFace Hub direct file download ────────────────────────────

def _download_hf_file(repo_id: str, filename: str, save_path: str):
    """Download a single file from a HuggingFace dataset repo."""
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset",
        )
        import shutil
        shutil.copy2(cached, save_path)
        return True
    except ImportError:
        pass

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    return _download_large_file(url, save_path)


def _download_large_file(url: str, save_path: str):
    """Stream-download a large file with progress."""
    import requests
    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                mb = downloaded // (1024 * 1024)
                print(f"\r  Progress: {mb} MB ({pct}%)", end="", flush=True)
    print()
    return True


# ── Google Drive download ────────────────────────────────────────────

def _download_gdrive(file_id_or_url: str, save_path: str):
    """
    Download from Google Drive, handling the virus-scan confirmation page.
    Tries gdown first (most robust), then falls back to a two-step manual flow.
    """
    import re
    import requests

    if file_id_or_url.startswith("http"):
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', file_id_or_url)
        file_id = match.group(1) if match else file_id_or_url
    else:
        file_id = file_id_or_url

    # Strategy 1: gdown (handles virus-scan pages automatically)
    try:
        import gdown
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(gdrive_url, save_path, quiet=False, fuzzy=True)
        if os.path.exists(save_path) and os.path.getsize(save_path) > 1024:
            return True
        print("  gdown produced a small/empty file, trying manual download...")
    except ImportError:
        print("  gdown not installed, trying manual download...")
    except Exception as e:
        print(f"  gdown failed: {e}, trying manual download...")

    # Strategy 2: two-step manual flow
    # Step 1: request the old /uc URL to get the confirmation page with uuid
    session = requests.Session()
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"  Step 1: Fetching confirmation page...")
    resp = session.get(initial_url, timeout=60)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        html = resp.content.decode("utf-8", errors="ignore")

        # Parse <form action="..."> and all <input type="hidden" name="..." value="...">
        action_match = re.search(r'<form[^>]+action="([^"]+)"', html)
        if not action_match:
            raise RuntimeError(
                "Could not parse Google Drive confirmation page. "
                "Install gdown (`pip install gdown`) for reliable downloads."
            )
        action_url = action_match.group(1).replace("&amp;", "&")

        params = {}
        for inp in re.finditer(
            r'<input[^>]*\bname="([^"]+)"[^>]*\bvalue="([^"]*)"', html
        ):
            params[inp.group(1)] = inp.group(2)

        # Step 2: submit the form to get the actual file
        print(f"  Step 2: Submitting confirmation (params: {list(params.keys())})...")
        resp = session.get(action_url, params=params, stream=True, timeout=300)
        resp.raise_for_status()

        # If still HTML, give up
        if "text/html" in resp.headers.get("content-type", ""):
            raise RuntimeError(
                "Google Drive still returned HTML after confirmation. "
                "Install gdown (`pip install gdown`) or download manually."
            )
    elif resp.headers.get("content-length", "0") != "0":
        pass  # direct download worked (small file or already confirmed)

    _write_stream(resp, save_path)
    _verify_not_html(save_path)
    return True


def _write_stream(resp, save_path: str):
    """Write a streaming response to disk with progress."""
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                mb = downloaded // (1024 * 1024)
                pct = downloaded * 100 // total
                print(f"\r  Progress: {mb} MB ({pct}%)", end="", flush=True)
    print()


def _verify_not_html(save_path: str):
    """Raise if the downloaded file is actually an HTML error page."""
    if os.path.getsize(save_path) < 4096:
        with open(save_path, "r", errors="ignore") as f:
            head = f.read(500)
        if "<html" in head.lower() or "<!doctype" in head.lower():
            os.remove(save_path)
            raise RuntimeError(
                "Google Drive returned an HTML page instead of data. "
                "Install gdown (`pip install gdown`) or download manually."
            )


# ── GitHub clone ─────────────────────────────────────────────────────

def _download_github_dataset(github_repo: str, data_path: str, save_dir: str):
    """Download dataset files from a GitHub repo using git sparse checkout."""
    import subprocess
    import tempfile
    import shutil

    clone_url = f"https://github.com/{github_repo}.git"
    print(f"  Cloning from GitHub: {clone_url} (path: {data_path})")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", clone_url, tmpdir],
                check=True, capture_output=True, text=True, timeout=120,
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", data_path],
                cwd=tmpdir, check=True, capture_output=True, text=True, timeout=30,
            )
            subprocess.run(
                ["git", "checkout"],
                cwd=tmpdir, check=True, capture_output=True, text=True, timeout=60,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  Sparse checkout failed, trying full clone...")
            shutil.rmtree(tmpdir, ignore_errors=True)
            os.makedirs(tmpdir, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, tmpdir],
                check=True, capture_output=True, text=True, timeout=300,
            )

        src = os.path.join(tmpdir, data_path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Path '{data_path}' not found in cloned repo")

        if os.path.isdir(src):
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(save_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(save_dir, os.path.basename(src)))

    print(f"  GitHub download complete -> {save_dir}")


# ── Main download orchestrator ───────────────────────────────────────

def download_dataset(dataset_key: str, output_dir: str, max_samples: int = None):
    """
    Download a single dataset using the best available strategy:
      1. Direct HF file download (CMtMedQA, Huatuo26M)
      2. Google Drive download (MedDialog-CN)
      3. GitHub clone (IMCS-21)
      4. Manual instructions (LCMDC)
    """
    info = DATASET_REGISTRY[dataset_key]
    save_dir = os.path.join(output_dir, dataset_key)
    os.makedirs(save_dir, exist_ok=True)

    source = info.get("hf_repo") or info.get("github_repo") or info.get("manual_url", "N/A")
    print(f"\n{'='*60}")
    print(f"Downloading: {info['name']}")
    print(f"  Source: {source}")
    print(f"  Description: {info['description']}")
    print(f"  Focus: {info['focus']}")
    print(f"  Save to: {save_dir}")
    print(f"{'='*60}")

    # Strategy A: Direct HF file download (most reliable for repos with raw files)
    hf_direct = info.get("hf_direct_files")
    hf_repo = info.get("hf_repo")
    if hf_direct and hf_repo:
        try:
            for finfo in hf_direct:
                save_path = os.path.join(save_dir, finfo["save_as"])
                print(f"  Downloading {finfo['filename']} from {hf_repo}...")
                _download_hf_file(hf_repo, finfo["filename"], save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                print(f"  Saved: {save_path} ({size_mb:.1f} MB)")
            print(f"✓ {info['name']} downloaded successfully")
            return
        except Exception as e:
            print(f"  Direct HF download failed: {e}")
            print(f"  Trying fallback methods...")

    # Strategy B: Google Drive download (MedDialog-CN)
    gdrive_files = info.get("gdrive_files")
    if gdrive_files:
        try:
            for split_name, gdrive_url in gdrive_files.items():
                save_path = os.path.join(save_dir, f"{split_name}.json")
                print(f"  Downloading {split_name} split from Google Drive...")
                _download_gdrive(gdrive_url, save_path)
                size_mb = os.path.getsize(save_path) / (1024 * 1024)
                print(f"  Saved: {save_path} ({size_mb:.1f} MB)")
            print(f"✓ {info['name']} downloaded successfully via Google Drive")
            return
        except Exception as e:
            print(f"  Google Drive download failed: {e}")
            print(f"  Trying fallback methods...")

    # Strategy C: GitHub clone (IMCS-21)
    github_repo = info.get("github_repo")
    github_data_path = info.get("github_data_path")
    if github_repo and github_data_path:
        try:
            _download_github_dataset(github_repo, github_data_path, save_dir)
            print(f"✓ {info['name']} downloaded successfully via GitHub")
            return
        except Exception as e:
            print(f"  GitHub download failed: {e}")

    # Strategy D: Manual instructions
    manual_note = info.get("manual_note")
    if manual_note:
        print(f"\nWARNING: Could not auto-download {info['name']}.")
        print(f"  {manual_note}")
        print(f"  Place data files in: {save_dir}")
    else:
        manual_url = info.get("manual_url", f"https://huggingface.co/datasets/{hf_repo}" if hf_repo else "")
        print(f"\nWARNING: Could not auto-download {info['name']}.")
        if manual_url:
            print(f"  Please download manually from: {manual_url}")
        print(f"  Place data files in: {save_dir}")
        print(f"  Expected format: JSON or JSONL files with dialogue records.")


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

    success_count = 0
    for dataset_key in datasets_to_download:
        try:
            download_dataset(dataset_key, args.output_dir, args.max_samples)
            success_count += 1
        except Exception as e:
            print(f"\nERROR downloading {dataset_key}: {e}")

    print(f"\n{'='*60}")
    print(f"Downloads complete: {success_count}/{len(datasets_to_download)} succeeded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

