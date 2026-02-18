"""
MedDialog-CN Examiner Agent
============================

Dataset: MedDialog-CN (UCSD / haodf.com)
Scale:   3.4M dialogues, 11.3M turns, 172 departments
Focus:   身份与病史记忆 — 每条数据均包含结构化的"患者状况与病史"描述

Multi-session synthesis strategy:
  按科室（department）分组 → 同一科室的多条独立问诊合并为同一患者的多次随访。
  典型场景：慢性病随访管理、复杂疾病诊疗。

Usage:
    python meddialog_cn_examiner.py \\
        --input_path /path/to/meddialog_cn/train.json \\
        --output_path /path/to/output/meddialog_cn_examiner.json \\
        --max_dialogues 100 \\
        --model gpt-4o-2024-11-20
"""

import json
import os
import sys
import argparse
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset_collectors.base_examiner import BaseExaminerAgent, DEFAULT_MODEL


class MedDialogCNExaminer(BaseExaminerAgent):
    """
    Examiner agent for MedDialog-CN dataset.

    Focus: Identity & Medical History Memory
    - Rich patient condition/history descriptions
    - Multi-department coverage (172 departments)

    Multi-session grouping: by department (科室).
    Same-department dialogues are merged into one patient's multiple visits.
    """

    # ── abstract method implementations ──────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """
        Load MedDialog-CN data.

        Supports:
          - Single JSON file (list of records)
          - JSONL file (one record per line)
          - Directory of JSON files
        """
        records = []

        if os.path.isdir(input_path):
            for fname in sorted(os.listdir(input_path)):
                if fname.endswith('.json') or fname.endswith('.jsonl'):
                    fpath = os.path.join(input_path, fname)
                    records.extend(self._load_file(fpath))
        else:
            records = self._load_file(input_path)

        print(f"[MedDialogCN] Loaded {len(records)} raw dialogues")
        return records

    def _load_file(self, fpath: str) -> List[Any]:
        records = []
        if fpath.endswith('.jsonl'):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # HuggingFace datasets format: might have splits
                for key in ["train", "test", "validation", "data"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                if not records:
                    records = [data]
        return records

    def normalize_dialogue(self, raw_record: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Normalize a MedDialog-CN record.

        Actual data format:
          ["病人：强制性脊柱炎，晚上...", "医生：应该没有问题..."]
          — a plain list of strings with "病人：" / "医生：" prefixes.

        Also supports dict formats for compatibility:
          Format A: {"utterances": ["患者：...", "医生：...", ...]}
          Format B: {"description": "...", "dialogue": [...]}
        """
        messages = []
        metadata = {}

        # Handle the actual format: list of strings
        if isinstance(raw_record, list):
            content_str = " ".join(str(s)[:100] for s in raw_record[:3])
            dialogue_id = hashlib.md5(content_str.encode()).hexdigest()[:12]
            for utt in raw_record:
                if isinstance(utt, str):
                    utt = utt.strip()
                    if utt:
                        role, content = self._parse_utterance(utt)
                        if content:
                            messages.append({"role": role, "content": content})
            return dialogue_id, messages, metadata

        # Dict formats (HF / alternative)
        dialogue_id = raw_record.get("id", raw_record.get("dialogue_id", None))
        if dialogue_id is None:
            content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
            dialogue_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

        description = raw_record.get("description", raw_record.get("patient_info", ""))
        if description:
            metadata["patient_description"] = description
            messages.append({"role": "patient", "content": description})

        utterances = raw_record.get("utterances", raw_record.get("dialogue", []))

        if isinstance(utterances, list):
            if len(utterances) > 0 and isinstance(utterances[0], str):
                for utt in utterances:
                    utt = utt.strip()
                    if not utt:
                        continue
                    role, content = self._parse_utterance(utt)
                    if content:
                        messages.append({"role": role, "content": content})

            elif len(utterances) > 0 and isinstance(utterances[0], dict):
                for utt in utterances:
                    speaker = utt.get("speaker", utt.get("role", "patient")).lower()
                    text = utt.get("text", utt.get("content", utt.get("utterance", "")))
                    role = "doctor" if speaker in ("doctor", "医生", "assistant") else "patient"
                    if text.strip():
                        messages.append({"role": role, "content": text.strip()})

        dept = raw_record.get("department", raw_record.get("科室", ""))
        if dept:
            metadata["department"] = dept

        return dialogue_id, messages, metadata

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Group by department (科室).
        Same-department dialogues become candidates for the same patient.
        """
        meta = normalised_record.get("metadata", {})
        dept = meta.get("department", "")
        if dept:
            return dept
        # Fallback: try to extract department hint from first patient message
        msgs = normalised_record.get("messages", [])
        if msgs:
            first_content = msgs[0].get("content", "")[:100]
            # Common department keywords
            for kw in ["内科", "外科", "妇科", "儿科", "骨科", "皮肤科",
                        "眼科", "耳鼻喉", "口腔", "肿瘤", "神经", "心血管",
                        "消化", "呼吸", "内分泌", "泌尿", "精神", "康复"]:
                if kw in first_content:
                    return kw
        return "综合科"

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_utterance(text: str) -> Tuple[str, str]:
        """Parse a single utterance string, detecting speaker from prefix."""
        prefixes_doctor = ["医生：", "医生:", "doctor:", "Doctor:", "医师：", "医师:"]
        prefixes_patient = ["患者：", "患者:", "patient:", "Patient:", "病人：", "病人:"]

        for prefix in prefixes_doctor:
            if text.startswith(prefix):
                return "doctor", text[len(prefix):].strip()
        for prefix in prefixes_patient:
            if text.startswith(prefix):
                return "patient", text[len(prefix):].strip()

        # No prefix detected — default patient
        return "patient", text


def main():
    parser = argparse.ArgumentParser(description="MedDialog-CN Examiner Agent")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to MedDialog-CN data (JSON/JSONL file or directory)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--max_dialogues", type=int, default=None,
                        help="Max dialogues to process (for testing)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--no_multi_session", action="store_true",
                        help="Disable multi-session synthesis")
    parser.add_argument("--sessions_min", type=int, default=2,
                        help="Min sessions per patient")
    parser.add_argument("--sessions_max", type=int, default=5,
                        help="Max sessions per patient")
    args = parser.parse_args()

    agent = MedDialogCNExaminer(
        model=args.model,
        language="zh",
        max_workers=args.workers,
        enable_multi_session=not args.no_multi_session,
        sessions_range=(args.sessions_min, args.sessions_max),
    )
    agent.run(args.input_path, args.output_path, args.max_dialogues)


if __name__ == "__main__":
    main()
