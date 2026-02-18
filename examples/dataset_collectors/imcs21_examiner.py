"""
IMCS-21 / KaMed Examiner Agent
================================

Dataset: IMCS-21 (Intelligent Medical Consultation System) / KaMed
Focus:   临床路径记忆 — 细粒度标注（诊断、报告、实体）

Multi-session synthesis strategy:
  按诊断（diagnosis）分组 → 同一诊断的多条独立问诊合并为同一患者的多次就诊。
  典型场景：复杂疾病诊疗、术后康复管理。

  核心特殊处理：
  - 支持 dict-of-dicts 格式（key=对话ID, value=对话记录）
  - 支持 "D"/"P" 单字母角色标识
  - _parse_dialogue_string() 解析纯文本对话

Usage:
    python imcs21_examiner.py \\
        --input_path /path/to/imcs21/data.json \\
        --output_path /path/to/output/imcs21_examiner.json
"""

import json
import os
import re
import sys
import argparse
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset_collectors.base_examiner import BaseExaminerAgent, DEFAULT_MODEL


class IMCS21Examiner(BaseExaminerAgent):
    """
    Examiner agent for IMCS-21 / KaMed dataset.

    Focus: Clinical Pathway Memory
    - Fine-grained annotations (diagnosis, report, entities)
    - Structured medical consultation records

    Multi-session grouping: by diagnosis.
    Same-diagnosis dialogues are merged into one patient's multiple visits.
    """

    # ── abstract method implementations ──────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """
        Load IMCS-21 data.

        Supports:
          - Single JSON file (list, dict-of-dicts, or HF format)
          - JSONL file
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
        print(f"[IMCS-21] Loaded {len(records)} raw records")
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
                # Check for HF splits
                for key in ["train", "test", "validation", "data"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                if not records:
                    # dict-of-dicts format: {dialogue_id: {dialogue_data}}
                    first_val = next(iter(data.values()), None) if data else None
                    if isinstance(first_val, dict):
                        for did, ddata in data.items():
                            if isinstance(ddata, dict):
                                ddata["_dialogue_id"] = did
                                records.append(ddata)
                    else:
                        records = [data]
        return records

    def normalize_dialogue(self, raw_record: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Normalize an IMCS-21 record.

        Expected formats:
          Format A: {"dialogue": [{"role": "D"/"P", "content": "..."}], "diagnosis": "...", "report": "..."}
          Format B: {"dialogue": "P: ... D: ...", "diagnosis": "..."}
          Format C: {"utterances": [...], "entities": {...}}
        """
        dialogue_id = raw_record.get(
            "_dialogue_id",
            raw_record.get("id", raw_record.get("dialogue_id", None))
        )
        if dialogue_id is None:
            content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
            dialogue_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

        messages = []
        metadata = {}

        # Extract fine-grained annotations
        for key in ["diagnosis", "report", "entities", "disease", "department"]:
            val = raw_record.get(key, "")
            if val:
                metadata[key] = val

        # Parse dialogue
        dialogue_raw = raw_record.get("dialogue", raw_record.get("utterances", []))

        if isinstance(dialogue_raw, str):
            # Plain text dialogue — parse it
            messages = self._parse_dialogue_string(dialogue_raw)
        elif isinstance(dialogue_raw, list):
            if len(dialogue_raw) > 0 and isinstance(dialogue_raw[0], str):
                # List of strings
                for utt in dialogue_raw:
                    utt = utt.strip()
                    if not utt:
                        continue
                    role, content = self._parse_single_utterance(utt)
                    if content:
                        messages.append({"role": role, "content": content})
            elif len(dialogue_raw) > 0 and isinstance(dialogue_raw[0], dict):
                # List of dicts
                for turn in dialogue_raw:
                    role_raw = turn.get("role", turn.get("speaker", "P")).strip()
                    content = turn.get("content", turn.get("text", turn.get("utterance", "")))
                    role = self._map_role(role_raw)
                    if content.strip():
                        messages.append({"role": role, "content": content.strip()})

        return dialogue_id, messages, metadata

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Group by diagnosis (诊断).
        Same-diagnosis dialogues become candidates for the same patient.
        """
        meta = normalised_record.get("metadata", {})
        # Priority: diagnosis > disease > department
        diag = meta.get("diagnosis", "")
        if isinstance(diag, str) and diag.strip():
            return diag.strip()
        if isinstance(diag, list) and diag:
            return str(diag[0])
        disease = meta.get("disease", "")
        if disease:
            return str(disease)
        dept = meta.get("department", "")
        if dept:
            return dept
        return "未分类"

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _map_role(role_raw: str) -> str:
        """Map various role labels to doctor/patient."""
        role_lower = role_raw.lower().strip()
        if role_lower in ("d", "doctor", "医生", "assistant"):
            return "doctor"
        if role_lower in ("p", "patient", "患者", "user", "病人"):
            return "patient"
        return "patient"

    @staticmethod
    def _parse_single_utterance(text: str) -> Tuple[str, str]:
        """Parse a single utterance with possible role prefix."""
        prefixes_doctor = ["D:", "D：", "医生:", "医生：", "doctor:", "Doctor:"]
        prefixes_patient = ["P:", "P：", "患者:", "患者：", "patient:", "Patient:", "病人:", "病人："]

        for prefix in prefixes_doctor:
            if text.startswith(prefix):
                return "doctor", text[len(prefix):].strip()
        for prefix in prefixes_patient:
            if text.startswith(prefix):
                return "patient", text[len(prefix):].strip()

        return "patient", text

    @staticmethod
    def _parse_dialogue_string(text: str) -> List[Dict[str, str]]:
        """
        Parse a plain text dialogue string into turns.
        Handles formats like:
          "P: 你好 D: 你好，请问有什么不舒服 P: 我头疼"
        """
        messages = []
        # Split by role markers
        parts = re.split(r'(?:^|\s)((?:P|D|患者|医生|patient|doctor)\s*[:：])', text, flags=re.IGNORECASE)

        current_role = "patient"
        current_text = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Check if this is a role marker
            role_match = re.match(r'^(P|D|患者|医生|patient|doctor)\s*[:：]$', part, re.IGNORECASE)
            if role_match:
                # Save previous turn
                if current_text.strip():
                    messages.append({"role": current_role, "content": current_text.strip()})
                marker = role_match.group(1).upper()
                current_role = "doctor" if marker in ("D", "医生", "DOCTOR") else "patient"
                current_text = ""
            else:
                current_text += " " + part

        # Don't forget last turn
        if current_text.strip():
            messages.append({"role": current_role, "content": current_text.strip()})

        return messages


def main():
    parser = argparse.ArgumentParser(description="IMCS-21 / KaMed Examiner Agent")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_multi_session", action="store_true",
                        help="Disable multi-session synthesis")
    parser.add_argument("--sessions_min", type=int, default=2)
    parser.add_argument("--sessions_max", type=int, default=4)
    args = parser.parse_args()

    agent = IMCS21Examiner(
        model=args.model,
        language="zh",
        max_workers=args.workers,
        enable_multi_session=not args.no_multi_session,
        sessions_range=(args.sessions_min, args.sessions_max),
    )
    agent.run(args.input_path, args.output_path, args.max_dialogues)


if __name__ == "__main__":
    main()
