"""
LCMDC Examiner Agent
=====================

Dataset: LCMDC (Large-scale Chinese Medical Dialogue Corpus)
Focus:   长程诊疗闭环 — 分诊→诊断→问诊多阶段

Multi-session synthesis strategy:
  LCMDC 天然包含三个子集（triage / diagnosis / consultation），
  _chain_subsets() 按 patient_id 将同一患者的分诊→诊断→问诊串联为
  跨阶段多会话记录，插入 "---诊疗分割线---"。
  对于无法匹配的单独子集记录，退化为按 department 分组合并。
  典型场景：复杂疾病诊疗、术后康复管理、罕见病鉴别诊断。

Usage:
    python lcmdc_examiner.py \\
        --input_path /path/to/lcmdc/ \\
        --output_path /path/to/output/lcmdc_examiner.json
"""

import json
import os
import sys
import argparse
import hashlib
import random
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.dataset_collectors.base_examiner import BaseExaminerAgent, DEFAULT_MODEL


# LCMDC subset names
LCMDC_SUBSETS = ["triage", "diagnosis", "consultation"]
SUBSET_ORDER = {s: i for i, s in enumerate(LCMDC_SUBSETS)}


class LCMDCExaminer(BaseExaminerAgent):
    """
    Examiner agent for LCMDC dataset.

    Focus: Long-range Diagnostic Closure
    - Three subsets: triage, diagnosis, consultation
    - Cross-session patient journey chaining

    Multi-session grouping:
      Primary: chain triage → diagnosis → consultation by patient_id.
      Fallback: group by department for unmatched records.
    """

    def __init__(self, **kwargs):
        # LCMDC already produces multi-session records via _chain_subsets,
        # so we set a wider session range for additional grouping of leftovers.
        kwargs.setdefault("sessions_range", (2, 4))
        super().__init__(**kwargs)
        # Track which records were already chained (skip in grouping)
        self._chained_ids = set()

    # ── abstract method implementations ──────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """
        Load LCMDC data.

        Expects either:
          - A directory with subdirs/files named triage*, diagnosis*, consultation*
          - A single JSON/JSONL file with a "subset" field per record
          - A directory of JSON/JSONL files (auto-detect subset from filename)
        """
        all_records = []

        if os.path.isdir(input_path):
            # Try to find subset-specific files/directories
            for item in sorted(os.listdir(input_path)):
                item_path = os.path.join(input_path, item)
                subset = self._detect_subset(item)

                if os.path.isdir(item_path):
                    for fname in sorted(os.listdir(item_path)):
                        if fname.endswith('.json') or fname.endswith('.jsonl'):
                            fpath = os.path.join(item_path, fname)
                            recs = self._load_file(fpath)
                            for r in recs:
                                r.setdefault("_subset", subset or self._detect_subset(fname))
                            all_records.extend(recs)
                elif item.endswith('.json') or item.endswith('.jsonl'):
                    recs = self._load_file(item_path)
                    for r in recs:
                        r.setdefault("_subset", subset)
                    all_records.extend(recs)
        else:
            all_records = self._load_file(input_path)

        print(f"[LCMDC] Loaded {len(all_records)} raw records")

        # Chain subsets by patient_id
        chained = self._chain_subsets(all_records)
        print(f"[LCMDC] Chained into {len(chained)} patient journeys "
              f"({len(self._chained_ids)} records used in chains)")
        return chained

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
                for key in ["train", "test", "validation", "data"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                if not records:
                    records = [data]
        return records

    def normalize_dialogue(self, raw_record: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Normalize an LCMDC record (possibly a chained multi-subset record).

        Chained records have:
          {"_chained": True, "patient_id": "...", "sessions": [...], "metadata": {...}}

        Single records have:
          {"dialogue": [...], "patient_id": "...", "_subset": "..."}
        """
        # Chained multi-subset record
        if raw_record.get("_chained"):
            patient_id = raw_record.get("patient_id", "")
            if not patient_id:
                content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
                patient_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

            messages = []
            metadata = raw_record.get("metadata", {})
            sessions = raw_record.get("sessions", [])

            for idx, session in enumerate(sessions):
                if idx > 0:
                    # Insert consultation separator
                    messages.append({
                        "role": "assistant",
                        "content": "---诊疗分割线---",
                    })
                # Parse session dialogue
                sess_messages = self._parse_session_dialogue(session)
                messages.extend(sess_messages)

                # Extract session-specific metadata
                subset = session.get("_subset", "")
                if subset:
                    metadata[f"session_{idx}_subset"] = subset
                for key in ["triage_result", "diagnosis", "prescription",
                             "department", "chief_complaint"]:
                    val = session.get(key, "")
                    if val:
                        metadata[f"session_{idx}_{key}"] = val

            return patient_id, messages, metadata

        # Single (unchained) record
        return self._normalize_single(raw_record)

    def _normalize_single(self, raw_record: Dict) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """Normalize a single LCMDC record (not chained)."""
        patient_id = raw_record.get("patient_id", raw_record.get("id", ""))
        if not patient_id:
            content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
            patient_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

        messages = self._parse_session_dialogue(raw_record)
        metadata = {}

        subset = raw_record.get("_subset", "")
        if subset:
            metadata["subset"] = subset

        for key in ["department", "diagnosis", "triage_result", "prescription",
                     "chief_complaint", "disease"]:
            val = raw_record.get(key, "")
            if val:
                metadata[key] = val

        return patient_id, messages, metadata

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Group by department or diagnosis.
        Chained records (already multi-session) get a unique key to avoid
        further merging with other patients.
        """
        meta = normalised_record.get("metadata", {})

        # If this was already chained, give it a unique key
        # (it's already multi-session, don't merge further)
        if meta.get("chained"):
            return f"_chained_{normalised_record.get('dialogue_id', random.random())}"

        # For unchained records, group by department
        dept = meta.get("department", "")
        if dept:
            return dept
        diag = meta.get("diagnosis", "")
        if diag:
            return str(diag)
        return "综合"

    # ── LCMDC-specific: cross-subset chaining ─────────────────────────

    def _chain_subsets(self, all_records: List[Dict]) -> List[Dict]:
        """
        Chain related sessions (triage → diagnosis → consultation) by patient_id
        into cross-session multi-stage dialogues.

        Records without a patient_id or without cross-subset matches are
        returned as standalone records for later grouping.
        """
        # Index records by patient_id
        by_patient: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        standalone = []

        for rec in all_records:
            pid = rec.get("patient_id", rec.get("id", ""))
            subset = rec.get("_subset", rec.get("subset", ""))

            if pid and subset:
                by_patient[pid][subset].append(rec)
            elif pid:
                by_patient[pid]["unknown"].append(rec)
            else:
                standalone.append(rec)

        # Build chained records
        chained = []
        for pid, subsets_dict in by_patient.items():
            # Sort subsets by defined order
            ordered_subsets = sorted(
                subsets_dict.keys(),
                key=lambda s: SUBSET_ORDER.get(s, 99)
            )

            sessions = []
            for subset_key in ordered_subsets:
                sessions.extend(subsets_dict[subset_key])

            if len(sessions) >= 2:
                # Multi-session chain
                meta = {
                    "chained": True,
                    "subset_count": len(ordered_subsets),
                    "subsets": ordered_subsets,
                }
                chained.append({
                    "_chained": True,
                    "patient_id": pid,
                    "sessions": sessions,
                    "metadata": meta,
                })
                for sess in sessions:
                    sid = sess.get("id", sess.get("patient_id", ""))
                    if sid:
                        self._chained_ids.add(sid)
            else:
                # Single session — treat as standalone
                standalone.extend(sessions)

        # Return chained + standalone (standalone will be grouped later)
        result = chained + standalone
        return result

    # ── helpers ───────────────────────────────────────────────────────

    def _parse_session_dialogue(self, session: Dict) -> List[Dict[str, str]]:
        """Parse dialogue from a single LCMDC session record."""
        messages = []
        dialogue = session.get("dialogue", session.get("utterances", []))

        if isinstance(dialogue, str):
            # Plain text
            return self._parse_dialogue_text(dialogue)

        if isinstance(dialogue, list):
            for turn in dialogue:
                if isinstance(turn, str):
                    role, content = self._parse_utterance(turn)
                    if content:
                        messages.append({"role": role, "content": content})
                elif isinstance(turn, dict):
                    role_raw = turn.get("role", turn.get("speaker", "patient"))
                    content = turn.get("content", turn.get("text", turn.get("utterance", "")))
                    role = self._map_role(role_raw)
                    if content.strip():
                        messages.append({"role": role, "content": content.strip()})

        return messages

    @staticmethod
    def _map_role(role_raw: str) -> str:
        role_lower = str(role_raw).lower().strip()
        if role_lower in ("doctor", "d", "医生", "assistant"):
            return "doctor"
        return "patient"

    @staticmethod
    def _parse_utterance(text: str) -> Tuple[str, str]:
        prefixes_doctor = ["医生：", "医生:", "D:", "D：", "doctor:", "Doctor:"]
        prefixes_patient = ["患者：", "患者:", "P:", "P：", "patient:", "Patient:", "病人:", "病人："]

        for prefix in prefixes_doctor:
            if text.startswith(prefix):
                return "doctor", text[len(prefix):].strip()
        for prefix in prefixes_patient:
            if text.startswith(prefix):
                return "patient", text[len(prefix):].strip()

        return "patient", text.strip()

    @staticmethod
    def _parse_dialogue_text(text: str) -> List[Dict[str, str]]:
        """Parse plain text dialogue."""
        import re
        messages = []
        parts = re.split(
            r'(?:^|\n)\s*((?:患者|医生|P|D|patient|doctor)\s*[:：])',
            text, flags=re.IGNORECASE
        )
        current_role = "patient"
        current_text = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            import re as _re
            role_match = _re.match(
                r'^(患者|医生|P|D|patient|doctor)\s*[:：]$', part, _re.IGNORECASE
            )
            if role_match:
                if current_text.strip():
                    messages.append({"role": current_role, "content": current_text.strip()})
                marker = role_match.group(1).upper()
                current_role = "doctor" if marker in ("D", "医生", "DOCTOR") else "patient"
                current_text = ""
            else:
                current_text += " " + part

        if current_text.strip():
            messages.append({"role": current_role, "content": current_text.strip()})
        return messages

    @staticmethod
    def _detect_subset(name: str) -> Optional[str]:
        """Detect LCMDC subset from filename or directory name."""
        name_lower = name.lower()
        for subset in LCMDC_SUBSETS:
            if subset in name_lower:
                return subset
        # Chinese names
        if "分诊" in name:
            return "triage"
        if "诊断" in name:
            return "diagnosis"
        if "问诊" in name or "咨询" in name:
            return "consultation"
        return None


def main():
    parser = argparse.ArgumentParser(description="LCMDC Examiner Agent")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to LCMDC data directory or file")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_multi_session", action="store_true",
                        help="Disable multi-session synthesis for unchained records")
    parser.add_argument("--sessions_min", type=int, default=2)
    parser.add_argument("--sessions_max", type=int, default=4)
    args = parser.parse_args()

    agent = LCMDCExaminer(
        model=args.model,
        language="zh",
        max_workers=args.workers,
        enable_multi_session=not args.no_multi_session,
        sessions_range=(args.sessions_min, args.sessions_max),
    )
    agent.run(args.input_path, args.output_path, args.max_dialogues)


if __name__ == "__main__":
    main()
