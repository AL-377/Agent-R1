"""
CMtMedQA Examiner Agent
========================

Dataset: CMtMedQA
Focus:   主动问诊与状态追踪 — 多轮主动问诊对话

Multi-session synthesis strategy:
  按疾病主题（department/category）分组 → 将同主题的多条Q&A或短对话
  串联为同一患者因同一类疾病多次就诊的记录。
  典型场景：慢性病随访管理、多疾病共病管理。

  核心特殊处理：_split_answer_into_turns() 将医生的长回答拆分为模拟多轮对话。

Usage:
    python cmtmedqa_examiner.py \\
        --input_path /path/to/cmtmedqa/data.json \\
        --output_path /path/to/output/cmtmedqa_examiner.json
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


class CMtMedQAExaminer(BaseExaminerAgent):
    """
    Examiner agent for CMtMedQA dataset.

    Focus: Active Inquiry & State Tracking
    - Simulates multi-turn active inquiry dialogues
    - Doctor's comprehensive answers are split into turns

    Multi-session grouping: by department / category.
    Same-category Q&A pairs are merged into one patient's multiple visits.
    """

    # ── abstract method implementations ──────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """Load CMtMedQA data (JSON / JSONL / directory)."""
        records = []
        if os.path.isdir(input_path):
            for fname in sorted(os.listdir(input_path)):
                if fname.endswith('.json') or fname.endswith('.jsonl'):
                    fpath = os.path.join(input_path, fname)
                    records.extend(self._load_file(fpath))
        else:
            records = self._load_file(input_path)
        print(f"[CMtMedQA] Loaded {len(records)} raw records")
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
                for key in ["train", "test", "validation", "data"]:
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                if not records:
                    records = [data]
        return records

    def normalize_dialogue(self, raw_record: Any) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Normalize a CMtMedQA record.

        Expected formats:
          Format A: {"question": "...", "answer": "..."}
          Format B: {"question": "...", "answer": "...", "department": "...", "title": "..."}
          Format C: {"dialogue": [{"role": "...", "content": "..."}]}
        """
        dialogue_id = raw_record.get("id", raw_record.get("dialogue_id", None))
        if dialogue_id is None:
            content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
            dialogue_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

        messages = []
        metadata = {}

        # Extract metadata
        for key in ["department", "title", "category", "科室"]:
            val = raw_record.get(key, "")
            if val:
                metadata[key] = val

        # Format C: pre-structured dialogue
        if "dialogue" in raw_record and isinstance(raw_record["dialogue"], list):
            for turn in raw_record["dialogue"]:
                role_raw = turn.get("role", turn.get("speaker", "patient")).lower()
                content = turn.get("content", turn.get("text", ""))
                role = "doctor" if role_raw in ("doctor", "医生", "assistant", "gpt") else "patient"
                if content.strip():
                    messages.append({"role": role, "content": content.strip()})
            return dialogue_id, messages, metadata

        # Format A/B: single Q&A → split answer into multi-turn
        question = raw_record.get("question", raw_record.get("query", ""))
        answer = raw_record.get("answer", raw_record.get("response", ""))

        if question.strip():
            messages.append({"role": "patient", "content": question.strip()})

        if answer.strip():
            # Split long doctor answer into simulated multi-turn dialogue
            turns = self._split_answer_into_turns(answer.strip())
            for i, turn_text in enumerate(turns):
                if i % 2 == 0:
                    messages.append({"role": "doctor", "content": turn_text})
                else:
                    # Insert simulated patient follow-up
                    messages.append({"role": "patient", "content": turn_text})

        return dialogue_id, messages, metadata

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Group by department or category.
        Same-department/category records become candidates for the same patient.
        """
        meta = normalised_record.get("metadata", {})
        # Priority: department > category > title keyword
        dept = meta.get("department", meta.get("科室", ""))
        if dept:
            return dept
        cat = meta.get("category", meta.get("title", ""))
        if cat:
            return cat
        # Fallback: extract keyword from first message
        msgs = normalised_record.get("messages", [])
        if msgs:
            first = msgs[0].get("content", "")[:100]
            for kw in ["糖尿病", "高血压", "感冒", "咳嗽", "头痛", "腹痛",
                        "皮肤", "骨折", "心脏", "肝", "肾", "肺",
                        "妇科", "儿科", "眼科", "耳鼻喉"]:
                if kw in first:
                    return kw
        return "综合"

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _split_answer_into_turns(answer: str) -> List[str]:
        """
        Split a long doctor answer into simulated multi-turn dialogue.

        Strategy:
          1. Split by numbered points (1. 2. 3. or 一、二、三、)
          2. Split by paragraph breaks
          3. If still too long, split by sentences

        Returns: list of text segments (odd indices = doctor, even = patient follow-up)
        """
        # Try numbered points
        segments = re.split(r'(?:(?:^|\n)\s*(?:\d+[.、)）]|[一二三四五六七八九十]+[、.]))', answer)
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) >= 3:
            return segments

        # Try paragraph breaks
        segments = [s.strip() for s in answer.split('\n\n') if s.strip()]
        if len(segments) >= 2:
            return segments

        # Try single newline
        segments = [s.strip() for s in answer.split('\n') if s.strip()]
        if len(segments) >= 2:
            return segments

        # Fallback: return as single turn
        return [answer]


def main():
    parser = argparse.ArgumentParser(description="CMtMedQA Examiner Agent")
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

    agent = CMtMedQAExaminer(
        model=args.model,
        language="zh",
        max_workers=args.workers,
        enable_multi_session=not args.no_multi_session,
        sessions_range=(args.sessions_min, args.sessions_max),
    )
    agent.run(args.input_path, args.output_path, args.max_dialogues)


if __name__ == "__main__":
    main()
