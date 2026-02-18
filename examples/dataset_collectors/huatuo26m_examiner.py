"""
Huatuo-26M Examiner Agent
===========================

Dataset: Huatuo-26M
Focus:   医学知识记忆 — 大规模医学问答对

Multi-session synthesis strategy:
  按 source/category/type 分组 → 将同类别的多条Q&A对串联为同一患者的多次问诊。
  _group_qa_pairs() 将 group_size 条连续Q&A合并为一次"问诊"，
  然后由 multi_session_synthesizer 将多次"问诊"合并为同一患者的多次就诊。
  典型场景：慢性病随访管理、罕见病鉴别诊断。

  核心特殊处理：原始数据是纯单轮Q&A，需要先合并为"伪多轮"再做多会话合成。

Usage:
    python huatuo26m_examiner.py \\
        --input_path /path/to/huatuo26m/data.json \\
        --output_path /path/to/output/huatuo26m_examiner.json
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


class Huatuo26MExaminer(BaseExaminerAgent):
    """
    Examiner agent for Huatuo-26M dataset.

    Focus: Medical Knowledge Memory
    - Large-scale medical Q&A pairs
    - Multiple sources (encyclopedias, forums, textbooks)

    Multi-session grouping: by category / source / type.
    Same-category Q&A pairs are first grouped into pseudo-sessions
    (group_size consecutive pairs = 1 session), then merged across sessions.
    """

    def __init__(self, group_size: int = 5, **kwargs):
        """
        Args:
            group_size: Number of Q&A pairs to merge into one pseudo-session.
        """
        super().__init__(**kwargs)
        self.group_size = group_size

    # ── abstract method implementations ──────────────────────────────

    def load_raw_data(self, input_path: str) -> List[Any]:
        """Load Huatuo-26M data."""
        records = []
        if os.path.isdir(input_path):
            for fname in sorted(os.listdir(input_path)):
                if fname.endswith('.json') or fname.endswith('.jsonl'):
                    fpath = os.path.join(input_path, fname)
                    records.extend(self._load_file(fpath))
        else:
            records = self._load_file(input_path)
        print(f"[Huatuo-26M] Loaded {len(records)} raw Q&A pairs")

        # Group Q&A pairs into pseudo-sessions BEFORE normalization
        grouped = self._group_qa_pairs(records)
        print(f"[Huatuo-26M] Grouped into {len(grouped)} pseudo-sessions "
              f"(group_size={self.group_size})")
        return grouped

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
        Normalize a grouped pseudo-session (list of Q&A pairs merged together).

        Input: a dict with {"qa_pairs": [...], "category": "...", "group_id": "..."}
        Each qa_pair: {"question": "...", "answer": "..."}
        """
        group_id = raw_record.get("group_id", "")
        category = raw_record.get("category", "")
        qa_pairs = raw_record.get("qa_pairs", [])

        if not group_id:
            content_str = json.dumps(raw_record, ensure_ascii=False, sort_keys=True)[:500]
            group_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

        messages = []
        metadata = {"category": category, "qa_count": len(qa_pairs)}

        for qa in qa_pairs:
            question = qa.get("question", qa.get("query", qa.get("input", "")))
            answer = qa.get("answer", qa.get("response", qa.get("output", "")))

            if question.strip():
                messages.append({"role": "patient", "content": question.strip()})
            if answer.strip():
                messages.append({"role": "doctor", "content": answer.strip()})

        return group_id, messages, metadata

    def grouping_key_fn(self, normalised_record: Dict[str, Any]) -> str:
        """
        Group by category.
        Same-category pseudo-sessions become candidates for the same patient.
        """
        meta = normalised_record.get("metadata", {})
        return meta.get("category", "综合医学")

    # ── helpers ───────────────────────────────────────────────────────

    def _group_qa_pairs(self, records: List[Dict]) -> List[Dict]:
        """
        Group raw Q&A pairs by category, then chunk into pseudo-sessions.

        Each record is expected to have:
          - "question" / "answer" (required)
          - "source" / "category" / "type" (optional, for grouping)

        Returns: list of {"qa_pairs": [...], "category": str, "group_id": str}
        """
        # Bucket by category
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for rec in records:
            key = self._extract_category(rec)
            buckets[key].append(rec)

        # Chunk each bucket
        grouped = []
        for cat, items in buckets.items():
            random.shuffle(items)
            for i in range(0, len(items), self.group_size):
                chunk = items[i : i + self.group_size]
                if not chunk:
                    continue
                gid = hashlib.md5(
                    f"{cat}_{i}_{random.random()}".encode()
                ).hexdigest()[:12]
                grouped.append({
                    "qa_pairs": chunk,
                    "category": cat,
                    "group_id": gid,
                })

        random.shuffle(grouped)
        return grouped

    @staticmethod
    def _extract_category(record: Dict) -> str:
        """Extract grouping category from a single Q&A record."""
        for key in ["category", "source", "type", "科室", "department"]:
            val = record.get(key, "")
            if val:
                return str(val)
        # Fallback: keyword extraction from question
        question = record.get("question", record.get("query", ""))[:100]
        for kw in ["糖尿病", "高血压", "心脏", "肝", "肾", "肺", "骨",
                    "皮肤", "眼", "耳", "口腔", "妇科", "儿科", "肿瘤",
                    "神经", "精神", "感染", "免疫"]:
            if kw in question:
                return kw
        return "综合医学"


def main():
    parser = argparse.ArgumentParser(description="Huatuo-26M Examiner Agent")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of Q&A pairs per pseudo-session")
    parser.add_argument("--no_multi_session", action="store_true",
                        help="Disable multi-session synthesis")
    parser.add_argument("--sessions_min", type=int, default=2)
    parser.add_argument("--sessions_max", type=int, default=4)
    args = parser.parse_args()

    agent = Huatuo26MExaminer(
        group_size=args.group_size,
        model=args.model,
        language="zh",
        max_workers=args.workers,
        enable_multi_session=not args.no_multi_session,
        sessions_range=(args.sessions_min, args.sessions_max),
    )
    agent.run(args.input_path, args.output_path, args.max_dialogues)


if __name__ == "__main__":
    main()
