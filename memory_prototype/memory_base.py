"""
MemoryBase: Four-layer hierarchical memory storage for medical dialogue.
Lightweight version without FAISS dependency for the prototype demo.
"""

import json
import uuid
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field


VALID_LAYERS = ("working", "identity", "history", "experience")


@dataclass
class MemoryItem:
    id: str
    layer: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBase:
    """Four-layer memory store for a single patient."""

    def __init__(self):
        self.memories: Dict[str, List[MemoryItem]] = {l: [] for l in VALID_LAYERS}

    def insert(self, layer: str, content, metadata: Optional[Dict] = None) -> str:
        if layer not in VALID_LAYERS:
            raise ValueError(f"Invalid layer: {layer}")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        mid = str(uuid.uuid4())[:8]
        item = MemoryItem(id=mid, layer=layer, content=content,
                          timestamp=time.time(), metadata=metadata or {})
        self.memories[layer].append(item)
        return mid

    def update(self, layer: str, memory_id: str, content) -> bool:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        for item in self.memories.get(layer, []):
            if item.id == memory_id:
                item.content = content
                return True
        return False

    def delete(self, layer: str, memory_id: str) -> bool:
        items = self.memories.get(layer, [])
        for i, item in enumerate(items):
            if item.id == memory_id:
                items.pop(i)
                return True
        return False

    def clear_working(self):
        self.memories["working"] = []

    def get_state_dict(self) -> Dict[str, List[Dict]]:
        return {
            layer: [asdict(it) for it in items]
            for layer, items in self.memories.items()
        }

    def get_state_text(self) -> str:
        parts = []
        for layer in VALID_LAYERS:
            items = self.memories[layer]
            parts.append(f"[{layer.upper()}] ({len(items)} items)")
            for it in items:
                parts.append(f"  - [{it.id}] {it.content}")
        return "\n".join(parts)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_state_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for layer in VALID_LAYERS:
            self.memories[layer] = [
                MemoryItem(**d) for d in data.get(layer, [])
            ]

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.memories.values())
