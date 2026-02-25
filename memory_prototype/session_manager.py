"""
SessionManager: Manages dialogue sessions and coordinates Memory Model + Chat Model.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple

from .memory_base import MemoryBase, VALID_LAYERS
from .model_adapter import ModelAdapter


class SessionManager:
    """Manages a single patient's dialogue sessions and orchestrates the memory pipeline."""

    def __init__(self, patient_id: str, adapter: ModelAdapter,
                 persist_dir: str = "./memory_data"):
        self.patient_id = patient_id
        self.adapter = adapter
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.memory = MemoryBase()
        self._load_memory()

        self.current_session_id: Optional[str] = None
        self.dialogue_history: List[Dict[str, str]] = []
        self.operation_logs: List[Dict[str, Any]] = []

    def _memory_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.patient_id}_memory.json")

    def _load_memory(self):
        p = self._memory_path()
        if os.path.exists(p):
            self.memory.load(p)

    def _save_memory(self):
        self.memory.save(self._memory_path())

    def start_session(self, session_id: str):
        if self.current_session_id and session_id != self.current_session_id:
            self.memory.clear_working()
        self.current_session_id = session_id
        self.dialogue_history = []
        self.operation_logs = []

    def process_turn(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process one dialogue turn:
        1. Call Memory Model to update memory
        2. Call Chat Model (with updated memory) to generate reply

        Returns (assistant_reply, turn_info).
        """
        self.dialogue_history.append({"role": "user", "content": user_message})

        dialogue_ctx = self._build_dialogue_context()
        mem_state_text = self.memory.get_state_text()

        mem_result = self.adapter.call_memory_model(dialogue_ctx, mem_state_text)
        executed = self._execute_actions(mem_result.get("actions", []))

        updated_mem_text = self.memory.get_state_text()
        reply = self.adapter.call_chat_model(self.dialogue_history, updated_mem_text)

        self.dialogue_history.append({"role": "assistant", "content": reply})
        self._save_memory()

        turn_info = {
            "think": mem_result.get("think", ""),
            "actions": mem_result.get("actions", []),
            "executed": executed,
            "memory_state": self.memory.get_state_dict(),
        }
        self.operation_logs.append(turn_info)
        return reply, turn_info

    def _build_dialogue_context(self) -> str:
        lines = []
        for m in self.dialogue_history:
            role = "患者" if m["role"] == "user" else "医生"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def _execute_actions(self, actions: List[Dict]) -> List[str]:
        logs = []
        for act in actions:
            name = act.get("name", "")
            args = act.get("arguments", {})
            try:
                if name == "memory_insert":
                    layer = args.get("layer", "working")
                    content = args.get("content", "")
                    if isinstance(content, dict):
                        content = json.dumps(content, ensure_ascii=False)
                    mid = self.memory.insert(layer, content)
                    logs.append(f"INSERT [{layer}] id={mid}: {content}")
                elif name == "memory_update":
                    layer = args.get("layer", "working")
                    mid = args.get("memory_id", "")
                    content = args.get("content", "")
                    if isinstance(content, dict):
                        content = json.dumps(content, ensure_ascii=False)
                    ok = self.memory.update(layer, mid, content)
                    logs.append(f"UPDATE [{layer}] id={mid}: {'OK' if ok else 'NOT FOUND'}")
                elif name == "memory_delete":
                    layer = args.get("layer", "working")
                    mid = args.get("memory_id", "")
                    ok = self.memory.delete(layer, mid)
                    logs.append(f"DELETE [{layer}] id={mid}: {'OK' if ok else 'NOT FOUND'}")
                elif name == "memory_wait":
                    logs.append("WAIT — no operation")
                else:
                    logs.append(f"UNKNOWN action: {name}")
            except Exception as e:
                logs.append(f"ERROR executing {name}: {e}")
        return logs

    def get_memory_display(self) -> Dict[str, str]:
        """Return per-layer formatted strings for the UI."""
        result = {}
        for layer in VALID_LAYERS:
            items = self.memory.memories[layer]
            if not items:
                result[layer] = "(empty)"
            else:
                lines = [f"[{it.id}] {it.content}" for it in items]
                result[layer] = "\n".join(lines)
        return result
