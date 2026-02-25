"""
ModelAdapter: Unified interface for calling LLM APIs.
Wraps the existing Agent-R1 LLM utilities.
"""

import sys
import os
import json
import re
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_r1.utils.llm import query_llm

MEMORY_PROMPT_TEMPLATE = (
    "你是一个智能医疗记忆管理助手，专注于医疗对话系统。你的职责是分析医疗对话内容，"
    "管理患者在不同记忆层中的信息，确保医疗数据的准确和有序存储。\n"
    "---\n"
    "当前对话上下文：\n{dialogue_context}\n\n"
    "当前记忆状态：\n{memory_state}\n\n"
    "基于对话上下文和当前记忆状态，决定需要执行哪些记忆操作。\n\n"
    "记忆层级：\n"
    "- working: 当前会话的临时关键信息（每次新会话清空）\n"
    "- identity: 患者身份与基础信息（如年龄、过敏史、基础疾病等，长期保留）\n"
    "- history: 历史诊疗记录（如检查结果、用药方案、手术记录，长期保留）\n"
    "- experience: 临床经验与医学知识（长期保留）\n\n"
    "可用的记忆操作工具：\n"
    '1. memory_insert(layer, content) — 向指定层插入新记忆\n'
    '2. memory_update(layer, memory_id, content) — 更新已有记忆\n'
    '3. memory_delete(layer, memory_id) — 删除已有记忆\n'
    '4. memory_wait() — 当前不需要任何记忆操作\n\n'
    "输出格式：\n"
    "<think>\n[你的分析推理过程]\n</think>\n\n"
    "推理后输出JSON格式的记忆操作：\n"
    '单个操作: {{"name": "memory_insert", "arguments": {{"layer": "working", "content": "..."}}}}\n'
    '多个操作: [{{"name": "memory_insert", "arguments": {{"layer": "identity", "content": "..."}}}}, ...]\n'
    '无操作: {{"name": "memory_wait", "arguments": {{}}}}\n\n'
    "重要规则：\n"
    "- content 字段必须是纯文本字符串，不要使用JSON对象\n"
    "- memory_update 和 memory_delete 必须提供当前记忆状态中存在的有效 memory_id\n"
    "- JSON 必须合法（单个对象或对象数组）\n"
    '- layer 只能是 "working", "identity", "history", "experience" 之一\n'
)

CHAT_PROMPT_TEMPLATE = (
    "你是一位专业的医疗对话助手，正在为患者提供诊疗咨询服务。"
    "请根据以下患者的记忆信息和当前对话历史，给出专业、准确、有温度的回复。\n\n"
    "患者记忆信息：\n{memory_state}\n\n"
    "请基于以上信息回复患者的最新消息。回复应简洁专业，体现对患者历史情况的了解。"
)


class ModelAdapter:
    """Unified adapter for Memory Model and Chat Model."""

    def __init__(self, memory_model: str = "gpt-4o-mini-2024-07-18",
                 chat_model: str = "gpt-4o-mini-2024-07-18"):
        self.memory_model = memory_model
        self.chat_model = chat_model

    def call_memory_model(self, dialogue_context: str, memory_state: str) -> Dict[str, Any]:
        prompt = MEMORY_PROMPT_TEMPLATE.format(
            dialogue_context=dialogue_context,
            memory_state=memory_state
        )
        result = query_llm(
            model_name=self.memory_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3,
        )
        raw = result.get("response", "")
        think, actions = self._parse_memory_output(raw)
        return {"think": think, "actions": actions, "raw": raw}

    def call_chat_model(self, messages: List[Dict[str, str]],
                        memory_state: str) -> str:
        system = CHAT_PROMPT_TEMPLATE.format(memory_state=memory_state)
        result = query_llm(
            model_name=self.chat_model,
            messages=messages,
            system=system,
            max_tokens=2048,
            temperature=0.7,
        )
        return result.get("response", "")

    @staticmethod
    def _parse_memory_output(raw: str):
        think = ""
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()

        after_think = raw
        if think_match:
            after_think = raw[think_match.end():]

        json_match = re.search(r'(\[.*\]|\{.*\})', after_think, re.DOTALL)
        actions = []
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    actions = [parsed]
                elif isinstance(parsed, list):
                    actions = parsed
            except json.JSONDecodeError:
                pass
        return think, actions
