"""
Memory-Enhanced Medical Dialogue Prototype — Gradio Demo
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from memory_prototype.memory_base import MemoryBase, VALID_LAYERS
from memory_prototype.model_adapter import ModelAdapter
from memory_prototype.session_manager import SessionManager
from memory_prototype.demo_data import ALL_SESSIONS

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "memory_data")
PATIENT_ID = "patient_demo_001"

adapter = ModelAdapter(
    memory_model="gpt-4o-mini-2024-07-18",
    chat_model="gpt-4o-mini-2024-07-18",
)
sm = SessionManager(patient_id=PATIENT_ID, adapter=adapter, persist_dir=PERSIST_DIR)
sm.start_session("session_init")


def reset_state():
    global sm
    sm = SessionManager(patient_id=PATIENT_ID, adapter=adapter, persist_dir=PERSIST_DIR)
    sm.start_session("session_init")
    return [], *_memory_outputs(), "", ""


def _memory_outputs():
    d = sm.get_memory_display()
    return d["working"], d["identity"], d["history"], d["experience"]


def chat_fn(user_msg, chat_history):
    if not user_msg.strip():
        return chat_history, *_memory_outputs(), "", ""

    reply, info = sm.process_turn(user_msg)

    chat_history = chat_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]

    think_text = info.get("think", "") or "(no think output)"
    action_lines = info.get("executed", [])
    action_text = "\n".join(action_lines) if action_lines else "(no action)"

    return chat_history, *_memory_outputs(), think_text, action_text


def auto_play(session_label):
    """Auto-play all messages from a predefined session."""
    target = None
    for s in ALL_SESSIONS:
        if s["label"] == session_label:
            target = s
            break
    if target is None:
        yield [], *_memory_outputs(), "", "未找到该会话"
        return

    sm.start_session(target["session_id"])
    chat_history = []

    for msg in target["messages"]:
        user_msg = msg["content"]
        reply, info = sm.process_turn(user_msg)
        chat_history = chat_history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        think_text = info.get("think", "") or "(no think output)"
        action_lines = info.get("executed", [])
        action_text = "\n".join(action_lines) if action_lines else "(no action)"
        yield chat_history, *_memory_outputs(), think_text, action_text


def build_demo(dev_mode=False):
    with gr.Blocks(title="记忆增强医疗对话原型系统") as demo:
        gr.Markdown(
            "# 记忆增强医疗对话原型系统\n"
            "基于分层记忆管理的医疗对话系统原型，支持 Working / Identity / History / Experience 四层记忆的自动管理。",
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="医疗对话", height=480)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入患者消息...",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                if dev_mode:
                    with gr.Row():
                        session_dd = gr.Dropdown(
                            choices=[s["label"] for s in ALL_SESSIONS],
                            value=ALL_SESSIONS[0]["label"],
                            label="预设会话场景",
                            scale=3,
                        )
                        play_btn = gr.Button("自动播放该会话", variant="secondary", scale=1)
                        reset_btn = gr.Button("重置系统", variant="stop", scale=1)

            with gr.Column(scale=2):
                gr.Markdown("### 四层记忆状态")
                mem_working = gr.Textbox(label="Working（当前会话）", lines=4,
                                          interactive=False, elem_classes="memory-box")
                mem_identity = gr.Textbox(label="Identity（患者身份）", lines=3,
                                           interactive=False, elem_classes="memory-box")
                mem_history = gr.Textbox(label="History（诊疗历史）", lines=4,
                                          interactive=False, elem_classes="memory-box")
                mem_experience = gr.Textbox(label="Experience（临床经验）", lines=3,
                                             interactive=False, elem_classes="memory-box")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Memory Model — Think")
                think_box = gr.Textbox(label="模型思考过程", lines=6, interactive=False)
            with gr.Column():
                gr.Markdown("### Memory Model — Actions")
                action_box = gr.Textbox(label="执行的记忆操作", lines=6, interactive=False)

        outputs = [chatbot, mem_working, mem_identity, mem_history, mem_experience,
                   think_box, action_box]

        send_btn.click(
            chat_fn,
            inputs=[msg_input, chatbot],
            outputs=outputs,
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            chat_fn,
            inputs=[msg_input, chatbot],
            outputs=outputs,
        ).then(lambda: "", outputs=[msg_input])

        if dev_mode:
            play_btn.click(auto_play, inputs=[session_dd], outputs=outputs)
            reset_btn.click(
                reset_state,
                outputs=[chatbot, mem_working, mem_identity, mem_history, mem_experience,
                         think_box, action_box],
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Enable developer controls")
    parser.add_argument("--session", type=str, default="session_init",
                        help="Initial session ID (use a new ID to trigger Working layer clear)")
    args = parser.parse_args()

    sm.start_session(args.session)

    demo = build_demo(dev_mode=args.dev)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
