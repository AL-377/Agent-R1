"""
Script to populate demo data and take clean screenshots.
Runs the sessions programmatically, then launches the clean UI for screenshot capture.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_prototype.model_adapter import ModelAdapter
from memory_prototype.session_manager import SessionManager
from memory_prototype.demo_data import ALL_SESSIONS

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "memory_data")
PATIENT_ID = "patient_demo_001"
OUT_DIR = os.path.join(os.path.dirname(__file__), "session_dumps")
os.makedirs(OUT_DIR, exist_ok=True)

adapter = ModelAdapter(
    memory_model="gpt-4o-mini-2024-07-18",
    chat_model="gpt-4o-mini-2024-07-18",
)


def run_session(session_info, sm):
    """Run a session and return the final state."""
    sm.start_session(session_info["session_id"])
    chat_history = []
    last_info = None

    for msg in session_info["messages"]:
        user_msg = msg["content"]
        print(f"  [Patient] {user_msg[:60]}...")
        reply, info = sm.process_turn(user_msg)
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": reply})
        print(f"  [Doctor]  {reply[:60]}...")
        print(f"  [Actions] {info.get('executed', [])}")
        print()
        last_info = info

    return chat_history, last_info


def main():
    sm = SessionManager(patient_id=PATIENT_ID, adapter=adapter, persist_dir=PERSIST_DIR)

    for i, sess in enumerate(ALL_SESSIONS):
        print(f"=== Session {i+1}: {sess['label']} ===")
        chat_history, last_info = run_session(sess, sm)

        dump = {
            "session": sess["label"],
            "chat_history": chat_history,
            "memory_state": sm.memory.get_state_dict(),
            "last_think": last_info.get("think", "") if last_info else "",
            "last_actions": last_info.get("executed", []) if last_info else [],
        }
        out_path = os.path.join(OUT_DIR, f"session_{i+1}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        print(f"Saved to {out_path}\n")

    print("All sessions complete. You can now launch the clean UI with:")
    print("  python -m memory_prototype.app")
    print("and take screenshots of the user-facing interface.")


if __name__ == "__main__":
    main()
