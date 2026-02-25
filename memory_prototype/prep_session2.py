#!/usr/bin/env python3
"""Clear Working layer for session 2, keeping Identity from session 1."""

import json
from pathlib import Path

MEMORY_PATH = Path(__file__).parent / "memory_data" / "patient_demo_001_memory.json"

def main():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["working"] = []
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Cleared Working layer. Identity preserved.")

if __name__ == "__main__":
    main()
