#!/bin/bash
# Re-run ONLY the agent_memory mode for one or more models.
#
# This script:
#   1. For eval_memory_agent (Dialogue Quality):
#      - Keeps existing oracle_memory / no_memory scores in traces
#      - Deletes old agent_traces/ so the Memory Agent pipeline re-runs
#      - Uses --force_rerun_modes agent_memory so eval scores are regenerated
#   2. For eval_medmem_agent (MED-MEM Accuracy):
#      - Uses --force_rerun to clear done_patients.txt and re-run all patients
#
# Usage:
#   bash scripts/rerun_agent_only.sh                       # all 3 models
#   bash scripts/rerun_agent_only.sh gpt-4o-2024-05-13     # single model
#   bash scripts/rerun_agent_only.sh DeepSeek-R1 qwen3-14b # two models

set -e

CACHE_DIRS="datasets/cmtmedqa/.examiner_cache \
            datasets/huatuo26m/.examiner_cache \
            datasets/meddialog_cn/.examiner_cache"

JUDGE="gpt-oss-120b"

if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="gpt-4o-2024-05-13 DeepSeek-R1 qwen3-14b"
fi

for MODEL in ${MODELS}; do
    QUALITY_DIR="eval_results/memory_agent_quality_${MODEL}"
    MEDMEM_DIR="eval_results/medmem_agent_${MODEL}"

    # ── Step 1: Dialogue Quality — re-run agent_memory only ─────────────
    echo ""
    echo "=========================================="
    echo " [${MODEL}] Re-run agent_memory — Dialogue Quality"
    echo "=========================================="

    if [ -d "${QUALITY_DIR}/agent_traces" ]; then
        echo "  Clearing old agent_traces..."
        rm -rf "${QUALITY_DIR}/agent_traces"
    fi

    python examples/eval/eval_memory_agent.py \
        --cache_dirs ${CACHE_DIRS} \
        --output_dir ${QUALITY_DIR} \
        --memory_model ${MODEL} \
        --chat_model ${MODEL} \
        --judge_model ${JUDGE} \
        --max_patients_per_source 20 \
        --max_eval_samples_per_patient 5 \
        --workers 4 \
        --retrieval_top_k 15 \
        --enable_reflection \
        --modes agent_memory oracle_memory no_memory \
        --force_rerun_modes agent_memory

    # ── Step 2: MED-MEM Accuracy — full re-run ─────────────────────────
    echo ""
    echo "=========================================="
    echo " [${MODEL}] Re-run agent — MED-MEM Accuracy"
    echo "=========================================="

    python examples/eval/eval_medmem_agent.py \
        --cache_dirs ${CACHE_DIRS} \
        --output_dir ${MEDMEM_DIR} \
        --memory_model ${MODEL} \
        --judge_model ${JUDGE} \
        --workers 4 \
        --max_patients_per_source 20 \
        --max_eval_samples_per_patient 5 \
        --memory_temperature 0.3 \
        --force_rerun

done

echo ""
echo "All agent-only re-runs complete."
