#!/bin/bash
# Enhanced Hierarchical Memory Agent Framework — Full Evaluation Suite
#
# Framework features (from related work):
#   - MemoryBank-style relevance scoring (Ebbinghaus decay + access strengthening)
#   - A-Mem-style conflict detection & auto-merge (character n-gram Jaccard)
#   - Generative Agents-style session-end reflection (working → permanent promotion)
#   - MemGPT-style selective retrieval (top-K by relevance, not full dump)
#   - Fine-Mem-style step verification (optional, --enable_step_verify)
#
# Evaluations:
#   1. Dialogue Quality: agent_memory vs oracle_memory vs no_memory
#   2. MED-MEM Accuracy: agent memory accuracy across sources and difficulty levels

set -e

CACHE_DIRS="datasets/cmtmedqa/.examiner_cache \
            datasets/huatuo26m/.examiner_cache \
            datasets/meddialog_cn/.examiner_cache"

JUDGE="gpt-oss-120b"

# ─── Evaluation 1: Dialogue Quality ──────────────────────────────────────────
echo "=========================================="
echo " Evaluation 1: Dialogue Quality (Enhanced Agent)"
echo "=========================================="

for MODEL in gpt-4o-2024-05-13; do
    echo ""
    echo ">>> Memory+Chat Model: ${MODEL}"
    python examples/eval/eval_memory_agent.py \
        --cache_dirs ${CACHE_DIRS} \
        --output_dir eval_results/memory_agent_quality_${MODEL} \
        --memory_model ${MODEL} \
        --chat_model ${MODEL} \
        --judge_model ${JUDGE} \
        --max_patients_per_source 20 \
        --max_eval_samples_per_patient 5 \
        --workers 4 \
        --retrieval_top_k 15 \
        --enable_reflection \
        --modes agent_memory oracle_memory no_memory
done

# ─── Evaluation 2: MED-MEM Memory Accuracy ──────────────────────────────────
echo ""
echo "=========================================="
echo " Evaluation 2: MED-MEM Agent Memory Accuracy (Enhanced)"
echo "=========================================="

for MODEL in gpt-4o-2024-05-13; do
    echo ""
    echo ">>> Memory Model: ${MODEL}"
    python examples/eval/eval_medmem_agent.py \
        --cache_dirs ${CACHE_DIRS} \
        --output_dir eval_results/medmem_agent_${MODEL} \
        --memory_model ${MODEL} \
        --judge_model ${JUDGE} \
        --workers 4 \
        --max_patients_per_source 20 \
        --max_eval_samples_per_patient 5 \
        --memory_temperature 0.3
done

echo ""
echo "All evaluations complete."
