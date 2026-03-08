python examples/eval/eval_dialogue_quality.py \
    --cache_dirs datasets/cmtmedqa/.examiner_cache \
                 datasets/huatuo26m/.examiner_cache \
                 datasets/meddialog_cn/.examiner_cache \
    --output_dir eval_results/dialogue_quality_qw3_14b \
    --chat_model qwen3-14b \
    --judge_model gpt-oss-120b \
    --max_samples_per_source 50 \
    --workers 4