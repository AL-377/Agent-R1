	
export no_proxy=""
export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"

python examples/eval/eval_dialogue_quality.py \
    --cache_dirs datasets/cmtmedqa/.examiner_cache \
                 datasets/huatuo26m/.examiner_cache \
                 datasets/meddialog_cn/.examiner_cache \
    --output_dir eval_results/dialogue_quality_gemini \
    --chat_model gemini-2.5-pro-preview-05-06 \
    --judge_model gpt-oss-120b \
    --max_samples_per_source 50 \
    --workers 4