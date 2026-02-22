# pure_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/cmtmedqa/cmtmedqa_pure_summary.parquet \
    --output_dir eval_results/cmtmedqa_ps_gpt4o \
    --model gpt-4o-2024-05-13 \
    --judge_model gpt-oss-120b \
    --max_tokens 4096 \
    --workers 8

# no_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/cmtmedqa/cmtmedqa_no_summary.parquet \
    --output_dir eval_results/cmtmedqa_ns_gpt4o \
    --model gpt-4o-2024-05-13 \
    --max_tokens 4096 \
    --workers 8

# pure_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/huatuo26m/huatuo26m_pure_summary.parquet \
    --output_dir eval_results/huatuo26m_ps_gpt4o \
    --model gpt-4o-2024-05-13 \
    --judge_model gpt-oss-120b \
    --max_tokens 4096 \
    --workers 8

# no_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/huatuo26m/huatuo26m_no_summary.parquet \
    --output_dir eval_results/huatuo26m_ns_gpt4o \
    --model gpt-4o-2024-05-13 \
    --max_tokens 4096 \
    --workers 8

# pure_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/meddialog_cn/meddialog_cn_pure_summary.parquet \
    --output_dir eval_results/meddialog_cn_ps_gpt4o \
    --model gpt-4o-2024-05-13 \
    --judge_model gpt-oss-120b \
    --max_tokens 4096 \
    --workers 8

# no_summary 评测
python examples/eval/eval_medmem.py \
    --input datasets/meddialog_cn/meddialog_cn_no_summary.parquet \
    --output_dir eval_results/meddialog_cn_ns_gpt4o \
    --model gpt-4o-2024-05-13 \
    --max_tokens 4096 \
    --workers 8

