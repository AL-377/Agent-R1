# 对比多个模型
python examples/eval/analyze_medmem.py \
    --result_dirs eval_results/cmtmedqa_ps_gpt4o \
                  eval_results/cmtmedqa_ps_deepseek \
                  eval_results/cmtmedqa_ns_gpt4o \
    --export_csv analysis.csv