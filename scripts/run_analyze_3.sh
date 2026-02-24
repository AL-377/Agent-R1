# 对比多个结果
python examples/eval/analyze_medmem.py \
    --result_dirs eval_results/huatuo26m_ns_dpskr1 \
                  eval_results/huatuo26m_ns_gemini25 \
                  eval_results/huatuo26m_ns_gpt4o \
                  eval_results/huatuo26m_ns_gpt4omini \
                  eval_results/huatuo26m_ns_gpt5 \
                  eval_results/huatuo26m_ns_q3w8b \
                  eval_results/huatuo26m_ns_q3w14b \
                  eval_results/huatuo26m_ps_dpskr1 \
                  eval_results/huatuo26m_ps_gemini25 \
                  eval_results/huatuo26m_ps_gpt4o \
                  eval_results/huatuo26m_ps_gpt4omini \
                  eval_results/huatuo26m_ps_gpt5 \
                  eval_results/huatuo26m_ps_q3w8b \
                  eval_results/huatuo26m_ps_q3w14b \
    --export_csv /opt/tiger/Agent-R1/datasets/huatuo26m_res.csv \
    --export_tables /opt/tiger/Agent-R1/datasets/huatuo26m_tables.xlsx