# 对比多个结果
python examples/eval/analyze_medmem.py \
    --result_dirs eval_results/meddialog_cn_ns_dpskr1 \
                  eval_results/meddialog_cn_ns_gemini25 \
                  eval_results/meddialog_cn_ns_gpt4o \
                  eval_results/meddialog_cn_ns_gpt4omini \
                  eval_results/meddialog_cn_ns_gpt5 \
                  eval_results/meddialog_cn_ns_q3w8b \
                  eval_results/meddialog_cn_ns_q3w14b \
                  eval_results/meddialog_cn_ps_dpskr1 \
                  eval_results/meddialog_cn_ps_gemini25 \
                  eval_results/meddialog_cn_ps_gpt4o \
                  eval_results/meddialog_cn_ps_gpt4omini \
                  eval_results/meddialog_cn_ps_gpt5 \
                  eval_results/meddialog_cn_ps_q3w8b \
                  eval_results/meddialog_cn_ps_q3w14b \
    --export_csv /opt/tiger/Agent-R1/datasets/meddialog_cn_res.csv \
    --export_tables /opt/tiger/Agent-R1/datasets/meddialog_cn_tables.xlsx