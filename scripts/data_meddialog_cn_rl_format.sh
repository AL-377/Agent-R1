# 只生成 pure_summary
python examples/data_preprocess/meddialog_cn_memory_rl.py \
    --input_dir datasets/meddialog_cn/.examiner_cache \
    --output datasets/meddialog_cn/meddialog_cn_pure_summary.parquet \
    --splits pure_summary

# 只生成 layers_summary
python examples/data_preprocess/meddialog_cn_memory_rl.py \
    --input_dir datasets/meddialog_cn/.examiner_cache \
    --output datasets/meddialog_cn/meddialog_cn_layers_summary.parquet \
    --splits layers_summary

# 只生成 no_summary
python examples/data_preprocess/meddialog_cn_memory_rl.py \
    --input_dir datasets/meddialog_cn/.examiner_cache \
    --output datasets/meddialog_cn/meddialog_cn_no_summary.parquet \
    --splits no_summary