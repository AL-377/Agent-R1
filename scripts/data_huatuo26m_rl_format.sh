# 只生成 pure_summary
python examples/data_preprocess/huatuo26m_memory_rl.py \
    --input_dir datasets/huatuo26m/.examiner_cache \
    --output datasets/huatuo26m/huatuo26m_pure_summary.parquet \
    --splits pure_summary

# 只生成 layers_summary
python examples/data_preprocess/huatuo26m_memory_rl.py \
    --input_dir datasets/huatuo26m/.examiner_cache \
    --output datasets/huatuo26m/huatuo26m_layers_summary.parquet \
    --splits layers_summary

# 只生成 no_summary
python examples/data_preprocess/huatuo26m_memory_rl.py \
    --input_dir datasets/huatuo26m/.examiner_cache \
    --output datasets/huatuo26m/huatuo26m_no_summary.parquet \
    --splits no_summary