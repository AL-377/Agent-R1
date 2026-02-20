# 只生成 pure_summary
python examples/data_preprocess/cmtmedqa_memory_rl.py \
    --input_dir datasets/cmtmedqa/.examiner_cache \
    --output datasets/cmtmedqa/cmtmedqa_pure_summary.parquet \
    --splits pure_summary

# 只生成 layers_summary
python examples/data_preprocess/cmtmedqa_memory_rl.py \
    --input_dir datasets/cmtmedqa/.examiner_cache \
    --output datasets/cmtmedqa/cmtmedqa_layers_summary.parquet \
    --splits layers_summary

# 只生成 no_summary
python examples/data_preprocess/cmtmedqa_memory_rl.py \
    --input_dir datasets/cmtmedqa/.examiner_cache \
    --output datasets/cmtmedqa/cmtmedqa_no_summary.parquet \
    --splits no_summary