python examples/dataset/carecall/examiner_agent_preprocess.py \
  --input_dir examples/dataset/carecall-cache \
  --output_dir examples/dataset/carecall-examiner \
  --workers 32 \
  --cache_dir .examiner_cache \
  --model_name DeepSeek-R1 \
  --trigger_prob 0.7