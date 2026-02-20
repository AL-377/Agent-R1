python examples/dataset_collectors/huatuo26m_examiner.py \
    --input_path datasets/huatuo26m/train.jsonl \
    --output_path datasets/huatuo26m/examiner.json \
    --max_dialogues 1000 \
    --sessions_min 5 \
    --sessions_max 10 \
    --workers 16 \
    --model gpt-5.2-2025-12-11