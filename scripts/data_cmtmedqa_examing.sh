python examples/dataset_collectors/cmtmedqa_examiner.py \
    --input_path datasets/cmtmedqa/train.json \
    --output_path datasets/cmtmedqa/examiner.json \
    --max_dialogues 1000 \
    --sessions_min 5 \
    --sessions_max 10 \
    --workers 16 \
    --model gpt-5.2-2025-12-11