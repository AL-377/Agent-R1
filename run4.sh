python examples/dataset_collectors/imcs21_examiner.py \
    --input_path datasets/imcs21/train.json \
    --output_path datasets/imcs21/examiner.json \
    --max_dialogues 1000 \
    --sessions_min 5 \
    --sessions_max 10 \
    --workers 16 \
    --model gpt-5.2-2025-12-11