
python examples/dataset_collectors/meddialog_cn_examiner.py \
    --input_path datasets/meddialog_cn/train.json \
    --output_path datasets/meddialog_cn/examiner.json \
    --max_dialogues 1000 \
    --sessions_min 5 \
    --sessions_max 10 \
    --workers 16 \
    --model gpt-5.2-2025-12-11-300