# python examples/dataset_collectors/cmtmedqa_examiner.py \
#     --input_path datasets/cmtmedqa/train.json \
#     --output_path datasets/cmtmedqa/examiner.json \
#     --max_dialogues 10 \
#     --sessions_min 2 \
#     --sessions_max 3


# python examples/dataset_collectors/huatuo26m_examiner.py \
#     --input_path datasets/huatuo26m/train.jsonl \
#     --output_path datasets/huatuo26m/examiner.json \
#     --max_dialogues 10 \
#     --sessions_min 2 \
#     --sessions_max 3

# python examples/dataset_collectors/imcs21_examiner.py \
#     --input_path datasets/imcs21/train.json \
#     --output_path datasets/imcs21/examiner.json \
#     --max_dialogues 10 \
#     --sessions_min 2 \
#     --sessions_max 3


python examples/dataset_collectors/meddialog_cn_examiner.py \
    --input_path datasets/meddialog_cn/train.json \
    --output_path datasets/meddialog_cn/examiner.json \
    --max_dialogues 10 \
    --sessions_min 2 \
    --sessions_max 3