python3 examples/dataset/carecall/memory_preprocess.py \
    --input examples/dataset/carecall/carecall-memory_en_auto_translated.json \
    --output examples/dataset/carecall/cmemory_eval_dataset_all.json \
    --max-patients 800 \
    --cache-dir /opt/tiger/Agent-R1/examples/dataset/carecall-cache \
    --model DeepSeek-R1 \
    --workers 8