#!/bin/bash
set -e

echo "=== Converting CareCall ==="
bash scripts/data_carecall_rl_format.sh

echo "=== Converting MedDialog-CN ==="
bash scripts/data_meddialog_cn_rl_format.sh

echo "=== Converting CMtMedQA ==="
bash scripts/data_cmtmedqa_rl_format.sh

echo "=== Converting Huatuo-26M ==="
bash scripts/data_huatuo26m_rl_format.sh

echo "=== All done ==="
