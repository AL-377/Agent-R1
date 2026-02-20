"""
Convert MedDialog-CN examiner-processed data into swalm RL training format.

Usage:
    python examples/data_preprocess/meddialog_cn_memory_rl.py \
        --input_dir datasets/meddialog_cn/.examiner_cache \
        --output datasets/meddialog_cn/meddialog_cn_memory_rl.parquet

    # Generate only specific splits:
    python examples/data_preprocess/meddialog_cn_memory_rl.py \
        --input_dir datasets/meddialog_cn/.examiner_cache \
        --output datasets/meddialog_cn/meddialog_cn_memory_rl.parquet \
        --splits pure_summary no_summary
"""

import argparse
from examiner_to_rl_common import process_dataset, ALL_SPLITS


DATA_SOURCE = "memory/meddialog_cn"


def main():
    parser = argparse.ArgumentParser(
        description="Convert MedDialog-CN examiner data to swalm RL format"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing patient_*.json files (examiner cache)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Max patients to process (for testing)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Which context/prompt splits to generate (default: all three)",
    )
    args = parser.parse_args()

    process_dataset(
        args.input_dir,
        args.output,
        data_source_tag=DATA_SOURCE,
        splits=args.splits,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()
