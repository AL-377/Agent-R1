"""
Analyze evaluation results from eval_medmem.py.

Loads results.parquet from one or more eval output directories, merges them
into a single DataFrame, and computes per-scenario, per-type, per-model
accuracy and compression ratio statistics.

Usage:
    # Single model results:
    python examples/eval/analyze_medmem.py \
        --result_dirs eval_results/cmtmedqa_pure_summary

    # Compare multiple models:
    python examples/eval/analyze_medmem.py \
        --result_dirs eval_results/cmtmedqa_ps_gpt4o \
                      eval_results/cmtmedqa_ps_deepseek \
                      eval_results/cmtmedqa_ns_gpt4o

    # Export to CSV:
    python examples/eval/analyze_medmem.py \
        --result_dirs eval_results/cmtmedqa_ps_gpt4o \
        --export_csv analysis_output.csv
"""

import argparse
import os
from typing import List

import pandas as pd


def load_results(result_dirs: List[str]) -> pd.DataFrame:
    dfs = []
    for d in result_dirs:
        path = os.path.join(d, "results.parquet")
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
            print(f"  Loaded {path}: {len(dfs[-1])} rows")
        else:
            print(f"  WARNING: {path} not found, skipping")
    if not dfs:
        raise FileNotFoundError("No results.parquet found in any of the provided dirs.")
    return pd.concat(dfs, ignore_index=True)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze(df: pd.DataFrame):
    valid = df[df["correct"] >= 0].copy()
    if valid.empty:
        print("No valid results to analyze.")
        return

    print(f"\nTotal samples: {len(df)}, Valid (correct >= 0): {len(valid)}")

    # ---- Overall per model x split ----
    print_section("Overall Accuracy: model x split")
    agg = valid.groupby(["model", "split"]).agg(
        count=("correct", "size"),
        correct_sum=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))

    # ---- Compression ratio (pure_summary, correct only) ----
    cr_valid = valid[(valid["split"] == "pure_summary") & (valid["correct"] == 1.0)]
    cr_valid = cr_valid.dropna(subset=["compression_ratio"])
    if not cr_valid.empty:
        print_section("Compression Ratio (pure_summary, correct only): model")
        cr_agg = cr_valid.groupby("model").agg(
            count=("compression_ratio", "size"),
            mean_cr=("compression_ratio", "mean"),
            median_cr=("compression_ratio", "median"),
            min_cr=("compression_ratio", "min"),
            max_cr=("compression_ratio", "max"),
        ).reset_index()
        print(cr_agg.to_string(index=False))

    # ---- Per scenario ----
    if valid["scenario"].notna().any() and (valid["scenario"] != "").any():
        print_section("Accuracy by scenario x model")
        sc_agg = valid.groupby(["scenario", "scenario_name", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(sc_agg.to_string(index=False))

        if not cr_valid.empty:
            print_section("Compression Ratio by scenario (correct only)")
            sc_cr = cr_valid.groupby(["scenario", "model"]).agg(
                count=("compression_ratio", "size"),
                mean_cr=("compression_ratio", "mean"),
            ).reset_index()
            print(sc_cr.to_string(index=False))

    # ---- Per query type ----
    if valid["query_type"].notna().any() and (valid["query_type"] != "").any():
        print_section("Accuracy by query_type x model")
        qt_agg = valid.groupby(["query_type", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(qt_agg.to_string(index=False))

        if not cr_valid.empty:
            print_section("Compression Ratio by query_type (correct only)")
            qt_cr = cr_valid.groupby(["query_type", "model"]).agg(
                count=("compression_ratio", "size"),
                mean_cr=("compression_ratio", "mean"),
            ).reset_index()
            print(qt_cr.to_string(index=False))

    # ---- Per difficulty ----
    if valid["difficulty"].notna().any() and (valid["difficulty"] != "").any():
        print_section("Accuracy by difficulty x model")
        diff_agg = valid.groupby(["difficulty", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(diff_agg.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze MedMem evaluation results")
    parser.add_argument(
        "--result_dirs", nargs="+", required=True,
        help="Directories containing results.parquet from eval_medmem.py",
    )
    parser.add_argument(
        "--export_csv", type=str, default=None,
        help="Export the merged DataFrame to CSV",
    )
    parser.add_argument(
        "--export_parquet", type=str, default=None,
        help="Export the merged DataFrame to parquet",
    )
    args = parser.parse_args()

    print("Loading results...")
    df = load_results(args.result_dirs)

    analyze(df)

    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"\nExported merged data to {args.export_csv}")
    if args.export_parquet:
        df.to_parquet(args.export_parquet, index=False)
        print(f"\nExported merged data to {args.export_parquet}")


if __name__ == "__main__":
    main()
