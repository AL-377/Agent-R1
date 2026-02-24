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

    # Export raw data + aggregated statistics:
    python examples/eval/analyze_medmem.py \
        --result_dirs eval_results/cmtmedqa_ps_gpt4o \
        --export_csv analysis_output.csv \
        --export_tables analysis_tables.xlsx
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List

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


def analyze(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all analyses. Returns an OrderedDict of {table_name: DataFrame}."""
    tables: Dict[str, pd.DataFrame] = OrderedDict()

    valid = df[df["correct"] >= 0].copy()
    if valid.empty:
        print("No valid results to analyze.")
        return tables

    print(f"\nTotal samples: {len(df)}, Valid (correct >= 0): {len(valid)}")

    # ---- Overall per model x split ----
    print_section("Overall Accuracy: model x split")
    agg = valid.groupby(["model", "split"]).agg(
        count=("correct", "size"),
        correct_sum=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    print(agg.to_string(index=False))
    tables["accuracy_model_split"] = agg

    # ---- layers_summary format validity ----
    layers = valid[valid["split"] == "layers_summary"]
    if not layers.empty and "format_valid" in layers.columns:
        print_section("layers_summary: Format Validity (model)")
        fmt_agg = layers.groupby("model").agg(
            count=("format_valid", "size"),
            valid_count=("format_valid", "sum"),
            valid_rate=("format_valid", "mean"),
        ).reset_index()
        print(fmt_agg.to_string(index=False))
        tables["format_validity"] = fmt_agg

    # ---- Compression ratio (pure_summary / layers_summary, correct only) ----
    cr_valid = valid[(valid["split"].isin(["pure_summary", "layers_summary"])) & (valid["correct"] == 1.0)]
    cr_valid = cr_valid.dropna(subset=["compression_ratio"])
    if not cr_valid.empty:
        print_section("Compression Ratio (correct only): model x split")
        cr_agg = cr_valid.groupby(["model", "split"]).agg(
            count=("compression_ratio", "size"),
            mean_cr=("compression_ratio", "mean"),
            median_cr=("compression_ratio", "median"),
            min_cr=("compression_ratio", "min"),
            max_cr=("compression_ratio", "max"),
        ).reset_index()
        print(cr_agg.to_string(index=False))
        tables["compress_ratio_model_split"] = cr_agg

    # ---- Per scenario ----
    if valid["scenario"].notna().any() and (valid["scenario"] != "").any():
        print_section("Accuracy by scenario x model")
        sc_agg = valid.groupby(["scenario", "scenario_name", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(sc_agg.to_string(index=False))
        tables["accuracy_scenario"] = sc_agg

        if not cr_valid.empty:
            print_section("Compression Ratio by scenario (correct only)")
            sc_cr = cr_valid.groupby(["scenario", "model"]).agg(
                count=("compression_ratio", "size"),
                mean_cr=("compression_ratio", "mean"),
            ).reset_index()
            print(sc_cr.to_string(index=False))
            tables["compress_ratio_scenario"] = sc_cr

    # ---- Per query type ----
    if valid["query_type"].notna().any() and (valid["query_type"] != "").any():
        print_section("Accuracy by query_type x model")
        qt_agg = valid.groupby(["query_type", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(qt_agg.to_string(index=False))
        tables["accuracy_query_type"] = qt_agg

        if not cr_valid.empty:
            print_section("Compression Ratio by query_type (correct only)")
            qt_cr = cr_valid.groupby(["query_type", "model"]).agg(
                count=("compression_ratio", "size"),
                mean_cr=("compression_ratio", "mean"),
            ).reset_index()
            print(qt_cr.to_string(index=False))
            tables["compress_ratio_query_type"] = qt_cr

    # ---- Per difficulty ----
    if valid["difficulty"].notna().any() and (valid["difficulty"] != "").any():
        print_section("Accuracy by difficulty x model")
        diff_agg = valid.groupby(["difficulty", "model", "split"]).agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
        ).reset_index()
        print(diff_agg.to_string(index=False))
        tables["accuracy_difficulty"] = diff_agg

    return tables


def export_tables(tables: Dict[str, pd.DataFrame], path: str):
    """Export all aggregated tables. Supports .xlsx (multi-sheet) and .csv (concatenated)."""
    if path.endswith(".xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, tbl in tables.items():
                sheet = name[:31]  # Excel sheet name limit
                tbl.to_excel(writer, sheet_name=sheet, index=False)
        print(f"\nExported {len(tables)} tables to {path}")
    else:
        with open(path, "w") as f:
            for name, tbl in tables.items():
                f.write(f"# {name}\n")
                tbl.to_csv(f, index=False)
                f.write("\n")
        print(f"\nExported {len(tables)} tables to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MedMem evaluation results")
    parser.add_argument(
        "--result_dirs", nargs="+", required=True,
        help="Directories containing results.parquet from eval_medmem.py",
    )
    parser.add_argument(
        "--export_csv", type=str, default=None,
        help="Export the merged raw DataFrame to CSV",
    )
    parser.add_argument(
        "--export_parquet", type=str, default=None,
        help="Export the merged raw DataFrame to parquet",
    )
    parser.add_argument(
        "--export_tables", type=str, default=None,
        help="Export aggregated statistics tables (.xlsx for multi-sheet, or .csv)",
    )
    args = parser.parse_args()

    print("Loading results...")
    df = load_results(args.result_dirs)

    tables = analyze(df)

    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"\nExported merged raw data to {args.export_csv}")
    if args.export_parquet:
        df.to_parquet(args.export_parquet, index=False)
        print(f"\nExported merged raw data to {args.export_parquet}")
    if args.export_tables and tables:
        export_tables(tables, args.export_tables)


if __name__ == "__main__":
    main()
