"""
Analyze dialogue quality evaluation results from eval_dialogue_quality.py.

Loads results.parquet, computes per-dimension score comparisons (with_memory
vs no_memory), and generates tables and charts suitable for the thesis.

Usage:
    python examples/eval/analyze_dialogue_quality.py \
        --result_dirs eval_results/dialogue_quality \
        --output_dir eval_results/dialogue_quality/analysis

    python examples/eval/analyze_dialogue_quality.py \
        --result_dirs eval_results/dq_gpt4o eval_results/dq_qwen3 \
        --output_dir eval_results/dialogue_quality/analysis \
        --export_tables analysis_tables.xlsx
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import pandas as pd

DIMENSIONS = [
    "medical_accuracy",
    "personalization",
    "consistency",
    "completeness",
    "safety",
]

DIMENSION_CN = {
    "medical_accuracy": "医学准确性",
    "personalization": "个性化程度",
    "consistency": "信息一致性",
    "completeness": "回答完整性",
    "safety": "安全性",
}


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
        raise FileNotFoundError("No results.parquet found.")
    return pd.concat(dfs, ignore_index=True)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def compute_delta_table(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """Compute with_memory vs no_memory score comparison grouped by given columns."""
    rows = []
    for group_vals, grp in df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        row = dict(zip(group_cols, group_vals))
        row["count"] = len(grp)
        for dim in DIMENSIONS:
            wm = grp[f"with_memory_{dim}"]
            nm = grp[f"no_memory_{dim}"]
            valid_wm = wm[wm > 0]
            valid_nm = nm[nm > 0]
            row[f"{dim}_with"] = valid_wm.mean() if len(valid_wm) > 0 else None
            row[f"{dim}_no"] = valid_nm.mean() if len(valid_nm) > 0 else None
            if row[f"{dim}_with"] is not None and row[f"{dim}_no"] is not None:
                row[f"{dim}_delta"] = row[f"{dim}_with"] - row[f"{dim}_no"]
            else:
                row[f"{dim}_delta"] = None
        # Average across dimensions
        deltas = [row[f"{d}_delta"] for d in DIMENSIONS if row.get(f"{d}_delta") is not None]
        row["avg_with"] = (
            sum(row[f"{d}_with"] for d in DIMENSIONS if row.get(f"{d}_with") is not None)
            / max(sum(1 for d in DIMENSIONS if row.get(f"{d}_with") is not None), 1)
        )
        row["avg_no"] = (
            sum(row[f"{d}_no"] for d in DIMENSIONS if row.get(f"{d}_no") is not None)
            / max(sum(1 for d in DIMENSIONS if row.get(f"{d}_no") is not None), 1)
        )
        row["avg_delta"] = sum(deltas) / max(len(deltas), 1) if deltas else None
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run all analyses. Returns OrderedDict of {name: DataFrame}."""
    tables: Dict[str, pd.DataFrame] = OrderedDict()

    # Filter valid rows (at least one dimension scored)
    has_scores = df[[f"with_memory_{d}" for d in DIMENSIONS]].max(axis=1) > 0
    valid = df[has_scores].copy()
    if valid.empty:
        print("No valid results to analyze.")
        return tables

    print(f"\nTotal rows: {len(df)}, Valid: {len(valid)}")

    # --- 1. Overall dimension comparison ---
    section("Overall Dimension Scores: with_memory vs no_memory")
    overall_rows = []
    for dim in DIMENSIONS:
        wm = valid[f"with_memory_{dim}"]
        nm = valid[f"no_memory_{dim}"]
        wm_valid = wm[wm > 0]
        nm_valid = nm[nm > 0]
        if len(wm_valid) > 0 and len(nm_valid) > 0:
            delta = wm_valid.mean() - nm_valid.mean()
            overall_rows.append({
                "dimension": dim,
                "dimension_cn": DIMENSION_CN.get(dim, dim),
                "with_memory": round(wm_valid.mean(), 3),
                "no_memory": round(nm_valid.mean(), 3),
                "delta": round(delta, 3),
                "count": len(wm_valid),
            })
    overall_df = pd.DataFrame(overall_rows)
    print(overall_df.to_string(index=False))
    tables["overall_dimensions"] = overall_df

    # --- 2. Per chat_model comparison ---
    if "chat_model" in valid.columns and valid["chat_model"].nunique() > 1:
        section("Per Chat Model Comparison")
        model_tbl = compute_delta_table(valid, ["chat_model"])
        print(model_tbl.to_string(index=False))
        tables["per_chat_model"] = model_tbl

    # --- 3. Per data source ---
    if "source" in valid.columns and valid["source"].nunique() > 1:
        section("Per Data Source Comparison")
        source_tbl = compute_delta_table(valid, ["source"])
        print(source_tbl.to_string(index=False))
        tables["per_source"] = source_tbl

    # --- 4. Per difficulty ---
    if "difficulty" in valid.columns and valid["difficulty"].notna().any():
        section("Per Difficulty Level")
        diff_tbl = compute_delta_table(valid, ["difficulty"])
        print(diff_tbl.to_string(index=False))
        tables["per_difficulty"] = diff_tbl

    # --- 5. Per query_type ---
    if "query_type" in valid.columns and valid["query_type"].notna().any():
        section("Per Query Type")
        qt_tbl = compute_delta_table(valid, ["query_type"])
        print(qt_tbl.to_string(index=False))
        tables["per_query_type"] = qt_tbl

    # --- 6. Score distribution ---
    section("Score Distribution (with_memory)")
    dist_rows = []
    for dim in DIMENSIONS:
        col = f"with_memory_{dim}"
        vals = valid[col][valid[col] > 0]
        if len(vals) > 0:
            dist_rows.append({
                "dimension": dim,
                "mean": round(vals.mean(), 2),
                "std": round(vals.std(), 2),
                "min": int(vals.min()),
                "median": round(vals.median(), 1),
                "max": int(vals.max()),
                "score_1_pct": round((vals == 1).mean() * 100, 1),
                "score_5_pct": round((vals == 5).mean() * 100, 1),
            })
    dist_df = pd.DataFrame(dist_rows)
    print(dist_df.to_string(index=False))
    tables["score_distribution_with_memory"] = dist_df

    # --- 7. Win/Tie/Loss analysis ---
    section("Win/Tie/Loss Analysis (with_memory vs no_memory)")
    wtl_rows = []
    for dim in DIMENSIONS:
        wm = valid[f"with_memory_{dim}"]
        nm = valid[f"no_memory_{dim}"]
        mask = (wm > 0) & (nm > 0)
        wm_v = wm[mask]
        nm_v = nm[mask]
        n = len(wm_v)
        if n > 0:
            win = (wm_v > nm_v).sum()
            tie = (wm_v == nm_v).sum()
            loss = (wm_v < nm_v).sum()
            wtl_rows.append({
                "dimension": dim,
                "win": win,
                "tie": tie,
                "loss": loss,
                "win_rate": round(win / n * 100, 1),
                "tie_rate": round(tie / n * 100, 1),
                "loss_rate": round(loss / n * 100, 1),
                "total": n,
            })
    wtl_df = pd.DataFrame(wtl_rows)
    print(wtl_df.to_string(index=False))
    tables["win_tie_loss"] = wtl_df

    return tables


def generate_charts(
    tables: Dict[str, pd.DataFrame],
    output_dir: str,
):
    """Generate PDF charts for the thesis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping chart generation.")
        return

    plt.rcParams["font.sans-serif"] = ["Heiti TC", "Hiragino Sans GB", "PingFang HK", "STHeiti", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Chart 1: Overall dimension comparison bar chart
    if "overall_dimensions" in tables:
        df = tables["overall_dimensions"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        width = 0.35
        bars1 = ax.bar(x - width / 2, df["with_memory"], width, label="With Memory", color="#4472C4")
        bars2 = ax.bar(x + width / 2, df["no_memory"], width, label="No Memory", color="#ED7D31")
        ax.set_xlabel("Evaluation Dimension")
        ax.set_ylabel("Average Score (1-5)")
        ax.set_title("Dialogue Quality: With Memory vs No Memory")
        ax.set_xticks(x)
        ax.set_xticklabels(df["dimension_cn"], rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 5.5)
        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, "dialogue_quality_comparison.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart: {path}")

    # Chart 2: Win/Tie/Loss stacked bar
    if "win_tie_loss" in tables:
        df = tables["win_tie_loss"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        ax.bar(x, df["win_rate"], label="Win (Memory better)", color="#4472C4")
        ax.bar(x, df["tie_rate"], bottom=df["win_rate"], label="Tie", color="#A5A5A5")
        ax.bar(x, df["loss_rate"], bottom=df["win_rate"] + df["tie_rate"],
               label="Loss (Memory worse)", color="#ED7D31")
        ax.set_xlabel("Evaluation Dimension")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Win/Tie/Loss: With Memory vs No Memory")
        ax.set_xticks(x)
        dim_labels = [DIMENSION_CN.get(d, d) for d in df["dimension"]]
        ax.set_xticklabels(dim_labels, rotation=15, ha="right")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 105)
        plt.tight_layout()
        path = os.path.join(output_dir, "dialogue_quality_win_tie_loss.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart: {path}")

    # Chart 3: Per-difficulty delta radar/bar chart
    if "per_difficulty" in tables:
        df = tables["per_difficulty"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        delta_cols = [f"{d}_delta" for d in DIMENSIONS]
        width = 0.15
        for i, dim in enumerate(DIMENSIONS):
            col = f"{dim}_delta"
            if col in df.columns:
                vals = df[col].fillna(0)
                ax.bar(x + i * width, vals, width, label=DIMENSION_CN.get(dim, dim))
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Score Delta (with_memory - no_memory)")
        ax.set_title("Memory Enhancement Effect by Difficulty Level")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df["difficulty"])
        ax.legend(fontsize=7, loc="upper left")
        ax.axhline(y=0, color="black", linewidth=0.5)
        plt.tight_layout()
        path = os.path.join(output_dir, "dialogue_quality_by_difficulty.pdf")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart: {path}")


def export_tables(tables: Dict[str, pd.DataFrame], path: str):
    if path.endswith(".xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, tbl in tables.items():
                tbl.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"\nExported {len(tables)} tables to {path}")
    else:
        with open(path, "w") as f:
            for name, tbl in tables.items():
                f.write(f"# {name}\n")
                tbl.to_csv(f, index=False)
                f.write("\n")
        print(f"\nExported {len(tables)} tables to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dialogue quality evaluation results")
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--export_tables", type=str, default=None)
    parser.add_argument("--no_charts", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dirs[0], "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    df = load_results(args.result_dirs)
    tables = analyze(df)

    # Write Overall Dimension Scores + Win/Tie/Loss to txt in output_dir
    if tables:
        txt_path = os.path.join(args.output_dir, "summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if "overall_dimensions" in tables:
                f.write("=" * 60 + "\n")
                f.write("  Overall Dimension Scores: with_memory vs no_memory\n")
                f.write("=" * 60 + "\n")
                cols = [c for c in tables["overall_dimensions"].columns if c != "delta"]
                f.write(tables["overall_dimensions"][cols].to_string(index=False))
                f.write("\n\n")
            if "win_tie_loss" in tables:
                f.write("=" * 60 + "\n")
                f.write("  Win/Tie/Loss Analysis\n")
                f.write("=" * 60 + "\n")
                f.write(tables["win_tie_loss"].to_string(index=False))
                f.write("\n\n")
            if "per_difficulty" in tables:
                f.write("=" * 60 + "\n")
                f.write("  Per Difficulty Level\n")
                f.write("=" * 60 + "\n")
                f.write(tables["per_difficulty"].to_string(index=False))
                f.write("\n")
        print(f"Summary saved to {txt_path}")

    if not args.no_charts:
        generate_charts(tables, args.output_dir)

    if args.export_tables and tables:
        export_tables(tables, args.export_tables)


if __name__ == "__main__":
    main()
