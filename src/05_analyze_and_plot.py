"""
Analyze metric-human alignment and create illustrative plots.
"""

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

sns.set_style("whitegrid")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_annotations(output_dir: Path) -> pd.DataFrame:
    annotations_path = output_dir / "human_annotations.csv"
    packet_path = output_dir / "human_eval_packet.csv"

    if annotations_path.exists():
        return pd.read_csv(annotations_path)
    if packet_path.exists():
        return pd.read_csv(packet_path)
    raise FileNotFoundError("No human annotation file found.")


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rouge1_lead",
        "rouge1_model",
        "rougeL_lead",
        "rougeL_model",
        "bert_lead",
        "bert_model",
    ]
    human_cols = [
        "faithfulness_lead",
        "faithfulness_model",
        "coverage_lead",
        "coverage_model",
    ]

    for col in human_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    rows = []
    for metric in metric_cols:
        for human in human_cols:
            corr = merged[[metric, human]].corr(method="spearman").iloc[0, 1]
            rows.append({"metric": metric, "human": human, "spearman": corr})
    return pd.DataFrame(rows)


def find_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["faithfulness_lead"] = pd.to_numeric(merged["faithfulness_lead"], errors="coerce")
    merged["faithfulness_model"] = pd.to_numeric(merged["faithfulness_model"], errors="coerce")

    disagreement = merged[
        (
            (merged["rougeL_model"] > merged["rougeL_lead"]) &
            (merged["faithfulness_model"] < merged["faithfulness_lead"])
        )
    ]
    return disagreement.sort_values(by="rougeL_model", ascending=False)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram(df: pd.DataFrame, column: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], bins=20, kde=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze metric-human alignment and plot results.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(output_dir / "metrics.csv")

    try:
        annotations_df = load_annotations(output_dir)
        merged = metrics_df.merge(annotations_df, on="id", how="left")
    except FileNotFoundError:
        print("No human annotations found; analysis limited to automatic metrics.")
        merged = metrics_df

    if {"faithfulness_lead", "faithfulness_model"}.issubset(merged.columns):
        correlation_df = compute_correlations(merged)
        correlation_path = output_dir / "metric_human_correlations.csv"
        correlation_df.to_csv(correlation_path, index=False)
        print(f"Saved correlation table to {correlation_path}")

        scatter_path = figures_dir / "rouge_vs_faithfulness.png"
        plot_scatter(
            merged,
            x="rougeL_model" if "rougeL_model" in merged else merged.columns[1],
            y="faithfulness_model",
            title="ROUGE-L vs. Human Faithfulness (Model)",
            path=scatter_path,
        )

    length_scatter = figures_dir / "rouge_vs_length.png"
    plot_scatter(
        metrics_df,
        x="model_len" if "model_len" in metrics_df.columns else metrics_df.columns[1],
        y="rougeL_model" if "rougeL_model" in metrics_df.columns else metrics_df.columns[2],
        title="ROUGE-L vs. Summary Length (Model)",
        path=length_scatter,
    )

    diff_column = "rougeL_model" if "rougeL_model" in metrics_df.columns else metrics_df.columns[2]
    base_column = "rougeL_lead" if "rougeL_lead" in metrics_df.columns else metrics_df.columns[1]
    metrics_df["rougeL_diff"] = metrics_df[diff_column] - metrics_df[base_column]
    diff_hist = figures_dir / "rouge_diff_hist.png"
    plot_histogram(metrics_df, "rougeL_diff", "ROUGE-L Difference (Model - Lead)", diff_hist)

    if {"faithfulness_lead", "faithfulness_model"}.issubset(merged.columns):
        disagreement_df = find_disagreements(merged)
        disagreement_path = output_dir / "disagreement_examples.csv"
        disagreement_df.to_csv(disagreement_path, index=False)
        print(f"Saved disagreement set to {disagreement_path}")


if __name__ == "__main__":
    main()
