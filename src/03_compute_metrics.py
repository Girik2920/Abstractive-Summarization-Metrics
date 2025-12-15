"""
Compute ROUGE and BERTScore for baseline and model summaries.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from evaluate import load as load_metric
from rouge_score import rouge_scorer


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, List[float]]:
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    results = {metric: [] for metric in ROUGE_TYPES}
    for reference, prediction in zip(references, predictions):
        scores = scorer.score(reference, prediction)
        for metric in ROUGE_TYPES:
            results[metric].append(scores[metric].fmeasure)
    return results


def compute_bertscore(references: List[str], predictions: List[str]) -> List[float]:
    bertscore = load_metric("bertscore")
    outputs = bertscore.compute(predictions=predictions, references=references, lang="en")
    return outputs["f1"]


def attach_metrics(df: pd.DataFrame, prefix: str, references: List[str], predictions: List[str]) -> pd.DataFrame:
    rouge_scores = compute_rouge(references, predictions)
    bert_scores = compute_bertscore(references, predictions)

    df[f"{prefix}_len"] = [len(pred.split()) for pred in predictions]
    for metric in ROUGE_TYPES:
        df[f"{metric}_{prefix}"] = rouge_scores[metric]
    df[f"bert_{prefix}"] = bert_scores
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROUGE and BERTScore metrics.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries_path = output_dir / "summaries.csv"
    summaries_df = pd.read_csv(summaries_path)

    metrics_df = pd.DataFrame()
    metrics_df["id"] = summaries_df["id"]

    metrics_df = attach_metrics(
        metrics_df,
        prefix="lead",
        references=summaries_df["reference"].tolist(),
        predictions=summaries_df["lead_summary"].tolist(),
    )
    metrics_df = attach_metrics(
        metrics_df,
        prefix="model",
        references=summaries_df["reference"].tolist(),
        predictions=summaries_df["model_summary"].tolist(),
    )

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
