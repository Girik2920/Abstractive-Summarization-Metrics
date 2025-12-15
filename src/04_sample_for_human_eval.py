"""
Sample examples for lightweight human evaluation.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample examples for human evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    random_seed = int(config.get("random_seed", 42))
    random.seed(random_seed)

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries_df = pd.read_csv(output_dir / "summaries.csv")

    sample_size = int(config.get("sample_size", 50))
    sample_df = summaries_df.sample(n=min(sample_size, len(summaries_df)), random_state=random_seed)

    annotation_columns = [
        "faithfulness_lead",
        "faithfulness_model",
        "coverage_lead",
        "coverage_model",
        "coherence_lead",
        "coherence_model",
        "notes",
    ]
    for column in annotation_columns:
        sample_df[column] = ""

    packet_path = output_dir / "human_eval_packet.csv"
    sample_df.to_csv(packet_path, index=False)
    print(f"Saved human evaluation packet to {packet_path}")


if __name__ == "__main__":
    main()
