"""
Prepare a reproducible subset of the chosen summarization dataset.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def prepare_subset(config: dict) -> pd.DataFrame:
    random_seed = int(config.get("random_seed", 42))
    random.seed(random_seed)

    dataset_name = config.get("dataset_name", "xsum")
    split = config.get("split", "test")
    subset_size = int(config.get("subset_size", 50))

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=random_seed)
    dataset = dataset.select(range(min(subset_size, len(dataset))))

    records = []
    for idx, row in enumerate(dataset):
        record_id = row.get("id", f"{split}-{idx}")
        records.append({
            "id": record_id,
            "source": row.get("document") or row.get("article"),
            "reference": row.get("summary"),
        })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset subset for summarization evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_df = prepare_subset(config)
    output_path = output_dir / "dataset_subset.jsonl"
    with output_path.open("w", encoding="utf-8") as fp:
        for record in subset_df.to_dict(orient="records"):
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(subset_df)} examples to {output_path}")


if __name__ == "__main__":
    main()
