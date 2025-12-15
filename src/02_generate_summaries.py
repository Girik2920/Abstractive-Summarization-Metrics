"""
Generate baseline (Lead-k) and neural model summaries for the dataset subset.
"""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import pipeline


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def sentence_split(text: str) -> List[str]:
    separators = ".!?"
    sentences = []
    current = []
    for char in text:
        current.append(char)
        if char in separators:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    if current:
        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)
    return sentences if sentences else [text.strip()]


def lead_k_summary(text: str, k: int) -> str:
    sentences = sentence_split(text)
    return " ".join(sentences[:k])


def generate_model_summaries(sources: List[str], config: dict) -> List[str]:
    model_name = config.get("model_name", "google/pegasus-xsum")
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("summarization", model=model_name, device=device)

    max_len = int(config.get("max_summary_length", 64))
    min_len = int(config.get("min_summary_length", 16))
    num_beams = int(config.get("num_beams", 4))

    summaries = []
    for text in tqdm(sources, desc="Generating model summaries"):
        output = generator(
            text,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_beams,
            do_sample=False,
        )[0]["summary_text"]
        summaries.append(output)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline and model summaries.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    random_seed = int(config.get("random_seed", 42))
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_path = output_dir / "dataset_subset.jsonl"
    subset_df = pd.read_json(subset_path, lines=True)

    k = int(config.get("baseline_sentences", 1))
    subset_df["lead_summary"] = subset_df["source"].apply(lambda x: lead_k_summary(str(x), k))

    subset_df["model_summary"] = generate_model_summaries(subset_df["source"].tolist(), config)

    summaries_path = output_dir / "summaries.csv"
    subset_df.to_csv(summaries_path, index=False)
    print(f"Saved summaries to {summaries_path}")


if __name__ == "__main__":
    main()
