# Abstractive Summarization Metrics

**Authors**: Girik Tripathi, Shriniketh Shankar, Harshita Murali, Pradeep Arjun

## Abstract

This project implements a reproducible pipeline to investigate the limitations of automatic summarization evaluation metrics (ROUGE, BERTScore) in capturing human notions of faithfulness. We evaluate baseline (Lead-k) and neural (BART) summarization approaches on 50 articles from the XSum dataset, computing multiple automatic metrics and preparing materials for human evaluation. The analysis reveals systematic disagreements between automatic metrics and expected quality judgments, highlighting the importance of multi-faceted evaluation approaches in summarization research.

## Introduction

Automatic evaluation metrics like ROUGE have become standard in summarization research due to their computational efficiency. However, these metrics often fail to capture semantic quality and factual faithfulness—critical dimensions of summary quality. This project provides a complete pipeline for metric validation and human evaluation, demonstrating where and why automatic metrics diverge from human judgment.

## Methodology

### Dataset
- **Name**: XSum (Extreme Summarization)
- **Source**: BBC News articles
- **Size**: 50 articles (test split)
- **Article Length**: ~600 words average
- **Reference Summaries**: 1 professional summary per article
- **Citation**: Narayan et al. (2018)

### Summarization Approaches
1. **Baseline**: Lead-k extraction (k=1)
   - Extracts first sentence
   - Serves as competitive baseline
   
2. **Neural**: facebook/bart-base
   - Encoder-decoder transformer (140M parameters)
   - Fine-tuned on CNN/DailyMail
   - Generates abstractive summaries

### Evaluation Metrics
| Metric | Type | Measures | Strengths | Limitations |
|--------|------|----------|-----------|-------------|
| **ROUGE-1/2/L** | Lexical | n-gram overlap | Simple, language-agnostic | Ignores semantics, paraphrase-sensitive |
| **BERTScore** | Semantic | Contextual similarity | Semantic awareness, robust to paraphrasing | Computationally expensive, not human-aligned |

## Key Findings

1. **Metric Disagreement**: ROUGE and BERTScore often rank summaries differently, suggesting they capture orthogonal quality dimensions

2. **Length Bias**: ROUGE exhibits clear correlation with summary length, favoring longer outputs

3. **Model Trade-offs**: 
   - Baseline (Lead-k) achieves higher ROUGE scores
   - Neural (BART) achieves higher BERTScore but lower ROUGE
   - Suggests metrics don't capture all quality aspects

4. **Faithfulness Gap**: Automatic metrics insufficient for evaluating factual accuracy—human judgment essential

## Dependencies

- **Python**: 3.8+
- **Core Libraries**: transformers, datasets, torch, pandas
- **Metrics**: rouge-score, bert-score
- **Visualization**: matplotlib, seaborn
- **Web Framework**: streamlit
- **Full list**: See `requirements.txt`

## Configuration

Edit `config.yaml` to customize:
```yaml
dataset_name: xsum              # Dataset to use
split: test                     # Train/validation/test split
subset_size: 50                 # Number of examples
model_name: facebook/bart-base  # Summarization model
max_summary_length: 64          # Max output tokens
num_beams: 4                    # Beam search width
```

## References

1. Narayan, S., Cohen, S. B., & Lapata, M. (2018). *Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization*. arXiv preprint arXiv:1808.08745.

2. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. arXiv preprint arXiv:1910.13461.

3. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. In Text summarization branches out (pp. 74-81).

4. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. arXiv preprint arXiv:1904.09675.

## License

This project is for educational purposes.

## Acknowledgments

- XSum dataset authors (University of Edinburgh NLP group)
- HuggingFace team for transformers and datasets libraries

## Project Structure
```
.
├── README.md                  # This file
├── PROJECT_REPORT.md          # Detailed technical report
├── config.yaml                # Configuration parameters
├── requirements.txt           # Python dependencies
├── data/                      # Input data directory
├── outputs/                   # Generated outputs
│   ├── dataset_subset.jsonl   # 50 articles with references
│   ├── summaries.csv          # Baseline + neural summaries
│   ├── metrics.csv            # ROUGE/BERTScore results
│   ├── human_eval_packet.csv  # Annotation-ready summaries
│   ├── metric_human_correlations.csv
│   ├── disagreement_examples.csv
│   └── figures/               # Visualization plots
│       ├── rouge_vs_length.png
│       ├── rouge_vs_faithfulness.png
│       └── rouge_diff_hist.png
├── src/                       # Source code
│   ├── 01_prepare_data.py          # Dataset preparation
│   ├── 02_generate_summaries.py    # Baseline + neural generation
│   ├── 03_compute_metrics.py       # ROUGE/BERTScore computation
│   ├── 04_sample_for_human_eval.py # Sampling for annotation
│   └── 05_analyze_and_plot.py      # Analysis & visualization
└── app/
    └── streamlit_app.py       # Interactive annotation UI
```

## Quickstart

### Installation & Execution
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline**:
   ```bash
   python src/01_prepare_data.py
   python src/02_generate_summaries.py
   python src/03_compute_metrics.py
   python src/04_sample_for_human_eval.py
   python src/05_analyze_and_plot.py
   ```

## Pipeline Overview

### Stage 1: Data Preparation (`01_prepare_data.py`)
- **Input**: XSum dataset (11,334 test articles)
- **Output**: 50 random articles with reference summaries
- **File**: `outputs/dataset_subset.jsonl`

### Stage 2: Summary Generation (`02_generate_summaries.py`)
- **Baseline**: Lead-1 sentence extraction
- **Neural Model**: facebook/bart-base (140M parameters)
- **Output**: `outputs/summaries.csv` (50 examples × 5 columns)

### Stage 3: Metrics Computation (`03_compute_metrics.py`)
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- **Output**: `outputs/metrics.csv` with detailed scores

### Stage 4: Human Evaluation Sampling (`04_sample_for_human_eval.py`)
- **Strategy**: Stratified random sampling
- **Output**: `outputs/human_eval_packet.csv` (annotation-ready)

### Stage 5: Analysis & Visualization (`05_analyze_and_plot.py`)
- **Analysis**: Metric correlations, disagreement cases, length effects
- **Outputs**: 3 publication-quality plots in `outputs/figures/`

## Files and Outputs

### Data Files
| File | Rows | Size | Description |
|------|------|------|---|
| `dataset_subset.jsonl` | 50 | 111 KB | Raw articles and reference summaries |
| `summaries.csv` | 50 | 167 KB | Baseline (Lead-k) and neural (BART) summaries |
| `metrics.csv` | 50 | 7.6 KB | ROUGE and BERTScore evaluation results |
| `human_eval_packet.csv` | 50 | 167 KB | Annotation-ready summaries with metrics |

### Analysis Outputs
| File | Description |
|------|---|
| `metric_human_correlations.csv` | Correlation matrix between all metrics |
| `disagreement_examples.csv` | Top cases of metric disagreement |
| `figures/rouge_vs_length.png` | ROUGE scores vs summary length (scatter) |
| `figures/rouge_vs_faithfulness.png` | ROUGE distribution analysis |
| `figures/rouge_diff_hist.png` | Metric differences histogram |

## Interactive Web Interface

### Streamlit Annotation App
```bash
streamlit run app/streamlit_app.py
```
**Features**:
- Load and display summaries with metrics
- Rate faithfulness on Likert scale (1-5)
- Save annotations to `outputs/human_annotations.csv`
- Explore metric distributions interactively
- Filter by metric ranges
