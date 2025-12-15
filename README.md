# When ROUGE Lies: A Faithfulness-Centered Evaluation of Summarization Metrics

This project implements a small, reproducible pipeline to probe where automatic summarization metrics (especially ROUGE) fail to capture human notions of faithfulness. The workflow uses a modest subset of a summarization dataset to generate baseline and neural summaries, compute metrics (ROUGE and BERTScore), and prepare materials for lightweight human evaluation and follow-up analysis.

## Project structure
```
.
├── README.md
├── config.yaml
├── requirements.txt
├── data/
├── outputs/
│   ├── figures/
│   ├── dataset_subset.jsonl
│   ├── summaries.csv
│   ├── metrics.csv
│   ├── human_eval_packet.csv
│   └── human_annotations.csv
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_generate_summaries.py
│   ├── 03_compute_metrics.py
│   ├── 04_sample_for_human_eval.py
│   └── 05_analyze_and_plot.py
└── app/
    └── streamlit_app.py (optional bonus UI)
```

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Review and adjust `config.yaml` for dataset, subset size, model, and generation settings.
3. Run the pipeline end-to-end from the repo root:
   ```bash
   python src/01_prepare_data.py && \
   python src/02_generate_summaries.py && \
   python src/03_compute_metrics.py && \
   python src/04_sample_for_human_eval.py && \
   python src/05_analyze_and_plot.py
   ```

## Files and outputs
- `outputs/dataset_subset.jsonl`: Cached dataset slice (id, source, reference).
- `outputs/summaries.csv`: Baseline (Lead-k) and neural model summaries.
- `outputs/metrics.csv`: Per-example ROUGE/BERTScore results with length statistics.
- `outputs/human_eval_packet.csv`: Lightweight annotation sheet for ~50 sampled examples.
- `outputs/figures/`: Plots illustrating metric–human score relationships and disagreement cases.

## Optional Streamlit demo
If you want an interactive interface for annotation, a stub `app/streamlit_app.py` is provided. Run it with:
```bash
streamlit run app/streamlit_app.py
```
The UI loads generated summaries/metrics and saves annotations to `outputs/human_annotations.csv`.
