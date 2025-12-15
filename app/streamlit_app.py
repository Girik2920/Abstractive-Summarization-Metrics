import pandas as pd
import streamlit as st
from pathlib import Path

OUTPUT_DIR = Path("outputs")
SUMMARIES_PATH = OUTPUT_DIR / "summaries.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"
ANNOTATION_PATH = OUTPUT_DIR / "human_annotations.csv"


def load_data():
    summaries = pd.read_csv(SUMMARIES_PATH) if SUMMARIES_PATH.exists() else pd.DataFrame()
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    return summaries, metrics


def save_annotation(row_id: str, lead_score: int, model_score: int, note: str):
    ANNOTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(ANNOTATION_PATH) if ANNOTATION_PATH.exists() else pd.DataFrame()
    new_row = pd.DataFrame([
        {
            "id": row_id,
            "faithfulness_lead": lead_score,
            "faithfulness_model": model_score,
            "notes": note,
        }
    ])
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(ANNOTATION_PATH, index=False)


def main():
    st.title("When ROUGE Lies: Faithfulness Rater")
    summaries, metrics = load_data()

    if summaries.empty:
        st.warning("Run the data preparation and summarization scripts first.")
        return

    summaries = summaries.reset_index(drop=True)
    example_ids = summaries["id"].tolist()
    selected_id = st.selectbox("Choose example ID", example_ids)

    row = summaries[summaries["id"] == selected_id].iloc[0]
    st.subheader("Source article")
    st.write(row["source"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Lead summary**")
        st.write(row["lead_summary"])
    with col2:
        st.markdown("**Model summary**")
        st.write(row["model_summary"])

    if not metrics.empty:
        metric_row = metrics[metrics["id"] == selected_id]
        if not metric_row.empty:
            st.markdown("### Automatic scores")
            st.write(metric_row[[
                "rouge1_lead", "rouge1_model", "rougeL_lead", "rougeL_model", "bert_lead", "bert_model"
            ]])

    st.markdown("### Faithfulness ratings (1–5)")
    lead_score = st.slider("Lead faithfulness", 1, 5, 3)
    model_score = st.slider("Model faithfulness", 1, 5, 3)
    notes = st.text_area("Notes", "")

    if st.button("Save annotation"):
        save_annotation(selected_id, lead_score, model_score, notes)
        st.success("Annotation saved.")


if __name__ == "__main__":
    main()
