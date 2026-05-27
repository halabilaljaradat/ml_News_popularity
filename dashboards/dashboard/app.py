"""Starter Streamlit dashboard for the DS230 final project.

Run from repository root:
    streamlit run dashboard/app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "classification_results.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "selected_features.csv"
ROBUSTNESS_PATH = PROJECT_ROOT / "data" / "processed" / "robustness_results.csv"

st.set_page_config(page_title="DS230 Online News Popularity", layout="wide")
st.title("DS230 Final Project — Online News Popularity")
st.caption("Starter dashboard for model comparison, selected features, and robustness results.")

if not RESULTS_PATH.exists():
    st.warning(
        "No generated results found yet. Run `python scripts/run_task_a_classification.py` first."
    )
    st.stop()

results_df = pd.read_csv(RESULTS_PATH)

st.header("Classification Model Comparison")
st.dataframe(results_df, use_container_width=True)

metric = st.selectbox(
    "Choose metric to compare",
    [col for col in results_df.columns if col not in ["model"]],
    index=1 if "test_acc" in results_df.columns else 0,
)
fig = px.bar(results_df.sort_values(metric), x=metric, y="model", orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.header("Selected Features")
if FEATURES_PATH.exists():
    features_df = pd.read_csv(FEATURES_PATH)
    st.write(f"Number of selected features: {len(features_df)}")
    st.dataframe(features_df, use_container_width=True)
else:
    st.info("Selected features file was not found.")

st.header("Robustness Test")
if ROBUSTNESS_PATH.exists():
    robustness_df = pd.read_csv(ROBUSTNESS_PATH)
    st.dataframe(robustness_df, use_container_width=True)
    fig2 = px.line(
        robustness_df,
        x="fraction_of_training_data",
        y="test_accuracy",
        markers=True,
        title="XGBoost accuracy vs. training-data fraction",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Robustness results were not found.")

st.header("Prediction Testing")
st.info(
    "Prediction testing can be added after saving the full preprocessing pipeline and final model. "
    "For now, this dashboard focuses on model comparison and generated results."
)
