"""
Entry point for the Streamlit multi-page dashboard.
Run with: streamlit run src/python/dashboard/app.py
Other pages appear automatically in the sidebar from the pages/ folder.
"""
import sys
from pathlib import Path

# Make the project root importable (credit-risk-scoring-platform/)
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.python.dashboard import theme

theme.setup("Home", "🏦")

theme.hero(
    "Explainable credit decisions, grounded in RBI regulation",
    "An end-to-end Medallion pipeline (Bronze → Silver → Gold → AI) that scores loan applicants with "
    "XGBoost, explains every decision with SHAP, and writes the decision letter with a local LLM that "
    "cites the official Reserve Bank of India directions.",
    eyebrow="AI Credit Risk Scoring Platform",
    chips=["XGBoost", "SHAP", "RAG · ChromaDB", "Ollama", "FastAPI", "Airflow", "PDPA-ready"],
)

theme.kpi_row(
    [
        {"label": "Architecture", "value": "Medallion", "hint": "Bronze · Silver · Gold", "accent": theme.INDIGO},
        {"label": "Model", "value": "XGBoost", "hint": "Optuna-tuned, MLflow-tracked", "accent": theme.TEAL},
        {"label": "Explainability", "value": "SHAP", "hint": "Top-5 factors per applicant", "accent": "#F59E0B"},
        {"label": "Regulation source", "value": "RBI PDF", "hint": "Original directions, cited by page", "accent": theme.VIOLET},
    ]
)

theme.section("Explore the platform", "Use the sidebar or jump straight in")
c1, c2, c3 = st.columns(3)
with c1:
    theme.feature_card("📊", "Overview", "Application volume, approval rate and the risk-tier mix at a glance.")
with c2:
    theme.feature_card("📈", "Model Performance", "AUC-ROC, KS, Gini, SHAP importance and PSI drift monitoring.")
with c3:
    theme.feature_card("🔍", "Decision Viewer", "Pick an applicant, see the SHAP waterfall and generate the decision letter.")
st.write("")
cn1, cn2, cn3 = st.columns(3)
with cn1:
    theme.feature_card("✍️", "Check Application", "Type in the features yourself (no ID needed) and get the decision, reasons and loan-advisor chat.")
st.write("")
c4, c5, c6 = st.columns(3)
with c4:
    theme.feature_card("📚", "Regulation Search", "Ask the RAG knowledge base a question and read the exact RBI paragraph, with page.")
with c5:
    theme.feature_card("🔒", "PDPA Compliance", "Data lineage, erasure-request queue and 72-hour SLA tracking.")
with c6:
    theme.feature_card("🧭", "How it flows", "Raw data → features → model → SHAP → RAG over RBI PDF → letter.")
