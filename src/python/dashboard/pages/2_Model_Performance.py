"""
Page 2 - Model Performance
AUC-ROC / KS / Gini, confusion matrix, hyperparameters, SHAP importance, PSI drift
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.python.dashboard import theme

# Read MLflow from the SQLite DB at the project root (forward slashes: safe on Windows)
mlflow.set_tracking_uri(f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT_NAME = "credit_risk_scoring"

theme.setup("Model Performance", "📈")
theme.page_header("📈", "Model Performance", "Discrimination metrics, explainability and drift on the champion model")


@st.cache_data(ttl=120)
def load_latest_run_metrics():
    """The run behind the current champion model (falls back to the newest run if none promoted)."""
    client = mlflow.tracking.MlflowClient()
    try:
        champ = client.get_model_version_by_alias("credit_risk_xgboost", "champion")
        return client.get_run(champ.run_id)
    except Exception:
        pass
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    return runs[0] if runs else None


run = load_latest_run_metrics()
if run is None:
    theme.empty_state("No MLflow run yet", "Run `python -m src.python.models.train_xgboost` first.", "🧪")
    st.stop()

metrics, params = run.data.metrics, run.data.params
auc_val = metrics.get("holdout_auc", metrics.get("test_auc", 0.0))
ks_val = metrics.get("holdout_ks", metrics.get("test_ks", 0.0))
gini_val = metrics.get("holdout_gini", metrics.get("test_gini", 0.0))
cv_mean, cv_std = metrics.get("cv_auc_mean", 0.0), metrics.get("cv_auc_std", 0.0)


def _accent(value: float, target: float) -> str:
    return theme.RISK_COLORS["LOW"] if value >= target else theme.RISK_COLORS["HIGH"]


theme.kpi_row(
    [
        {"label": "AUC-ROC", "value": f"{auc_val:.4f}", "hint": "target > 0.85", "accent": _accent(auc_val, 0.85)},
        {"label": "KS statistic", "value": f"{ks_val:.4f}", "hint": "target > 0.45", "accent": _accent(ks_val, 0.45)},
        {"label": "Gini", "value": f"{gini_val:.4f}", "hint": "target > 0.70", "accent": _accent(gini_val, 0.70)},
        (
            {"label": "CV AUC (mean ± std)", "value": f"{cv_mean:.4f}", "hint": f"± {cv_std:.4f}", "accent": theme.INDIGO}
            if cv_mean > 0
            else {"label": "Run ID", "value": run.info.run_id[:8], "hint": "MLflow", "accent": theme.INDIGO}
        ),
    ]
)

theme.section("Holdout evaluation", "Confusion matrix and the tuned hyperparameters")
col_left, col_right = st.columns(2)

with col_left:
    try:
        artifact_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="confusion_matrix.json")
        with open(artifact_path) as f:
            cm_data = json.load(f)
        fig = go.Figure(
            go.Heatmap(
                z=cm_data["confusion_matrix"],
                x=[f"Predicted: {l}" for l in cm_data["labels"]],
                y=[f"Actual: {l}" for l in cm_data["labels"]],
                text=cm_data["confusion_matrix"], texttemplate="%{text}",
                colorscale=[[0, "#EEF2FF"], [1, theme.INDIGO]], showscale=False,
            )
        )
        fig.update_layout(title="Confusion matrix (holdout)")
        st.plotly_chart(theme.style_fig(fig, legend=False), width="stretch")
    except Exception:
        theme.empty_state("Confusion matrix not logged", "The artifact is missing from this MLflow run.", "🧩")

with col_right:
    if params:
        param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
        st.markdown("**Best hyperparameters (Optuna)**")
        st.dataframe(param_df, width="stretch", hide_index=True, height=340)
    else:
        theme.empty_state("No hyperparameters recorded", "Tuning has not been logged yet.", "🎛️")

theme.section("Global feature importance", "Mean absolute SHAP value across the scored population")
shap_path = ROOT_DIR / "docs" / "shap_global_importance.csv"
if shap_path.exists():
    imp = pd.read_csv(shap_path).head(15).sort_values("mean_abs_shap")
    fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                 labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""})
    fig.update_traces(marker_color=theme.VIOLET, marker_cornerradius=5)
    st.plotly_chart(theme.style_fig(fig, height=460, legend=False), width="stretch")
else:
    theme.empty_state("No SHAP importance yet", "Run `python -m src.python.explainability.shap_explainer`.", "🧠")

theme.section("Drift monitoring", "Population Stability Index per feature (alert above 0.2)")
psi_path = ROOT_DIR / "docs" / "psi_drift_report.csv"
if psi_path.exists():
    psi_df = pd.read_csv(psi_path)
    psi_df["status"] = psi_df["drift_alert"].map({True: "Drift", False: "Stable"})
    fig = px.bar(psi_df, x="feature", y="psi", color="status",
                 color_discrete_map={"Drift": theme.RISK_COLORS["VERY_HIGH"], "Stable": theme.RISK_COLORS["LOW"]},
                 labels={"psi": "PSI", "feature": ""})
    fig.add_hline(y=0.2, line_dash="dash", line_color=theme.RISK_COLORS["VERY_HIGH"],
                  annotation_text="Alert threshold 0.2", annotation_position="top left")
    st.plotly_chart(theme.style_fig(fig, height=400), width="stretch")
else:
    theme.empty_state("No PSI report yet", "Run the drift_monitor DAG or `python -m src.python.monitoring.drift_monitor`.", "📡")
