"""
SHAP Explainability
- shap.TreeExplainer on the champion model
- Top-5 SHAP features per applicant saved to prediction_explanations (DuckDB)
- Global feature importance summary

The feature matrix is built with the SAME contract the model was trained with
(features.build_feature_matrix + the feature list stored with the model), so every SHAP value
is attributed to the real feature name. The previous version built the matrix from "whatever
numeric columns exist" and padded it with zero columns named `padded_feature_N`.
"""
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shap

from src.python.config import DUCKDB_PATH
from src.python.models.features import ID_COL, SPLIT_COL, build_feature_matrix
from src.python.models.model_registry import load_champion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_gold_features() -> pd.DataFrame:
    """Fetch the gold layer feature store directly from DuckDB."""
    logger.info(f"Reading applicant_feature_store from DuckDB: {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM applicant_feature_store").fetchdf()
    finally:
        con.close()
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values using TreeExplainer (log-odds space)."""
    logger.info(f"Computing SHAP TreeExplainer for {len(X)} rows...")
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X.values)


def top_n_shap_features(shap_row: np.ndarray, feature_names: list[str], n: int = 5) -> list[dict]:
    """Extract top-N features with the largest absolute SHAP value for a single applicant."""
    idx_sorted = np.argsort(-np.abs(shap_row))[:n]
    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_row[i]), 4),
            "direction": "increases_risk" if shap_row[i] > 0 else "decreases_risk",
        }
        for i in idx_sorted
    ]


def build_prediction_explanations(
    df: pd.DataFrame, shap_values: np.ndarray, feature_names: list[str], proba: np.ndarray
) -> pd.DataFrame:
    """Construct structured DataFrame containing top feature explanations per applicant."""
    ids = df[ID_COL].to_numpy()
    splits = df[SPLIT_COL].to_numpy() if SPLIT_COL in df.columns else np.full(len(df), "unknown")
    records = [
        {
            ID_COL: ids[i],
            "predicted_proba": round(float(proba[i]), 4),
            "top_shap_features_json": json.dumps(top_n_shap_features(shap_values[i], feature_names, n=5)),
            "data_split": splits[i],
        }
        for i in range(len(df))
    ]
    return pd.DataFrame(records)


def save_global_importance(shap_values: np.ndarray, feature_names: list[str], out_path: Path) -> pd.DataFrame:
    """Calculate and export global mean absolute SHAP feature importance."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    importance_df.to_csv(out_path, index=False)
    logger.info(f"Global SHAP importance saved to {out_path}")
    logger.info(f"\nTop 10 features:\n{importance_df.head(10).to_string(index=False)}")
    return importance_df


def save_to_duckdb(explanations_df: pd.DataFrame) -> None:
    """Write prediction explanations back to DuckDB layer."""
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.register("explanations_tmp", explanations_df)
        con.execute("CREATE OR REPLACE TABLE main.prediction_explanations AS SELECT * FROM explanations_tmp")
    finally:
        con.close()
    logger.info(f"prediction_explanations saved to DuckDB ({len(explanations_df)} rows)")


def run_shap_pipeline(sample_size: int | None = None) -> pd.DataFrame:
    """Execute end-to-end SHAP explanation pipeline."""
    df = load_gold_features()
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    model, feature_names = load_champion()
    X = build_feature_matrix(df, feature_names)
    logger.info(f"Feature matrix X prepared with shape: {X.shape}")

    proba = model.predict_proba(X.values)[:, 1]
    shap_values = compute_shap_values(model, X)

    explanations_df = build_prediction_explanations(df, shap_values, feature_names, proba)
    save_to_duckdb(explanations_df)

    save_global_importance(shap_values, feature_names, Path("docs") / "shap_global_importance.csv")
    return explanations_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="explain only N random applicants (much faster on small machines; default: all)")
    run_shap_pipeline(sample_size=ap.parse_args().sample)
