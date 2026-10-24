"""
XGBoost training: Optuna search (CV on the training split only), honest held-out evaluation,
MLflow logging and registry promotion.

Guarantees (each covered by a test or an assertion):
  * The model sees ONLY features in features.MODEL_FEATURES, never the target/id/split columns.
  * Train/test membership is the deterministic `data_split` assigned in dbt, so the held-out set
    is identical across runs and shared with scoring/reporting.
  * Probabilities are on the natural scale (no class re-weighting); calibration is measured and logged.
  * The feature list is stored with the model and a new model only becomes champion if it
    beats the current champion on held-out AUC.
"""
import argparse
import logging

import duckdb
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

from src.python.config import DUCKDB_PATH, MLFLOW_TRACKING_URI
from src.python.models.features import (
    ID_COL,
    MODEL_FEATURES,
    SPLIT_COL,
    TARGET_COL,
    build_feature_matrix,
)
from src.python.models.hyperparameter_tuning import run_optuna_search
from src.python.scoring.decision_rules import MEDIUM_MAX
from src.python.models.model_registry import MODEL_NAME, FEATURES_ARTIFACT, promote_if_better

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = "credit_risk_scoring"
FORBIDDEN_FEATURES = {TARGET_COL, ID_COL, SPLIT_COL, "SeriousDlqin2yrs", "feature_store_updated_at"}


def load_gold_features() -> pd.DataFrame:
    """Load the gold feature store from DuckDB."""
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB file not found: {DUCKDB_PATH}. Run `cd dbt && dbt run` first.")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM applicant_feature_store").df()
    finally:
        con.close()
    logger.info(f"Loaded Gold Feature Store: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def assert_no_leakage_columns(feature_cols: list[str]) -> None:
    bad = FORBIDDEN_FEATURES.intersection(feature_cols)
    if bad:
        raise AssertionError(f"Leakage guard: forbidden columns in model features: {sorted(bad)}")


def prepare_train_data(df: pd.DataFrame, feature_cols: list[str] | None = None):
    """Split by the dbt-assigned `data_split`. Returns (train_df, test_df, feature_cols)."""
    feature_cols = feature_cols or MODEL_FEATURES
    assert_no_leakage_columns(feature_cols)
    train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    test_df = df[df[SPLIT_COL] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("data_split produced an empty train or test set")
    return train_df, test_df, feature_cols


def calculate_ks_stat(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    pos = y_pred_proba[y_true == 1]
    neg = y_pred_proba[y_true == 0]
    ks_stat, _ = ks_2samp(pos, neg)
    return float(ks_stat)


def calculate_gini(auc: float) -> float:
    return 2 * auc - 1


def calibration_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per probability decile (quantile bins)."""
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
    return out


def _version_for_run(run_id: str) -> str:
    """Registered-model version created by this run (works across MLflow 2.x/3.x)."""
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    mine = [v for v in versions if v.run_id == run_id]
    if not mine:
        raise RuntimeError(f"No registered version found for run {run_id}")
    return str(max(int(v.version) for v in mine))


def train_and_evaluate(n_trials: int = 30, cv_folds: int = 5, seed: int = 42) -> dict:
    df = load_gold_features()
    train_df, test_df, feature_cols = prepare_train_data(df)

    X_train = build_feature_matrix(train_df, feature_cols).values
    y_train = train_df[TARGET_COL].astype(int).values
    X_test = build_feature_matrix(test_df, feature_cols).values
    y_test = test_df[TARGET_COL].astype(int).values
    logger.info(
        f"Train: {len(y_train)} rows ({y_train.mean():.2%} positive) | "
        f"Test: {len(y_test)} rows ({y_test.mean():.2%} positive) | {len(feature_cols)} features"
    )

    best_params, best_cv_auc, best_cv_std = run_optuna_search(X_train, y_train, n_trials=n_trials, seed=seed, n_splits=cv_folds)

    final_model = xgb.XGBClassifier(
        **best_params, objective="binary:logistic", eval_metric="auc", random_state=seed, n_jobs=-1
    )
    final_model.fit(X_train, y_train, verbose=False)

    proba = final_model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, proba))
    test_ks = calculate_ks_stat(y_test, proba)
    test_gini = calculate_gini(test_auc)
    brier = float(brier_score_loss(y_test, proba))
    calib = calibration_table(y_test, proba)
    logger.info(
        f"Held-out test | AUC {test_auc:.4f} | KS {test_ks:.4f} | Gini {test_gini:.4f} | "
        f"Brier {brier:.4f} | mean pred {proba.mean():.4f} vs observed {y_test.mean():.4f}"
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_params({"n_trials": n_trials, "cv_folds": cv_folds, "n_features": len(feature_cols)})
        mlflow.log_metrics(
            {
                "cv_auc": best_cv_auc,
                "cv_auc_mean": best_cv_auc,
                "cv_auc_std": best_cv_std,
                "test_auc": test_auc,
                "test_ks": test_ks,
                "test_gini": test_gini,
                "test_brier": brier,
                "test_mean_pred": float(proba.mean()),
                "test_base_rate": float(y_test.mean()),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
        )
        mlflow.log_dict(feature_cols, FEATURES_ARTIFACT)
        mlflow.log_table(calib, "calibration_by_decile.json")
        # Confusion matrix at the reject threshold used by the decision policy (HIGH and above).
        cm = confusion_matrix(y_test, (proba >= MEDIUM_MAX).astype(int), labels=[0, 1])
        mlflow.log_dict(
            {"threshold": MEDIUM_MAX, "labels": ["no_default", "default"], "confusion_matrix": cm.tolist()},
            "confusion_matrix.json",
        )

        input_example = pd.DataFrame(X_train[:5], columns=feature_cols)
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
        )
        version = _version_for_run(run.info.run_id)
        logger.info(f"Model logged and registered as '{MODEL_NAME}' v{version} (run {run.info.run_id})")

    promoted = promote_if_better(version, run.info.run_id)
    return {
        "run_id": run.info.run_id,
        "version": version,
        "promoted": promoted,
        "cv_auc": best_cv_auc,
        "test_auc": test_auc,
        "test_ks": test_ks,
        "test_gini": test_gini,
        "test_brier": brier,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()
    print(train_and_evaluate(n_trials=args.trials, cv_folds=args.cv_folds))
