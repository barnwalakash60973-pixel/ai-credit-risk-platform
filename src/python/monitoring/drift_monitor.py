"""
Model Drift Monitoring (Population Stability Index / PSI)
Compare new feature distributions vs. training distribution (baseline) per feature.
PSI > 0.2 -> alert & trigger retrain flag.
"""
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PSI_ALERT_THRESHOLD = 0.2


def _psi_from_pcts(baseline_pct: np.ndarray, current_pct: np.ndarray) -> float:
    baseline_pct = np.where(baseline_pct == 0, 1e-4, baseline_pct)
    current_pct = np.where(current_pct == 0, 1e-4, current_pct)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
    baseline = training distribution, current = latest data distribution.

    Two fixes over the first version:
      * outer bin edges are +/- infinity, so current values outside the training range are counted
        (np.histogram used to silently drop them, hiding exactly the drift we want to see);
      * zero-inflated count features (past-due counts, dependents, flags) collapse to <4 distinct
        quantile edges, which used to give a single bin and PSI = 0 whatever the shift. Those are
        compared as categories 0,1,2,...,5+ instead.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.percentile(baseline, np.linspace(0, 100, bins + 1)))

    if len(edges) < 4:  # low-cardinality / zero-inflated: categorical comparison
        cap = 5
        cats = np.arange(0, cap + 1)
        b = np.clip(np.round(baseline), 0, cap)
        c = np.clip(np.round(current), 0, cap)
        baseline_pct = np.array([(b == k).mean() for k in cats])
        current_pct = np.array([(c == k).mean() for k in cats])
    else:
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        baseline_pct = np.histogram(baseline, bins=edges)[0] / len(baseline)
        current_pct = np.histogram(current, bins=edges)[0] / len(current)

    return round(_psi_from_pcts(baseline_pct, current_pct), 4)


def run_drift_check(baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    results = []
    for col in feature_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        baseline_vals = baseline_df[col].dropna().values
        current_vals = current_df[col].dropna().values
        if len(baseline_vals) < 10 or len(current_vals) < 10:
            continue

        psi = calculate_psi(baseline_vals, current_vals)
        results.append(
            {
                "feature": col,
                "psi": psi,
                "drift_alert": psi > PSI_ALERT_THRESHOLD,
            }
        )

    result_df = pd.DataFrame(results).sort_values("psi", ascending=False)
    n_alerts = result_df["drift_alert"].sum() if not result_df.empty else 0
    logger.info(f"Drift check completed: {n_alerts} features with PSI > {PSI_ALERT_THRESHOLD}")

    if n_alerts > 2:
        logger.warning(f"AUTO-RETRAIN TRIGGER: {n_alerts} features showing significant drift (>2 threshold)")

    return result_df


def run_drift_report(out_path: Path | None = None) -> pd.DataFrame:
    """PSI of each model feature: training split (baseline) vs the latest batch of new applications.

    If no new-applications file exists for today, falls back to the held-out test split and says
    so (weak signal: both splits come from the same population). The report is written to
    docs/psi_drift_report.csv (absolute path, independent of the working directory), which is
    what the dashboard's Model Performance page reads.
    """
    # Imported lazily so the pure PSI functions above stay importable without mlflow/duckdb.
    from src.python.config import BASE_DIR, RAW_DIR
    from src.python.models.features import MODEL_FEATURES, SPLIT_COL, raw_to_model_features
    from src.python.models.train_xgboost import load_gold_features

    out_path = out_path or (BASE_DIR / "docs" / "psi_drift_report.csv")
    df = load_gold_features()
    baseline_df = df[df[SPLIT_COL] == "train"]

    new_path = RAW_DIR / "new_applications" / f"new_applications_{date.today().isoformat()}.csv"
    if new_path.exists():
        logger.info(f"Comparing training split vs new applications: {new_path.name}")
        current_df = raw_to_model_features(pd.read_csv(new_path))
    else:
        logger.warning("No new-applications file for today; comparing train vs held-out test split (weak signal).")
        current_df = df[df[SPLIT_COL] == "test"]

    result_df = run_drift_check(baseline_df, current_df, MODEL_FEATURES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    logger.info(f"PSI report saved to {out_path}")
    return result_df


if __name__ == "__main__":
    print(run_drift_report().to_string(index=False))
