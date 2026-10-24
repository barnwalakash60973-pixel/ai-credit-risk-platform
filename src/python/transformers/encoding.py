"""
Encoding
- One-hot: loan_purpose, employment_type (synthetic, demo-only; not model features)
- Ordinal: credit_grade (A-E, heuristic risk proxy; scorecard only, not a model feature)
- Feature selection: drop high-correlation columns, never touching protected model features
- Save final result to the SQL database (PostgreSQL, or SQLite when no credentials) silver.applicants_silver

Removed: leave-one-out target encoding of `occupation_category`. That column was generated from a
hash of the member id (pure noise), and leave-one-out encoding leaks the row's own label:
inside a category, a row's encoded value is LOWER when its label is 1, so a tree model can read
the label back out. On this data it produced AUC > 0.94 (vs ~0.86 for the real features).
`target_encode_oof` below is the leak-safe replacement should a real categorical ever need it.
"""
import logging

import numpy as np
import pandas as pd
from src.python.db import backend_name, get_engine
from sklearn.model_selection import KFold

from src.python.config import SILVER_DIR, settings
from src.python.transformers.synthetic_enrichment import enrich_categorical_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CREDIT_GRADE_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# Never dropped by the correlation filter: raw model inputs, engineered model inputs, target, id.
PROTECTED_COLS = [
    "SeriousDlqin2yrs",
    "member_id_hash",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "total_past_due",
    "DebtRatio",
    "MonthlyIncome",
    "income_missing",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    # columns the dbt staging model reads (scorecard / demo): dropping one breaks `dbt run`
    "credit_grade",
    "credit_grade_ordinal",
    "payment_velocity_3m",
    "income_stability_score",
    "employment_years",
    "income_consistency",
]


def assign_credit_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Credit grade A-E based on a risk proxy (utilization + dti + delinquency), avoiding target leakage
    (does not use SeriousDlqin2yrs). Heuristic scorecard value; redundant with its inputs so it is
    not fed to the model."""
    df = df.copy()
    risk_score = (
        df["credit_utilization"].clip(0, 1) * 0.4
        + df["dti_ratio"].fillna(1.0).clip(0, 2) / 2 * 0.3
        + (df["delinquency_count_12m"].clip(0, 10) / 10) * 0.3
    )
    bins = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["A", "B", "C", "D", "E"]
    df["credit_grade"] = pd.cut(risk_score, bins=bins, labels=labels).astype(str)
    df["credit_grade_ordinal"] = df["credit_grade"].map(CREDIT_GRADE_ORDER)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=["loan_purpose", "employment_type"], prefix=["purpose", "emp"])
    return df


def target_encode_oof(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
) -> pd.Series:
    """Out-of-fold smoothed target encoding.

    Each row is encoded with statistics computed ONLY from the other folds, so the row's own
    label never influences its encoded value in a way correlated with that label. For scoring
    new data, fit the mapping on the full training set (not provided here: no real
    categorical currently needs it). Call this on TRAINING rows only."""
    encoded = pd.Series(np.nan, index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(df):
        fit, enc = df.iloc[fit_idx], df.iloc[enc_idx]
        gm = fit[target_col].mean()
        stats = fit.groupby(cat_col)[target_col].agg(["sum", "count"])
        smoothed = (stats["sum"] + smoothing * gm) / (stats["count"] + smoothing)
        encoded.iloc[enc_idx] = enc[cat_col].map(smoothed).fillna(gm).to_numpy()
    return encoded


def drop_high_correlation_features(
    df: pd.DataFrame, threshold: float = 0.9, protect: list[str] | None = None
) -> pd.DataFrame:
    """Drop one feature from any pair of numeric features with |corr| > threshold.

    Columns in `protect` (and the target) are never dropped, so a correlated pair of model inputs
    keeps both; the unprotected partner is dropped instead."""
    protect = set(protect or [])
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().abs()
    to_drop: list[str] = []
    cols = list(numeric_df.columns)
    for idx, col in enumerate(cols):
        if col in protect:
            continue
        # compare against columns that will survive: earlier kept ones, plus any protected one
        rivals = [c for j, c in enumerate(cols) if c != col and c not in to_drop and (j < idx or c in protect)]
        if any(corr.loc[col, r] > threshold for r in rivals):
            to_drop.append(col)
    if to_drop:
        logger.info(f"Dropping high-correlation columns (>{threshold}): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        logger.info("No columns found with correlation > threshold; nothing dropped.")
    return df


def save_to_postgres(df: pd.DataFrame, table_name: str = "applicants_silver", schema: str = "silver") -> None:
    engine = get_engine()
    df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False, chunksize=5000)
    logger.info(f"Saved to {backend_name()}: {schema}.{table_name} ({len(df)} rows)")


def run_encoding_pipeline() -> str:
    core_path = SILVER_DIR / "core_features.parquet"
    logger.info(f"Reading core features from {core_path}")
    df = pd.read_parquet(core_path, engine="pyarrow")

    logger.info("Enriching categorical fields (loan_purpose, employment_type; demo-only) ...")
    df = enrich_categorical_fields(df)

    logger.info("Assigning credit_grade (ordinal, heuristic scorecard value) ...")
    df = assign_credit_grade(df)

    logger.info("One-hot encoding loan_purpose & employment_type ...")
    df = one_hot_encode(df)

    logger.info("Dropping high-correlation features (model inputs protected) ...")
    df = drop_high_correlation_features(df, threshold=0.9, protect=PROTECTED_COLS)

    out_path = SILVER_DIR / "applicants_silver_final.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Final Silver dataset saved to {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    try:
        save_to_postgres(df)
    except Exception as e:
        logger.warning(f"Failed to save to SQL database ({e}), but local parquet file was saved successfully.")

    return str(out_path)


if __name__ == "__main__":
    run_encoding_pipeline()
