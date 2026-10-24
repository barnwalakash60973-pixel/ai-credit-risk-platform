"""
Feature Engineering (Core Features)
- dti_ratio, credit_utilization, payment_velocity_3m,
  delinquency_count_12m, income_stability_score
Input: intermediate cleaned parquet (from Block 1)
Output: core_features.parquet
"""
import logging
import numpy as np
import pandas as pd

from src.python.config import SILVER_DIR
from src.python.transformers.synthetic_enrichment import (
    enrich_employment_and_income_consistency,
    generate_synthetic_transactions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_cleaned_data() -> pd.DataFrame:
    path = SILVER_DIR / "intermediate"
    logger.info(f"Reading cleaned data from {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} rows")
    return df


def compute_dti_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """dti_ratio = monthly_debt / monthly_income.
    The dataset provides DebtRatio directly (= monthly_debt/monthly_income by Kaggle definition),
    so we derive monthly_debt explicitly first for transparency, then recompute the ratio.
    Missing income (or a DebtRatio nulled during cleaning) stays NaN rather than becoming 0."""
    df = df.copy()
    income = df["MonthlyIncome"]
    df["monthly_debt"] = (df["DebtRatio"] * income).round(2)
    ratio = np.where(income > 0, df["monthly_debt"] / income.where(income > 0), 0.0)
    df["dti_ratio"] = np.where(income.isna() | df["DebtRatio"].isna(), np.nan, ratio).round(4)
    return df


def compute_credit_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """credit_utilization = balance / credit_limit.
    RevolvingUtilizationOfUnsecuredLines from the dataset ALREADY matches this exact definition."""
    df = df.copy()
    df["credit_utilization"] = df["RevolvingUtilizationOfUnsecuredLines"].round(4)
    return df


def compute_payment_velocity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """payment_velocity_3m = average delinquency flag over the last 3 months (synthetic transactions)."""
    transactions_df = generate_synthetic_transactions(df, n_months=3)
    velocity = (
        transactions_df.groupby("member_id_hash")["delinquency_flag"]
        .mean()
        .rename("payment_velocity_3m")
        .round(4)
        .reset_index()
    )
    df = df.merge(velocity, on="member_id_hash", how="left")
    return df, transactions_df


PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def compute_total_past_due(df: pd.DataFrame) -> pd.DataFrame:
    """Total past-due events over the 2-year window (the model feature).
    NaN only when all three counts are unknown (sentinel rows cleaned upstream)."""
    df = df.copy()
    df["total_past_due"] = df[PAST_DUE_COLS].sum(axis=1, min_count=1)
    return df


def compute_delinquency_count_12m(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy delinquency_count_12m from the sum of 3 delinquency categories (dataset window = 2 years,
    we approximate half of it to represent the last 12 months). Used only for the heuristic
    credit grade / scorecard, not as a model feature."""
    df = df.copy()
    total_24m = df[PAST_DUE_COLS].fillna(0).sum(axis=1)
    df["delinquency_count_12m"] = (total_24m / 2).round().astype(int)
    return df


def run_feature_engineering() -> tuple[str, str]:
    df = load_cleaned_data()

    logger.info("Computing dti_ratio ...")
    df = compute_dti_ratio(df)

    logger.info("Computing credit_utilization ...")
    df = compute_credit_utilization(df)

    logger.info("Computing payment_velocity_3m (DEMO-ONLY synthetic transactions, not a model feature) ...")
    df, transactions_df = compute_payment_velocity(df)

    logger.info("Computing total_past_due & delinquency_count_12m ...")
    df = compute_total_past_due(df)
    df = compute_delinquency_count_12m(df)

    logger.info("Enriching employment_years, income_consistency, income_stability_score ...")
    df = enrich_employment_and_income_consistency(df)

    out_path = SILVER_DIR / "core_features.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Core features saved to {out_path} ({len(df)} rows, {len(df.columns)} cols)")

    tx_path = SILVER_DIR / "transactions.parquet"
    transactions_df.to_parquet(tx_path, index=False, engine="pyarrow")
    logger.info(f"Synthetic transactions saved to {tx_path} ({len(transactions_df)} rows)")

    return str(out_path), str(tx_path)


if __name__ == "__main__":
    run_feature_engineering()