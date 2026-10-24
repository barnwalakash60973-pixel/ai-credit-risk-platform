"""
Single source of truth for the model's feature contract.

Training, scoring and SHAP explanation ALL import from here, so the three can never
disagree about which columns the model sees or in which order (previously scoring used a
hardcoded list and SHAP padded the matrix with zero columns named `padded_feature_N`).

Synthetic columns (see ADR-006) are deliberately NOT model features: they are derived from a
hash of the member id, so they carry no real signal, and any apparent signal would be an
artefact (e.g. leakage through a target encoding).
"""
from __future__ import annotations

import pandas as pd

TARGET_COL = "is_serious_delinquent"
ID_COL = "member_id_hash"
SPLIT_COL = "data_split"  # 'train' | 'test', assigned deterministically in dbt

# Real, observable credit-bureau features (Give Me Some Credit) + labelled-free engineered ones.
MODEL_FEATURES: list[str] = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "past_due_60_89",
    "past_due_90",
    "total_past_due",
    "debt_ratio",
    "monthly_income",
    "income_missing",
    "open_credit_lines",
    "real_estate_loans",
    "number_of_dependents",
]

# Human-readable names used in decision letters / regulation lookup.
FEATURE_DISPLAY: dict[str, list[str]] = {
    "revolving_utilization": ["utilization", "credit usage", "revolving credit"],
    "age": ["age", "years old"],
    "past_due_30_59": ["30-59", "late payment", "past due", "delinquen"],
    "past_due_60_89": ["60-89", "late payment", "past due", "delinquen"],
    "past_due_90": ["90 days", "late payment", "past due", "delinquen"],
    "total_past_due": ["late payment", "past due", "delinquen", "payment history"],
    "debt_ratio": ["debt ratio", "debt to income", "debt-to-income", "dti"],
    "monthly_income": ["monthly income", "income", "earning"],
    "income_missing": ["income not", "income was not", "unreported income", "income"],
    "open_credit_lines": ["open credit", "credit lines", "open loans", "open accounts"],
    "real_estate_loans": ["real estate", "mortgage", "home loan"],
    "number_of_dependents": ["dependents", "family members", "children"],
}


class FeatureContractError(ValueError):
    """Raised when a dataframe does not satisfy the model's feature contract."""


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Return df[feature_cols] as float, in the exact training order.

    Raises FeatureContractError on missing columns instead of silently zero-filling them.
    NaNs are left in place: XGBoost handles missing values natively, and filling with 0
    (the old behaviour) would make "unknown income" look like "zero income".
    """
    feature_cols = feature_cols or MODEL_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise FeatureContractError(f"Missing required model features: {missing}")
    return df[feature_cols].astype(float)


# --- raw (Kaggle-schema) -> model-feature mapping, used for NEW applications -----------------

RAW_TO_MODEL = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfTimes90DaysLate": "past_due_90",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfDependents": "number_of_dependents",
}
_PAST_DUE = ["past_due_30_59", "past_due_60_89", "past_due_90"]


def raw_to_model_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules the batch (Spark) pipeline applies, to a raw Kaggle-schema frame.

    Keeps online/new-application scoring consistent with training: 96/98 sentinels -> NaN,
    income_missing flag, DebtRatio -> NaN when income is missing, total_past_due. Winsorisation
    caps are not re-applied: tree models split on thresholds well inside the capped range, so
    values beyond the 99th percentile fall on the same side of every split either way.
    """
    df = raw.rename(columns=RAW_TO_MODEL).copy()
    for c in _PAST_DUE:
        df[c] = df[c].where(~df[c].isin([96, 98]))
    df["income_missing"] = df["monthly_income"].isna().astype(int)
    df["debt_ratio"] = df["debt_ratio"].where(df["monthly_income"].notna())
    df["total_past_due"] = df[_PAST_DUE].sum(axis=1, min_count=1)
    df["number_of_dependents"] = df["number_of_dependents"].fillna(0)
    return df
