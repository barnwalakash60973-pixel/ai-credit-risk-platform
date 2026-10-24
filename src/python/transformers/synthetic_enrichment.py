"""
Deterministic synthetic enrichment (seeded from member_id_hash) for columns
not available in the public Give Me Some Credit dataset, but commonly found
in production lending systems: loan_purpose, employment_type, employment_years,
income_consistency, and monthly payment history (for payment_velocity).

See ADR-006 for justification.
"""
import hashlib

import numpy as np
import pandas as pd

LOAN_PURPOSES = ["debt_consolidation", "home_improvement", "auto", "education", "personal", "business"]
EMPLOYMENT_TYPES = ["full_time", "part_time", "self_employed", "contract", "unemployed"]


def _seed_from_hash(hash_str: str) -> int:
    """Convert part of a hex hash into an integer seed, deterministic per applicant."""
    return int(hashlib.sha256(hash_str.encode()).hexdigest()[:8], 16)


def enrich_categorical_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add loan_purpose & employment_type, deterministic per member_id_hash."""
    df = df.copy()

    def pick(hash_str: str, options: list[str], salt: str) -> str:
        seed = _seed_from_hash(hash_str + salt)
        return options[seed % len(options)]

    df["loan_purpose"] = df["member_id_hash"].apply(lambda h: pick(h, LOAN_PURPOSES, "purpose"))
    df["employment_type"] = df["member_id_hash"].apply(lambda h: pick(h, EMPLOYMENT_TYPES, "employment"))
    return df


def enrich_employment_and_income_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    employment_years: proxy derived from age (tenure cannot exceed realistic working age: age-18),
    with deterministic variation per applicant.
    income_consistency: proxy 0-1, applicants with high MonthlyIncome & low DebtRatio
    are assumed to be more stable (simple heuristic, documented as a proxy, not original data).
    """
    df = df.copy()
    rng_seeds = df["member_id_hash"].apply(_seed_from_hash)

    max_tenure = (df["age"] - 18).clip(lower=0)
    tenure_fraction = (rng_seeds % 1000) / 1000.0  # 0.0 - 0.999, deterministic
    df["employment_years"] = (max_tenure * tenure_fraction).round(1)

    # income_consistency: 0-1 heuristic based on relative stability (low DebtRatio = more stable)
    debt_ratio_clipped = df["DebtRatio"].fillna(1.0).clip(0, 2)
    df["income_consistency"] = (1 - (debt_ratio_clipped / 2)).clip(0.1, 1.0).round(3)

    df["income_stability_score"] = (df["employment_years"] * df["income_consistency"]).round(2)
    return df


def generate_synthetic_transactions(df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    """
    Generate synthetic monthly payment snapshot per applicant (stg_transactions source).
    The delinquency_flag for that month is probabilistically influenced by actual delinquency history
    (NumberOfTime30-59/60-89/90DaysLate) to remain statistically consistent with the target.

    DEMO-ONLY data (ADR-006): derived from the same columns the model already sees, so it adds
    no information and is not used as a model feature. Missing counts (sentinels cleaned to NaN)
    are treated as 0 here.
    """
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    hashes = df["member_id_hash"].to_numpy()
    total_delinquency = df[past_due_cols].fillna(0).sum(axis=1).to_numpy()
    incomes = df["MonthlyIncome"].fillna(0).to_numpy() if "MonthlyIncome" in df else np.zeros(len(df))

    records = []
    for h, total, income in zip(hashes, total_delinquency, incomes):
        rng = np.random.default_rng(_seed_from_hash(h))
        # Monthly delinquent probability, calibrated from a 24-month history (dataset window)
        p_delinquent = min(total / 24.0, 0.9)
        for m in range(n_months):
            income_noise = rng.normal(1.0, 0.05)
            records.append(
                (h, m + 1, round(float(income) * max(income_noise, 0), 2), int(rng.random() < p_delinquent))
            )
    return pd.DataFrame(
        records, columns=["member_id_hash", "month_offset", "snapshot_income", "delinquency_flag"]
    )
