import numpy as np
import pandas as pd

from src.python.transformers.feature_engineering import (
    compute_credit_utilization,
    compute_delinquency_count_12m,
    compute_dti_ratio,
    compute_total_past_due,
)

PD = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def test_compute_dti_ratio():
    df = pd.DataFrame({"DebtRatio": [0.3, 0.5], "MonthlyIncome": [5000.0, 0.0]})
    result = compute_dti_ratio(df)
    assert result.loc[0, "dti_ratio"] == 0.3
    assert result.loc[1, "dti_ratio"] == 0  # zero income: avoid div by zero


def test_compute_dti_ratio_missing_income_stays_missing_not_zero():
    df = pd.DataFrame({"DebtRatio": [np.nan], "MonthlyIncome": [np.nan]})
    assert np.isnan(compute_dti_ratio(df).loc[0, "dti_ratio"])


def test_compute_credit_utilization():
    df = pd.DataFrame({"RevolvingUtilizationOfUnsecuredLines": [0.234567]})
    assert compute_credit_utilization(df).loc[0, "credit_utilization"] == 0.2346


def test_compute_delinquency_count_12m():
    df = pd.DataFrame({PD[0]: [2], PD[1]: [1], PD[2]: [1]})
    assert compute_delinquency_count_12m(df).loc[0, "delinquency_count_12m"] == 2  # (2+1+1)/2


def test_delinquency_count_survives_nan_counts():
    """Sentinel rows are NaN after cleaning; astype(int) used to crash on them."""
    df = pd.DataFrame({PD[0]: [np.nan], PD[1]: [np.nan], PD[2]: [np.nan]})
    assert compute_delinquency_count_12m(df).loc[0, "delinquency_count_12m"] == 0


def test_total_past_due_sums_and_keeps_unknown_as_nan():
    df = pd.DataFrame({PD[0]: [2, np.nan], PD[1]: [1, np.nan], PD[2]: [1, np.nan]})
    out = compute_total_past_due(df)
    assert out.loc[0, "total_past_due"] == 4
    assert np.isnan(out.loc[1, "total_past_due"])
