"""The feature contract shared by training, scoring and SHAP."""
import numpy as np
import pandas as pd
import pytest

from src.python.models.features import (
    FEATURE_DISPLAY,
    MODEL_FEATURES,
    FeatureContractError,
    build_feature_matrix,
    raw_to_model_features,
)
from src.python.models.train_xgboost import FORBIDDEN_FEATURES, assert_no_leakage_columns


def _frame(**over):
    base = {c: [1.0, 2.0] for c in MODEL_FEATURES}
    base.update(over)
    return pd.DataFrame(base)


def test_matrix_has_exact_columns_in_training_order():
    df = _frame()[MODEL_FEATURES[::-1]]  # scrambled column order
    X = build_feature_matrix(df)
    assert list(X.columns) == MODEL_FEATURES


def test_missing_feature_raises_instead_of_silently_zero_filling():
    df = _frame().drop(columns=["debt_ratio"])
    with pytest.raises(FeatureContractError, match="debt_ratio"):
        build_feature_matrix(df)


def test_nan_is_preserved_not_filled_with_zero():
    X = build_feature_matrix(_frame(monthly_income=[np.nan, 5000.0]))
    assert np.isnan(X.loc[0, "monthly_income"])


def test_no_forbidden_columns_in_model_features():
    assert not FORBIDDEN_FEATURES.intersection(MODEL_FEATURES)
    assert_no_leakage_columns(MODEL_FEATURES)
    with pytest.raises(AssertionError):
        assert_no_leakage_columns(MODEL_FEATURES + ["is_serious_delinquent"])


def test_no_synthetic_or_target_derived_columns_are_model_features():
    banned = ("occupation", "loan_purpose", "employment", "payment_velocity", "income_stability",
              "income_consistency", "credit_grade", "target_enc", "padded")
    for f in MODEL_FEATURES:
        assert not any(b in f for b in banned), f


def test_every_model_feature_has_display_keywords():
    assert set(MODEL_FEATURES) <= set(FEATURE_DISPLAY)


def test_raw_to_model_features_matches_batch_cleaning_rules():
    raw = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.5, 0.2],
            "age": [40, 30],
            "NumberOfTime30-59DaysPastDueNotWorse": [98, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [98, 0],
            "NumberOfTimes90DaysLate": [98, 2],
            "DebtRatio": [3500.0, 0.4],
            "MonthlyIncome": [np.nan, 4000.0],
            "NumberOfOpenCreditLinesAndLoans": [5, 3],
            "NumberRealEstateLoansOrLines": [1, 0],
            "NumberOfDependents": [np.nan, 2],
        }
    )
    out = raw_to_model_features(raw)
    assert out.loc[0, ["past_due_30_59", "past_due_60_89", "past_due_90"]].isna().all()  # sentinels -> NaN
    assert out.loc[0, "income_missing"] == 1 and out.loc[1, "income_missing"] == 0
    assert np.isnan(out.loc[0, "debt_ratio"])  # absolute amount, not a ratio, when income missing
    assert out.loc[1, "total_past_due"] == 3
    assert out.loc[0, "number_of_dependents"] == 0
    assert set(MODEL_FEATURES) <= set(out.columns)
