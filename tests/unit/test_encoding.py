import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.python.transformers.encoding import (
    CREDIT_GRADE_ORDER,
    PROTECTED_COLS,
    assign_credit_grade,
    drop_high_correlation_features,
    one_hot_encode,
    target_encode_oof,
)


def test_assign_credit_grade_produces_valid_labels():
    df = pd.DataFrame(
        {"credit_utilization": [0.1, 0.9], "dti_ratio": [0.2, 1.8], "delinquency_count_12m": [0, 9]}
    )
    result = assign_credit_grade(df)
    assert set(result["credit_grade"]).issubset(set(CREDIT_GRADE_ORDER.keys()))
    assert result["credit_grade_ordinal"].isnull().sum() == 0


def test_assign_credit_grade_handles_missing_dti():
    df = pd.DataFrame({"credit_utilization": [0.5], "dti_ratio": [np.nan], "delinquency_count_12m": [0]})
    assert assign_credit_grade(df)["credit_grade"].iloc[0] in CREDIT_GRADE_ORDER


def test_one_hot_encode_creates_dummy_columns():
    df = pd.DataFrame({"loan_purpose": ["auto", "education"], "employment_type": ["full_time", "contract"]})
    result = one_hot_encode(df)
    assert "purpose_auto" in result.columns
    assert "emp_full_time" in result.columns
    assert "loan_purpose" not in result.columns


def _noise_frame(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"cat": rng.integers(0, 7, n), "y": (rng.random(n) < 0.07).astype(int)})


def _naive_loo(df):
    """The encoding that was REMOVED from the pipeline, kept here as a leakage canary."""
    g = df.groupby("cat")["y"]
    return ((g.transform("sum") - df["y"]) + 10 * df["y"].mean()) / ((g.transform("count") - 1) + 10)


def _mean_within_category_auc(df, enc):
    """Average over categories of AUC(label vs -encoding) measured INSIDE each category."""
    aucs = [
        roc_auc_score(g["y"], -enc.loc[g.index]) for _, g in df.groupby("cat") if g["y"].nunique() == 2
    ]
    return float(np.mean(aucs))


def test_naive_loo_target_encoding_leaks_the_label():
    """Canary: inside every category, leave-one-out encoding is LOWER for positives than negatives,
    so it perfectly separates the label even though the category is pure noise. This is why
    occupation_category_target_enc was removed (it let the model reach AUC>0.9 on random data)."""
    df = _noise_frame()
    assert _mean_within_category_auc(df, _naive_loo(df)) > 0.99


def test_target_encode_oof_does_not_leak_on_noise():
    df = _noise_frame()
    enc = target_encode_oof(df, "cat", "y")
    auc = _mean_within_category_auc(df, enc)
    assert abs(auc - 0.5) < 0.03, f"OOF encoding should carry no label information, got AUC {auc:.3f}"


def test_drop_high_correlation_features():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [5, 1, 9, 2]})  # b = 2*a, perfect corr
    result = drop_high_correlation_features(df, threshold=0.9)
    assert "b" not in result.columns
    assert "a" in result.columns
    assert "c" in result.columns


def test_drop_high_correlation_never_drops_protected_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [5, 1, 9, 2]})
    result = drop_high_correlation_features(df, threshold=0.9, protect=["b"])
    assert "b" in result.columns  # protected, so its unprotected partner goes instead
    assert "a" not in result.columns


def test_correlation_filter_keeps_all_model_inputs_even_when_they_correlate():
    """Regression: the three past-due counts correlate >0.9 and used to be dropped automatically."""
    rng = np.random.default_rng(1)
    base = rng.poisson(0.3, 500)
    df = pd.DataFrame(
        {
            "SeriousDlqin2yrs": base > 0,  # even a 'target' that correlates must survive
            "NumberOfTime30-59DaysPastDueNotWorse": base,
            "NumberOfTime60-89DaysPastDueNotWorse": base,
            "NumberOfTimes90DaysLate": base,
            "dup_of_them": base,
        }
    ).astype({"SeriousDlqin2yrs": int})
    out = drop_high_correlation_features(df, 0.9, protect=PROTECTED_COLS)
    for c in ["SeriousDlqin2yrs", "NumberOfTime30-59DaysPastDueNotWorse",
              "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]:
        assert c in out.columns
    assert "dup_of_them" not in out.columns
