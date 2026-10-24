"""
Score ONE application from typed-in feature values (what the dashboard form and POST /predict use).

It goes through the SAME code path as batch scoring so a form entry can never disagree with the
batch result for the same person:

    typed values -> features.raw_to_model_features()  (income_missing, DebtRatio NaN when income is
                    missing, total_past_due, ...)  ->  champion model  ->  risk tier / decision
                 -> SHAP TreeExplainer (reasons)

Inputs (9 values the applicant knows; the other 3 model features are derived):

    revolving_utilization, age, past_due_30_59, past_due_60_89, past_due_90, debt_ratio,
    monthly_income (None = not reported), open_credit_lines, real_estate_loans, number_of_dependents
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.python.models.features import MODEL_FEATURES, build_feature_matrix, raw_to_model_features
from src.python.scoring.decision_rules import decision_from_tier, risk_tier_from_proba

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- input definitions
# Shared by the dashboard form and the API so both show / enforce the same limits.
# `min`/`max` are HARD limits (sanity checks), not the model's training range.
INPUT_FIELDS: list[dict] = [
    dict(key="revolving_utilization", label="Credit utilization", kind="float", min=0.0, max=5.0, default=0.15, step=0.01,
         help="Share of your revolving credit limits (credit cards, overdrafts) you are using. 0.30 = 30%, 1.00 = fully used. "
              "Training data was capped at about 1.09."),
    dict(key="age", label="Age (years)", kind="int", min=18, max=110, default=45, step=1, help="Applicant age."),
    dict(key="past_due_30_59", label="Times 30-59 days late", kind="int", min=0, max=30, default=0, step=1,
         help="Number of times a payment was 30-59 days overdue in the last 2 years."),
    dict(key="past_due_60_89", label="Times 60-89 days late", kind="int", min=0, max=30, default=0, step=1,
         help="Number of times a payment was 60-89 days overdue in the last 2 years."),
    dict(key="past_due_90", label="Times 90+ days late", kind="int", min=0, max=30, default=0, step=1,
         help="Number of times a payment was 90 or more days overdue in the last 2 years."),
    dict(key="debt_ratio", label="Debt ratio", kind="float", min=0.0, max=10000.0, default=0.30, step=0.01,
         help="Monthly debt payments (loans, rent/mortgage, alimony...) divided by monthly income. 0.30 = 30% of income."),
    dict(key="monthly_income", label="Monthly income", kind="float", min=0.0, max=10_000_000.0, default=5400.0, step=100.0,
         help="Gross monthly income, in the units of the training data (Give Me Some Credit, US$). "
              "Tick 'income not reported' if unknown - the model handles that case explicitly."),
    dict(key="open_credit_lines", label="Open credit lines & loans", kind="int", min=0, max=100, default=8, step=1,
         help="Number of open loans (instalment, car, mortgage) plus credit cards/lines of credit."),
    dict(key="real_estate_loans", label="Real-estate loans", kind="int", min=0, max=100, default=1, step=1,
         help="Number of mortgage and home-equity loans/lines."),
    dict(key="number_of_dependents", label="Dependents", kind="int", min=0, max=30, default=0, step=1,
         help="Family members who depend on the applicant (not counting the applicant)."),
]
FIELD = {f["key"]: f for f in INPUT_FIELDS}

# Readable names for ALL 12 model features (incl. the derived ones).
FEATURE_LABELS: dict[str, str] = {f["key"]: f["label"] for f in INPUT_FIELDS} | {
    "total_past_due": "Total late payments",
    "income_missing": "Income not reported",
}


# ----------------------------------------------------------------------------- validation
def _num(inputs: dict, key: str, required: bool = True) -> float | None:
    v = inputs.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        if required:
            raise ValueError(f"'{key}' is required")
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number, got {v!r}")
    spec = FIELD[key]
    if not (spec["min"] <= v <= spec["max"]) or math.isnan(v):
        raise ValueError(f"'{key}' must be between {spec['min']} and {spec['max']}, got {v}")
    return v


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Typed values -> one-row frame with all model features (project's own cleaning rules applied)."""
    income = _num(inputs, "monthly_income", required=False)  # None / NaN = not reported
    row = {
        "revolving_utilization": _num(inputs, "revolving_utilization"),
        "age": _num(inputs, "age"),
        "past_due_30_59": _num(inputs, "past_due_30_59"),
        "past_due_60_89": _num(inputs, "past_due_60_89"),
        "past_due_90": _num(inputs, "past_due_90"),
        "monthly_income": np.nan if income is None else income,
        "open_credit_lines": _num(inputs, "open_credit_lines"),
        "real_estate_loans": _num(inputs, "real_estate_loans"),
        "number_of_dependents": _num(inputs, "number_of_dependents"),
        # Without income a "ratio to income" is meaningless; raw_to_model_features blanks it anyway.
        "debt_ratio": _num(inputs, "debt_ratio", required=income is not None) if income is not None else np.nan,
    }
    return raw_to_model_features(pd.DataFrame([row]))


# ----------------------------------------------------------------------------- model (cached)
@lru_cache(maxsize=1)
def _champion():
    import shap

    from src.python.models.model_registry import load_champion

    model, feature_cols = load_champion()
    return model, feature_cols, shap.TreeExplainer(model)


@lru_cache(maxsize=1)
def training_base_rate() -> float | None:
    """Share of serious delinquents in the training data (for context in the UI); None if unavailable."""
    try:
        import duckdb

        from src.python.config import DUCKDB_PATH

        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            return float(con.execute("SELECT avg(is_serious_delinquent) FROM applicant_feature_store").fetchone()[0])
        finally:
            con.close()
    except Exception:
        return None


def predict_application(inputs: dict) -> dict:
    """Score one application. Raises ValueError on invalid input."""
    from src.python.explainability.shap_explainer import top_n_shap_features

    df = build_feature_row(inputs)
    model, feature_cols, explainer = _champion()
    X = build_feature_matrix(df, feature_cols)

    proba = round(float(model.predict_proba(X.values)[:, 1][0]), 4)
    shap_row = np.asarray(explainer.shap_values(X.values))[0]
    tier = risk_tier_from_proba(proba)

    values = X.iloc[0].to_dict()
    all_factors = sorted(
        (
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "value": None if pd.isna(values[f]) else float(values[f]),
                "shap_value": round(float(shap_row[i]), 4),
                "direction": "increases_risk" if shap_row[i] > 0 else "decreases_risk",
            }
            for i, f in enumerate(feature_cols)
        ),
        key=lambda d: -abs(d["shap_value"]),
    )

    notes = []
    if pd.isna(values.get("monthly_income")):
        notes.append("Income not reported: the debt ratio is ignored (it only makes sense relative to income), "
                     "exactly as in training.")
    return {
        "predicted_proba": proba,
        "risk_tier": tier,
        "decision": decision_from_tier(tier),
        "top_shap_features": top_n_shap_features(shap_row, feature_cols, n=5),  # same shape as the DB table
        "all_factors": all_factors,
        "features_used": {k: (None if pd.isna(v) else v) for k, v in values.items()},
        "notes": notes,
    }
