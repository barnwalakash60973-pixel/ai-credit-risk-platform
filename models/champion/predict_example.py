"""Score applicants with the exported champion model - NO MLflow, Spark, dbt or retraining needed.

    pip install xgboost pandas
    python predict_example.py

Input columns = feature_columns.json (already cleaned, same rules as training):
  - past-due counts of 96/98 are sentinels -> set to NaN before scoring
  - income_missing = 1 if monthly_income is NaN (and then debt_ratio is NaN too)
  - total_past_due = past_due_30_59 + past_due_60_89 + past_due_90
For raw Kaggle-schema files use  src.python.models.features.raw_to_model_features()  first.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
LOW_MAX, MEDIUM_MAX, HIGH_MAX = 0.10, 0.30, 0.60  # same policy as src/python/scoring/decision_rules.py


def load():
    model = xgb.XGBClassifier()
    model.load_model(str(HERE / "model.json"))
    return model, json.loads((HERE / "feature_columns.json").read_text())


def tier(p: float) -> str:
    return "LOW" if p < LOW_MAX else "MEDIUM" if p < MEDIUM_MAX else "HIGH" if p < HIGH_MAX else "VERY_HIGH"


def score(df: pd.DataFrame) -> pd.DataFrame:
    model, features = load()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    p = model.predict_proba(df[features].astype(float).values)[:, 1]
    out = pd.DataFrame({"predicted_proba": p.round(4)})
    out["risk_tier"] = out["predicted_proba"].map(tier)
    out["decision"] = np.where(out["risk_tier"].isin(["LOW", "MEDIUM"]), "APPROVED", "REJECTED")
    return out


if __name__ == "__main__":
    applicants = pd.DataFrame(
        [
            # healthy: low utilisation, no late payments
            dict(revolving_utilization=0.10, age=45, past_due_30_59=0, past_due_60_89=0, past_due_90=0, total_past_due=0,
                 debt_ratio=0.25, monthly_income=6500, income_missing=0, open_credit_lines=8, real_estate_loans=1, number_of_dependents=2),
            # risky: maxed-out cards, repeated late payments
            dict(revolving_utilization=1.05, age=31, past_due_30_59=3, past_due_60_89=2, past_due_90=2, total_past_due=7,
                 debt_ratio=0.9, monthly_income=2800, income_missing=0, open_credit_lines=12, real_estate_loans=0, number_of_dependents=3),
        ]
    )
    print(pd.concat([applicants[["revolving_utilization", "total_past_due"]], score(applicants)], axis=1).to_string(index=False))
