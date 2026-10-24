"""Risk-tier / decision policy, defined once and shared by scoring and the decision-letter RAG.

Thresholds are on the model's predicted probability of serious delinquency within 2 years.
The model is trained WITHOUT class re-weighting, so probabilities are on the natural scale
(portfolio base rate ~6.7%); calibration is verified at training time (see train_xgboost).
"""

LOW_MAX = 0.10
MEDIUM_MAX = 0.30
HIGH_MAX = 0.60


def risk_tier_from_proba(p: float) -> str:
    if p < LOW_MAX:
        return "LOW"
    elif p < MEDIUM_MAX:
        return "MEDIUM"
    elif p < HIGH_MAX:
        return "HIGH"
    return "VERY_HIGH"


def decision_from_tier(tier: str) -> str:
    return "APPROVED" if tier in ("LOW", "MEDIUM") else "REJECTED"
