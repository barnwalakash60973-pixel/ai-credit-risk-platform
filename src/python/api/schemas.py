"""Pydantic schemas for FastAPI request/response."""
from pydantic import BaseModel


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str


class CreditDecisionResponse(BaseModel):
    member_id_hash: str
    predicted_proba: float
    risk_tier: str
    decision: str
    narrative: str
    top_shap_features: list[ShapFeature]
    regulation_references: list[str]
    quality_check: dict


class ErrorResponse(BaseModel):
    detail: str


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    grounded: bool


class ApplicationInput(BaseModel):
    """The 10 values an applicant knows; the other model features are derived (see scoring/live_predict.py)."""
    revolving_utilization: float
    age: int
    past_due_30_59: int = 0
    past_due_60_89: int = 0
    past_due_90: int = 0
    debt_ratio: float | None = None          # required when monthly_income is given
    monthly_income: float | None = None      # null = income not reported
    open_credit_lines: int
    real_estate_loans: int = 0
    number_of_dependents: int = 0


class PredictionResponse(BaseModel):
    predicted_proba: float
    risk_tier: str
    decision: str
    top_shap_features: list[dict]
    notes: list[str]
