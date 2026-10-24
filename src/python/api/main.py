"""
FastAPI endpoint
POST /credit-decision/{applicant_id} -> score, risk_tier, decision, narrative, SHAP factors
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.python.api.audit_log import create_audit_table, log_audit_entry, log_llmops_metrics
from src.python.api.schemas import (
    ApplicationInput,
    ChatRequest,
    ChatResponse,
    CreditDecisionResponse,
    ErrorResponse,
    PredictionResponse,
)
from src.python.rag.credit_decision_explainer import generate_credit_decision_letter
from src.python.rag.rejection_chatbot import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Decision API",
    description="AI-powered credit decision explanation API with SHAP + RAG regulatory context",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    create_audit_table()
    logger.info("FastAPI startup: audit table is ready")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/credit-decision/{applicant_id}",
    response_model=CreditDecisionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_credit_decision(applicant_id: str, requested_by: str = "api_user"):
    start = time.time()
    try:
        result = generate_credit_decision_letter(applicant_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate credit decision")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    total_latency = time.time() - start

    try:
        log_audit_entry(
            member_id_hash=applicant_id,
            requested_by=requested_by,
            narrative=result["narrative"],
            latency_seconds=total_latency,
        )
        log_llmops_metrics(applicant_id, result["llm_metadata"], result["quality_check"])
    except Exception as e:
        logger.warning(f"Audit/LLMOps logging failed (non-fatal): {e}")

    return CreditDecisionResponse(
        member_id_hash=result["member_id_hash"],
        predicted_proba=result["predicted_proba"],
        risk_tier=result["risk_tier"],
        decision=result["decision"],
        narrative=result["narrative"],
        top_shap_features=result["top_shap_features"],
        regulation_references=result["regulation_references"],
        quality_check=result["quality_check"],
    )



@app.post("/credit-decision/{applicant_id}/chat", response_model=ChatResponse)
def chat_after_rejection(applicant_id: str, body: ChatRequest):
    """Follow-up Q&A after a decline, grounded in the RBI FAME booklet."""
    try:
        return answer_question(applicant_id, body.question, [t.model_dump() for t in body.history])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
def predict(body: ApplicationInput):
    """Score an application from typed-in feature values (no applicant id needed)."""
    from src.python.scoring.live_predict import predict_application

    try:
        r = predict_application(body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(**{k: r[k] for k in ("predicted_proba", "risk_tier", "decision", "top_shap_features", "notes")})
