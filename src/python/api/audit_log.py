"""
Audit log + LLMOps logging for every credit-decision request.
- Audit log: who_requested, when, output_hash (compliance)
- LLMOps: log to MLflow (prompt_version, tokens, latency)
"""
import hashlib
import logging
from datetime import datetime

import mlflow
from sqlalchemy import text

from src.python.db import get_engine, serial_pk

from src.python.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1.0"
MLFLOW_LLMOPS_EXPERIMENT = "credit-decision-rag-llmops"


def create_audit_table() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS audit.decision_request_log (
                    id {serial_pk()},
                    member_id_hash VARCHAR(64) NOT NULL,
                    requested_by VARCHAR(100),
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    output_hash VARCHAR(64),
                    prompt_version VARCHAR(20),
                    latency_seconds DOUBLE PRECISION,
                    status VARCHAR(20) DEFAULT 'SUCCESS'
                )
                """
            )
        )
    logger.info("Table audit.decision_request_log is ready")


def hash_output(narrative: str) -> str:
    return hashlib.sha256(narrative.encode()).hexdigest()


def log_audit_entry(
    member_id_hash: str,
    requested_by: str,
    narrative: str,
    latency_seconds: float,
    status: str = "SUCCESS",
) -> None:
    engine = get_engine()
    output_hash = hash_output(narrative)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO audit.decision_request_log
                    (member_id_hash, requested_by, output_hash, prompt_version, latency_seconds, status)
                VALUES (:hash, :by, :out_hash, :version, :latency, :status)
                """
            ),
            {
                "hash": member_id_hash,
                "by": requested_by,
                "out_hash": output_hash,
                "version": PROMPT_VERSION,
                "latency": latency_seconds,
                "status": status,
            },
        )
    logger.info(f"Audit logged: {member_id_hash} by {requested_by} | output_hash={output_hash[:12]}...")


def log_llmops_metrics(member_id_hash: str, llm_metadata: dict, quality_check: dict) -> None:
    """Log to MLflow: prompt_version, tokens, latency, quality metrics per request."""
    mlflow.set_experiment(MLFLOW_LLMOPS_EXPERIMENT)
    with mlflow.start_run(run_name=f"decision_{member_id_hash[:8]}_{datetime.utcnow().isoformat()}"):
        mlflow.log_param("member_id_hash", member_id_hash)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("model", llm_metadata.get("model"))

        mlflow.log_metric("latency_seconds", llm_metadata.get("latency_seconds", 0))
        mlflow.log_metric("prompt_tokens_estimate", llm_metadata.get("prompt_tokens_estimate", 0))
        mlflow.log_metric("response_tokens_estimate", llm_metadata.get("response_tokens_estimate", 0))
        mlflow.log_metric("quality_coverage_ratio", quality_check.get("coverage_ratio", 0))
        mlflow.log_metric("all_features_mentioned", int(quality_check.get("all_features_mentioned", False)))

    logger.info(f"LLMOps metrics logged for {member_id_hash}")