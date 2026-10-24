"""
RAG Credit Decision Explainer
Input: applicant_id -> fetch SHAP top-5 (from Day 3) -> query ChromaDB for relevant regulation
-> build prompt -> generate decision letter narrative via Ollama (Mistral 7B)
"""
import ast
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUCKDB_PATH = PROJECT_ROOT / "dbt" / "local_dev.duckdb"

from src.python.models.features import FEATURE_DISPLAY
from src.python.rag.chroma_store import format_citation, query_regulation
from src.python.scoring.decision_rules import decision_from_tier, risk_tier_from_proba  # single source of truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


@dataclass
class DecisionContext:
    member_id_hash: str
    predicted_proba: float
    risk_tier: str
    decision: str
    top_shap_features: list[dict]
    regulation_context: list[dict] = field(default_factory=list)


def fetch_shap_explanation(member_id_hash: str) -> dict:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    row = con.execute(
        """
        SELECT member_id_hash, predicted_proba, top_shap_features_json
        FROM prediction_explanations
        WHERE member_id_hash = ?
        """,
        [member_id_hash],
    ).fetchone()
    con.close()

    if row is None:
        raise ValueError(f"No explanation found for applicant {member_id_hash}")

    member_id_hash, predicted_proba, top_features_raw = row
    try:
        top_features = json.loads(top_features_raw)
    except (json.JSONDecodeError, TypeError):
        top_features = ast.literal_eval(top_features_raw)

    return {
        "member_id_hash": member_id_hash,
        "predicted_proba": predicted_proba,
        "top_shap_features": top_features,
    }


def build_regulation_query(top_features: list[dict]) -> str:
    """Build a regulation search query based on the top SHAP features that increase risk."""
    risk_features = [f["feature"] for f in top_features if f["direction"] == "increases_risk"]
    feature_map = {
        "debt_ratio": "borrower repayment capacity debt servicing obligations to income",
        "revolving_utilization": "credit limit utilisation revolving credit exposure",
        "past_due_30_59": "overdue payments delinquency asset classification",
        "past_due_60_89": "overdue payments delinquency asset classification",
        "past_due_90": "non-performing asset overdue more than 90 days",
        "total_past_due": "overdue payments delinquency asset classification",
        "monthly_income": "borrower income assessment repayment capacity",
        "income_missing": "income verification borrower assessment",
        "open_credit_lines": "multiple credit facilities total exposure of the borrower",
        "real_estate_loans": "housing loan real estate exposure",
        "number_of_dependents": "household financial obligations repayment capacity",
        "age": "credit history borrower track record",
    }
    query_terms = [feature_map.get(f, f) for f in risk_features[:3]]
    return " and ".join(query_terms) if query_terms else "general creditworthiness assessment"


def build_decision_context(member_id_hash: str) -> DecisionContext:
    shap_data = fetch_shap_explanation(member_id_hash)
    proba = shap_data["predicted_proba"]
    tier = risk_tier_from_proba(proba)
    decision = decision_from_tier(tier)

    reg_query = build_regulation_query(shap_data["top_shap_features"])
    regulation_context = query_regulation(reg_query, n_results=3)

    return DecisionContext(
        member_id_hash=member_id_hash,
        predicted_proba=proba,
        risk_tier=tier,
        decision=decision,
        top_shap_features=shap_data["top_shap_features"],
        regulation_context=regulation_context,
    )


def build_prompt(ctx: DecisionContext) -> str:
    shap_lines = "\n".join(
        f"- {f['feature']}: contribution {f['shap_value']:+.4f} ({f['direction'].replace('_', ' ')})"
        for f in ctx.top_shap_features
    )
    reg_lines = "\n".join(
        f"- [{format_citation(r['metadata'])}]: {r['content'][:600]}"
        for r in ctx.regulation_context
    )

    prompt = f"""You are an assistant that writes formal credit decision letters for a financial institution.
Write a formal credit decision letter in English based on the following data.

DECISION DATA:
- Customer ID: {ctx.member_id_hash[:12]}...
- Risk Score (probability of default): {ctx.predicted_proba:.2%}
- Risk Tier: {ctx.risk_tier}
- Decision: {ctx.decision}

KEY FACTORS (SHAP top-5, ordered by influence):
{shap_lines}

RELEVANT REGULATORY REFERENCES (excerpts from official Reserve Bank of India documents):
{reg_lines}

INSTRUCTIONS:
1. Write a formal decision letter with the structure: Header (Customer, Score, Risk Tier, Date),
   Decision (APPROVED/REJECTED), at least 3 Reasons for the Decision (mention ALL the SHAP factors
   above in language easily understood by a layperson customer, not technical SHAP terminology),
   Regulatory References (cite ONLY the references listed above, by document, chapter/paragraph and page),
   and Customer Rights (appeal, data access).
2. Use formal but clear language, avoid machine learning technical jargon.
3. Every SHAP factor MUST be mentioned in the narrative with an explanation of its contribution.
4. Do not fabricate information beyond the data provided. Never invent clause numbers, thresholds or
   page numbers that do not appear in the references above.
"""
    return prompt


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 300) -> tuple[str, dict]:
    """Call the Ollama local API, return (generated_text, metadata for LLMOps logging)."""
    start = time.time()
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    latency = time.time() - start

    data = response.json()
    generated_text = data.get("response", "")

    metadata = {
        "model": model,
        "prompt_tokens_estimate": len(prompt.split()),
        "response_tokens_estimate": len(generated_text.split()),
        "latency_seconds": round(latency, 2),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
    }
    return generated_text, metadata


def quality_check_narrative(narrative: str, ctx: DecisionContext) -> dict:
    narrative_lower = narrative.lower()
    feature_display_map = FEATURE_DISPLAY

    mentioned = []
    missing = []
    for f in ctx.top_shap_features:
        keywords = feature_display_map.get(f["feature"], [f["feature"].replace("_", " ")])
        if any(kw in narrative_lower for kw in keywords):
            mentioned.append(f["feature"])
        else:
            missing.append(f["feature"])

    return {
        "all_features_mentioned": len(missing) == 0,
        "mentioned_features": mentioned,
        "missing_features": missing,
        "coverage_ratio": round(len(mentioned) / max(len(ctx.top_shap_features), 1), 2),
    }

def generate_credit_decision_letter(member_id_hash: str) -> dict:
    ctx = build_decision_context(member_id_hash)
    prompt = build_prompt(ctx)

    logger.info(f"Generating narrative for {member_id_hash} via {OLLAMA_MODEL} ...")
    narrative, llm_metadata = call_ollama(prompt)

    quality = quality_check_narrative(narrative, ctx)
    if not quality["all_features_mentioned"]:
        logger.warning(f"Quality check: features not mentioned -> {quality['missing_features']}")

    return {
        "member_id_hash": ctx.member_id_hash,
        "predicted_proba": ctx.predicted_proba,
        "risk_tier": ctx.risk_tier,
        "decision": ctx.decision,
        "narrative": narrative,
        "top_shap_features": ctx.top_shap_features,
        "regulation_references": [format_citation(r["metadata"]) for r in ctx.regulation_context],
        "llm_metadata": llm_metadata,
        "quality_check": quality,
    }


if __name__ == "__main__":
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    sample_id = con.execute("SELECT member_id_hash FROM prediction_explanations LIMIT 1").fetchone()
    con.close()

    if sample_id:
        result = generate_credit_decision_letter(sample_id[0])
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))