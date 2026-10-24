"""
Post-rejection RAG chatbot.

Context given to the LLM:
  1. the applicant's own SHAP reasons (why the model rejected them)
  2. retrieved passages from the RBI FAME booklet (what advice we are allowed to cite)
Answers must come from (2); if retrieval is weak we say so instead of inventing advice.
"""
import logging

import requests

from src.python.rag.credit_decision_explainer import (
    OLLAMA_MODEL, OLLAMA_URL, fetch_shap_explanation,
)
from src.python.rag.fame_kb import query_fame

logger = logging.getLogger(__name__)

# cosine distance from Chroma; tune on ~20 real questions. Above this = "PDF doesn't cover it".
MAX_DISTANCE = 0.85
MAX_HISTORY_TURNS = 6

SYSTEM = """You are a helpful assistant for a loan applicant whose application was declined.
Answer ONLY using the REFERENCE EXCERPTS below (from the RBI 'Financial Awareness Messages' booklet).
Rules:
- Be practical and friendly: give concrete steps the applicant can take.
- Cite the source as (RBI FAME, Message N, p.X) after each point that uses an excerpt.
- If the excerpts do not cover part of the question, say so plainly. You may add one short
  sentence of general guidance, clearly labelled "General guidance (not from the RBI booklet)".
- Never invent numbers, thresholds, timelines or guarantees (e.g. "your score will rise by 50 points").
- Never promise the loan will be approved after following the advice.
- Do not give legal or investment advice."""


def _build_query(question: str, shap_features: list[dict]) -> str:
    # Add the applicant's top risk drivers so vague questions ("how do I fix this?") still retrieve well.
    drivers = ", ".join(f["feature"].replace("_", " ")
                        for f in shap_features if f["direction"] == "increases_risk")[:200]
    return f"{question}. Related factors: {drivers}" if drivers else question


def _format_history(history: list[dict]) -> str:
    return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in history[-MAX_HISTORY_TURNS * 2:])


def answer_question(applicant_id: str | None, question: str, history: list[dict] | None = None,
                    shap_features: list[dict] | None = None) -> dict:
    """Answer a follow-up question after a decline.

    Pass `shap_features` (list of {feature, direction, ...}) for an applicant who is NOT in the database
    (e.g. features typed into the dashboard form); otherwise the reasons are read from the DB by id.
    """
    feats = shap_features if shap_features is not None else fetch_shap_explanation(applicant_id)["top_shap_features"]

    hits = query_fame(_build_query(question, feats), n=3)
    good = [h for h in hits if h["distance"] <= MAX_DISTANCE]

    refs = "\n\n".join(
        f"[Message {h['metadata']['message_no']} - {h['metadata']['title']}, p.{h['metadata']['page_start']}]\n{h['content']}"
        for h in good
    ) or "(no relevant excerpt found)"

    reasons = "\n".join(f"- {f['feature'].replace('_', ' ')} ({f['direction'].replace('_', ' ')})" for f in feats)

    prompt = (
        f"{SYSTEM}\n\nAPPLICANT'S MAIN DECLINE FACTORS:\n{reasons}\n\n"
        f"REFERENCE EXCERPTS:\n{refs}\n\n"
        f"CONVERSATION SO FAR:\n{_format_history(history or [])}\n\n"
        f"APPLICANT QUESTION: {question}\n\nANSWER:"
    )
    r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                        "options": {"temperature": 0.2}}, timeout=300)
    r.raise_for_status()

    return {
        "answer": r.json().get("response", "").strip(),
        "sources": [{"message": h["metadata"]["message_no"], "title": h["metadata"]["title"],
                     "page": h["metadata"]["page_start"], "distance": round(h["distance"], 3)} for h in good],
        "grounded": bool(good),
    }
