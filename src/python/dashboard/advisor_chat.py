"""
Reusable "Loan Advisor" chat box (Streamlit). Used by:
  * page 6  - applicants already in the database (reasons read from DB by id)
  * page 3b - applications typed into the form (reasons passed in directly)
"""
from __future__ import annotations

import streamlit as st

# Question starters keyed by the SHAP feature that pushed the applicant's risk up.
SUGGESTIONS = {
    "revolving_utilization": "My loan application was rejected because of high credit utilization. How can I improve?",
    "debt_ratio": "My loan was rejected because I have too much debt compared to my income. What should I do?",
    "past_due_30_59": "My loan was rejected because of late payments. How can I rebuild my credit?",
    "past_due_60_89": "My loan was rejected because of late payments. How can I rebuild my credit?",
    "past_due_90": "My loan was rejected because of overdue payments. How can I rebuild my credit?",
    "total_past_due": "My loan was rejected because of late payments. How can I rebuild my credit?",
    "monthly_income": "My loan was rejected partly because of my income. What can I do?",
    "income_missing": "My loan was rejected partly because my income was not reported. What can I do?",
    "open_credit_lines": "My loan was rejected because I have many open credit accounts. What should I do?",
}
GENERIC = [
    "How can I improve my credit score?",
    "How do I borrow responsibly?",
    "I think this decision is wrong. How do I file a complaint?",
]


def _render_sources(msg: dict) -> None:
    if msg["role"] != "assistant":
        return
    if not msg.get("grounded", True):
        st.caption("⚠️ The RBI booklet has no closely matching passage for this question.")
    elif msg.get("sources"):
        with st.expander("📚 Sources from the RBI booklet"):
            for s in msg["sources"]:
                st.markdown(f"- Message {s['message']} — {s['title']} (p.{s['page']})")


def render_advisor_chat(chat_key: str, shap_features: list[dict], applicant_id: str | None = None) -> None:
    """Draw suggestions + history + chat box. `chat_key` separates conversations (one per applicant/application)."""
    from src.python.rag.rejection_chatbot import answer_question  # lazy: heavy imports only when chat is shown

    chats = st.session_state.setdefault("advisor_chats", {})
    history = chats.setdefault(chat_key, [])
    pending_key = f"pending_q::{chat_key}"

    with st.sidebar:
        if st.button("🗑️ Clear this conversation", key=f"clear::{chat_key}"):
            chats[chat_key] = []
            st.rerun()
        st.caption("Source: RBI Financial Awareness Messages (FAME), 3rd ed., July 2020. Educational guidance only — "
                   "it does not guarantee approval of a future application.")

    if not history:
        drivers = [f["feature"] for f in shap_features if f["direction"] == "increases_risk"]
        suggestions = []
        for f in drivers:
            s = SUGGESTIONS.get(f)
            if s and s not in suggestions:
                suggestions.append(s)
        for g in GENERIC:
            if len(suggestions) >= 4:
                break
            if g not in suggestions:
                suggestions.append(g)
        cols = st.columns(2)
        for i, s in enumerate(suggestions[:4]):
            if cols[i % 2].button(s, key=f"sugg::{chat_key}::{i}", width="stretch"):
                st.session_state[pending_key] = s
                st.rerun()

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg)

    question = st.chat_input("e.g. How can I improve my credit score?", key=f"input::{chat_key}") or st.session_state.pop(pending_key, None)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    llm_history = [{"role": m["role"], "content": m["content"]} for m in history]  # without UI-only fields
    history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Looking through the RBI booklet…"):
            try:
                result = answer_question(applicant_id, question, llm_history, shap_features=shap_features)
            except Exception as e:
                history.pop()  # let the user retry without a dangling question
                st.error(f"Could not get an answer: {e}. Is the FAME index built (`python -m src.python.rag.fame_kb`) "
                         "and Ollama running (`ollama serve`)?")
                st.stop()
        msg = {"role": "assistant", "content": result["answer"], "sources": result["sources"], "grounded": result["grounded"]}
        st.markdown(msg["content"])
        _render_sources(msg)
    history.append(msg)
