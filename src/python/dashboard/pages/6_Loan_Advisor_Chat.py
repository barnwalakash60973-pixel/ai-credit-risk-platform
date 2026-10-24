"""
Page 6 - Loan Advisor Chat
After a loan is declined, the applicant can ask follow-up questions
("How can I improve?") and get answers grounded in the RBI FAME booklet.
(For someone who is not in the database, use page 3 "Check Application" - it has the same chat.)
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

import streamlit as st

from src.python.dashboard import theme
from src.python.dashboard.advisor_chat import render_advisor_chat
from src.python.dashboard.db_utils import get_duckdb_connection
from src.python.scoring.decision_rules import MEDIUM_MAX, decision_from_tier, risk_tier_from_proba

theme.setup("Loan Advisor Chat", "💬")
theme.page_header("💬", "Loan Advisor Chat", "Ask what to do next after a declined application — answers come from the RBI FAME booklet")

# ----------------------------------------------------------------------------- applicant picker
con = get_duckdb_connection()
try:
    ids = con.execute(
        "SELECT member_id_hash FROM prediction_explanations WHERE predicted_proba >= ? ORDER BY predicted_proba DESC LIMIT 200",
        [MEDIUM_MAX],
    ).fetchdf()["member_id_hash"].tolist()
except Exception:
    ids = []

if not ids:
    theme.empty_state("No declined applicants found", "Run the SHAP explainer first, or check that some applicants were rejected.", "🧠")
    st.stop()

with st.container(border=True):
    applicant_id = st.selectbox("Declined applicant", ids, help="Only rejected applicants are listed — this chat is for post-decline guidance.")

row = con.execute("SELECT predicted_proba, top_shap_features_json FROM prediction_explanations WHERE member_id_hash = ?", [applicant_id]).fetchone()
proba, raw = float(row[0]), row[1]
try:
    top_features = json.loads(raw)
except (json.JSONDecodeError, TypeError):
    top_features = ast.literal_eval(raw)
tier = risk_tier_from_proba(proba)

st.write("")
theme.decision_banner(decision_from_tier(tier), proba, tier)

risk_drivers = [f["feature"] for f in top_features if f["direction"] == "increases_risk"]
if risk_drivers:
    st.caption("Main reasons for the decision: " + ", ".join(d.replace("_", " ") for d in risk_drivers[:3]))

theme.section("Ask a question", "Type below or pick a suggestion")
render_advisor_chat(chat_key=applicant_id, shap_features=top_features, applicant_id=applicant_id)
