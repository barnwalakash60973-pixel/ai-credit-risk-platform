"""
Page 3 - Check Application (enter the features yourself)
Type the applicant's details -> the champion model scores them -> decision, probability, SHAP reasons,
and the RBI-grounded Loan Advisor chat. No applicant ID needed.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

import plotly.graph_objects as go
import streamlit as st

from src.python.dashboard import theme
from src.python.dashboard.advisor_chat import render_advisor_chat
from src.python.scoring.live_predict import FIELD, INPUT_FIELDS, predict_application, training_base_rate

theme.setup("Check Application", "✍️")
theme.page_header("✍️", "Check an Application", "Enter the applicant's details (the same features the model was trained on) and get the decision, with reasons")

# ----------------------------------------------------------------------------- example presets
PRESETS = {
    "— start from an example —": None,
    "Typical low-risk applicant": dict(revolving_utilization=0.10, age=52, past_due_30_59=0, past_due_60_89=0, past_due_90=0,
                                       debt_ratio=0.25, monthly_income=7000.0, open_credit_lines=8, real_estate_loans=1,
                                       number_of_dependents=0, income_reported=True),
    "Maxed-out cards + late payments": dict(revolving_utilization=1.0, age=30, past_due_30_59=3, past_due_60_89=2, past_due_90=2,
                                            debt_ratio=0.8, monthly_income=2500.0, open_credit_lines=12, real_estate_loans=0,
                                            number_of_dependents=3, income_reported=True),
    "Income not reported": dict(revolving_utilization=0.30, age=40, past_due_30_59=0, past_due_60_89=0, past_due_90=0,
                                debt_ratio=0.30, monthly_income=5400.0, open_credit_lines=6, real_estate_loans=1,
                                number_of_dependents=2, income_reported=False),
}


def _apply_preset() -> None:
    p = PRESETS.get(st.session_state["ca_preset"])
    if p:
        for k, v in p.items():
            st.session_state[f"ca_{k}"] = v


for f in INPUT_FIELDS:  # initialise widget state once (widgets below then take no `value=`)
    st.session_state.setdefault(f"ca_{f['key']}", f["default"])
st.session_state.setdefault("ca_income_reported", True)


def num(key: str, **kw):
    f = FIELD[key]
    cast = int if f["kind"] == "int" else float
    return st.number_input(f["label"], min_value=cast(f["min"]), max_value=cast(f["max"]), step=cast(f["step"]),
                           help=f["help"], key=f"ca_{key}", **kw)


# ----------------------------------------------------------------------------- the form
st.selectbox("Example", list(PRESETS), key="ca_preset", on_change=_apply_preset, label_visibility="collapsed")

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("**💳 Credit usage & accounts**")
    num("revolving_utilization", format="%.2f")
    num("open_credit_lines")
    num("real_estate_loans")
with c2, st.container(border=True):
    st.markdown("**🕒 Payment history** (last 2 years)")
    num("past_due_30_59")
    num("past_due_60_89")
    num("past_due_90")
with c3, st.container(border=True):
    st.markdown("**👤 Income & personal**")
    num("age")
    num("number_of_dependents")
    income_reported = st.checkbox("Income reported", key="ca_income_reported",
                                  help="Untick if the applicant did not report an income. The model treats that explicitly.")
    num("monthly_income", disabled=not income_reported, format="%.0f")
    num("debt_ratio", disabled=not income_reported, format="%.2f")


def current_inputs() -> dict:
    inp = {f["key"]: st.session_state[f"ca_{f['key']}"] for f in INPUT_FIELDS}
    if not st.session_state["ca_income_reported"]:
        inp["monthly_income"] = None
        inp["debt_ratio"] = None
    return inp


if st.button("🔎 Check application", type="primary"):
    try:
        with st.spinner("Scoring…"):
            st.session_state["ca_result"] = {"inputs": current_inputs(), "result": predict_application(current_inputs())}
    except Exception as e:
        st.session_state.pop("ca_result", None)
        st.error(f"Could not score this application: {e}")

stored = st.session_state.get("ca_result")
if not stored:
    st.write("")
    theme.empty_state("No result yet", "Fill in the details above (or pick an example) and click “Check application”.", "✍️")
    st.stop()

result, used_inputs = stored["result"], stored["inputs"]
if used_inputs != current_inputs():
    st.info("You changed the details since the last check — click **Check application** again to update the result.")

# ----------------------------------------------------------------------------- the result
proba, tier, decision = result["predicted_proba"], result["risk_tier"], result["decision"]
st.write("")
b1, b2 = st.columns([3, 2])
with b1:
    theme.decision_banner(decision, proba, tier)
with b2:
    base = training_base_rate()
    theme.kpi_row([{"label": "Default probability", "value": f"{proba:.2%}",
                    "hint": (f"average applicant ≈ {base:.1%}" if base else f"tier {tier.replace('_', ' ')}"),
                    "accent": theme.RISK_COLORS[tier]}])

for n in result["notes"]:
    st.caption(f"ℹ️ {n}")

theme.section("Why this decision", "Red bars push the risk up, green bars pull it down (SHAP, in log-odds)")
factors = sorted(result["all_factors"], key=lambda d: abs(d["shap_value"]))  # smallest first -> largest at top
labels = [f"{d['label']} = {'n/a' if d['value'] is None else (int(d['value']) if float(d['value']).is_integer() else round(d['value'], 3))}"
          for d in factors]
fig = go.Figure(go.Bar(
    x=[d["shap_value"] for d in factors], y=labels, orientation="h",
    marker_color=[theme.RISK_COLORS["VERY_HIGH"] if d["shap_value"] > 0 else theme.RISK_COLORS["LOW"] for d in factors],
    text=[f"{d['shap_value']:+.2f}" for d in factors], textposition="outside",
))
fig.update_layout(xaxis_title="Contribution to risk (log-odds)")
st.plotly_chart(theme.style_fig(fig, height=420, legend=False), width="stretch")

up = [d for d in result["all_factors"] if d["shap_value"] > 0.02][:4]
down = [d for d in result["all_factors"] if d["shap_value"] < -0.02][:4]
r1, r2 = st.columns(2)
with r1:
    st.markdown("**⬆ What raises the risk**")
    st.markdown("\n".join(f"- {d['label']}" for d in up) or "- nothing significant")
with r2:
    st.markdown("**⬇ What lowers the risk**")
    st.markdown("\n".join(f"- {d['label']}" for d in down) or "- nothing significant")

st.caption("Demo model trained on the public *Give Me Some Credit* dataset — not a real lending decision.")

# ----------------------------------------------------------------------------- loan advisor chat
if decision == "REJECTED":
    theme.section("Not approved? Ask what to do next", "Answers come from the RBI Financial Awareness Messages booklet")
else:
    theme.section("Questions? Ask the Loan Advisor", "Answers come from the RBI Financial Awareness Messages booklet")
chat_key = "live::" + hashlib.md5(json.dumps(used_inputs, sort_keys=True).encode()).hexdigest()[:10]
render_advisor_chat(chat_key=chat_key, shap_features=result["top_shap_features"], applicant_id=None)
