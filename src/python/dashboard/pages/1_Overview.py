"""
Page 1 - Overview
Risk tier donut chart, daily volume, approval rate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

import plotly.express as px
import streamlit as st

from src.python.dashboard import theme
from src.python.dashboard.db_utils import query_postgres

theme.setup("Overview", "📊")
theme.page_header("📊", "Overview", "Portfolio volume, approval rate and risk-tier mix")

try:
    decisions_df = query_postgres(
        """
        SELECT scoring_date, risk_tier, decision, predicted_proba
        FROM gold.credit_decisions
        ORDER BY scoring_date DESC
        """
    )
except Exception as e:
    theme.empty_state("Could not load decisions", f"{e}", "⚠️")
    st.stop()

if decisions_df.empty:
    theme.empty_state("No decisions yet", "Run scoring_pipeline.py to populate gold.credit_decisions.", "📭")
    st.stop()

total = len(decisions_df)
approved = int((decisions_df["decision"] == "APPROVED").sum())
rejected = int((decisions_df["decision"] == "REJECTED").sum())
rate = approved / total * 100 if total else 0

theme.kpi_row(
    [
        {"label": "Total applications", "value": f"{total:,}", "hint": "all scored applicants", "accent": theme.INDIGO},
        {"label": "Approved", "value": f"{approved:,}", "hint": "LOW + MEDIUM tiers", "accent": theme.RISK_COLORS["LOW"]},
        {"label": "Rejected", "value": f"{rejected:,}", "hint": "HIGH + VERY HIGH tiers", "accent": theme.RISK_COLORS["VERY_HIGH"]},
        {"label": "Approval rate", "value": f"{rate:.1f}%", "hint": "approved / total", "accent": theme.TEAL},
    ]
)

theme.section("Portfolio mix", "Where applicants land and when they arrive")
col_left, col_right = st.columns(2)

with col_left:
    counts = decisions_df["risk_tier"].value_counts().reset_index()
    counts.columns = ["risk_tier", "count"]
    order = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    counts["risk_tier"] = counts["risk_tier"].astype(str)
    fig = px.pie(
        counts, names="risk_tier", values="count", hole=0.62, color="risk_tier",
        color_discrete_map=theme.RISK_COLORS, category_orders={"risk_tier": order},
        title="Risk tier distribution",
    )
    fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#fff", width=3)))
    st.plotly_chart(theme.style_fig(fig), width="stretch")

with col_right:
    daily = decisions_df.groupby("scoring_date").size().reset_index(name="volume")
    fig = px.bar(daily, x="scoring_date", y="volume", title="Daily volume",
                 labels={"scoring_date": "Date", "volume": "Applications"})
    fig.update_traces(marker_color=theme.INDIGO, marker_cornerradius=6)
    st.plotly_chart(theme.style_fig(fig, legend=False), width="stretch")

theme.section("Score distribution", "Predicted probability of serious delinquency")
fig = px.histogram(decisions_df, x="predicted_proba", nbins=30, color="decision",
                   color_discrete_map={"APPROVED": theme.RISK_COLORS["LOW"], "REJECTED": theme.RISK_COLORS["VERY_HIGH"]},
                   labels={"predicted_proba": "Predicted probability"}, barmode="overlay", opacity=0.8)
st.plotly_chart(theme.style_fig(fig, height=360), width="stretch")
