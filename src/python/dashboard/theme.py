"""
Shared look & feel for the Streamlit dashboard.

Everything visual lives here so every page stays consistent:
  * ``setup()``            - page config + global CSS (call first on every page)
  * ``hero()``/``page_header()`` - banners
  * ``kpi_row()``          - KPI cards
  * ``badge()``/``decision_banner()`` - risk tier / decision chips
  * ``style_fig()``        - one Plotly style for all charts
"""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

# ----------------------------------------------------------------------------- palette
NAVY = "#0B1B3A"
INDIGO = "#4F46E5"
VIOLET = "#7C3AED"
TEAL = "#14B8A6"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E3E8F2"
BG = "#F4F6FB"

RISK_COLORS = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#F97316",
    "VERY_HIGH": "#EF4444",
}
COLORWAY = [INDIGO, TEAL, "#F59E0B", "#EF4444", VIOLET, "#0EA5E9", "#10B981"]

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --navy: {NAVY}; --indigo: {INDIGO}; --violet: {VIOLET}; --teal: {TEAL};
  --ink: {INK}; --muted: {MUTED}; --border: {BORDER}; --bg: {BG};
  --radius: 16px; --shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06);
}}

html, body, [class*="css"], .stApp, button, input, textarea {{
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}}
.stApp {{ background: var(--bg); color: var(--ink); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}

/* ---- chrome ---- */
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---- headings ---- */
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.02em; font-weight: 700; }}
h2 {{ font-size: 1.35rem !important; }}
h3 {{ font-size: 1.1rem !important; }}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {NAVY} 0%, #14275C 100%);
  border-right: 0;
}}
section[data-testid="stSidebar"] * {{ color: #E2E8F8; }}
[data-testid="stSidebarNav"]::before {{
  content: "\\1F3E6  CreditRisk AI";
  display: block; padding: 1.4rem 1.2rem .4rem; font-size: 1.25rem; font-weight: 800;
  letter-spacing: -0.02em; color: #fff;
}}
[data-testid="stSidebarNav"]::after {{
  content: "Explainable credit decisions";
  display: block; padding: 0 1.2rem 1rem; font-size: .74rem; color: #93A4D3;
  border-bottom: 1px solid rgba(255,255,255,.08); margin-bottom: .6rem;
}}
[data-testid="stSidebarNav"] a {{
  border-radius: 10px; margin: 2px 10px; padding: .5rem .8rem; transition: all .15s ease;
}}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
  background: linear-gradient(90deg, rgba(79,70,229,.9), rgba(124,58,237,.75));
  box-shadow: 0 6px 18px rgba(79,70,229,.35);
}}
[data-testid="stSidebarNav"] a[aria-current="page"] * {{ color: #fff !important; font-weight: 600; }}
.side-foot {{ font-size: .72rem; color: #8FA0CF; padding: .6rem .4rem; line-height: 1.5; }}
.side-foot b {{ color: #fff; }}

/* ---- hero ---- */
.hero {{
  position: relative; overflow: hidden; border-radius: 22px; padding: 2.4rem 2.6rem; margin-bottom: 1.6rem;
  background: radial-gradient(1200px 400px at 90% -20%, rgba(20,184,166,.35), transparent 60%),
              linear-gradient(135deg, {NAVY} 0%, #1E2A78 55%, {INDIGO} 100%);
  color: #fff; box-shadow: 0 18px 40px rgba(30,42,120,.28);
}}
.hero .eyebrow {{ font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; color: #9FB3FF; font-weight: 600; }}
.hero h1 {{ color: #fff; font-size: 2.35rem; line-height: 1.15; margin: .5rem 0 .7rem; font-weight: 800; }}
.hero p {{ color: #D5DDFB; font-size: 1.02rem; max-width: 760px; margin: 0; line-height: 1.6; }}
.chips {{ margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: .5rem; }}
.chip {{
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); color: #fff;
  padding: .28rem .75rem; border-radius: 999px; font-size: .78rem; font-weight: 500; backdrop-filter: blur(4px);
}}

.page-head {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1.4rem; }}
.page-head .ico {{
  width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center; font-size: 1.5rem;
  background: linear-gradient(135deg, {INDIGO}, {VIOLET}); box-shadow: 0 10px 22px rgba(79,70,229,.3);
}}
.page-head h1 {{ margin: 0; font-size: 1.75rem; font-weight: 800; }}
.page-head p {{ margin: .15rem 0 0; color: var(--muted); font-size: .95rem; }}

/* ---- cards ---- */
.card {{
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.2rem 1.3rem; box-shadow: var(--shadow); height: 100%;
}}
.card h4 {{ margin: 0 0 .35rem; font-size: 1rem; }}
.card p {{ margin: 0; color: var(--muted); font-size: .9rem; line-height: 1.55; }}
.card .ico {{ font-size: 1.5rem; margin-bottom: .5rem; }}
.card.link:hover {{ transform: translateY(-2px); transition: .15s; border-color: #C7D2FE; }}

.kpi {{
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.05rem 1.2rem; box-shadow: var(--shadow); position: relative; overflow: hidden;
}}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent, {INDIGO}); }}
.kpi .lbl {{ font-size: .74rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }}
.kpi .val {{ font-size: 1.85rem; font-weight: 800; letter-spacing: -0.03em; margin-top: .25rem; color: var(--ink); }}
.kpi .hint {{ font-size: .78rem; color: var(--muted); margin-top: .15rem; }}

/* native st.metric -> card look */
[data-testid="stMetric"] {{
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem 1.2rem; box-shadow: var(--shadow);
}}
[data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-weight: 800; letter-spacing: -0.03em; }}

/* charts / tables as cards */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
  background: #fff; border: 1px solid var(--border); border-radius: var(--radius);
  padding: .5rem; box-shadow: var(--shadow); overflow: hidden;
}}

/* ---- badges ---- */
.badge {{ display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .7rem; border-radius: 999px;
  font-size: .75rem; font-weight: 700; letter-spacing: .02em; border: 1px solid transparent; }}
.badge.low {{ background: #D1FAE5; color: #047857; border-color: #A7F3D0; }}
.badge.medium {{ background: #FEF3C7; color: #B45309; border-color: #FDE68A; }}
.badge.high {{ background: #FFEDD5; color: #C2410C; border-color: #FED7AA; }}
.badge.very_high {{ background: #FEE2E2; color: #B91C1C; border-color: #FECACA; }}
.badge.info {{ background: #E0E7FF; color: #3730A3; border-color: #C7D2FE; }}
.badge.neutral {{ background: #F1F5F9; color: #475569; border-color: #E2E8F0; }}

.decision {{ border-radius: var(--radius); padding: 1.15rem 1.4rem; display: flex; align-items: center; gap: 1rem;
  border: 1px solid; box-shadow: var(--shadow); }}
.decision .big {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }}
.decision .sub {{ font-size: .85rem; opacity: .85; }}
.decision.approved {{ background: linear-gradient(135deg,#ECFDF5,#D1FAE5); border-color: #A7F3D0; color: #065F46; }}
.decision.rejected {{ background: linear-gradient(135deg,#FEF2F2,#FEE2E2); border-color: #FECACA; color: #991B1B; }}
.decision .ico {{ font-size: 2rem; }}

/* ---- regulation result ---- */
.reg {{ background: #fff; border: 1px solid var(--border); border-left: 4px solid {INDIGO}; border-radius: 12px;
  padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: var(--shadow); }}
.reg .meta {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .55rem; }}
.reg .txt {{ color: #334155; font-size: .92rem; line-height: 1.65; }}

.letter {{ background: #fff; border: 1px solid var(--border); border-radius: var(--radius); padding: 1.6rem 2rem;
  box-shadow: var(--shadow); line-height: 1.7; }}

.empty {{ text-align: center; padding: 2.4rem 1rem; background: #fff; border: 1px dashed #CBD5E1; border-radius: var(--radius); }}
.empty .ico {{ font-size: 2.2rem; }} .empty h4 {{ margin: .4rem 0 .2rem; }} .empty p {{ color: var(--muted); margin: 0; }}

.section-title {{ display: flex; align-items: baseline; gap: .6rem; margin: 1.6rem 0 .7rem; }}
.section-title h3 {{ margin: 0; }} .section-title span {{ color: var(--muted); font-size: .85rem; }}

/* ---- widgets ---- */
.stButton > button, .stDownloadButton > button {{
  border-radius: 12px; font-weight: 600; padding: .55rem 1.2rem; border: 1px solid var(--border); transition: all .15s;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {INDIGO}, {VIOLET}); border: 0; color: #fff;
  box-shadow: 0 8px 20px rgba(79,70,229,.32);
}}
.stButton > button[kind="primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 12px 26px rgba(79,70,229,.4); }}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {{ border-radius: 12px !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: .55rem 1rem; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {INDIGO}; }}
[data-testid="stExpander"] {{ background: #fff; border: 1px solid var(--border) !important; border-radius: 14px; }}
div[data-testid="stAlert"] {{ border-radius: 14px; }}
hr {{ border-color: var(--border) !important; }}
</style>
"""


# ----------------------------------------------------------------------------- setup
def setup(page_title: str, icon: str = "🏦", layout: str = "wide") -> None:
    """Must be the FIRST Streamlit call on a page."""
    st.set_page_config(page_title=f"{page_title} · CreditRisk AI", page_icon=icon, layout=layout)
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            "<div class='side-foot'><b>Stack</b><br>XGBoost · SHAP · ChromaDB<br>Ollama · FastAPI · Airflow<br><br>"
            "<b>Knowledge base</b><br>Official RBI directions (PDF)</div>",
            unsafe_allow_html=True,
        )


def _h(s: str) -> str:
    return html.escape(str(s), quote=True)


def _md(markup: str) -> None:
    """Render raw HTML; leading whitespace would turn markup into a code block."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- components
def hero(title: str, subtitle: str, eyebrow: str = "", chips: Iterable[str] = ()) -> None:
    chip_html = "".join(f"<span class='chip'>{_h(c)}</span>" for c in chips)
    _md(
        f"""<div class="hero">
        <div class="eyebrow">{_h(eyebrow)}</div>
        <h1>{_h(title)}</h1>
        <p>{_h(subtitle)}</p>
        <div class="chips">{chip_html}</div>
        </div>"""
    )


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    _md(
        f"""<div class="page-head"><div class="ico">{icon}</div>
        <div><h1>{_h(title)}</h1><p>{_h(subtitle)}</p></div></div>"""
    )


def section(title: str, caption: str = "") -> None:
    _md(f"<div class='section-title'><h3>{_h(title)}</h3><span>{_h(caption)}</span></div>")


def feature_card(icon: str, title: str, body: str) -> None:
    _md(f"<div class='card link'><div class='ico'>{icon}</div><h4>{_h(title)}</h4><p>{_h(body)}</p></div>")


def kpi_row(items: list[dict]) -> None:
    """items: [{label, value, hint?, accent?}] rendered as responsive KPI cards."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            _md(
                f"""<div class="kpi" style="--accent:{it.get('accent', INDIGO)}">
                <div class="lbl">{_h(it['label'])}</div>
                <div class="val">{_h(it['value'])}</div>
                <div class="hint">{_h(it.get('hint', ''))}</div></div>"""
            )


def badge(text: str, kind: str = "neutral") -> str:
    """Returns an HTML snippet (embed inside _md / st.markdown(unsafe_allow_html=True))."""
    return f"<span class='badge {kind.lower()}'>{_h(text)}</span>"


def risk_badge(tier: str) -> str:
    return badge(tier.replace("_", " "), tier.lower())


def decision_banner(decision: str, proba: float, tier: str) -> None:
    ok = decision == "APPROVED"
    _md(
        f"""<div class="decision {'approved' if ok else 'rejected'}">
        <div class="ico">{'✅' if ok else '⛔'}</div>
        <div><div class="big">{_h(decision)}</div>
        <div class="sub">Default probability {proba:.2%} · {risk_badge(tier)}</div></div></div>"""
    )


def empty_state(title: str, body: str, icon: str = "🗂️") -> None:
    _md(f"<div class='empty'><div class='ico'>{icon}</div><h4>{_h(title)}</h4><p>{_h(body)}</p></div>")


def regulation_card(rank: int, citation: str, text: str, distance: float | None = None, tags: Iterable[str] = ()) -> None:
    tag_html = "".join(badge(t, "info") for t in tags)
    rel = ""
    if distance is not None:
        rel = badge(f"relevance {max(0.0, 1 - distance / 2):.0%}", "neutral")
    _md(
        f"""<div class="reg"><div class="meta">{badge(f'#{rank}', 'neutral')}{tag_html}{rel}</div>
        <div style="font-weight:700;margin-bottom:.35rem">{_h(citation)}</div>
        <div class="txt">{_h(text)}</div></div>"""
    )


# ----------------------------------------------------------------------------- charts
def style_fig(fig, height: int | None = 380, legend: bool = True):
    """Apply the shared Plotly look to any figure."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=16, r=16, t=48, b=16),
        title=dict(font=dict(size=15, color=INK), x=0.01),
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=BORDER),
        height=height,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#EDF0F7", zeroline=False, tickfont=dict(color=MUTED))
    return fig
