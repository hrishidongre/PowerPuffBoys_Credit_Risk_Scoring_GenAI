"""
agent_app.py  —  Phase 2 Streamlit UI  —  Agentic Credit Risk Assistant
command: streamlit run agent_app.py
"""

import os
import sys
import logging

# Set logging to WARNING to hide info/debug logs in production
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# CRITICAL FIX FOR APPLE SILICON SEGFAULT:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LGBM_AVOID_FAST_PREDICT"] = "1"

import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Page Config
st.set_page_config(
    page_title="Credit Risk Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: #0a0a0a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #a1a1aa;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #a1a1aa; font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    .block-container { padding: 3rem 4rem 4rem !important; max-width: 1200px; }

    .hero-wrapper { text-align: center; padding: 2.5rem 0 3rem; }
    .hero-title {
        font-size: 2.8rem; font-weight: 900; color: #ffffff;
        letter-spacing: -0.04em; line-height: 1.15; margin-bottom: 0.6rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #FBBF24, #F59E0B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle { color: #71717a; font-size: 1.05rem; font-weight: 400; line-height: 1.6; }

    .section-header {
        font-size: 0.82rem; font-weight: 700; color: #FBBF24;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 3rem 0 1.6rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .cards-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.2rem; margin-bottom: 1.2rem; }
    .metric-card {
        background: #161616; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.6rem 1.4rem; text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(251,191,36,0.3); background: #1a1a1a;
        box-shadow: 0 8px 30px rgba(251,191,36,0.04); transform: translateY(-2px);
    }
    .metric-label { font-size: 0.65rem; color: #52525b; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.7rem; font-weight: 700; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #f4f4f5; letter-spacing: -0.02em; }
    .metric-value-sm { font-size: 0.95rem; font-weight: 600; color: #f4f4f5; letter-spacing: -0.01em; line-height: 1.4; }

    .stTabs { margin-top: 0.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; padding: 0.4rem 0; }
    .stTabs [data-baseweb="tab"] {
        background: #161616; border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        color: #71717a; font-weight: 600; font-size: 0.82rem;
        padding: 0.65rem 1.3rem; letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #a1a1aa; border-color: rgba(255,255,255,0.12); }
    .stTabs [aria-selected="true"] { background: #1c1c1c !important; color: #FBBF24 !important; border-color: #FBBF24 !important; }
    .stTabs [data-baseweb="tab-panel"] { padding: 1.8rem 0 0.5rem; }

    .stNumberInput label, .stSlider label {
        color: #a1a1aa !important; font-weight: 500 !important;
        font-size: 0.85rem !important; margin-bottom: 0.3rem !important;
    }
    .stNumberInput > div { margin-bottom: 1.2rem; }
    .stSelectbox label { color: #a1a1aa !important; font-weight: 500 !important; font-size: 0.85rem !important; }

    .stButton > button[kind="primary"] {
        background: #FBBF24 !important; color: #0a0a0a !important;
        border: none !important; font-weight: 700 !important;
        border-radius: 50px !important; padding: 0.85rem 3rem !important;
        font-size: 0.95rem !important; letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 20px rgba(251,191,36,0.15) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #F59E0B !important; transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(251,191,36,0.25) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: transparent !important; color: #a1a1aa !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important; font-weight: 500 !important;
        font-size: 0.82rem !important; padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover { border-color: #FBBF24 !important; color: #FBBF24 !important; }

    .risk-result {
        border-radius: 20px; padding: 3rem 2.5rem;
        text-align: center; margin: 2.5rem auto; max-width: 520px;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.96) translateY(8px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }
    .risk-label { font-size: 2.2rem; font-weight: 900; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .risk-sub   { font-size: 0.85rem; color: #71717a; font-weight: 400; }

    .overview-box {
        background: #161616; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 1.4rem 1.8rem;
        color: #a1a1aa; font-size: 0.88rem; line-height: 1.75;
        margin-bottom: 0.8rem;
    }
    .overview-rec {
        color: #f4f4f5; font-weight: 600; font-size: 0.9rem;
        margin-top: 0.8rem; padding-top: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    .prob-section { max-width: 700px; margin: 0 auto; padding: 0.5rem 0 1rem; }
    .prob-container { margin: 0.75rem 0; }
    .prob-bar-bg { background: #161616; border-radius: 8px; height: 32px; overflow: hidden; border: 1px solid rgba(255,255,255,0.04); }
    .prob-bar-fill {
        height: 100%; border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
        display: flex; align-items: center;
        padding-left: 12px; font-size: 0.78rem; font-weight: 700; color: #0a0a0a;
    }
    .prob-label { font-size: 0.72rem; color: #71717a; margin-bottom: 0.35rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }

    .factor-item {
        background: #161616; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    }
    .factor-title { color: #f4f4f5; font-size: 0.88rem; font-weight: 600; margin-bottom: 0.3rem; }
    .factor-body  { color: #71717a; font-size: 0.81rem; line-height: 1.6; }

    .action-item {
        padding: 0.55rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
        color: #a1a1aa; font-size: 0.85rem; line-height: 1.5;
    }
    .action-item:last-child { border-bottom: none; }
    .bullet { display: inline-block; width: 5px; height: 5px; background: #FBBF24; border-radius: 50%; margin-right: 10px; vertical-align: middle; }

    .guideline-ref {
        background: #111; border-left: 3px solid #FBBF24;
        border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
        margin-bottom: 0.6rem; color: #a1a1aa;
        font-size: 0.81rem; line-height: 1.6; font-style: italic;
    }

    .chunk-card { background: #161616; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.7rem; }
    .chunk-source { font-size: 0.68rem; color: #FBBF24; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
    .chunk-text   { color: #71717a; font-size: 0.8rem; line-height: 1.6; }

    .styled-divider { height: 1px; background: rgba(255,255,255,0.04); margin: 2rem 0; }

    .sidebar-brand { text-align: center; padding: 1.8rem 0 1rem; }
    .sidebar-brand-icon { width: 48px; height: 48px; background: #FBBF24; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 900; color: #0a0a0a; margin-bottom: 0.8rem; }
    .sidebar-brand-name { font-size: 1.05rem; font-weight: 700; color: #f4f4f5; letter-spacing: -0.01em; }
    .sidebar-brand-sub  { font-size: 0.7rem; color: #52525b; margin-top: 0.25rem; }

    .app-footer { text-align: center; color: #3f3f46; font-size: 0.72rem; padding: 2rem 0; line-height: 1.7; }
</style>
""",
    unsafe_allow_html=True,
)


#  Data & Model
@st.cache_resource
def get_agent_runner():
    try:
        from graph import run_agent

        return run_agent
    except Exception as e:
        logger.exception("Failed to load Agent Runner!")
        raise e


@st.cache_data
def load_cibil():
    path = "./Dataset/Unseen_CIBL_Data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


cibil_df = load_cibil()

RISK_LEVELS = {
    "P1": ("VERY LOW RISK", "#22C55E", "#0d1f12", "#22C55E"),
    "P2": ("LOW RISK", "#FBBF24", "#1f1a0a", "#FBBF24"),
    "P3": ("MEDIUM RISK", "#F59E0B", "#1f180a", "#F59E0B"),
    "P4": ("HIGH RISK", "#EF4444", "#1f0d0d", "#EF4444"),
}


#  Session State
if "selected_prospect" not in st.session_state:
    st.session_state.selected_prospect = (
        str(cibil_df["PROSPECT_ID"].values[0]) if not cibil_df.empty else "MANUAL_001"
    )
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None


#  SIDEBAR
with st.sidebar:
    st.markdown(
        """
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">CR</div>
        <div class="sidebar-brand-name">Credit Risk Agent</div>
        <div class="sidebar-brand-sub">LangGraph + RAG + Groq LLM</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if not cibil_df.empty:
        prospect_ids = [str(p) for p in cibil_df["PROSPECT_ID"].unique()]
        options = ["Manual Entry"] + prospect_ids
        st.session_state.selected_prospect = st.selectbox(
            "Select Prospect",
            options,
            index=options.index(st.session_state.selected_prospect)
            if st.session_state.selected_prospect in options
            else 0,
        )
    else:
        st.session_state.selected_prospect = "Manual Entry"
        st.info("No dataset found at Dataset/Unseen_CIBL_Data.csv")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Random", use_container_width=True) and not cibil_df.empty:
            st.session_state.selected_prospect = str(
                np.random.choice(cibil_df["PROSPECT_ID"].unique())
            )
            st.session_state.agent_result = None
            st.rerun()
    with col_b:
        if st.button("Clear", use_container_width=True):
            st.session_state.agent_result = None
            st.rerun()

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div style="font-size:0.68rem; color:#3f3f46; text-align:center; padding:1rem 0; line-height:1.7;">
        <span style="color:#71717a; font-weight:600;">Team PowerPuff Boys</span><br>
        Agentic Credit Risk Classification<br>
        LightGBM + SMOTE · 80.4% Accuracy
    </div>
    """,
        unsafe_allow_html=True,
    )


#  HERO
st.markdown(
    """
<div class="hero-wrapper">
    <div class="hero-title">Agentic Credit Risk <span>Intelligence</span></div>
    <div class="hero-subtitle">
        ML risk classification &nbsp;·&nbsp; RAG guideline retrieval &nbsp;·&nbsp; LLM-generated report
    </div>
</div>
""",
    unsafe_allow_html=True,
)


#  TABS
tab_report, tab_arch = st.tabs(["Risk Analysis", "System Architecture"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab_report:
    # Pre-fill from selected prospect
    prefill = {}
    prospect_id = "MANUAL_001"
    selected = st.session_state.selected_prospect

    if selected != "Manual Entry" and not cibil_df.empty:
        row = cibil_df[cibil_df["PROSPECT_ID"].astype(str) == selected]
        if not row.empty:
            prefill = row.iloc[0].to_dict()
            prospect_id = selected

    # ── Prospect Overview ──────────────────────────────────────────────────────
    if prefill:
        st.markdown(
            f'<div class="section-header">Prospect Overview  /  ID #{prospect_id}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="cards-grid">
    <div class="metric-card">
        <div class="metric-label">Gender</div>
        <div class="metric-value">{prefill.get("GENDER", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Marital Status</div>
        <div class="metric-value-sm">{prefill.get("MARITALSTATUS", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Education</div>
        <div class="metric-value-sm">{prefill.get("EDUCATION", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Monthly Income</div>
        <div class="metric-value">Rs. {int(prefill.get("NETMONTHLYINCOME", 0)):,}</div>
    </div>
</div>
<div class="cards-grid">
    <div class="metric-card">
        <div class="metric-label">Enquiries (3 Mo)</div>
        <div class="metric-value">{int(prefill.get("enq_L3m", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Missed Payments</div>
        <div class="metric-value">{int(prefill.get("Tot_Missed_Pmnt", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Oldest Account</div>
        <div class="metric-value">{int(prefill.get("Age_Oldest_TL", 0))} mo</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Trade Lines</div>
        <div class="metric-value">{int(prefill.get("Total_TL", 0))}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    #  Feature Input Tabs
    st.markdown(
        '<div class="section-header">Bureau Data Inputs</div>', unsafe_allow_html=True
    )

    ftab1, ftab2, ftab3 = st.tabs(
        ["Enquiry & Trade Lines", "Delinquency & Payments", "Demographics"]
    )

    with ftab1:
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            enq_l3m = st.number_input(
                "Enquiries — Last 3 Months",
                value=int(prefill.get("enq_L3m", 3)),
                min_value=0,
            )
            enq_l6m = st.number_input(
                "Enquiries — Last 6 Months",
                value=int(prefill.get("enq_L6m", 5)),
                min_value=0,
            )
            enq_l12m = st.number_input(
                "Enquiries — Last 12 Months",
                value=int(prefill.get("enq_L12m", 7)),
                min_value=0,
            )
        with c2:
            age_oldest = st.number_input(
                "Oldest Account (months)",
                value=int(prefill.get("Age_Oldest_TL", 36)),
                min_value=0,
            )
            age_newest = st.number_input(
                "Newest Account (months)",
                value=int(prefill.get("Age_Newest_TL", 6)),
                min_value=0,
            )
            tot_tl = st.number_input(
                "Total Trade Lines", value=int(prefill.get("Total_TL", 5)), min_value=0
            )
        with c3:
            num_std = st.number_input(
                "Standard Payments (total)",
                value=int(prefill.get("num_std", 10)),
                min_value=0,
            )
            num_std_12mts = st.number_input(
                "Standard Payments (12 months)",
                value=int(prefill.get("num_std_12mts", 8)),
                min_value=0,
            )
            tot_missed = st.number_input(
                "Total Missed Payments",
                value=int(prefill.get("Tot_Missed_Pmnt", 0)),
                min_value=0,
            )

    with ftab2:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            time_recent = st.number_input(
                "Time Since Recent Delinquency (months)",
                value=float(prefill.get("time_since_recent_deliquency", 0)),
                min_value=0.0,
            )
            time_first = st.number_input(
                "Time Since First Delinquency (months)",
                value=float(prefill.get("time_since_first_deliquency", 0)),
                min_value=0.0,
            )
        with c2:
            max_deliq = st.number_input(
                "Max Delinquency Level (12 months)",
                value=int(prefill.get("max_deliq_12mts", 0)),
                min_value=0,
            )
            num_deliq = st.number_input(
                "Times Delinquent (12 months)",
                value=int(prefill.get("num_deliq_12mts", 0)),
                min_value=0,
            )

    with ftab3:
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            age = st.number_input(
                "Age", value=int(prefill.get("AGE", 30)), min_value=18, max_value=80
            )
            income = st.number_input(
                "Net Monthly Income",
                value=float(prefill.get("NETMONTHLYINCOME", 40000.0)),
                min_value=0.0,
            )
        with c2:
            gender = st.selectbox("Gender", ["M", "F"])
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced"]
            )
            education = st.selectbox(
                "Education",
                [
                    "SSC",
                    "12TH",
                    "GRADUATE",
                    "UNDER GRADUATE",
                    "POST-GRADUATE",
                    "OTHERS",
                    "PROFESSIONAL",
                ],
                index=2,
            )
        with c3:
            last_prod = st.selectbox(
                "Last Product Enquiry", ["PL", "CC", "AL", "HL", "others"]
            )
            first_prod = st.selectbox(
                "First Product Enquiry", ["PL", "CC", "AL", "HL", "others"]
            )

    features = {
        "enq_L3m": enq_l3m,
        "enq_L6m": enq_l6m,
        "enq_L12m": enq_l12m,
        "Age_Oldest_TL": age_oldest,
        "Age_Newest_TL": age_newest,
        "Total_TL": tot_tl,
        "time_since_recent_deliquency": time_recent,
        "time_since_first_deliquency": time_first,
        "max_deliq_12mts": max_deliq,
        "num_deliq_12mts": num_deliq,
        "num_std": num_std,
        "num_std_12mts": num_std_12mts,
        "AGE": age,
        "NETMONTHLYINCOME": income,
        "Tot_Missed_Pmnt": tot_missed,
        "GENDER": gender,
        "MARITALSTATUS": marital_status,
        "EDUCATION": education,
        "last_prod_enq2": last_prod,
        "first_prod_enq2": first_prod,
    }

    #  Analyze Button
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1.5, 2, 1.5])
    with btn_col:
        analyze_clicked = st.button(
            "Run Agentic Analysis", use_container_width=True, type="primary"
        )

    if analyze_clicked:
        if not os.environ.get("GROQ_API_KEY"):
            st.error(
                "GROQ_API_KEY not found. Add it to your .env file and restart the app."
            )
        else:
            try:
                with st.spinner(
                    "Running agent  —  Risk Analysis  >  Guideline Retrieval  >  Report Generation  (approx 10s)"
                ):
                    run_agent_cached = get_agent_runner()
                    st.session_state.agent_result = run_agent_cached(
                        prospect_id, features
                    )
            except Exception as e:
                st.session_state.agent_result = {"error": f"Agent failed to start: {e}"}
                st.error(
                    "Agent failed to start. Check terminal logs for dependency or model errors."
                )

    #  Results
    result = st.session_state.agent_result

    if result:
        if result.get("error"):
            st.error(f"Agent error: {result['error']}")
        else:
            report = result.get("report", {})
            summary = report.get("summary", {})
            deep_dive = report.get("deep_dive", {})
            confidence = report.get("confidence", {})

            tier = summary.get("risk_tier", "P2")
            lbl, clr, bg, border = RISK_LEVELS.get(tier, RISK_LEVELS["P2"])

            #  Risk Card
            st.markdown(
                f"""
<div class="risk-result" style="background:{bg}; border:2px solid {border};">
    <div class="risk-label" style="color:{clr};">{lbl}</div>
    <div class="risk-sub">Prospect #{summary.get("prospect_id", prospect_id)} &nbsp;—&nbsp; Tier {tier}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            #  Overview
            st.markdown(
                '<div class="section-header">Risk Overview</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
<div class="overview-box">
    {summary.get("overview", "")}
    <div class="overview-rec">{summary.get("recommendation", "")}</div>
</div>
""",
                unsafe_allow_html=True,
            )

            #  Probability Breakdown
            st.markdown(
                '<div class="section-header">Risk Probability Breakdown</div>',
                unsafe_allow_html=True,
            )

            cards_html = '<div class="cards-grid">'
            for p_tier in ["P1", "P2", "P3", "P4"]:
                p_val = confidence.get(p_tier, 0.0)
                p_lbl, p_clr, _, p_bdr = RISK_LEVELS[p_tier]
                cards_html += f"""
<div class="metric-card" style="border-color:{p_bdr}33;">
    <div class="metric-label" style="color:{p_clr};">{p_lbl}</div>
    <div class="metric-value" style="color:{p_clr};">{p_val * 100:.1f}%</div>
</div>"""
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)

            bars_html = '<div class="prob-section">'
            for p_tier in ["P1", "P2", "P3", "P4"]:
                p_val = confidence.get(p_tier, 0.0)
                p_lbl, p_clr, _, _ = RISK_LEVELS[p_tier]
                w = max(p_val * 100, 2)
                bars_html += f"""
<div class="prob-container">
    <div class="prob-label">{p_lbl}</div>
    <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{w}%; background:{p_clr};">{p_val * 100:.1f}%</div>
    </div>
</div>"""
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

            # Deep Dive
            st.markdown(
                '<div class="section-header">Deep Dive Analysis</div>',
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                st.markdown(
                    "<div style='color:#71717a;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;'>Top Risk Factors</div>",
                    unsafe_allow_html=True,
                )
                factors = deep_dive.get("top_risk_factors", [])
                explanations = deep_dive.get("factor_explanations", [])
                for i, factor in enumerate(factors):
                    exp = explanations[i] if i < len(explanations) else ""
                    st.markdown(
                        f'<div class="factor-item">'
                        f'<div class="factor-title">{i + 1}. {factor}</div>'
                        f'<div class="factor-body">{exp}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with col_right:
                st.markdown(
                    "<div style='color:#71717a;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;'>Suggested Actions</div>",
                    unsafe_allow_html=True,
                )
                for action in deep_dive.get("suggested_actions", []):
                    st.markdown(
                        f'<div class="action-item"><span class="bullet"></span>{action}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='color:#71717a;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;'>Guideline References</div>",
                    unsafe_allow_html=True,
                )
                for ref in deep_dive.get("guideline_references", []):
                    st.markdown(
                        f'<div class="guideline-ref">{ref}</div>',
                        unsafe_allow_html=True,
                    )

            # Retrieved Chunks
            with st.expander("Retrieved Knowledge Base Chunks"):
                chunks = result.get("retrieved_chunks", [])
                if chunks:
                    for i, c in enumerate(chunks, 1):
                        txt = c["text"][:500] + ("..." if len(c["text"]) > 500 else "")
                        st.markdown(
                            f'<div class="chunk-card">'
                            f'<div class="chunk-source">Chunk {i} — {c["source"]}  (distance: {c["distance"]})</div>'
                            f'<div class="chunk-text">{txt}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No chunks retrieved.")

            with st.expander("Raw JSON Report"):
                st.json(report)

            st.markdown(
                """
<div class="app-footer">
    LangGraph agentic workflow &nbsp;·&nbsp; LightGBM + SMOTE classifier &nbsp;·&nbsp; 80.4% accuracy &nbsp;·&nbsp; 100 features<br>
    RAG: ChromaDB + all-MiniLM-L6-v2 &nbsp;·&nbsp; LLM: Groq llama-3.3-70b-versatile
</div>
""",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown(
        """
<div class="hero-wrapper" style="padding:1.5rem 0 2rem;">
    <div class="hero-title" style="font-size:2rem;">System <span>Architecture</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">LangGraph Workflow</div>', unsafe_allow_html=True
    )

    # Flow as styled cards
    for title, body in [
        (
            "Node 1 — Risk Analyzer",
            "Loads LightGBM + SMOTE model (80.4% accuracy, 100 features). Applies the same preprocessing pipeline as training: sentinel replacement, 5 engineered behavioral features, one-hot encoding, column alignment. Outputs risk tier P1–P4, class probabilities, and top-5 feature importances.",
        ),
        (
            "Node 2 — Guideline Retriever (RAG)",
            "Builds a semantic query from the risk tier and top features. Queries ChromaDB (2,931 chunks from 3 RBI/SEBI documents) using all-MiniLM-L6-v2 embeddings with cosine similarity. Returns the 4 most relevant guideline chunks.",
        ),
        (
            "Node 3 — Report Generator",
            "Combines risk output and retrieved guidelines into a structured prompt. Calls Groq API (llama-3.3-70b-versatile) at temperature 0.1 for deterministic, factual output. Enforces JSON-only schema. Model probability scores are injected directly, never trusted from LLM output.",
        ),
    ]:
        st.markdown(
            f"""
<div class="factor-item" style="margin-bottom:0.5rem;">
    <div class="factor-title" style="color:#FBBF24;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;">{title}</div>
    <div class="factor-body" style="margin-top:0.4rem;">{body}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center;color:#3f3f46;font-size:1.1rem;margin:-0.2rem 0;'>&#8595;</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-header">Technology Stack</div>', unsafe_allow_html=True
    )

    stack = [
        ("Workflow Orchestration", "LangGraph"),
        ("LLM", "Groq — llama-3.3-70b-versatile"),
        ("Embeddings", "HuggingFace all-MiniLM-L6-v2 (local)"),
        ("Vector Store", "ChromaDB persistent — 2,931 chunks"),
        ("ML Model", "LightGBM + SMOTE — 80.4% accuracy, 100 features"),
        ("UI", "Streamlit"),
    ]
    cards_html = '<div class="cards-grid" style="grid-template-columns:repeat(3,1fr);">'
    for label, value in stack:
        cards_html += f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value-sm">{value}</div></div>'
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-header">RAG Knowledge Base</div>', unsafe_allow_html=True
    )

    kb_cards = '<div class="cards-grid">'
    for src, pages in [
        ("RBI Basel III Capital Regulations", "328 pages"),
        ("IRACP Norms — Asset Classification", "77 pages"),
        ("SEBI Credit Rating Agencies Circular", "109 pages"),
    ]:
        kb_cards += f'<div class="metric-card"><div class="metric-label">{pages}</div><div class="metric-value-sm">{src}</div></div>'
    kb_cards += '<div class="metric-card"><div class="metric-label">Total Chunks</div><div class="metric-value">2,931</div></div>'
    kb_cards += "</div>"
    st.markdown(kb_cards, unsafe_allow_html=True)
