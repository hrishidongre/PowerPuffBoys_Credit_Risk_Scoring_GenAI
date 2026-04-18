import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Page Configuration
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS — FinNuvora-inspired Black/Gold Theme
# ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ══════════ Global ══════════ */
    .stApp {
        background: #0a0a0a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #a1a1aa;
    }
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* ══════════ Sidebar ══════════ */
    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #a1a1aa;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ══════════ Main Content Area ══════════ */
    .block-container {
        padding: 3rem 4rem 4rem !important;
        max-width: 1200px;
    }

    /* ══════════ Hero ══════════ */
    .hero-wrapper {
        text-align: center;
        padding: 2.5rem 0 3rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-bottom: 0.6rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #FBBF24, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: #71717a;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.6;
    }

    /* ══════════ Section Headers ══════════ */
    .section-header {
        font-size: 0.82rem;
        font-weight: 700;
        color: #FBBF24;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 3rem 0 1.6rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ══════════ Metric Cards ══════════ */
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #161616;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(251,191,36,0.3);
        background: #1a1a1a;
        box-shadow: 0 8px 30px rgba(251,191,36,0.04);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.65rem;
        color: #52525b;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f4f4f5;
        letter-spacing: -0.02em;
    }

    /* ══════════ Tabs ══════════ */
    .stTabs {
        margin-top: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        padding: 0.4rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #161616;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        color: #71717a;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.65rem 1.3rem;
        letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #a1a1aa;
        border-color: rgba(255,255,255,0.12);
    }
    .stTabs [aria-selected="true"] {
        background: #1c1c1c !important;
        color: #FBBF24 !important;
        border-color: #FBBF24 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.8rem 0 0.5rem;
    }

    /* ══════════ Input Widgets ══════════ */
    .stNumberInput label, .stSlider label {
        color: #a1a1aa !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stNumberInput > div {
        margin-bottom: 1.2rem;
    }
    .stSlider > div {
        margin-bottom: 1rem;
    }

    /* ══════════ Primary Button ══════════ */
    .stButton > button[kind="primary"] {
        background: #FBBF24 !important;
        color: #0a0a0a !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 50px !important;
        padding: 0.85rem 3rem !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 20px rgba(251,191,36,0.15) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #F59E0B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(251,191,36,0.25) !important;
    }

    /* ══════════ Secondary Button ══════════ */
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        color: #a1a1aa !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #FBBF24 !important;
        color: #FBBF24 !important;
    }

    /* ══════════ Risk Result ══════════ */
    .risk-result {
        border-radius: 20px;
        padding: 3rem 2.5rem;
        text-align: center;
        margin: 2.5rem auto;
        max-width: 520px;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.96) translateY(8px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }
    .risk-label {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }
    .risk-sub {
        font-size: 0.85rem;
        color: #71717a;
        font-weight: 400;
    }

    /* ══════════ Probability Bars ══════════ */
    .prob-section {
        max-width: 700px;
        margin: 0 auto;
        padding: 0.5rem 0 1rem;
    }
    .prob-container {
        margin: 0.75rem 0;
    }
    .prob-bar-bg {
        background: #161616;
        border-radius: 8px;
        height: 32px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
        display: flex;
        align-items: center;
        padding-left: 12px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #0a0a0a;
    }
    .prob-label {
        font-size: 0.72rem;
        color: #71717a;
        margin-bottom: 0.35rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ══════════ Dividers ══════════ */
    .styled-divider {
        height: 1px;
        background: rgba(255,255,255,0.04);
        margin: 2rem 0;
    }

    /* ══════════ Sidebar Brand ══════════ */
    .sidebar-brand {
        text-align: center;
        padding: 1.8rem 0 1rem;
    }
    .sidebar-brand-icon {
        width: 48px;
        height: 48px;
        background: #FBBF24;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        font-weight: 900;
        color: #0a0a0a;
        margin-bottom: 0.8rem;
    }
    .sidebar-brand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f4f4f5;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.7rem;
        color: #52525b;
        margin-top: 0.25rem;
    }

    /* ══════════ Footer ══════════ */
    .app-footer {
        text-align: center;
        color: #3f3f46;
        font-size: 0.72rem;
        padding: 2rem 0;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)



# Data & Model Loading
# ──────────────────────────────────────────────────
@st.cache_data
def load_cibil():
    return pd.read_csv("./Dataset/Unseen_CIBL_Data.csv")

@st.cache_resource
def load_model():
    return joblib.load("./models/finalized_model.joblib")

cibil_df = load_cibil()
model = load_model()
FEATURES = joblib.load("./models/feature_columns.joblib")



# Labels & Configuration
# ──────────────────────────────────────────────────
LABELS = {
    "Total_TL":             "Total Trade Lines",
    "Tot_Closed_TL":        "Closed Accounts",
    "Tot_Active_TL":        "Active Accounts",
    "Total_TL_opened_L6M":  "Opened (Last 6 Mo)",
    "Tot_TL_closed_L6M":    "Closed (Last 6 Mo)",
    "pct_tl_open_L6M":      "% Opened (Last 6 Mo)",
    "pct_tl_closed_L6M":    "% Closed (Last 6 Mo)",
    "pct_active_tl":        "% Active Accounts",
    "pct_closed_tl":        "% Closed Accounts",
    "Total_TL_opened_L12M": "Opened (Last 12 Mo)",
    "Tot_TL_closed_L12M":   "Closed (Last 12 Mo)",
    "pct_tl_open_L12M":     "% Opened (Last 12 Mo)",
    "pct_tl_closed_L12M":   "% Closed (Last 12 Mo)",
    "Tot_Missed_Pmnt":      "Missed Payments",
    "Auto_TL":              "Auto Loans",
    "CC_TL":                "Credit Cards",
    "Consumer_TL":          "Consumer Loans",
    "Gold_TL":              "Gold Loans",
    "Home_TL":              "Home Loans",
    "PL_TL":                "Personal Loans",
    "Secured_TL":           "Secured Accounts",
    "Unsecured_TL":         "Unsecured Accounts",
    "Other_TL":             "Other Accounts",
    "Age_Oldest_TL":        "Oldest Account Age (Mo)",
    "Age_Newest_TL":        "Newest Account Age (Mo)",
}

RANGES = {
    "Total_TL": (0, 25), "Tot_Closed_TL": (0, 25), "Tot_Active_TL": (0, 25),
    "Total_TL_opened_L6M": (0, 10), "Tot_TL_closed_L6M": (0, 10),
    "Total_TL_opened_L12M": (0, 20), "Tot_TL_closed_L12M": (0, 20),
    "Tot_Missed_Pmnt": (0, 100),
    "Auto_TL": (0, 10), "CC_TL": (0, 10), "Consumer_TL": (0, 20),
    "Gold_TL": (0, 10), "Home_TL": (0, 10), "PL_TL": (0, 10),
    "Secured_TL": (0, 20), "Unsecured_TL": (0, 20), "Other_TL": (0, 10),
    "Age_Oldest_TL": (0, 500), "Age_Newest_TL": (0, 300),
}

DEFAULTS = {
    "Total_TL": 8, "Tot_Closed_TL": 3, "Tot_Active_TL": 5,
    "Total_TL_opened_L6M": 1, "Tot_TL_closed_L6M": 0,
    "pct_tl_open_L6M": 0.15, "pct_tl_closed_L6M": 0.05,
    "pct_active_tl": 0.65, "pct_closed_tl": 0.35,
    "Total_TL_opened_L12M": 2, "Tot_TL_closed_L12M": 1,
    "pct_tl_open_L12M": 0.25, "pct_tl_closed_L12M": 0.10,
    "Tot_Missed_Pmnt": 0,
    "Auto_TL": 1, "CC_TL": 3, "Consumer_TL": 2, "Gold_TL": 1,
    "Home_TL": 1, "PL_TL": 0,
    "Secured_TL": 4, "Unsecured_TL": 4, "Other_TL": 0,
    "Age_Oldest_TL": 120, "Age_Newest_TL": 12,
}

RISK_LEVELS = {
    0: ("VERY LOW RISK",  "#22C55E"),
    1: ("LOW RISK",       "#FBBF24"),
    2: ("MEDIUM RISK",    "#F59E0B"),
    3: ("HIGH RISK",      "#EF4444"),
}
RISK_BG = { 0: "#0d1f12", 1: "#1f1a0a", 2: "#1f180a", 3: "#1f0d0d" }
RISK_BORDER = { 0: "#22C55E", 1: "#FBBF24", 2: "#F59E0B", 3: "#EF4444" }



# Session State
# ──────────────────────────────────────────────────
if "selected_prospect" not in st.session_state:
    st.session_state.selected_prospect = cibil_df["PROSPECT_ID"].values[0]
if "trade_inputs" not in st.session_state:
    st.session_state.trade_inputs = DEFAULTS.copy()



# SIDEBAR
# ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">CR</div>
        <div class="sidebar-brand-name">Credit Risk Engine</div>
        <div class="sidebar-brand-sub">LightGBM + SMOTE Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    prospect_ids = cibil_df["PROSPECT_ID"].unique()
    st.session_state.selected_prospect = st.selectbox(
        "Select Prospect",
        prospect_ids,
        index=list(prospect_ids).index(st.session_state.selected_prospect),
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Random", use_container_width=True):
            st.session_state.selected_prospect = np.random.choice(prospect_ids)
            st.rerun()
    with col_b:
        if st.button("Reset", use_container_width=True):
            st.session_state.trade_inputs = DEFAULTS.copy()
            st.rerun()

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.68rem; color:#3f3f46; text-align:center; padding:1rem 0; line-height:1.7;">
        <span style="color:#71717a; font-weight:600;">Team PowerPuff Boys</span><br>
        Behavioral Credit Risk Classification<br>
        using Classical Machine Learning
    </div>
    """, unsafe_allow_html=True)



# HERO
# ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-title">Credit Risk <span>Scoring Engine</span></div>
    <div class="hero-subtitle">
        Behavioral credit risk classification powered by machine learning
    </div>
</div>
""", unsafe_allow_html=True)



# PROSPECT OVERVIEW
# ──────────────────────────────────────────────────
selected_row = cibil_df[cibil_df["PROSPECT_ID"] == st.session_state.selected_prospect].iloc[0]

st.markdown(
    '<div class="section-header">Prospect Overview  /  ID #'
    + str(st.session_state.selected_prospect)
    + '</div>',
    unsafe_allow_html=True,
)

# Row 1
st.markdown(f"""
<div class="cards-grid">
    <div class="metric-card">
        <div class="metric-label">Gender</div>
        <div class="metric-value">{selected_row.get("GENDER", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Marital Status</div>
        <div class="metric-value">{selected_row.get("MARITALSTATUS", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Education</div>
        <div class="metric-value">{selected_row.get("EDUCATION", "N/A")}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Monthly Income</div>
        <div class="metric-value">Rs. {int(selected_row.get("NETMONTHLYINCOME", 0)):,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 2
st.markdown(f"""
<div class="cards-grid">
    <div class="metric-card">
        <div class="metric-label">Employer Tenure</div>
        <div class="metric-value">{int(selected_row.get("Time_With_Curr_Empr", 0))} mo</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Credit Cards</div>
        <div class="metric-value">{int(selected_row.get("CC_TL", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Personal Loans</div>
        <div class="metric-value">{int(selected_row.get("PL_TL", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Home Loans</div>
        <div class="metric-value">{int(selected_row.get("Home_TL", 0))}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 3
st.markdown(f"""
<div class="cards-grid">
    <div class="metric-card">
        <div class="metric-label">Recent Enquiries (3 Mo)</div>
        <div class="metric-value">{int(selected_row.get("enq_L3m", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Missed Payments</div>
        <div class="metric-value">{int(selected_row.get("Tot_Missed_Pmnt", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Secured Accounts</div>
        <div class="metric-value">{int(selected_row.get("Secured_TL", 0))}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Unsecured Accounts</div>
        <div class="metric-value">{int(selected_row.get("Unsecured_TL", 0))}</div>
    </div>
</div>
""", unsafe_allow_html=True)



# TRADE LINE INPUTS
# ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Bureau Trade Line Adjustments</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Account Summary",
    "Recent Activity",
    "Account Types",
    "Age & Security",
])


def render_input(feature):
    label = LABELS.get(feature, feature)
    if "pct" in feature:
        st.session_state.trade_inputs[feature] = st.slider(
            label, 0.0, 1.0,
            float(st.session_state.trade_inputs[feature]),
            0.01, key=feature,
        )
    else:
        mn, mx = RANGES[feature]
        st.session_state.trade_inputs[feature] = st.number_input(
            label,
            min_value=float(mn), max_value=float(mx),
            value=float(st.session_state.trade_inputs[feature]),
            key=feature,
        )


with tab1:
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        render_input("Total_TL")
        render_input("pct_active_tl")
    with c2:
        render_input("Tot_Active_TL")
        render_input("pct_closed_tl")
    with c3:
        render_input("Tot_Closed_TL")
        render_input("Tot_Missed_Pmnt")

with tab2:
    st.markdown("**Last 6 Months**")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: render_input("Total_TL_opened_L6M")
    with c2: render_input("Tot_TL_closed_L6M")
    with c3: render_input("pct_tl_open_L6M")
    with c4: render_input("pct_tl_closed_L6M")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("**Last 12 Months**")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: render_input("Total_TL_opened_L12M")
    with c2: render_input("Tot_TL_closed_L12M")
    with c3: render_input("pct_tl_open_L12M")
    with c4: render_input("pct_tl_closed_L12M")

with tab3:
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        render_input("CC_TL")
        render_input("PL_TL")
        render_input("Auto_TL")
    with c2:
        render_input("Consumer_TL")
        render_input("Gold_TL")
        render_input("Home_TL")
    with c3:
        render_input("Other_TL")

with tab4:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_input("Age_Oldest_TL")
        render_input("Secured_TL")
    with c2:
        render_input("Age_Newest_TL")
        render_input("Unsecured_TL")

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)



# PREDICT
# ──────────────────────────────────────────────────
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    predict_clicked = st.button("Predict Credit Risk", use_container_width=True, type="primary")

if predict_clicked:
    final_input = selected_row.to_dict()
    for key, value in st.session_state.trade_inputs.items():
        final_input[key] = value

    input_df = pd.DataFrame([final_input])
    input_df.replace(-99999, np.nan, inplace=True)

    for col in ["time_since_first_deliquency", "time_since_recent_deliquency",
                 "max_delinquency_level", "max_deliq_6mts",
                 "max_deliq_12mts", "max_unsec_exposure_inPct"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(0)

    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_cols] = input_df[numeric_cols].fillna(input_df[numeric_cols].median())

    # Engineered features
    def _get(col):
        return input_df[col] if col in input_df.columns else 0

    input_df['delinq_burden']    = _get('num_times_delinquent') * _get('max_delinquency_level')
    input_df['enq_pressure']     = _get('enq_L3m') / (_get('enq_L12m') + 1)
    input_df['account_health']   = _get('num_std') / (_get('num_sub') + _get('num_dbt') + _get('num_lss') + 1)
    input_df['income_stability'] = _get('NETMONTHLYINCOME') * np.log1p(_get('Time_With_Curr_Empr'))
    input_df['recency_risk']     = 1 / (_get('time_since_recent_deliquency') + 1)

    cat_cols = ["MARITALSTATUS", "EDUCATION", "GENDER", "last_prod_enq2", "first_prod_enq2"]
    input_df = pd.get_dummies(
        input_df,
        columns=[c for c in cat_cols if c in input_df.columns],
        drop_first=True,
    )
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[FEATURES]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    risk_label, risk_color = RISK_LEVELS[prediction]

    # ── Result Card ──
    st.markdown(f"""
    <div class="risk-result" style="background:{RISK_BG[prediction]}; border:2px solid {RISK_BORDER[prediction]};">
        <div class="risk-label" style="color:{risk_color};">{risk_label}</div>
        <div class="risk-sub">Prospect #{st.session_state.selected_prospect}  —  Predicted Risk Classification</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Cards ──
    st.markdown('<div class="section-header">Risk Probability Breakdown</div>', unsafe_allow_html=True)

    cards = '<div class="cards-grid">'
    for i, prob in enumerate(probabilities):
        cls = model.classes_[i]
        lbl, clr = RISK_LEVELS[cls]
        cards += f'''
        <div class="metric-card" style="border-color:{RISK_BORDER[cls]};">
            <div class="metric-label" style="color:{clr};">{lbl}</div>
            <div class="metric-value" style="color:{clr};">{prob*100:.1f}%</div>
        </div>'''
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    # ── Probability Bars ──
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    bars_html = '<div class="prob-section">'
    for i, prob in enumerate(probabilities):
        cls = model.classes_[i]
        lbl, clr = RISK_LEVELS[cls]
        w = max(prob * 100, 2)
        bars_html += f"""
        <div class="prob-container">
            <div class="prob-label">{lbl}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{w}%; background:{clr};">{prob*100:.1f}%</div>
            </div>
        </div>"""
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="app-footer">
        Prediction powered by LightGBM + SMOTE classifier<br>
        Trained on 51,336 records · 80.4% accuracy · 100 features
    </div>
    """, unsafe_allow_html=True)