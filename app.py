import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_data
def load_cibil():
    return pd.read_csv("./Dataset/Unseen_CIBL_Data.csv")

@st.cache_resource
def load_model():
    return joblib.load("./models/finalized_model.joblib")

cibil_df = load_cibil()
model = load_model()
FEATURES = list(model.feature_names_in_)


TRADE_LINE_FEATURES = [
    "Total_TL","Tot_Closed_TL","Tot_Active_TL",
    "Total_TL_opened_L6M","Tot_TL_closed_L6M",
    "pct_tl_open_L6M","pct_tl_closed_L6M",
    "pct_active_tl","pct_closed_tl",
    "Total_TL_opened_L12M","Tot_TL_closed_L12M",
    "pct_tl_open_L12M","pct_tl_closed_L12M",
    "Tot_Missed_Pmnt",
    "Auto_TL","CC_TL","Consumer_TL","Gold_TL",
    "Home_TL","PL_TL","Secured_TL","Unsecured_TL",
    "Other_TL","Age_Oldest_TL","Age_Newest_TL"
]

RANGES = {
    "Total_TL": (0, 25),
    "Tot_Closed_TL": (0, 25),
    "Tot_Active_TL": (0, 25),
    "Total_TL_opened_L6M": (0, 10),
    "Tot_TL_closed_L6M": (0, 10),
    "Total_TL_opened_L12M": (0, 20),
    "Tot_TL_closed_L12M": (0, 20),
    "Tot_Missed_Pmnt": (0, 100),
    "Auto_TL": (0, 10),
    "CC_TL": (0, 10),
    "Consumer_TL": (0, 20),
    "Gold_TL": (0, 10),
    "Home_TL": (0, 10),
    "PL_TL": (0, 10),
    "Secured_TL": (0, 20),
    "Unsecured_TL": (0, 20),
    "Other_TL": (0, 10),
    "Age_Oldest_TL": (0, 500),
    "Age_Newest_TL": (0, 300)
}

DEFAULTS = {
    "Total_TL": 8,
    "Tot_Closed_TL": 3,
    "Tot_Active_TL": 5,
    "Total_TL_opened_L6M": 1,
    "Tot_TL_closed_L6M": 0,
    "pct_tl_open_L6M": 0.15,
    "pct_tl_closed_L6M": 0.05,
    "pct_active_tl": 0.65,
    "pct_closed_tl": 0.35,
    "Total_TL_opened_L12M": 2,
    "Tot_TL_closed_L12M": 1,
    "pct_tl_open_L12M": 0.25,
    "pct_tl_closed_L12M": 0.10,
    "Tot_Missed_Pmnt": 0,
    "Auto_TL": 1,
    "CC_TL": 3,
    "Consumer_TL": 2,
    "Gold_TL": 1,
    "Home_TL": 1,
    "PL_TL": 0,
    "Secured_TL": 4,
    "Unsecured_TL": 4,
    "Other_TL": 0,
    "Age_Oldest_TL": 120,
    "Age_Newest_TL": 12
}

RISK_LEVELS = {
    0: ("Very Low Risk", "#2ecc71"),
    1: ("Low Risk", "#3498db"),
    2: ("Medium Risk", "#f39c12"),
    3: ("High Risk", "#e74c3c")
}


if "selected_prospect" not in st.session_state:
    st.session_state.selected_prospect = np.random.choice(cibil_df["PROSPECT_ID"].values)

if "trade_inputs" not in st.session_state:
    st.session_state.trade_inputs = DEFAULTS.copy()

st.title("Intelligent Credit Risk Scoring Engine")

col1, col2 = st.columns([3, 1])

with col1:
    prospect_ids = cibil_df["PROSPECT_ID"].unique()
    st.session_state.selected_prospect = st.selectbox(
        "Select Prospect ID",
        prospect_ids,
        index=list(prospect_ids).index(st.session_state.selected_prospect)
    )
    st.write(cibil_df.loc[st.session_state.selected_prospect])

with col2:
    if st.button("Randomize Prospect", use_container_width=True):
        st.session_state.selected_prospect = np.random.choice(prospect_ids)
        st.rerun()


st.subheader("Bureau Trade Line Inputs")

if st.button("Reset to Defaults", use_container_width=True):
    st.session_state.trade_inputs = DEFAULTS.copy()
    st.rerun()

# Render inputs
for feature in TRADE_LINE_FEATURES:
    if "pct" in feature:
        st.session_state.trade_inputs[feature] = st.slider(
            feature,
            0.0, 1.0,
            float(st.session_state.trade_inputs[feature]),
            0.01,
            key=feature
        )
    else:
        min_val, max_val = RANGES[feature]
        
        st.session_state.trade_inputs[feature] = st.number_input(
            feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state.trade_inputs[feature]),
            key=feature
        )


if st.button("Predict Risk", use_container_width=True, type="primary"):
    selected_row = cibil_df[cibil_df["PROSPECT_ID"] == st.session_state.selected_prospect].iloc[0]
    final_input = selected_row.to_dict()
    
    for key, value in st.session_state.trade_inputs.items():
        final_input[key] = value
    
    input_df = pd.DataFrame([final_input])
    
    input_df.replace(-99999, np.nan, inplace=True)
    
    delinquency_cols = [
        "time_since_first_deliquency",
        "time_since_recent_deliquency",
        "max_delinquency_level",
        "max_deliq_6mts",
        "max_deliq_12mts",
        "max_unsec_exposure_inPct"
    ]
    
    for col in delinquency_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna(0)
    
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    input_df[numeric_cols] = input_df[numeric_cols].fillna(input_df[numeric_cols].median())
    
    cat_cols = [
        "MARITALSTATUS","EDUCATION","GENDER",
        "last_prod_enq2","first_prod_enq2"
    ]
    
    input_df = pd.get_dummies(
        input_df,
        columns=[c for c in cat_cols if c in input_df.columns],
        drop_first=True
    )
    
    for col in FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[FEATURES]
    
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    risk_label, color = RISK_LEVELS[prediction]
    
    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:{color};
            color:white;
            font-size:20px;
            font-weight:bold;
            text-align:center;">
            {risk_label}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("**Risk Probabilities:**")
    for i, prob in enumerate(probabilities):
        label, _ = RISK_LEVELS[model.classes_[i]]
        st.write(f"{label}: {prob:.4f}")