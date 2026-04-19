"""
graph.py

Flow:
    Node 1 (risk_analyzer)       → LightGBM+SMOTE model, risk tier + top features
    Node 2 (guideline_retriever) → RAG: fetch relevant chunks from ChromaDB
    Node 3 (report_generator)    → Groq LLM → structured JSON report
"""

import os

# Ensure thread safety on Apple Silicon before any heavy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import joblib
import numpy as np
import pandas as pd
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from groq import Groq

from retriever import retrieve, build_context_string

load_dotenv()

# Config
MODEL_PATH = "models/finalized_model.joblib"
FEATURES_PATH = "models/feature_columns.joblib"
GROQ_MODEL = "llama-3.3-70b-versatile"


class AgentState(TypedDict):
    prospect_id: str
    features: dict
    risk_tier: Optional[str]
    probabilities: Optional[dict]
    top_risk_factors: Optional[list[str]]
    retrieved_chunks: Optional[list[dict]]
    report: Optional[dict]
    error: Optional[str]


#  Globals
_model = None
_feat_cols = None
_groq_client = None


def _truncate(text: str, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_feat_cols() -> list[str]:
    global _feat_cols
    if _feat_cols is None:
        _feat_cols = joblib.load(FEATURES_PATH)
    return _feat_cols


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Falling back to local report generation.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client



def _build_fallback_report(state: AgentState, note: Optional[str] = None) -> dict:
    """Resilient non-LLM report so the Streamlit app remains deployable and usable."""
    prospect_id = state.get("prospect_id", "UNKNOWN")
    tier = state.get("risk_tier") or "P2"
    probs = state.get("probabilities") or {"P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0}
    chunks = state.get("retrieved_chunks") or []
    factors = (state.get("top_risk_factors") or ["No significant model factors available."])[:3]

    severity_map = {"P1": "Very Low", "P2": "Low", "P3": "Medium", "P4": "High"}
    highest_band = max(probs, key=probs.get) if probs else tier

    recommendation_map = {
        "P1": "Recommendation: Proceed with standard underwriting checks.",
        "P2": "Recommendation: Proceed with normal review and verify current income and obligations.",
        "P3": "Recommendation: Escalate for analyst review before approval and validate recent repayment behavior.",
        "P4": "Recommendation: Apply tight underwriting controls or decline unless strong compensating factors are documented.",
    }
    action_map = {
        "P1": [
            "Complete standard KYC and bureau verification.",
            "Confirm current employment and income continuity.",
            "Monitor exposure limits before sanction.",
        ],
        "P2": [
            "Verify declared income and existing repayment burden.",
            "Review recent enquiries and any fresh credit lines.",
            "Approve within standard policy limits if checks remain clean.",
        ],
        "P3": [
            "Review bank statements and repayment discipline for the last 6–12 months.",
            "Ask for additional income or collateral support where policy permits.",
            "Route the file to manual credit review before final decision.",
        ],
        "P4": [
            "Investigate delinquencies and overdue patterns in detail.",
            "Require senior credit approval and strong mitigants if the case is pursued.",
            "Consider decline when repayment stress cannot be reasonably explained.",
        ],
    }

    guideline_references = []
    for chunk in chunks[:3]:
        src = chunk.get("source", "unknown")
        txt = _truncate(chunk.get("text", ""))
        if txt:
            guideline_references.append(f"{src}: {txt}")
    if not guideline_references:
        guideline_references = ["Insufficient guideline context."]

    factor_explanations = [
        f"The model identified {factor} as a meaningful contributor to the current risk assessment."
        for factor in factors
    ]

    overview = (
        f"Prospect {prospect_id} is assessed as {tier} ({severity_map.get(tier, 'Unknown')}). "
        f"The model assigns its highest confidence to {highest_band} at {round(float(probs.get(highest_band, 0.0)) * 100, 1)}%. "
        f"Retrieved policy context was used to keep the recommendation grounded in the available guidance."
    )
    if note:
        overview += f" {note}"

    return {
        "summary": {
            "prospect_id": prospect_id,
            "risk_tier": tier,
            "severity": severity_map.get(tier, "Low"),
            "recommendation": recommendation_map.get(tier, recommendation_map["P2"]),
            "overview": overview,
        },
        "deep_dive": {
            "top_risk_factors": factors,
            "factor_explanations": factor_explanations,
            "guideline_references": guideline_references,
            "suggested_actions": action_map.get(tier, action_map["P2"]),
        },
        "confidence": probs,
    }


#  Preprocessing
CAT_COLS = ["MARITALSTATUS", "EDUCATION", "GENDER", "last_prod_enq2", "first_prod_enq2"]

DELINQUENCY_COLS = [
    "time_since_first_deliquency",
    "time_since_recent_deliquency",
    "max_delinquency_level",
    "max_deliq_6mts",
    "max_deliq_12mts",
    "max_unsec_exposure_inPct",
]

DROP_COLS = [
    "CC_utilization",
    "PL_utilization",
    "Credit_Score",
    "PROSPECTID",
    "PROSPECT_ID",
]


def _safe(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0


def preprocess_features(raw: dict, train_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df.replace(-99999, np.nan, inplace=True)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    for col in DELINQUENCY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Engineered features — must match training pipeline
    df["delinq_burden"] = _safe(df, "num_times_delinquent") * _safe(
        df, "max_delinquency_level"
    )
    df["enq_pressure"] = _safe(df, "enq_L3m") / (_safe(df, "enq_L12m") + 1)
    df["account_health"] = _safe(df, "num_std") / (
        _safe(df, "num_sub") + _safe(df, "num_dbt") + _safe(df, "num_lss") + 1
    )
    df["income_stability"] = _safe(df, "NETMONTHLYINCOME") * np.log1p(
        _safe(df, "Time_With_Curr_Empr")
    )
    df["recency_risk"] = 1 / (_safe(df, "time_since_recent_deliquency") + 1)

    present_cats = [c for c in CAT_COLS if c in df.columns]
    if present_cats:
        df = pd.get_dummies(df, columns=present_cats, drop_first=True)

    df = df.reindex(columns=train_columns, fill_value=0)
    return df


#  Node 1: Risk Analyzer
def risk_analyzer(state: AgentState) -> AgentState:
    try:
        model = _get_model()
        train_cols = _get_feat_cols()
        X = preprocess_features(state["features"], train_cols)

        # TOTAL BYPASS FOR APPLE SILICON CRASH:
        # We will use the model's Booster directly to skip the buggy C++ predict_np2d
        # path and use the thread-safe array-based path.
        X_np = X.to_numpy().astype(np.float32)

        # Use Booster.predict which is the lowest-level stable API
        booster = model.booster_
        proba = booster.predict(X_np)[0]

        # In case it returns single value (unlikely for 4 classes, but safe-coding)
        if isinstance(proba, (float, np.float32, np.float64)):
            # Handle binary case if it ever happens
            proba = [1 - proba, proba]

        pred = np.argmax(proba)

        tier = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}.get(int(pred), str(pred))
        probs = {f"P{i + 1}": round(float(proba[i]), 3) for i in range(4)}

        top_factors = []
        # Accessing feature_importances_ is safe
        if hasattr(model, "feature_importances_"):
            feat_series = pd.Series(model.feature_importances_, index=train_cols)
            for feat, imp in feat_series.nlargest(5).items():
                val = state["features"].get(feat, "N/A")
                top_factors.append(f"{feat} = {val}  (importance: {imp:.3f})")
        else:
            for f in [
                "enq_L3m",
                "Age_Oldest_TL",
                "time_since_recent_deliquency",
                "num_std_12mts",
                "Tot_Missed_Pmnt",
            ]:
                if f in state["features"]:
                    top_factors.append(f"{f} = {state['features'][f]}")

        return {
            **state,
            "risk_tier": tier,
            "probabilities": probs,
            "top_risk_factors": top_factors,
            "error": None,
        }

    except Exception as e:
        return {**state, "error": f"risk_analyzer failed: {e}"}


#  Node 2: Guideline Retriever
def guideline_retriever(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        query = " ".join(
            [f"credit risk {state['risk_tier']} borrower lending guidelines"]
            + (state.get("top_risk_factors") or [])[:3]
        )
        return {**state, "retrieved_chunks": retrieve(query)}
    except Exception as e:
        return {**state, "error": f"guideline_retriever failed: {e}"}


#  Node 3: Report Generator
SYSTEM_PROMPT = """You are a credit risk analyst AI at a bank.
Your job is to generate a structured risk assessment report for a loan prospect.

STRICT RULES:
1. Only use facts from the PROVIDED GUIDELINES. Do not invent regulations or statistics.
2. If guidelines don't cover a point, say "Insufficient guideline context."
3. Respond ONLY with a valid JSON object — no markdown, no preamble, no explanation.
4. All string values must be factual and concise.

JSON SCHEMA (respond exactly in this shape):
{
  "summary": {
    "prospect_id": "string",
    "risk_tier": "P1|P2|P3|P4",
    "severity": "Very Low|Low|Medium|High",
    "recommendation": "string (1 sentence action)",
    "overview": "string (2-3 sentences explaining the risk profile)"
  },
  "deep_dive": {
    "top_risk_factors": ["string", "string", "string"],
    "factor_explanations": ["string explaining each factor above"],
    "guideline_references": ["exact quote or paraphrase from retrieved guidelines"],
    "suggested_actions": ["string", "string", "string"]
  },
  "confidence": {
    "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0
  }
}"""


def report_generator(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    raw_text = ""
    try:
        client = _get_groq_client()
        context = build_context_string(state.get("retrieved_chunks", []))
        tier = state["risk_tier"]
        probs = state["probabilities"]
        factors = state.get("top_risk_factors", [])
        severity_map = {"P1": "Very Low", "P2": "Low", "P3": "Medium", "P4": "High"}

        user_message = f"""Prospect ID: {state["prospect_id"]}
Risk Tier: {tier} ({severity_map.get(tier, "Unknown")} Risk)
Model Probabilities: {json.dumps(probs)}

Top Risk Factors from ML Model:
{chr(10).join(f"- {f}" for f in factors)}

RETRIEVED GUIDELINES (use ONLY these for recommendations):
{context}

Generate the risk assessment report in the JSON schema specified."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        report = json.loads(raw_text)
        report["confidence"] = probs  # always use model probs, not LLM's
        return {**state, "report": report, "error": None}

    except json.JSONDecodeError:
        fallback = _build_fallback_report(
            state,
            note="The hosted LLM response could not be parsed, so a local structured fallback report was generated.",
        )
        return {**state, "report": fallback, "error": None}
    except Exception as e:
        fallback = _build_fallback_report(
            state,
            note=f"The hosted LLM was unavailable ({e}), so a local structured fallback report was generated.",
        )
        return {**state, "report": fallback, "error": None}


#  Graph
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("risk_analyzer", risk_analyzer)
    g.add_node("guideline_retriever", guideline_retriever)
    g.add_node("report_generator", report_generator)
    g.set_entry_point("risk_analyzer")
    g.add_edge("risk_analyzer", "guideline_retriever")
    g.add_edge("guideline_retriever", "report_generator")
    g.add_edge("report_generator", END)
    return g.compile()


_compiled_graph = None


def run_agent(prospect_id: str, features: dict) -> dict:
    """Entry point called from agent_app.py."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    return _compiled_graph.invoke(
        {
            "prospect_id": prospect_id,
            "features": features,
            "risk_tier": None,
            "probabilities": None,
            "top_risk_factors": None,
            "retrieved_chunks": None,
            "report": None,
            "error": None,
        }
    )
