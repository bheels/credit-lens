"""
SME Loan Default Risk Analyzer - Streamlit Web App
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CreditLens",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #0f172a;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    .risk-score-box {
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .risk-high   { background: #fef2f2; border: 2px solid #ef4444; }
    .risk-medium { background: #fffbeb; border: 2px solid #f59e0b; }
    .risk-low    { background: #f0fdf4; border: 2px solid #22c55e; }

    .risk-score-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
    }

    .risk-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .factor-positive { color: #ef4444; }
    .factor-negative { color: #22c55e; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.5rem 0 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }

    .stButton>button {
        background: #0f172a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton>button:hover {
        background: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Artifacts ───────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load('models/xgb_loan_default.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, explainer

@st.cache_data
def load_artifacts():
    with open('models/feature_names.json') as f:
        features = json.load(f)
    with open('models/state_mapping.json') as f:
        state_map = json.load(f)
    with open('models/model_stats.json') as f:
        stats = json.load(f)
    return features, state_map, stats


# ─── Industry Mapping ─────────────────────────────────────────────────────────

NAICS_MAP = {
    'Agriculture / Forestry / Fishing': 11,
    'Mining / Oil & Gas': 21,
    'Utilities': 22,
    'Construction': 23,
    'Manufacturing': 31,
    'Wholesale Trade': 42,
    'Retail Trade': 44,
    'Transportation & Warehousing': 48,
    'Information / Media / Telecom': 51,
    'Finance & Insurance': 52,
    'Real Estate / Rental': 53,
    'Professional / Scientific Services': 54,
    'Management / Consulting': 55,
    'Administrative / Support Services': 56,
    'Education Services': 61,
    'Health Care / Social Assistance': 62,
    'Arts / Entertainment / Recreation': 71,
    'Food Service / Hospitality': 72,
    'Other Services': 81,
    'Public Administration': 92,
}


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🏦 CreditLens</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine learning-powered SME credit risk scoring with SHAP explainability</p>', unsafe_allow_html=True)

# ─── Try Loading Model ────────────────────────────────────────────────────────

try:
    model, explainer = load_model()
    features, state_map, stats = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"""
    **Model not found.** Run the notebooks first to train and save the model:
    1. `notebooks/01_data_exploration.ipynb`
    2. `notebooks/02_feature_engineering.ipynb`
    3. `notebooks/03_model_training.ipynb`

    Error: `{e}`
    """)


# ─── Sidebar: Model Stats ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### About CreditLens")
    if model_loaded:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", f"{stats['roc_auc']:.3f}")
        with col2:
            st.metric("Avg Precision", f"{stats['avg_precision']:.3f}")

        st.markdown(f"""
        **Algorithm:** XGBoost Classifier  
        **Training samples:** {stats['train_size']:,}  
        **Test samples:** {stats['test_size']:,}  
        **Historical default rate:** {stats['default_rate']:.1%}
        """)

    st.markdown("---")
    st.markdown("### What is this?")
    st.markdown("""
    This tool predicts the probability that a small business loan will default, 
    based on 900,000+ real SBA-backed loans.

    **SHAP values** explain *why* each score was given — which factors increase 
    or decrease risk.
    """)

    st.markdown("---")
    st.markdown("### Data Source")
    st.markdown("[SBA Loan Dataset on Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied)")


# ─── Input Form ───────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Loan Details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=1000, max_value=5_000_000,
        value=150_000, step=5_000, format="%d"
    )
    term = st.slider("Loan Term (months)", min_value=12, max_value=360, value=84, step=12)
    sba_guarantee = st.slider("SBA Guarantee %", min_value=0, max_value=100, value=75)

with col2:
    industry = st.selectbox("Industry", options=list(NAICS_MAP.keys()))
    state = st.selectbox("State", options=sorted(state_map.keys()))
    is_new = st.radio("Business Type", ["Existing Business", "New Business"], horizontal=True)

with col3:
    employees = st.number_input("Number of Employees", min_value=0, max_value=500, value=10)
    jobs_created = st.number_input("Jobs Created", min_value=0, max_value=200, value=2)
    jobs_retained = st.number_input("Jobs Retained", min_value=0, max_value=200, value=8)
    is_franchise = st.checkbox("Is a franchise?")
    low_doc = st.checkbox("Low documentation loan?")
    rev_line = st.checkbox("Revolving line of credit?")

st.markdown("")
predict_btn = st.button("⚡  Analyze Loan Risk")


# ─── Prediction ───────────────────────────────────────────────────────────────

if predict_btn and model_loaded:

    # Build feature vector
    sba_appv = loan_amount * (sba_guarantee / 100)
    total_jobs = jobs_created + jobs_retained
    loan_per_job = loan_amount / max(total_jobs, 1)

    input_data = {
        'Term': term,
        'NoEmp': employees,
        'CreateJob': jobs_created,
        'RetainedJob': jobs_retained,
        'TotalJobs': total_jobs,
        'UrbanRural': 1,
        'RevLineCr': int(rev_line),
        'LowDoc': int(low_doc),
        'DisbursementGross': loan_amount,
        'GrAppv': loan_amount,
        'SBA_Appv': sba_appv,
        'SBA_GuaranteeRatio': sba_guarantee / 100,
        'DisbursementRatio': 1.0,
        'LoanPerJob': min(loan_per_job, 500_000),
        'IsNewBusiness': int(is_new == "New Business"),
        'IsFranchise': int(is_franchise),
        'NAICSCode': NAICS_MAP[industry],
        'StateEncoded': state_map.get(state, 0),
        'ApprovalFY': 2024,
    }

    input_df = pd.DataFrame([input_data])[features]
    risk_score = model.predict_proba(input_df)[0][1]

    # ─── Risk Display ─────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

    result_col, explain_col = st.columns([1, 2])

    with result_col:
        if risk_score >= 0.6:
            risk_class = "risk-high"
            risk_label = "🔴 High Risk"
            risk_color = "#ef4444"
        elif risk_score >= 0.35:
            risk_class = "risk-medium"
            risk_label = "🟡 Medium Risk"
            risk_color = "#f59e0b"
        else:
            risk_class = "risk-low"
            risk_label = "🟢 Low Risk"
            risk_color = "#22c55e"

        st.markdown(f"""
        <div class="risk-score-box {risk_class}">
            <div class="risk-score-number" style="color:{risk_color}">{risk_score:.0%}</div>
            <div class="risk-label">{risk_label}</div>
            <div style="font-size:0.85rem;color:#64748b;margin-top:0.5rem">Probability of Default</div>
        </div>
        """, unsafe_allow_html=True)

        # Benchmark
        baseline = stats['default_rate']
        rel_risk = risk_score / baseline
        st.info(f"**{rel_risk:.1f}x** the historical average default rate ({baseline:.1%})")

    # ─── SHAP Explanation ─────────────────────────────────────────────────────

    with explain_col:
        st.markdown("**Why this score?** Top contributing factors:")

        shap_vals = explainer.shap_values(input_df)[0]
        feature_impact = pd.DataFrame({
            'Feature': features,
            'Value': input_df.values[0],
            'SHAP': shap_vals
        }).sort_values('SHAP', key=abs, ascending=False).head(8)

        # Friendly feature name map
        friendly_names = {
            'Term': 'Loan Term',
            'SBA_GuaranteeRatio': 'SBA Guarantee %',
            'GrAppv': 'Loan Amount',
            'IsNewBusiness': 'New Business',
            'NAICSCode': 'Industry',
            'LowDoc': 'Low Documentation',
            'NoEmp': 'Employees',
            'LoanPerJob': 'Loan Per Job',
            'StateEncoded': 'State',
            'DisbursementRatio': 'Disbursement Ratio',
            'TotalJobs': 'Total Jobs',
            'IsFranchise': 'Franchise',
            'RevLineCr': 'Revolving Credit',
            'CreateJob': 'Jobs Created',
            'RetainedJob': 'Jobs Retained',
        }

        for _, row in feature_impact.iterrows():
            fname = friendly_names.get(row['Feature'], row['Feature'])
            direction = "↑ Increases risk" if row['SHAP'] > 0 else "↓ Decreases risk"
            color = "#ef4444" if row['SHAP'] > 0 else "#22c55e"
            bar_width = min(abs(row['SHAP']) * 100 * 3, 100)

            st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
                    <span style="font-size:0.9rem;font-weight:500">{fname}</span>
                    <span style="font-size:0.8rem;color:{color};font-weight:600">{direction}</span>
                </div>
                <div style="background:#e2e8f0;border-radius:4px;height:6px;">
                    <div style="background:{color};width:{bar_width}%;height:6px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── SHAP Waterfall Plot ──────────────────────────────────────────────────

    st.markdown('<p class="section-header">SHAP Waterfall — Full Breakdown</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap_display = pd.DataFrame({
        'Feature': [friendly_names.get(f, f) for f in features],
        'SHAP Value': shap_vals
    }).sort_values('SHAP Value').tail(12)

    colors = ['#ef4444' if v > 0 else '#22c55e' for v in shap_display['SHAP Value']]
    ax.barh(shap_display['Feature'], shap_display['SHAP Value'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value (impact on default probability)')
    ax.set_title('Feature Impact on This Prediction', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ─── Loan Summary Table ───────────────────────────────────────────────────

    st.markdown('<p class="section-header">Loan Summary</p>', unsafe_allow_html=True)

    summary = {
        "Loan Amount": f"${loan_amount:,}",
        "SBA Guaranteed": f"${sba_appv:,.0f} ({sba_guarantee}%)",
        "Term": f"{term} months ({term//12} years)",
        "Industry": industry,
        "State": state,
        "Employees": employees,
        "Total Jobs": total_jobs,
        "Business Type": is_new,
        "Risk Score": f"{risk_score:.1%}",
        "Risk Level": risk_label,
    }

    col_a, col_b = st.columns(2)
    items = list(summary.items())
    for i, (k, v) in enumerate(items):
        (col_a if i % 2 == 0 else col_b).markdown(f"**{k}:** {v}")


elif predict_btn and not model_loaded:
    st.error("Please train the model first by running the three notebooks in order.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#94a3b8;font-size:0.8rem;">
    CreditLens · Built with XGBoost · SHAP · Streamlit · 
    <a href="https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied" 
       style="color:#94a3b8">SBA Loan Dataset</a>
</div>
""", unsafe_allow_html=True)
