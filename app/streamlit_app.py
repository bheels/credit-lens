"""
CreditLens - SME loan default risk app.
Run with: streamlit run app/streamlit_app.py
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

matplotlib.use('Agg')


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CreditLens",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --ink:        #0f172a;
        --ink-soft:   #334155;
        --muted:      #64748b;
        --hairline:   #e5e7eb;
        --surface:    #ffffff;
        --canvas:     #fafafa;
        --accent:     #0f172a;
        --red:        #dc2626;
        --amber:      #d97706;
        --green:      #16a34a;
    }

    html, body, [class*="css"], .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: var(--ink);
    }

    .stApp {
        background-color: var(--canvas);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .brand {
        font-family: 'Inter', sans-serif;
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.1;
    }

    .brand-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-left: 0.85rem;
        vertical-align: 0.32rem;
    }

    .subtitle {
        font-size: 1rem;
        color: var(--muted);
        margin: 0.4rem 0 2rem;
        font-weight: 400;
    }

    /* Section labels */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--ink-soft);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin: 2rem 0 1rem;
    }

    /* Risk score card */
    .risk-card {
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 14px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: box-shadow 0.25s ease;
    }

    .risk-card:hover {
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.07);
    }

    .risk-card.risk-high   { border-top: 3px solid var(--red); }
    .risk-card.risk-medium { border-top: 3px solid var(--amber); }
    .risk-card.risk-low    { border-top: 3px solid var(--green); }

    .risk-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.4rem;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -0.02em;
    }

    .risk-label {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.65rem;
        letter-spacing: 0.02em;
    }

    .risk-caption {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }

    /* Factor rows */
    .factor-row {
        margin-bottom: 0.85rem;
    }
    .factor-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .factor-name {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--ink);
    }
    .factor-dir {
        font-size: 0.78rem;
        font-weight: 600;
    }
    .factor-bar {
        background: #f1f5f9;
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
    }
    .factor-bar-fill {
        height: 6px;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    /* Summary list */
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 0.55rem 0;
        border-bottom: 1px solid var(--hairline);
        font-size: 0.93rem;
    }
    .summary-row:last-child {
        border-bottom: none;
    }
    .summary-key { color: var(--muted); }
    .summary-val { color: var(--ink); font-weight: 500; }

    /* Button */
    .stButton>button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        width: 100%;
        letter-spacing: 0.01em;
        transition: background 0.2s ease, transform 0.05s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.1);
    }
    .stButton>button:hover {
        background: #1e293b;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--hairline);
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--ink-soft);
        margin-top: 1.2rem;
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer { visibility: hidden; }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--hairline);
    }
    .footer a { color: var(--muted); text-decoration: none; }
    .footer a:hover { color: var(--ink); }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Load model + artifacts ───────────────────────────────────────────────────

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


# ─── Reference data ───────────────────────────────────────────────────────────

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

FRIENDLY_NAMES = {
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


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 class="brand">CreditLens<span class="brand-tag">SBA risk model</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Estimate the probability a small-business loan defaults, '
    'and see which loan attributes drove the score.</p>',
    unsafe_allow_html=True,
)


# ─── Model load ───────────────────────────────────────────────────────────────

try:
    model, explainer = load_model()
    features, state_map, stats = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(
        "Model artifacts not found. Run the notebooks in `notebooks/` to "
        f"train and save the model first.\n\nError: `{e}`"
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model")
    if model_loaded:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", f"{stats['roc_auc']:.3f}")
        with col2:
            st.metric("Avg Precision", f"{stats['avg_precision']:.3f}")

        st.markdown(
            f"""
<div style="font-size:0.88rem;color:var(--ink-soft);line-height:1.7;margin-top:0.6rem">
<span style="color:var(--muted)">Algorithm</span> &nbsp; XGBoost classifier<br>
<span style="color:var(--muted)">Training rows</span> &nbsp; {stats['train_size']:,}<br>
<span style="color:var(--muted)">Test rows</span> &nbsp; {stats['test_size']:,}<br>
<span style="color:var(--muted)">Default rate</span> &nbsp; {stats['default_rate']:.1%}
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### About")
    st.markdown(
        "Trained on roughly 900,000 SBA-backed loans. Each prediction comes with "
        "SHAP values, so you can see which loan attributes pushed the score up or down."
    )

    st.markdown("### Data")
    st.markdown(
        "[SBA national loan dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied)"
    )


# ─── Input form ───────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Loan details</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    loan_amount = st.number_input(
        "Loan amount ($)",
        min_value=1000, max_value=5_000_000,
        value=150_000, step=5_000, format="%d",
    )
    term = st.slider("Loan term (months)", min_value=12, max_value=360, value=84, step=12)
    sba_guarantee = st.slider("SBA guarantee (%)", min_value=0, max_value=100, value=75)

with col2:
    industry = st.selectbox("Industry", options=list(NAICS_MAP.keys()))
    if model_loaded:
        state = st.selectbox("State", options=sorted(state_map.keys()))
    else:
        state = st.selectbox("State", options=["CA", "TX", "NY", "FL"])
    is_new = st.radio("Business type", ["Existing business", "New business"], horizontal=True)

with col3:
    employees = st.number_input("Number of employees", min_value=0, max_value=500, value=10)
    jobs_created = st.number_input("Jobs created", min_value=0, max_value=200, value=2)
    jobs_retained = st.number_input("Jobs retained", min_value=0, max_value=200, value=8)
    is_franchise = st.checkbox("Franchise")
    low_doc = st.checkbox("Low documentation loan")
    rev_line = st.checkbox("Revolving line of credit")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
predict_btn = st.button("Analyze loan risk")


# ─── Prediction ───────────────────────────────────────────────────────────────

if predict_btn and model_loaded:

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
        'IsNewBusiness': int(is_new == "New business"),
        'IsFranchise': int(is_franchise),
        'NAICSCode': NAICS_MAP[industry],
        'StateEncoded': state_map.get(state, 0),
        'ApprovalFY': 2024,
    }

    input_df = pd.DataFrame([input_data])[features]
    risk_score = float(model.predict_proba(input_df)[0][1])

    # Risk band
    if risk_score >= 0.6:
        risk_class, risk_label, risk_color = "risk-high", "High risk", "var(--red)"
    elif risk_score >= 0.35:
        risk_class, risk_label, risk_color = "risk-medium", "Medium risk", "var(--amber)"
    else:
        risk_class, risk_label, risk_color = "risk-low", "Low risk", "var(--green)"

    st.markdown('<p class="section-header">Risk assessment</p>', unsafe_allow_html=True)

    result_col, explain_col = st.columns([1, 2], gap="large")

    with result_col:
        st.markdown(
            f"""
<div class="risk-card {risk_class}">
    <div class="risk-score" style="color:{risk_color}">{risk_score:.0%}</div>
    <div class="risk-label" style="color:{risk_color}">{risk_label}</div>
    <div class="risk-caption">Probability of default</div>
</div>
""",
            unsafe_allow_html=True,
        )

        baseline = stats['default_rate']
        rel_risk = risk_score / baseline
        st.markdown(
            f"<div style='margin-top:1rem;font-size:0.88rem;color:var(--ink-soft);"
            f"text-align:center;line-height:1.55'>"
            f"<span style='font-family:IBM Plex Mono,monospace;font-weight:600;color:var(--ink)'>{rel_risk:.1f}×</span> "
            f"the historical average default rate of {baseline:.1%}.</div>",
            unsafe_allow_html=True,
        )

    with explain_col:
        st.markdown(
            "<div style='font-size:0.95rem;font-weight:500;margin-bottom:0.85rem;color:var(--ink)'>"
            "Top contributing factors</div>",
            unsafe_allow_html=True,
        )

        shap_vals = explainer.shap_values(input_df)[0]
        feature_impact = pd.DataFrame({
            'Feature': features,
            'Value': input_df.values[0],
            'SHAP': shap_vals,
        }).sort_values('SHAP', key=abs, ascending=False).head(8)

        for _, row in feature_impact.iterrows():
            fname = FRIENDLY_NAMES.get(row['Feature'], row['Feature'])
            increases = row['SHAP'] > 0
            direction = "Increases risk" if increases else "Decreases risk"
            color = "var(--red)" if increases else "var(--green)"
            bar_width = min(abs(row['SHAP']) * 300, 100)

            st.markdown(
                f"""
<div class="factor-row">
    <div class="factor-meta">
        <span class="factor-name">{fname}</span>
        <span class="factor-dir" style="color:{color}">{direction}</span>
    </div>
    <div class="factor-bar">
        <div class="factor-bar-fill" style="background:{color};width:{bar_width}%"></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    # SHAP waterfall
    st.markdown('<p class="section-header">Full SHAP breakdown</p>', unsafe_allow_html=True)

    shap_display = pd.DataFrame({
        'Feature': [FRIENDLY_NAMES.get(f, f) for f in features],
        'SHAP Value': shap_vals,
    }).sort_values('SHAP Value', key=abs, ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#dc2626' if v > 0 else '#16a34a' for v in shap_display['SHAP Value']]
    ax.barh(shap_display['Feature'], shap_display['SHAP Value'], color=colors, height=0.65)
    ax.axvline(0, color='#cbd5e1', linewidth=0.8)
    ax.set_xlabel('SHAP value (impact on default probability)', fontsize=10, color='#475569')
    ax.tick_params(axis='both', colors='#475569', labelsize=10)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.grid(axis='x', color='#f1f5f9', linewidth=0.7)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary
    st.markdown('<p class="section-header">Loan summary</p>', unsafe_allow_html=True)

    summary = [
        ("Loan amount", f"${loan_amount:,}"),
        ("SBA guaranteed", f"${sba_appv:,.0f} ({sba_guarantee}%)"),
        ("Term", f"{term} months ({term // 12} years)"),
        ("Industry", industry),
        ("State", state),
        ("Employees", f"{employees:,}"),
        ("Total jobs", f"{total_jobs:,}"),
        ("Business type", is_new),
        ("Risk score", f"{risk_score:.1%}"),
        ("Risk level", risk_label),
    ]

    col_a, col_b = st.columns(2, gap="large")
    half = (len(summary) + 1) // 2
    for col, items in ((col_a, summary[:half]), (col_b, summary[half:])):
        rows = "".join(
            f"<div class='summary-row'><span class='summary-key'>{k}</span>"
            f"<span class='summary-val'>{v}</span></div>"
            for k, v in items
        )
        col.markdown(rows, unsafe_allow_html=True)


elif predict_btn and not model_loaded:
    st.error("Train the model first by running the three notebooks in order.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="footer">
    CreditLens &middot; XGBoost &middot; SHAP &middot; Streamlit
    &middot; <a href="https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied">SBA loan dataset</a>
</div>
""",
    unsafe_allow_html=True,
)
