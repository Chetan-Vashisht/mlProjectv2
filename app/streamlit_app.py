import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.predict import (
    load_artifacts, predict_rate, predict_all_models, SCALE_COLS
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Interest Rate Recommender",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a polished fintech look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #00bcd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #666;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
    .disclaimer {
        text-align: center;
        color: #999;
        font-size: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .rate-display {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .rate-number {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .rate-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }

    .rate-low { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); }
    .rate-low .rate-number { color: #2e7d32; }

    .rate-mid { background: linear-gradient(135deg, #fff3e0, #ffe0b2); }
    .rate-mid .rate-number { color: #ef6c00; }

    .rate-high { background: linear-gradient(135deg, #ffebee, #ffcdd2); }
    .rate-high .rate-number { color: #c62828; }

    .range-bar {
        background: #f0f0f0;
        border-radius: 8px;
        height: 12px;
        position: relative;
        margin: 0.5rem 0;
    }
    .range-fill {
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        border-radius: 8px;
        height: 100%;
    }

    .grade-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.5rem 0;
    }
    .grade-a { background: #e8f5e9; color: #2e7d32; }
    .grade-b { background: #e3f2fd; color: #1565c0; }
    .grade-c { background: #fff3e0; color: #ef6c00; }
    .grade-d { background: #fff8e1; color: #f9a825; }
    .grade-e { background: #fbe9e7; color: #d84315; }
    .grade-f { background: #ffebee; color: #c62828; }
    .grade-g { background: #fce4ec; color: #ad1457; }

    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model Artifacts ---
@st.cache_resource
def load_all_artifacts():
    try:
        model = joblib.load('models/XGBoost.pkl')
    except FileNotFoundError:
        model = joblib.load('models/LinearRegression.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, feature_columns

model, scaler, feature_columns = load_all_artifacts()

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>💰 Loan Interest Rate Recommender</h1>
    <p>Enter applicant details to get a data-driven, explainable rate recommendation</p>
</div>
<p class="disclaimer">⚠️ For academic demonstration purposes only — not a real lending decision tool</p>
""", unsafe_allow_html=True)

# --- Two-column layout ---
col_input, col_spacer, col_output = st.columns([4, 0.5, 5])

with col_input:
    st.markdown("### 📋 Applicant Details")

    loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 12000, step=500,
                           help="Amount requested by the borrower")

    annual_inc = st.slider("Annual Income ($)", 10000, 250000, 45000, step=1000,
                            help="Self-reported annual income")

    emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                          '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
    emp_length_str = st.selectbox("Employment Length", emp_length_options, index=3)
    emp_length = emp_length_options.index(emp_length_str)

    dti = st.slider("Debt-to-Income Ratio (%)", 0, 50, 20,
                     help="Monthly debt payments divided by gross monthly income")

    fico_avg = st.slider("FICO Score", 300, 850, 680,
                          help="Credit score (average of range)")

    term_months = st.radio("Loan Term", [36, 60], horizontal=True,
                            format_func=lambda x: f"{x} months")

    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

    purpose_options = ["debt_consolidation", "credit_card", "home_improvement",
                       "major_purchase", "other", "medical", "small_business",
                       "moving", "vacation", "house", "wedding", "renewable_energy",
                       "educational"]
    purpose = st.selectbox("Loan Purpose", purpose_options,
                            format_func=lambda x: x.replace('_', ' ').title())

    open_acc = st.slider("Open Credit Lines", 0, 30, 8)
    pub_rec = st.slider("Public Records", 0, 5, 0,
                         help="Number of derogatory public records")

    revol_util = st.slider("Revolving Credit Utilization (%)", 0.0, 100.0, 50.0, step=0.5,
                            help="Amount of revolving credit used relative to limit")

with col_output:
    st.markdown("### 📊 Model Output")

    # --- Predict ---
    prediction = predict_rate(
        model, scaler, feature_columns,
        loan_amnt=loan_amnt, annual_inc=annual_inc, emp_length=emp_length,
        dti=dti, fico_avg=fico_avg, term_months=term_months,
        home_ownership=home_ownership, purpose=purpose,
        open_acc=open_acc, pub_rec=pub_rec, revol_util=revol_util
    )

    # Confidence range (±10% heuristic, matching AGENTS.md placeholder)
    lower = prediction * 0.90
    upper = prediction * 1.10

    # --- Element 1: Big number display ---
    if prediction < 10:
        rate_class = "rate-low"
        risk_label = "Low Risk Pricing"
    elif prediction < 15:
        rate_class = "rate-mid"
        risk_label = "Moderate Risk Pricing"
    else:
        rate_class = "rate-high"
        risk_label = "High Risk Pricing"

    st.markdown(f"""
    <div class="rate-display {rate_class}">
        <div class="rate-number">{prediction:.2f}%</div>
        <div class="rate-label">Predicted Interest Rate • {risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Element 2: Confidence range ---
    st.markdown(f"**Estimated Range:** {lower:.2f}% – {upper:.2f}% *(±10% band)*")

    # Progress bar visualization for the range
    range_pct = min(max((prediction - 5) / 25 * 100, 0), 100)  # Map 5%–30% to 0–100
    st.progress(int(range_pct) / 100)

    # --- Element 3: Grade equivalent badge ---
    grade_ranges = [
        (7.5, 'A', 'grade-a'), (10.5, 'B', 'grade-b'), (14.5, 'C', 'grade-c'),
        (19, 'D', 'grade-d'), (24, 'E', 'grade-e'), (28, 'F', 'grade-f'),
        (35, 'G', 'grade-g')
    ]
    equiv_grade = 'G'
    equiv_class = 'grade-g'
    for threshold, grade, css_class in grade_ranges:
        if prediction < threshold:
            equiv_grade = grade
            equiv_class = css_class
            break

    st.markdown(f"""
    <span class="grade-badge {equiv_class}">≈ LendingClub Grade {equiv_grade}</span>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Element 4: Key factors summary ---
    st.markdown("#### 🔍 Key Input Factors")
    factor_cols = st.columns(3)
    with factor_cols[0]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{fico_avg}</div>
            <div class="metric-label">FICO Score</div>
        </div>""", unsafe_allow_html=True)
    with factor_cols[1]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{dti}%</div>
            <div class="metric-label">DTI Ratio</div>
        </div>""", unsafe_allow_html=True)
    with factor_cols[2]:
        lti = loan_amnt / annual_inc if annual_inc > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{lti:.2f}</div>
            <div class="metric-label">Loan-to-Income</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- Element 5: Model comparison ---
    with st.expander("📈 See how other models would price this loan"):
        all_predictions = predict_all_models(
            scaler, feature_columns,
            loan_amnt=loan_amnt, annual_inc=annual_inc, emp_length=emp_length,
            dti=dti, fico_avg=fico_avg, term_months=term_months,
            home_ownership=home_ownership, purpose=purpose,
            open_acc=open_acc, pub_rec=pub_rec, revol_util=revol_util
        )

        comp_df = pd.DataFrame({
            'Model': list(all_predictions.keys()),
            'Predicted Rate (%)': [f"{v:.2f}" if v is not None else "N/A"
                                   for v in all_predictions.values()]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
