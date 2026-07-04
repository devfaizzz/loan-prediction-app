import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from predict import predict_loan

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .approved-card {
        background: linear-gradient(135deg, rgba(56, 178, 172, 0.2), rgba(72, 187, 120, 0.2));
        border: 2px solid #48bb78;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .rejected-card {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.2), rgba(229, 62, 62, 0.2));
        border: 2px solid #fc8181;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .result-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .result-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .confidence-text {
        font-size: 1.2rem;
        color: #e2e8f0;
    }

    .section-header {
        color: #ffd200;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(255, 210, 0, 0.3);
        padding-bottom: 0.5rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    .stSidebar {
        background: rgba(255, 255, 255, 0.03) !important;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    .stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200) !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(247, 151, 30, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(247, 151, 30, 0.6) !important;
    }

    .stSelectbox label, .stNumberInput label, .stRadio label {
        color: #a0aec0 !important;
        font-weight: 500 !important;
    }

    label[data-testid="stWidgetLabel"] {
        color: #cbd5e0 !important;
    }

    .tip-box {
        background: rgba(247, 151, 30, 0.1);
        border-left: 4px solid #f7971e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #fbd38d;
        font-size: 0.85rem;
    }

    button[data-testid="collapsedControl"] {
        display: none !important;
    }

    .stAppDeployButton {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #4a5568;
        font-size: 0.82rem;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin-top: 2rem;
    }
    .footer span {
        color: #ffd200;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">FIJ Loan Approval Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter applicant details below to predict loan approval status</p>', unsafe_allow_html=True)

st.markdown("---")

with st.form("loan_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        married = st.selectbox("Marital Status", ["Yes", "No"], key="married")
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], key="dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="education")
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], key="self_employed")

    with col2:
        st.markdown('<p class="section-header">💰 Financial Information</p>', unsafe_allow_html=True)
        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)",
            min_value=0, max_value=1000000,
            value=5000, step=500, key="app_income"
        )
        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income (₹)",
            min_value=0, max_value=1000000,
            value=0, step=500, key="coapp_income"
        )
        loan_amount = st.number_input(
            "Loan Amount (in thousands ₹)",
            min_value=1, max_value=10000,
            value=150, step=10, key="loan_amt"
        )

    with col3:
        st.markdown('<p class="section-header">🏠 Loan Details</p>', unsafe_allow_html=True)
        loan_amount_term = st.selectbox(
            "Loan Amount Term (months)",
            [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
            index=9, key="loan_term"
        )
        credit_history = st.selectbox(
            "Credit History",
            [1, 0],
            format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)",
            key="credit_hist"
        )
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"],
            key="prop_area"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict Loan Status")

if submitted:

    errors = []

    if applicant_income <= 0:
        errors.append("❌ Applicant income must be greater than 0")
    if coapplicant_income < 0:
        errors.append("❌ Co-applicant income cannot be negative")
    if loan_amount <= 0:
        errors.append("❌ Loan amount must be greater than 0")

    if errors:
        for err in errors:
            st.error(err)
    else:
        try:
            result, confidence, probs = predict_loan(
                gender=gender,
                married=married,
                dependents=int(dependents),
                education=education,
                self_employed=self_employed,
                applicant_income=applicant_income,
                coapplicant_income=coapplicant_income,
                loan_amount=loan_amount,
                loan_amount_term=loan_amount_term,
                credit_history=credit_history,
                property_area=property_area
            )

            st.markdown("---")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                if result == "Approved":
                    st.markdown(f"""
                    <div class="approved-card">
                        <div class="result-emoji">✅</div>
                        <div class="result-text" style="color: #68d391;">Loan Approved!</div>
                        <div class="confidence-text">Confidence: <b>{confidence}%</b></div>
                        <br>
                        <div style="color: #9ae6b4; font-size: 0.95rem;">
                            Congratulations! Based on the provided information,<br>
                            your loan application is likely to be approved.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="rejected-card">
                        <div class="result-emoji">❌</div>
                        <div class="result-text" style="color: #fc8181;">Loan Rejected</div>
                        <div class="confidence-text">Confidence: <b>{confidence}%</b></div>
                        <br>
                        <div style="color: #feb2b2; font-size: 0.95rem;">
                            Based on the provided information,<br>
                            this loan application may be rejected.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<p class="section-header">📊 Prediction Probabilities</p>', unsafe_allow_html=True)

                approved_pct = round(probs[1] * 100, 1)
                rejected_pct = round(probs[0] * 100, 1)

                st.markdown(f"**Approval Probability**: {approved_pct}%")
                st.progress(int(approved_pct))

                st.markdown(f"**Rejection Probability**: {rejected_pct}%")
                st.progress(int(rejected_pct))

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<p class="section-header">📋 Your Application Summary</p>', unsafe_allow_html=True)

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.markdown(f"""
                <div class="card">
                    <b style="color: #ffd200;">Personal Details</b><br><br>
                    <div class="info-row"><span>Gender</span><span>{gender}</span></div>
                    <div class="info-row"><span>Married</span><span>{married}</span></div>
                    <div class="info-row"><span>Dependents</span><span>{dependents}</span></div>
                    <div class="info-row"><span>Education</span><span>{education}</span></div>
                    <div class="info-row"><span>Self Employed</span><span>{self_employed}</span></div>
                </div>
                """, unsafe_allow_html=True)

            with summary_col2:
                st.markdown(f"""
                <div class="card">
                    <b style="color: #ffd200;">Financial Details</b><br><br>
                    <div class="info-row"><span>Applicant Income</span><span>₹{applicant_income:,}</span></div>
                    <div class="info-row"><span>Coapplicant Income</span><span>₹{coapplicant_income:,}</span></div>
                    <div class="info-row"><span>Total Income</span><span>₹{applicant_income + coapplicant_income:,}</span></div>
                    <div class="info-row"><span>Loan Amount</span><span>₹{loan_amount}K</span></div>
                    <div class="info-row"><span>Loan Term</span><span>{loan_amount_term} months</span></div>
                </div>
                """, unsafe_allow_html=True)

            with summary_col3:
                credit_label = "Good ✅" if credit_history == 1 else "Bad ❌"
                st.markdown(f"""
                <div class="card">
                    <b style="color: #ffd200;">Loan Details</b><br><br>
                    <div class="info-row"><span>Credit History</span><span>{credit_label}</span></div>
                    <div class="info-row"><span>Property Area</span><span>{property_area}</span></div>
                    <div class="info-row"><span>Predicted Status</span><span>{result}</span></div>
                    <div class="info-row"><span>Confidence Score</span><span>{confidence}%</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if result == "Approved":
                st.markdown("""
                <div class="tip-box">
                💡 <b>Tip:</b> Even with approval prediction, final decisions depend on the bank's internal policies and document verification.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="tip-box">
                💡 <b>Tip:</b> To improve your chances — maintain good credit history, reduce existing debts, and consider a co-applicant with stable income.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please make sure you have trained the model first by running: `python train.py`")

st.markdown("""
<div class="footer">
    © 2024 Copyright — Team <span>Faiz &nbsp;•&nbsp; Insha &nbsp;•&nbsp; Jahanavi</span><br>
    AIML Summer Internship Capstone Project &nbsp;|&nbsp; Loan Approval Prediction System
</div>
""", unsafe_allow_html=True)
