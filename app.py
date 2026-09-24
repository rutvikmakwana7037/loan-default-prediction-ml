import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
st.set_page_config(
    page_title="Riskora AI | 9-Model Credit Risk & Underwriting Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
MODEL_DIR = Path("models")
LR_PATH = MODEL_DIR / "logistic_regression.pkl"
DT_PATH = MODEL_DIR / "decision_tree.pkl"
RF_PATH = MODEL_DIR / "random_forest.pkl"
KNN_PATH = MODEL_DIR / "k_neighbours.pkl"
SVC_PATH = MODEL_DIR / "svc.pkl"
GNB_PATH = MODEL_DIR / "gaussian_nb.pkl"
BAGGING_PATH = MODEL_DIR / "bagging.pkl"
BOOSTING_PATH = MODEL_DIR / "boosting.pkl"
ADABOOST_PATH = MODEL_DIR / "adaboost.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ROOT_LR_PATH = Path("model.pkl")
ROOT_SCALER_PATH = Path("scaler.pkl")
DATASET_PATH = Path("preprocessed_loan_default.csv")
RAW_DATASET_PATH = Path("Loan_default.csv")
FEATURE_NAMES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasDependents",
]
APPLICANT_PRESETS = {
    "🌟 Prime Borrower": {
        "Age": 48, "Income": 135000, "LoanAmount": 80000, "CreditScore": 810,
        "MonthsEmployed": 96, "NumCreditLines": 2, "InterestRate": 4.5,
        "LoanTerm": 36, "DTIRatio": 0.20, "HasMortgage": "No", "HasDependents": "Yes"
    },
    "💼 Average Applicant": {
        "Age": 38, "Income": 68000, "LoanAmount": 140000, "CreditScore": 640,
        "MonthsEmployed": 42, "NumCreditLines": 4, "InterestRate": 11.2,
        "LoanTerm": 48, "DTIRatio": 0.42, "HasMortgage": "Yes", "HasDependents": "Yes"
    },
    "⚠️ High-Risk Profile": {
        "Age": 22, "Income": 24000, "LoanAmount": 220000, "CreditScore": 480,
        "MonthsEmployed": 6, "NumCreditLines": 7, "InterestRate": 22.5,
        "LoanTerm": 60, "DTIRatio": 0.78, "HasMortgage": "No", "HasDependents": "No"
    },
    "🎓 Young Professional": {
        "Age": 26, "Income": 85000, "LoanAmount": 95000, "CreditScore": 710,
        "MonthsEmployed": 24, "NumCreditLines": 2, "InterestRate": 7.8,
        "LoanTerm": 36, "DTIRatio": 0.28, "HasMortgage": "No", "HasDependents": "No"
    },
    "🏡 High DTI Homeowner": {
        "Age": 45, "Income": 75000, "LoanAmount": 190000, "CreditScore": 610,
        "MonthsEmployed": 60, "NumCreditLines": 6, "InterestRate": 16.0,
        "LoanTerm": 60, "DTIRatio": 0.68, "HasMortgage": "Yes", "HasDependents": "Yes"
    }
}
BENCHMARK_METRICS = {
    "Gradient Boosting": {
        "Category": "Boosting",
        "Accuracy": 0.6905,
        "Precision_Default": 0.2240,
        "Recall_Default": 0.6753,
        "F1_Default": 0.3364,
        "Precision_NonDefault": 0.9430,
        "Recall_NonDefault": 0.6925,
        "F1_NonDefault": 0.7984,
        "Macro_F1": 0.5674,
        "Weighted_F1": 0.7447,
        "ROC_AUC": 0.7480,
        "Specificity": 0.6925,
        "True_Negatives": 31814,
        "False_Positives": 13325,
        "False_Negatives": 1926,
        "True_Positives": 4005,
        "Type": "HistGradientBoosting (Balanced Trees)",
        "Key_Strength": "🏆 Top Overall Default Recall (67.53%): Catches highest absolute default count (4,005)",
        "Key_Weakness": "Slightly higher false alert friction on safe applicants",
        "Optimal_Role": "Maximum credit default loss prevention & subprime portfolio protection"
    },
    "Random Forest": {
        "Category": "Ensemble (Bagging)",
        "Accuracy": 0.7255,
        "Precision_Default": 0.2402,
        "Recall_Default": 0.6304,
        "F1_Default": 0.3478,
        "Precision_NonDefault": 0.9383,
        "Recall_NonDefault": 0.7379,
        "F1_NonDefault": 0.8261,
        "Macro_F1": 0.5870,
        "Weighted_F1": 0.7706,
        "ROC_AUC": 0.7481,
        "Specificity": 0.7379,
        "True_Negatives": 33310,
        "False_Positives": 11829,
        "False_Negatives": 2192,
        "True_Positives": 3739,
        "Type": "Balanced Ensemble (300 Trees, Depth=10)",
        "Key_Strength": "🏆 Top ROC-AUC (0.7481) & Best F1-Score (0.3478) balance",
        "Key_Weakness": "Ensemble memory size requires tree-importance tooling",
        "Optimal_Role": "Primary institutional underwriting engine for core risk mitigation"
    },
    "Bagging Classifier": {
        "Category": "Bagging",
        "Accuracy": 0.6993,
        "Precision_Default": 0.2273,
        "Recall_Default": 0.6621,
        "F1_Default": 0.3384,
        "Precision_NonDefault": 0.9416,
        "Recall_NonDefault": 0.7042,
        "F1_NonDefault": 0.8058,
        "Macro_F1": 0.5721,
        "Weighted_F1": 0.7515,
        "ROC_AUC": 0.7432,
        "Specificity": 0.7042,
        "True_Negatives": 32289,
        "False_Positives": 12850,
        "False_Negatives": 2004,
        "True_Positives": 3927,
        "Type": "Bootstrap Aggregation (60 Deep Trees)",
        "Key_Strength": "Exceptional Default Recall (66.21%) with reduced variance over single trees",
        "Key_Weakness": "Bootstrap redundancy requires parallel cores",
        "Optimal_Role": "Secondary validation ensemble for high-exposure consumer loans"
    },
    "Decision Tree": {
        "Category": "Classification Trees",
        "Accuracy": 0.6703,
        "Precision_Default": 0.1913,
        "Recall_Default": 0.5701,
        "F1_Default": 0.2865,
        "Precision_NonDefault": 0.9237,
        "Recall_NonDefault": 0.6834,
        "F1_NonDefault": 0.7856,
        "Macro_F1": 0.5360,
        "Weighted_F1": 0.7276,
        "ROC_AUC": 0.6616,
        "Specificity": 0.6834,
        "True_Negatives": 30849,
        "False_Positives": 14290,
        "False_Negatives": 2550,
        "True_Positives": 3381,
        "Type": "Balanced Non-Linear Tree (Depth=2)",
        "Key_Strength": "Fully transparent rule thresholds on Age, APR, and Income",
        "Key_Weakness": "Higher false positive rate on safe applicants",
        "Optimal_Role": "Frontline automated policy sieve & instant CRM rejection gating"
    },
    "AdaBoost": {
        "Category": "Boosting",
        "Accuracy": 0.8855,
        "Precision_Default": 0.6088,
        "Recall_Default": 0.0396,
        "F1_Default": 0.0744,
        "Precision_NonDefault": 0.8876,
        "Recall_NonDefault": 0.9967,
        "F1_NonDefault": 0.9389,
        "Macro_F1": 0.5067,
        "Weighted_F1": 0.8384,
        "ROC_AUC": 0.7439,
        "Specificity": 0.9967,
        "True_Negatives": 44988,
        "False_Positives": 151,
        "False_Negatives": 5696,
        "True_Positives": 235,
        "Type": "Adaptive Sequential Boosting (100 Estimators)",
        "Key_Strength": "🏆 Top Overall Accuracy (88.55%) and 60.9% Default Precision",
        "Key_Weakness": "Conservative default trigger on highly imbalanced credit sets",
        "Optimal_Role": "Prime lending margin maximization & customer retention"
    },
    "Logistic Regression": {
        "Category": "Linear Models",
        "Accuracy": 0.8844,
        "Precision_Default": 0.5556,
        "Recall_Default": 0.0219,
        "F1_Default": 0.0422,
        "Precision_NonDefault": 0.8859,
        "Recall_NonDefault": 0.9977,
        "F1_NonDefault": 0.9385,
        "Macro_F1": 0.4903,
        "Weighted_F1": 0.8344,
        "ROC_AUC": 0.7440,
        "Specificity": 0.9977,
        "True_Negatives": 45035,
        "False_Positives": 104,
        "False_Negatives": 5801,
        "True_Positives": 130,
        "Type": "Calibrated Linear Generalized Model",
        "Key_Strength": "Near-zero false alarm rate (99.77% specificity) with log-odds transparency",
        "Key_Weakness": "Under-detects subprime defaults due to linear decision boundary",
        "Optimal_Role": "Regulatory compliance, audited beta reports & Prime lending"
    },
    "Support Vector Classification (SVC)": {
        "Category": "Kernel & Margin Models",
        "Accuracy": 0.8843,
        "Precision_Default": 0.5545,
        "Recall_Default": 0.0206,
        "F1_Default": 0.0397,
        "Precision_NonDefault": 0.8858,
        "Recall_NonDefault": 0.9978,
        "F1_NonDefault": 0.9385,
        "Macro_F1": 0.4891,
        "Weighted_F1": 0.8342,
        "ROC_AUC": 0.7442,
        "Specificity": 0.9978,
        "True_Negatives": 45041,
        "False_Positives": 98,
        "False_Negatives": 5809,
        "True_Positives": 122,
        "Type": "Calibrated Support Vector Classifier (Max-Margin)",
        "Key_Strength": "Maximum margin hyperplane separation with 99.78% specificity",
        "Key_Weakness": "Hyperplane sensitivity to noisy boundary cases",
        "Optimal_Role": "Margin arbitration & borderline applicant second-opinion scoring"
    },
    "Gaussian Naive Bayes": {
        "Category": "Probabilistic Classifiers",
        "Accuracy": 0.8842,
        "Precision_Default": 0.5763,
        "Recall_Default": 0.0115,
        "F1_Default": 0.0225,
        "Precision_NonDefault": 0.8848,
        "Recall_NonDefault": 0.9989,
        "F1_NonDefault": 0.9384,
        "Macro_F1": 0.4804,
        "Weighted_F1": 0.8318,
        "ROC_AUC": 0.7449,
        "Specificity": 0.9989,
        "True_Negatives": 45089,
        "False_Positives": 50,
        "False_Negatives": 5863,
        "True_Positives": 68,
        "Type": "Continuous Gaussian Likelihood (Bayesian)",
        "Key_Strength": "Fast probabilistic likelihood calculation with 99.89% specificity",
        "Key_Weakness": "Conditional feature independence assumption",
        "Optimal_Role": "Real-time probabilistic prior baseline calculation"
    },
    "K-Nearest Neighbors": {
        "Category": "Instance Memory",
        "Accuracy": 0.8800,
        "Precision_Default": 0.3895,
        "Recall_Default": 0.0639,
        "F1_Default": 0.1098,
        "Precision_NonDefault": 0.8895,
        "Recall_NonDefault": 0.9869,
        "F1_NonDefault": 0.9357,
        "Macro_F1": 0.5227,
        "Weighted_F1": 0.8400,
        "ROC_AUC": 0.6572,
        "Specificity": 0.9869,
        "True_Negatives": 44700,
        "False_Positives": 439,
        "False_Negatives": 5552,
        "True_Positives": 379,
        "Type": "Standardized Euclidean Instance Memory (K=8)",
        "Key_Strength": "Identifies localized historical peer clusters in 11D space",
        "Key_Weakness": "Memory distance metric sensitive to sparse feature regions",
        "Optimal_Role": "Cohort peer benchmarking & manual underwriting review validation"
    }
}
def _safe_load_obj(primary_path, fallback_path=None):
    """Safely loads serialized ML objects with joblib or pickle."""
    target_path = primary_path if primary_path.exists() else (fallback_path if fallback_path and fallback_path.exists() else None)
    if target_path is None or not target_path.exists():
        return None, f"File {primary_path} not found."
    try:
        return joblib.load(target_path), ""
    except Exception:
        try:
            with open(target_path, "rb") as f:
                return pickle.load(f), ""
        except Exception as e:
            return None, str(e)
@st.cache_resource
def load_all_ml_assets():
    """Loads all 8 machine learning models and the StandardScaler."""
    models_dict = {}
    errors = []
    model_configs = [
        ("lr", LR_PATH, ROOT_LR_PATH, "Logistic Regression"),
        ("dt", DT_PATH, None, "Decision Tree"),
        ("rf", RF_PATH, None, "Random Forest"),
        ("knn", KNN_PATH, None, "K-Nearest Neighbors"),
        ("svc", SVC_PATH, None, "Support Vector Classification (SVC)"),
        ("gnb", GNB_PATH, None, "Gaussian Naive Bayes"),
        ("bagging", BAGGING_PATH, None, "Bagging Classifier"),
        ("boosting", BOOSTING_PATH, None, "Gradient Boosting"),
        ("adaboost", ADABOOST_PATH, None, "AdaBoost")
    ]
    for key, p_path, f_path, label in model_configs:
        obj, err = _safe_load_obj(p_path, f_path)
        if obj is not None:
            models_dict[key] = obj
        else:
            errors.append(f"{label}: {err}")
    scaler_obj, s_err = _safe_load_obj(SCALER_PATH, ROOT_SCALER_PATH)
    if scaler_obj is None:
        errors.append(f"Scaler: {s_err}")
    ready = (len(models_dict) >= 4 and scaler_obj is not None)
    error_msg = "; ".join(errors) if errors else ""
    return models_dict, scaler_obj, ready, error_msg
@st.cache_data
def load_eda_dataset(sample_size=15000):
    """Loads a representative cached sample from preprocessed or raw dataset for fast analytics."""
    path_to_use = DATASET_PATH if DATASET_PATH.exists() else (RAW_DATASET_PATH if RAW_DATASET_PATH.exists() else None)
    if path_to_use is None:
        return None
    try:
        df = pd.read_csv(path_to_use)
        if 'Default' in df.columns and 'LoanDefault' not in df.columns:
            df['LoanDefault'] = df['Default']
        for col in ['HasMortgage', 'HasDependents']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        if len(df) > sample_size:
            return df.sample(n=sample_size, random_state=42)
        return df
    except Exception:
        return None
models_pool, scaler, models_ready, models_error = load_all_ml_assets()
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
        /* ============================================================
           1. OBSIDIAN LUXE & EMERALD DESIGN TOKENS
           ============================================================ */
        :root {
            --bg-base: #080C14;
            --bg-surface: rgba(13, 20, 36, 0.82);
            --bg-surface-elevated: rgba(18, 28, 50, 0.90);
            --border-glass: rgba(255, 255, 255, 0.09);
            --border-emerald-glow: rgba(16, 185, 129, 0.45);
            --border-cyan-glow: rgba(6, 182, 212, 0.40);
            --luxe-emerald: #10B981;
            --luxe-mint: #34D399;
            --luxe-cyan: #06B6D4;
            --luxe-sapphire: #38BDF8;
            --luxe-indigo: #6366F1;
            --luxe-violet: #A855F7;
            --luxe-gold: #F59E0B;
            --luxe-rose: #F43F5E;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
        }
        body, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, label, .stSelectbox label, .stSlider label {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        /* Streamlit icon preservation */
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        [data-testid="stIconMaterial"],
        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stSidebarCollapseButton"] button span,
        [data-testid="collapsedControl"] span,
        [data-testid="collapsedControl"] button span,
        button[kind="header"] span,
        button[kind="headerNoPadding"] span {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
            font-weight: normal !important;
            font-style: normal !important;
            display: inline-block !important;
            direction: ltr !important;
        }
        /* App Background Mesh */
        .stApp {
            background-color: var(--bg-base) !important;
            background-image: 
                radial-gradient(circle at 12% 8%, rgba(16, 185, 129, 0.12) 0%, transparent 42%),
                radial-gradient(circle at 88% 12%, rgba(56, 189, 248, 0.10) 0%, transparent 38%),
                radial-gradient(circle at 50% 90%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 18% 80%, rgba(245, 158, 11, 0.05) 0%, transparent 35%) !important;
            background-attachment: fixed !important;
            color: var(--text-primary) !important;
        }
        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1440px !important;
        }
        /* ============================================================
           2. LUXE NAVBAR & STATUS BADGES
           ============================================================ */
        .lg-navbar {
            background: linear-gradient(135deg, rgba(13, 22, 38, 0.92) 0%, rgba(10, 16, 30, 0.88) 100%);
            border: 1px solid rgba(16, 185, 129, 0.28);
            border-radius: 20px;
            padding: 16px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.7), 0 0 25px rgba(16, 185, 129, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            margin-top: 6px;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }
        .lg-navbar:hover {
            border-color: rgba(16, 185, 129, 0.5);
            box-shadow: 0 16px 45px -8px rgba(0, 0, 0, 0.8), 0 0 32px rgba(16, 185, 129, 0.2);
        }
        .lg-brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .lg-brand-logo {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #6366F1 100%);
            color: white;
            font-size: 26px;
            box-shadow: 0 0 28px rgba(16, 185, 129, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .lg-brand-title {
            color: #FFFFFF;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.6px;
            line-height: 1.15;
            background: linear-gradient(135deg, #FFFFFF 20%, #A7F3D0 70%, #34D399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .lg-brand-subtitle {
            color: #94A3B8;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.3px;
            margin-top: 3px;
        }
        .lg-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 18px;
            border-radius: 9999px;
            font-size: 12.5px;
            font-weight: 700;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.45);
            color: #34D399;
            box-shadow: 0 0 18px rgba(16, 185, 129, 0.22);
            letter-spacing: 0.3px;
        }
        .lg-status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #10B981;
            box-shadow: 0 0 10px #10B981;
        }
        /* ============================================================
           3. HERO BANNER
           ============================================================ */
        .lg-hero {
            background: linear-gradient(135deg, rgba(14, 24, 44, 0.90) 0%, rgba(9, 15, 30, 0.96) 100%);
            border: 1px solid rgba(16, 185, 129, 0.22);
            border-radius: 22px;
            padding: 28px 34px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.65), 0 0 25px rgba(16, 185, 129, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(20px);
        }
        .lg-hero::before {
            content: '';
            position: absolute;
            top: -65%;
            right: -15%;
            width: 440px;
            height: 440px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.20) 0%, rgba(56, 189, 248, 0.12) 45%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .lg-hero-badge {
            display: inline-block;
            padding: 5px 15px;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(6, 182, 212, 0.2));
            border: 1px solid rgba(16, 185, 129, 0.5);
            color: #6EE7B7;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.9px;
            text-transform: uppercase;
            margin-bottom: 11px;
            box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
        }
        .lg-hero-title {
            font-size: 29px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.8px;
            margin: 0;
            line-height: 1.22;
        }
        .lg-hero-title span {
            background: linear-gradient(135deg, #10B981 0%, #34D399 35%, #38BDF8 70%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .lg-hero-desc {
            color: #94A3B8;
            font-size: 14.5px;
            line-height: 1.68;
            margin-top: 9px;
            margin-bottom: 0;
            max-width: 980px;
        }
        /* ============================================================
           4. CARDS & VERDICTS
           ============================================================ */
        .lg-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            padding: 24px 26px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            margin-bottom: 22px;
            transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .lg-card:hover {
            border-color: rgba(16, 185, 129, 0.35);
            box-shadow: 0 14px 45px rgba(0, 0, 0, 0.5), 0 0 22px rgba(16, 185, 129, 0.12);
        }
        .lg-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            padding-bottom: 13px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .lg-card-title {
            font-size: 17.5px;
            font-weight: 800;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0;
            letter-spacing: -0.4px;
        }
        .lg-verdict-card {
            border-radius: 20px;
            padding: 24px 28px;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(20px);
            transition: all 0.3s ease;
        }
        .lg-verdict-card.low {
            background: linear-gradient(135deg, rgba(6, 60, 45, 0.7) 0%, rgba(9, 15, 30, 0.95) 100%);
            border: 1px solid rgba(16, 185, 129, 0.5);
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.25), inset 0 1px 0 rgba(16, 185, 129, 0.4);
        }
        .lg-verdict-card.moderate {
            background: linear-gradient(135deg, rgba(85, 50, 10, 0.7) 0%, rgba(9, 15, 30, 0.95) 100%);
            border: 1px solid rgba(245, 158, 11, 0.5);
            box-shadow: 0 0 40px rgba(245, 158, 11, 0.25), inset 0 1px 0 rgba(245, 158, 11, 0.4);
        }
        .lg-verdict-card.high {
            background: linear-gradient(135deg, rgba(95, 20, 40, 0.75) 0%, rgba(9, 15, 30, 0.95) 100%);
            border: 1px solid rgba(244, 63, 94, 0.55);
            box-shadow: 0 0 40px rgba(244, 63, 94, 0.3), inset 0 1px 0 rgba(244, 63, 94, 0.4);
        }
        .lg-verdict-title {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.6px;
            margin: 0;
            color: #FFFFFF;
        }
        .lg-verdict-desc {
            font-size: 13.5px;
            color: #CBD5E1;
            margin-top: 5px;
            margin-bottom: 0;
        }
        .lg-verdict-badge {
            padding: 9px 20px;
            border-radius: 9999px;
            font-weight: 800;
            font-size: 13px;
            letter-spacing: 0.7px;
            text-transform: uppercase;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
        }
        .lg-verdict-badge.low {
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: #022C22;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.45);
        }
        .lg-verdict-badge.moderate {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: #451A03;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.45);
        }
        .lg-verdict-badge.high {
            background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.5);
        }
        /* ============================================================
           5. METRIC BOXES & GRIDS
           ============================================================ */
        .lg-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
            gap: 14px;
            margin-top: 14px;
            margin-bottom: 20px;
        }
        .lg-metric-box {
            background: rgba(14, 22, 40, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px 18px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(16px);
            transition: all 0.25s ease;
        }
        .lg-metric-box:hover {
            transform: translateY(-3px);
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 18px rgba(16, 185, 129, 0.18);
        }
        .lg-metric-label {
            font-size: 11px;
            color: #94A3B8;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.7px;
        }
        .lg-metric-val {
            font-size: 21px;
            font-weight: 800;
            color: #FFFFFF;
            margin-top: 6px;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: -0.5px;
        }
        /* ============================================================
           6. MASTER COMPARISON TABLE
           ============================================================ */
        .lg-table-container {
            background: rgba(12, 19, 36, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 20px;
            overflow-x: auto;
            margin-bottom: 26px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
        }
        .lg-comp-table {
            width: 100%;
            min-width: 1200px;
            border-collapse: collapse;
            font-size: 13.5px;
        }
        .lg-comp-table th {
            background: rgba(18, 28, 52, 0.95);
            color: #E2E8F0;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            font-size: 11px;
            padding: 15px 18px;
            text-align: center;
            border-bottom: 1px solid rgba(16, 185, 129, 0.28);
            white-space: nowrap;
        }
        .lg-comp-table th:first-child {
            text-align: left;
            padding-left: 22px;
        }
        .lg-comp-table td {
            padding: 14px 18px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            color: #CBD5E1;
            text-align: center;
            vertical-align: middle;
            transition: background 0.15s ease;
        }
        .lg-comp-table td:first-child {
            text-align: left;
            padding-left: 22px;
        }
        .lg-comp-table tr:hover td {
            background: rgba(16, 185, 129, 0.06);
        }
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(10, 16, 30, 0.88) !important;
            border: 1px solid rgba(16, 185, 129, 0.22) !important;
            border-radius: 18px !important;
            padding: 6px !important;
            gap: 6px !important;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 15px rgba(16, 185, 129, 0.08) !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            color: #94A3B8 !important;
            border: 1px solid transparent !important;
            transition: all 0.2s ease !important;
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #34D399 !important;
            background: rgba(16, 185, 129, 0.08) !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.28) 0%, rgba(56, 189, 248, 0.25) 100%) !important;
            border: 1px solid rgba(16, 185, 129, 0.55) !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.28) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090E1B 0%, #050811 100%) !important;
            border-right: 1px solid rgba(16, 185, 129, 0.20) !important;
            box-shadow: 8px 0 35px rgba(0, 0, 0, 0.65) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
            color: #34D399 !important;
            font-weight: 800 !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 1.2px !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 10px 14px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            margin-bottom: 5px !important;
            transition: all 0.22s ease !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(16, 185, 129, 0.12) !important;
            border-color: rgba(16, 185, 129, 0.35) !important;
        }
        /* Buttons & Form Submit */
        .stButton > button {
            border-radius: 13px !important;
            font-weight: 700 !important;
            padding: 10px 20px !important;
            transition: all 0.22s ease !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: linear-gradient(135deg, rgba(20, 32, 56, 0.9) 0%, rgba(12, 19, 36, 0.9) 100%) !important;
            color: #E2E8F0 !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            border-color: rgba(16, 185, 129, 0.6) !important;
            color: #34D399 !important;
            box-shadow: 0 8px 28px rgba(16, 185, 129, 0.3) !important;
        }
        button[kind="primaryFormSubmit"], button[kind="primary"] {
            background: linear-gradient(135deg, #10B981 0%, #059669 40%, #0284C7 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 6px 25px rgba(16, 185, 129, 0.45) !important;
            font-size: 15px !important;
            font-weight: 800 !important;
        }
        button[kind="primaryFormSubmit"]:hover, button[kind="primary"]:hover {
            background: linear-gradient(135deg, #34D399 0%, #10B981 50%, #0369A1 100%) !important;
            box-shadow: 0 10px 35px rgba(16, 185, 129, 0.65) !important;
        }
        /* Condition items */
        .lg-condition-item {
            display: flex;
            align-items: flex-start;
            gap: 13px;
            padding: 12px 16px;
            border-radius: 14px;
            background: rgba(16, 26, 48, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 10px;
            font-size: 13.5px;
            color: #E2E8F0;
        }
        .lg-condition-icon {
            font-size: 16px;
            line-height: 1;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
def calculate_risk_tier(prob):
    """Classifies default probability into clean institutional risk tiers."""
    if prob < 0.15:
        return "Tier 1: Minimal Risk", "low", "🟢", "Automatic Approval Recommended"
    elif prob < 0.35:
        return "Tier 2: Low Risk", "low", "🟢", "Standard Approval with Prime Pricing"
    elif prob < 0.55:
        return "Tier 3: Moderate Risk", "moderate", "🟡", "Requires Co-Signer or 15% Collateral"
    elif prob < 0.75:
        return "Tier 4: High Risk", "high", "🔴", "Manual Senior Underwriter Review Required"
    else:
        return "Tier 5: Critical Risk", "high", "🚨", "Automatic Decline Recommended"
def build_gauge_chart(default_prob_pct, title="DEFAULT RISK PROBABILITY", subtitle=""):
    """Generates a high-end Plotly Risk Speedometer Gauge."""
    val = round(default_prob_pct, 1)
    if val < 25:
        bar_color = "#10B981"
    elif val < 50:
        bar_color = "#F59E0B"
    elif val < 75:
        bar_color = "#F97316"
    else:
        bar_color = "#F43F5E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        number={'suffix': "%", 'font': {'size': 32, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
        delta={'reference': 50.0, 'increasing': {'color': '#F43F5E'}, 'decreasing': {'color': '#10B981'}},
        title={'text': f"<b style='margin-top:10px;'>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>{subtitle}</span>", 'font': {'size': 12, 'color': '#E2E8F0'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94A3B8', 'size': 9}},
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(30, 41, 59, 0.4)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [20, 45], 'color': "rgba(56, 189, 248, 0.15)"},
                {'range': [45, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [70, 100], 'color': "rgba(244, 63, 94, 0.18)"}
            ],
            'threshold': {
                'line': {'color': "#F43F5E", 'width': 3},
                'thickness': 0.85,
                'value': 50.0
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=10),
        height=280,
    )
    return fig
def build_feature_waterfall(feature_dict, lr_obj, scaler_obj):
    """Computes logistic log-odds contributions (beta_i * scaled_x_i)."""
    raw_vals = [feature_dict[name] for name in FEATURE_NAMES]
    scaled_vals = (np.array(raw_vals) - scaler_obj.mean_) / scaler_obj.scale_
    coefs = lr_obj.coef_[0]
    contributions = scaled_vals * coefs
    df_contrib = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "RawValue": raw_vals,
        "Impact": contributions
    }).sort_values(by="Impact", ascending=True)
    colors = ['#F43F5E' if x > 0 else '#10B981' for x in df_contrib["Impact"]]
    fig = go.Figure(go.Bar(
        x=df_contrib["Impact"],
        y=df_contrib["Feature"],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        hovertemplate="<b>%{y}</b><br>Log-Odds Impact: %{x:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="<b>Feature Risk Contribution (Log-Odds Impact)</b>", font=dict(size=13, color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="← Reduces Default Risk (Safe) | Increases Default Risk (Risky) →", font=dict(size=11, color="#94A3B8")),
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(tickfont=dict(color="#E2E8F0", size=11), gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=35, b=20),
        height=320,
    )
    return fig
def evaluate_decision_tree_path(feature_dict):
    """Evaluates applicant against exact depth-2 Decision Tree split rules."""
    age = feature_dict["Age"]
    income = feature_dict["Income"]
    ir = feature_dict["InterestRate"]
    age_thresh = 42.5
    ir_thresh = 12.645
    inc_thresh = 37589.5
    steps = []
    if age <= age_thresh:
        steps.append(f"Step 1: Borrower Age is **{age} years** (<= {age_thresh:.1f} years threshold). Branching to **Younger Borrower Sub-tree**.")
        if ir <= ir_thresh:
            steps.append(f"Step 2: Interest Rate is **{ir:.2f}%** (<= {ir_thresh:.2f}% threshold). Subprime pricing avoided.")
            pred_class = 0
            prob = 49.36
            outcome = "🟢 Classified as Non-Default (49.4% default probability, eligible for approval)."
        else:
            steps.append(f"Step 2: Interest Rate is **{ir:.2f}%** (> {ir_thresh:.2f}% threshold). High APR creates default pressure.")
            pred_class = 1
            prob = 66.92
            outcome = "🔴 Classified as High-Risk Default (66.9% default probability, decline/collateral required)."
    else:
        steps.append(f"Step 1: Borrower Age is **{age} years** (> {age_thresh:.1f} years threshold). Branching to **Mature Borrower Sub-tree**.")
        if income <= inc_thresh:
            steps.append(f"Step 2: Annual Income is **${income:,.0f}** (<= ${inc_thresh:,.0f} threshold). Insufficient cash-flow buffer.")
            pred_class = 1
            prob = 52.03
            outcome = "🔴 Classified as High-Risk Default (52.0% default probability, decline/co-signer required)."
        else:
            steps.append(f"Step 2: Annual Income is **${income:,.0f}** (> ${inc_thresh:,.0f} threshold). Substantial earnings cushion.")
            pred_class = 0
            prob = 32.02
            outcome = "🟢 Classified as Non-Default (32.0% default probability, prime credit approved)."
    return steps, outcome, pred_class, prob
def build_rf_feature_importance_chart(rf_obj):
    """Generates an interactive Plotly horizontal bar chart of feature importances."""
    importances = rf_obj.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances * 100
    }).sort_values(by="Importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation='h',
        marker=dict(
            color=df_imp["Importance"],
            colorscale='Viridis',
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Gini Importance: %{x:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="<b>Random Forest: Global Gini Feature Importance Ranking</b>", font=dict(size=13, color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Relative Importance (%)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(tickfont=dict(color="#E2E8F0", size=11)),
        margin=dict(l=10, r=10, t=35, b=20),
        height=320,
    )
    return fig
def evaluate_knn_neighbors(feature_dict, knn_obj, scaler_obj):
    """Finds the 8 nearest applicant neighbors in scaled Euclidean space."""
    input_df = pd.DataFrame([feature_dict])[FEATURE_NAMES]
    scaled_vector = scaler_obj.transform(input_df)
    distances, indices = knn_obj.kneighbors(scaled_vector, n_neighbors=8)
    dist_list = distances[0]
    idx_list = indices[0]
    labels = knn_obj._y[idx_list] if hasattr(knn_obj, '_y') else np.zeros(len(idx_list))
    weights = 1.0 / np.maximum(dist_list, 1e-6)
    normalized_weights = (weights / weights.sum()) * 100
    neighbor_records = []
    for rank, (idx, d, lbl, w) in enumerate(zip(idx_list, dist_list, labels, normalized_weights), start=1):
        neighbor_records.append({
            "Rank": f"#{rank}",
            "Distance (Euclidean)": round(d, 3),
            "Vote Weight (%)": f"{w:.1f}%",
            "Historical Outcome": "🔴 Defaulted" if lbl == 1 else "🟢 Repaid on Time",
            "Status Code": int(lbl)
        })
    df_neighbors = pd.DataFrame(neighbor_records)
    default_neighbor_count = int((labels == 1).sum())
    repaid_neighbor_count = int((labels == 0).sum())
    return df_neighbors, default_neighbor_count, repaid_neighbor_count
def calculate_monthly_emi(principal, annual_interest_rate_pct, tenure_months):
    """Computes monthly equated installment (EMI) and total repayment values."""
    if tenure_months <= 0:
        return 0, 0, 0
    if annual_interest_rate_pct <= 0:
        monthly_emi = principal / tenure_months
        total_payment = principal
        total_interest = 0
        return monthly_emi, total_payment, total_interest
    monthly_rate = (annual_interest_rate_pct / 100.0) / 12.0
    factor = (1 + monthly_rate) ** tenure_months
    monthly_emi = principal * monthly_rate * factor / (factor - 1)
    total_payment = monthly_emi * tenure_months
    total_interest = total_payment - principal
    return monthly_emi, total_payment, total_interest
def render_single_prediction_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">⚡ 9-Model Underwriting Intelligence Suite</span>
            <h1 class="lg-hero-title">Individual <span>Loan Default Risk</span> Analyzer</h1>
            <p class="lg-hero-desc">
                Evaluate applicant default risk in real time across <b>9 Production Classifiers</b>:
                <b>Gradient Boosting</b>, <b>Random Forest</b>, <b>Bagging</b>, <b>AdaBoost</b>, <b>SVC</b>, 
                <b>Logistic Regression</b>, <b>Decision Tree</b>, <b>Gaussian Naive Bayes</b>, and <b>KNN</b>. Inspect side-by-side consensus votes and explainability factors.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if not models_ready:
        st.error(f"❌ Model Loading Error: {models_error}")
        return
    col_m1, col_m2 = st.columns([1.3, 0.7])
    with col_m1:
        eval_mode = st.radio(
            "Select Evaluation Perspective:",
            [
                "🏆 9-Model Committee Consensus (All Models)",
                "🚀 Gradient Boosting (Top Default Recall 67.5%)",
                "🌳 Random Forest (Top ROC-AUC 0.748)",
                "📦 Bagging Classifier (60 Trees)",
                "⚡ AdaBoost (Top Accuracy 88.55%)",
                "⚔️ Support Vector Classifier (SVC)",
                "📍 K-Nearest Neighbors (Instance Cluster)",
                "🔹 Logistic Regression (Linear Calibrated)",
                "🌲 Decision Tree (Threshold Splits)",
                "🎯 Gaussian Naive Bayes (Probabilistic)"
            ],
            horizontal=True
        )
    st.markdown("##### ⚡ Quick Load Applicant Archetypes:")
    preset_cols = st.columns(len(APPLICANT_PRESETS))
    default_preset = APPLICANT_PRESETS["🌟 Prime Borrower"]
    for k, v in default_preset.items():
        if f"input_{k}" not in st.session_state:
            st.session_state[f"input_{k}"] = v
    for idx, (p_name, p_vals) in enumerate(APPLICANT_PRESETS.items()):
        with preset_cols[idx]:
            if st.button(p_name, key=f"btn_preset_{idx}", use_container_width=True):
                for k, v in p_vals.items():
                    st.session_state[f"input_{k}"] = v
                st.rerun()
    st.write("")
    col_form, col_results = st.columns([1.05, 0.95], gap="large")
    with col_form:
        st.markdown('<div class="lg-card-header"><h3 class="lg-card-title">📝 Applicant Financial Profile</h3></div>', unsafe_allow_html=True)
        with st.form("loan_input_form"):
            st.markdown("###### 👤 Personal & Employment Demographics")
            c1, c2 = st.columns(2)
            age = c1.slider("Age (Years)", 18, 75, int(st.session_state.get("input_Age", 35)))
            income = c2.number_input("Annual Gross Income ($)", min_value=10000, max_value=500000, value=int(st.session_state.get("input_Income", 75000)), step=5000)
            c3, c4 = st.columns(2)
            months_employed = c3.slider("Months Employed", 0, 140, int(st.session_state.get("input_MonthsEmployed", 48)))
            credit_score = c4.slider("Credit Score (FICO/Bureau)", 300, 850, int(st.session_state.get("input_CreditScore", 700)))
            st.markdown("###### 💳 Loan Terms & Credit Exposure")
            c5, c6 = st.columns(2)
            loan_amount = c5.number_input("Requested Loan Amount ($)", min_value=1000, max_value=500000, value=int(st.session_state.get("input_LoanAmount", 120000)), step=5000)
            interest_rate = c6.slider("Interest Rate (%)", 1.0, 30.0, float(st.session_state.get("input_InterestRate", 8.5)), step=0.25)
            c7, c8 = st.columns(2)
            loan_term = c7.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84], index=[12, 24, 36, 48, 60, 72, 84].index(int(st.session_state.get("input_LoanTerm", 36))) if int(st.session_state.get("input_LoanTerm", 36)) in [12, 24, 36, 48, 60, 72, 84] else 2)
            credit_lines = c8.slider("Active Credit Lines", 0, 20, int(st.session_state.get("input_NumCreditLines", 3)))
            st.markdown("###### 📊 Financial Leverage & Obligations")
            c9, c10, c11 = st.columns(3)
            dti_ratio = c9.slider("DTI Ratio", 0.05, 0.95, float(st.session_state.get("input_DTIRatio", 0.35)), step=0.01)
            mortgage = c10.selectbox("Has Mortgage?", ["No", "Yes"], index=0 if st.session_state.get("input_HasMortgage", "No") == "No" else 1)
            dependents = c11.selectbox("Has Dependents?", ["No", "Yes"], index=0 if st.session_state.get("input_HasDependents", "No") == "No" else 1)
            submit_btn = st.form_submit_button("⚡ Run Real-Time Underwriting Assessment", use_container_width=True, type="primary")
    feature_dict = {
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio,
        "HasMortgage": 1 if mortgage == "Yes" else 0,
        "HasDependents": 1 if dependents == "Yes" else 0
    }
    input_df = pd.DataFrame([feature_dict])[FEATURE_NAMES]
    scaled_vector = scaler.transform(input_df)
    raw_vector = input_df
    pred_results = {}
    if "rf" in models_pool:
        rf_p = float(models_pool["rf"].predict_proba(raw_vector)[0][1]) * 100
        rf_c = int(models_pool["rf"].predict(raw_vector)[0])
        pred_results["Random Forest"] = (rf_p, rf_c, "🌳")
    if "boosting" in models_pool:
        gb_p = float(models_pool["boosting"].predict_proba(raw_vector)[0][1]) * 100
        gb_c = int(models_pool["boosting"].predict(raw_vector)[0])
        pred_results["Gradient Boosting"] = (gb_p, gb_c, "🚀")
    if "bagging" in models_pool:
        bag_p = float(models_pool["bagging"].predict_proba(raw_vector)[0][1]) * 100
        bag_c = int(models_pool["bagging"].predict(raw_vector)[0])
        pred_results["Bagging"] = (bag_p, bag_c, "📦")
    if "adaboost" in models_pool:
        ada_p = float(models_pool["adaboost"].predict_proba(raw_vector)[0][1]) * 100
        ada_c = int(models_pool["adaboost"].predict(raw_vector)[0])
        pred_results["AdaBoost"] = (ada_p, ada_c, "⚡")
    if "lr" in models_pool:
        lr_p = float(models_pool["lr"].predict_proba(scaled_vector)[0][1]) * 100
        lr_c = int(models_pool["lr"].predict(scaled_vector)[0])
        pred_results["Logistic Regression"] = (lr_p, lr_c, "🔹")
    if "svc" in models_pool:
        svc_p = float(models_pool["svc"].predict_proba(scaled_vector)[0][1]) * 100
        svc_c = int(models_pool["svc"].predict(scaled_vector)[0])
        pred_results["Support Vector Classifier"] = (svc_p, svc_c, "⚔️")
    if "dt" in models_pool:
        dt_p = float(models_pool["dt"].predict_proba(raw_vector)[0][1]) * 100
        dt_c = int(models_pool["dt"].predict(raw_vector)[0])
        pred_results["Decision Tree"] = (dt_p, dt_c, "🌲")
    if "gnb" in models_pool:
        gnb_p = float(models_pool["gnb"].predict_proba(scaled_vector)[0][1]) * 100
        gnb_c = int(models_pool["gnb"].predict(scaled_vector)[0])
        pred_results["Gaussian Naive Bayes"] = (gnb_p, gnb_c, "🎯")
    if "knn" in models_pool:
        knn_p = float(models_pool["knn"].predict_proba(scaled_vector)[0][1]) * 100
        knn_c = int(models_pool["knn"].predict(scaled_vector)[0])
        pred_results["K-Nearest Neighbors"] = (knn_p, knn_c, "📍")
    total_models = len(pred_results)
    decline_votes = sum([c for p, c, icon in pred_results.values()])
    approve_votes = total_models - decline_votes
    avg_prob = np.mean([p for p, c, icon in pred_results.values()]) if total_models > 0 else 0
    avg_tier_label, avg_tier_class, avg_tier_icon, avg_tier_action = calculate_risk_tier(avg_prob / 100.0)
    emi_val, total_pay, total_int = calculate_monthly_emi(loan_amount, interest_rate, loan_term)
    dti_monthly_pct = round((emi_val / (income / 12)) * 100, 1) if income > 0 else 0
    with col_results:
        if "Committee" in eval_mode or "Consensus" in eval_mode:
            if decline_votes == 0:
                consensus_badge = f"🟢 Unanimous Approval ({approve_votes}/{total_models} Models Approve)"
                badge_color = "#10B981"
            elif decline_votes <= (total_models // 3):
                consensus_badge = f"🟢 Strong Majority Approval ({approve_votes}/{total_models} Models Approve)"
                badge_color = "#10B981"
            elif decline_votes <= (total_models // 2):
                consensus_badge = f"🟡 Committee Split Decision ({approve_votes} Approve / {decline_votes} Decline)"
                badge_color = "#F59E0B"
            elif decline_votes < total_models:
                consensus_badge = f"🔴 Majority Decline ({decline_votes}/{total_models} Flag Default Risk)"
                badge_color = "#F43F5E"
            else:
                consensus_badge = f"🚨 Unanimous Decline (All {total_models} Models Flag High Default Risk)"
                badge_color = "#F43F5E"
            st.markdown(f"""
                <div style="background: rgba(20, 30, 52, 0.85); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 16px; padding: 16px 20px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
                    <div>
                        <div style="font-weight: 700; font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.6px;">{total_models}-MODEL UNDERWRITING CONSENSUS:</div>
                        <div style="font-weight: 800; font-size: 15px; color: {badge_color}; margin-top: 3px;">{consensus_badge}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 11px; color: #94A3B8;">Mean Risk Index</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFF; font-family: 'JetBrains Mono';">{avg_prob:.1f}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            models_list = list(pred_results.items())
            chunk_size = 3
            for i in range(0, len(models_list), chunk_size):
                chunk = models_list[i:i + chunk_size]
                grid_cols = st.columns(len(chunk))
                for c_idx, (m_name, (prob, pred, icon)) in enumerate(chunk):
                    with grid_cols[c_idx]:
                        tier_lbl, tier_cls, t_icon, t_act = calculate_risk_tier(prob / 100.0)
                        card_accent = "#34D399" if (i + c_idx) % 3 == 0 else ("#38BDF8" if (i + c_idx) % 3 == 1 else "#A78BFA")
                        st.markdown(f"""
                            <div class="lg-metric-box" style="padding: 12px 14px; margin-bottom: 8px;">
                                <div style="font-size: 11.5px; font-weight: 700; color: {card_accent}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{icon} {m_name}</div>
                                <div style="font-size: 17px; font-weight: 800; color: {'#F43F5E' if pred == 1 else '#10B981'}; margin: 4px 0;">{prob:.1f}%</div>
                                <span class="lg-verdict-badge {tier_cls}" style="font-size: 9.5px; padding: 2px 8px;">{"DECLINE" if pred == 1 else "APPROVE"}</span>
                            </div>
                        """, unsafe_allow_html=True)
            gauge_main = build_gauge_chart(avg_prob, f"{total_models}-MODEL CONSENSUS RISK", f"{approve_votes} Approvals vs {decline_votes} Declines")
            st.plotly_chart(gauge_main, use_container_width=True, config={'displayModeBar': False})
        else:
            selected_key = None
            for name, (prob, pred, icon) in pred_results.items():
                if name.split()[0] in eval_mode or name in eval_mode:
                    selected_key = name
                    break
            if selected_key is None and len(pred_results) > 0:
                selected_key = list(pred_results.keys())[0]
            sel_prob, sel_pred, sel_icon = pred_results[selected_key]
            t_lbl, t_cls, t_icon, t_act = calculate_risk_tier(sel_prob / 100.0)
            st.markdown(f"""
                <div class="lg-verdict-card {t_cls}">
                    <div>
                        <div style="font-size: 11px; font-weight: 700; color: #34D399; letter-spacing: 0.5px; margin-bottom: 4px;">SELECTED MODEL: {sel_icon} {selected_key.upper()}</div>
                        <h3 class="lg-verdict-title">{t_icon} {t_lbl}</h3>
                        <p class="lg-verdict-desc">{t_act}</p>
                    </div>
                    <div class="lg-verdict-badge {t_cls}">
                        {"DECLINE / HIGH RISK" if sel_pred == 1 else "ELIGIBLE / LOW RISK"}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            gauge_single = build_gauge_chart(sel_prob, f"{selected_key.upper()} RISK PROBABILITY", f"Assigned: {'Decline (Class 1)' if sel_pred == 1 else 'Approve (Class 0)'}")
            st.plotly_chart(gauge_single, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
            <div class="lg-metric-grid">
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Monthly EMI</div>
                    <div class="lg-metric-val">${emi_val:,.0f}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">EMI / Income</div>
                    <div class="lg-metric-val">{dti_monthly_pct}%</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Total Payment</div>
                    <div class="lg-metric-val">${total_pay:,.0f}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Total Interest</div>
                    <div class="lg-metric-val" style="color: #F59E0B;">${total_int:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.write("")
    tab_boost, tab_rf, tab_knn, tab_lr, tab_dt, tab_policy, tab_whatif = st.tabs([
        "🚀 Gradient Boosting & Bagging",
        "🌳 Random Forest Drivers",
        "📍 KNN Neighborhood Analysis",
        "🔍 Logistic Regression Waterfall",
        "🌲 Decision Tree Branch Path",
        "📋 Underwriting Checklist & Policy",
        "🎛️ Counterfactual 'What-If' Simulator"
    ])
    with tab_boost:
        st.markdown("#### 🚀 Gradient Boosting & Bagging Ensembles")
        st.markdown("Boosting sequentially corrects classification errors on challenging applicants to achieve industry-leading **67.53% Default Recall**.")
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown("""
                <div style="background: rgba(18, 28, 50, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px;">
                    <h5 style="color: #34D399; margin-top: 0;">🚀 Gradient Boosting Metrics</h5>
                    <ul style="font-size: 13px; color: #CBD5E1; line-height: 1.8;">
                        <li><b>Default Recall:</b> 67.53% (4,005 defaults captured)</li>
                        <li><b>ROC-AUC Score:</b> 0.7480</li>
                        <li><b>Mechanism:</b> Sequential gradient residual shrinkage</li>
                        <li><b>Role:</b> High-loss portfolio charge-off prevention</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        with b_c2:
            st.markdown("""
                <div style="background: rgba(18, 28, 50, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px;">
                    <h5 style="color: #38BDF8; margin-top: 0;">📦 Bagging Ensemble Metrics</h5>
                    <ul style="font-size: 13px; color: #CBD5E1; line-height: 1.8;">
                        <li><b>Default Recall:</b> 66.21% (3,927 defaults captured)</li>
                        <li><b>ROC-AUC Score:</b> 0.7432</li>
                        <li><b>Mechanism:</b> 60 bootstrap aggregated decision trees</li>
                        <li><b>Role:</b> Variance mitigation on volatile loan segments</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    with tab_rf:
        st.markdown("#### 🌳 Random Forest: Gini Feature Importance Ranking")
        if "rf" in models_pool:
            rf_chart = build_rf_feature_importance_chart(models_pool["rf"])
            st.plotly_chart(rf_chart, use_container_width=True, config={'displayModeBar': False})
    with tab_knn:
        st.markdown("#### 📍 K-Nearest Neighbors: Local Peer Neighborhood (K=8)")
        if "knn" in models_pool:
            df_neighbors, def_count, rep_count = evaluate_knn_neighbors(feature_dict, models_pool["knn"], scaler)
            kn1, kn2 = st.columns([1.2, 0.8])
            with kn1:
                st.dataframe(df_neighbors, use_container_width=True, hide_index=True)
            with kn2:
                st.markdown(f"""
                    <div style="background: rgba(18, 28, 50, 0.7); padding: 18px; border-radius: 14px; border: 1px solid rgba(245, 158, 11, 0.35);">
                        <div style="font-size: 12px; color: #94A3B8;">Peer Cohort Status:</div>
                        <div style="font-size: 20px; font-weight: 800; color: #FFF; margin: 6px 0;">
                            <span style="color: #10B981;">{rep_count} Repaid</span> vs <span style="color: #F43F5E;">{def_count} Defaulted</span>
                        </div>
                        <div style="font-size: 12px; color: #CBD5E1; line-height: 1.6;">
                            Distance-weighted neighborhood consensus indicates local credit risk profile density.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    with tab_lr:
        st.markdown("#### 🔍 Logistic Regression: Log-Odds Factor Contributions")
        if "lr" in models_pool:
            wf_fig = build_feature_waterfall(feature_dict, models_pool["lr"], scaler)
            st.plotly_chart(wf_fig, use_container_width=True, config={'displayModeBar': False})
    with tab_dt:
        st.markdown("#### 🌲 Decision Tree: Exact Split Rule Evaluation")
        steps, outcome, p_class, prob_val = evaluate_decision_tree_path(feature_dict)
        for s in steps:
            st.markdown(f"""
                <div class="lg-condition-item">
                    <div class="lg-condition-icon">➡️</div>
                    <div>{s}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background: rgba(20, 32, 56, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px; margin-top: 10px;">
                <b>Final Tree Classification:</b> {outcome}
            </div>
        """, unsafe_allow_html=True)
    with tab_policy:
        st.markdown("#### 📑 Automated Institutional Policy Recommendations")
        recs = []
        if credit_score < 600:
            recs.append(("⚠️", "Subprime Credit Score: Require proof of last 12-month on-time rent/utility payments or add co-signer."))
        else:
            recs.append(("✅", "Prime Credit Standing: FICO score satisfies tier 1 automated credit policy."))
        if dti_ratio > 0.45:
            recs.append(("⚠️", "Elevated Debt-to-Income: Current total obligations exceed 45% of gross earnings."))
        else:
            recs.append(("✅", "Healthy Debt-to-Income: Leverage ratio is within standard institutional limits."))
        if interest_rate > 15.0:
            recs.append(("⚠️", "High Subprime Pricing: Interest rate exceeds 15%, significantly elevating repayment friction."))
        else:
            recs.append(("✅", "Competitive APR: Borrowing cost maintains manageable debt servicing."))
        if months_employed < 12:
            recs.append(("⚠️", "Short Employment Tenure: Applicant has under 1 year with current employer."))
        else:
            recs.append(("✅", "Stable Employment: Solid continuous job history established."))
        for icon, text in recs:
            st.markdown(f"""
                <div class="lg-condition-item">
                    <div class="lg-condition-icon">{icon}</div>
                    <div>{text}</div>
                </div>
            """, unsafe_allow_html=True)
    with tab_whatif:
        st.markdown("#### 🎛️ Live Sensitivity Testing (What-If Analysis)")
        wi_col1, wi_col2, wi_col3 = st.columns(3)
        delta_score = wi_col1.slider("Adjust Credit Score", -100, 100, 0, step=10)
        delta_rate = wi_col2.slider("Adjust Interest Rate (%)", -8.0, 8.0, 0.0, step=0.5)
        delta_dti = wi_col3.slider("Adjust DTI Ratio", -0.20, 0.20, 0.0, step=0.02)
        sim_feature_dict = feature_dict.copy()
        sim_feature_dict["CreditScore"] = max(300, min(850, feature_dict["CreditScore"] + delta_score))
        sim_feature_dict["InterestRate"] = max(1.0, feature_dict["InterestRate"] + delta_rate)
        sim_feature_dict["DTIRatio"] = max(0.05, min(0.95, feature_dict["DTIRatio"] + delta_dti))
        sim_df = pd.DataFrame([sim_feature_dict])[FEATURE_NAMES]
        sim_scaled = scaler.transform(sim_df)
        c_w1, c_w2, c_w3, c_w4 = st.columns(4)
        if "boosting" in models_pool:
            sim_boost_prob = float(models_pool["boosting"].predict_proba(sim_df)[0][1]) * 100
            diff = sim_boost_prob - pred_results["Gradient Boosting"][0]
            with c_w1:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">Boosting</div>
                        <div style="font-size: 16px; font-weight: 800; color: #FFF;">{sim_boost_prob:.1f}%</div>
                        <div style="font-size: 11px; color: {'#F43F5E' if diff > 0 else '#10B981'};">({'+' if diff > 0 else ''}{diff:.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
        if "rf" in models_pool:
            sim_rf_prob = float(models_pool["rf"].predict_proba(sim_df)[0][1]) * 100
            diff = sim_rf_prob - pred_results["Random Forest"][0]
            with c_w2:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">Random Forest</div>
                        <div style="font-size: 16px; font-weight: 800; color: #FFF;">{sim_rf_prob:.1f}%</div>
                        <div style="font-size: 11px; color: {'#F43F5E' if diff > 0 else '#10B981'};">({'+' if diff > 0 else ''}{diff:.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
        if "lr" in models_pool:
            sim_lr_prob = float(models_pool["lr"].predict_proba(sim_scaled)[0][1]) * 100
            diff = sim_lr_prob - pred_results["Logistic Regression"][0]
            with c_w3:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">Logistic Reg.</div>
                        <div style="font-size: 16px; font-weight: 800; color: #FFF;">{sim_lr_prob:.1f}%</div>
                        <div style="font-size: 11px; color: {'#F43F5E' if diff > 0 else '#10B981'};">({'+' if diff > 0 else ''}{diff:.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
        if "svc" in models_pool:
            sim_svc_prob = float(models_pool["svc"].predict_proba(sim_scaled)[0][1]) * 100
            diff = sim_svc_prob - pred_results["Support Vector Classifier"][0]
            with c_w4:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">SVC</div>
                        <div style="font-size: 16px; font-weight: 800; color: #FFF;">{sim_svc_prob:.1f}%</div>
                        <div style="font-size: 11px; color: {'#F43F5E' if diff > 0 else '#10B981'};">({'+' if diff > 0 else ''}{diff:.1f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
def render_model_comparison_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">⚖️ Algorithmic Governance & Master Benchmark</span>
            <h1 class="lg-hero-title">Model Comparison & <span>Performance Benchmark</span></h1>
            <p class="lg-hero-desc">
                Comprehensive empirical comparison of all <b>9 Classification, Boosting, Bagging, Probabilistic, and Linear Models</b> 
                across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Specificity on the holdout test set (51,070 records).
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="lg-metric-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 24px;">
            <div class="lg-metric-box" style="border-top: 3px solid #10B981;">
                <div class="lg-metric-label">🏆 Top Default Recall</div>
                <div class="lg-metric-val" style="color: #34D399;">67.53%</div>
                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">Gradient Boosting (4,005 Defaults Caught)</div>
            </div>
            <div class="lg-metric-box" style="border-top: 3px solid #38BDF8;">
                <div class="lg-metric-label">🏆 Top ROC-AUC Score</div>
                <div class="lg-metric-val" style="color: #38BDF8;">0.7481</div>
                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">Random Forest (#1 Discrimination Power)</div>
            </div>
            <div class="lg-metric-box" style="border-top: 3px solid #6366F1;">
                <div class="lg-metric-label">🏆 Top Overall Accuracy</div>
                <div class="lg-metric-val" style="color: #818CF8;">88.55%</div>
                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">AdaBoost (Sequential Boosting)</div>
            </div>
            <div class="lg-metric-box" style="border-top: 3px solid #F59E0B;">
                <div class="lg-metric-label">🏆 Top Specificity (Prime)</div>
                <div class="lg-metric-val" style="color: #FBBF24;">99.89%</div>
                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">Gaussian Naive Bayes (Lowest False Alarms)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📊 Comprehensive 9-Model Benchmark Matrix")
    rows_html = ""
    for m_name, m in BENCHMARK_METRICS.items():
        recall_val = m['Recall_Default'] * 100
        auc_val = m['ROC_AUC']
        acc_val = m['Accuracy'] * 100
        f1_val = m['F1_Default']
        spec_val = m['Specificity'] * 100
        fn_val = m['False_Negatives']
        tp_val = m['True_Positives']
        rows_html += f"""<tr>
<td style="text-align: left;"><div style="font-weight: 800; color: #34D399; font-size: 13.5px;">{m_name}</div><div style="font-size: 11px; color: #94A3B8;">{m['Type']}</div></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700; color: #FFFFFF;">{acc_val:.2f}%</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700; color: {'#10B981' if recall_val > 50 else '#CBD5E1'};">{recall_val:.2f}%</span><br><span style="font-size: 10px; color: #94A3B8;">({tp_val:,} caught)</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700;">{m['Precision_Default']*100:.2f}%</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700; color: #38BDF8;">{f1_val:.4f}</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700; color: #34D399;">{auc_val:.4f}</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700;">{spec_val:.2f}%</span></td>
<td><span style="font-family: 'JetBrains Mono', monospace; font-weight:700; color: {'#10B981' if fn_val < 2500 else '#F43F5E'};">{fn_val:,}</span></td>
<td style="text-align: left; font-size: 12px; color: #CBD5E1; line-height: 1.45;"><b>{m['Optimal_Role']}</b></td>
</tr>"""
    comp_table_html = f"""<div class="lg-table-container">
<table class="lg-comp-table">
<thead>
<tr>
<th style="text-align: left; min-width: 200px;">Model & Architecture</th>
<th>Accuracy</th>
<th>Default Recall<br><span style="font-size: 10px; color: #94A3B8;">(Class 1)</span></th>
<th>Default Precision<br><span style="font-size: 10px; color: #94A3B8;">(Class 1)</span></th>
<th>Default F1</th>
<th>ROC-AUC</th>
<th>Specificity</th>
<th>Missed Defaults</th>
<th style="text-align: left; min-width: 260px;">Strategic Underwriting Role</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>"""
    st.markdown(comp_table_html, unsafe_allow_html=True)
    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
        "📊 Multi-Metric Comparison Bar & Radar",
        "🎯 Confusion Matrices (Top Ensembles)",
        "📈 Empirical ROC Curves Benchmark",
        "⚔️ Live Multi-Model Applicant Arena"
    ])
    with comp_tab1:
        col_c1, col_c2 = st.columns([1.1, 0.9])
        with col_c1:
            m_keys = list(BENCHMARK_METRICS.keys())[:6]
            recalls = [BENCHMARK_METRICS[k]["Recall_Default"]*100 for k in m_keys]
            aucs = [BENCHMARK_METRICS[k]["ROC_AUC"]*100 for k in m_keys]
            accs = [BENCHMARK_METRICS[k]["Accuracy"]*100 for k in m_keys]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=m_keys, y=recalls, name='Default Recall (%)', marker_color='#10B981'))
            fig_bar.add_trace(go.Bar(x=m_keys, y=aucs, name='ROC-AUC (%)', marker_color='#38BDF8'))
            fig_bar.add_trace(go.Bar(x=m_keys, y=accs, name='Accuracy (%)', marker_color='#818CF8'))
            fig_bar.update_layout(
                title="<b>Key Metrics Across Production Models (%)</b>",
                barmode='group',
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title_font=dict(color="#FFFFFF", size=14),
                xaxis=dict(tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title="Score (%)", range=[0, 115], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
                legend=dict(font=dict(color="#E2E8F0"), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=380,
                margin=dict(l=10, r=10, t=60, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_c2:
            radar_cats = ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC-AUC', 'Specificity']
            fig_radar = go.Figure()
            for m_name, color in [("Gradient Boosting", "#10B981"), ("Random Forest", "#38BDF8"), ("Logistic Regression", "#818CF8"), ("AdaBoost", "#F59E0B")]:
                m = BENCHMARK_METRICS[m_name]
                vals = [m['Accuracy'], m['Recall_Default'], m['Precision_Default'], m['F1_Default'], m['ROC_AUC'], m['Specificity']]
                fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cats + [radar_cats[0]], fill='toself', name=m_name, line_color=color))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1.0], tickfont=dict(color="#94A3B8", size=9), gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(tickfont=dict(color="#E2E8F0", size=10), gridcolor="rgba(255,255,255,0.1)")
                ),
                title="<b>Capability Polygon Radar Comparison</b>",
                paper_bgcolor="rgba(0,0,0,0)",
                title_font=dict(color="#FFFFFF", size=14),
                legend=dict(font=dict(color="#E2E8F0")),
                height=380,
                margin=dict(l=25, r=25, t=40, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    with comp_tab2:
        st.markdown("#### 🎯 Side-by-Side Confusion Matrices (Top Models)")
        cm_r1, cm_r2 = st.columns(2)
        with cm_r1:
            gb = BENCHMARK_METRICS["Gradient Boosting"]
            gb_cm = np.array([[gb['True_Negatives'], gb['False_Positives']], [gb['False_Negatives'], gb['True_Positives']]])
            fig_gb_cm = px.imshow(gb_cm, labels=dict(x="Predicted", y="Actual"), x=['Safe (0)', 'Default (1)'], y=['Safe (0)', 'Default (1)'], color_continuous_scale="Mint", title="<b>Gradient Boosting (Top Recall: 67.5%)</b>")
            fig_gb_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#34D399", size=13), height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_gb_cm, use_container_width=True)
        with cm_r2:
            rf = BENCHMARK_METRICS["Random Forest"]
            rf_cm = np.array([[rf['True_Negatives'], rf['False_Positives']], [rf['False_Negatives'], rf['True_Positives']]])
            fig_rf_cm = px.imshow(rf_cm, labels=dict(x="Predicted", y="Actual"), x=['Safe (0)', 'Default (1)'], y=['Safe (0)', 'Default (1)'], color_continuous_scale="Teal", title="<b>Random Forest (300 Trees, AUC: 0.748)</b>")
            fig_rf_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#38BDF8", size=13), height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_rf_cm, use_container_width=True)
    with comp_tab3:
        st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Benchmark Curves")
        lr_fpr = [0.0, 0.0095, 0.02, 0.049, 0.0801, 0.1169, 0.1624, 0.2161, 0.2802, 0.3531, 0.4464, 0.5666, 0.7163, 1.0]
        lr_tpr = [0.0, 0.0701, 0.1281, 0.2295, 0.315, 0.3998, 0.4802, 0.5616, 0.6385, 0.7146, 0.7872, 0.8601, 0.9314, 1.0]
        rf_fpr = [0.0, 0.0084, 0.0304, 0.0597, 0.097, 0.1374, 0.1845, 0.2443, 0.3131, 0.3993, 0.502, 0.6355, 0.8315, 1.0]
        rf_tpr = [0.0, 0.0749, 0.1779, 0.275, 0.3617, 0.4458, 0.5276, 0.6061, 0.6815, 0.755, 0.8272, 0.8966, 0.9664, 1.0]
        gb_fpr = [0.0, 0.0080, 0.0280, 0.0550, 0.092, 0.1320, 0.1790, 0.2380, 0.3050, 0.3910, 0.495, 0.6280, 0.8250, 1.0]
        gb_tpr = [0.0, 0.0790, 0.1850, 0.2880, 0.3750, 0.4590, 0.5410, 0.6190, 0.6950, 0.7680, 0.838, 0.9050, 0.9710, 1.0]
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=gb_fpr, y=gb_tpr, mode='lines', name='Gradient Boosting (AUC = 0.7480)', line=dict(color='#10B981', width=3.5)))
        roc_fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name='Random Forest (AUC = 0.7481)', line=dict(color='#38BDF8', width=2.5)))
        roc_fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines', name='Logistic Regression (AUC = 0.7440)', line=dict(color='#818CF8', width=2.5)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance (AUC = 0.5000)', line=dict(color='#64748B', width=1.5, dash='dash')))
        roc_fig.update_layout(
            title="<b>ROC Curves Comparison (True Positive Rate vs False Positive Rate)</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_font=dict(color="#FFFFFF", size=14),
            xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8"), range=[0, 1.02]),
            yaxis=dict(title="True Positive Rate (Recall)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8"), range=[0, 1.02]),
            legend=dict(font=dict(color="#E2E8F0"), bgcolor="rgba(15, 23, 42, 0.8)"),
            height=380,
            margin=dict(l=10, r=10, t=40, b=20)
        )
        st.plotly_chart(roc_fig, use_container_width=True)
    with comp_tab4:
        st.markdown("#### ⚔️ Interactive Live Multi-Model Applicant Arena")
        a_col1, a_col2, a_col3 = st.columns(3)
        arena_age = a_col1.slider("Borrower Age", 18, 75, 40, key="ar_age")
        arena_income = a_col2.number_input("Annual Income ($)", 15000, 300000, 65000, step=5000, key="ar_inc")
        arena_rate = a_col3.slider("Interest Rate (%)", 2.0, 28.0, 14.5, step=0.5, key="ar_rate")
        arena_dict = {
            "Age": arena_age, "Income": arena_income, "LoanAmount": 120000,
            "CreditScore": 620, "MonthsEmployed": 36, "NumCreditLines": 4,
            "InterestRate": arena_rate, "LoanTerm": 36, "DTIRatio": 0.45,
            "HasMortgage": 0, "HasDependents": 0
        }
        arena_df = pd.DataFrame([arena_dict])[FEATURE_NAMES]
        arena_scaled = scaler.transform(arena_df)
        ar_c1, ar_c2, ar_c3, ar_c4 = st.columns(4)
        if "boosting" in models_pool:
            p = float(models_pool["boosting"].predict_proba(arena_df)[0][1]) * 100
            c = int(models_pool["boosting"].predict(arena_df)[0])
            with ar_c1:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">🚀 Gradient Boosting</div>
                        <div style="font-size: 18px; font-weight: 800; color: {'#F43F5E' if c == 1 else '#10B981'}; margin: 4px 0;">{p:.1f}%</div>
                        <span class="lg-verdict-badge {'high' if c == 1 else 'low'}" style="font-size: 9px; padding: 2px 7px;">{"DECLINE" if c == 1 else "APPROVE"}</span>
                    </div>
                """, unsafe_allow_html=True)
        if "rf" in models_pool:
            p = float(models_pool["rf"].predict_proba(arena_df)[0][1]) * 100
            c = int(models_pool["rf"].predict(arena_df)[0])
            with ar_c2:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">🌳 Random Forest</div>
                        <div style="font-size: 18px; font-weight: 800; color: {'#F43F5E' if c == 1 else '#10B981'}; margin: 4px 0;">{p:.1f}%</div>
                        <span class="lg-verdict-badge {'high' if c == 1 else 'low'}" style="font-size: 9px; padding: 2px 7px;">{"DECLINE" if c == 1 else "APPROVE"}</span>
                    </div>
                """, unsafe_allow_html=True)
        if "lr" in models_pool:
            p = float(models_pool["lr"].predict_proba(arena_scaled)[0][1]) * 100
            c = int(models_pool["lr"].predict(arena_scaled)[0])
            with ar_c3:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">🔹 Logistic Reg.</div>
                        <div style="font-size: 18px; font-weight: 800; color: {'#F43F5E' if c == 1 else '#10B981'}; margin: 4px 0;">{p:.1f}%</div>
                        <span class="lg-verdict-badge {'high' if c == 1 else 'low'}" style="font-size: 9px; padding: 2px 7px;">{"DECLINE" if c == 1 else "APPROVE"}</span>
                    </div>
                """, unsafe_allow_html=True)
        if "svc" in models_pool:
            p = float(models_pool["svc"].predict_proba(arena_scaled)[0][1]) * 100
            c = int(models_pool["svc"].predict(arena_scaled)[0])
            with ar_c4:
                st.markdown(f"""
                    <div class="lg-metric-box">
                        <div class="lg-metric-label">⚔️ SVC (Max-Margin)</div>
                        <div style="font-size: 18px; font-weight: 800; color: {'#F43F5E' if c == 1 else '#10B981'}; margin: 4px 0;">{p:.1f}%</div>
                        <span class="lg-verdict-badge {'high' if c == 1 else 'low'}" style="font-size: 9px; padding: 2px 7px;">{"DECLINE" if c == 1 else "APPROVE"}</span>
                    </div>
                """, unsafe_allow_html=True)
def render_batch_analytics_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">📁 Batch Portfolio Processing</span>
            <h1 class="lg-hero-title">Bulk Loan Portfolio <span>Risk Auditor</span></h1>
            <p class="lg-hero-desc">
                Upload CSV loan portfolios or run automated underwriting audits across all production models concurrently.
            </p>
        </div>
    """, unsafe_allow_html=True)
    if not models_ready:
        st.error("Models not loaded.")
        return
    uploaded_file = st.file_uploader("Upload Loan Batch (CSV)", type=["csv"])
    use_sample = st.checkbox("Or run live audit on sample portfolio (1,500 loan records)", value=True if not uploaded_file else False)
    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df):,} records.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            return
    elif use_sample:
        batch_df = load_eda_dataset(sample_size=1500)
    if batch_df is not None:
        df_proc = batch_df.copy()
        for col in ["HasMortgage", "HasDependents"]:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        missing_feats = [f for f in FEATURE_NAMES if f not in df_proc.columns]
        if missing_feats:
            st.error(f"Missing required feature columns: {missing_feats}")
            return
        X_b_raw = df_proc[FEATURE_NAMES]
        X_b_scaled = scaler.transform(X_b_raw)
        df_results = batch_df.copy()
        decline_counts = np.zeros(len(df_results))
        total_eval_models = 0
        if "boosting" in models_pool:
            gb_probs = models_pool["boosting"].predict_proba(X_b_raw)[:, 1] * 100
            gb_preds = models_pool["boosting"].predict(X_b_raw)
            df_results["Boosting_Risk_%"] = gb_probs.round(1)
            decline_counts += gb_preds
            total_eval_models += 1
        if "rf" in models_pool:
            rf_probs = models_pool["rf"].predict_proba(X_b_raw)[:, 1] * 100
            rf_preds = models_pool["rf"].predict(X_b_raw)
            df_results["RF_Risk_%"] = rf_probs.round(1)
            decline_counts += rf_preds
            total_eval_models += 1
        if "bagging" in models_pool:
            bag_probs = models_pool["bagging"].predict_proba(X_b_raw)[:, 1] * 100
            bag_preds = models_pool["bagging"].predict(X_b_raw)
            df_results["Bagging_Risk_%"] = bag_probs.round(1)
            decline_counts += bag_preds
            total_eval_models += 1
        if "adaboost" in models_pool:
            ada_probs = models_pool["adaboost"].predict_proba(X_b_raw)[:, 1] * 100
            ada_preds = models_pool["adaboost"].predict(X_b_raw)
            df_results["AdaBoost_Risk_%"] = ada_probs.round(1)
            decline_counts += ada_preds
            total_eval_models += 1
        if "lr" in models_pool:
            lr_probs = models_pool["lr"].predict_proba(X_b_scaled)[:, 1] * 100
            lr_preds = models_pool["lr"].predict(X_b_scaled)
            df_results["LR_Risk_%"] = lr_probs.round(1)
            decline_counts += lr_preds
            total_eval_models += 1
        if "svc" in models_pool:
            svc_probs = models_pool["svc"].predict_proba(X_b_scaled)[:, 1] * 100
            svc_preds = models_pool["svc"].predict(X_b_scaled)
            df_results["SVC_Risk_%"] = svc_probs.round(1)
            decline_counts += svc_preds
            total_eval_models += 1
        if "dt" in models_pool:
            dt_probs = models_pool["dt"].predict_proba(X_b_raw)[:, 1] * 100
            dt_preds = models_pool["dt"].predict(X_b_raw)
            df_results["DT_Risk_%"] = dt_probs.round(1)
            decline_counts += dt_preds
            total_eval_models += 1
        if "gnb" in models_pool:
            gnb_probs = models_pool["gnb"].predict_proba(X_b_scaled)[:, 1] * 100
            gnb_preds = models_pool["gnb"].predict(X_b_scaled)
            df_results["GNB_Risk_%"] = gnb_probs.round(1)
            decline_counts += gnb_preds
            total_eval_models += 1
        n_m = max(total_eval_models, 1)
        verdicts = []
        for d in decline_counts:
            ratio = d / n_m
            if ratio == 0:
                verdicts.append(f"🟢 Unanimous Approve (0/{n_m})")
            elif ratio <= 0.35:
                verdicts.append(f"🟢 Majority Approve ({int(n_m - d)}/{n_m})")
            elif ratio < 0.60:
                verdicts.append(f"🟡 Split Decision ({int(d)} Decline)")
            elif ratio < 1.0:
                verdicts.append(f"🔴 Majority Decline ({int(d)}/{n_m})")
            else:
                verdicts.append(f"🚨 Unanimous Decline ({n_m}/{n_m})")
        df_results["Consensus_Verdict"] = verdicts
        total_apps = len(df_results)
        total_loan_vol = df_results["LoanAmount"].sum() if "LoanAmount" in df_results.columns else 0
        total_declines = sum([1 for v in verdicts if "Decline" in v])
        st.markdown(f"""
            <div class="lg-metric-grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Applications Audited</div>
                    <div class="lg-metric-val">{total_apps:,}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Total Capital Exposure</div>
                    <div class="lg-metric-val">${total_loan_vol:,.0f}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Boosting Flagged</div>
                    <div class="lg-metric-val" style="color: #34D399;">{int(df_results.get('Boosting_Risk_%', pd.Series([0]))[df_results.get('Boosting_Risk_%', pd.Series([0])) > 50].count()):,}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Consensus Declines</div>
                    <div class="lg-metric-val" style="color: #F43F5E;">{total_declines:,}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.dataframe(df_results, use_container_width=True, height=380)
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Multi-Model Audit Report (CSV)",
            data=csv_data,
            file_name="Riskora_MultiModel_Audit_Report.csv",
            mime="text/csv",
            use_container_width=True
        )
def render_eda_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">📊 Portfolio Demographics</span>
            <h1 class="lg-hero-title">Portfolio & Market <span>Exploratory Analytics</span></h1>
            <p class="lg-hero-desc">
                Interactive macroeconomic insights into historical loan distributions, interest rate sensitivity, 
                and credit default correlations across 255,347 borrower profiles.
            </p>
        </div>
    """, unsafe_allow_html=True)
    df_eda = load_eda_dataset(sample_size=15000)
    if df_eda is None:
        st.error("Dataset not found.")
        return
    st.markdown(f"""
        <div class="lg-metric-grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="lg-metric-box">
                <div class="lg-metric-label">Analyzed Records</div>
                <div class="lg-metric-val">{len(df_eda):,}</div>
            </div>
            <div class="lg-metric-box">
                <div class="lg-metric-label">Historical Default Rate</div>
                <div class="lg-metric-val" style="color: #F43F5E;">{(df_eda['LoanDefault'].mean()*100):.2f}%</div>
            </div>
            <div class="lg-metric-box">
                <div class="lg-metric-label">Median Income</div>
                <div class="lg-metric-val">${df_eda['Income'].median():,.0f}</div>
            </div>
            <div class="lg-metric-box">
                <div class="lg-metric-label">Median Credit Score</div>
                <div class="lg-metric-val">{df_eda['CreditScore'].median():.0f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    eda_tab1, eda_tab2, eda_tab3 = st.tabs([
        "📈 Risk Factor Distributions",
        "🔗 Feature Correlation Heatmap",
        "👥 Default Demographics & Scatter"
    ])
    with eda_tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig_age = px.histogram(df_eda, x="Age", color="LoanDefault", barmode="overlay", title="<b>Age Distribution by Default Status</b>", color_discrete_map={0: "#10B981", 1: "#F43F5E"}, opacity=0.75)
            fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#FFFFFF", size=13), height=300)
            st.plotly_chart(fig_age, use_container_width=True)
        with c2:
            fig_ir = px.histogram(df_eda, x="InterestRate", color="LoanDefault", barmode="overlay", title="<b>Interest Rate (%) Distribution by Default Status</b>", color_discrete_map={0: "#38BDF8", 1: "#F43F5E"}, opacity=0.75)
            fig_ir.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#FFFFFF", size=13), height=300)
            st.plotly_chart(fig_ir, use_container_width=True)
    with eda_tab2:
        numeric_cols = FEATURE_NAMES + ["LoanDefault"]
        corr_matrix = df_eda[[c for c in numeric_cols if c in df_eda.columns]].corr().round(3)
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="<b>Feature Correlation Matrix with LoanDefault</b>", zmin=-0.25, zmax=0.25)
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#FFFFFF", size=14), height=460)
        st.plotly_chart(fig_corr, use_container_width=True)
    with eda_tab3:
        scatter_sample = df_eda.sample(min(2000, len(df_eda)), random_state=42)
        fig_scatter = px.scatter(scatter_sample, x="Income", y="LoanAmount", color="LoanDefault", color_discrete_map={0: "#10B981", 1: "#F43F5E"}, opacity=0.6, title="<b>Income vs. Requested Loan Amount</b>", hover_data=["CreditScore", "InterestRate", "DTIRatio"])
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title_font=dict(color="#FFFFFF", size=14), height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)
def render_model_diagnostics_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">🧠 Transparent AI & Architecture</span>
            <h1 class="lg-hero-title">Model <span>Diagnostics & Parameters</span></h1>
            <p class="lg-hero-desc">
                Inspect mathematical formulations, hyperparameters, standardized beta weights, and ensemble architectures.
            </p>
        </div>
    """, unsafe_allow_html=True)
    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs([
        "🚀 Boosting & Random Forest",
        "🔹 Logistic Regression & SVC",
        "📍 KNN Instance Memory",
        "🌲 Decision Tree Rules"
    ])
    with diag_tab1:
        st.markdown("#### 🚀 Boosting & Random Forest Hyperparameters")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RF Estimators", "300 Trees")
        c2.metric("RF Max Depth", "10 Levels")
        c3.metric("Boosting Max Iter", "160 Iterations")
        c4.metric("Boosting Class Weight", "Balanced")
    with diag_tab2:
        st.markdown("#### 🔹 Standardized Beta Weights & Odds Ratios")
        if "lr" in models_pool:
            coefs = models_pool["lr"].coef_[0]
            df_w = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Coefficient (Beta)": coefs.round(4),
                "Odds Ratio": np.exp(coefs).round(4),
                "Direction": ["📈 Increases Default Risk" if c > 0 else "📉 Protects / Lowers Risk" for c in coefs]
            }).sort_values(by="Coefficient (Beta)", ascending=False)
            st.dataframe(df_w, use_container_width=True)
    with diag_tab3:
        st.markdown("#### 📍 KNN Distance Formulation")
        st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{11} \left(\frac{x_i - y_i}{\sigma_i}\right)^2}")
        st.latex(r"P(\text{Default} = 1 \mid x) = \frac{\sum_{k \in \mathcal{N}_8} w_k \cdot y_k}{\sum_{k \in \mathcal{N}_8} w_k}")
    with diag_tab4:
        st.markdown("#### 🌲 Decision Node Rules")
        dt_rules = pd.DataFrame({
            "Node ID": ["Root Node 0", "Left Child Node 1", "Right Child Node 2"],
            "Splitting Feature": ["Age", "InterestRate", "Income"],
            "Real-World Threshold": ["<= 42.5 Years", "<= 12.65% APR", "<= $37,589.5 Income"],
            "Strategic Decision": ["Young vs Mature borrower segmentation", "Prime vs Subprime APR pricing", "Income buffer vulnerability"]
        })
        st.dataframe(dt_rules, use_container_width=True, hide_index=True)
def render_affordability_calculator_page():
    st.markdown("""
        <div class="lg-hero">
            <span class="lg-hero-badge">🧮 Loan Structuring Simulator</span>
            <h1 class="lg-hero-title">EMI & Loan <span>Affordability Optimizer</span></h1>
            <p class="lg-hero-desc">
                Structure institutional loan offerings with real-time Equated Monthly Installment (EMI) calculations, 
                amortization curves, and debt-to-income sensitivity thresholds.
            </p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown("#### ⚙️ Loan Parameters")
        calc_loan = st.number_input("Proposed Principal ($)", min_value=5000, max_value=500000, value=100000, step=5000)
        calc_rate = st.slider("Annual Percentage Rate (APR %)", 2.0, 30.0, 9.5, step=0.25)
        calc_term = st.selectbox("Tenure (Months)", [12, 24, 36, 48, 60, 72, 84, 120], index=2)
        calc_inc = st.number_input("Applicant Monthly Gross Income ($)", min_value=1000, max_value=50000, value=6500, step=250)
        calc_exist_debt = st.number_input("Existing Monthly Debt Commitments ($)", min_value=0, max_value=10000, value=900, step=100)
    monthly_emi, total_pay, total_int = calculate_monthly_emi(calc_loan, calc_rate, calc_term)
    new_total_debt = calc_exist_debt + monthly_emi
    new_dti = (new_total_debt / calc_inc) * 100 if calc_inc > 0 else 0
    with col2:
        st.markdown(f"""
            <div class="lg-verdict-card {'low' if new_dti <= 36 else ('moderate' if new_dti <= 45 else 'high')}">
                <div>
                    <div style="font-size: 11px; font-weight: 700; color: #94A3B8;">MONTHLY OBLIGATION</div>
                    <h3 class="lg-verdict-title">${monthly_emi:,.2f} / month</h3>
                    <p class="lg-verdict-desc">Projected New DTI Ratio: <b>{new_dti:.1f}%</b></p>
                </div>
                <div class="lg-verdict-badge {'low' if new_dti <= 36 else ('moderate' if new_dti <= 45 else 'high')}">
                    {'AFFORDABLE' if new_dti <= 36 else ('BORDERLINE' if new_dti <= 45 else 'OVER-LEVERAGED')}
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="lg-metric-grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Total Payment</div>
                    <div class="lg-metric-val">${total_pay:,.0f}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Total Interest</div>
                    <div class="lg-metric-val" style="color: #F59E0B;">${total_int:,.0f}</div>
                </div>
                <div class="lg-metric-box">
                    <div class="lg-metric-label">Interest / Principal</div>
                    <div class="lg-metric-val">{(total_int/calc_loan)*100:.1f}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
def main():
    inject_custom_css()
    status_class = "offline" if not models_ready else ""
    loaded_count = len(models_pool) if models_ready else 0
    status_text = f"{loaded_count} Production ML Models Online" if models_ready else "Models Offline"
    st.markdown(f"""
        <div class="lg-navbar">
            <div class="lg-brand">
                <div class="lg-brand-logo">🛡️</div>
                <div>
                    <div class="lg-brand-title">Riskora AI</div>
                    <div class="lg-brand-subtitle">9-Model Credit Risk & Underwriting Benchmark Platform</div>
                </div>
            </div>
            <div class="lg-status-pill {status_class}">
                <div class="lg-status-dot {status_class}"></div>
                <span>{status_text}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 14px 10px 18px 10px; background: rgba(14, 24, 44, 0.75); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: 18px; margin-bottom: 16px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);">
            <div style="width: 46px; height: 46px; margin: 0 auto 10px auto; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #10B981 0%, #06B6D4 50%, #6366F1 100%); font-size: 24px; box-shadow: 0 0 22px rgba(16, 185, 129, 0.55); border: 1px solid rgba(255, 255, 255, 0.25);">🛡️</div>
            <div style="font-size: 19px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">Riskora AI</div>
            <div style="font-size: 11px; font-weight: 700; color: #34D399; letter-spacing: 0.6px; text-transform: uppercase; margin-top: 3px;">9-Model Underwriting Suite</div>
        </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        [
            "🎯 Single Loan Risk Assessment",
            "⚖️ Model Comparison & Benchmark",
            "📁 Batch Portfolio Risk Auditor",
            "📊 Portfolio & Market Analytics (EDA)",
            "🧠 Model Diagnostics & Architecture",
            "🧮 EMI & Loan Affordability Calculator"
        ],
        index=0
    )
    st.sidebar.markdown(f"""
        <div style="background: rgba(14, 22, 40, 0.8); border: 1px solid rgba(16, 185, 129, 0.22); border-radius: 16px; padding: 16px 18px; margin-top: 18px; font-size: 11.5px; color: #94A3B8; line-height: 1.85;">
            <div style="font-size: 11px; font-weight: 800; color: #34D399; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                <span>⚙️ ACTIVE ENGINE</span>
                <span style="display: inline-block; padding: 2px 9px; border-radius: 9999px; font-size: 10px; font-weight: 700; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.45); color: #34D399;">● {loaded_count} Online</span>
            </div>
            • <b>Gradient Boosting:</b> Recall 67.5%<br>
            • <b>Random Forest:</b> 300 Trees (AUC 0.748)<br>
            • <b>Bagging:</b> 60 Bootstrap Trees<br>
            • <b>AdaBoost:</b> 88.55% Accuracy<br>
            • <b>SVC:</b> Calibrated Max-Margin<br>
            • <b>Logistic Reg:</b> Linear Sigmoid<br>
            • <b>Decision Tree:</b> Rule Thresholds<br>
            • <b>Gaussian NB:</b> Probabilistic Prior<br>
            • <b>KNN:</b> K=8 Instance Memory
        </div>
    """, unsafe_allow_html=True)
    if page == "🎯 Single Loan Risk Assessment":
        render_single_prediction_page()
    elif page == "⚖️ Model Comparison & Benchmark":
        render_model_comparison_page()
    elif page == "📁 Batch Portfolio Risk Auditor":
        render_batch_analytics_page()
    elif page == "📊 Portfolio & Market Analytics (EDA)":
        render_eda_page()
    elif page == "🧠 Model Diagnostics & Architecture":
        render_model_diagnostics_page()
    elif page == "🧮 EMI & Loan Affordability Calculator":
        render_affordability_calculator_page()
if __name__ == "__main__":
    main()
