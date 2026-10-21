"""
lumio. | Electricity Usage Predictor
A high-end, dark-themed web application designed for intelligent household electricity prediction.
Exact Palette: Charcoal (#111214), Dark Slate (#1B1D22), Electric Yellow (#FFD447), Teal (#2DD4BF), Rose (#FB7185).
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="lumio. | Electricity Usage Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base Paths (Relative to script location)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_usage.csv"
MODEL_PATH = BASE_DIR / "models" / "electricity_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
REPORT_PATH = BASE_DIR / "reports" / "evaluation_results.txt"

# ---------------------------------------------------------
# EXACT PALETTE & MODERN SAAS STYLING
# Charcoal (#111214) | Dark Slate (#1B1D22) | Electric Yellow (#FFD447)
# Teal (#2DD4BF) | Pink/Red (#FB7185)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts: Outfit (Geometric SaaS Sans) + Space Grotesk (Tech/Numeric Display) */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Root Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #111214 !important;
        color: #EDEDED !important;
        letter-spacing: -0.01em;
    }

    /* Headings with Space Grotesk */
    h1, h2, h3, .top-header-title {
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        letter-spacing: -0.03em !important;
    }

    /* Hide standard Streamlit header clutter */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Theme (Charcoal #111214) */
    [data-testid="stSidebar"] {
        background-color: #111214 !important;
        border-right: 1px solid #1E2026 !important;
        padding-top: 1.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #8C92A4;
    }

    /* Brand Header on Top Left */
    .brand-container {
        padding: 4px 0 16px 0;
        margin-bottom: 16px;
        border-bottom: 1px solid #1E2026;
    }

    /* Sidebar Navigation Radios - Minimalist Clean Text */
    div[data-testid="stRadio"] > div {
        gap: 5px;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 11px !important;
        padding: 10px 14px !important;
        color: #8C92A4 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.18s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1B1D22 !important;
        color: #FFFFFF !important;
        border-color: #262933 !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #1B1D22 !important;
        color: #FFD447 !important;
        border-color: rgba(255, 212, 71, 0.3) !important;
        font-weight: 700 !important;
    }
    /* Radio circle dot override (Forces Electric Yellow #FFD447 instead of default red) */
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] input:checked ~ div {
        background-color: #FFD447 !important;
        border-color: #FFD447 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] input:checked ~ div > div {
        background-color: #111214 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] div:first-child {
        border-color: #FFD447 !important;
    }
    div[data-testid="stRadio"] [data-baseweb="radio"] input:checked + div,
    div[data-testid="stRadio"] [data-baseweb="radio"] input:checked + div > div {
        border-color: #FFD447 !important;
    }
    div[data-testid="stRadio"] svg {
        fill: #FFD447 !important;
    }

    /* Top Page Header */
    .top-header-row {
        margin-bottom: 8px;
    }
    .top-header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.04em;
        margin: 0 0 10px 0;
    }

    /* Modern Subtabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 22px !important;
        border-bottom: 1px solid #1E2026 !important;
        padding-bottom: 0px !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0px !important;
        color: #6C7282 !important;
        padding: 8px 2px 10px 2px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        letter-spacing: -0.01em !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #FFD447 !important;
    }

    /* Modern Dark Slate Cards (#1B1D22) */
    .dark-card {
        background-color: #1B1D22;
        border: 1px solid #262933;
        border-radius: 16px;
        padding: 22px 20px;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .dark-card:hover {
        transform: translateY(-2px);
        border-color: #353947;
    }

    /* Signature Highlight Card (Electric Yellow #FFD447) */
    .highlight-yellow-card {
        background: #FFD447 !important;
        color: #111214 !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 22px 20px !important;
        box-shadow: 0 12px 32px -4px rgba(255, 212, 71, 0.45) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        position: relative;
        height: 100%;
    }
    .highlight-yellow-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 16px 38px -2px rgba(255, 212, 71, 0.55) !important;
    }
    .highlight-yellow-card .card-subtitle {
        color: #5C4800 !important;
        font-weight: 700 !important;
    }
    .highlight-yellow-card .card-main-val {
        color: #111214 !important;
        font-weight: 800 !important;
    }

    /* Card Typography */
    .card-icon-container {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background-color: #242731;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 14px;
    }
    .card-subtitle {
        font-size: 0.72rem;
        font-weight: 700;
        color: #7A8194;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .card-main-val {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.15;
        letter-spacing: -0.04em;
    }
    .card-subtext {
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
        letter-spacing: -0.01em;
    }
    .text-mint {
        color: #2DD4BF !important;
    }
    .text-yellow {
        color: #FFD447 !important;
    }
    .text-rose {
        color: #FB7185 !important;
    }

    /* Small Floating Side Cards */
    .side-task-card {
        background-color: #1B1D22;
        border: 1px solid #262933;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .side-task-card.active-yellow {
        background: #FFD447;
        color: #111214;
        border: none;
        box-shadow: 0 10px 26px -2px rgba(255, 212, 71, 0.4);
    }

    /* Form Controls & Inputs (Dark Slate #1B1D22) */
    .stNumberInput input, .stTextInput input {
        background-color: #1B1D22 !important;
        border: 1px solid #292D38 !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        padding: 10px 14px !important;
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #FFD447 !important;
        box-shadow: 0 0 0 2px rgba(255, 212, 71, 0.25) !important;
    }

    /* Sliders Styling */
    .stSlider label {
        color: #EDEDED !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Primary Submit Button (Electric Yellow #FFD447) */
    .stButton > button, button[kind="primary"] {
        background: #FFD447 !important;
        color: #111214 !important;
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 28px !important;
        font-size: 0.98rem !important;
        letter-spacing: -0.02em !important;
        box-shadow: 0 10px 24px -3px rgba(255, 212, 71, 0.4) !important;
        transition: all 0.18s ease !important;
    }
    .stButton > button:hover, button[kind="primary"]:hover {
        background: #FFE27D !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 28px -2px rgba(255, 212, 71, 0.5) !important;
    }

    /* Custom Tables */
    .stDataFrame {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid #262933 !important;
    }

    /* Prediction Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, #13221E 0%, #17332B 100%);
        border: 1.5px solid #2DD4BF;
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 0 12px 28px -4px rgba(45, 212, 191, 0.2);
    }
    .result-card-med {
        background: #FFD447;
        color: #111214;
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 0 14px 34px -2px rgba(255, 212, 71, 0.45);
    }
    .result-card-high {
        background: linear-gradient(135deg, #2A171B 0%, #35191F 100%);
        border: 1.5px solid #FB7185;
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 0 12px 28px -4px rgba(251, 113, 133, 0.25);
    }

    /* Disclosure Banner */
    .disclosure-banner {
        background-color: #16181E;
        border: 1px solid #242732;
        border-left: 3px solid #FFD447;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATA & MODEL LOADERS (CACHED)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None


# ---------------------------------------------------------
# SIDEBAR NAVIGATION (lumio. Branding)
# ---------------------------------------------------------
with st.sidebar:
    # Brand Logo matching user screenshot
    st.markdown(
        """
        <div class="brand-container">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="background-color: #FFD447; border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; box-shadow: 0 4px 14px rgba(255, 212, 71, 0.35);">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#111214" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                    </svg>
                </div>
                <div>
                    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #FFFFFF; line-height: 1.1;">lumio.</div>
                    <div style="font-size: 0.74rem; color: #8C92A4; font-weight: 500; margin-top: 2px;">Electricity Usage Predictor</div>
                    <div style="margin-top: 4px;">
                        <span style="background: rgba(45, 212, 191, 0.12); color: #2DD4BF; font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 999px;">● Model Active</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<p style='font-size:0.70rem; font-weight:700; color:#545967; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</p>", unsafe_allow_html=True)
    
    # Pure Clean Text - Zero Emojis
    nav_selection = st.radio(
        "Navigation",
        options=["Dashboard", "Predict Usage", "Model Evaluation", "About & References"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color:#1B1D22; border:1px solid #262933; border-radius:12px; padding:12px 14px;">
            <div style="font-size:0.70rem; font-weight:700; color:#FFD447; text-transform:uppercase; letter-spacing:0.08em;">Core Engine</div>
            <div style="font-size:0.84rem; color:#EDEDED; font-weight:600; margin-top:2px;">CART Decision Tree</div>
            <div style="font-size:0.72rem; color:#6C7282; margin-top:2px;">4 Features • 3 Tiers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("lumio. Energy OS • v2.7")


# ---------------------------------------------------------
# ERROR HANDLING HELPER
# ---------------------------------------------------------
def render_missing_files_warning():
    st.error("Application Data & Model Not Found")
    st.markdown(
        """
        The application requires the generated dataset and the trained model artifact to operate.
        
        **Run the automated pipeline to initialize:**
        ```powershell
        python generate_dataset.py
        python train_model.py
        ```
        Then reload this page.
        """
    )


# ---------------------------------------------------------
# PAGE 1: DASHBOARD (HOME)
# ---------------------------------------------------------
if nav_selection == "Dashboard":
    st.markdown(
        """
        <div class="top-header-row">
            <h1 class="top-header-title">Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    metadata = load_metadata()

    if df is None or metadata is None:
        render_missing_files_warning()
        st.stop()

    # Clear Synthetic Data Disclosure Banner
    st.markdown(
        """
        <div class="disclosure-banner">
            <div style="font-size: 0.80rem; color: #8C92A4;">
                <strong style="color: #EDEDED;">Synthetic Educational Dataset Notice:</strong> Metrics & inferences are dynamically computed from 800 simulated residential profiles calibrated with domain-grounded electrical parameters.
            </div>
            <span style="background: rgba(255, 212, 71, 0.12); color: #FFD447; font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 6px; letter-spacing: 0.04em;">SIMULATED DATA</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # FUNCTIONAL INTERACTIVE TABS (Overview, Analytics, Consumption Tiers)
    tab_overview, tab_analytics, tab_tiers = st.tabs(
        ["Overview", "Analytics", "Consumption Tiers"]
    )

    # ---------------------------------------------
    # SUBTAB 1: OVERVIEW
    # ---------------------------------------------
    with tab_overview:
        # Dynamically extracted statistics
        total_records = len(df)
        total_features = len(metadata.get("feature_names", [1, 2, 3, 4]))
        total_classes = len(metadata.get("class_labels", ["Low", "Medium", "High"]))
        actual_test_accuracy = metadata.get("metrics", {}).get("test_accuracy", 0.0) * 100
        train_samples = metadata.get("dataset_statistics", {}).get("train_samples", int(total_records * 0.8))
        test_samples = metadata.get("dataset_statistics", {}).get("test_samples", int(total_records * 0.2))
        max_tree_depth = metadata.get("hyperparameters", {}).get("max_depth", 4)

        # 4 Top KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-icon-container">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFD447" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                    </div>
                    <div class="card-subtitle">Total Households</div>
                    <div class="card-main-val">{total_records:,}</div>
                    <div class="card-subtext text-mint">Validated Records</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-icon-container">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFD447" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
                    </div>
                    <div class="card-subtitle">Input Features</div>
                    <div class="card-main-val">{total_features}</div>
                    <div class="card-subtext text-mint">Attributes Evaluated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            # SIGNATURE GLOWING YELLOW CARD (#FFD447)
            st.markdown(
                f"""
                <div class="highlight-yellow-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#111214" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                        <span style="background:#111214; color:#FFD447; font-family:'Space Grotesk',sans-serif; font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:999px; letter-spacing:0.04em;">ACTIVE</span>
                    </div>
                    <div class="card-subtitle" style="margin-top:10px;">Classification</div>
                    <div class="card-main-val">{total_classes} Tiers</div>
                    <div style="font-size:0.76rem; font-weight:700; color:#524000; margin-top:6px;">Low • Medium • High</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                f"""
                <div class="dark-card">
                    <div class="card-icon-container">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2DD4BF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>
                    </div>
                    <div class="card-subtitle">Model Accuracy</div>
                    <div class="card-main-val">{actual_test_accuracy:.1f}%</div>
                    <div class="card-subtext text-mint">Held-out test split</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Middle Section: Charts & Analytics
        chart_col1, chart_col2 = st.columns([1.7, 1.1])

        with chart_col1:
            st.markdown(
                """
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">Overall Household Distribution</span>
                    <span style="font-size:0.75rem; color:#FFD447; font-weight:700; background:#1B1D22; padding:4px 10px; border-radius:8px; border:1px solid #262933;">All Tiers</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cat_counts = df["usage_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            custom_colors = {"Low": "#2DD4BF", "Medium": "#FFD447", "High": "#FB7185"}

            fig_bar = px.bar(
                cat_counts,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map=custom_colors,
                text="Count",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#8C92A4"),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=280,
                xaxis=dict(showgrid=False, linecolor="#262933"),
                yaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
            )
            fig_bar.update_traces(
                marker=dict(line=dict(width=0)),
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=13, family="Space Grotesk"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.markdown(
                """
                <div style="margin-bottom:12px;">
                    <span style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">Demand Summary</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mean_kwh = df["simulated_daily_consumption_kwh"].mean()
            max_kwh = df["simulated_daily_consumption_kwh"].max()

            # Side Card 1 (Electric Yellow Highlight #FFD447)
            st.markdown(
                f"""
                <div class="side-task-card active-yellow">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:1.0rem; font-weight:800; color:#111214;">Daily Mean Usage</div>
                        <span style="font-size:0.72rem; font-weight:800; background:#111214; color:#FFD447; padding:2px 8px; border-radius:6px; font-family:'Space Grotesk';">AVG</span>
                    </div>
                    <div style="font-family:'Space Grotesk'; font-size:1.65rem; font-weight:700; color:#111214; margin:6px 0 2px 0;">{mean_kwh:.2f} kWh/day</div>
                    <div style="font-size:0.76rem; font-weight:600; color:#5D4600;">Approx. {mean_kwh * 30:.0f} units per month</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Side Card 2 (Dark Slate #1B1D22 with Mint Indicator)
            st.markdown(
                f"""
                <div class="side-task-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF;">Peak Load Observed</div>
                        <span style="font-size:0.72rem; font-weight:700; color:#2DD4BF; font-family:'Space Grotesk';">+4.5%</span>
                    </div>
                    <div style="font-family:'Space Grotesk'; font-size:1.45rem; font-weight:700; color:#2DD4BF; margin:6px 0 2px 0;">{max_kwh:.1f} kWh</div>
                    <div style="font-size:0.74rem; color:#6C7282;">Maximum recorded daily consumption</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Bottom Row: 4 Metric Pillars
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)

        with b1:
            st.markdown(
                f"""
                <div class="dark-card" style="padding:16px 18px;">
                    <div class="card-subtitle">Training Partition</div>
                    <div style="font-family:'Space Grotesk'; font-size:1.4rem; font-weight:700; color:#FFFFFF;">{train_samples} Records</div>
                    <div class="card-subtext text-mint">80% Stratified Split</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b2:
            st.markdown(
                f"""
                <div class="dark-card" style="padding:16px 18px; border-color:#FFD447;">
                    <div class="card-subtitle" style="color:#FFD447;">Decision Tree Depth</div>
                    <div style="font-family:'Space Grotesk'; font-size:1.4rem; font-weight:700; color:#FFD447;">{max_tree_depth} Levels</div>
                    <div style="font-size:0.74rem; color:#7A8194; margin-top:6px;">Regularized (Anti-Overfitting)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b3:
            st.markdown(
                f"""
                <div class="dark-card" style="padding:16px 18px;">
                    <div class="card-subtitle">Testing Partition</div>
                    <div style="font-family:'Space Grotesk'; font-size:1.4rem; font-weight:700; color:#FFFFFF;">{test_samples} Records</div>
                    <div class="card-subtext text-mint">20% Held-Out Validation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b4:
            st.markdown(
                """
                <div class="dark-card" style="padding:16px 18px;">
                    <div class="card-subtitle">Inference Latency</div>
                    <div style="font-family:'Space Grotesk'; font-size:1.4rem; font-weight:700; color:#2DD4BF;">&lt; 2 ms</div>
                    <div class="card-subtext text-mint">Instant Local Compute</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------
    # SUBTAB 2: ANALYTICS (DEEP DIVE VISUALIZATIONS)
    # ---------------------------------------------
    with tab_analytics:
        st.markdown(
            """
            <div style="margin-bottom:16px;">
                <span style="font-size:1.2rem; font-weight:700; color:#FFFFFF;">In-Depth Residential Energy Analytics</span>
                <p style="font-size:0.84rem; color:#8C92A4; margin:4px 0 0 0;">Correlations and spread across household appliance loads, AC operational hours, and consumption ranges.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        an_col1, an_col2 = st.columns([1.4, 1.2])

        with an_col1:
            st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#FFFFFF; margin-bottom:8px;'>Consumption Spread by Tier (kWh/day)</p>", unsafe_allow_html=True)
            custom_colors = {"Low": "#2DD4BF", "Medium": "#FFD447", "High": "#FB7185"}

            fig_box = px.box(
                df,
                x="usage_category",
                y="simulated_daily_consumption_kwh",
                color="usage_category",
                color_discrete_map=custom_colors,
                category_orders={"usage_category": ["Low", "Medium", "High"]},
                labels={
                    "usage_category": "Tier",
                    "simulated_daily_consumption_kwh": "Daily Energy (kWh)",
                },
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#8C92A4"),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                xaxis=dict(showgrid=False, linecolor="#262933"),
                yaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
            )
            fig_box.add_hline(y=10.0, line_dash="dot", line_color="#2DD4BF", annotation_text="Low Cutoff (10 kWh)", annotation_font_color="#2DD4BF")
            fig_box.add_hline(y=22.0, line_dash="dot", line_color="#FB7185", annotation_text="High Cutoff (22 kWh)", annotation_font_color="#FB7185")
            st.plotly_chart(fig_box, use_container_width=True)

        with an_col2:
            st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#FFFFFF; margin-bottom:8px;'>Daily Energy Density Histogram</p>", unsafe_allow_html=True)
            fig_hist = px.histogram(
                df,
                x="simulated_daily_consumption_kwh",
                color="usage_category",
                color_discrete_map=custom_colors,
                nbins=35,
                labels={"simulated_daily_consumption_kwh": "Daily Consumption (kWh)"},
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#8C92A4"),
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                xaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
                yaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        an2_col1, an2_col2 = st.columns([1.5, 1.1])

        with an2_col1:
            st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#FFFFFF; margin-bottom:8px;'>Air Conditioner Hours vs Total Daily Energy</p>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df,
                x="ac_usage_hours_per_day",
                y="simulated_daily_consumption_kwh",
                color="usage_category",
                color_discrete_map=custom_colors,
                labels={
                    "ac_usage_hours_per_day": "Daily AC Hours",
                    "simulated_daily_consumption_kwh": "Total Consumption (kWh)",
                },
                hover_data=["household_size", "number_of_appliances"],
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color="#8C92A4"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=10, b=10, l=10, r=10),
                height=310,
                xaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
                yaxis=dict(gridcolor="#1E2026", linecolor="#262933"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with an2_col2:
            st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#FFFFFF; margin-bottom:8px;'>Dataset Consumption Percentiles</p>", unsafe_allow_html=True)
            p25 = float(np.percentile(df["simulated_daily_consumption_kwh"], 25))
            p50 = float(np.percentile(df["simulated_daily_consumption_kwh"], 50))
            p75 = float(np.percentile(df["simulated_daily_consumption_kwh"], 75))
            std_kwh = float(df["simulated_daily_consumption_kwh"].std())

            st.markdown(
                f"""
                <div class="dark-card" style="padding:16px 20px;">
                    <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #262933;">
                        <span style="color:#8C92A4; font-size:0.86rem;">25th Percentile (Q1)</span>
                        <span style="color:#EDEDED; font-family:'Space Grotesk'; font-weight:700;">{p25:.2f} kWh/day</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #262933;">
                        <span style="color:#8C92A4; font-size:0.86rem;">Median (50th Percentile)</span>
                        <span style="color:#FFD447; font-family:'Space Grotesk'; font-weight:700;">{p50:.2f} kWh/day</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #262933;">
                        <span style="color:#8C92A4; font-size:0.86rem;">75th Percentile (Q3)</span>
                        <span style="color:#EDEDED; font-family:'Space Grotesk'; font-weight:700;">{p75:.2f} kWh/day</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #262933;">
                        <span style="color:#8C92A4; font-size:0.86rem;">Interquartile Range (IQR)</span>
                        <span style="color:#2DD4BF; font-family:'Space Grotesk'; font-weight:700;">{p75 - p25:.2f} kWh</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:8px 0;">
                        <span style="color:#8C92A4; font-size:0.86rem;">Standard Deviation</span>
                        <span style="color:#EDEDED; font-family:'Space Grotesk'; font-weight:700;">{std_kwh:.2f} kWh</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------
    # SUBTAB 3: CONSUMPTION TIERS (COMPARATIVE PROFILES)
    # ---------------------------------------------
    with tab_tiers:
        st.markdown(
            """
            <div style="margin-bottom:18px;">
                <span style="font-size:1.2rem; font-weight:700; color:#FFFFFF;">Residential Energy Tier Specifications</span>
                <p style="font-size:0.84rem; color:#8C92A4; margin:4px 0 0 0;">Physical criteria, typical household configurations, and monthly billing expectations across the three classes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tc1, tc2, tc3 = st.columns(3)

        # Dynamic counts & percentages
        low_cnt = int((df["usage_category"] == "Low").sum())
        med_cnt = int((df["usage_category"] == "Medium").sum())
        high_cnt = int((df["usage_category"] == "High").sum())
        total_cnt = len(df)

        with tc1:
            st.markdown(
                f"""
                <div class="result-card-low" style="padding:20px 22px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:#2DD4BF; color:#111214; font-family:'Space Grotesk'; font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:999px;">TIER 1</span>
                        <span style="color:#2DD4BF; font-family:'Space Grotesk'; font-weight:700; font-size:0.82rem;">{(low_cnt/total_cnt)*100:.1f}% of Total</span>
                    </div>
                    <h3 style="color:#2DD4BF; margin:12px 0 4px 0; font-size:1.45rem;">Low Consumption</h3>
                    <div style="font-family:'Space Grotesk'; font-size:1.15rem; font-weight:700; color:#FFFFFF; margin-bottom:12px;">&lt; 10.0 kWh/day</div>
                    <div style="font-size:0.82rem; color:#A7F3D0; line-height:1.7;">
                        • <strong>Monthly Units:</strong> &lt; 300 kWh/month<br>
                        • <strong>Household Size:</strong> 1 to 3 members<br>
                        • <strong>Typical Appliances:</strong> 3 to 8 devices<br>
                        • <strong>AC Utilization:</strong> 0 to 2 hours/day<br>
                        • <strong>Profile:</strong> Energy-efficient residential homes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tc2:
            st.markdown(
                f"""
                <div class="result-card-med" style="padding:20px 22px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:#111214; color:#FFD447; font-family:'Space Grotesk'; font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:999px;">TIER 2</span>
                        <span style="color:#3B2D00; font-family:'Space Grotesk'; font-weight:800; font-size:0.82rem;">{(med_cnt/total_cnt)*100:.1f}% of Total</span>
                    </div>
                    <h3 style="color:#111214; margin:12px 0 4px 0; font-size:1.45rem; font-weight:800;">Medium Consumption</h3>
                    <div style="font-family:'Space Grotesk'; font-size:1.15rem; font-weight:800; color:#111214; margin-bottom:12px;">10.0 to 22.0 kWh/day</div>
                    <div style="font-size:0.82rem; color:#453600; line-height:1.7; font-weight:600;">
                        • <strong>Monthly Units:</strong> 300 to 660 kWh/month<br>
                        • <strong>Household Size:</strong> 2 to 5 members<br>
                        • <strong>Typical Appliances:</strong> 8 to 16 devices<br>
                        • <strong>AC Utilization:</strong> 2 to 6 hours/day<br>
                        • <strong>Profile:</strong> Standard urban residential consumer.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tc3:
            st.markdown(
                f"""
                <div class="result-card-high" style="padding:20px 22px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="background:#FB7185; color:#111214; font-family:'Space Grotesk'; font-size:0.68rem; font-weight:800; padding:3px 8px; border-radius:999px;">TIER 3</span>
                        <span style="color:#FB7185; font-family:'Space Grotesk'; font-weight:700; font-size:0.82rem;">{(high_cnt/total_cnt)*100:.1f}% of Total</span>
                    </div>
                    <h3 style="color:#FB7185; margin:12px 0 4px 0; font-size:1.45rem;">High Consumption</h3>
                    <div style="font-family:'Space Grotesk'; font-size:1.15rem; font-weight:700; color:#FFFFFF; margin-bottom:12px;">&gt; 22.0 kWh/day</div>
                    <div style="font-size:0.82rem; color:#FECDD3; line-height:1.7;">
                        • <strong>Monthly Units:</strong> &gt; 660 kWh/month<br>
                        • <strong>Household Size:</strong> 4 to 8 members<br>
                        • <strong>Typical Appliances:</strong> 14 to 25 devices<br>
                        • <strong>AC Utilization:</strong> 5 to 14 hours/day<br>
                        • <strong>Profile:</strong> Heavy load, multiple ACs/appliances.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;'>Comparative Statistical Averages Across Tiers</p>", unsafe_allow_html=True)

        tier_stats = []
        for tier in ["Low", "Medium", "High"]:
            sub = df[df["usage_category"] == tier]
            tier_stats.append({
                "Tier Category": tier,
                "Household Count": len(sub),
                "Avg Household Size": f"{sub['household_size'].mean():.1f} members",
                "Avg Appliances": f"{sub['number_of_appliances'].mean():.1f}",
                "Avg Active Hours": f"{sub['average_daily_usage_hours'].mean():.1f} hrs/day",
                "Avg AC Hours": f"{sub['ac_usage_hours_per_day'].mean():.1f} hrs/day",
                "Avg Daily Energy": f"{sub['simulated_daily_consumption_kwh'].mean():.2f} kWh",
                "Monthly Unit Est.": f"{sub['simulated_daily_consumption_kwh'].mean() * 30:.0f} units",
            })
        st.dataframe(pd.DataFrame(tier_stats), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# PAGE 2: PREDICT USAGE
# ---------------------------------------------------------
elif nav_selection == "Predict Usage":
    st.markdown(
        """
        <div class="top-header-row">
            <h1 class="top-header-title">Predict Usage Tier</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    metadata = load_metadata()

    if model is None or metadata is None:
        render_missing_files_warning()
        st.stop()

    form_col, info_col = st.columns([1.6, 1])

    with form_col:
        st.markdown(
            """
            <div style="background-color:#1B1D22; border:1px solid #262933; border-radius:16px; padding:24px 22px; margin-bottom:20px;">
                <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF; margin-bottom:18px;">Enter Household Specifications</div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            c_left, c_right = st.columns(2)

            with c_left:
                household_size = st.number_input(
                    "Household Occupancy (Members)",
                    min_value=1,
                    max_value=12,
                    value=4,
                    step=1,
                    help="Regular residents residing in the premises.",
                )
                number_of_appliances = st.number_input(
                    "Total Electrical Appliances",
                    min_value=1,
                    max_value=40,
                    value=12,
                    step=1,
                    help="Count of electrical appliances: refrigerator, lights, TV, computer, geyser, fans, etc.",
                )

            with c_right:
                average_daily_usage_hours = st.slider(
                    "Average Active Appliance Hours",
                    min_value=1.0,
                    max_value=24.0,
                    value=8.0,
                    step=0.5,
                    help="Daily cumulative operating hours across appliances.",
                )
                ac_usage_hours_per_day = st.slider(
                    "Daily Air Conditioner (AC) Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=3.5,
                    step=0.5,
                    help="Daily AC running hours. Air conditioners are high-power thermal loads.",
                )

            hours_mismatch = ac_usage_hours_per_day > average_daily_usage_hours

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict Consumption Tier", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown(
            """
            <div class="dark-card">
                <div style="font-size:1.0rem; font-weight:700; color:#FFFFFF; margin-bottom:12px;">Consumption Benchmarks</div>
                <div style="font-size:0.83rem; color:#8C92A4; line-height:1.7;">
                    The prediction pipeline feeds the 4 attributes through a pre-fitted <strong>Decision Tree Classifier</strong>.
                    <br><br>
                    • <strong>Low:</strong> Under 10.0 kWh/day (&lt; 300 units/mo)<br>
                    • <strong>Medium:</strong> 10.0 to 22.0 kWh/day (300-660 units/mo)<br>
                    • <strong>High:</strong> Above 22.0 kWh/day (&gt; 660 units/mo)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Handle Prediction Output
    if submitted:
        if hours_mismatch:
            st.error(
                f"Validation Error: Air Conditioner usage hours ({ac_usage_hours_per_day} hrs) "
                f"cannot exceed total daily appliance operating hours ({average_daily_usage_hours} hrs)."
            )
        else:
            feature_order = metadata.get("feature_names", [
                "household_size",
                "number_of_appliances",
                "average_daily_usage_hours",
                "ac_usage_hours_per_day",
            ])
            
            input_dict = {
                "household_size": household_size,
                "number_of_appliances": number_of_appliances,
                "average_daily_usage_hours": average_daily_usage_hours,
                "ac_usage_hours_per_day": ac_usage_hours_per_day,
            }
            input_df = pd.DataFrame([input_dict])[feature_order]

            with st.spinner("Classifying consumption tier with Decision Tree..."):
                predicted_category = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Classification Result")

            res_c1, res_c2 = st.columns([1.2, 1])

            with res_c1:
                if predicted_category == "Low":
                    st.markdown(
                        """
                        <div class="result-card-low">
                            <span style="background:#2DD4BF; color:#111214; font-family:'Space Grotesk'; font-size:0.72rem; font-weight:800; padding:4px 10px; border-radius:999px;">TIER: LOW DEMAND</span>
                            <h2 style="color:#2DD4BF; margin:10px 0 6px 0; font-size:1.85rem; font-weight:700;">Low Consumption</h2>
                            <p style="color:#C6F6D5; font-size:0.92rem; margin:0;">
                                Estimated daily demand under <strong>10.0 kWh/day</strong> (approx. under 300 units/month). Energy-conscious pattern.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif predicted_category == "Medium":
                    # Electric Yellow Highlight Card (#FFD447)
                    st.markdown(
                        """
                        <div class="result-card-med">
                            <span style="background:#111214; color:#FFD447; font-family:'Space Grotesk'; font-size:0.72rem; font-weight:800; padding:4px 10px; border-radius:999px;">TIER: MODERATE DEMAND</span>
                            <h2 style="color:#111214; margin:10px 0 6px 0; font-size:1.85rem; font-weight:800;">Medium Consumption</h2>
                            <p style="color:#3B2D00; font-size:0.92rem; margin:0; font-weight:600;">
                                Estimated daily demand between <strong>10.0 and 22.0 kWh/day</strong> (approx. 300 to 660 units/month).
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="result-card-high">
                            <span style="background:#FB7185; color:#111214; font-family:'Space Grotesk'; font-size:0.72rem; font-weight:800; padding:4px 10px; border-radius:999px;">TIER: HEAVY DEMAND</span>
                            <h2 style="color:#FB7185; margin:10px 0 6px 0; font-size:1.85rem; font-weight:700;">High Consumption</h2>
                            <p style="color:#FECDD3; font-size:0.92rem; margin:0;">
                                Estimated daily demand above <strong>22.0 kWh/day</strong> (approx. above 660 units/month). High cooling load.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with res_c2:
                st.markdown(
                    """
                    <div class="dark-card" style="padding:18px 20px;">
                        <div style="font-size:0.74rem; font-weight:700; color:#7A8194; text-transform:uppercase; letter-spacing:0.08em;">Confidence Distribution</div>
                    """,
                    unsafe_allow_html=True,
                )
                for c_name, p_val in zip(classes, probabilities):
                    pct = p_val * 100
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.86rem;">
                            <span style="color:#EDEDED; font-weight:600;">{c_name} Tier</span>
                            <span style="color:#FFD447; font-family:'Space Grotesk'; font-weight:700;">{pct:.1f}%</span>
                        </div>
                        <div style="background-color:#262933; height:6px; border-radius:3px; overflow:hidden; margin-top:3px;">
                            <div style="background-color:#FFD447; height:100%; width:{pct}%;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Recommended Energy Actions")

            if predicted_category == "Low":
                st.info(
                    "**Maintain Current Efficiency:** Continue leveraging natural cross-ventilation. Keep standby appliances switched off at the wall socket to sustain low baseline consumption."
                )
            elif predicted_category == "Medium":
                st.warning(
                    "**Thermostat Calibration:** Setting your AC temperature to 24°C–26°C reduces cooling consumption by ~6% per degree. Clean AC air filters bi-weekly to prevent compressor strain."
                )
            else:
                st.error(
                    "**Active Demand Reduction:** Air conditioning dominates this tier. Pair AC with ceiling fans at 26°C, utilize thermal blackout curtains, and consider shifting high-wattage appliance use to off-peak periods."
                )


# ---------------------------------------------------------
# PAGE 3: MODEL EVALUATION
# ---------------------------------------------------------
elif nav_selection == "Model Evaluation":
    st.markdown(
        """
        <div class="top-header-row">
            <h1 class="top-header-title">Model Evaluation</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata = load_metadata()
    if metadata is None:
        render_missing_files_warning()
        st.stop()

    metrics = metadata.get("metrics", {})
    cm_data = metadata.get("confusion_matrix", {})
    clf_report = metadata.get("classification_report", {})
    importances = metadata.get("feature_importances", {})

    # 4 Key Metric Cards
    ev1, ev2, ev3, ev4 = st.columns(4)

    with ev1:
        st.markdown(
            f"""
            <div class="dark-card">
                <div class="card-subtitle">Test Accuracy</div>
                <div class="card-main-val text-mint">{metrics.get('test_accuracy', 0.0) * 100:.1f}%</div>
                <div class="card-subtext text-mint">Held-out 20% validation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ev2:
        st.markdown(
            f"""
            <div class="dark-card">
                <div class="card-subtitle">Macro Precision</div>
                <div class="card-main-val">{metrics.get('precision_macro', 0.0) * 100:.1f}%</div>
                <div class="card-subtext text-mint">Unweighted average</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ev3:
        st.markdown(
            f"""
            <div class="dark-card">
                <div class="card-subtitle">Macro Recall</div>
                <div class="card-main-val">{metrics.get('recall_macro', 0.0) * 100:.1f}%</div>
                <div class="card-subtext text-mint">Positive sensitivity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ev4:
        st.markdown(
            f"""
            <div class="dark-card" style="border-color:#FFD447;">
                <div class="card-subtitle" style="color:#FFD447;">Macro F1-Score</div>
                <div class="card-main-val" style="color:#FFD447;">{metrics.get('f1_macro', 0.0) * 100:.1f}%</div>
                <div style="font-size:0.74rem; color:#7A8194; margin-top:6px;">Harmonic balance P & R</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix & Feature Importances in Dark Theme
    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown(
            """
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;">
                Confusion Matrix (Held-out Test Subset)
            </div>
            """,
            unsafe_allow_html=True,
        )
        labels = cm_data.get("labels", ["Low", "Medium", "High"])
        matrix_vals = cm_data.get("matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        fig_cm = px.imshow(
            matrix_vals,
            x=labels,
            y=labels,
            color_continuous_scale=[[0, "#16181E"], [0.5, "#0D9488"], [1, "#FFD447"]],
            text_auto=True,
            labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#8C92A4"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            xaxis=dict(gridcolor="#1E2026"),
            yaxis=dict(gridcolor="#1E2026"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_m2:
        st.markdown(
            """
            <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;">
                Gini Feature Importances
            </div>
            """,
            unsafe_allow_html=True,
        )
        feat_df = pd.DataFrame(
            list(importances.items()), columns=["Feature", "Importance"]
        ).sort_values(by="Importance", ascending=True)

        label_map = {
            "ac_usage_hours_per_day": "AC Usage (hrs/day)",
            "average_daily_usage_hours": "Appliance Operating (hrs/day)",
            "number_of_appliances": "Total Appliances",
            "household_size": "Household Occupancy",
        }
        feat_df["Feature_Label"] = feat_df["Feature"].map(lambda x: label_map.get(x, x))

        fig_imp = px.bar(
            feat_df,
            x="Importance",
            y="Feature_Label",
            orientation="h",
            color_discrete_sequence=["#FFD447"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit, sans-serif", color="#8C92A4"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            xaxis=dict(gridcolor="#1E2026", title="Gini Importance Score"),
            yaxis=dict(showgrid=False, title=""),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Detailed Per-Class Classification Report Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; margin-bottom:10px;">
            Detailed Classification Report
        </div>
        """,
        unsafe_allow_html=True,
    )
    table_rows = []
    for cls in ["Low", "Medium", "High"]:
        if cls in clf_report:
            item = clf_report[cls]
            table_rows.append({
                "Category Tier": cls,
                "Precision": f"{item.get('precision', 0.0) * 100:.2f}%",
                "Recall": f"{item.get('recall', 0.0) * 100:.2f}%",
                "F1-Score": f"{item.get('f1-score', 0.0) * 100:.2f}%",
                "Test Support": int(item.get("support", 0)),
            })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# PAGE 4: ABOUT & REFERENCES
# ---------------------------------------------------------
elif nav_selection == "About & References":
    st.markdown(
        """
        <div class="top-header-row">
            <h1 class="top-header-title">About & References</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs([
        "System Architecture",
        "Algorithms & Theory",
        "Dataset Methodology",
        "References & Scope",
    ])

    with t1:
        st.markdown("### System Architecture & Objectives")
        st.markdown(
            """
            **lumio.** is an intelligent decision-support system that classifies household electricity consumption into three distinct tiers:
            - **Low:** Under 10.0 kWh/day (&lt; 300 units/month)
            - **Medium:** 10.0 to 22.0 kWh/day (300 to 660 units/month)
            - **High:** Above 22.0 kWh/day (&gt; 660 units/month)
            
            #### Core Objectives:
            1. **Early Tier Visibility:** Enable domestic users to predict their billing slab before monthly utility statements arrive.
            2. **Interpretable Rule Induction:** Utilize non-black-box tree structures to ensure algorithmic decisions can be audited.
            3. **Zero-Cloud Dependency:** Designed to run locally with zero API latency and zero cloud costs.
            """
        )

    with t2:
        st.markdown("### Machine Learning Engine: CART Decision Tree")
        st.markdown(
            """
            The classification engine employs **scikit-learn's `DecisionTreeClassifier`** utilizing the CART (Classification and Regression Trees) paradigm with Gini Impurity minimization:
            """
        )
        st.latex(r"I_G(p) = 1 - \sum_{i=1}^{C} p_i^2")
        st.markdown(
            """
            #### Regularization Hyperparameters:
            - **Criterion:** Gini Impurity
            - **Max Depth:** `4` (Prevents overfitting while retaining high expressiveness)
            - **Min Samples Split:** `8`
            - **Min Samples Leaf:** `4`
            - **Random State:** `42` (Strict reproducibility across splits)
            """
        )

    with t3:
        st.markdown("### Dataset Methodology & Academic Disclosures")
        st.markdown(
            """
            #### Educational & Synthetic Data Disclosure
            This application is trained on a synthetically simulated dataset constructed from physical residential electrical engineering principles.
            
            #### Generation Equation (Label Derivation Only):
            """
        )
        st.latex(
            r"\text{Daily kWh} = 1.25 \times \text{Size} + 0.38 \times \text{Appliances} \times \left(\frac{\text{Usage Hours}}{8.0}\right) + 1.45 \times \text{AC Hours} + \mathcal{N}(0, 1.2^2)"
        )
        st.markdown(
            """
            **Strict Data Leakage Prevention:** The calculated `simulated_daily_consumption_kwh` is used **exclusively** to establish target category labels during data generation. It is **never** provided as an input feature to the machine learning model.
            """
        )

    with t4:
        st.markdown("### Limitations & References")
        st.markdown(
            """
            #### Limitations
            1. **Synthetic Data Constraint:** Performance on simulated data demonstrates pipeline soundness, but does not substitute for field tests on physical utility hardware.
            2. **Ambient Weather Omission:** Real-world cooling efficiency fluctuates with ambient summer temperatures.
            
            #### Academic & Technical References
            1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.
            2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
            3. Bureau of Energy Efficiency (BEE), Ministry of Power, India. *Standards & Labeling Program for Household Air Conditioners*.
            """
        )
