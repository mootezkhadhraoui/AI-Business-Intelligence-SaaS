import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Business Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Global */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Hide Streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1F2937;
}

/* Titles */
h1, h2, h3 {
    color: white !important;
    font-weight: 700 !important;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #374151;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* Metric styling */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #374151;
    padding: 20px;
    border-radius: 16px;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-weight: 700;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg, #4338CA, #6D28D9);
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: #111827;
    border-radius: 15px;
    padding: 20px;
    border: 1px dashed #4F46E5;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* Main spacing */
.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    padding: 40px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# ================= ROOT =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ================= ENV =================
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ================= SAFE IMPORTS =================
try:
    from gemini_analysis import analyze_data
except Exception:
    analyze_data = None

try:
    from src.logger import log_event
except Exception:
    def log_event(*args, **kwargs):
        pass

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1>🤖 AI Business Intelligence Platform</h1>
    <p style="font-size:18px;">
        MLOps • MLflow • DagsHub • Streamlit • Gemini AI
    </p>
    <p>
        Smart churn prediction platform with real-time analytics and AI insights.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= MODEL LOADER =================
@st.cache_resource
def load_model_safe():

    try:
        import joblib

        path = os.path.join(
            PROJECT_ROOT,
            "models",
            "champion_model.pkl"
        )

        if os.path.exists(path):
            return joblib.load(path)

    except Exception:
        pass

    try:
        from src.champion import load_champion_model
        return load_champion_model()

    except Exception:
        return None

# ================= FEATURES =================
features_path = os.path.join(
    PROJECT_ROOT,
    "models",
    "features.json"
)

if not os.path.exists(features_path):
    st.error("❌ features.json introuvable")
    st.stop()

with open(features_path, "r") as f:
    features = json.load(f)

# ================= SIDEBAR =================
st.sidebar.markdown("# 🤖 AI SaaS")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Prediction",
        "📁 Batch AI",
        "📊 Analytics",
        "🧠 AI Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Platform Online")

# ================= LOAD DATA =================
data_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "customer_churn.csv"
)

df_main = pd.read_csv(data_path)

df_main["Churn"] = (
    df_main["Churn"]
    .map({"Yes": 1, "No": 0})
    .fillna(0)
)

# ================= OVERVIEW =================
if menu == "🏠 Overview":

    st.subheader("📊 Executive Dashboard")

    total_clients = len(df_main)
    churn_rate = round(df_main["Churn"].mean() * 100, 2)
    active_clients = int((df_main["Churn"] == 0).sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👥 Total Clients",
            total_clients
        )

    with col2:
        st.metric(
            "📉 Churn Rate",
            f"{churn_rate}%"
        )

    with col3:
        st.metric(
            "✅ Active Clients",
            active_clients
        )

    st.markdown("## 📈 Business Analytics")

    col4, col5 = st.columns([1.2, 1])

    with col4:

        churn_chart = px.pie(
            names=["Retained", "Churned"],
            values=[
                active_clients,
                total_clients - active_clients
            ],
            title="Customer Distribution",
            hole=0.55
        )

        churn_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            churn_chart,
            use_container_width=True
        )

    with col5:

        contract_chart = px.histogram(
            df_main,
            x="tenure",
            title="Customer Tenure Distribution"
        )

        contract_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            contract_chart,
            use_container_width=True
        )

    st.markdown("## 📄 Dataset Preview")

    st.dataframe(
        df_main.head(10),
        use_container_width=True
    )

# ================= PREDICTION =================
elif menu == "🔮 Prediction":

    st.subheader("🔮 Smart Customer Prediction")

    st.info(
        "Enter customer data to predict churn probability."
    )

    input_data = []

    cols = st.columns(2)

    for i, feature in enumerate(features):

        with cols[i % 2]:

            value = st.number_input(
                feature,
                value=0.0
            )

            input_data.append(value)

    if st.button("🚀 Predict Customer"):

        with st.spinner("🤖 AI model analyzing..."):

            model = load_model_safe()

            if model is None:
                st.error("❌ Model unavailable")
            else:

                try:

                    prediction = model.predict(
                        np.array(input_data).reshape(1, -1)
                    )[0]

                    log_event(
                        "prediction",
                        str(input_data)
                    )

                    st.success("✅ Prediction completed")

                    if prediction == 1:

                        st.error("⚠️ HIGH CHURN RISK")

                        st.markdown("""
                        ### 🔴 Customer likely to churn

                        Recommended actions:
                        - Offer retention discount
                        - Contact customer support
                        - Send loyalty promotion
                        """)

                    else:

                        st.success("🟢 LOW CHURN RISK")

                        st.markdown("""
                        ### ✅ Customer likely retained

                        Customer relationship is stable.
                        """)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ================= BATCH =================
elif menu == "📁 Batch AI":

    st.subheader("📁 Batch Prediction Engine")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.markdown("### 📄 Uploaded Dataset")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        if st.button("⚡ Run AI Prediction"):

            with st.spinner("🤖 Processing predictions..."):

                model = load_model_safe()

                if model is None:
                    st.error("❌ Model unavailable")
                    st.stop()

                try:

                    original_df = df.copy()

                    df = df.drop(
                        columns=["customerID"],
                        errors="ignore"
                    )

                    df = df.replace({
                        "Yes": 1,
                        "No": 0,
                        "Female": 0,
                        "Male": 1
                    })

                    df = df.apply(
                        pd.to_numeric,
                        errors="coerce"
                    ).fillna(0)

                    df = df[features]

                    preds = model.predict(df)

                    original_df["Prediction"] = preds

                    st.success(
                        "✅ Batch prediction completed"
                    )

                    st.dataframe(
                        original_df,
                        use_container_width=True
                    )

                    csv = original_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Batch error: {e}")

# ================= ANALYTICS =================
elif menu == "📊 Analytics":

    st.subheader("📊 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:

        chart1 = px.histogram(
            df_main,
            x="MonthlyCharges",
            nbins=30,
            title="Monthly Charges Distribution"
        )

        chart1.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            chart1,
            use_container_width=True
        )

    with col2:

        chart2 = px.scatter(
            df_main,
            x="tenure",
            y="MonthlyCharges",
            color="Churn",
            title="Tenure vs Monthly Charges"
        )

        chart2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            chart2,
            use_container_width=True
        )

    st.markdown("## 📊 Churn Comparison")

    churn_bar = px.bar(
        df_main["Churn"].value_counts(),
        title="Churn Distribution"
    )

    churn_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    st.plotly_chart(
        churn_bar,
        use_container_width=True
    )

# ================= AI INSIGHTS =================
elif menu == "🧠 AI Insights":

    st.subheader("🧠 Gemini AI Business Insights")

    st.dataframe(
        df_main.head(),
        use_container_width=True
    )

    if analyze_data is None:

        st.error("❌ Gemini module unavailable")

    else:

        if st.button("✨ Generate AI Insights"):

            with st.spinner("🤖 Gemini analyzing dataset..."):

                result = analyze_data(df_main)

            st.success("✅ AI Analysis completed")

            st.markdown(result)