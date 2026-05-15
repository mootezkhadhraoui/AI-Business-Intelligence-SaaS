import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

# ================= CONFIG UI =================
st.set_page_config(
    page_title="AI Business Intelligence App",
    layout="wide"
)

# ================= UX STYLE =================
st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: #ffffff;
}

.stApp {
    background: #0f1117;
}

/* Buttons */
.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #3730A3;
    transform: scale(1.02);
}

/* Metrics cards */
div[data-testid="metric-container"] {
    background-color: #1c1f2a;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #2a2f3a;
}

/* Layout spacing */
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🤖 AI Business Intelligence Platform")
st.caption("MLOps • MLflow • Streamlit • Gemini AI • Smart Predictions")
st.success("System Online ✅ All services running")

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

# ================= MODEL LOADER (SAFE + FALLBACK) =================
def load_model_safe():
    try:
        import joblib
        path = os.path.join(PROJECT_ROOT, "models", "champion_model.pkl")

        if os.path.exists(path):
            return joblib.load(path)
    except:
        pass

    try:
        from src.champion import load_champion_model
        return load_champion_model()
    except Exception:
        return None

# ================= FEATURES =================
features_path = os.path.join(PROJECT_ROOT, "models", "features.json")

if not os.path.exists(features_path):
    st.error("features.json introuvable")
    st.stop()

with open(features_path, "r") as f:
    features = json.load(f)

# ================= SIDEBAR =================
st.sidebar.title("🧭 Navigation Panel")
st.sidebar.info("AI SaaS Dashboard")

menu = st.sidebar.radio(
    "Choose module:",
    ["🏠 Overview", "🔮 Prediction", "📁 Batch AI", "📊 Analytics", "🧠 AI Insights"]
)

# ================= OVERVIEW =================
if menu == "🏠 Overview":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0}).fillna(0)

    st.subheader("📊 Business Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Clients", len(df))
    col2.metric("📉 Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
    col3.metric("✅ Active Clients", int((df["Churn"] == 0).sum()))

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.bar_chart(df["Churn"].value_counts())

    with col5:
        st.dataframe(df.head(), use_container_width=True)

# ================= PREDICTION =================
elif menu == "🔮 Prediction":

    st.subheader("🔮 Customer Churn Prediction")

    input_data = []

    cols = st.columns(2)

    for i, feature in enumerate(features):
        with cols[i % 2]:
            input_data.append(st.number_input(feature, value=0.0))

    if st.button("🚀 Predict Now"):

        with st.spinner("🧠 AI model analyzing customer..."):

            model = load_model_safe()

            if model is None:
                st.error("❌ Model unavailable")
            else:
                try:
                    pred = model.predict(np.array(input_data).reshape(1, -1))[0]

                    log_event("prediction", str(input_data))

                    st.success("Analysis complete")

                    if pred == 1:
                        st.error("⚠️ HIGH RISK CUSTOMER")
                        st.warning("Customer likely to churn")
                    else:
                        st.success("🟢 LOW RISK CUSTOMER")
                        st.info("Customer retained")

                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ================= BATCH =================
elif menu == "📁 Batch AI":

    st.subheader("📁 Batch Prediction Engine")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("⚡ Run Prediction"):

            with st.spinner("⚙️ Loading model..."):
                model = load_model_safe()

            if model is None:
                st.error("Model unavailable")
                st.stop()

            with st.spinner("📊 Processing data..."):

                df = df.drop(columns=["customerID"], errors="ignore")
                df = df.replace({"Yes":1,"No":0,"Female":0,"Male":1})
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

                df = df[features]

            with st.spinner("🤖 Running predictions..."):
                preds = model.predict(df)

            df["Prediction"] = preds

            st.success("✅ Batch prediction completed")

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )

# ================= ANALYTICS =================
elif menu == "📊 Analytics":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0}).fillna(0)

    st.subheader("📊 Analytics Dashboard")

    st.metric("Total Clients", len(df))
    st.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")

    st.bar_chart(df["Churn"].value_counts())

# ================= GEMINI =================
elif menu == "🧠 AI Insights":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))

    st.subheader("🧠 AI Business Insights")

    st.dataframe(df.head(), use_container_width=True)

    if analyze_data is None:
        st.error("Gemini module not available")
    else:
        if st.button("✨ Generate Insights"):

            with st.spinner("AI analyzing data..."):
                result = analyze_data(df)

            st.success("Analysis completed")
            st.write(result)