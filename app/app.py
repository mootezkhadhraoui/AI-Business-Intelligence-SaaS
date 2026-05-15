import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

# ================= CONFIG UI =================
st.set_page_config(page_title="AI Business Intelligence App", layout="wide")

# ================= STYLE UX =================
st.markdown("""
<style>

.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: #ffffff;
}

.stButton>button {
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #3730A3;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Business Intelligence SaaS")
st.success("App démarrée correctement ✅")

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
except:
    analyze_data = None

try:
    from src.logger import log_event
except:
    def log_event(*args, **kwargs):
        pass

# ================= MODEL SAFE LOADER =================
def load_model_safe():
    try:
        import joblib
        path = os.path.join(PROJECT_ROOT, "models/champion_model.pkl")

        if os.path.exists(path):
            return joblib.load(path)

    except Exception:
        pass

    try:
        from src.champion import load_champion_model
        return load_champion_model()

    except Exception:
        return None


# ================= LOAD FEATURES =================
features_path = os.path.join(PROJECT_ROOT, "models", "features.json")

if not os.path.exists(features_path):
    st.error("features.json introuvable")
    st.stop()

with open(features_path, "r") as f:
    features = json.load(f)

# ================= SIDEBAR =================
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Prédiction unitaire", "Upload CSV", "Dashboard", "IA"]
)

# ================= ACCUEIL =================
if menu == "Accueil":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0}).fillna(0)

    st.title("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Clients", len(df))
    col2.metric("📉 Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
    col3.metric("✅ Active Clients", int((df["Churn"] == 0).sum()))

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("📊 Churn Distribution")
        st.bar_chart(df["Churn"].value_counts())

    with col5:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)


# ================= PREDICTION =================
elif menu == "Prédiction unitaire":

    st.title("🔮 AI Prediction")

    input_data = []

    cols = st.columns(2)

    for i, feature in enumerate(features):
        with cols[i % 2]:
            input_data.append(st.number_input(feature, value=0.0))

    if st.button("🚀 Predict Now"):

        with st.spinner("AI is thinking... 🤖"):

            model = load_model_safe()

            if model is None:
                st.error("Model unavailable")
            else:
                pred = model.predict(np.array(input_data).reshape(1, -1))[0]

                log_event("prediction", str(input_data))

                if pred == 1:
                    st.error("❌ HIGH RISK: Customer will churn")
                else:
                    st.success("✅ LOW RISK: Customer retained")


# ================= UPLOAD CSV =================
elif menu == "Upload CSV":

    st.title("📁 Batch AI Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("⚡ Run Prediction"):

            with st.spinner("Processing..."):

                model = load_model_safe()

                if model is None:
                    st.error("Model unavailable")
                    st.stop()

                df = df.drop(columns=["customerID"], errors="ignore")
                df = df.replace({"Yes":1,"No":0,"Female":0,"Male":1})
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

                df = df[features]

                preds = model.predict(df)
                df["Prediction"] = preds

                st.success("✅ Prediction completed!")

                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "⬇️ Download Results",
                    df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )


# ================= DASHBOARD =================
elif menu == "Dashboard":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0}).fillna(0)

    st.title("📊 Analytics Dashboard")

    st.metric("Total Clients", len(df))
    st.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")

    st.bar_chart(df["Churn"].value_counts())


# ================= IA =================
elif menu == "IA":

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/customer_churn.csv"))
    st.dataframe(df.head())

    if analyze_data and st.button("✨ Generate Insights"):

        with st.spinner("AI analyzing data..."):

            result = analyze_data(df)

        st.success("Analysis completed")
        st.write(result)