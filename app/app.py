import os
import sys
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

# ---------------- PATH FIX ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---------------- SAFE IMPORT ----------------
try:
    from src.gemini_analysis import analyze_data
except Exception as e:
    print("IMPORT ERROR:", e)
    analyze_data = None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Business Intelligence App", layout="wide")

st.title("🤖 AI Business Intelligence SaaS")

# ---------------- LOAD MODEL ----------------
model = joblib.load(os.path.join(PROJECT_ROOT, "models", "champion_model.pkl"))

# ---------------- LOAD FEATURES ----------------
with open(os.path.join(PROJECT_ROOT, "models", "features.json"), "r") as f:
    features = json.load(f)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Prédiction unitaire", "Upload CSV", "Dashboard", "🧠 IA Générative"]
)

# ---------------- ACCUEIL ----------------
if menu == "Accueil":

    st.header("📊 SaaS AI Business Intelligence Dashboard")

    st.write("Bienvenue dans ton plateforme IA de churn prediction 🚀")

    # Load dataset
    df = pd.read_csv("data/raw/customer_churn.csv")

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # KPIs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📦 Total clients", len(df))

    with col2:
        st.metric("📉 Churn rate", f"{round(df['Churn'].mean()*100,2)}%")

    with col3:
        st.metric("✅ Active clients", int((df["Churn"] == 0).sum()))

    st.divider()

    # Preview data
    st.subheader("📄 Aperçu dataset")
    st.dataframe(df.head())

    # Charts
    st.subheader("📊 Churn distribution")
    st.bar_chart(df["Churn"].value_counts())

    st.subheader("💰 Monthly Charges")
    if "MonthlyCharges" in df.columns:
        st.line_chart(df["MonthlyCharges"])

# ---------------- PREDICTION ----------------
elif menu == "Prédiction unitaire":
    st.header("🔮 Prédiction client")

    input_data = []
    for feature in features:
        value = st.number_input(feature, value=0.0)
        input_data.append(value)

    if st.button("Prédire"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        if prediction == 1:
            st.error("❌ Client risque de churn")
        else:
            st.success("✅ Client fidèle")

# ---------------- UPLOAD CSV ----------------
elif menu == "Upload CSV":

    st.header("📁 Prédiction batch")

    file = st.file_uploader("Uploader CSV")

    if file:

        df = pd.read_csv(file)
        st.dataframe(df.head())

        if st.button("Lancer prédictions"):

            df_clean = df.copy()

            # ---------------- DROP NON FEATURES ----------------
            drop_cols = ["customerID", "Churn"]
            df_clean = df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns])

            # ---------------- ENCODING CATEGORICAL ----------------
            df_clean = pd.get_dummies(df_clean)

            # ---------------- ALIGN WITH TRAIN FEATURES ----------------
            df_clean = df_clean.reindex(columns=features, fill_value=0)

            # ---------------- PREDICT ----------------
            preds = model.predict(df_clean)

            df["Prediction"] = preds

            st.dataframe(df)
# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":
    st.header("📊 SaaS Analytics Dashboard")

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn.csv"))

    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📦 Total clients", len(df))

    with col2:
        st.metric("📉 Churn rate", f"{round(df['Churn'].mean()*100,2)}%")

    with col3:
        st.metric("✅ Clients actifs", int((df["Churn"] == 0).sum()))

# ---------------- IA ----------------
elif menu == "🧠 IA Générative":

    st.header("🧠 Analyse IA (Gemini)")

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn.csv"))

    st.dataframe(df.head())

    api_key = os.getenv("GEMINI_API_KEY")

    if analyze_data is None:
        st.error("❌ Module Gemini introuvable")
    elif not api_key:
        st.error("❌ GEMINI_API_KEY manquante dans .env")
    else:
        if st.button("Analyser avec IA"):
            with st.spinner("Analyse en cours..."):
                result = analyze_data(df)
                st.success("Analyse terminée")
                st.write(result)