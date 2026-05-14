import sys
import os
sys.path.append(os.path.abspath("src"))
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.logger import log_event
from gemini_analysis import analyze_data

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Business Intelligence App", layout="wide")

st.title("🤖 AI Business Intelligence App")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/champion_model.pkl")

# ---------------- LOAD FEATURES ----------------
import json

with open("models/features.json", "r") as f:
    features = json.load(f)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Prédiction unitaire", "Upload CSV", "Dashboard", "🧠 IA Générative"]
)

# ---------------- ACCUEIL ----------------
if menu == "Accueil":

    st.header("📊 Présentation du projet")

    st.write("""
    ✅ Machine Learning multi-modèles  
    ✅ MLflow + DagsHub  
    ✅ Modèle Champion automatique  
    ✅ Streamlit App  
    ✅ IA Générative (Gemini)  
    """)

# ---------------- PREDICTION UNITAIRE ----------------
elif menu == "Prédiction unitaire":

    st.header("🔮 Prédiction client")

    input_data = []

    for feature in features:
        value = st.number_input(f"{feature}", value=0)
        input_data.append(value)

    if st.button("Prédire"):

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]

        if prediction == 1:
            st.error("❌ Client va quitter (Churn)")
        else:
            st.success("✅ Client fidèle")

# ---------------- UPLOAD CSV ----------------
elif menu == "Upload CSV":

    st.header("📁 Prédiction batch CSV")

    file = st.file_uploader("Uploader un fichier CSV")

    if file:

        df = pd.read_csv(file)

        st.write("Aperçu des données")
        st.dataframe(df.head())

        if st.button("Prédire"):

            predictions = model.predict(df)

            df["Prediction"] = predictions

            st.write(df)

# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":

    st.header("📊 Dashboard Analytics")

    df = pd.read_csv("data/raw/customer_churn.csv")

    st.subheader("Aperçu dataset")
    st.dataframe(df.head())

    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    st.subheader("Répartition Churn")

    st.bar_chart(df["Churn"].value_counts())

# ---------------- IA GENERATIVE ----------------
elif menu == "🧠 IA Générative":

    st.header("🧠 Analyse IA avec Gemini")

    df = pd.read_csv("data/raw/customer_churn.csv")

    st.write("Aperçu dataset")
    st.dataframe(df.head())

    if st.button("Analyser avec IA"):

        with st.spinner("Analyse en cours..."):

            result = analyze_data(df)

            st.success("Analyse terminée")

            st.write(result)