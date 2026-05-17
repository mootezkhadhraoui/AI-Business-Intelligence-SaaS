import os
import pandas as pd
import time
from google import genai
import streamlit as st

# ================= API KEY =================
API_KEY = st.secrets.get("GEMINI_API_KEY", None)

client = genai.Client(api_key=API_KEY) if API_KEY else None


# ================= DATA ANALYSIS =================
def analyze_data(df: pd.DataFrame):

    if client is None:
        return "⚠️ GEMINI_API_KEY missing"

    summary = f"""
Dataset Overview:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Missing values: {df.isna().sum().sum()}

Sample data:
{df.head(5).to_string()}
"""

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Analyse ce dataset et donne des insights business:\n" + summary
            )
            return response.text

        except Exception as e:
            last_error = str(e)
            time.sleep(2 * (attempt + 1))

    return f"""
⚠️ Gemini temporarily unavailable

Reason: {last_error}
"""


# ================= CHAT FUNCTION =================
def chat_with_gemini(prompt: str, df: pd.DataFrame):

    if client is None:
        return "⚠️ GEMINI_API_KEY missing"

    context = f"""
You are a Business Intelligence AI assistant.

Dataset info:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Sample:
{df.head(5).to_string()}

User question:
{prompt}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=context
        )
        return response.text

    except Exception as e:
        return f"⚠️ Error: {str(e)}"