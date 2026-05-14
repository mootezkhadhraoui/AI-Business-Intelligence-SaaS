import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


def analyze_data(df: pd.DataFrame):

    if client is None:
        return "❌ API key manquante"

    summary = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing: {df.isna().sum().sum()}
Head:
{df.head(5).to_string()}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Analyse ce dataset:\n" + summary
        )

        return response.text

    except Exception as e:
        return f"❌ Gemini error: {str(e)}"