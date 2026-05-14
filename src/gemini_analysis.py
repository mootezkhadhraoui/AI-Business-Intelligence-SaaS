import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

def analyze_data(df: pd.DataFrame):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "❌ GEMINI_API_KEY introuvable dans .env"

    client = genai.Client(api_key=api_key)

    summary = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Head:
{df.head(5).to_string()}
Missing:
{df.isna().sum().to_string()}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
Analyse ce dataset :

{summary}
"""
        )

        return response.text

    except Exception as e:
        return f"❌ Gemini error: {str(e)}"