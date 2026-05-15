import os
import pandas as pd
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def analyze_data(df: pd.DataFrame):

    if client is None:
        return "⚠️ API key manquante (GEMINI_API_KEY)"

    summary = f"""
Dataset Overview:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Missing values: {df.isna().sum().sum()}

Sample data:
{df.head(5).to_string()}
"""

    # 🔥 retry system (important pour éviter 503)
    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Analyse ce dataset et donne des insights business:\n" + summary
            )

            return response.text

        except Exception as e:
            wait_time = 2 * (attempt + 1)
            time.sleep(wait_time)

            last_error = str(e)

    return f"""
⚠️ Gemini temporairement indisponible

Cause probable: surcharge API (503)
Solution: réessayer dans quelques minutes

Détail technique: {last_error}
"""