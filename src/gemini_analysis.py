from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_data(df):

    prompt = f"""
Tu es un data analyst expert en business intelligence.

Analyse ce dataset :

1. Résumé business
2. Causes du churn
3. Insights importants
4. Recommandations

DATA:
{df.head(20).to_string()}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text