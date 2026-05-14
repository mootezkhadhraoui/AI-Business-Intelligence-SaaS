import os
from dotenv import load_dotenv
import google.generativeai as genai

# charger .env
load_dotenv()

# config API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# lister les modèles
models = genai.list_models()

print("📌 MODELS DISPONIBLES:\n")

for m in models:
    print("➡️", m.name, "|", m.supported_generation_methods)