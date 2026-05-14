import google.generativeai as genai

genai.configure(api_key="TON_API_KEY_ICI")

print("\n🔍 Liste des modèles Gemini disponibles :\n")

for model in genai.list_models():
    print(model.name)