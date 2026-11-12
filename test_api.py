import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ La clé GEMINI_API_KEY n'est pas trouvée dans le fichier .env")
else:
    print(f"✅ Clé trouvée (commence par {api_key[:5]}...)")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-3.8-flash', contents="Réponds juste OK")
        print(f"✅ Gemini fonctionne : {response.text}")
    except Exception as e:
        print(f"❌ Erreur Gemini : {e}")