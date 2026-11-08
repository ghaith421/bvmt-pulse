import os
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

print("--- TEST DES CLÉS API ---")

# Test Gemini
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("❌ GEMINI_API_KEY n'est pas trouvée dans le .env")
else:
    print(f"✅ Clé Gemini trouvée (commence par {gemini_key[:5]}...)")
    try:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents="Réponds juste par le mot: OK"
        )
        print(f"✅ Gemini fonctionne ! Réponse : {response.text}")
    except Exception as e:
        print(f"❌ Erreur Gemini : {e}")

print("-" * 20)

# Test Groq
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("❌ GROQ_API_KEY n'est pas trouvée dans le .env")
else:
    print(f"✅ Clé Groq trouvée (commence par {groq_key[:5]}...)")
    try:
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Réponds juste par le mot: OK"}],
            model="llama-3.1-8b-instant",
        )
        print(f"✅ Groq fonctionne ! Réponse : {completion.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Erreur Groq : {e}")