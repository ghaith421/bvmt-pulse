import feedparser
import urllib.parse
import os
import json
import re
from google import genai
from google.genai import types
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

client_gemini = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
client_groq = Groq(api_key=groq_api_key) if groq_api_key else None

def collecter_news(banque_nom: str, max_articles: int = 5) -> list:
    query = urllib.parse.quote(f"{banque_nom} Tunisie")
    url = f"https://news.google.com/rss/search?q={query}&hl=fr&gl=TN&ceid=TN:fr"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({"titre": entry.title, "date": entry.published, "lien": entry.link})
    return articles

def extraire_json(texte: str) -> dict:
    try:
        match = re.search(r'\{.*\}', texte, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(texte)
    except Exception:
        return {"score": 50, "justification": "Erreur de format"}

def analyser_sentiment_ia(articles: list) -> list:
    prompt_template = """
    Tu es un analyste financier. Analyse ce titre d'actualité sur une banque tunisienne :
    "{titre}"
    Donne un score de sentiment de 0 (négatif) à 100 (positif) et une justification courte.
    Format STRICT (JSON uniquement) : {{"score": 75, "justification": "Bonne nouvelle"}}
    """
    
    resultats = []
    for i, art in enumerate(articles, 1):
        print(f"\n--- Analyse News {i}/{len(articles)} ---")
        prompt = prompt_template.format(titre=art['titre'])
        analyse_reussie = False

        # --- TENTATIVE 1 : GROQ (openai/gpt-oss-20b) ---
        if not analyse_reussie and client_groq:
            try:
                print("⚡ Essai 1 : Groq (openai/gpt-oss-20b)...")
                completion = client_groq.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="openai/gpt-oss-20b", # Modèle actif recommandé par Groq
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                reponse = extraire_json(completion.choices[0].message.content)
                art["score_sentiment"] = reponse.get("score", 50)
                art["justification"] = reponse.get("justification", "")
                analyse_reussie = True
                print("✅ Succès Groq")
            except Exception as e:
                print(f"⚠️ Échec Groq : {e}")

        # --- TENTATIVE 2 : GEMINI (gemini-2.5-flash) ---
        if not analyse_reussie and client_gemini:
            try:
                print("⏳ Essai 2 : Gemini 2.5 Flash...")
                response = client_gemini.models.generate_content(
                    model='gemini-2.5-flash', # Modèle Gemini actuel
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    )
                )
                reponse = extraire_json(response.text)
                art["score_sentiment"] = reponse.get("score", 50)
                art["justification"] = reponse.get("justification", "")
                analyse_reussie = True
                print("✅ Succès Gemini")
            except Exception as e:
                print(f"⚠️ Échec Gemini : {e}")

        # --- SI TOUT ÉCHOUE ---
        if not analyse_reussie:
            art["score_sentiment"] = 50
            art["justification"] = "Erreur IA (Tous les modèles ont échoué)"
            
        resultats.append(art)
    
    return resultats

def score_sentiment_global(articles_analyses: list) -> int:
    if not articles_analyses: return 50
    scores = [art.get("score_sentiment", 50) for art in articles_analyses]
    return int(sum(scores) / len(scores))