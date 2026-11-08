import os
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import requests
from io import StringIO

# ... (Gardez BANQUES_ILBOURSA et excel_date_to_datetime) ...

def get_historique_banque(symbol: str, length: int = 180) -> pd.DataFrame:
    # ... (Gardez la partie TENTATIVE 1 : SCRAPING EN LIGNE avec Playwright) ...

    # --- TENTATIVE 2 : MODE SECOURS (LECTURE DEPUIS GITHUB) ---
    print("🔄 Passage en mode secours (lecture depuis GitHub)...")
    
    # ⚠️ REMPLACEZ PAR VOTRE NOM D'UTILISATEUR GITHUB
    GITHUB_USER = "ghaith421" 
    GITHUB_REPO = "bvmt-pulse"
    
    # URL "Raw" de votre fichier CSV sur GitHub
    github_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/data/{symbol}_historique.csv"
    
    try:
        # On ajoute un paramètre de temps pour éviter le cache du navigateur
        url_avec_cache_buster = f"{github_url}?t={datetime.now().timestamp()}"
        response = requests.get(url_avec_cache_buster, timeout=10)
        
        if response.status_code == 200:
            df_fallback = pd.read_csv(StringIO(response.text), index_col="Date", parse_dates=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df_fallback[col] = pd.to_numeric(df_fallback[col], errors='coerce')
            print("✅ Succès local ! Données chargées depuis GitHub.")
            return df_fallback
        else:
            print(f"❌ Échec de la récupération depuis GitHub (Code {response.status_code})")
    except Exception as e:
        print(f"❌ Erreur lors de la lecture depuis GitHub : {e}")

    # Dernier recours : le fichier local (si vous êtes sur votre PC)
    if os.path.exists(f"data/{symbol}_historique.csv"):
        return pd.read_csv(f"data/{symbol}_historique.csv", index_col="Date", parse_dates=True)

    return pd.DataFrame()