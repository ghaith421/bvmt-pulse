import os
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import requests
from io import StringIO

# Dictionnaire des banques (Nom -> Ticker ilboursa)
BANQUES_ILBOURSA = {
    "Amen Bank": "AB",
    "Attijari Bank": "TJARI",
    "Arab Tunisian Bank": "ATB",
    "Banque de Tunisie": "BT",
    "Banque Nationale Agricole": "BNA",
    "Banque de l'Habitat": "BH",
    "Société Tunisienne de Banque": "STB",
    "BIAT": "BIAT",
    "Union Internationale de Banques": "UIB",
    "Union Bancaire pour le Commerce et l'Industrie": "UBCI",
    "Banque de Tunisie et des Emirats": "BTE",
    "Wifak International Bank": "WIFAK"
}

def excel_date_to_datetime(excel_date: int) -> datetime:
    """Convertit une date sérialisée Excel en objet datetime Python."""
    return datetime(1899, 12, 30) + timedelta(days=excel_date)

def get_historique_banque(symbol: str, length: int = 180) -> pd.DataFrame:
    """
    Récupère l'historique des cours.
    1. Tente le scraping (si pas bloqué).
    2. Tente de lire sur GitHub (pour le cloud).
    3. Tente de lire en local (pour le PC).
    """
    csv_path = f"data/{symbol}_historique.csv"
    url_api = f"https://www.ilboursa.com/api/charting/GetTicksEOD?symbol={symbol}&length={length}&period=0&addIntradayArray=false"
    
    # --- TENTATIVE 1 : SCRAPING EN LIGNE ---
    print(f"⏳ Tentative de récupération pour {symbol} via le web...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = browser.new_context()
            page = context.new_page()
            
            page.goto("https://www.ilboursa.com/", timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            
            response = page.request.get(
                url_api,
                headers={"Referer": "https://www.ilboursa.com/", "X-Requested-With": "XMLHttpRequest"}
            )
            
            if response.status == 200:
                data = response.json()
                if data and "QuoteTab" in data and data["QuoteTab"]:
                    df = pd.DataFrame(data["QuoteTab"])
                    df = df.rename(columns={"d": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
                    df["Date"] = df["Date"].apply(excel_date_to_datetime)
                    df = df.set_index("Date").sort_index()
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(csv_path)
                    print(f"✅ Succès Web ! Sauvegardé dans {csv_path}")
                    return df
    except Exception as e:
        print(f"⚠️ Échec du scraping : {e}")
    finally:
        try:
            browser.close()
        except:
            pass

    # --- TENTATIVE 2 : LECTURE DEPUIS GITHUB (Pour Streamlit Cloud) ---
    print("🔄 Passage en mode secours (lecture depuis GitHub)...")
    # ⚠️ REMPLACEZ PAR VOTRE NOM D'UTILISATEUR GITHUB SI CELA A CHANGÉ
    GITHUB_USER = "ghaith421" 
    GITHUB_REPO = "bvmt-pulse"
    
    github_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/data/{symbol}_historique.csv"
    
    try:
        url_avec_cache_buster = f"{github_url}?t={datetime.now().timestamp()}"
        res = requests.get(url_avec_cache_buster, timeout=10)
        
        if res.status_code == 200:
            df_fallback = pd.read_csv(StringIO(res.text), index_col="Date", parse_dates=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df_fallback[col] = pd.to_numeric(df_fallback[col], errors='coerce')
            print("✅ Succès ! Données chargées depuis GitHub.")
            return df_fallback
    except Exception as e:
        print(f"⚠️ Échec lecture GitHub : {e}")

    # --- TENTATIVE 3 : LECTURE LOCALE (Si sur PC) ---
    if os.path.exists(csv_path):
        try:
            df_local = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
            print("✅ Succès ! Données chargées depuis le fichier local.")
            return df_local
        except Exception as e:
            print(f"❌ Erreur lecture locale : {e}")

    return pd.DataFrame()