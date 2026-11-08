import os
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta

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
    Récupère l'historique des cours pour une banque.
    Si le scraping échoue (ex: bloqué par le cloud), utilise un fichier CSV local en secours.
    """
    # On définit le chemin du fichier CSV de secours
    csv_path = f"data/{symbol}_historique.csv"
    url_api = f"https://www.ilboursa.com/api/charting/GetTicksEOD?symbol={symbol}&length={length}&period=0&addIntradayArray=false"
    
    print(f"⏳ Tentative de récupération des données pour {symbol} via le web...")
    
    # --- TENTATIVE 1 : SCRAPING EN LIGNE ---
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = browser.new_context()
            page = context.new_page()
            
            # On réduit le timeout pour ne pas bloquer l'utilisateur trop longtemps
            page.goto("https://www.ilboursa.com/", timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            
            response = page.request.get(
                url_api,
                headers={
                    "Referer": "https://www.ilboursa.com/",
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json, text/javascript, */*; q=0.01"
                }
            )
            
            if response.status == 200:
                data = response.json()
                
                if data and "QuoteTab" in data and data["QuoteTab"]:
                    df = pd.DataFrame(data["QuoteTab"])
                    
                    # Nettoyage et formatage
                    df = df.rename(columns={"d": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
                    df["Date"] = df["Date"].apply(excel_date_to_datetime)
                    df = df.set_index("Date").sort_index()
                    
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # SAUVEGARDE POUR LE MODE SECOURS
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(csv_path)
                    print(f"✅ Succès Web ! Données sauvegardées dans {csv_path}")
                    
                    return df
            else:
                print(f"⚠️ Le site a renvoyé une erreur HTTP {response.status}.")

    except Exception as e:
        print(f"⚠️ Échec du scraping web : {e}")
    finally:
        try:
            browser.close()
        except:
            pass

    # --- TENTATIVE 2 : MODE SECOURS (LECTURE DU CSV) ---
    print(f"🔄 Passage en mode secours (lecture de {csv_path})...")
    if os.path.exists(csv_path):
        try:
            df_fallback = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            # On s'assure que les colonnes numériques sont bien typées
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df_fallback[col] = pd.to_numeric(df_fallback[col], errors='coerce')
            
            print("✅ Succès local ! Données chargées depuis le fichier CSV.")
            return df_fallback
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
    else:
        print(f"❌ Aucun fichier de secours trouvé à l'emplacement : {csv_path}")

    # Si tout échoue, on renvoie un DataFrame vide
    return pd.DataFrame()