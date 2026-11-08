import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime

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

def get_historique_banque(symbol: str, length: int = 180) -> pd.DataFrame:
    """
    Récupère l'historique des cours pour une banque spécifique via Playwright.
    """
    url_api = f"https://www.ilboursa.com/api/charting/GetTicksEOD?symbol={symbol}&length={length}&period=0&addIntradayArray=false"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # 1. Initialisation de la session
            page.goto("https://www.ilboursa.com/", timeout=15000, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            
            # 2. Appel de l'API
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
                
                if not data or "QuoteTab" not in data or not data["QuoteTab"]:
                    print(f"⚠️ Données vides ou introuvables pour {symbol}")
                    return pd.DataFrame()

                df = pd.DataFrame(data["QuoteTab"])
                
                # 3. Renommer les colonnes (sans utiliser inplace=True pour éviter les warnings)
                df = df.rename(columns={
                    "d": "Date_Orm", "o": "Open", "h": "High", 
                    "l": "Low", "c": "Close", "v": "Volume"
                })
                
                # 4. Créer un calendrier boursier fiable (jours ouvrés se terminant aujourd'hui)
                df['Date'] = pd.bdate_range(end=datetime.today(), periods=len(df))
                
                # 5. Définir la date comme index et trier
                df = df.set_index("Date")
                df = df.sort_index()
                
                # 6. S'assurer que les colonnes sont numériques
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # On garde uniquement les colonnes utiles
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                
                return df
            else:
                print(f"❌ Erreur HTTP {response.status} pour {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Erreur inattendue pour {symbol} : {e}")
            return pd.DataFrame()
        finally:
            browser.close()

if __name__ == "__main__":
    print("=== Test de collecte pour la BIAT ===")
    
    df_biat = get_historique_banque("BIAT", length=180)
    
    if not df_biat.empty:
        print("✅ Superbe ! Voici les 5 derniers jours de la BIAT avec de vraies dates :")
        print(df_biat.tail(5))
        print(f"\nNombre total de jours d'historique : {len(df_biat)}")