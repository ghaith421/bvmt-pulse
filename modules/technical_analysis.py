import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def calculer_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques de base."""
    df = df.copy()
    
    # Moyennes mobiles
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (14 jours)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogramme'] = df['MACD'] - df['Signal']
    
    return df

def calculer_score_technique(df: pd.DataFrame) -> int:
    """Attribue un score technique sur 100 basé sur les indicateurs actuels."""
    if df.empty or len(df) < 50:
        return 50 # Neutre si pas assez de données
        
    dernier = df.iloc[-1]
    score = 50 # Base neutre
    
    # 1. Tendance (Prix vs SMA 50) -> +/- 20 points
    if dernier['Close'] > dernier['SMA_50']:
        score += 20
    else:
        score -= 20
        
    # 2. RSI (Surachat / Survente) -> +/- 15 points
    if dernier['RSI'] < 30:
        score += 15 # Survente = potentiel d'achat
    elif dernier['RSI'] > 70:
        score -= 15 # Surachat = potentiel de vente
        
    # 3. MACD (Croisement) -> +/- 15 points
    if dernier['MACD'] > dernier['Signal']:
        score += 15 # Tendance haussière
    else:
        score -= 15 # Tendance baissière
        
    # On s'assure que le score reste entre 0 et 100
    return max(0, min(100, int(score)))

def predire_tendance_ml(df: pd.DataFrame) -> int:
    """
    Entraîne un modèle Random Forest pour prédire la probabilité 
    que le cours monte le lendemain (en %).
    """
    df = df.copy()
    
    # 1. Création de la cible (Target) : 1 si le cours monte le lendemain, 0 sinon
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # On supprime la dernière ligne car on ne connaît pas encore le cours de demain
    df = df.dropna()
    
    if len(df) < 50:
        return 50 # Pas assez de données pour entraîner l'IA

    # 2. Sélection des variables (Features)
    features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']
    X = df[features]
    y = df['Target']
    
    # 3. Séparation des données (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Entraînement du modèle Random Forest
    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    
    # 5. Prédiction sur la toute dernière ligne (aujourd'hui)
    derniere_ligne = X.iloc[-1:]
    probabilite_hausse = modele.predict_proba(derniere_ligne)[0][1] # Probabilité de la classe 1 (Hausse)
    
    return int(probabilite_hausse * 100)