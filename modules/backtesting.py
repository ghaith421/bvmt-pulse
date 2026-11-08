import pandas as pd
import numpy as np
from modules.technical_analysis import calculer_score_technique

def simuler_strategie(df: pd.DataFrame, capital_initial: float = 10000.0):
    """
    Simule une stratégie de trading basée sur le score technique.
    Règle : Achat si score >= 60, Vente si score <= 40.
    """
    df = df.copy()
    
    # On commence à trader après 50 jours pour laisser le temps aux moyennes mobiles de se former
    if len(df) < 60:
        return None, None
        
    df_sim = df.iloc[50:].copy()
    
    # 1. Génération des signaux historiques
    signaux = []
    for i in range(len(df_sim)):
        # On calcule le score sur les données disponibles jusqu'à ce jour J
        score_jour = calculer_score_technique(df.iloc[:50+i])
        if score_jour >= 60:
            signaux.append(1) # Achat / Conserver
        elif score_jour <= 40:
            signaux.append(0) # Vente / Cash
        else:
            # Si neutre, on garde la position précédente
            if len(signaux) > 0:
                signaux.append(signaux[-1])
            else:
                signaux.append(0) # Au début, on attend
                
    df_sim['Position'] = signaux
    
    # 2. Calcul des rendements quotidiens
    df_sim['Rendement_Quotidien'] = df_sim['Close'].pct_change()
    
    # Rendement de la stratégie (on applique la position de la VEILLE)
    df_sim['Rendement_Strategie'] = df_sim['Position'].shift(1) * df_sim['Rendement_Quotidien']
    
    # 3. Calcul de la valeur du portefeuille au fil du temps
    df_sim['Valeur_Strategie'] = capital_initial * (1 + df_sim['Rendement_Strategie']).cumprod()
    df_sim['Valeur_Buy_Hold'] = capital_initial * (1 + df_sim['Rendement_Quotidien']).cumprod()
    
    # Remplir les NaN du début par le capital initial
    df_sim['Valeur_Strategie'] = df_sim['Valeur_Strategie'].fillna(capital_initial)
    df_sim['Valeur_Buy_Hold'] = df_sim['Valeur_Buy_Hold'].fillna(capital_initial)
    
    return df_sim, capital_initial