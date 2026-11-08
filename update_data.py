# update_data.py
import sys
import os

# Ajoute le dossier courant au chemin pour importer vos modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.market_data import get_historique_banque, BANQUES_ILBOURSA

def main():
    print("Début de la mise à jour des données...")
    for nom_banque, ticker in BANQUES_ILBOURSA.items():
        print(f"Mise à jour de {nom_banque} ({ticker})...")
        # L'appel à cette fonction va scraper le site et écraser le fichier CSV correspondant
        df = get_historique_banque(ticker, length=180)
        if not df.empty:
            print(f"✅ {nom_banque} mis à jour avec succès.")
        else:
            print(f"❌ Échec pour {nom_banque}.")
    print("Mise à jour terminée.")

if __name__ == "__main__":
    main()