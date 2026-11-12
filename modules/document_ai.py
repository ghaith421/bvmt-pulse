#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
document_ai.py — Analyse fondamentale d'un rapport annuel bancaire (PDF) — VERSION 6
====================================================================================

OBJECTIF
--------
Extraire TOUT ce qui est disponible dans un rapport annuel de banque tunisienne (français / anglais) :
états financiers, notes annexes, rapport de gestion (texte), ratios publiés, pages scannées (images) ;
puis recalculer, croiser et noter.

ARCHITECTURE (à présenter au jury) — « 3 niveaux + 1 filet de sécurité »
------------------------------------------------------------------------

   PDF ──► [0] LECTURE MULTI-SOURCES
              • texte natif (PyMuPDF)                    → lignes reconstruites géométriquement
              • pages scannées / images                  → OCR local RapidOCR (gratuit, rapide)
              • OCR peu fiable sur une page financière   → relecture par Gemini Vision (transcription)
              Toutes les sources sont ramenées au MÊME format : des « lignes » « Libellé | N | N-1 ».
        │
        ├─► [1] NIVEAU GÉOMÉTRIQUE (sans IA)  : chaque ligne de tableau est comparée au DICTIONNAIRE
        │        MASSIF FR/EN (section 3, ~700 synonymes). Le chiffre est lu par le CODE, colonne N / N-1.
        │
        ├─► [2] NIVEAU SÉMANTIQUE (sans IA)   : recherche par regex, basée sur le MÊME dictionnaire, dans
        │        le texte courant (rapport de gestion, notes) : « Notre ROE s'élève à 14,5 % »,
        │        « Les dépôts ont atteint 1 200 millions de dinars », « passe de 75 % à 78 % »…
        │
        ├─► [3] NIVEAU « LLM POINTEUR » (seulement si nécessaire) : le LLM ne recopie JAMAIS un chiffre.
        │        On lui présente des lignes/phrases numérotées (p12:L034) ; il répond uniquement
        │        « la donnée X est sur p12:L034, 1er nombre ». Le CODE lit alors le chiffre dans la
        │        ligne réelle. ⇒ une hallucination numérique est structurellement impossible.
        │        Il sert (a) à combler les champs introuvables, (b) à arbitrer les candidats ambigus.
        │
        └─► [4] VALIDATION CROISÉE (Python pur) : cohérence comptable (bilan équilibré, CP < actif…),
                 recalcul de chaque ratio à partir des montants bruts et comparaison au ratio publié
                 (alerte si écart > 10 %), calcul automatique des ratios manquants, score pondéré,
                 indice de confiance.

POINTS CLÉS À EXPLIQUER
-----------------------
• Le dictionnaire est UNIQUE et déclaratif (section 3) : une seule source de vérité qui alimente
  (1) la notation des pages, (2) l'appariement des libellés de tableaux, (3) la recherche dans le texte,
  (4) la présélection des lignes montrées au LLM. Ajouter un synonyme = ajouter un mot dans une chaîne.
• Normalisation linguistique légère (« racinisation » FR/EN, section 2) : « créances classées »,
  « créance classée », « Créances Classées » → même clé ; « liabilities »/« liability » idem.
• Plusieurs niveaux de priorité (forts / moyens / faibles) + exclusions : « Total actifs » ≠
  « Total actifs financiers » ; « Capitaux propres » ≠ « Capitaux propres avant résultat ».
• Le choix final d'un chiffre = SCORE (précision du libellé, type de page, colonne N-1, périmètre
  consolidé, corroboration par plusieurs pages/sources, cohérence avec le total bilan…).

Dépendances obligatoires : pip install pymupdf python-dotenv
Dépendances facultatives : pip install rapidocr-onnxruntime numpy   (OCR local)
                           pip install google-genai groq openai       (LLM pointeur + Vision)
Variables .env (toutes facultatives) : GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY,
  LLM_ORDER, GEMINI_MODEL, GEMINI_VISION_MODEL, OPENROUTER_MODEL, GROQ_MODEL, HEDGE_DELAY_S,
  OCR_DPI, OCR_WORKERS, OCR_BUDGET_S, MAX_PAGES_VISION, VERIFICATION_STRICTE=1, ANALYSE_CACHE_DIR
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

import pymupdf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # .env facultatif
    pass

try:
    import numpy as np
except ImportError:
    np = None
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# OCR local : RapidOCR (ancien paquet « rapidocr_onnxruntime » ou nouveau paquet « rapidocr »)
OCR_CLASSE = None
for _nom_module in ("rapidocr_onnxruntime", "rapidocr"):
    try:
        OCR_CLASSE = getattr(__import__(_nom_module), "RapidOCR")
        break
    except Exception:
        continue

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
VERBOSE = True
PROMPT_VERSION = "v6.0"                      # changer => invalide le cache
CACHE_DIR = Path(os.getenv("ANALYSE_CACHE_DIR", ".cache_analyses"))

LLM_ORDER = [p.strip() for p in os.getenv("LLM_ORDER", "gemini,openrouter,groq").split(",") if p.strip()]
MODEL_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MODEL_VISION = os.getenv("GEMINI_VISION_MODEL", MODEL_GEMINI)
MODEL_OPENROUTER = os.getenv("OPENROUTER_MODEL", "openrouter/free")
MODEL_GROQ = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
LLM_TIMEOUT_S = 45
HEDGE_DELAY_S = float(os.getenv("HEDGE_DELAY_S", "12"))
LIMITE_CARACTERES = {"groq": 22_000}

# --- lecture du PDF ---
MAX_PAGES_SCAN = 800          # pages scannées au maximum
SCORE_PAGE_MIN = 3.0          # score minimal pour qu'une page soit retenue pour un thème
PAGES_PAR_TOPIC = 14          # meilleures pages retenues par thème (+ voisines des 3 premières)
MAX_PAGES_LIGNES = 120        # nombre maximal de pages dont on reconstruit les lignes de tableaux

# --- OCR / Vision ---
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_WORKERS = int(os.getenv("OCR_WORKERS", str(min(4, os.cpu_count() or 2))))
OCR_BUDGET_S = float(os.getenv("OCR_BUDGET_S", "70"))   # temps max alloué à l'OCR (puis on s'arrête)
MAX_PAGES_OCR = 80           # pages scannées OCRisées au maximum
MAX_PAGES_MIXTES = 40         # pages texte contenant de grosses images (infographies de ratios)
SEUIL_CONF_OCR = 0.85         # en dessous, une page financière est relue par Gemini Vision
MAX_PAGES_VISION = int(os.getenv("MAX_PAGES_VISION", "4"))
VISION_DPI = 150

# --- validation croisée ---
SEUIL_ECART_PCT = 10.0        # écart relatif ratio publié / recalculé au-delà duquel on alerte
ECART_ABS_MIN_PTS = 0.2       # ... et seulement si l'écart absolu dépasse 0,2 point (évite le bruit sur ROA)
VERIFICATION_STRICTE = os.getenv("VERIFICATION_STRICTE", "0") == "1"

# --- recherche sémantique (texte courant) ---
FENETRE_AVANT = 200           # caractères examinés avant un chiffre pour retrouver son libellé
DISTANCE_MAX_LIBELLE = 130    # distance maximale (caractères) entre le mot-clé et le chiffre


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# 2. Outils sur les nombres et le texte
# ---------------------------------------------------------------------------
_VALEURS_NULLES = {"", "n/a", "na", "nd", "n.d", "n.d.", "null", "none", "-", "—", "–", "nan"}


def parse_nombre(valeur) -> float | None:
    """Convertit '1 234 567,8' / '1.234.567,8' / '1,234,567.8' / '(1 234)' / '-3,6 %' en float."""
    if valeur is None or isinstance(valeur, bool):
        return None
    if isinstance(valeur, (int, float)):
        return float(valeur) if math.isfinite(valeur) else None
    s = str(valeur).strip()
    if s.lower() in _VALEURS_NULLES:
        return None
    negatif = bool(re.match(r"^\(.*\)$", s)) or bool(re.match(r"^[-−–]\s*\d", s)) or bool(re.match(r"^\([-−–]", s))
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") > 1:
        s = s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    else:
        s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if negatif else x


def plier(texte: str) -> str:
    """Minuscules, sans accents, apostrophes/tirets normalisés (le signe moins reste un « - »)."""
    t = (texte.replace("’", "'").replace("‘", "'").replace("\u00a0", " ").replace("\u202f", " ")
         .replace("−", "-").replace("–", "-").replace("—", "-").replace("œ", "oe").replace("Œ", "OE")
         .replace("æ", "ae"))
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()


def _plat(texte: str) -> str:
    """Texte plié sur une seule ligne (les libellés coupés sur 2 lignes restent détectables)."""
    return re.sub(r"\s+", " ", plier(texte))


# --- « Racinisation » légère FR/EN : rend « créances classées » = « créance classée » = « Créances Classées » ---
STOPS = {"de", "du", "des", "la", "le", "les", "l", "d", "en", "au", "aux", "sur", "pour", "par", "of", "the",
         "a", "an", "to", "for", "on", "in", "and", "et", "ou", "or", "with", "avec", "sous"}


@lru_cache(maxsize=300_000)
def racine(m: str) -> str:
    """Pluriels, féminins et suffixes usuels FR/EN. Appliquée de façon identique au dictionnaire ET au texte,
    donc les deux restent comparables même si la racine n'est pas un « vrai » radical."""
    if len(m) <= 3 or not m.isalpha():
        return m
    if m.endswith("euses"):
        return m[:-5] + "eu"
    if m.endswith("euse"):
        return m[:-4] + "eu"
    if m.endswith("eux"):
        return m[:-3] + "eu"          # douteux / douteuse / douteuses -> douteu
    if m.endswith("aux"):
        return m[:-3] + "al"          # capitaux -> capital, generaux -> general
    if m.endswith("sses"):
        m = m[:-2]                    # losses -> loss
    elif m.endswith("ies") and len(m) > 5:
        m = m[:-3] + "y"              # liabilities -> liability
    elif m.endswith(("ss", "us", "is")):
        pass
    elif m.endswith(("s", "x")):
        m = m[:-1]
    if len(m) > 4 and m.endswith("ee"):
        m = m[:-1]                    # classee -> classe
    if len(m) > 4 and m.endswith("e"):
        m = m[:-1]                    # classe -> class
    return m


_RE_MOT = re.compile(r"[a-z]+")


def racinise(texte_plie: str) -> str:
    return _RE_MOT.sub(lambda m: racine(m.group()), texte_plie)


# Unités abrégées sensibles à la casse (mDT = milliers, MDT = millions) : on les « protège » AVANT de passer
# en minuscules, sinon « mDT » et « MDT » deviennent indiscernables.
_RE_U_KILO = re.compile(r"(?<![A-Za-z])(?:mDT|KDT|kDT|mD|KD|kD|kTND|KTND|mTND|kEUR|k€|K€)(?![A-Za-z])")
_RE_U_MEGA = re.compile(r"(?<![A-Za-z])(?:MDT|MD|MTND|MEUR|M€|MMAD|MDH|MDZD)(?![A-Za-z])")


def _proteger_unites(texte: str) -> str:
    return _RE_U_MEGA.sub(" zzmega ", _RE_U_KILO.sub(" zzkilo ", texte))


def normaliser_texte(texte: str) -> str:
    """Texte « de recherche » : sans accents, minuscules, racinisé, sur une seule ligne, unités protégées."""
    t = plier(_proteger_unites(texte)).replace("'", " ")
    return re.sub(r"\s+", " ", racinise(t)).strip()


_RE_NON_LABEL = re.compile(r"[^a-z0-9 ]+")
_RE_CODE = re.compile(r"^(?:[a-z]{1,3}\d{1,2}[a-z]?|\d{1,2}[a-z]?|[ivx]{1,4}|[a-d])\s+(?=[a-z])")
_CODES_A_GARDER = {"cet1", "t1", "ct1", "at1", "t2", "tier1"}


def norm_label(s: str) -> str:
    """Libellé normalisé et racinisé, sans code comptable en tête (« AC3 - », « CP7 », « 1. »)."""
    t = _RE_NON_LABEL.sub(" ", plier(s).replace("'", " "))
    t = re.sub(r"\s+", " ", t).strip()
    m = _RE_CODE.match(t)
    if m and m.group(0).split()[0] not in _CODES_A_GARDER:
        t = t[m.end():].strip()
    return racinise(t)


def extraire_json(texte: str) -> dict:
    """Premier objet JSON valide trouvé dans la réponse du modèle (tolère ```json et bavardage)."""
    if not texte:
        return {}
    texte = re.sub(r"```(?:json)?", "", texte)
    decodeur = json.JSONDecoder()
    for m in re.finditer(r"\{", texte):
        try:
            obj, _ = decodeur.raw_decode(texte[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _entier(v) -> int | None:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 3. LE DICTIONNAIRE MASSIF FR/EN  (CŒUR DE LA PRÉCISION)
# ---------------------------------------------------------------------------
# Chaque indicateur est décrit UNE SEULE FOIS ci-dessous. Les listes de synonymes sont des chaînes
# séparées par « ; » (accents et casse sans importance, pluriels/féminins gérés par la racinisation).
#
#   F = libellés FORTS     (poids 10) : le libellé exact attendu dans un état financier / un tableau
#   M = libellés MOYENS    (poids 7)  : variantes fréquentes, IFRS anglais, formulations de rapport de gestion
#   f = libellés FAIBLES   (poids 4)  : libellés génériques, acceptés seulement en tableau (jamais en texte)
#   X = EXCLUSIONS                    : si le libellé contient un de ces mots, le champ est rejeté
#
# Genres : "montant" (devise, converti en unités), "pct" (ratio en %), "unitaire" (par action, sans conversion).
# prefixe=True : le libellé peut se prolonger (ex. « Taux de couverture des créances classées par les provisions »)
# prefixe=False: le libellé doit être EXACTEMENT la phrase du dictionnaire (+ mots neutres : consolidé, net…).
ACRONYMES_TEXTE = {"pnb", "roe", "roa", "roaa", "npl", "npls", "lcr", "nsfr", "cet1", "rwa", "nbi", "eps", "dps",
                   "nim", "ldr", "ltd", "cir", "rbe", "bpa"}
ACRO_STEMS = ACRONYMES_TEXTE | {racine(a) for a in ACRONYMES_TEXTE}

_STOP_ALT = "|".join(sorted(STOPS, key=len, reverse=True))
_SEP = r"(?: (?:%s))* " % _STOP_ALT                 # espace + mots vides optionnels entre deux mots-clés
_STOPS_OPT = r"(?: (?:%s))*" % _STOP_ALT
_NEUTRES = STOPS | {racine(w) for w in (
    "consolide consolides net nette nettes groupe part exercice periode annee apres modification modifications "
    "comptable comptables ifrs individuel annuel millier milliers million millions milliard milliards dinar dinars "
    "dt mdt kdt tnd ktnd md devise euro euros eur usd unite unites zzkilo zzmega").split()}
_NEUTRES_ALT = "|".join(sorted(_NEUTRES, key=len, reverse=True)) + r"|20\d\d|\d{1,3}"


def _lst(chaine: str) -> list:
    return [p.strip() for p in chaine.split(";") if p.strip()]


def _tokens_phrase(phrase: str) -> list:
    return [t for t in norm_label(phrase).split() if t not in STOPS]


def _rx_phrase(tokens: list) -> str:
    return _SEP.join(re.escape(t) for t in tokens)


def _rx_exclusion(phrases: list):
    """Exclusion par PRÉFIXE de mot (« financ » exclut financier, financement, financiers…)."""
    alts = []
    for p in phrases:
        toks = _tokens_phrase(p)
        if toks:
            alts.append(_SEP.join(re.escape(t) + r"\w*" for t in toks))
    return re.compile(r"(?:^| )(?:" + "|".join(alts) + r")") if alts else None


class Champ:
    """Un indicateur à extraire + ses regex compilées (construites automatiquement depuis le dictionnaire)."""

    def __init__(self, nom, desc, genre, topic, F, M, f, X, prefixe, plage, texte, essentiel, prefere_total):
        self.nom, self.desc, self.genre, self.topic = nom, desc, genre, topic
        self.forts, self.moyens, self.faibles, self.exclure = _lst(F), _lst(M), _lst(f), _lst(X)
        self.prefixe, self.plage, self.texte = prefixe, plage, texte
        self.essentiel, self.prefere_total = essentiel, prefere_total
        self.tiers, self.rx_excl, self.rx_texte, self.pivots = [], None, None, ()

    def compiler(self) -> None:
        # (a) appariement d'un libellé de TABLEAU : regex ancrées, par niveau de priorité
        for poids, phrases in ((10, self.forts), (7, self.moyens), (4, self.faibles)):
            alts = set()
            for p in phrases:
                toks = _tokens_phrase(p)
                if len(toks) > 1 and toks[0] == "total":
                    toks = toks[1:]                                  # « total » est géré par le préfixe optionnel
                if toks:
                    alts.add(_rx_phrase(toks))
            if not alts:
                continue
            corps = "|".join(sorted(alts, key=len, reverse=True))
            if self.prefixe:
                rx = re.compile(r"^(?:total%s )?(?:%s)(?: |$)" % (_STOPS_OPT, corps))
            else:
                rx = re.compile(r"^(?:total%s )?(?:%s)(?: (?:%s))*$" % (_STOPS_OPT, corps, _NEUTRES_ALT))
            self.tiers.append((poids, rx))
        self.rx_excl = _rx_exclusion(self.exclure)
        # (b) recherche dans le TEXTE COURANT : non ancrée, uniquement phrases d'au moins 2 mots (ou acronymes)
        alts_t, pivots = set(), set()
        for p in self.forts + self.moyens:
            toks = _tokens_phrase(p)
            if toks and (len(toks) >= 2 or toks[0] in ACRO_STEMS):
                alts_t.add(_rx_phrase(toks))
                pivots.add(toks[0])
        if alts_t:
            self.rx_texte = re.compile(r"(?<![a-z0-9])(?:" + "|".join(sorted(alts_t, key=len, reverse=True))
                                       + r")(?![a-z0-9])")
        self.pivots = tuple(pivots)

    def apparier(self, variante: str):
        """Poids (0-10) si le libellé de tableau correspond à ce champ, sinon None."""
        if self.rx_excl and self.rx_excl.search(variante):
            return None
        for poids, rx in self.tiers:
            if rx.match(variante):
                return poids
        return None

    def plage_ok(self, v: float) -> bool:
        return self.plage is None or self.plage[0] <= v <= self.plage[1]


CHAMPS: dict = {}


def _c(nom, desc, genre, topic, F, M="", f="", X="", prefixe=False, plage=None, texte=True,
       essentiel=False, prefere_total=False):
    CHAMPS[nom] = Champ(nom, desc, genre, topic, F, M, f, X, prefixe, plage, texte, essentiel, prefere_total)


# ======================= BILAN =======================
_c("total_bilan", "Total de l'actif (= total passif + capitaux propres), le plus grand chiffre du bilan. Tunisie : « Total actifs ».",
   "montant", "bilan", essentiel=True,
   F="total actif; total des actifs; total de l'actif; total actifs; total du bilan; total bilan; total general de l'actif; "
     "total general; total assets; total asset; balance sheet total; total balance sheet; total of assets; total actif net; "
     "total des actifs consolides; total actifs consolides",
   M="actif total; bilan total; total des emplois; total emplois; assets total; total net assets; total des actifs nets",
   X="ponder; moyen; hors bilan; off balance; financ; courant; immobilis; gere; gestion; evolution; variation; "
     "croissance; growth; average; weighted; progression; ratio; taux")
_c("total_passif_cp", "Total du passif ET des capitaux propres (sert uniquement à vérifier l'équilibre du bilan).",
   "montant", "bilan", texte=False,
   F="total passif et capitaux propres; total passifs et capitaux propres; total capitaux propres et passifs; "
     "total des passifs et des capitaux propres; total des capitaux propres et des passifs; total passif et fonds propres; "
     "total du passif et des capitaux propres; total equity and liabilities; total liabilities and equity; "
     "total liabilities and shareholders equity; total liabilities and stockholders equity; "
     "total liabilities equity and minority interests; total liabilities and capital",
   X="ponder; moyen; hors bilan; ratio")
_c("capitaux_propres", "Total des capitaux propres (ligne de total, en bas du passif). PAS le capital social seul, PAS « avant résultat de l'exercice ».",
   "montant", "bilan", essentiel=True,
   F="total capitaux propres; total des capitaux propres; capitaux propres; total equity; total shareholders equity; "
     "total stockholders equity; shareholders equity; stockholders equity; equity attributable to owners of the parent; "
     "equity attributable to equity holders of the bank; equity attributable to shareholders of the bank; "
     "total equity attributable to owners of the parent; capitaux propres part du groupe; total capitaux propres part du groupe; "
     "capitaux propres attribuables aux actionnaires; capitaux propres revenant aux actionnaires de la banque; "
     "capitaux propres consolides; capitaux propres apres resultat de l'exercice; total capitaux propres apres resultat",
   M="total fonds propres; total des fonds propres; fonds propres comptables; equity; net equity; shareholders funds; "
     "total shareholders funds; fonds propres consolides; capitaux propres apres affectation; "
     "capitaux propres de l'ensemble consolide",
   f="situation nette; net worth",
   X="avant; hors resultat; passif; dette; liabilit; minoritair; minorit; non controlling; ponder; reglementair; regulatory; "
     "tier; cet1; ratio; rentabilit; return; taux; moyen; average; evolution; variation; croissance; growth; "
     "per share; par action; et passif; and liabilit")
_c("capital_social", "Capital social (nominal souscrit et libéré).",
   "montant", "bilan", texte=False,
   F="capital social; capital; share capital; capital stock; capital souscrit; capital libere; capital emis; issued capital; "
     "paid up capital; capital social souscrit et libere; capital appele",
   X="ratio; risque; variation; adequacy; adequation; ponder; propre")
_c("reserves", "Réserves (consolidées ou non).",
   "montant", "bilan", texte=False,
   F="reserves; total reserves; reserves consolidees; autres reserves; reserves et primes; reserves legales; other reserves; "
     "reserves consolidees part du groupe; total reserves and retained earnings",
   X="ponder; provision; agios; creance; risque; reglementair; regulatory")
_c("resultats_reportes", "Résultats reportés / report à nouveau.",
   "montant", "bilan", texte=False,
   F="resultats reportes; resultat reporte; report a nouveau; reports a nouveau; retained earnings; resultats non distribues; "
     "resultats cumules; resultats reportes consolides; benefices non distribues",
   X="avant; dividende; dividend")
_c("credits_clientele", "Créances / crédits sur la clientèle au bilan (loans and advances to customers). Tunisie : « Créances sur la clientèle » (actif).",
   "montant", "bilan", essentiel=True,
   F="creances sur la clientele; creances nettes sur la clientele; creances a la clientele; creances clientele; "
     "creances sur clientele; credits a la clientele; credits clientele; credits sur la clientele; prets a la clientele; "
     "prets et creances a la clientele; prets et avances a la clientele; prets et creances sur la clientele; "
     "creances sur la clientele nettes; encours de credits; encours des credits; loans and advances to customers; "
     "net loans and advances to customers; loans to customers; net loans to customers; customer loans; net customer loans; "
     "loans and receivables to customers; loans and receivables from customers; loans and advances to customers net; "
     "creances sur les clients; creances nettes sur les clients",
   M="loans and advances; net loans; total loans; total net loans; total des credits; credits nets; prets nets; "
     "portefeuille de credits; encours credit clientele; encours clientele; loans to clients; client loans",
   X="etablissement; banque; bancair; financ; hors bilan; brut; provision; depreciation; interet; agios; moyen; variation; "
     "bank; credit institution; central bank; evolution; growth; croissance; average; impaired; non performing; douteu; "
     "classe; ratio; taux; rate; revenu; produit; income; commission; gross; charge")
_c("depots_clientele", "Dépôts et avoirs de la clientèle au passif (customer deposits). Tunisie : « Dépôts et avoirs de la clientèle ».",
   "montant", "bilan", essentiel=True,
   F="depots et avoirs de la clientele; depots de la clientele; depots clientele; depots clients; depots des clients; "
     "avoirs de la clientele; comptes crediteurs de la clientele; dettes envers la clientele; dettes envers les clients; "
     "depots et comptes crediteurs de la clientele; customer deposits; deposits from customers; amounts due to customers; "
     "due to customers; deposits and customer accounts; customer accounts; deposits from clients; client deposits; "
     "deposits and amounts due to customers; customers deposits; total deposits",
   M="ressources de la clientele; collecte clientele; collecte des depots; encours des depots; encours de depots; "
     "depots collectes; total des depots; ressources collectees; deposit base; deposits; depots de la clientele et assimiles",
   f="depots",
   X="etablissement; banque; bank; financ; moyen; variation; charge; interet; cout; cost; expense; growth; croissance; "
     "evolution; part de marche; market share; taux; ratio; rate; centrale; central bank; interbancair; interbank")
_c("creances_etablissements", "Créances sur les établissements bancaires et financiers (actif).",
   "montant", "bilan",
   F="creances sur les etablissements bancaires et financiers; creances sur les etablissements de credit; creances sur les banques; "
     "creances aupres des etablissements bancaires et financiers; prets et avances aux etablissements de credit; "
     "loans and advances to banks; due from banks; due from banks and financial institutions; "
     "amounts due from credit institutions; interbank loans; loans to banks; creances sur les etablissements financiers; "
     "avoirs aupres des banques; creances sur les etablissements bancaires",
   X="provision; depreciation; interet; variation; charge; taux; produit")
_c("depots_etablissements", "Dépôts et avoirs des établissements bancaires et financiers (passif).",
   "montant", "bilan",
   F="depots et avoirs des etablissements bancaires et financiers; depots des etablissements bancaires et financiers; "
     "dettes envers les etablissements de credit; dettes envers les banques; deposits from banks; due to banks; "
     "due to banks and financial institutions; amounts due to credit institutions; interbank deposits; "
     "deposits from credit institutions; depots des banques; depots et avoirs des etablissements bancaires",
   X="clientele; customer; interet; charge; cout; variation")
_c("caisse_bct", "Caisse et avoirs auprès de la BCT / banque centrale (liquidités).",
   "montant", "bilan",
   F="caisse et avoirs aupres de la bct ccp et tgt; caisse et avoirs aupres de la bct; caisse et avoirs aupres de la banque centrale; "
     "caisse et avoirs aupres de la bct et ccp; caisse bct ccp et tgt; cash and balances with central banks; "
     "cash and balances with the central bank; cash and due from central banks; caisse et avoirs aupres de la bct ccp",
   M="cash and cash equivalents; liquid assets; liquidites et avoirs; caisse et banque centrale",
   X="interet; variation")
_c("portefeuille_commercial", "Portefeuille-titres commercial / de transaction (actif).",
   "montant", "bilan",
   F="portefeuille titres commercial; portefeuille-titres commercial; portefeuille titres de transaction; titres de transaction; "
     "trading portfolio; trading assets; financial assets held for trading; financial assets at fair value through profit or loss; "
     "titres de placement; titres detenus a des fins de transaction; actifs financiers detenus a des fins de transaction",
   X="gain; revenu; produit; interet; charge; variation; plus value; resultat")
_c("portefeuille_investissement", "Portefeuille-titres d'investissement (actif).",
   "montant", "bilan",
   F="portefeuille titres d'investissement; portefeuille-titres d'investissement; portefeuille d'investissement; "
     "titres d'investissement; investment portfolio; investment securities; investment securities at amortised cost; "
     "titres de participation et d'investissement; actifs financiers detenus jusqu'a l'echeance; held to maturity investments; "
     "investments held to maturity; financial investments; financial assets at fair value through other comprehensive income",
   X="gain; revenu; produit; interet; charge; variation; resultat")
_c("emprunts_ressources_speciales", "Emprunts et ressources spéciales / dettes subordonnées (passif).",
   "montant", "bilan",
   F="emprunts et ressources speciales; ressources speciales; borrowings; borrowed funds; other borrowings; subordinated debt; "
     "emprunts subordonnes; debt securities issued; dettes subordonnees; funds borrowed",
   X="charge; interet; variation; remboursement; cout")

# ======================= COMPTE DE RÉSULTAT =======================
_c("produit_net_bancaire", "Produit net bancaire (PNB) / Net banking income / total net revenues. Ligne « PRODUIT NET BANCAIRE » de l'état de résultat.",
   "montant", "resultat", essentiel=True,
   F="produit net bancaire; total produit net bancaire; pnb; p n b; net banking income; nbi; total net banking income; "
     "net operating income; total operating income; total net revenue; total net revenues; net revenue; net revenues; "
     "total revenue; total revenues; net banking revenue; net banking revenues; produit net bancaire consolide",
   M="produits d'exploitation bancaire nets; revenu net bancaire; produit bancaire net; net operating revenue; "
     "total income net of interest expense; net interest and other income; total operating revenues; "
     "income from banking operations; revenus nets d'exploitation; net income from banking operations; "
     "operating income before provisions; operating income before impairment",
   f="produit bancaire",
   X="avant impot; hors; par action; variation; evolution; croissance; growth; progression; ratio; taux; marge nette; "
     "per share; moyen; coefficient")
_c("resultat_net", "Résultat net de l'exercice (compte de résultat). Si comptes consolidés : part du groupe acceptée.",
   "montant", "resultat", essentiel=True,
   F="resultat net; resultat net de l'exercice; resultat net de la periode; resultat net de l'annee; resultat net consolide; "
     "resultat net part du groupe; resultat net part groupe; resultat net de l'exercice part du groupe; resultat de l'exercice; "
     "resultat de la periode; resultat de l'exercice apres modifications comptables; resultat net apres modifications comptables; "
     "resultat net apres impot; benefice net; benefice net de l'exercice; benefice de l'exercice; benefice de la periode; "
     "net income; net profit; net income for the year; net profit for the year; net income for the period; "
     "profit for the year; profit for the period; profit after tax; net profit after tax; net result; net earnings; "
     "profit attributable to owners of the parent; profit attributable to equity holders; "
     "net income attributable to shareholders; net income group share; resultat net revenant aux actionnaires; "
     "resultat net attribuable aux proprietaires; resultat net attribuable aux actionnaires de la banque",
   M="resultat apres impot; resultat net apres impots; resultat de l'exercice net d'impot; profit; benefice; "
     "income for the year; resultat comptable; benefice net apres impot; resultat net comptable; "
     "profit attributable to shareholders; resultat net en part du groupe",
   X="avant; hors; minoritair; minority; non controlling; interets minoritaires; par action; per share; ordinaire; growth; "
     "croissance; evolution; variation; marge; margin; ratio; taux; dividende; dividend; affect; reporte; retained; "
     "distribuable; before; operating")
_c("resultat_avant_impot", "Résultat avant impôt.",
   "montant", "resultat",
   F="resultat avant impot; resultat avant impots; resultat courant avant impot; resultat des activites ordinaires avant impot; "
     "resultat avant impot sur les benefices; resultat avant impots sur les societes; profit before tax; income before tax; "
     "income before income taxes; profit before income tax; earnings before tax; pre tax profit; pre tax income; "
     "profit before taxation; pbt; resultat avant impot et interets minoritaires",
   X="par action; per share; variation; ratio")
_c("impot_benefices", "Impôt sur les bénéfices / charge d'impôt de l'exercice.",
   "montant", "resultat",
   F="impot sur les benefices; impots sur les benefices; impot sur les societes; impots sur les societes; impot sur le resultat; "
     "charge d'impot; charge d'impot sur le resultat; impot sur le benefice; income tax; income tax expense; income taxes; "
     "tax expense; taxation; corporate tax; provision for income taxes; impots et taxes sur les benefices",
   X="avant; apres; differe; deferred; ratio; taux; rate; variation")
_c("resultat_exploitation", "Résultat d'exploitation (solde après PNB, charges et dotations).",
   "montant", "resultat",
   F="resultat d'exploitation; operating profit; operating income; operating result; profit from operations; "
     "income from operations; resultat d'exploitation consolide",
   M="resultat brut d'exploitation; rbe; gross operating income; operating profit before provisions; "
     "resultat d'exploitation avant provisions",
   X="par action; ratio; taux; variation; marge; margin; avant impot")
_c("charges_exploitation", "Total des charges d'exploitation NON bancaires / charges opératoires / frais généraux (personnel + charges générales + amortissements). Valeur positive. Ce n'est PAS « charges d'exploitation bancaire » (intérêts encourus). null si aucun total n'est imprimé.",
   "montant", "resultat", essentiel=True,
   F="total charges d'exploitation; total des charges d'exploitation; charges operatoires; total charges operatoires; "
     "charges d'exploitation non bancaires; total des frais generaux; frais generaux d'exploitation; frais generaux; "
     "total frais generaux; operating expenses; total operating expenses; non interest expenses; total non interest expenses; "
     "non interest expense; operating costs; total operating costs; administrative expenses and depreciation; "
     "total charges operationnelles; charges operationnelles",
   M="charges d'exploitation; charges de fonctionnement; frais de fonctionnement; total charges de fonctionnement; "
     "charges d'exploitation hors charges bancaires; total opex; opex; cost base; operating charges; "
     "total des charges operationnelles; total costs",
   f="total des charges",
   X="interet; commission; provision pour; variation; ratio; coefficient; taux; croissance; growth; evolution; "
     "per share; moyen")
_c("frais_personnel", "Frais de personnel / charges de personnel de l'exercice (compte de résultat).",
   "montant", "resultat",
   F="frais de personnel; charges de personnel; frais du personnel; charges du personnel; total frais de personnel; "
     "masse salariale; remunerations du personnel; salaires et charges sociales; salaires et avantages; personnel expenses; "
     "staff costs; staff expenses; employee benefits; employee benefit expenses; employee benefits expense; "
     "employee compensation; compensation and benefits; salaries and employee benefits; salaries and benefits; "
     "personnel costs; payroll costs; wages and salaries; remuneration and employee benefits",
   X="avant; variation; ratio; taux; nombre; effectif; headcount; number of; moyen; average; par employe; per employee; "
     "evolution; croissance; growth; part; poids")
_c("charges_generales", "Charges générales d'exploitation (hors personnel) de l'exercice (compte de résultat).",
   "montant", "resultat",
   F="charges generales d'exploitation; autres charges d'exploitation; frais administratifs; autres frais administratifs; "
     "charges generales; frais generaux administratifs; frais d'administration; charges administratives; "
     "general and administrative expenses; general administrative expenses; administrative expenses; "
     "general operating expenses; other operating expenses; other administrative expenses; other general expenses; "
     "operating and administrative expenses; other expenses; autres charges generales",
   M="autres charges; charges externes; achats et charges externes; external expenses; services exterieurs",
   X="avant; variation; ratio; taux; evolution; croissance; growth; per share; interet; commission; bancair")
_c("dotations_amortissements", "Dotations aux amortissements et aux provisions sur immobilisations de l'exercice (compte de résultat).",
   "montant", "resultat",
   F="dotations aux amortissements; dotations aux amortissements et aux provisions sur immobilisations; "
     "dotations aux amortissements et provisions sur immobilisations; dotations aux amortissements des immobilisations; "
     "dotations aux amortissements sur immobilisations; dotations aux amortissements et aux depreciations; "
     "dotation aux amortissements; amortissements; depreciation and amortization; depreciation and amortisation; "
     "depreciation amortisation and impairment; depreciation and amortization of fixed assets; "
     "depreciation of property and equipment; amortization; depreciation; depreciation expense; "
     "dotations aux amortissements des immobilisations incorporelles et corporelles; "
     "dotations aux amortissements et depreciations des immobilisations corporelles et incorporelles",
   X="provision pour risque; creance; credit; reprise; variation; cumul; valeur; brute; ratio; taux")
_c("interets_produits", "Intérêts et revenus assimilés (produits d'intérêts).",
   "montant", "resultat",
   F="interets et revenus assimiles; interets et produits assimiles; interets et revenus similaires; "
     "interets et produits similaires; produits d'interets; revenus d'interets; produits d'interets et assimiles; "
     "total produits d'interets; interest income; interest and similar income; interest and related income; "
     "total interest income; interest and other similar income; interest and dividend income",
   M="revenus assimiles; total interets et revenus assimiles; interest revenue",
   X="net; marge; charge; expense; encouru; variation; croissance; ratio; taux")
_c("interets_charges", "Intérêts encourus et charges assimilées (charges d'intérêts).",
   "montant", "resultat",
   F="interets encourus et charges assimilees; interets encourus; charges d'interets; charges d'interets et assimilees; "
     "interets et charges assimilees; interets et charges similaires; total interets encourus et charges assimilees; "
     "interest expense; interest and similar expense; interest and similar expenses; interest and related expense; "
     "total interest expense; interest charges",
   X="net; marge; variation; croissance; ratio; taux; income; produit")
_c("marge_interets", "Marge d'intérêts (intérêts perçus – intérêts encourus) / net interest income.",
   "montant", "resultat",
   F="marge d'interet; marge d'interets; marge nette d'interets; marge d'interet nette; marge nette d'interet; "
     "net interest income; net interest revenue; net interest and similar income; interest margin; produit net d'interets; "
     "revenu net d'interets; resultat d'interets; net interest earnings; marge d'intermediation",
   X="taux; ratio; moyen; sur actif; on assets; after provision; apres provision")
_c("commissions", "Commissions (en produits) / commissions nettes / fee and commission income.",
   "montant", "resultat",
   F="commissions en produits; commissions percues; commissions; produits de commissions; commissions nettes; "
     "fee and commission income; commission income; fees and commissions income; net fee and commission income; "
     "net commissions; commission and fee income; fees and commissions; produits des commissions",
   X="encouru; charge; expense; variation; ratio; taux; croissance; growth")
_c("gains_portefeuille_commercial", "Gains sur portefeuille-titres commercial et opérations financières (résultat de marché / change).",
   "montant", "resultat",
   F="gains sur portefeuille titres commercial et operations financieres; gains sur portefeuille titres commercial; "
     "gains sur portefeuille commercial; gains nets sur portefeuille titres commercial et operations financieres; "
     "trading income; net trading income; net gains on financial instruments at fair value; "
     "net gain from financial transactions; foreign exchange income; gains sur operations de change",
   X="variation; ratio; taux")
_c("revenus_portefeuille_investissement", "Revenus du portefeuille d'investissement.",
   "montant", "resultat",
   F="revenus du portefeuille d'investissement; revenus du portefeuille titres d'investissement; "
     "revenus du portefeuille-titres d'investissement; investment income; income from investment securities; "
     "dividend income; revenus des titres d'investissement",
   X="variation; ratio; taux")
_c("cout_du_risque", "Coût du risque de l'EXERCICE (compte de résultat). Tunisie : « Dotations aux provisions et résultat des corrections de valeurs sur créances, hors bilan et passif ». Ce n'est PAS le stock de provisions.",
   "montant", "resultat", essentiel=True, prefixe=True,
   F="cout du risque; cout du risque net; cout net du risque; cout du risque de credit; cout du risque de l'exercice; cost of risk; "
     "net cost of risk; credit cost; "
     "dotations aux provisions et resultat des corrections de valeurs sur creances hors bilan et passif; "
     "dotation aux provisions et resultat des corrections de valeurs sur creances hors bilan et passif; "
     "dotations aux provisions et resultat des corrections de valeurs sur creances; "
     "dotation aux provisions et resultat des corrections de valeurs sur creances; "
     "dotations nettes aux provisions sur creances; dotations nettes aux provisions pour risques de credit; "
     "dotations nettes aux provisions pour creances douteuses; charges nettes pour risque de credit; "
     "charge nette du risque de credit; provision for credit losses; provision for loan losses; provisions for credit losses; "
     "net impairment loss on financial assets; net impairment losses on loans and advances; "
     "impairment losses on loans and advances; impairment charges on loans and advances; impairment charge for credit losses; "
     "expected credit losses; expected credit loss charge; credit impairment charges; net credit impairment; "
     "loan impairment charges",
   M="dotations aux provisions sur creances; dotations aux provisions pour risques de credit; "
     "dotations aux provisions pour risque de contrepartie; charge de depreciation des creances; pertes de valeur sur creances; "
     "risque de credit; loan loss provisions; provisions pour creances douteuses; impairment losses; net impairment charge",
   X="reprise; taux; ratio; autres elements; autres actifs; immobilis; bps; pourcent; stock; cumul; solde des provisions; "
     "encours; en pourcentage; as a percentage; participation; titres; portefeuille; risques et charges")

# ======================= QUALITÉ DES ACTIFS =======================
_c("creances_douteuses_brutes", "MONTANT brut des créances / engagements classés / douteux / NPL (en montant, pas en %).",
   "montant", "risque", prefixe=True, prefere_total=True,
   F="creances classees; creances douteuses; total des creances classees; total creances classees; creances classees brutes; "
     "engagements classes; total des engagements classes; total engagements classes; creances douteuses et litigieuses; "
     "creances compromises; creances en souffrance; creances litigieuses; actifs classes; actifs non performants; "
     "creances non performantes; prets non performants; encours classes; encours des creances classees; encours douteux; "
     "engagements douteux; creances contentieuses; creances irrecouvrables; creances classees brutes avant provisions; "
     "non performing loans; non performing loans npl; non performing assets; non performing exposures; "
     "gross non performing loans; total non performing loans; total npl; npls; npl; nonperforming loans; impaired loans; "
     "gross impaired loans; impaired loans and advances; impaired assets; stage 3 loans; loans in stage 3; stage 3 exposures; "
     "credit impaired loans; credit impaired assets; defaulted loans; defaulted exposures; doubtful loans; doubtful debts; "
     "doubtful receivables; bad loans; problem loans",
   M="creances impayees; creances en retard; impayes; creances sensibles; creances classees b2 b3 et b4; "
     "creances classees b2 b3 b4; classes b2 b3 et b4; creances de classes 2 3 et 4; actifs de classe 2 3 et 4; "
     "past due loans; loans past due; non performing",
   X="taux; ratio; couverture; coverage; pourcent; par action; provision; agios; garanti; reserv; couvert; dotation; reprise; "
     "variation; evolution; nouveau; repartition; part de; share of; proportion; percentage; en pourcentage; croissance; "
     "growth; cost; cout; hors bilan")
_c("provisions_creances", "STOCK total des provisions / dépréciations sur créances ou engagements classés (note risque). Pas le coût du risque annuel.",
   "montant", "risque", prefixe=True, prefere_total=True,
   F="provisions sur creances; provisions pour creances; provisions sur creances classees; provisions sur creances douteuses; "
     "provisions individuelles; total provisions; total des provisions; provisions et agios reserves; "
     "provisions et agios reserves sur creances; provisions et agios reserves affectes aux creances classees; "
     "provisions sur engagements; provisions pour pertes sur creances; provisions pour depreciation des creances; "
     "provisions pour risques de credit; depreciation des creances; depreciations sur creances; depreciations individuelles; "
     "stock de provisions; stock des provisions; provisions constituees; provisions constituees sur creances classees; "
     "provisions affectees aux creances classees; provisions affectees; "
     "loan loss allowance; loan loss allowances; loan loss reserve; loan loss reserves; allowance for loan losses; "
     "allowance for credit losses; allowance for impairment; allowances for impairment losses; allowances for loan losses; "
     "reserve for loan losses; total allowances; total allowance for credit losses; impairment allowance; "
     "impairment allowances; loss allowance; loss allowances; total loan loss provisions",
   M="loan loss provisions; provisions for impairment; provisions on loans; provisions for loan losses; "
     "provisions pour depreciation; depreciations; provisions sur la clientele",
   f="provisions",
   X="dotation; reprise; taux; ratio; couverture; variation; charge; cout; net des; coverage; expense; collectiv; "
     "risques et charges; participation; titres; portefeuille; immobilis")
_c("provisions_collectives", "Provisions collectives / additionnelles / générales (stock).",
   "montant", "risque", prefixe=True,
   F="provisions collectives; provision collective; provisions collectives sur creances; provisions collectives sur engagements; "
     "provisions additionnelles; provisions a caractere general; provisions generales; collective provisions; "
     "collective allowance; collective impairment; general provisions; general allowance; "
     "stage 1 and stage 2 allowance; depreciations collectives",
   X="dotation; reprise; taux; ratio; variation; charge")
_c("agios_reserves", "Stock des agios réservés (intérêts réservés sur créances classées), s'il est présenté séparément.",
   "montant", "risque", prefixe=True,
   F="agios reserves; agios reserves sur creances; agios reserves sur creances classees; agios reserves sur engagements; "
     "agios reserves sur clientele; total des agios reserves; interets reserves; interets reserves sur creances classees; "
     "interets en suspens; interets suspendus; suspended interest; interest in suspense",
   M="produits reserves; revenus reserves; agios et interets reserves",
   X="dotation; reprise; variation; provision et")
_c("engagements_bruts", "Total des engagements bruts (bilan + hors bilan) ou encours brut des crédits : dénominateur du taux de créances classées.",
   "montant", "risque", prefixe=True, prefere_total=True,
   F="total des engagements; total engagements; engagements bruts; total des engagements bilan et hors bilan; "
     "total des engagements de la clientele; engagements de la clientele; encours total des engagements; "
     "total exposures; gross loans; gross loans and advances; gross loans and advances to customers; total gross loans; "
     "encours brut des credits; creances brutes sur la clientele; encours brut; total des engagements bruts",
   X="classe; douteu; non perform; impaired; provision; agios; taux; ratio; variation; croissance; growth; hors bilan seul")

# ======================= RATIOS PUBLIÉS =======================
_c("taux_npl_rapporte", "Taux de créances/engagements classés / NPL ratio publié, en % (ex. 3.6).",
   "pct", "risque", prefixe=True, plage=(0.1, 80),
   F="taux des creances classees; taux de creances classees; taux des creances douteuses; taux de creances douteuses; "
     "taux des engagements classes; taux d'engagements classes; taux des actifs classes; taux des actifs non performants; "
     "taux d'actifs classes; taux des creances non performantes; taux de creances non performantes; taux des creances impayees; "
     "taux d'impayes; taux des impayes; ratio des creances classees; ratio de creances classees; ratio des creances douteuses; "
     "ratio des engagements classes; ratio d'actifs classes; part des creances classees; part des engagements classes; "
     "part des creances douteuses; poids des creances classees; proportion des creances classees; "
     "creances classees en pourcentage des engagements; creances classees sur total engagements; "
     "creances classees rapportees aux engagements; npl ratio; npl rate; non performing loans ratio; "
     "non performing loans to total loans; non performing loans as a percentage of total loans; "
     "non performing loans as percentage of gross loans; non performing loan ratio; nonperforming loans ratio; "
     "npl to gross loans; npl to total loans; gross npl ratio; impaired loans ratio; impaired loans to gross loans; "
     "impaired loans as a percentage of gross loans; bad loans ratio; bad debt ratio; problem loans ratio; stage 3 ratio; "
     "stage 3 loans ratio; stage 3 loans as percentage of gross loans; ratio of non performing loans; ratio of impaired loans; "
     "doubtful loans ratio; taux de creances douteuses et litigieuses; taux des engagements douteux; "
     "taux de creances en souffrance; taux de creances compromises; taux de contentieux; taux de creances irrecouvrables",
   M="creances classees representent; engagements classes representent; creances classees ressortent; "
     "creances douteuses representent; npls represent; non performing loans represent; non performing loans stood at; "
     "creances classees en pourcentage du total des engagements; ratio npl",
   X="couverture; coverage; provision")
_c("taux_couverture_rapporte", "Taux de couverture des créances classées publié, en % (ex. 75).",
   "pct", "risque", prefixe=True, plage=(5, 300),
   F="taux de couverture; taux de couverture des creances classees; taux de couverture des creances classees par les provisions; "
     "taux de couverture des creances classees par les provisions et agios reserves; taux de couverture des creances douteuses; "
     "taux de couverture des engagements classes; taux de couverture des actifs classes; "
     "taux de couverture des actifs non performants; taux de couverture globale; taux de couverture des npl; "
     "taux de provisionnement; taux de provisionnement des creances classees; ratio de couverture; "
     "ratio de couverture des creances classees; ratio de couverture des creances douteuses; ratio de provisionnement; "
     "couverture des creances classees; couverture des creances douteuses; couverture des engagements classes; "
     "couverture des npl; taux de couverture par les provisions; taux de couverture hors agios; coverage ratio; "
     "npl coverage; npl coverage ratio; non performing loans coverage; non performing loan coverage ratio; "
     "provision coverage; provision coverage ratio; provisioning coverage; provisioning ratio; loan loss coverage; "
     "loan loss coverage ratio; loan loss reserve ratio; loan loss reserves to non performing loans; allowance coverage; "
     "allowance for loan losses to non performing loans; reserves to npl; provisions to npl; "
     "provisions to non performing loans; impaired loans coverage ratio; stage 3 coverage; stage 3 coverage ratio; "
     "coverage of non performing loans; coverage of impaired loans; bad debt coverage",
   X="emploi; ressource; liquid; lcr; depots; import; export; cout; charge; interet; marge; change; dividende; sociale; "
     "risque de; besoin")
_c("roe_rapporte", "ROE / rentabilité des capitaux propres publiée, en %.",
   "pct", "ratios", prefixe=True, plage=(-100, 80),
   F="roe; return on equity; return on average equity; return on shareholders equity; return on average shareholders equity; "
     "return on common equity; return on equity roe; rentabilite des capitaux propres; rentabilite des fonds propres; "
     "rentabilite financiere; rendement des capitaux propres; rendement des fonds propres; "
     "rentabilite des capitaux propres moyens; taux de rentabilite des capitaux propres; "
     "ratio de rentabilite des capitaux propres; retour sur capitaux propres; retour sur fonds propres; "
     "rendement sur fonds propres; rentabilite des capitaux propres roe; resultat net sur capitaux propres moyens; "
     "resultat net sur capitaux propres; resultat net rapporte aux capitaux propres",
   M="net income to equity; return on net worth; rentabilite des capitaux propres apres impot; roe net; roe consolide; "
     "roe de la banque",
   X="cible; objectif; target; ajuste; adjusted; tangible; underlying; attendu; budget; guidance; previsionnel; minimum; "
     "rotce; avant impot; before tax")
_c("roa_rapporte", "ROA / rentabilité des actifs publiée, en %.",
   "pct", "ratios", prefixe=True, plage=(-10, 10),
   F="roa; roaa; return on assets; return on average assets; return on total assets; rentabilite des actifs; "
     "rentabilite des actifs moyens; rentabilite economique; rendement des actifs; rendement de l'actif; "
     "rentabilite de l'actif; taux de rentabilite des actifs; retour sur actifs; retour sur actif; "
     "resultat net sur total actif; resultat net rapporte au total actif; return on assets roa; rentabilite des actifs roa",
   M="roa net; roa consolide; roa de la banque; return on average total assets",
   X="cible; objectif; target; ajuste; adjusted; attendu; budget; guidance; previsionnel; minimum")
_c("coeff_exploitation_rapporte", "Coefficient d'exploitation / cost-to-income publié, en %.",
   "pct", "ratios", prefixe=True, plage=(10, 250),
   F="coefficient d'exploitation; coeff d'exploitation; coefficient d'exploitation du groupe; ratio d'exploitation; "
     "ratio d'efficience; coefficient d'efficience; efficience operationnelle; ratio d'efficacite; ratio charges produits; "
     "ratio charges sur produits; charges d'exploitation sur pnb; charges sur pnb; frais generaux sur pnb; "
     "charges d'exploitation sur produit net bancaire; charges d'exploitation rapportees au pnb; cost to income; "
     "cost to income ratio; cost income ratio; cost income; cir; efficiency ratio; operating efficiency ratio; "
     "cost to income ratio cir; ratio de cout; ratio cout revenu; overhead ratio; cost efficiency ratio; "
     "expense to revenue ratio",
   X="cible; objectif; target; budget; guidance; previsionnel; ajuste; adjusted")
_c("ratio_solvabilite_rapporte", "Ratio de solvabilité global (fonds propres nets / actifs pondérés) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(2, 60),
   F="ratio de solvabilite; ratio de solvabilite global; ratio de solvabilite globale; ratio global de solvabilite; "
     "taux de solvabilite; ratio de solvabilite total; ratio de fonds propres totaux; ratio de fonds propres; "
     "ratio de fonds propres global; ratio de capital total; ratio de capital global; ratio de solvabilite reglementaire; "
     "ratio de solvabilite de la banque; fonds propres nets sur risques encourus; fonds propres nets sur risques ponderes; "
     "fonds propres nets rapportes aux risques encourus; ratio de couverture des risques par les fonds propres; "
     "ratio cooke; ratio mcdonough; ratio bale; ratio bale iii; solvency ratio; total solvency ratio; "
     "overall solvency ratio; capital adequacy ratio; capital adequacy ratio car; car; total capital ratio; "
     "total capital adequacy ratio; total capital to risk weighted assets; total capital to risk weighted assets ratio; "
     "capital to risk weighted assets; regulatory capital ratio; risk based capital ratio; total risk based capital ratio; "
     "bis ratio; basel ratio; basel iii ratio; basel iii total capital ratio; own funds ratio; total own funds ratio",
   X="tier; cet1; de base; minimum; exige; requis; seuil; cible; buffer; coussin; levier; leverage; requirement; exigence; "
     "plancher; floor; norme; limite; additional; complementaire")
_c("ratio_tier1_rapporte", "Ratio Tier 1 (fonds propres de base / actifs pondérés) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(2, 60),
   F="ratio tier 1; ratio tier1; tier 1 ratio; tier 1 capital ratio; tier one ratio; ratio de fonds propres de base; "
     "ratio de fonds propres tier 1; ratio de solvabilite tier 1; ratio de solvabilite de base; taux de solvabilite tier 1; "
     "ratio de fonds propres de categorie 1; ratio fonds propres de categorie 1; ratio tier 1 consolide; t1 ratio; "
     "tier 1 capital to risk weighted assets; tier 1 capital adequacy ratio; ratio de fonds propres de niveau 1; "
     "fonds propres de base sur risques ponderes; tier 1 capital to rwa; tier 1 to risk weighted assets; "
     "ratio de fonds propres nets de base",
   X="minimum; exige; requis; seuil; cible; target; buffer; coussin; levier; leverage; requirement; exigence; additional; "
     "cet1; common equity; core tier; commun; dur")
_c("ratio_cet1_rapporte", "Ratio CET1 (Common Equity Tier 1) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(2, 60),
   F="cet1; cet 1; cet1 ratio; ratio cet1; common equity tier 1; common equity tier 1 ratio; common equity tier one; "
     "ratio de fonds propres de base de categorie 1; ratio de fonds propres durs; ratio core tier 1; core tier 1; "
     "core tier 1 ratio; ratio de capital ordinaire de categorie 1; ratio ct1; ratio de fonds propres cet1; "
     "fonds propres cet1 sur actifs ponderes; ratio de fonds propres de base durs",
   X="minimum; exige; requis; seuil; cible; target; buffer; coussin; requirement; exigence")
_c("lcr_rapporte", "LCR (Liquidity Coverage Ratio) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(30, 1500),
   F="lcr; liquidity coverage ratio; liquidity coverage ratio lcr; ratio de liquidite a court terme; "
     "ratio de couverture des besoins de liquidite; ratio de couverture de liquidite; ratio lcr; taux lcr; "
     "ratio de liquidite a court terme lcr; ratio de couverture de liquidite a 30 jours; ratio de liquidite court terme; "
     "lcr ratio; ratio de couverture des sorties nettes de tresorerie; liquidity ratio lcr; ratio de liquidite lcr; "
     "ratio de couverture des besoins de liquidite lcr",
   X="minimum; exige; requis; seuil; cible; target; limite; requirement; exigence")
_c("nsfr_rapporte", "NSFR (Net Stable Funding Ratio) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(30, 600),
   F="nsfr; net stable funding ratio; net stable funding ratio nsfr; ratio de financement stable net; "
     "ratio de liquidite a long terme; ratio nsfr; ratio de financement stable; taux nsfr; ratio structurel de liquidite; "
     "ratio de liquidite structurelle; ratio de liquidite a long terme nsfr; nsfr ratio; ratio de refinancement stable",
   X="minimum; exige; requis; seuil; cible; target; limite; requirement; exigence")
_c("ratio_liquidite_rapporte", "Ratio de liquidité réglementaire (BCT) publié, en % (minimum réglementaire 100 %).",
   "pct", "ratios", prefixe=True, plage=(20, 1500),
   F="ratio de liquidite; ratio de liquidite reglementaire; ratio de liquidite bct; coefficient de liquidite; "
     "ratio de liquidite global; liquidity ratio; regulatory liquidity ratio; liquidity ratio bct; ratio de liquidite bct; "
     "taux de liquidite",
   X="lcr; nsfr; coverage; couverture; minimum; exige; requis; seuil; cible; limite; requirement; court terme; long terme; "
     "short term; long term; structurel")
_c("ldr_rapporte", "Ratio crédits / dépôts (LDR / transformation) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(20, 300),
   F="ratio credits depots; ratio credits sur depots; ratio de transformation; ratio de transformation des depots en credits; "
     "ratio loans to deposits; loans to deposits ratio; loan to deposit ratio; loans to deposits; loan to deposit; ldr; "
     "loan deposit ratio; ratio creances sur la clientele depots clientele; taux d'emploi des depots; "
     "ratio de credits sur depots; ratio de transformation credits depots; ratio ltd; ltd ratio; "
     "loan to deposit ratio ldr; ratio loan to deposit; ratio prets depots",
   X="minimum; maximum; exige; requis; seuil; cible; limite; requirement; plafond")
_c("cout_risque_rapporte", "Coût du risque publié en % des crédits (ou en points de base, converti en %).",
   "pct", "ratios", prefixe=True, plage=(0.01, 20),
   F="taux de cout du risque; cout du risque en pourcentage des credits; cout du risque sur encours; "
     "cout du risque rapporte aux credits; cout du risque en pourcent; cout du risque en pb; "
     "cout du risque en points de base; cost of risk ratio; cost of risk in bps; cost of risk bps; credit cost ratio; "
     "credit loss ratio; loan loss provision ratio; provision charge to average loans; "
     "cost of risk as a percentage of loans; cost of risk bps; ratio cout du risque; cout du risque rapporte aux creances; "
     "cout du risque sur encours moyen; taux du cout du risque; cout du risque sur credits",
   M="cost of risk; cout du risque; cout du risque net",
   X="cible; objectif; target; budget; guidance")
_c("nim_rapporte", "Marge nette d'intérêt (NIM) publiée, en %.",
   "pct", "ratios", prefixe=True, plage=(0.1, 15),
   F="marge nette d'interet; marge d'interet nette; net interest margin; nim; marge nette d'intermediation; "
     "marge d'intermediation; taux de marge nette d'interet; marge d'interet en pourcentage des actifs; "
     "marge nette d'interets sur actifs productifs; net interest margin nim; net interest spread",
   X="cible; objectif; target; budget; guidance")
_c("ratio_levier_rapporte", "Ratio de levier (Bâle III) publié, en %.",
   "pct", "ratios", prefixe=True, plage=(1, 30),
   F="ratio de levier; leverage ratio; basel leverage ratio; ratio de levier bale iii; ratio d'effet de levier; "
     "tier 1 leverage ratio",
   X="minimum; exige; requis; seuil; cible; target; requirement")
_c("dividende_par_action", "Dividende (proposé/distribué) par action, en devise (non converti).",
   "unitaire", "resultat", plage=(0.001, 1000),
   F="dividende par action; dividendes par action; dividende propose par action; dividende net par action; "
     "dividende brut par action; dividend per share; dividends per share; dps; dividende unitaire; "
     "dividende distribue par action; dividende par titre; proposed dividend per share",
   X="taux; rendement; yield; payout; distribution")
_c("resultat_par_action", "Résultat (bénéfice) par action, en devise (non converti).",
   "unitaire", "resultat", plage=(-1000, 1000),
   F="resultat par action; benefice par action; resultat net par action; resultat de base par action; "
     "resultat dilue par action; bpa; earnings per share; eps; basic earnings per share; diluted earnings per share; "
     "basic eps; net income per share; earning per share; resultat par action de base; resultat net de base par action",
   X="croissance; growth; variation; ratio; per sh")
_c("actifs_ponderes", "Actifs / risques encourus pondérés (RWA) : total, dénominateur des ratios de solvabilité.",
   "montant", "ratios", prefere_total=True,
   F="risques encourus ponderes; total des risques encourus ponderes; actifs ponderes des risques; actifs ponderes; "
     "total actifs ponderes; risques ponderes; total risques ponderes; actifs ponderes en fonction des risques; "
     "encours pondere; encours ponderes; risk weighted assets; rwa; total rwa; total risk weighted assets; "
     "risk weighted exposure; risk weighted exposures; total risk weighted exposure; weighted risk assets; "
     "actifs a risque ponderes; total des risques ponderes; expositions ponderees; total des expositions ponderees",
   X="ratio; taux; densite; density; variation; evolution; hors bilan; off balance; croissance; growth")
_c("fonds_propres_nets", "Fonds propres nets réglementaires (Tier 1 + Tier 2), numérateur du ratio de solvabilité global.",
   "montant", "ratios", prefere_total=True,
   F="fonds propres nets; fonds propres nets reglementaires; total des fonds propres nets; fonds propres reglementaires; "
     "fonds propres prudentiels; fonds propres globaux; fonds propres totaux; total fonds propres nets; "
     "fonds propres nets globaux; total regulatory capital; regulatory capital; total capital; total own funds; own funds; "
     "total eligible capital; total capital resources; capital base; total capital base; regulatory own funds; "
     "net own funds; fonds propres eligibles; fonds propres totaux reglementaires",
   X="ratio; taux; tier 1; base; complementaire; additional; supplementary; tier 2; cet1; variation; evolution; minimum; "
     "exige; requis; besoin; requirement; insuffisance; excedent; deduction")
_c("fonds_propres_tier1", "Fonds propres de base (Tier 1 / FPNB), numérateur du ratio Tier 1.",
   "montant", "ratios", prefere_total=True,
   F="fonds propres nets de base; fonds propres de base; fonds propres de base tier 1; total des fonds propres nets de base; "
     "fonds propres tier 1; fonds propres de categorie 1; fonds propres de niveau 1; tier 1 capital; total tier 1 capital; "
     "tier 1; tier 1 capital base; tier one capital; core capital; fonds propres de base tier 1; fonds propres noyau dur; "
     "fonds propres cet1; common equity tier 1 capital; cet1 capital; total cet1 capital; "
     "fonds propres de base de categorie 1; fonds propres durs",
   X="ratio; taux; complementaire; additional; tier 2; supplementary; minimum; exige; requis; requirement; variation; "
     "evolution; deduction; avant")

for _ch in CHAMPS.values():
    _ch.compiler()

# --- Vues dérivées du dictionnaire ---
TOPICS: dict = defaultdict(list)
for _ch in CHAMPS.values():
    TOPICS[_ch.topic].append(_ch.nom)
TOPIC_NOMS = ["bilan", "resultat", "risque", "ratios"]
assert set(TOPICS) <= set(TOPIC_NOMS)
PCT = {n for n, c in CHAMPS.items() if c.genre == "pct"}
ESSENTIELS_IA = [n for n, c in CHAMPS.items() if c.essentiel]
FACTEURS_UNITE = {"unites": 1.0, "milliers": 1e3, "millions": 1e6, "milliards": 1e9}

# Index inversé « mot racinisé -> champs » : avant de tester 55 champs sur un libellé, on ne retient que ceux
# qui partagent au moins un mot avec lui (accélère la lecture des tableaux d'un facteur ~10).
TOKEN_INDEX: dict = defaultdict(set)
for _ch in CHAMPS.values():
    for _p in _ch.forts + _ch.moyens + _ch.faibles:
        for _t in _tokens_phrase(_p):
            if _t != "total":
                TOKEN_INDEX[_t].add(_ch.nom)
_VIDE: set = set()

# Un ratio présenté dans un tableau sans signe « % » doit au moins avoir un libellé de ratio.
_RE_LABEL_RATIO = re.compile(r"(?:^| )(?:" + "|".join(racine(w) for w in (
    "taux ratio rate part poids proportion pourcentage roe roa roaa npl nsfr lcr cet1 tier solvabilite couverture coverage "
    "rentabilite return margin marge coefficient efficience liquidite levier leverage cost cout rendement").split()) + r")\w*")

_RE_ANCRES_PAGE = re.compile(
    r"total|resultat|benefice|profit|bilan|balance sheet|clientel|customer|pnb|produit|banking|npl|creanc|engagement|"
    r"couvertur|coverage|risqu|risk|cout|cost|solvab|\broe\b|\broa\b|exploitation|impairment|provision|agio|tier|"
    r"equity|capital|liabilit|asset|actif|depot|deposit|loan|personnel|staff|frais|charg|dotation|amortis|"
    r"allowance|class|douteu|performing|coeff|efficien|tal|income|revenue|operating|liquid|lcr|nsfr")
_RE_DENSITE = re.compile(r"\d[\d .,]{2,}\d")
_RE_SOMMAIRE = re.compile(r"sommair|table (?:de|des) matier|table of content|\.{6,}")
_RE_CHIFFRE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# 4. Lecture du PDF : lignes reconstruites, pages scannées (OCR / Vision), unités, périmètre
# ---------------------------------------------------------------------------
def scores_page(plat: str, nb_nombres: int) -> tuple:
    """Un score par thème = 2 × nombre de postes financiers DISTINCTS présents sur la page (+ densité numérique)."""
    zeros = (0.0,) * len(TOPIC_NOMS)
    if len(plat) < 200 or not _RE_ANCRES_PAGE.search(plat):
        return zeros
    touches = set()
    for nom, ch in CHAMPS.items():
        if ch.rx_texte is not None and any(p in plat for p in ch.pivots) and ch.rx_texte.search(plat):
            touches.add(nom)
    if not touches:
        return zeros
    bonus = min(nb_nombres / 40, 3) - (6 if _RE_SOMMAIRE.search(plat) else 0)
    sortie = []
    for topic in TOPIC_NOMS:
        k = sum(1 for c in TOPICS.get(topic, ()) if c in touches)
        sortie.append(max(2 * k + bonus, 0.0) if k else 0.0)
    return tuple(sortie)


def _pages_cibles(scores: list, n: int) -> list:
    cibles: set = set()
    for t in range(len(TOPIC_NOMS)):
        classement = sorted(((s[t], i) for i, s in enumerate(scores) if s[t] >= SCORE_PAGE_MIN), reverse=True)
        top = [i for _, i in classement[:PAGES_PAR_TOPIC]]
        cibles.update(top)
        for i in top[:3]:                               # un état financier s'étale souvent sur 2 pages
            cibles.update(v for v in (i - 1, i + 1) if 0 <= v < n)
    if len(cibles) > MAX_PAGES_LIGNES:                  # on garde les pages au meilleur score global
        cibles = set(sorted(cibles, key=lambda i: -max(scores[i]))[:MAX_PAGES_LIGNES])
    return sorted(cibles)


# Unités : abréviations sensibles à la casse (mDT = milliers, MDT = millions) puis formulations longues
_PRE = r"(?:\ben |\(|[Uu]nit[eé]s?\s*:?\s*(?:en )?)"
_UNITES_BRUT = {
    "milliers": [re.compile(_PRE + r"(?:mDT|KDT|kDT|mD|KD|kD|kTND|KTND|kEUR|k€|K€)(?!\w)")],
    "millions": [re.compile(_PRE + r"(?:MDT|MD|MTND|MEUR|M€|MMAD|MDH|MDZD)(?!\w)")],
}
_UNITES_PLIE = {
    "milliards": [re.compile(p) for p in (r"\ben milliards\b", r"\bin billions\b")],
    "millions": [re.compile(p) for p in (r"\ben millions\b", r"\bin millions\b", r"\(\s*millions? d[e']")],
    "milliers": [re.compile(p) for p in (
        r"\ben milliers\b", r"\bin thousands\b", r"\(\s*milliers? d[e']", r"\bmilliers de (?:dinars?|dt|euros?|dirhams?)\b",
        r"\ben (?:1 ?000|1000) ?(?:dt|d|dinars?|tnd|eur|euros?|mad|dh|dzd|usd)\b",
        r"unite ?:? ?(?:en )?(?:1 ?000 |milliers)")],
}


def detecter_unite(texte: str, texte_plie: str):
    compte: Counter = Counter()
    for unite, regexes in _UNITES_BRUT.items():
        compte[unite] += sum(len(rx.findall(texte)) for rx in regexes)
    for unite, regexes in _UNITES_PLIE.items():
        compte[unite] += sum(len(rx.findall(texte_plie)) for rx in regexes)
    compte = +compte
    return compte.most_common(1)[0][0] if compte else None


def detecter_perimetre(texte: str):
    """'consolide' / 'individuel' d'après le titre de la page (début), à défaut le pied de page."""
    for morceau in (texte[:900], texte[-500:]):
        p = plier(morceau)
        indiv = re.search(r"individuel|etats? financiers? sociaux|comptes sociaux|separate|statutory", p)
        conso = re.search(r"consolid", p)
        if conso and not indiv:
            return "consolide"
        if indiv and not conso:
            return "individuel"
    return None


def detecter_annee(textes: list):
    c: Counter = Counter()
    for t in textes:
        p = plier(t)
        c.update(re.findall(r"(?:31 decembre|31 december|december 31,?)\s*(20\d\d)", p))
        c.update(re.findall(r"(?:exercice|annee|year ended|year)\s*(?:clos le )?(?:au )?(20[12]\d)\b", p))
    return int(max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]) if c else None


def _assembler_lignes(mots: list, tol_y: float = 3.0, ecart_colonne: float = 9.0) -> list:
    """Reconstruit les lignes visuelles à partir de mots positionnés [(x0, y0, x1, y1, texte)] (en points).
    Format de sortie : 'Total actifs | 12 434 225 | 11 000 000' (les colonnes sont séparées par « | »).
    Ce format est IDENTIQUE pour le texte natif, l'OCR et la transcription Vision."""
    if not mots:
        return []
    mots = sorted(mots, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lignes, courante, y_ref = [], [], None
    for w in mots:
        yc = (w[1] + w[3]) / 2
        if y_ref is None or abs(yc - y_ref) <= tol_y:
            if y_ref is None:
                y_ref = yc
            courante.append(w)
        else:
            lignes.append(courante)
            courante, y_ref = [w], yc
    lignes.append(courante)
    sortie = []
    for ligne in lignes:
        ligne.sort(key=lambda w: w[0])
        morceaux, x_prec = [], None
        for w in ligne:
            if x_prec is not None:
                morceaux.append(" | " if w[0] - x_prec > ecart_colonne else " ")
            morceaux.append(w[4])
            x_prec = w[2]
        sortie.append("".join(morceaux))
    return sortie


def lignes_page(page) -> list:
    mots = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
    return _assembler_lignes(mots)


# ---- OCR local (RapidOCR) ----
_OCR_LOCAL = threading.local()
_RE_NOMBRE_OCR = re.compile(r"^[\dOoIl|.,\s()%\-]+$")


def _corriger_ocr(txt: str) -> str:
    """Confusions classiques de l'OCR dans les nombres (O→0, I/l→1) — seulement si le jeton est surtout numérique."""
    if sum(c.isdigit() for c in txt) >= 2 and _RE_NOMBRE_OCR.match(txt):
        return txt.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))
    return txt


def ocr_disponible() -> bool:
    return OCR_CLASSE is not None and np is not None


def _ocr_image(arr, echelle: float):
    """OCR d'une image (ndarray). Retourne ([(x0, y0, x1, y1, texte)] en points PDF, confiance moyenne)."""
    if not hasattr(_OCR_LOCAL, "moteur"):
        _OCR_LOCAL.moteur = OCR_CLASSE()                     # un moteur par thread (sûr et sans verrou)
    sortie = _OCR_LOCAL.moteur(arr)
    elements = []
    if isinstance(sortie, tuple):                            # ancienne API : (résultat, temps)
        elements = [(b, t, s) for b, t, s in (sortie[0] or [])]
    else:                                                    # nouvelle API : objet .boxes / .txts / .scores
        boxes, txts = getattr(sortie, "boxes", None), getattr(sortie, "txts", None)
        scores = getattr(sortie, "scores", None)
        if boxes is not None and txts is not None:
            scores = scores if scores is not None else [1.0] * len(txts)
            elements = list(zip(boxes, txts, scores))
    mots, confs = [], []
    for box, txt, sc in elements:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        mots.append((min(xs) / echelle, min(ys) / echelle, max(xs) / echelle, max(ys) / echelle, _corriger_ocr(str(txt))))
        try:
            confs.append(float(sc))
        except (TypeError, ValueError):
            pass
    return mots, (sum(confs) / len(confs) if confs else 0.0)


def _rendre(page, dpi: int, clip=None):
    pix = page.get_pixmap(dpi=dpi, alpha=False, clip=clip) if clip is not None else page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif pix.n == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def _ocr_pages(doc, taches: list, alertes: list, on_progress=None) -> dict:
    """taches = [(index_page, clip_ou_None)]. OCR en parallèle (le rendu reste dans le thread principal, PyMuPDF
    n'étant pas thread-safe). Mémoire bornée : au plus 2×OCR_WORKERS images en vol. Arrêt au budget de temps."""
    resultats: dict = defaultdict(lambda: {"lignes": [], "confs": []})
    t_fin = time.perf_counter() + OCR_BUDGET_S
    echelle = OCR_DPI / 72.0
    en_cours, faits, abandon = {}, 0, False

    def collecter(termines):
        nonlocal faits
        for f in termines:
            i = en_cours.pop(f)
            try:
                mots, conf = f.result()
                resultats[i]["lignes"].extend(_assembler_lignes(mots))
                resultats[i]["confs"].append(conf)
            except Exception as e:
                alertes.append(f"OCR page {i + 1} échoué : {str(e)[:120]}")
            faits += 1
            if on_progress and faits % 5 == 0:
                on_progress(f"OCR : {faits}/{len(taches)} zones lues...", 10)

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        for i, clip in taches:
            if time.perf_counter() > t_fin:
                abandon = True
                break
            try:
                arr = _rendre(doc[i], OCR_DPI, clip)
            except Exception as e:
                alertes.append(f"Rendu page {i + 1} impossible : {str(e)[:120]}")
                continue
            en_cours[pool.submit(_ocr_image, arr, echelle)] = i
            while len(en_cours) >= 2 * OCR_WORKERS:
                termines, _ = wait(list(en_cours), return_when=FIRST_COMPLETED)
                collecter(termines)
        while en_cours:
            termines, _ = wait(list(en_cours), return_when=FIRST_COMPLETED)
            collecter(termines)
    if abandon:
        alertes.append(f"OCR interrompu après {OCR_BUDGET_S:.0f}s (budget) : certaines pages scannées n'ont pas été lues "
                       "(augmentez OCR_BUDGET_S ou activez la Vision).")
    return resultats


# ---- Gemini Vision : transcription fidèle d'une page image (secours / relecture de l'OCR) ----
PROMPT_VISION = (
    "Transcris fidèlement cette page d'un rapport annuel de banque en texte brut (français ou anglais).\n"
    "- Pour chaque tableau : UNE ligne de texte par ligne du tableau, cellules séparées par « | ».\n"
    "- Recopie les nombres EXACTEMENT comme imprimés (séparateurs de milliers, virgule décimale, parenthèses "
    "pour les négatifs, signe %). Garde les en-têtes de colonnes (années) et la mention d'unité (« en milliers de DT »...).\n"
    "- N'invente rien, ne calcule rien, ne résume pas. Cellule illisible : [?].\n"
    "- Pour les graphiques/infographies : écris chaque étiquette suivie de sa valeur (ex. « ROE | 14,5 % »).\n"
    "Réponds uniquement avec la transcription, sans commentaire ni balise markdown.")


def _vision_page(png: bytes) -> str:
    client = CLIENTS.get("gemini")
    if client is None:
        raise RuntimeError("Gemini indisponible (GEMINI_API_KEY manquante)")
    part = types.Part.from_bytes(data=png, mime_type="image/png")
    base = dict(temperature=0.0, max_output_tokens=8192)
    try:
        cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), **base)
        r = client.models.generate_content(model=MODEL_VISION, contents=[part, PROMPT_VISION], config=cfg)
    except Exception as e:
        if "thinking" not in str(e).lower() and getattr(e, "code", None) != 400:
            raise
        r = client.models.generate_content(model=MODEL_VISION, contents=[part, PROMPT_VISION],
                                           config=types.GenerateContentConfig(**base))
    return r.text or ""


def _nettoyer_transcription(texte: str) -> list:
    """Transforme une sortie Vision (parfois en tableau markdown) en lignes « a | b | c »."""
    lignes = []
    for brut in texte.replace("**", "").split("\n"):
        l = brut.strip()
        if not l or l.startswith("```") or re.fullmatch(r"[\s|:\-]+", l):
            continue
        lignes.append(re.sub(r"\s+", " ", l.strip("|").strip()))
    return lignes


def _nb_nombres(lignes: list) -> int:
    return sum(len(_RE_DENSITE.findall(l)) for l in lignes)


def _vision_pages(doc, indices: list, alertes: list) -> dict:
    """Relit des pages par Gemini Vision (en parallèle). Retourne {index_page: [lignes]}."""
    sorties: dict = {}
    rendus = {}
    for i in indices:
        try:
            pix = doc[i].get_pixmap(dpi=VISION_DPI, alpha=False)
            rendus[i] = pix.tobytes("png")
        except Exception as e:
            alertes.append(f"Rendu Vision page {i + 1} impossible : {str(e)[:100]}")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {i: pool.submit(_vision_page, png) for i, png in rendus.items()}
        for i, f in futs.items():
            try:
                lignes = _nettoyer_transcription(f.result(timeout=LLM_TIMEOUT_S * 2))
                if lignes:
                    sorties[i] = lignes
            except Exception as e:
                alertes.append(f"Vision page {i + 1} échouée : {str(e)[:120]}")
    return sorties


def _classer_page(page, texte: str) -> dict:
    """type : 'texte' (exploitable), 'scan' (image sans texte), 'mixte' (texte + grosses images pauvres en chiffres),
    'vide'. Pour 'mixte' on mémorise les zones d'image à passer à l'OCR."""
    nb = len(texte.strip())
    try:
        infos_img = page.get_image_info()
    except Exception:
        infos_img = []
    pw, ph = page.rect.width or 1, page.rect.height or 1
    zones = []
    for info in infos_img:
        x0, y0, x1, y1 = info["bbox"]
        aire = max(0.0, min(x1, pw) - max(x0, 0.0)) * max(0.0, min(y1, ph) - max(y0, 0.0))
        if aire / (pw * ph) >= 0.12:
            zones.append((max(x0, 0.0), max(y0, 0.0), min(x1, pw), min(y1, ph)))
    if nb < 80:
        return {"type": "scan" if infos_img else "vide", "zones": []}
    if zones and len(_RE_DENSITE.findall(texte)) < 30:
        return {"type": "mixte", "zones": zones[:3]}
    return {"type": "texte", "zones": []}


def lire_pdf(file_bytes: bytes, utiliser_ocr: bool = True, utiliser_vision: bool = True, on_progress=None) -> dict:
    """Lit le PDF et renvoie une « lecture » : lignes de tableaux, texte plat racinisé de TOUTES les pages,
    scores par thème, unités, périmètre, année. Gère les pages scannées (OCR RapidOCR, relecture Gemini Vision)."""
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    alertes: list = []
    try:
        n = min(len(doc), MAX_PAGES_SCAN)
        if n == 0:
            return {"erreur": "PDF vide."}
        textes = [doc[i].get_text("text") for i in range(n)]
        infos = [_classer_page(doc[i], textes[i]) for i in range(n)]
        sources = {}                                  # index -> 'ocr' / 'vision' (absent = texte natif)
        lignes_extra: dict = {}                       # lignes issues de l'OCR/Vision, par page
        confs: dict = {}
        scans = [i for i in range(n) if infos[i]["type"] == "scan"]
        mixtes = [i for i in range(n) if infos[i]["type"] == "mixte"][:MAX_PAGES_MIXTES]
        stats = {"pages": n, "pages_scan": len(scans), "pages_mixtes": len(mixtes), "pages_ocr": 0, "pages_vision": 0}

        # ---------- (a) OCR local des pages scannées + des images « riches » des pages mixtes ----------
        if utiliser_ocr and (scans or mixtes):
            if not ocr_disponible():
                alertes.append(f"{len(scans)} page(s) scannée(s) détectée(s) mais RapidOCR n'est pas installé "
                               "(pip install rapidocr-onnxruntime numpy).")
            else:
                if on_progress:
                    on_progress(f"OCR de {len(scans)} page(s) scannée(s) et {len(mixtes)} infographie(s)...", 8)
                # les états financiers sont plutôt au milieu / fin du rapport : on les traite en premier
                prio = sorted(scans, key=lambda i: abs(i / n - 0.65))[:MAX_PAGES_OCR]
                taches = [(i, None) for i in prio]
                taches += [(i, pymupdf.Rect(*z)) for i in mixtes for z in infos[i]["zones"]]
                res = _ocr_pages(doc, taches, alertes, on_progress)
                for i, r in res.items():
                    if r["lignes"]:
                        lignes_extra[i] = r["lignes"]
                        confs[i] = sum(r["confs"]) / len(r["confs"]) if r["confs"] else 0.0
                        sources[i] = "ocr"
                stats["pages_ocr"] = len(res)

        # ---------- (b) Gemini Vision : pages sans OCR, ou pages financières à OCR peu fiable ----------
        if utiliser_vision and "gemini" in CLIENTS and (scans or mixtes):
            candidates = []
            for i in scans + mixtes:
                if i not in lignes_extra:                                  # rien lu du tout (OCR absent/échoué)
                    candidates.append((0.0 if i in scans else 1.0, abs(i / n - 0.65), i))
                else:                                                      # OCR lu mais douteux sur une page financière
                    plat_ocr = normaliser_texte("\n".join(lignes_extra[i]))
                    sc = max(scores_page(plat_ocr, _nb_nombres(lignes_extra[i])))
                    if sc >= SCORE_PAGE_MIN and confs.get(i, 1.0) < SEUIL_CONF_OCR:
                        candidates.append((-1.0, confs.get(i, 0.0), i))
            ordre = [i for _, _, i in sorted(candidates)][:MAX_PAGES_VISION]
            if ordre:
                if on_progress:
                    on_progress(f"Relecture Gemini Vision de {len(ordre)} page(s)...", 18)
                vis = _vision_pages(doc, ordre, alertes)
                for i, lignes in vis.items():
                    ancien = lignes_extra.get(i)
                    if ancien is None or _nb_nombres(lignes) >= 0.7 * _nb_nombres(ancien):
                        lignes_extra[i] = lignes
                        sources[i] = "vision"
                stats["pages_vision"] = len(vis)
        elif (scans or mixtes) and utiliser_vision and "gemini" not in CLIENTS and "ocr" not in sources.values():
            alertes.append("Pages scannées non lues : ni OCR (RapidOCR) ni Gemini Vision (GEMINI_API_KEY) disponibles.")

        # ---------- (c) texte final de chaque page ----------
        for i, lignes in lignes_extra.items():
            if infos[i]["type"] == "scan":
                textes[i] = "\n".join(lignes)
            else:                                                          # mixte : texte natif + texte des images
                textes[i] = textes[i] + "\n" + "\n".join(lignes)
        avec_texte = sum(1 for t in textes if len(t.strip()) > 100)
        if avec_texte < 0.2 * n:
            return {"erreur": "PDF scanné (image) : aucun texte exploitable. Installez RapidOCR "
                              "(`pip install rapidocr-onnxruntime numpy`) ou renseignez GEMINI_API_KEY pour la Vision, "
                              "ou passez un OCR externe (`ocrmypdf entree.pdf sortie.pdf`)."}

        plats = [normaliser_texte(t) for t in textes]
        nb_nombres = [len(_RE_DENSITE.findall(p)) for p in plats]
        scores = [scores_page(plats[i], nb_nombres[i]) for i in range(n)]
        cibles = _pages_cibles(scores, n)

        lignes: dict = {}
        for i in cibles:
            if infos[i]["type"] == "scan":
                lignes[i] = list(lignes_extra.get(i, []))
            else:
                lignes[i] = lignes_page(doc[i]) + list(lignes_extra.get(i, []) if infos[i]["type"] == "mixte" else [])
        unites = {i + 1: u for i in range(n) if (u := detecter_unite(textes[i], _plat(textes[i])))}
        unite_cibles = [unites[i + 1] for i in cibles if (i + 1) in unites]
        return {
            "nb_pages": n, "textes": textes, "plats": plats, "scores": scores, "lignes": lignes,
            "perimetres": {i: detecter_perimetre(textes[i]) for i in range(n)},
            "sources_pages": sources, "unites": unites,
            "unite_doc": (Counter(unite_cibles) or Counter(unites.values())).most_common(1)[0][0] if unites else None,
            "annee": detecter_annee([textes[i] for i in cibles]),
            "stats_lecture": stats, "alertes_lecture": alertes, "_cache_phrases": {},
        }
    finally:
        doc.close()


def _unite_page(page, unites: dict, unite_doc):
    """Unité de la page, sinon des pages voisines (l'en-tête d'unité n'est pas répété partout), sinon du document."""
    if page is not None:
        for d in (0, -1, 1, -2, 2, -3, 3):
            if unites.get(page + d):
                return unites[page + d]
    return unite_doc


# ---------------------------------------------------------------------------
# 5. NIVEAU 1 — Lecture géométrique des tableaux, pilotée par le dictionnaire
# ---------------------------------------------------------------------------
_TOK = (r"\(?[-−–]?\s?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
        r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?|\d+(?:[.,]\d+)?)\)?\s?%?")
_RE_TOK = re.compile(_TOK)
_RE_NOTE = re.compile(r"^\d{1,2}(?:[.,]\d)?$")


def _tokens_cellule(c: str):
    """Liste des nombres d'une cellule si elle ne contient QUE des nombres, sinon None."""
    if not c or len(c) > 60 or not _RE_CHIFFRE.search(c):
        return None
    toks = [m.group().strip() for m in _RE_TOK.finditer(c)]
    reste = _RE_TOK.sub("", c).strip(" ()%-−–.,")
    return toks if toks and not reste else None


def _segments(ligne: str) -> list:
    """[(libellé, [nombres])] — gère aussi deux tableaux côte à côte (actif | passif) sur la même ligne."""
    segs, label, nums = [], [], []
    cellules = [x.strip() for x in re.split(r"\s*\|\s*", ligne) if x.strip()]
    if cellules and not any(_tokens_cellule(c) is not None for c in cellules):
        mots = cellules[-1].split(" ")                     # libellé suivi de nombres collés (pas de colonnes détectées)
        k = len(mots)
        while k > 1 and re.match(r"^\(?[-−–]?[\d.,]+\)?%?$|^%$", mots[k - 1]):
            k -= 1
        if k < len(mots) and re.search(r"[A-Za-z]", " ".join(mots[:k])):
            cellules[-1:] = [" ".join(mots[:k]), " ".join(mots[k:])]
    for c in cellules:
        toks = _tokens_cellule(c)
        if toks is not None:
            nums.extend(toks)
        else:
            if nums:
                segs.append((" ".join(label), nums))
                label, nums = [], []
            label.append(c)
    segs.append((" ".join(label), nums))
    return segs


def _valeurs(toks: list, genre: str) -> list:
    """Valeurs [N, N-1] d'une ligne ; ignore la colonne « Notes » (petit entier en tête)."""
    toks = list(toks)
    if genre == "pct":
        vals = [v for v in (parse_nombre(t.replace("%", "")) for t in toks) if v is not None]
        if len(vals) >= 3 and toks and _RE_NOTE.match(toks[0].strip("() ")):
            vals = vals[1:]
        return vals[:2]
    if any("%" in t for t in toks):
        return []
    note = _RE_NOTE.match(toks[0].strip("() ")) if toks else None
    if note and len(toks) >= 3:
        toks.pop(0)
    elif note and len(toks) == 2 and " " not in toks[0] and abs(parse_nombre(toks[1]) or 0) >= 1000:
        toks.pop(0)
    if genre != "unitaire":                               # « 1,234 » = 1 234 (format anglais) sauf valeurs par action
        toks = [re.sub(r",", "", t) if re.match(r"^\(?[-−–]?\d{1,3},\d{3}\)?$", t) else t for t in toks]
    return [v for v in (parse_nombre(t) for t in toks) if v is not None][:2]


def _variantes(lab: str, prev: list) -> list:
    """Libellé seul, puis précédé de 1 ou 2 lignes sans chiffres (libellé coupé sur plusieurs lignes)."""
    v = []
    if len(prev) >= 2:
        v.append(" ".join(prev[-2:] + ([lab] if lab else [])).strip())
    if prev:
        v.append(" ".join(prev[-1:] + ([lab] if lab else [])).strip())
    if lab:
        v.append(lab)
    return [x for x in dict.fromkeys(v) if x]


def extraire_par_libelles(lecture: dict) -> dict:
    """NIVEAU 1. Retourne {champ: [candidats]} pour toutes les lignes de tableaux (texte natif, OCR, Vision)."""
    cands: dict = defaultdict(list)
    for i in sorted(lecture["lignes"]):
        page, per = i + 1, lecture["perimetres"].get(i)
        src_page = lecture["sources_pages"].get(i, "pdf")
        prev: list = []
        for k, ligne in enumerate(lecture["lignes"][i]):
            if not ligne.strip():
                continue
            ref = f"p{page}:L{k:03d}"
            for label, toks in _segments(ligne):
                lab = norm_label(label)
                if not toks:
                    prev = (prev + [lab])[-2:] if lab and len(lab) <= 110 else []
                    continue
                variantes = _variantes(lab, prev)
                prev = []
                if not variantes:
                    continue
                possibles: set = set()
                for v in variantes:
                    for t in v.split():
                        possibles |= TOKEN_INDEX.get(t, _VIDE)
                for nom in possibles:
                    ch = CHAMPS[nom]
                    for v in variantes:                       # la variante la plus longue (la plus précise) d'abord
                        poids = ch.apparier(v)
                        if poids is None:
                            continue
                        vals = _valeurs(toks, ch.genre)
                        if not vals or vals[0] == 0:
                            break
                        if ch.genre == "pct" and not (any("%" in t for t in toks) or _RE_LABEL_RATIO.search(v)):
                            break
                        if ch.genre in ("pct", "unitaire") and not ch.plage_ok(vals[0]):
                            break
                        cands[nom].append({
                            "valeur": vals[0], "n1": vals[1] if len(vals) > 1 else None, "page": page, "prio": poids,
                            "libelle": v, "perimetre": per, "ligne": ligne.strip()[:200], "ref": ref,
                            "source": "libellé", "source_page": src_page, "total": v.startswith("total ")})
                        break
    return cands


# ---------------------------------------------------------------------------
# 6. NIVEAU 2 — Recherche sémantique dans le texte courant (rapport de gestion, notes…)
# ---------------------------------------------------------------------------
# On part du CHIFFRE (« 14,5 % », « 1 200 millions de dinars ») et on remonte dans la phrase pour retrouver
# le mot-clé le plus proche dans le dictionnaire. Cela traite aussi « passe de 75 % à 78 % » et « 14,5 % contre 13,2 % ».
_NUM = r"\d{1,3}(?:[ ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_RE_PCT = re.compile(r"(?P<n>(?<![\d.,])-?\d{1,4}(?:[.,]\d{1,3})?)\s?(?P<u>%|pb\b|bps\b|points? de base|basis points?)")
_RE_MONTANT_TXT = re.compile(
    r"(?P<n>(?<![\d.,])(?:" + _NUM + r"))\s?(?:de )?(?P<u>milliard|mrd|billion|million|zzmega|millier|zzkilo|"
    r"dinar|dt|tnd|eur|euro|usd|dollar)\b")
_FACTEUR_TEXTE = {"milliard": 1e9, "mrd": 1e9, "billion": 1e9, "million": 1e6, "zzmega": 1e6, "millier": 1e3,
                  "zzkilo": 1e3, "dinar": 1.0, "dt": 1.0, "tnd": 1.0, "eur": 1.0, "euro": 1.0, "usd": 1.0, "dollar": 1.0}
_MOTS_UNITES = ("milliard", "mrd", "billion", "million", "zzmega", "millier", "zzkilo", "dinar", " dt", "tnd", "euro", "usd")
_RE_ACCEPTE_N = re.compile(r"(?:\ba|\bto|\bat|atteindre|\bvers|jusqu a|reaching|reach)\s*$")
_RE_N1_APRES_PCT = re.compile(
    r"^[\s,(]*(?:contre|versus|vs\.?|against|compare[d]? (?:to|with)|par rapport a|soit)\s*[a-z ]{0,25}?"
    r"(?P<n>-?\d{1,4}(?:[.,]\d{1,3})?)\s?%")
_RE_SOMMAIRE_DEBUT = re.compile(r"sommair|table (?:de|des) matier|table of content")


def _fenetre_avant(plat: str, pos: int) -> str:
    """Texte qui précède un chiffre, coupé à la frontière de phrase la plus proche."""
    fen = plat[max(0, pos - FENETRE_AVANT): pos]
    cut = max(fen.rfind(". "), fen.rfind("; "), fen.rfind("! "), fen.rfind("? "))
    return fen[cut + 2:] if cut >= 0 else fen


def _meilleur_champ(fenetre: str, champs: list, union):
    """Le champ du dictionnaire dont le mot-clé est le plus PROCHE de la fin de la fenêtre (donc du chiffre)."""
    if not union.search(fenetre):
        return None
    essais = []
    for ch in champs:
        ms = list(ch.rx_texte.finditer(fenetre))
        if not ms:
            continue
        mm = ms[-1]
        if ch.rx_excl and ch.rx_excl.search(fenetre[mm.start():]):
            continue
        essais.append((len(fenetre) - mm.end(), -(mm.end() - mm.start()), ch.nom, mm))
    if not essais:
        return None
    essais.sort(key=lambda e: (e[0], e[1]))
    dist, _, nom, mm = essais[0]
    return nom, mm, dist


def _annee_contexte(inter: str, apres: str):
    ys = re.findall(r"\b(20[12]\d)\b", inter[-50:])
    if ys:
        return int(ys[-1])
    m = re.match(r"\s*(?:\(|en |a fin |fin |au 31 decembr\w* |at |in )?\s*(?:31 decembr\w* |december 31,? )?(20[12]\d)\b", apres)
    return int(m.group(1)) if m else None


def _union(champs: list):
    return re.compile("|".join(ch.rx_texte.pattern for ch in champs)) if champs else None


def recherche_semantique(lecture: dict) -> dict:
    """NIVEAU 2. Parcourt le texte de TOUTES les pages. Retourne {champ: [candidats]} (source « texte »)."""
    cands: dict = defaultdict(list)
    ch_pct = [c for c in CHAMPS.values() if c.genre == "pct" and c.rx_texte is not None]
    ch_mt = [c for c in CHAMPS.values() if c.genre in ("montant", "unitaire") and c.texte and c.rx_texte is not None]
    u_pct, u_mt = _union(ch_pct), _union(ch_mt)
    deja: set = set()
    for i, plat in enumerate(lecture["plats"]):
        if len(plat) < 120 or _RE_SOMMAIRE_DEBUT.search(plat[:400]):
            continue
        page, per = i + 1, lecture["perimetres"].get(i)
        src_page = lecture["sources_pages"].get(i, "pdf")

        # ---------- ratios publiés (pourcentages, points de base) ----------
        if "%" in plat or "basis" in plat or " pb" in plat or " bps" in plat:
            for m in _RE_PCT.finditer(plat):
                fen = _fenetre_avant(plat, m.start())
                trouve = _meilleur_champ(fen, ch_pct, u_pct)
                if not trouve:
                    continue
                nom, mm, dist = trouve
                inter = fen[mm.end():]
                if dist > DISTANCE_MAX_LIBELLE:
                    continue
                n1 = None
                autres = list(_RE_PCT.finditer(inter))
                if autres:                                   # un autre % entre le mot-clé et ce chiffre
                    if _RE_ACCEPTE_N.search(inter[-25:]):    # « passe de 75 % À 78 % » : 78 est N, 75 est N-1
                        n1 = parse_nombre(autres[-1].group("n"))
                    else:                                    # « 14,5 % contre 13,2 % » : ce chiffre est N-1
                        continue
                v = parse_nombre(m.group("n"))
                if v is None or v == 0:
                    continue
                unite = m.group("u")
                if unite not in ("%",):                      # points de base -> %
                    v = v / 100.0
                if n1 is None:
                    q = _RE_N1_APRES_PCT.match(plat[m.end(): m.end() + 70])
                    n1 = parse_nombre(q.group("n")) if q else None
                if not CHAMPS[nom].plage_ok(v):
                    continue
                cle = (nom, page, round(v, 3))
                if cle in deja:
                    continue
                deja.add(cle)
                cands[nom].append({
                    "valeur": v, "n1": n1, "page": page, "prio": 6, "libelle": mm.group(), "perimetre": per,
                    "ligne": (fen[max(0, mm.start() - 15):] + " " + m.group()).strip()[-220:], "ref": None,
                    "source": "texte", "source_page": src_page, "annee": _annee_contexte(inter, plat[m.end(): m.end() + 25]),
                    "num": m.group("n"), "total": False})

        # ---------- montants cités dans le texte (« 1 200 millions de dinars », « 1,2 milliard de DT », « 950 MD ») ----------
        if any(u in plat for u in _MOTS_UNITES):
            for m in _RE_MONTANT_TXT.finditer(plat):
                fen = _fenetre_avant(plat, m.start())
                trouve = _meilleur_champ(fen, ch_mt, u_mt)
                if not trouve:
                    continue
                nom, mm, dist = trouve
                if dist > DISTANCE_MAX_LIBELLE:
                    continue
                inter = fen[mm.end():]
                if _RE_MONTANT_TXT.search(inter) and not _RE_ACCEPTE_N.search(inter[-25:]):
                    continue                                 # « 1 200 MD en 2024 contre 1 100 MD » : 2e chiffre = N-1
                facteur = _FACTEUR_TEXTE[m.group("u")]
                brut = parse_nombre(m.group("n"))
                if brut is None or brut == 0:
                    continue
                ch = CHAMPS[nom]
                if ch.genre == "unitaire" and (facteur != 1.0 or not ch.plage_ok(brut)):
                    continue
                if ch.genre == "montant" and facteur == 1.0 and brut < 1000:
                    continue                                 # « 5 dinars » : pas un montant d'état financier
                v = brut * facteur
                cle = (nom, page, round(v, 0))
                if cle in deja:
                    continue
                deja.add(cle)
                cands[nom].append({
                    "valeur": v, "n1": None, "page": page, "prio": 5, "libelle": mm.group(), "perimetre": per,
                    "ligne": (fen[max(0, mm.start() - 15):] + " " + m.group()).strip()[-220:], "ref": None,
                    "source": "texte", "source_page": src_page, "en_unites": True, "unite_txt": m.group("u"),
                    "annee": _annee_contexte(inter, plat[m.end(): m.end() + 25]), "num": m.group("n"), "total": False})
    return cands


def extrait_brut(lecture: dict, cand: dict, nom: str) -> str:
    """Retrouve dans le texte ORIGINAL (accents, casse) la phrase d'où provient un chiffre lu dans le texte racinisé."""
    num, i = cand.get("num"), cand["page"] - 1
    if not num or i >= len(lecture["textes"]):
        return cand["ligne"][:200]
    flat = re.sub(r"\s+", " ", lecture["textes"][i])
    motif = re.compile(re.escape(num).replace(r"\ ", r"\s?"))
    ch = CHAMPS[nom]
    for k, m in enumerate(motif.finditer(flat)):
        if k >= 8:
            break
        fen = flat[max(0, m.start() - 170): m.end() + 40]
        if ch.rx_texte and ch.rx_texte.search(normaliser_texte(fen)):
            return fen.strip()
    return cand["ligne"][:200]


# ---------------------------------------------------------------------------
# 7. Classement des candidats, cohérence comptable, construction des champs
# ---------------------------------------------------------------------------
def _perimetre_cible(cands: dict, pref: str):
    souhait = "consolide" if pref == "auto" else pref
    pers = Counter(c["perimetre"] for nom in ("total_bilan", "resultat_net", "produit_net_bancaire")
                   for c in cands.get(nom, []) if c.get("perimetre") and c["source"] != "texte")
    if pers.get(souhait):
        return souhait
    return pers.most_common(1)[0][0] if pers else None


def _valeur_unites(c: dict, nom: str, lecture: dict):
    """(valeur, n1, unité, unité_inconnue) — valeur en UNITÉS de la devise (sauf pct et unitaire)."""
    ch = CHAMPS[nom]
    if ch.genre == "pct":
        return c["valeur"], c.get("n1"), "%", False
    if ch.genre == "unitaire":
        return c["valeur"], c.get("n1"), "devise/action", False
    if c.get("en_unites"):
        return c["valeur"], c.get("n1"), c.get("unite_txt", "texte"), False
    unite = _unite_page(c["page"], lecture["unites"], lecture["unite_doc"])
    f = FACTEURS_UNITE[unite or "unites"]
    n1 = c["n1"] * f if c.get("n1") not in (None, 0) else None
    return c["valeur"] * f, n1, unite or "unites", unite is None


def classer_candidats(cands: dict, lecture: dict, pref: str = "auto") -> tuple:
    """Pour chaque champ : liste de candidats triée par SCORE décroissant. Retourne (classes, périmètre retenu)."""
    cible = _perimetre_cible(cands, pref)
    annee = lecture["annee"]
    classes = {}
    for nom, liste in cands.items():
        ch = CHAMPS[nom]
        t = TOPIC_NOMS.index(ch.topic)
        for c in liste:
            c["vu"], c["n1u"], c["unite"], c["unite_inconnue"] = _valeur_unites(c, nom, lecture)
        for c in liste:
            tol = 0.02 if ch.genre == "montant" else 0.005
            pages = {o["page"] for o in liste
                     if o is not c and abs(o["vu"] - c["vu"]) <= tol * max(abs(c["vu"]), 1e-9)}
            c["corroboration"] = len(pages)
            s = float(c["prio"])
            s += min(lecture["scores"][c["page"] - 1][t] / 6, 2.5)                  # la page est du bon type
            s += 1.5 if (c.get("n1") is not None and ch.genre != "pct") else 0      # tableau à 2 colonnes
            if c.get("perimetre") and cible:
                s += 4 if c["perimetre"] == cible else -4
            s += min(c["corroboration"], 3)                                         # même valeur ailleurs
            if c["source"] == "texte":
                s -= 3
            if c.get("source_page") in ("ocr", "vision"):
                s -= 0.5
            if ch.prefere_total and c.get("total"):
                s += 1
            if c.get("annee") and annee and c["annee"] != annee:
                s -= 4 if c["annee"] < annee else 1
            if nom == "total_bilan":
                s += 0.2 * math.log10(abs(c["vu"]) + 1)                             # le total est le plus grand
            c["score"] = round(s, 2)
        classes[nom] = sorted(liste, key=lambda c: (c["score"], abs(c["vu"])), reverse=True)
    return classes, cible


RELATIONS_BILAN = {"capitaux_propres": (0.005, 0.5), "credits_clientele": (0.05, 0.95), "depots_clientele": (0.05, 0.98)}


def arbitrer_coherence(classes: dict, alertes: list) -> dict:
    """Choisit un candidat par champ, puis le corrige si les relations comptables sont violées
    (capitaux propres << total bilan, crédits < bilan, |résultat net| <= PNB…). Aucune IA : pure logique."""
    choix = {nom: lst[0] for nom, lst in classes.items() if lst}

    def vu(nom):
        c = choix.get(nom)
        return c["vu"] if c else None

    # 1) le total bilan est celui qui rend le plus de relations cohérentes
    if len(classes.get("total_bilan", [])) > 1:
        def coherence(b):
            ok = 0
            for nom, (lo, hi) in RELATIONS_BILAN.items():
                v = vu(nom)
                ok += 1 if (v and b and lo <= v / b <= hi) else 0
            v = vu("produit_net_bancaire")
            ok += 1 if (v and b and 0.003 <= v / b <= 0.15) else 0
            return ok
        meilleur = max(classes["total_bilan"][:5], key=lambda c: (coherence(c["vu"]), c["score"]))
        if meilleur is not choix["total_bilan"]:
            alertes.append(f"total_bilan : valeur p.{meilleur['page']} préférée car plus cohérente avec les autres postes.")
            choix["total_bilan"] = meilleur
    # 2) les autres postes doivent rester cohérents avec le total bilan
    b = vu("total_bilan")
    if b:
        for nom, (lo, hi) in RELATIONS_BILAN.items():
            lst = classes.get(nom, [])
            if lst and not lo <= lst[0]["vu"] / b <= hi:
                alt = next((c for c in lst[1:6] if lo <= c["vu"] / b <= hi), None)
                if alt:
                    choix[nom] = alt
                    alertes.append(f"{nom} : premier candidat incohérent avec le total bilan → remplacé (p.{alt['page']}).")
    # 3) |résultat net| <= PNB
    rn, pnb = choix.get("resultat_net"), choix.get("produit_net_bancaire")
    if rn and pnb and abs(rn["vu"]) > pnb["vu"]:
        alt = next((c for c in classes["resultat_net"][1:6] if abs(c["vu"]) <= pnb["vu"]), None)
        if alt:
            choix["resultat_net"] = alt
            alertes.append(f"resultat_net : valeur initiale > PNB → remplacée (p.{alt['page']}).")
    return choix


def _champ_vide() -> dict:
    return {"valeur": None, "valeur_n1": None, "valeur_brute": None, "unite": None, "page": None, "extrait": None,
            "verifie": None, "source": None, "libelle": None, "valeur_milliers": None, "ref": None, "score": None,
            "corroboration": 0, "alternatives": []}


def champ_depuis_candidat(nom: str, c: dict, alternatives: list, lecture: dict, alertes: list) -> dict:
    champ = _champ_vide()
    src = c["source"] if c.get("source_page", "pdf") == "pdf" else f"{c['source']} ({c['source_page']})"
    # Un chiffre lu par Vision (LLM) n'est considéré comme vérifié que s'il est corroboré ailleurs.
    verifie = not (c.get("source_page") == "vision" and c.get("corroboration", 0) == 0)
    champ.update(valeur=c["vu"], valeur_n1=c["n1u"], valeur_brute=c["valeur"], unite=c["unite"], page=c["page"],
                 verifie=verifie, source=src, libelle=c["libelle"], ref=c.get("ref"), score=c.get("score"),
                 corroboration=c.get("corroboration", 0),
                 extrait=(extrait_brut(lecture, c, nom) if c["source"] == "texte" else c["ligne"][:200]),
                 alternatives=[{"valeur": a["vu"], "page": a["page"], "score": a["score"], "source": a["source"]}
                               for a in alternatives[:3]])
    if c["source"] == "texte":
        alertes.append(f"{nom} : lu dans une phrase du rapport (p.{c['page']}) — à vérifier.")
    if c.get("unite_inconnue"):
        alertes.append(f"{nom} : unité introuvable, 'unites' supposée (vérifier).")
    if not verifie:
        alertes.append(f"{nom} : lu par Gemini Vision (p.{c['page']}) sans corroboration — à vérifier.")
    return champ


def construire_champs(classes: dict, choix: dict, lecture: dict, alertes: list) -> dict:
    champs = {nom: _champ_vide() for nom in CHAMPS}
    for nom, c in choix.items():
        autres = [x for x in classes[nom] if x is not c and abs(x["vu"] - c["vu"]) > 1e-9]
        champs[nom] = champ_depuis_candidat(nom, c, autres, lecture, alertes)
    return champs


def reconstituer(champs: dict, alertes: list) -> None:
    """Postes dérivés quand le total n'est pas imprimé : charges d'exploitation, marge d'intérêts, total bilan."""
    def a(c, cle="valeur"):
        return abs(c[cle]) if c[cle] is not None else None

    if champs["charges_exploitation"]["valeur"] is None:
        fp, cg, da = (champs[k] for k in ("frais_personnel", "charges_generales", "dotations_amortissements"))
        if fp["valeur"] is not None and cg["valeur"] is not None:
            parts = [c for c in (fp, cg, da) if c["valeur"] is not None]
            total = sum(a(c) for c in parts)
            n1s = [a(c, "valeur_n1") for c in parts]
            n1 = sum(n1s) if all(x is not None for x in n1s) else None
            libelle = "frais de personnel + charges générales" + (" + amortissements" if da["valeur"] is not None else "")
            champs["charges_exploitation"].update(valeur=total, valeur_n1=n1, valeur_brute=total, unite=fp["unite"],
                                                  page=fp["page"], verifie=True, source="reconstitué", libelle=libelle)
            alertes.append(f"charges_exploitation reconstituées ({libelle}).")
        elif cg["valeur"] is not None:
            champs["charges_exploitation"].update(valeur=a(cg), valeur_n1=a(cg, "valeur_n1"), valeur_brute=a(cg),
                                                  unite=cg["unite"], page=cg["page"], verifie=True, source="reconstitué",
                                                  libelle="charges générales (total supposé)")
            alertes.append("charges_exploitation : seul le poste « charges générales » a été trouvé, pris comme total.")
    ip, ic, mi = champs["interets_produits"], champs["interets_charges"], champs["marge_interets"]
    if mi["valeur"] is None and ip["valeur"] is not None and ic["valeur"] is not None:
        mi.update(valeur=ip["valeur"] - abs(ic["valeur"]), unite=ip["unite"], page=ip["page"], verifie=True,
                  source="reconstitué", libelle="intérêts perçus − intérêts encourus",
                  valeur_n1=(ip["valeur_n1"] - abs(ic["valeur_n1"]) if ip["valeur_n1"] is not None
                             and ic["valeur_n1"] is not None else None))
    tb, tp = champs["total_bilan"], champs["total_passif_cp"]
    if tb["valeur"] is None and tp["valeur"] is not None:
        tb.update({k: tp[k] for k in ("valeur", "valeur_n1", "valeur_brute", "unite", "page", "verifie", "libelle", "extrait")},
                  source="reconstitué")
        alertes.append("total_bilan absent : total « passif et capitaux propres » utilisé à la place.")


def champs_a_completer(champs: dict) -> list:
    ok = lambda k: champs[k]["valeur"] is not None
    m = [k for k in ESSENTIELS_IA if not ok(k)]
    if not ok("charges_exploitation"):
        m += [k for k in ("frais_personnel", "charges_generales", "dotations_amortissements") if not ok(k)]
    if not (ok("taux_npl_rapporte") or ok("creances_douteuses_brutes")):
        m += ["taux_npl_rapporte", "creances_douteuses_brutes"]
    if not (ok("taux_couverture_rapporte") or ok("provisions_creances")):
        m += ["taux_couverture_rapporte", "provisions_creances"]
    if not ok("ratio_solvabilite_rapporte"):
        m.append("ratio_solvabilite_rapporte")
    return list(dict.fromkeys(m))


def champs_ambigus(classes: dict) -> list:
    """Champs essentiels dont les deux meilleurs candidats ont des valeurs différentes et des scores proches."""
    sortie = []
    for nom in ESSENTIELS_IA:
        lst = classes.get(nom, [])
        if len(lst) >= 2 and lst[0]["score"] < 14 and lst[0]["score"] - lst[1]["score"] < 2.0 \
                and abs(lst[0]["vu"] - lst[1]["vu"]) > 0.005 * abs(lst[0]["vu"]):
            sortie.append(nom)
    return sortie[:6]


# ---------------------------------------------------------------------------
# 8. NIVEAU 3 — « LLM pointeur » : le LLM DÉSIGNE une ligne, le code LIT le chiffre
# ---------------------------------------------------------------------------
def _creer_clients() -> dict:
    clients = {}
    if os.getenv("GEMINI_API_KEY") and genai:
        clients["gemini"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
                                         http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000))
    if os.getenv("OPENROUTER_API_KEY") and OpenAI:
        clients["openrouter"] = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                                       timeout=LLM_TIMEOUT_S, max_retries=0)
    if os.getenv("GROQ_API_KEY") and Groq:
        clients["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=LLM_TIMEOUT_S, max_retries=0)
    return clients


CLIENTS = _creer_clients()


def _appel_gemini(client, system, user, json_mode):
    base = dict(system_instruction=system, temperature=0.0,
                response_mime_type="application/json" if json_mode else None)
    try:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
            max_output_tokens=4096, **base)
        return client.models.generate_content(model=MODEL_GEMINI, contents=user, config=config).text or ""
    except Exception as e:
        if "thinking" not in str(e).lower() and getattr(e, "code", None) != 400:
            raise
        config = types.GenerateContentConfig(**base)
        return client.models.generate_content(model=MODEL_GEMINI, contents=user, config=config).text or ""


def _appel_openai_compatible(client, modele, system, user, json_mode):
    kwargs = dict(model=modele, temperature=0.0,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if json_mode:
        try:
            r = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                raise
    return client.chat.completions.create(**kwargs).choices[0].message.content or ""


_APPELS = {
    "gemini": _appel_gemini,
    "openrouter": lambda c, s, u, j: _appel_openai_compatible(c, MODEL_OPENROUTER, s, u, j),
    "groq": lambda c, s, u, j: _appel_openai_compatible(c, MODEL_GROQ, s, u, j),
}


def appeler_llm(system: str, user: str, *, json_mode: bool = True, valider=None, delai_hedge=None):
    """1er fournisseur ; en cas d'échec le suivant démarre aussitôt ; s'il traîne (> delai_hedge) le suivant démarre en
    parallèle et la première réponse valide gagne. Retourne (résultat, fournisseur)."""
    delai = HEDGE_DELAY_S if delai_hedge is None else delai_hedge
    noms = [n for n in LLM_ORDER if n in CLIENTS]
    if not noms:
        raise RuntimeError("Aucun fournisseur LLM configuré (clés API manquantes dans .env).")
    file: queue.Queue = queue.Queue()

    def tache(nom):
        try:
            limite = LIMITE_CARACTERES.get(nom)
            texte = _APPELS[nom](CLIENTS[nom], system, user[:limite] if limite else user, json_mode)
            file.put((nom, valider(texte) if valider else texte, None))
        except Exception as e:
            file.put((nom, None, e))

    def lancer(i):
        threading.Thread(target=tache, args=(noms[i],), daemon=True).start()

    erreurs, lances, en_cours = [], 0, 0
    while True:
        if lances < len(noms) and en_cours == 0:
            lancer(lances)
            lances += 1
            en_cours += 1
        if en_cours == 0:
            break
        try:
            nom, resultat, erreur = file.get(timeout=delai if lances < len(noms) else None)
        except queue.Empty:
            lancer(lances)
            lances += 1
            en_cours += 1
            continue
        en_cours -= 1
        if erreur is None:
            return resultat, nom
        erreurs.append(f"{nom} : {str(erreur)[:150]}")
    raise RuntimeError("Aucun fournisseur LLM n'a répondu. " + " | ".join(erreurs[-4:]))


def phrases_page(lecture: dict, i: int) -> list:
    """Phrases du texte courant d'une page (découpage déterministe : l'index S sert de référence au LLM)."""
    cache = lecture["_cache_phrases"]
    if i not in cache:
        flat = re.sub(r"\s+", " ", lecture["textes"][i])
        morceaux = re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"(•\-])", flat)
        cache[i] = [p.strip()[:350] for p in morceaux if len(p.strip()) >= 25]
    return cache[i]


def construire_contexte_pointeur(cibles: list, lecture: dict, classes: dict, max_car: int = 18_000):
    """Pour chaque champ : une courte liste de LIGNES de tableaux et de PHRASES numérotées (p12:L034 / p45:S007) choisies
    grâce au dictionnaire. Le LLM ne voit que ces candidats — jamais le PDF entier."""
    refs: dict = defaultdict(dict)                     # champ -> {ref: texte affiché}
    cache_norm: dict = {}

    def norm(i, k, ligne):
        if (i, k) not in cache_norm:
            cache_norm[(i, k)] = normaliser_texte(ligne)
        return cache_norm[(i, k)]

    for nom in cibles:
        ch = CHAMPS[nom]
        if ch.rx_texte is None:
            continue
        t = TOPIC_NOMS.index(ch.topic)
        trouves = []
        for i, lignes in lecture["lignes"].items():
            sp = lecture["scores"][i][t]
            for k, ligne in enumerate(lignes):
                if not _RE_CHIFFRE.search(ligne):
                    continue
                pl, prec = norm(i, k, ligne), ""
                m = ch.rx_texte.search(pl)
                if not m and k > 0 and not _RE_CHIFFRE.search(lignes[k - 1]):          # libellé coupé sur 2 lignes
                    prec = lignes[k - 1]
                    pl = norm(i, k - 1, prec) + " " + pl
                    m = ch.rx_texte.search(pl)
                if not m or (ch.rx_excl and ch.rx_excl.search(pl[m.start():])):
                    continue
                trouves.append((sp + min(len(_RE_DENSITE.findall(ligne)) / 3, 2), f"p{i + 1}:L{k:03d}",
                                (prec + " / " if prec else "") + ligne))
        for c in classes.get(nom, [])[:4]:             # candidats existants (mode arbitrage)
            if c.get("ref"):
                trouves.append((99, c["ref"], c["ligne"]))
        if ch.texte:
            n_pages = 0
            for i, plat in enumerate(lecture["plats"]):
                if n_pages >= 30 or not any(p in plat for p in ch.pivots) or not ch.rx_texte.search(plat):
                    continue
                n_pages += 1
                for idx, ph in enumerate(phrases_page(lecture, i)):
                    pn = normaliser_texte(ph)
                    m = ch.rx_texte.search(pn)
                    if not m or (ch.rx_excl and ch.rx_excl.search(pn[m.start():])):
                        continue
                    motif = _RE_PCT if ch.genre == "pct" else _RE_MONTANT_TXT
                    if motif.search(pn):
                        trouves.append((lecture["scores"][i][t] - 2, f"p{i + 1}:S{idx:03d}", ph))
        trouves.sort(key=lambda x: -x[0])
        vus = set()
        for _, ref, texte in trouves:
            if ref in vus or len([1 for r in refs[nom] if r[len(r.split(':')[0]) + 1] == ref[len(ref.split(':')[0]) + 1]]) >= 5:
                continue
            vus.add(ref)
            refs[nom][ref] = texte[:230]
            if len(refs[nom]) >= 7:
                break
    # texte du prompt (borné)
    while True:
        blocs = []
        for nom, lst in refs.items():
            ch = CHAMPS[nom]
            type_ = "pourcentage" if ch.genre == "pct" else "montant"
            lignes = "\n".join(f"  [{r}] {t}" for r, t in lst.items())
            blocs.append(f"## {nom} — {ch.desc} (type : {type_})\n{lignes}")
        corps = "\n\n".join(blocs)
        if len(corps) <= max_car or all(len(l) <= 2 for l in refs.values()):
            return corps, refs
        for nom in refs:                                # réduit progressivement
            if len(refs[nom]) > 2:
                refs[nom].pop(list(refs[nom])[-1])


SYSTEM_POINTEUR = (
    "Tu es un analyste financier spécialisé dans les banques tunisiennes (français/anglais, normes BCT et IFRS).\n"
    "Tu ne lis AUCUN chiffre à voix haute : tu DÉSIGNES la ligne ou la phrase qui contient la donnée demandée.\n\n"
    "RÈGLES STRICTES\n"
    "1. Pour chaque champ, choisis UNE référence parmi celles listées sous ce champ, recopiée exactement "
    "(ex. \"p12:L034\" = ligne de tableau, \"p45:S007\" = phrase du texte). Toute autre référence sera rejetée.\n"
    "2. \"rang\" = position du nombre voulu dans cette ligne/phrase, en comptant à partir de 1, de gauche à droite, "
    "en ignorant la colonne « Notes » (petit entier de renvoi). Dans un tableau à deux colonnes d'exercice : "
    "1 = exercice le plus récent (N), 2 = exercice précédent (N-1). Dans une phrase : rang du nombre voulu parmi "
    "les pourcentages (champ de type pourcentage) ou parmi les montants (champ de type montant).\n"
    "3. Si aucune ligne ne correspond vraiment à la définition (mauvais poste, stock au lieu de flux, minimum "
    "réglementaire au lieu du ratio réalisé, exercice précédent seulement...), mets null. Mieux vaut null qu'une erreur.\n"
    "4. Préfère les comptes consolidés aux comptes individuels, et l'exercice le plus récent.\n"
    "5. Ne calcule rien, n'invente rien, ne recopie aucun nombre.\n"
    "6. Réponds par un unique objet JSON compact, sans texte autour.")


def interroger_pointeur(cibles: list, lecture: dict, classes: dict):
    """Un seul appel LLM pour tous les champs à compléter/arbitrer. Retourne (reponse_validee, refs, fournisseur)."""
    corps, refs = construire_contexte_pointeur(cibles, lecture, classes)
    refs = {n: r for n, r in refs.items() if r}
    if not refs:
        return {}, {}, None
    definitions = "\n".join(f"- \"{n}\" : {CHAMPS[n].desc}" for n in refs)
    user = (f"CHAMPS À LOCALISER :\n{definitions}\n\n"
            "FORMAT DE SORTIE :\n"
            '{"champs": {"nom_du_champ": {"ref": "p12:L034", "rang": 1}, "autre_champ": null}}\n\n'
            f"CANDIDATS (lignes et phrases numérotées) :\n{corps}")

    def valider(texte: str) -> dict:
        obj = extraire_json(texte)
        if not isinstance(obj.get("champs"), dict):
            raise ValueError("JSON sans 'champs'")
        return obj

    obj, fournisseur = appeler_llm(SYSTEM_POINTEUR, user, valider=valider)
    return obj["champs"], refs, fournisseur


def _nombres_phrase(ch: Champ, phrase: str) -> list:
    """Nombres d'une phrase pour un champ : [(valeur, unité_label, en_unites)] dans l'ordre de lecture."""
    pn, sortie = normaliser_texte(phrase), []
    if ch.genre == "pct":
        for m in _RE_PCT.finditer(pn):
            v = parse_nombre(m.group("n"))
            if v is not None:
                sortie.append((v if m.group("u") == "%" else v / 100.0, "%", False))
    else:
        for m in _RE_MONTANT_TXT.finditer(pn):
            v = parse_nombre(m.group("n"))
            if v is not None:
                sortie.append((v * _FACTEUR_TEXTE[m.group("u")], m.group("u"), True))
    return sortie


def lire_reference(ref: str, rang, nom: str, lecture: dict):
    """Le CODE lit le chiffre désigné par le LLM. Retourne un candidat (prio 11) ou None."""
    m = re.match(r"^p(\d+):([LS])(\d+)$", str(ref).strip())
    if not m:
        return None
    page, typ, idx = int(m.group(1)), m.group(2), int(m.group(3))
    i, ch = page - 1, CHAMPS[nom]
    r = max(1, _entier(rang) or 1)
    base = {"page": page, "prio": 11, "perimetre": lecture["perimetres"].get(i), "ref": ref, "source": "LLM-pointeur",
            "source_page": lecture["sources_pages"].get(i, "pdf"), "total": False}
    try:
        if typ == "L":
            ligne = lecture["lignes"][i][idx]
            segs = [(l, t) for l, t in _segments(ligne) if t]
            toks = next((t for l, t in segs if ch.rx_texte and ch.rx_texte.search(normaliser_texte(l))),
                        segs[0][1] if segs else None)
            vals = _valeurs(toks, ch.genre) if toks else []
            if not vals:
                return None
            r = r if r <= len(vals) else 1
            if ch.genre == "pct" and not ch.plage_ok(vals[r - 1]):
                return None
            return {**base, "valeur": vals[r - 1], "n1": vals[1] if r == 1 and len(vals) > 1 else None,
                    "libelle": norm_label(ligne.split("|")[0])[:80], "ligne": ligne.strip()[:200]}
        phrase = phrases_page(lecture, i)[idx]
        nums = _nombres_phrase(ch, phrase)
        if not nums:
            return None
        r = r if r <= len(nums) else 1
        v, u, en_unites = nums[r - 1]
        if ch.genre == "pct" and not ch.plage_ok(v):
            return None
        cand = {**base, "valeur": v, "n1": None, "libelle": "phrase", "ligne": phrase[:200], "num": None}
        if en_unites:
            cand.update(en_unites=True, unite_txt=u)
        return cand
    except (IndexError, KeyError):
        return None


def appliquer_pointeurs(reponse: dict, refs: dict, lecture: dict, cands: dict, alertes: list) -> int:
    """Valide chaque pointeur (la référence doit faire partie des candidats montrés) puis lit le chiffre dans le PDF."""
    n_ok = 0
    for nom, rep in reponse.items():
        if nom not in refs or not isinstance(rep, dict):
            continue
        ref = str(rep.get("ref", "")).strip()
        if ref not in refs[nom]:
            alertes.append(f"{nom} : référence '{ref}' proposée par le LLM hors de la liste fournie → ignorée.")
            continue
        cand = lire_reference(ref, rep.get("rang"), nom, lecture)
        if cand:
            cands[nom].append(cand)
            n_ok += 1
    return n_ok


# ---------------------------------------------------------------------------
# 9. VALIDATION CROISÉE, ratios calculés et score
# ---------------------------------------------------------------------------
def _pct(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den * 100


def _moy(a, b):
    return (a + b) / 2 if a is not None and b is not None else None


def _interp(x: float, mauvais: float, bon: float) -> float:
    t = (x - mauvais) / (bon - mauvais)
    return 100 * max(0.0, min(1.0, t))


def _note_ldr(ldr: float) -> float:
    if 80 <= ldr <= 100:
        return 100
    return _interp(ldr, 130, 100) if ldr > 100 else _interp(ldr, 40, 80)


def _appreciation(score: int) -> str:
    return "Solide" if score >= 75 else "Correct" if score >= 55 else "Fragile" if score >= 40 else "Risqué"


def _comparer(nom: str, publie, calcules: dict, lo: float, hi: float, alertes: list, validation: list):
    """Compare le ratio PUBLIÉ aux ratios RECALCULÉS (plusieurs bases possibles : fin de période, moyenne…).
    • publié ET calculable : on retient la base de calcul la plus proche ; alerte si écart relatif > 10 %.
    • publié seul          : on garde le publié.
    • calculable seul      : le ratio manquant est CALCULÉ automatiquement (première base).
    Retourne (valeur_retenue, 'rapporté' | 'calculé' | None)."""
    calcules = {b: v for b, v in calcules.items() if v is not None and lo <= v <= hi}
    pub_ok = publie is not None and lo <= publie <= hi
    if publie is not None and not pub_ok:
        alertes.append(f"{nom} publié ({publie:.2f}) hors plage plausible [{lo}, {hi}] → ignoré.")
    rec = {"indicateur": nom, "publie": round(publie, 2) if pub_ok else None, "calcule": None, "base_calcul": None,
           "ecart_relatif_pct": None, "statut": None}
    if pub_ok and calcules:
        base, calc = min(calcules.items(), key=lambda kv: abs(publie - kv[1]))
        ecart_abs = abs(publie - calc)
        ecart_rel = ecart_abs / max(abs(calc), 1e-9) * 100
        rec.update(calcule=round(calc, 2), base_calcul=base, ecart_relatif_pct=round(ecart_rel, 1))
        if ecart_rel > SEUIL_ECART_PCT and ecart_abs > ECART_ABS_MIN_PTS:
            rec["statut"] = "ALERTE"
            alertes.append(f"{nom} : publié {publie:.2f} vs recalculé {calc:.2f} ({base}) → écart {ecart_rel:.0f} % "
                           f"(> {SEUIL_ECART_PCT:.0f} %) : base de calcul différente (moyennes, périmètre) ou extraction erronée ?")
        else:
            rec["statut"] = "OK"
        validation.append(rec)
        return publie, "rapporté"
    if pub_ok:
        rec["statut"] = "PUBLIE_SEUL"
        validation.append(rec)
        return publie, "rapporté"
    if calcules:
        base, calc = next(iter(calcules.items()))
        rec.update(calcule=round(calc, 2), base_calcul=base, statut="CALCULE_SEUL")
        validation.append(rec)
        return calc, "calculé"
    return None, None


def calculer_ratios_fondamentaux(donnees: dict) -> dict:
    """Entrée : {champ: {"valeur", "valeur_n1", "verifie", ...}} (montants en unité commune)."""
    try:
        alertes, erreurs, validation = [], [], []
        val = {k: parse_nombre((donnees.get(k) or {}).get("valeur")) for k in CHAMPS}
        val1 = {k: parse_nombre((donnees.get(k) or {}).get("valeur_n1")) for k in CHAMPS}

        def invalider(k, motif):
            val[k] = None
            alertes.append(f"{k} écarté : {motif}")

        non_verifies = [k for k in CHAMPS if val[k] is not None and (donnees.get(k) or {}).get("verifie") is False]
        for k in non_verifies:
            if VERIFICATION_STRICTE:
                invalider(k, "valeur non corroborée (lecture OCR/Vision) — mode strict")
            else:
                alertes.append(f"{k} : valeur non corroborée — à vérifier manuellement.")

        # ---- contrôles de cohérence comptable ----
        bilan = val["total_bilan"]
        if bilan:
            echecs = [k for k, (lo, hi) in RELATIONS_BILAN.items() if val[k] and not lo <= val[k] / bilan <= hi]
            if len(echecs) >= 2:
                invalider("total_bilan", f"incohérent avec {', '.join(echecs)} (erreur d'unité ou mauvaise ligne)")
            else:
                for k in echecs:
                    invalider(k, f"{k}/total_bilan = {val[k] / bilan * 100:.2f}% (erreur d'unité ou mauvaise ligne ?)")
        if val["total_bilan"] and val["total_passif_cp"]:
            ecart = abs(val["total_bilan"] - val["total_passif_cp"]) / val["total_bilan"] * 100
            if ecart > 0.5:
                alertes.append(f"Bilan non équilibré : total actif ≠ total passif + capitaux propres (écart {ecart:.2f} %).")
            else:
                validation.append({"indicateur": "Équilibre du bilan", "publie": None, "calcule": round(ecart, 3),
                                   "base_calcul": "écart actif / passif (%)", "ecart_relatif_pct": round(ecart, 3), "statut": "OK"})
        if val["resultat_net"] and val["produit_net_bancaire"] and abs(val["resultat_net"]) > val["produit_net_bancaire"]:
            alertes.append("Résultat net > PNB : l'un des deux chiffres est probablement faux.")
        if val["produit_net_bancaire"] and val["total_bilan"]:
            r = val["produit_net_bancaire"] / val["total_bilan"] * 100
            if not 0.3 <= r <= 15:
                alertes.append(f"PNB/Total bilan = {r:.2f}% : vérifier les unités.")
        if all(val[k] is not None for k in ("produit_net_bancaire", "charges_exploitation", "cout_du_risque", "resultat_avant_impot")):
            theo = val["produit_net_bancaire"] - abs(val["charges_exploitation"]) - abs(val["cout_du_risque"])
            if abs(val["resultat_avant_impot"]) > 0 and abs(theo - val["resultat_avant_impot"]) > 0.25 * abs(val["resultat_avant_impot"]):
                alertes.append("Info : PNB − charges − coût du risque s'écarte de plus de 25 % du résultat avant impôt "
                               "(autres produits/charges ordinaires non extraits ?).")

        rn, cp, pnb = val["resultat_net"], val["capitaux_propres"], val["produit_net_bancaire"]
        rn1, cp1, bilan1 = val1["resultat_net"], val1["capitaux_propres"], val1["total_bilan"]
        bilan = val["total_bilan"]
        charges = abs(val["charges_exploitation"]) if val["charges_exploitation"] is not None else None
        fp, cg = val["frais_personnel"], val["charges_generales"]
        npl, prov, agios, eng = val["creances_douteuses_brutes"], val["provisions_creances"], val["agios_reserves"], val["engagements_bruts"]
        cred, dep, cred1 = val["credits_clientele"], val["depots_clientele"], val1["credits_clientele"]
        cdr = abs(val["cout_du_risque"]) if val["cout_du_risque"] is not None else None

        # couverture = (provisions + agios réservés) / créances classées, sauf si le libellé les inclut déjà
        prov_total = prov
        if prov is not None and agios is not None and "agios" not in str((donnees.get("provisions_creances") or {}).get("libelle") or ""):
            prov_total = prov + abs(agios)

        sources = {}
        roe, sources["ROE (%)"] = _comparer("ROE", val["roe_rapporte"], {
            "capitaux propres de fin de période": _pct(rn, cp) if cp and cp > 0 else None,
            "capitaux propres moyens": _pct(rn, _moy(cp, cp1)) if _moy(cp, cp1) and _moy(cp, cp1) > 0 else None},
            -100, 80, alertes, validation)
        roa, sources["ROA (%)"] = _comparer("ROA", val["roa_rapporte"], {
            "total actif de fin de période": _pct(rn, bilan), "actif moyen": _pct(rn, _moy(bilan, bilan1))},
            -10, 10, alertes, validation)
        coeff, sources["Coeff. Exploitation (%)"] = _comparer("Coefficient d'exploitation", val["coeff_exploitation_rapporte"], {
            "charges d'exploitation / PNB": _pct(charges, pnb) if pnb and pnb > 0 else None,
            "(personnel + charges générales) / PNB": _pct(abs(fp) + abs(cg), pnb) if fp is not None and cg is not None and pnb and pnb > 0 else None},
            10, 250, alertes, validation)
        taux_npl, sources["Taux NPL (%)"] = _comparer("Taux NPL", val["taux_npl_rapporte"], {
            "créances classées / engagements bruts": _pct(npl, eng),
            "créances classées / (crédits nets + provisions + agios)": _pct(npl, cred + (prov or 0) + abs(agios or 0)) if cred else None,
            "créances classées / crédits nets": _pct(npl, cred)}, 0.1, 80, alertes, validation)
        couv, sources["Taux de Couverture (%)"] = _comparer("Taux de couverture", val["taux_couverture_rapporte"], {
            "(provisions + agios réservés) / créances classées": _pct(prov_total, npl) if npl and npl > 0 else None,
            "provisions / créances classées": _pct(prov, npl) if npl and npl > 0 else None}, 5, 300, alertes, validation)
        ldr_cal = _pct(cred, dep) if cred and dep and dep > 0 else None
        if cred and dep and abs(cred - dep) / dep < 0.001:
            alertes.append("Crédits et dépôts identiques : même ligne extraite deux fois → LDR calculé ignoré.")
            ldr_cal = None
        ldr, sources["LDR (%)"] = _comparer("LDR (crédits/dépôts)", val["ldr_rapporte"], {"crédits / dépôts": ldr_cal},
                                            10, 300, alertes, validation)
        cout_risque, sources["Coût du Risque (% crédits)"] = _comparer("Coût du risque", val["cout_risque_rapporte"], {
            "coût du risque / crédits": _pct(cdr, cred), "coût du risque / crédits moyens": _pct(cdr, _moy(cred, cred1))},
            0.01, 15, alertes, validation)
        solva, sources["Ratio de Solvabilité (%)"] = _comparer("Ratio de solvabilité", val["ratio_solvabilite_rapporte"], {
            "fonds propres nets / actifs pondérés": _pct(val["fonds_propres_nets"], val["actifs_ponderes"])},
            2, 60, alertes, validation)
        tier1, sources["Ratio Tier 1 (%)"] = _comparer("Ratio Tier 1", val["ratio_tier1_rapporte"], {
            "fonds propres de base / actifs pondérés": _pct(val["fonds_propres_tier1"], val["actifs_ponderes"])},
            2, 60, alertes, validation)
        nim, sources["Marge Nette d'Intérêt (%)"] = _comparer("Marge nette d'intérêt", val["nim_rapporte"], {
            "marge d'intérêts / total actif": _pct(val["marge_interets"], bilan),
            "marge d'intérêts / actif moyen": _pct(val["marge_interets"], _moy(bilan, bilan1))},
            0.1, 15, alertes, validation)
        cet1, lcr, nsfr = val["ratio_cet1_rapporte"], val["lcr_rapporte"], val["nsfr_rapporte"]
        liq_bct, levier = val["ratio_liquidite_rapporte"], val["ratio_levier_rapporte"]

        marge = _pct(rn, pnb) if pnb and pnb > 0 else None
        marge = marge if marge is not None and -100 <= marge <= 100 else None
        cp_actif = _pct(cp, bilan)

        def croissance(k):
            if val[k] is None or val1[k] in (None, 0):
                return None
            g = (val[k] - val1[k]) / abs(val1[k]) * 100
            return g if -90 <= g <= 500 else None

        for etiquette, v in (("Taux NPL", taux_npl), ("ROE", roe), ("LDR", ldr), ("Taux de couverture", couv)):
            if v is None:
                erreurs.append(f"Donnée manquante : {etiquette}.")

        liquidite = min((x for x in (lcr, liq_bct) if x is not None), default=None)
        indicateurs = {
            "ROE": (roe, 14, lambda x: _interp(x, 0, 15)),
            "ROA": (roa, 8, lambda x: _interp(x, 0, 1.5)),
            "Coefficient d'exploitation": (coeff, 10, lambda x: _interp(x, 80, 50)),
            "Taux NPL": (taux_npl, 16, lambda x: _interp(x, 20, 4)),
            "Taux de couverture": (couv, 14, lambda x: _interp(x, 40, 80)),
            "LDR": (ldr, 8, _note_ldr),
            "Solvabilité": (solva, 10, lambda x: _interp(x, 8, 15)),
            "Tier 1": (tier1 if tier1 is not None else cet1, 5, lambda x: _interp(x, 6, 11)),
            "Liquidité": (liquidite, 7, lambda x: _interp(x, 90, 150)),
            "Coût du risque": (cout_risque, 8, lambda x: _interp(x, 3, 0.5)),
        }
        notes = {k: f(v) for k, (v, _, f) in indicateurs.items() if v is not None}
        poids_total = sum(p for _, p, _ in indicateurs.values())
        poids_dispo = sum(indicateurs[k][1] for k in notes)
        if poids_dispo == 0:
            return {"erreur": "Aucun indicateur exploitable : extraction insuffisante.",
                    "alertes": alertes, "erreurs_validation": erreurs, "validation_croisee": validation}
        score = round(sum(indicateurs[k][1] * n for k, n in notes.items()) / poids_dispo)
        nb_alertes_ecart = sum(1 for r in validation if r["statut"] == "ALERTE")
        nb_ok = sum(1 for r in validation if r["statut"] == "OK")
        confiance = round(100 * poids_dispo / poids_total) - 7 * len([k for k in non_verifies if val[k] is not None]) \
            - 4 * nb_alertes_ecart + min(3 * nb_ok, 9)
        confiance = max(0, min(100, confiance))
        if confiance < 50:
            alertes.append(f"Confiance faible ({confiance}%) : trop d'indicateurs manquants ou non vérifiés.")

        def r2(x):
            return round(x, 2) if x is not None else "N/A"

        return {
            "donnees_brutes": donnees,
            "erreurs_validation": erreurs,
            "alertes": alertes,
            "validation_croisee": validation,
            "ratios": {
                "ROE (%)": r2(roe), "ROA (%)": r2(roa), "Marge Nette (%)": r2(marge),
                "Coeff. Exploitation (%)": r2(coeff), "Taux NPL (%)": r2(taux_npl),
                "Taux de Couverture (%)": r2(couv), "LDR (%)": r2(ldr),
                "Coût du Risque (% crédits)": r2(cout_risque), "Ratio de Solvabilité (%)": r2(solva),
                "Ratio Tier 1 (%)": r2(tier1), "Ratio CET1 (%)": r2(cet1), "LCR (%)": r2(lcr), "NSFR (%)": r2(nsfr),
                "Ratio de Liquidité BCT (%)": r2(liq_bct), "Ratio de Levier (%)": r2(levier),
                "Marge Nette d'Intérêt (%)": r2(nim),
                "Capitaux Propres / Actif (%)": r2(cp_actif),
                "Commissions / PNB (%)": r2(_pct(val["commissions"], pnb) if pnb and pnb > 0 else None),
                "Frais de Personnel / PNB (%)": r2(_pct(abs(fp), pnb) if fp is not None and pnb and pnb > 0 else None),
                "Croissance Résultat Net (%)": r2(croissance("resultat_net")),
                "Croissance Total Bilan (%)": r2(croissance("total_bilan")),
                "Croissance Crédits (%)": r2(croissance("credits_clientele")),
                "Croissance Dépôts (%)": r2(croissance("depots_clientele")),
                "Croissance PNB (%)": r2(croissance("produit_net_bancaire")),
                "Dividende par action": r2(val["dividende_par_action"]),
                "Résultat par action": r2(val["resultat_par_action"]),
            },
            "sources_ratios": sources,
            "score_fondamental": score,
            "appreciation": _appreciation(score),
            "confiance": confiance,
            "detail_score": {k: round(n) for k, n in notes.items()},
        }
    except Exception as e:
        return {"erreur": f"Erreur de calcul : {e}"}


# ---------------------------------------------------------------------------
# 10. Résumé, cache et point d'entrée
# ---------------------------------------------------------------------------
def _resume_regles(res: dict) -> str:
    r = res["ratios"]

    def ok(k):
        return r.get(k) not in (None, "N/A")

    phrases = [f"Score fondamental de {res['score_fondamental']}/100 ({res['appreciation']}), "
               f"confiance {res['confiance']}% sur les données extraites."]
    if ok("ROE (%)"):
        roe = r["ROE (%)"]
        niveau = "élevée" if roe >= 15 else "correcte" if roe >= 10 else "modeste" if roe >= 0 else "négative"
        phrases.append(f"Rentabilité {niveau} (ROE {roe:.1f}%" + (f", ROA {r['ROA (%)']:.2f}%" if ok("ROA (%)") else "") + ").")
    if ok("Taux NPL (%)"):
        npl = r["Taux NPL (%)"]
        niveau = "maîtrisée" if npl < 5 else "à surveiller" if npl < 10 else "dégradée"
        phrases.append(f"Qualité des actifs {niveau} (créances classées {npl:.1f}%"
                       + (f", couverture {r['Taux de Couverture (%)']:.0f}%" if ok("Taux de Couverture (%)") else "") + ").")
    if ok("Coeff. Exploitation (%)"):
        c = r["Coeff. Exploitation (%)"]
        phrases.append(f"Efficacité {'bonne' if c < 60 else 'moyenne' if c <= 75 else 'faible'} (coefficient d'exploitation {c:.1f}%).")
    if ok("LDR (%)"):
        ldr = r["LDR (%)"]
        phrases.append(f"Transformation {'prudente' if ldr < 80 else 'équilibrée' if ldr <= 100 else 'tendue'} (crédits/dépôts {ldr:.0f}%).")
    if ok("Ratio de Solvabilité (%)"):
        phrases.append(f"Ratio de solvabilité de {r['Ratio de Solvabilité (%)']:.1f}%.")
    if ok("LCR (%)"):
        phrases.append(f"LCR de {r['LCR (%)']:.0f}%.")
    nb_alertes = sum(1 for v in res.get("validation_croisee", []) if v["statut"] == "ALERTE")
    if nb_alertes:
        phrases.append(f"{nb_alertes} ratio(s) publié(s) s'écartent de plus de {SEUIL_ECART_PCT:.0f}% du recalcul.")
    if res["erreurs_validation"]:
        phrases.append("Limites : " + " ".join(res["erreurs_validation"]))
    elif res["alertes"]:
        phrases.append(f"{len(res['alertes'])} alerte(s) de contrôle à consulter.")
    return " ".join(phrases)


SYSTEM_RESUME = ("Tu es analyste bancaire. Rédige en français 4 à 6 phrases à partir UNIQUEMENT des indicateurs fournis. "
                 "N'invente aucun chiffre. Cite les points forts, les points de vigilance et les limites de l'analyse "
                 "(données manquantes, alertes, écarts publié/recalculé). Pas de liste, pas de titre.")


def _lire_cache(cle: str):
    f = CACHE_DIR / f"{cle}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _ecrire_cache(cle: str, contenu: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{cle}.json").write_text(json.dumps(contenu, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:
        _log(f"⚠️ Cache non écrit : {e}")


def generer_resume_ia(res: dict) -> str:
    payload = json.dumps({
        "ratios": {k: v for k, v in res["ratios"].items() if v != "N/A"},
        "score": res["score_fondamental"], "appreciation": res["appreciation"], "confiance": res["confiance"],
        "alertes": res["alertes"][:6], "donnees_manquantes": res["erreurs_validation"],
        "validation_croisee": [v for v in res.get("validation_croisee", []) if v["statut"] == "ALERTE"][:4],
    }, ensure_ascii=False, sort_keys=True)
    cle = "resume_" + hashlib.sha256((PROMPT_VERSION + payload).encode()).hexdigest()[:20]
    en_cache = _lire_cache(cle)
    if en_cache:
        return en_cache["texte"]

    def valider(texte: str) -> str:
        if len(texte.strip()) < 40:
            raise ValueError("résumé vide")
        return texte.strip()

    try:
        texte, _ = appeler_llm(SYSTEM_RESUME, payload, json_mode=False, valider=valider)
        _ecrire_cache(cle, {"texte": texte})
        return texte
    except Exception:
        return _resume_regles(res)


def _ajouter_milliers(champs: dict) -> None:
    """Pour l'affichage : colonne « valeur en milliers » (kTND) à côté de la valeur en unités (TND)."""
    for nom, c in champs.items():
        if CHAMPS[nom].genre == "montant" and c.get("valeur") is not None:
            c["valeur_milliers"] = c["valeur"] / 1000
            c["valeur_n1_milliers"] = c["valeur_n1"] / 1000 if c.get("valeur_n1") is not None else None


def _compter_sources(champs: dict) -> dict:
    return dict(Counter((c["source"] or "").split(" (")[0] for c in champs.values() if c["valeur"] is not None))


def extraire_champs(file_bytes: bytes, perimetre: str, utiliser_ocr: bool, utiliser_vision: bool,
                    etape, durees: dict, alertes_amont: list, alertes_norm: list):
    """Pipeline d'extraction complet (niveaux 0 à 3). Retourne (champs, meta) ou un dict {'erreur': ...}."""
    etape("⏳ 1/5 Lecture du PDF (texte natif, OCR, Vision) et repérage des pages financières...", 5)
    t = time.perf_counter()
    lecture = lire_pdf(file_bytes, utiliser_ocr, utiliser_vision, on_progress=lambda m, p: etape(m, p))
    durees["lecture_pdf_s"] = round(time.perf_counter() - t, 2)
    if "erreur" in lecture:
        return lecture
    alertes_amont.extend(lecture["alertes_lecture"])
    if not lecture["lignes"]:
        return {"erreur": "Aucune page contenant des tableaux financiers n'a été trouvée."}
    _log(f"   {lecture['nb_pages']} pages en {durees['lecture_pdf_s']}s — {len(lecture['lignes'])} pages de tableaux : "
         f"{[i + 1 for i in sorted(lecture['lignes'])][:40]}{'...' if len(lecture['lignes']) > 40 else ''} — "
         f"scans : {lecture['stats_lecture']['pages_scan']} (OCR {lecture['stats_lecture']['pages_ocr']}, "
         f"Vision {lecture['stats_lecture']['pages_vision']})")

    etape("🔎 2/5 Niveau 1 : lecture géométrique des tableaux par libellés (dictionnaire FR/EN)...", 30)
    t = time.perf_counter()
    cands = extraire_par_libelles(lecture)
    n_tab = sum(len(v) for v in cands.values())
    durees["niveau1_tableaux_s"] = round(time.perf_counter() - t, 2)

    etape("📖 3/5 Niveau 2 : recherche sémantique dans le texte (rapport de gestion, notes)...", 45)
    t = time.perf_counter()
    for nom, lst in recherche_semantique(lecture).items():
        cands[nom].extend(lst)
    durees["niveau2_texte_s"] = round(time.perf_counter() - t, 2)
    _log(f"   {n_tab} candidats en tableaux, {sum(len(v) for v in cands.values()) - n_tab} dans le texte "
         f"({durees['niveau1_tableaux_s']}s + {durees['niveau2_texte_s']}s)")

    classes, per_cible = classer_candidats(cands, lecture, perimetre)
    choix = arbitrer_coherence(classes, alertes_norm)
    champs = construire_champs(classes, choix, lecture, alertes_norm)
    reconstituer(champs, alertes_norm)
    manquants = champs_a_completer(champs)
    ambigus = champs_ambigus(classes)
    _log(f"   {sum(1 for c in champs.values() if c['valeur'] is not None)}/{len(CHAMPS)} champs trouvés — "
         f"manquants essentiels : {manquants or 'aucun'} — ambigus : {ambigus or 'aucun'}")

    fournisseur = None
    cibles = list(dict.fromkeys(manquants + ambigus))
    if cibles and CLIENTS:
        etape(f"🎯 4/5 Niveau 3 : LLM pointeur ({len(cibles)} champ(s) à compléter/arbitrer)...", 60)
        t = time.perf_counter()
        try:
            reponse, refs, fournisseur = interroger_pointeur(cibles, lecture, classes)
            if reponse:
                n_ok = appliquer_pointeurs(reponse, refs, lecture, cands, alertes_norm)
                if n_ok:
                    alertes_nouvelles: list = []
                    classes, per_cible = classer_candidats(cands, lecture, perimetre)
                    choix = arbitrer_coherence(classes, alertes_nouvelles)
                    champs = construire_champs(classes, choix, lecture, alertes_nouvelles)
                    reconstituer(champs, alertes_nouvelles)
                    alertes_norm[:] = [a for a in alertes_norm if "lu dans une phrase" not in a] + alertes_nouvelles
        except Exception as e:
            alertes_amont.append(f"Niveau 3 (LLM pointeur) échoué : {str(e)[:200]}")
        durees["niveau3_llm_s"] = round(time.perf_counter() - t, 2)
    elif cibles:
        alertes_amont.append("Aucun fournisseur LLM configuré : champs non complétés → " + ", ".join(cibles))

    meta = {"annee": lecture["annee"], "perimetre": per_cible, "unite_document": lecture["unite_doc"],
            "pages_tableaux": [i + 1 for i in sorted(lecture["lignes"])], "fournisseur_llm": fournisseur,
            "lecture": lecture["stats_lecture"], "sources_champs": _compter_sources(champs),
            "champs_manquants_apres_extraction": [k for k in CHAMPS if champs[k]["valeur"] is None]}
    return champs, meta


def analyser_document_financier(file_bytes: bytes, utiliser_cache: bool = True, avec_resume_ia: bool = False,
                                on_progress=None, perimetre: str = "auto", utiliser_ocr: bool = True,
                                utiliser_vision: bool = True) -> dict:
    """
    perimetre : "auto" (consolidé s'il existe, sinon individuel), "consolide" ou "individuel".
    utiliser_ocr / utiliser_vision : lecture des pages scannées (RapidOCR / Gemini Vision).
    on_progress(message, pourcentage) : rappel optionnel pour une barre de progression (ex. st.progress).

    Unités : toutes les valeurs « valeur » des montants sont en UNITÉS de la devise (ex. TND). Utiliser
    « valeur_milliers » pour afficher des kTND. Les ratios sont en %.
    """
    t0 = time.perf_counter()
    durees: dict = {}

    def etape(message: str, pct: int):
        _log(message)
        if on_progress:
            try:
                on_progress(message, pct)
            except Exception:
                pass

    options = f"{PROMPT_VERSION}|{perimetre}|ocr={int(utiliser_ocr)}|vision={int(utiliser_vision)}"
    cle = hashlib.sha256(file_bytes + options.encode()).hexdigest()[:24]
    cache = _lire_cache(cle) if utiliser_cache else None
    alertes_amont: list = []
    alertes_norm: list = []
    if cache:
        etape("⚡ Extraction récupérée du cache (analyse instantanée).", 80)
        champs, alertes_norm, meta = cache["champs"], cache["alertes"], cache["meta"]
    else:
        sortie = extraire_champs(file_bytes, perimetre, utiliser_ocr, utiliser_vision, etape, durees,
                                 alertes_amont, alertes_norm)
        if isinstance(sortie, dict):
            return sortie
        champs, meta = sortie
        if utiliser_cache and not alertes_amont:
            _ecrire_cache(cle, {"champs": champs, "alertes": alertes_norm, "meta": meta})

    etape("🔍 5/5 Validation croisée, calcul des ratios manquants et score...", 90)
    _ajouter_milliers(champs)
    for nom, c in champs.items():
        if c["valeur"] is not None:
            marque = "✓" if c["verifie"] else "?"
            _log(f"   {marque} {nom:<32} {c['valeur']:>20,.2f}  [{c['unite']}, p.{c['page']}, {c['source']}]")
    resultat = calculer_ratios_fondamentaux(champs)
    if "erreur" in resultat:
        resultat.setdefault("alertes", []).extend(alertes_amont + alertes_norm)
        return resultat
    resultat["alertes"] = alertes_amont + alertes_norm + resultat["alertes"]

    if avec_resume_ia:
        t = time.perf_counter()
        resultat["resume_ia"] = generer_resume_ia(resultat)
        durees["resume_s"] = round(time.perf_counter() - t, 2)
    else:
        resultat["resume_ia"] = _resume_regles(resultat)

    durees["total_s"] = round(time.perf_counter() - t0, 2)
    resultat["metadonnees"] = {**meta, "durees": durees, "version": PROMPT_VERSION,
                               "unite_valeurs": "unités de la devise (TND) ; voir valeur_milliers pour les kTND"}
    etape(f"🏁 Terminé en {durees['total_s']}s", 100)
    return resultat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse fondamentale d'un rapport annuel bancaire (PDF) — v6")
    parser.add_argument("pdf", help="chemin du fichier PDF")
    parser.add_argument("--sans-cache", action="store_true", help="ignore le cache disque")
    parser.add_argument("--resume-ia", action="store_true", help="résumé rédigé par l'IA (un appel de plus)")
    parser.add_argument("--perimetre", choices=["auto", "consolide", "individuel"], default="auto")
    parser.add_argument("--sans-ocr", action="store_true", help="n'utilise pas RapidOCR")
    parser.add_argument("--sans-vision", action="store_true", help="n'utilise pas Gemini Vision")
    parser.add_argument("--court", action="store_true", help="n'affiche pas le détail des données brutes")
    args = parser.parse_args()
    sortie = analyser_document_financier(Path(args.pdf).read_bytes(), utiliser_cache=not args.sans_cache,
                                         avec_resume_ia=args.resume_ia, perimetre=args.perimetre,
                                         utiliser_ocr=not args.sans_ocr, utiliser_vision=not args.sans_vision)
    if args.court:
        sortie.pop("donnees_brutes", None)
    print(json.dumps(sortie, ensure_ascii=False, indent=2, default=str))