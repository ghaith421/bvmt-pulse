"""
Analyse fondamentale d'un rapport annuel bancaire (PDF) — version 4 (précise + rapide)

Ce qui change par rapport à la v3
---------------------------------
La v3 envoyait quelques pages « au hasard du score » à l'IA : si le compte de résultat n'était pas parmi
les pages choisies, PNB / charges / coût du risque restaient à None. La v4 fonctionne en deux temps :

  1. REPÉRAGE PAR LIBELLÉS (sans IA, quelques secondes) : chaque page est notée selon le nombre de
     postes financiers distincts qu'elle contient (bilan / résultat / risque / ratios), les lignes de
     tableaux sont reconstruites, puis chaque poste est lu directement à partir de son libellé
     (vocabulaire tunisien/maghrébin + IFRS anglais). Le bon candidat est choisi par un score
     (précision du libellé, type de page, colonne N-1 présente, périmètre consolidé/individuel,
     même valeur retrouvée sur plusieurs pages).
  2. IA SEULEMENT POUR CE QUI MANQUE : un contexte court (meilleures pages + lignes repérées par mots-clés)
     est envoyé pour les seuls champs introuvables. Si tout est trouvé à l'étape 1, aucun appel IA.

  Ensuite : normalisation des unités (détection « en 1000 DT », « mDT », « milliers de dinars »…),
  reconstitution des charges d'exploitation (frais de personnel + charges générales + amortissements),
  vérification d'ancrage des chiffres de l'IA, cohérence croisée, ratios, score pondéré + confiance.

Dépendances : pip install pymupdf python-dotenv google-genai groq openai
Variables .env : GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY (facultatives si le repérage suffit)
Optionnelles   : LLM_ORDER, GEMINI_MODEL, OPENROUTER_MODEL, GROQ_MODEL, GEMINI_THINKING_BUDGET (0),
                 HEDGE_DELAY_S (12), VERIFICATION_STRICTE=1, ANALYSE_CACHE_DIR
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from dotenv import load_dotenv

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

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERBOSE = True
PROMPT_VERSION = "v4.0"                      # change => invalide le cache
CACHE_DIR = Path(os.getenv("ANALYSE_CACHE_DIR", ".cache_analyses"))

LLM_ORDER = [p.strip() for p in os.getenv("LLM_ORDER", "gemini,openrouter,groq").split(",") if p.strip()]
MODEL_GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_OPENROUTER = os.getenv("OPENROUTER_MODEL", "openrouter/free")
MODEL_GROQ = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
LLM_TIMEOUT_S = 45
HEDGE_DELAY_S = float(os.getenv("HEDGE_DELAY_S", "12"))
LIMITE_CARACTERES = {"groq": 22_000}

MAX_PAGES_SCAN = 800         # pages scannées au maximum
SCORE_PAGE_MIN = 3.0         # score minimal pour qu'une page soit retenue pour un thème
PAGES_PAR_TOPIC = 10         # meilleures pages retenues par thème (+ voisines des 3 premières)
PAGES_LLM, CARACTERES_LLM = 6, 22_000
CARACTERES_PAR_PAGE = 7_000
VERIFICATION_STRICTE = os.getenv("VERIFICATION_STRICTE", "0") == "1"


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Outils sur les nombres et le texte
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


_RE_JETON = re.compile(r"\d{1,3}(?:[ \u00a0\u202f.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


def nombres_de_texte(texte: str) -> set[float]:
    """Ensemble des valeurs absolues (arrondies) présentes dans un texte, pour l'ancrage."""
    sortie: set[float] = set()
    for m in _RE_JETON.finditer(texte):
        jeton = m.group()
        for cand in (parse_nombre(jeton), parse_nombre(re.sub(r"[ \u00a0\u202f.,]", "", jeton))):
            if cand is not None:
                sortie.add(round(abs(cand), 3))
    return sortie


def plier(texte: str) -> str:
    """Minuscules, sans accents, apostrophes normalisées."""
    t = texte.replace("’", "'").replace("\u00a0", " ").replace("\u202f", " ")
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()


def _plat(texte: str) -> str:
    """Texte plié sur une seule ligne (les libellés coupés sur 2 lignes restent détectables)."""
    return re.sub(r"\s+", " ", plier(texte))


_RE_NON_LABEL = re.compile(r"[^a-z0-9' ]+")
_RE_CODE = re.compile(r"^(?:[a-z]{1,3}\d{1,2}[a-z]?|\d{1,2}[a-z]?|[ivx]{1,4}|[a-d])\s+(?=[a-z])")


def norm_label(s: str) -> str:
    """Libellé normalisé : sans accents/ponctuation, sans code comptable (« AC3 - », « CP7 », « 1. »)."""
    t = _RE_NON_LABEL.sub(" ", plier(s))
    t = re.sub(r"\s+", " ", t).strip()
    return _RE_CODE.sub("", t, count=1).strip()


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
# 2. Champs à extraire
# ---------------------------------------------------------------------------
# nom: (description pour l'IA, type)   type = "montant" ou "pct"
CHAMPS_ETATS = {
    "total_bilan": ("Total de l'actif (= total du passif et des capitaux propres), le plus grand chiffre du bilan. Tunisie : « Total actifs ».", "montant"),
    "capitaux_propres": ("Total des capitaux propres (ligne de total, en bas du passif). PAS le capital social seul, PAS « avant résultat de l'exercice ».", "montant"),
    "resultat_net": ("Résultat net de l'exercice (compte de résultat / état de résultat). Si comptes consolidés : part du groupe acceptée.", "montant"),
    "produit_net_bancaire": ("Produit net bancaire (PNB) / Net banking income / total net revenues. Ligne « PRODUIT NET BANCAIRE » de l'état de résultat.", "montant"),
    "charges_exploitation": ("Total des charges d'exploitation NON bancaires / charges opératoires / frais généraux (personnel + charges générales + amortissements). Valeur positive. Ce n'est PAS « charges d'exploitation bancaire » (intérêts encourus). null si aucun total n'est imprimé.", "montant"),
    "credits_clientele": ("Créances / crédits sur la clientèle au bilan (loans and advances to customers). Tunisie : « Créances sur la clientèle » (actif).", "montant"),
    "depots_clientele": ("Dépôts et avoirs de la clientèle au passif (customer deposits). Tunisie : « Dépôts et avoirs de la clientèle ».", "montant"),
    "cout_du_risque": ("Coût du risque de l'EXERCICE (compte de résultat). Tunisie : « Dotations aux provisions et résultat des corrections de valeurs sur créances, hors bilan et passif ». Ce n'est PAS le stock de provisions.", "montant"),
    "frais_personnel": ("Frais de personnel / charges de personnel de l'exercice (compte de résultat).", "montant"),
    "charges_generales": ("Charges générales d'exploitation (hors personnel) de l'exercice (compte de résultat).", "montant"),
    "dotations_amortissements": ("Dotations aux amortissements et aux provisions sur immobilisations de l'exercice (compte de résultat).", "montant"),
}
CHAMPS_RISQUE = {
    "creances_douteuses_brutes": ("MONTANT brut des créances / engagements classés / douteux / NPL (en montant, pas en %).", "montant"),
    "provisions_creances": ("STOCK total des provisions / dépréciations sur créances ou engagements classés (note risque). Pas le coût du risque annuel.", "montant"),
    "agios_reserves": ("Stock des agios réservés (intérêts réservés sur créances classées), s'il est présenté séparément.", "montant"),
    "taux_npl_rapporte": ("Taux de créances/engagements classés / NPL ratio publié, en % (ex. 3.6).", "pct"),
    "taux_couverture_rapporte": ("Taux de couverture des créances classées publié, en % (ex. 75).", "pct"),
    "roe_rapporte": ("ROE / rentabilité des capitaux propres publiée, en %.", "pct"),
    "roa_rapporte": ("ROA / rentabilité des actifs publiée, en %.", "pct"),
    "coeff_exploitation_rapporte": ("Coefficient d'exploitation / cost-to-income publié, en %.", "pct"),
    "ratio_solvabilite_rapporte": ("Ratio de solvabilité global (fonds propres / actifs pondérés) publié, en %.", "pct"),
}
TOUS_CHAMPS = {**CHAMPS_ETATS, **CHAMPS_RISQUE}
FACTEURS_UNITE = {"unites": 1.0, "milliers": 1e3, "millions": 1e6, "milliards": 1e9}
PCT = {k for k, (_, g) in TOUS_CHAMPS.items() if g == "pct"}

TOPICS = {
    "bilan": ("total_bilan", "capitaux_propres", "credits_clientele", "depots_clientele"),
    "resultat": ("produit_net_bancaire", "resultat_net", "cout_du_risque", "charges_exploitation",
                 "frais_personnel", "charges_generales", "dotations_amortissements"),
    "risque": ("creances_douteuses_brutes", "provisions_creances", "agios_reserves",
               "taux_npl_rapporte", "taux_couverture_rapporte"),
    "ratios": ("roe_rapporte", "roa_rapporte", "coeff_exploitation_rapporte", "ratio_solvabilite_rapporte"),
}
TOPIC_NOMS = list(TOPICS)
TOPIC_DE = {c: t for t, cs in TOPICS.items() for c in cs}
assert set(TOPIC_DE) == set(TOUS_CHAMPS)

# Champs que l'IA doit compléter s'ils manquent après le repérage par libellés
ESSENTIELS_IA = ["total_bilan", "capitaux_propres", "resultat_net", "produit_net_bancaire",
                 "credits_clientele", "depots_clientele", "cout_du_risque", "charges_exploitation"]

# ---------------------------------------------------------------------------
# 3. Dictionnaire de libellés (CÅ’UR DE LA PRÉCISION — ajouter un synonyme = une ligne)
#    Les regex s'appliquent au libellé normalisé (minuscules, sans accents ni ponctuation, sans code
#    comptable). Ancrées avec ^ ... $ pour ne pas confondre « Total actifs » et « Total actifs financiers ».
#    Format : champ: ([(regex, priorité 0-10), ...], regex_d_exclusion)
# ---------------------------------------------------------------------------
_EXCL_TAUX = r"taux|ratio|couverture|coverage|pourcent|par action"

LIBELLES_BRUTS = {
    "total_bilan": ([
        (r"^total (?:de l'|des |du |de la )?(?:actifs?|bilan|general)(?: net)?(?: consolide)?$", 10),
        (r"^total assets$", 10),
        (r"^total (?:du )?(?:passifs? et capitaux propres|capitaux propres et passifs?)$", 8),
        (r"^total (?:equity and liabilities|liabilities and (?:shareholders'? |stockholders'? )?equity)$", 8),
    ], r"ponder|moyen|hors bilan|financ|courant|immobilis|gere|gestion"),
    "capitaux_propres": ([
        (r"^total (?:des |du |de )?(?:capitaux propres|fonds propres)(?: part du groupe| du groupe| consolides)?$", 10),
        (r"^total (?:shareholders'? |stockholders'? )?equity$", 10),
        (r"^capitaux propres(?: part du groupe| consolides)?$", 6),
    ], r"avant|passif|dettes|liabilit|minoritaire|ponder|reglementaire"),
    "resultat_net": ([
        (r"^resultat net(?: consolide)?(?: de (?:l'exercice|la periode|l'annee))?(?: consolide)?"
         r"(?: part du groupe| part groupe)?(?: apres modifications comptables)?$", 10),
        (r"^benefice net(?: de l'exercice)?$", 9),
        (r"^resultat de l'exercice$", 8),
        (r"^net (?:income|profit)(?: for the (?:year|period))?$", 10),
        (r"^profit for the (?:year|period)$", 9),
    ], r"avant|hors|minoritaire|interets|par action|ordinaires"),
    "produit_net_bancaire": ([
        (r"^(?:total )?(?:du )?produit net bancaire(?: pnb)?(?: consolide)?$", 10),
        (r"^pnb$", 9),
        (r"^(?:total )?net banking income(?: nbi)?$", 10),
        (r"^total net (?:operating )?(?:revenues?|income)$", 6),
        (r"^net operating income$", 5),
    ], r"avant|hors|par action|variation|evolution"),
    "charges_exploitation": ([
        (r"^total (?:des )?charges d'exploitation$", 10),
        (r"^charges operatoires$", 10),
        (r"^(?:total )?(?:des )?frais generaux(?: d'exploitation)?$", 9),
        (r"^(?:total )?(?:operating expenses|non interest expenses?)$", 9),
        (r"^charges d'exploitation non bancaires?$", 8),
    ], r"(?<!non )bancaire"),
    "frais_personnel": ([
        (r"^(?:frais|charges) de personnel$", 10),
        (r"^(?:personnel|staff|employee(?: benefits?)?) (?:expenses|costs)$", 9),
        (r"^salaries and (?:employee )?benefits$", 8),
    ], r"avant|variation"),
    "charges_generales": ([
        (r"^charges generales d'exploitation$", 10),
        (r"^(?:autres )?frais administratifs$", 8),
        (r"^(?:general (?:and )?administrative expenses|general operating expenses|other operating expenses)$", 8),
    ], r"avant|variation"),
    "dotations_amortissements": ([
        (r"^dotations aux amortissements(?: et aux (?:provisions|depreciations?))?(?: sur immobilisations)?"
         r"(?: corporelles et incorporelles)?$", 10),
        (r"^depreciation(?: and| &)? amorti[sz]ation(?: and impairment)?$", 8),
    ], r"avant|variation"),
    "credits_clientele": ([
        (r"^creances (?:nettes )?(?:sur|a|aupres de) (?:la )?clientele(?: nettes?)?$", 10),
        (r"^(?:credits|prets) (?:et creances )?(?:a|sur|aupres de) (?:la )?clientele$", 10),
        (r"^prets et creances (?:a|sur) la clientele", 9),
        (r"^(?:net )?loans and advances to customers(?: net)?$", 10),
        (r"^(?:net )?loans(?: and receivables)? to customers$", 9),
    ], r"etablissement|banque|bancaire|financ|hors bilan|brut|provision|depreciation|interet|agios|moyen|variation|dont"),
    "depots_clientele": ([
        (r"^depots(?: et avoirs)? (?:de|des) (?:la )?clientele$", 10),
        (r"^depots (?:de la |des )?clients?$", 9),
        (r"^(?:comptes crediteurs|dettes) (?:de|envers) (?:la )?clientele$", 9),
        (r"^(?:customer deposits|deposits from customers|amounts due to customers|due to customers)$", 10),
    ], r"etablissement|banque|financ|moyen|variation|charge|interet"),
    "cout_du_risque": ([
        (r"^cout du risque(?: net)?(?: de credit)?(?: de l'exercice)?$", 10),
        (r"^cost of risk$", 10),
        (r"^dotations? aux provisions et resultat des corrections de valeurs sur creances hors bilan et passif$", 10),
        (r"^dotations? aux provisions et resultat des corrections de valeurs sur creances", 9),
        (r"^dotations? aux provisions (?:pour|sur) (?:risques? de credit|creances)(?: nettes?)?$", 7),
        (r"^(?:net )?(?:impairment (?:losses|charges?) on (?:loans|financial assets)|provision for (?:credit|loan) losses)$", 8),
        (r"^charges? nettes? (?:pour|de) risque de credit$", 8),
    ], r"reprise|taux|ratio|autres elements|autres actifs|immobilis|bps"),
    "creances_douteuses_brutes": ([
        (r"^(?:total )?(?:des )?(?:creances|engagements|actifs|prets|encours) (?:brut(?:e)?s? )?"
         r"(?:classe(?:e)?s?|douteu(?:x|ses)|non performants?|en souffrance|compromis)(?: .*)?$", 9),
        (r"^(?:total )?(?:gross )?(?:non performing|nonperforming|impaired) (?:loans|assets|exposures|loans and advances)(?: .*)?$", 10),
        (r"^(?:total )?npls?(?: brut)?$", 9),
    ], _EXCL_TAUX + r"|provision|agios|garanti|reserv|couvert|dotation|reprise|variation|evolution|nouveau|repartition"),
    "provisions_creances": ([
        (r"^(?:total )?(?:des )?provisions? (?:et|ou) agios reserves$", 9),
        (r"^(?:total )?(?:des )?(?:provisions?|depreciations?) (?:individuelles? |collectives? )?(?:sur|pour|des) "
         r"(?:creances|engagements)(?: sur la clientele| classees?)?$", 9),
        (r"^(?:total )?(?:loan|credit) loss (?:allowances?|reserves?)$", 10),
        (r"^(?:total )?(?:allowances?|reserves?) for (?:loan|credit)(?: and lease)? losses$", 10),
        (r"^total (?:des )?provisions$", 7),
    ], r"dotation|reprise|taux|ratio|couverture|variation|charge|cout|net des"),
    "agios_reserves": ([
        (r"^(?:total )?(?:des )?agios reserves(?: sur (?:creances|engagements|clientele).*)?$", 9),
    ], r"dotation|reprise|variation"),
    # --- ratios publiés (regex non ancrées à droite : « Taux de couverture des créances classées par … ») ---
    "taux_npl_rapporte": ([
        (r"^taux (?:des |de |d')?(?:creances|engagements|actifs|prets|encours) (?:brut(?:e)?s? )?"
         r"(?:classe|douteu|non perf|en souffrance|impaye)\w*", 10),
        (r"^ratio (?:des |de )?(?:creances|engagements) (?:classe|douteu)\w*", 9),
        (r"^(?:taux npl|npl ratio|npl rate|non performing loans? ratio|part des (?:creances|engagements) classe\w*)", 10),
    ], r"couverture|coverage|provision"),
    "taux_couverture_rapporte": ([
        (r"^taux de couverture(?: des (?:creances|engagements|actifs)(?: brut\w*)? (?:classe|douteu|non perf)\w*)?", 10),
        (r"^(?:coverage ratio|provision(?:ing)? coverage|taux de provisionnement|npl coverage|loan loss coverage)", 9),
    ], r"emploi|ressource|liquid|lcr|depots|import|export|cout|charge|interet|marge"),
    "roe_rapporte": ([
        (r"^(?:roe\b|return on (?:average )?equity|rentabilite des capitaux propres|rentabilite financiere|"
         r"rendement des capitaux propres)", 10),
    ], r"cible|objectif|target|ajuste|tangible"),
    "roa_rapporte": ([
        (r"^(?:roa\b|return on (?:average )?assets?|rentabilite des actifs|rentabilite economique|rendement des actifs)", 10),
    ], r"cible|objectif|target|ajuste"),
    "coeff_exploitation_rapporte": ([
        (r"^(?:coefficient d'exploitation|coeff(?:icient)? d'exploitation|cost (?:to )?income(?: ratio)?|"
         r"ratio d'efficience|coefficient d'efficience)", 10),
    ], r"cible|objectif|target"),
    "ratio_solvabilite_rapporte": ([
        (r"^(?:ratio (?:global )?de solvabilite|ratio de solvabilite global|solvency ratio|capital adequacy ratio|"
         r"total capital ratio|ratio de fonds propres totaux)", 10),
    ], r"tier ?1|cet ?1|de base|minimum|exige|reglementaire|requis|seuil|cible|buffer|coussin|levier|leverage"),
}
assert set(LIBELLES_BRUTS) == set(TOUS_CHAMPS)

LIBELLES = {nom: ([(re.compile(p), w) for p, w in pats], re.compile(excl) if excl else None, TOUS_CHAMPS[nom][1])
            for nom, (pats, excl) in LIBELLES_BRUTS.items()}


def _sans_ancres(p: str) -> str:
    return p.replace("^", "").replace("$", "").replace(".*", "")


# Mots-clés non ancrés, dérivés automatiquement des libellés : sert à noter les pages et à repérer des lignes.
MOTS_CLES = {nom: re.compile(r"\b(?:" + "|".join("(?:" + _sans_ancres(p) + ")" for p, _ in pats) + ")")
             for nom, (pats, _) in LIBELLES_BRUTS.items()}
PATTERNS_TOUS = [(rx, 1) for rx in MOTS_CLES.values()]
LIBRES_PCT = {nom: MOTS_CLES[nom] for nom in PCT}                  # recherche dans le texte narratif
PLAGES_PCT = {"taux_npl_rapporte": (0, 100), "taux_couverture_rapporte": (5, 300), "roe_rapporte": (-100, 80),
              "roa_rapporte": (-10, 10), "coeff_exploitation_rapporte": (10, 200), "ratio_solvabilite_rapporte": (2, 50)}

_RE_ANCRES = re.compile(
    r"total|resultat|benefice|profit|bilan|balance sheet|clientele|customer|pnb|produit|banking|npl|npe|creance|"
    r"engagement|couverture|coverage|risque|risk|cout|cost|solvab|\broe\b|\broa\b|exploitation|impairment|provision|"
    r"agios|tier|cet ?1|equity|capitaux|liabilit|asset|actif|depot|deposit|loans|personnel|staff|frais|charges|"
    r"dotation|amortis|depreciation|allowance|classe|douteu|performing|coeff|efficien|taux|income|revenue|operating")
_RE_DENSITE = re.compile(r"\d[\d .,]{2,}\d")
_RE_SOMMAIRE = re.compile(r"sommaire|table des matieres|table of contents|\.{6,}")
_RE_CHIFFRE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# 4. Lecture du PDF : notation des pages, lignes reconstruites, unités, périmètre
# ---------------------------------------------------------------------------
def scores_page(plat: str) -> tuple:
    """Un score par thème = 2 × nombre de postes financiers DISTINCTS présents sur la page (+ densité numérique)."""
    zeros = (0.0,) * len(TOPIC_NOMS)
    if len(plat) < 200 or not _RE_ANCRES.search(plat):
        return zeros
    touches = {nom for nom, rx in MOTS_CLES.items() if rx.search(plat)}
    if not touches:
        return zeros
    bonus = min(len(_RE_DENSITE.findall(plat)) / 40, 3) - (6 if _RE_SOMMAIRE.search(plat) else 0)
    sortie = []
    for champs in TOPICS.values():
        k = sum(1 for c in champs if c in touches)
        sortie.append(max(2 * k + bonus, 0.0) if k else 0.0)
    return tuple(sortie)


def _pages_cibles(scores: list, n: int) -> list[int]:
    cibles: set[int] = set()
    for t in range(len(TOPIC_NOMS)):
        classement = sorted(((s[t], i) for i, s in enumerate(scores) if s[t] >= SCORE_PAGE_MIN), reverse=True)
        top = [i for _, i in classement[:PAGES_PAR_TOPIC]]
        cibles.update(top)
        for i in top[:3]:                               # un état financier s'étale souvent sur 2 pages
            cibles.update(v for v in (i - 1, i + 1) if 0 <= v < n)
    return sorted(cibles)


# Abréviations sensibles à la casse (mDT = milliers, MDT = millions) puis formulations longues
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


def detecter_unite(texte: str, texte_plie: str) -> str | None:
    compte: Counter = Counter()
    for unite, regexes in _UNITES_BRUT.items():
        compte[unite] += sum(len(rx.findall(texte)) for rx in regexes)
    for unite, regexes in _UNITES_PLIE.items():
        compte[unite] += sum(len(rx.findall(texte_plie)) for rx in regexes)
    compte = +compte
    return compte.most_common(1)[0][0] if compte else None


def detecter_perimetre(texte: str) -> str | None:
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


def detecter_annee(plats: dict) -> int | None:
    c: Counter = Counter()
    for t in plats.values():
        c.update(re.findall(r"(?:31 decembre|31 december|december 31,?)\s*(20\d\d)", t))
    return int(max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]) if c else None


def lignes_page(page, tol_y: float = 3.0, ecart_colonne: float = 9.0) -> str:
    """Reconstruit les lignes visuelles : 'Total actifs | 12 434 225 | 11 000 000' (colonnes séparées par « | »)."""
    mots = page.get_text("words")
    if not mots:
        return ""
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
    return "\n".join(sortie)


def lire_pdf(file_bytes: bytes) -> dict:
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    try:
        n = min(len(doc), MAX_PAGES_SCAN)
        textes = [doc[i].get_text("text") for i in range(n)]
        avec_texte = sum(1 for t in textes if len(t.strip()) > 100)
        if n == 0 or avec_texte < 0.2 * n:
            return {"erreur": "PDF scanné (image) : aucun texte exploitable. Passez d'abord un OCR "
                              "(ex. `ocrmypdf entree.pdf sortie.pdf`) puis relancez l'analyse."}
        plats = [_plat(t) for t in textes]
        scores = [scores_page(p) for p in plats]
        cibles = _pages_cibles(scores, n)
        unites = {i + 1: detecter_unite(textes[i], plats[i]) for i in range(n)}
        unites = {p: u for p, u in unites.items() if u}
        unite_cibles = [unites[i + 1] for i in cibles if (i + 1) in unites]
        return {
            "nb_pages": n,
            "scores": scores,                                            # index 0-based
            "lignes": {i: lignes_page(doc[i]) for i in cibles},          # index 0-based
            "plats": {i: plats[i] for i in cibles},
            "perimetres": {i: detecter_perimetre(textes[i]) for i in cibles},
            "unites": unites,                                            # clés 1-based (n° de page)
            "unite_doc": (Counter(unite_cibles) or Counter(unites.values())).most_common(1)[0][0] if unites else None,
            "annee": detecter_annee({i: plats[i] for i in cibles}),
        }
    finally:
        doc.close()


def _unite_page(page: int | None, unites: dict, unite_doc):
    """Unité de la page, sinon des pages voisines (l'en-tête d'unité n'est pas répété partout), sinon du document."""
    if page is not None:
        for d in (0, -1, 1, -2, 2, -3, 3):
            if unites.get(page + d):
                return unites[page + d]
    return unite_doc


# ---------------------------------------------------------------------------
# 5. Repérage par libellés (sans IA)
# ---------------------------------------------------------------------------
_TOK = (r"\(?[-−–]?\s?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
        r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?|\d+(?:[.,]\d+)?)\)?\s?%?")
_RE_TOK = re.compile(_TOK)
_RE_NOTE = re.compile(r"^\d{1,2}(?:[.,]\d)?$")
_RE_PCT_QUEUE = re.compile(r"(-?\d{1,3}(?:[.,]\d{1,2})?) ?%")


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
    if genre == "pct":
        for t in toks:
            v = parse_nombre(t)
            if v is not None:
                return [v]
        return []
    if any("%" in t for t in toks):
        return []
    toks = list(toks)
    note = _RE_NOTE.match(toks[0].strip("() ")) if toks else None
    if note and len(toks) >= 3:
        toks.pop(0)
    elif note and len(toks) == 2 and " " not in toks[0] and abs(parse_nombre(toks[1]) or 0) >= 1000:
        toks.pop(0)
    toks = [re.sub(r",", "", t) if re.match(r"^\(?[-−–]?\d{1,3},\d{3}\)?$", t) else t for t in toks]   # 1,234 = milliers
    vals = [v for v in (parse_nombre(t) for t in toks) if v is not None]
    return vals[:2]


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


def _plage_ok(nom: str, v: float) -> bool:
    lo, hi = PLAGES_PCT.get(nom, (-1e18, 1e18))
    return lo <= v <= hi


def extraire_par_libelles(lecture: dict) -> dict:
    """Retourne {champ: [candidats]} pour toutes les lignes de tableaux (et phrases pour les ratios)."""
    cands: dict = defaultdict(list)
    for i, texte in lecture["lignes"].items():
        page, per = i + 1, lecture["perimetres"].get(i)
        prev: list = []
        for ligne in texte.split("\n"):
            if not ligne.strip():
                continue
            for label, toks in _segments(ligne):
                lab = norm_label(label)
                if not toks:
                    prev = (prev + [lab])[-2:] if lab and len(lab) <= 110 else []
                    continue
                variantes = _variantes(lab, prev)
                prev = []
                if not variantes or not any(_RE_ANCRES.search(v) for v in variantes):
                    continue
                for nom, (pats, excl, genre) in LIBELLES.items():
                    for v in variantes:                       # la variante la plus longue (la plus précise) d'abord
                        if excl and excl.search(v):
                            continue
                        poids = max((w for rx, w in pats if rx.match(v)), default=None)
                        if poids is None:
                            continue
                        vals = _valeurs(toks, genre)
                        if vals and vals[0] != 0 and (genre != "pct" or _plage_ok(nom, vals[0])):
                            cands[nom].append({"valeur": vals[0], "n1": vals[1] if len(vals) > 1 else None,
                                               "page": page, "prio": poids, "libelle": v, "perimetre": per,
                                               "ligne": ligne.strip()[:200], "source": "libellé"})
                        break
        # ratios cités dans le texte courant (« le taux de couverture s'établit à 75,3 % »)
        plat = lecture["plats"].get(i, "")
        for nom, rx in LIBRES_PCT.items():
            excl = LIBELLES[nom][1]
            for m in rx.finditer(plat):
                if excl and excl.search(m.group()):
                    continue
                q = _RE_PCT_QUEUE.search(plat[m.end(): m.end() + 100])
                v = parse_nombre(q.group(1)) if q else None
                if v is not None and v != 0 and _plage_ok(nom, v):
                    cands[nom].append({"valeur": v, "n1": None, "page": page, "prio": 6, "libelle": m.group(),
                                       "perimetre": per, "ligne": plat[max(0, m.start() - 20): m.end() + 60],
                                       "source": "texte"})
                    break
    return cands


def _perimetre_cible(cands: dict, pref: str) -> str | None:
    souhait = "consolide" if pref == "auto" else pref
    pers = Counter(c["perimetre"] for nom in ("total_bilan", "resultat_net", "produit_net_bancaire")
                   for c in cands.get(nom, []) if c["perimetre"])
    if pers.get(souhait):
        return souhait
    return pers.most_common(1)[0][0] if pers else None


def choisir_candidats(cands: dict, lecture: dict, pref: str = "auto"):
    """Pour chaque champ, le candidat au meilleur score. Retourne ({champ: candidat}, périmètre retenu)."""
    cible = _perimetre_cible(cands, pref)
    choisis = {}
    for nom, liste in cands.items():
        t = TOPIC_NOMS.index(TOPIC_DE[nom])
        pages_par_valeur: dict = defaultdict(set)
        for c in liste:
            pages_par_valeur[round(abs(c["valeur"]), 2)].add(c["page"])
        meilleur, meilleur_score = None, -1e9
        for c in liste:
            s = float(c["prio"])
            s += min(lecture["scores"][c["page"] - 1][t] / 6, 2.5)                   # la page est du bon type
            s += 1.5 if (c["n1"] is not None and nom not in PCT) else 0              # tableau à 2 colonnes
            if c["perimetre"] and cible:
                s += 4 if c["perimetre"] == cible else -4
            s += min(len(pages_par_valeur[round(abs(c["valeur"]), 2)]) - 1, 3)       # même valeur ailleurs
            if c["source"] == "texte":
                s -= 3
            if nom == "total_bilan":
                s += 0.2 * math.log10(abs(c["valeur"]) + 1)                          # le total est le plus grand
            if s > meilleur_score:
                meilleur, meilleur_score = c, s
        meilleur["score"] = round(meilleur_score, 2)
        choisis[nom] = meilleur
    return choisis, cible


def _champ_vide() -> dict:
    return {"valeur": None, "valeur_n1": None, "valeur_brute": None, "unite": None, "page": None,
            "extrait": None, "verifie": None, "source": None, "libelle": None, "valeur_milliers": None}


def _champ_depuis_candidat(nom: str, c: dict, lecture: dict, alertes: list) -> dict:
    champ = _champ_vide()
    champ.update(valeur_brute=c["valeur"], page=c["page"], verifie=True, source=c["source"],
                 libelle=c["libelle"], extrait=c["ligne"][:200])
    if nom in PCT:
        champ.update(valeur=c["valeur"], unite="%")
        if c["source"] == "texte":
            alertes.append(f"{nom} : ratio lu dans une phrase du rapport (p.{c['page']}) — à vérifier.")
        return champ
    unite = _unite_page(c["page"], lecture["unites"], lecture["unite_doc"])
    if unite is None:
        alertes.append(f"{nom} : unité introuvable, 'unites' supposée (vérifier).")
        unite = "unites"
    f = FACTEURS_UNITE[unite]
    champ.update(valeur=c["valeur"] * f, unite=unite,
                 valeur_n1=(c["n1"] * f if c["n1"] not in (None, 0) else None))
    return champ


def _reconstituer_charges(champs: dict, alertes: list) -> None:
    """Si aucun total de charges d'exploitation n'est imprimé : frais de personnel + charges générales + amortissements."""
    if champs["charges_exploitation"]["valeur"] is not None:
        return
    fp, cg, da = (champs[k] for k in ("frais_personnel", "charges_generales", "dotations_amortissements"))

    def a(c, cle="valeur"):
        return abs(c[cle]) if c[cle] is not None else None

    if fp["valeur"] is not None and cg["valeur"] is not None:
        parts = [c for c in (fp, cg, da) if c["valeur"] is not None]
        total = sum(a(c) for c in parts)
        n1s = [a(c, "valeur_n1") for c in parts]
        n1 = sum(n1s) if all(x is not None for x in n1s) else None
        libelle = "frais de personnel + charges générales" + (" + amortissements" if da["valeur"] is not None else "")
        base = fp
    elif cg["valeur"] is not None:
        total, n1, libelle, base = a(cg), a(cg, "valeur_n1"), "charges générales (total supposé)", cg
        alertes.append("charges_exploitation : seul le poste « charges générales » a été trouvé, pris comme total.")
    else:
        return
    champs["charges_exploitation"].update(
        valeur=total, valeur_n1=n1, valeur_brute=total, unite=base["unite"], page=base["page"], verifie=True,
        source="reconstitué", libelle=libelle, extrait=None)
    alertes.append(f"charges_exploitation reconstituées ({libelle}).")


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


# ---------------------------------------------------------------------------
# 6. Complément par l'IA (uniquement pour les champs manquants)
# ---------------------------------------------------------------------------
def filtrer_lignes(texte: str, patterns=PATTERNS_TOUS) -> str:
    """Ne garde que les lignes utiles : chiffres, mots-clés financiers, ou libellé court avant une ligne chiffrée."""
    lignes = texte.split("\n")
    gardees = []
    for i, ligne in enumerate(lignes):
        if i < 3 or _RE_CHIFFRE.search(ligne):
            gardees.append(ligne)
            continue
        suivante = lignes[i + 1] if i + 1 < len(lignes) else ""
        if len(ligne) <= 70 and _RE_CHIFFRE.search(suivante):
            gardees.append(ligne)
            continue
        pl = plier(ligne)
        if any(rx.search(pl) for rx, _ in patterns):
            gardees.append(ligne)
    return "\n".join(gardees)


def _formater_contexte(pages: dict, unites: dict) -> str:
    morceaux = []
    for num in sorted(pages):
        u = unites.get(num)
        entete = f"--- PAGE {num}" + (f" | unité détectée : {u}" if u else "") + " ---"
        morceaux.append(f"{entete}\n{pages[num]}")
    return "\n\n".join(morceaux)


def contexte_llm(champs: list, lecture: dict) -> str:
    """Meilleures pages pour ces champs + lignes repérées par mots-clés (même hors des pages classées)."""
    topics = {TOPIC_NOMS.index(TOPIC_DE[c]) for c in champs}
    classement = sorted(((max(lecture["scores"][i][t] for t in topics), i) for i in lecture["lignes"]), reverse=True)
    pages, total = {}, 0
    for s, i in classement:
        if s <= 0 or len(pages) >= PAGES_LLM:
            break
        texte = filtrer_lignes(lecture["lignes"][i])[:CARACTERES_PAR_PAGE]
        if pages and total + len(texte) > CARACTERES_LLM:
            continue
        pages[i + 1] = texte
        total += len(texte)
    reperes = []
    pliees = {i: [plier(l) for l in t.split("\n")] for i, t in lecture["lignes"].items()}
    for c in champs:
        n = 0
        for i in sorted(lecture["lignes"]):
            brutes = lecture["lignes"][i].split("\n")
            for k, pl in enumerate(pliees[i]):
                if n >= 4:
                    break
                if MOTS_CLES[c].search(pl) and (_RE_CHIFFRE.search(brutes[k]) or
                                                (k + 1 < len(brutes) and _RE_CHIFFRE.search(brutes[k + 1]))):
                    suite = brutes[k + 1] if not _RE_CHIFFRE.search(brutes[k]) and k + 1 < len(brutes) else ""
                    reperes.append(f"[{c}] [p.{i + 1}] {brutes[k][:150]} {suite[:100]}".strip())
                    n += 1
    ctx = _formater_contexte(pages, lecture["unites"])
    if reperes:
        ctx += "\n\n--- LIGNES REPÉRÉES PAR MOTS-CLÉS (préfixe [p.n] = numéro de page) ---\n" + "\n".join(reperes)
    return ctx


# ---- Appels LLM : requêtes « hedgées », fallback immédiat, validation de la réponse ----
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


def appeler_llm(system: str, user: str, *, json_mode: bool = True, valider=None, delai_hedge: float | None = None):
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


def _regles_extraction() -> str:
    regles = [
        "Recopie uniquement ce qui est imprimé dans le texte. Ne calcule rien, n'additionne rien, ne déduis rien, n'estime rien.",
        "Exercice N = l'exercice le plus récent (lis les années en en-tête de colonnes). \"valeur\" = colonne N ; \"valeur_n1\" = colonne N-1 (si demandé).",
        "Les tableaux ont souvent une colonne « Notes » (petits entiers 1 à 40) AVANT les montants : ne la prends JAMAIS pour une valeur.",
        "\"valeur\" = le nombre tel qu'imprimé, SANS conversion d'unité, sans espaces ni séparateurs de milliers, point comme séparateur décimal (« 1 234 567,8 » -> 1234567.8). Nombre entre parenthèses ou précédé de « - » -> négatif.",
        "\"unite\" = unité du tableau lue dans son en-tête : \"unites\", \"milliers\", \"millions\" ou \"milliards\" (« en milliers de dinars » ou « en 1000 DT » -> \"milliers\"). Pour un pourcentage : \"%\".",
        "\"page\" = numéro n du marqueur « --- PAGE n --- » (ou du préfixe [p.n]) où se trouve le chiffre.",
        "Donnée introuvable -> la clé entière vaut null. N'écris JAMAIS 0 par défaut.",
        "Si le texte contient des comptes consolidés ET individuels, utilise les consolidés. Indique le périmètre dans \"perimetre\".",
        "Les lignes ont été reconstruites depuis le PDF : les colonnes d'un tableau sont séparées par « | ». Un libellé peut être coupé sur 2 lignes.",
        "Réponds directement par un unique objet JSON compact, sans texte autour, sans explication, sans commentaire.",
    ]
    corps = "\n".join(f"{i}. {r}" for i, r in enumerate(regles, 1))
    return ("Tu es un analyste financier spécialisé dans les banques (Tunisie, Maghreb, IFRS). Tu extrais des chiffres d'un rapport annuel.\n\n"
            "RÈGLES STRICTES\n" + corps)


SYSTEM_EXTRACTION = _regles_extraction()


def construire_prompts(section: str, contexte: str, champs_cibles=None) -> tuple[str, str]:
    avec_n1 = section == "etats"
    base = CHAMPS_ETATS if avec_n1 else CHAMPS_RISQUE
    champs = {k: v for k, v in base.items() if champs_cibles is None or k in champs_cibles}
    definitions = "\n".join(f'- "{nom}" : {desc}' for nom, (desc, _) in champs.items())
    exemple = ('{"valeur": 1234567.8, "valeur_n1": 1100000, "unite": "milliers", "page": 45}'
               if avec_n1 else '{"valeur": 3.6, "unite": "%", "page": 60}')
    titre = "états financiers" if avec_n1 else "qualité des actifs et ratios publiés"
    user = (f"CHAMPS À EXTRAIRE ({titre}) :\n{definitions}\n\n"
            "FORMAT DE SORTIE (toutes les clés ci-dessus doivent figurer dans \"champs\", null si introuvable) :\n"
            '{"annee_n": 2024, "perimetre": "consolidé", "champs": {"nom_du_champ": ' + exemple + ", ...}}\n\n"
            f"TEXTE DU RAPPORT :\n{contexte}")
    return SYSTEM_EXTRACTION, user


def extraire_section(section: str, contexte: str, champs_cibles=None):
    system, user = construire_prompts(section, contexte, champs_cibles)

    def valider(texte: str) -> dict:
        obj = extraire_json(texte)
        if not isinstance(obj.get("champs"), dict):
            raise ValueError("JSON sans 'champs'")
        return obj

    return appeler_llm(system, user, valider=valider)


# ---------------------------------------------------------------------------
# 7. Vérification (ancrage dans le texte) et normalisation des unités — champs issus de l'IA
# ---------------------------------------------------------------------------
def _normaliser_unite(u) -> str | None:
    if not u:
        return None
    u = plier(str(u))
    if "milliard" in u or "billion" in u:
        return "milliards"
    if "million" in u:
        return "millions"
    if "millier" in u or "thousand" in u or u in ("k", "kdt", "1000", "1 000"):
        return "milliers"
    if any(m in u for m in ("unit", "dinar", "dirham", "tnd", "dt", "eur", "usd", "mad")):
        return "unites"
    return None


def _verifier(x: float, page: int | None, nombres_pages: dict, tous: set):
    cible = round(abs(x), 3)
    candidates = [page, page - 1, page + 1] if page is not None else []
    for p in candidates:
        if p in nombres_pages and cible in nombres_pages[p]:
            return True, p
    if cible in tous:
        for p, ensemble in nombres_pages.items():
            if cible in ensemble:
                return True, p
    return False, None


def normaliser_champs(champs_llm: dict, pages: dict, unites_pages: dict, unite_doc, alertes: list) -> dict:
    nombres_pages = {p: nombres_de_texte(t) for p, t in pages.items()}
    tous = set().union(*nombres_pages.values()) if nombres_pages else set()
    sortie = {}
    for nom, (_, genre) in TOUS_CHAMPS.items():
        champ = _champ_vide()
        sortie[nom] = champ
        brut = champs_llm.get(nom)
        if not isinstance(brut, dict):
            continue
        x = parse_nombre(brut.get("valeur"))
        if x is None or x == 0:
            continue
        page = _entier(brut.get("page"))
        ok, page_trouvee = _verifier(x, page, nombres_pages, tous)
        champ.update(valeur_brute=x, page=page_trouvee or page, verifie=ok, source="IA",
                     extrait=str(brut.get("extrait") or "")[:200] or None)
        x1 = parse_nombre(brut.get("valeur_n1"))
        if genre == "pct":
            if not _plage_ok(nom, x):
                alertes.append(f"{nom} : {x} hors plage plausible → ignoré.")
                champ.update(valeur_brute=None, verifie=None, source=None)
                continue
            champ.update(valeur=x, unite="%")
            continue
        u_llm, u_det = _normaliser_unite(brut.get("unite")), unites_pages.get(champ["page"])
        if u_llm and u_det and u_llm != u_det:
            alertes.append(f"{nom} : unité annoncée par l'IA ({u_llm}) ≠ unité détectée p.{champ['page']} "
                           f"({u_det}) → '{u_det}' retenue.")
            unite = u_det
        else:
            unite = u_llm or _unite_page(champ["page"], unites_pages, unite_doc)
        if unite is None:
            alertes.append(f"{nom} : unité introuvable, 'unites' supposée (vérifier).")
            unite = "unites"
        f = FACTEURS_UNITE[unite]
        champ.update(valeur=x * f, unite=unite, valeur_n1=(x1 * f if x1 not in (None, 0) else None))
    return sortie


def completer_par_ia(champs: dict, manquants: list, lecture: dict, alertes_amont: list, alertes_norm: list) -> dict:
    if not CLIENTS:
        alertes_amont.append("Aucun fournisseur IA configuré : champs non complétés → " + ", ".join(manquants))
        return {}
    par_section = {"etats": [c for c in manquants if c in CHAMPS_ETATS],
                   "risque": [c for c in manquants if c in CHAMPS_RISQUE]}
    taches = {s: cs for s, cs in par_section.items() if cs}
    contextes = {s: contexte_llm(cs, lecture) for s, cs in taches.items()}
    fournisseurs, resultats = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, len(taches))) as pool:
        futurs = {s: pool.submit(extraire_section, s, contextes[s], cs) for s, cs in taches.items()}
        for s, f in futurs.items():
            try:
                res, fournisseurs[s] = f.result()
                resultats[s] = res
            except Exception as e:
                alertes_amont.append(f"Complément IA '{s}' échoué : {str(e)[:200]}")
    champs_llm = {k: v for r in resultats.values() for k, v in r["champs"].items()}
    pages_verif = {i + 1: t for i, t in lecture["lignes"].items()}
    normalises = normaliser_champs(champs_llm, pages_verif, lecture["unites"], lecture["unite_doc"], alertes_norm)
    for nom in manquants:
        if normalises[nom]["valeur"] is not None:
            champs[nom] = normalises[nom]
    return fournisseurs


# ---------------------------------------------------------------------------
# 8. Cohérence, ratios et score
# ---------------------------------------------------------------------------
def _choisir(nom, rapporte, calcule, lo, hi, alertes):
    """Préfère le ratio publié s'il est plausible, sinon le ratio recalculé. Alerte si fort écart."""
    rap_ok = rapporte is not None and lo <= rapporte <= hi
    cal_ok = calcule is not None and lo <= calcule <= hi
    if rapporte is not None and not rap_ok:
        alertes.append(f"{nom} publié ({rapporte:.2f}) hors plage plausible [{lo}, {hi}] → ignoré.")
    if rap_ok and cal_ok:
        ecart = abs(rapporte - calcule)
        if ecart > 2 and ecart > 0.4 * abs(calcule):
            alertes.append(f"{nom} : publié {rapporte:.2f} vs recalculé {calcule:.2f} (écart notable : "
                           "base de calcul différente ? capitaux moyens, périmètre…).")
    if rap_ok:
        return rapporte, "rapporté"
    if cal_ok:
        return calcule, "calculé"
    return None, None


def _pct(num, den):
    if num is None or den is None or den == 0:
        return None
    return num / den * 100


def _interp(x: float, mauvais: float, bon: float) -> float:
    t = (x - mauvais) / (bon - mauvais)
    return 100 * max(0.0, min(1.0, t))


def _note_ldr(ldr: float) -> float:
    if 80 <= ldr <= 100:
        return 100
    return _interp(ldr, 130, 100) if ldr > 100 else _interp(ldr, 40, 80)


def _appreciation(score: int) -> str:
    return "Solide" if score >= 75 else "Correct" if score >= 55 else "Fragile" if score >= 40 else "Risqué"


def calculer_ratios_fondamentaux(donnees: dict) -> dict:
    """Entrée : {champ: {"valeur", "valeur_n1", "verifie", ...}} (montants en unité commune)."""
    try:
        alertes, erreurs = [], []
        val = {k: parse_nombre((donnees.get(k) or {}).get("valeur")) for k in TOUS_CHAMPS}
        val1 = {k: parse_nombre((donnees.get(k) or {}).get("valeur_n1")) for k in TOUS_CHAMPS}

        def invalider(k, motif):
            val[k] = None
            alertes.append(f"{k} écarté : {motif}")

        non_verifies = [k for k in TOUS_CHAMPS if val[k] is not None and (donnees.get(k) or {}).get("verifie") is False]
        for k in non_verifies:
            if VERIFICATION_STRICTE:
                invalider(k, "valeur introuvable dans le texte du PDF (hallucination probable)")
            else:
                alertes.append(f"{k} : valeur non retrouvée dans le texte du PDF — à vérifier manuellement.")

        bilan = val["total_bilan"]
        if bilan:
            tests = {"capitaux_propres": (0.005, 0.5), "credits_clientele": (0.05, 0.95), "depots_clientele": (0.05, 0.98)}
            echecs = [k for k, (lo, hi) in tests.items() if val[k] and not lo <= val[k] / bilan <= hi]
            if len(echecs) >= 2:
                invalider("total_bilan", f"incohérent avec {', '.join(echecs)} (erreur d'unité ou mauvaise ligne)")
            else:
                for k in echecs:
                    invalider(k, f"{k}/total_bilan = {val[k] / bilan * 100:.2f}% (erreur d'unité ou mauvaise ligne ?)")
        if val["resultat_net"] and val["produit_net_bancaire"] and abs(val["resultat_net"]) > val["produit_net_bancaire"]:
            alertes.append("Résultat net > PNB : l'un des deux chiffres est probablement faux.")
        if val["produit_net_bancaire"] and val["total_bilan"]:
            r = val["produit_net_bancaire"] / val["total_bilan"] * 100
            if not 0.3 <= r <= 15:
                alertes.append(f"PNB/Total bilan = {r:.2f}% : vérifier les unités.")

        rn, cp, pnb = val["resultat_net"], val["capitaux_propres"], val["produit_net_bancaire"]
        charges = abs(val["charges_exploitation"]) if val["charges_exploitation"] is not None else None
        npl, prov, agios = val["creances_douteuses_brutes"], val["provisions_creances"], val["agios_reserves"]
        cred, dep, bilan = val["credits_clientele"], val["depots_clientele"], val["total_bilan"]
        cdr = abs(val["cout_du_risque"]) if val["cout_du_risque"] is not None else None

        # couverture = (provisions + agios réservés) / créances classées, sauf si le libellé les inclut déjà
        prov_total = prov
        if prov is not None and agios is not None and "agios" not in str((donnees.get("provisions_creances") or {}).get("libelle") or ""):
            prov_total = prov + abs(agios)

        sources = {}
        roe, sources["ROE (%)"] = _choisir("ROE", val["roe_rapporte"], _pct(rn, cp) if cp and cp > 0 else None, -100, 80, alertes)
        roa, sources["ROA (%)"] = _choisir("ROA", val["roa_rapporte"], _pct(rn, bilan), -10, 10, alertes)
        coeff, sources["Coeff. Exploitation (%)"] = _choisir("Coeff. d'exploitation", val["coeff_exploitation_rapporte"],
                                                              _pct(charges, pnb) if pnb and pnb > 0 else None, 10, 200, alertes)
        taux_npl, sources["Taux NPL (%)"] = _choisir("Taux NPL", val["taux_npl_rapporte"], _pct(npl, cred), 0, 100, alertes)
        couv, sources["Taux de Couverture (%)"] = _choisir("Taux de couverture", val["taux_couverture_rapporte"],
                                                           _pct(prov_total, npl) if npl and npl > 0 else None, 5, 300, alertes)

        marge = _pct(rn, pnb) if pnb and pnb > 0 else None
        marge = marge if marge is not None and -100 <= marge <= 100 else None

        ldr = None
        if cred and dep and dep > 0:
            if abs(cred - dep) / dep < 0.001:
                alertes.append("Crédits et dépôts identiques : même ligne extraite deux fois → LDR ignoré.")
            else:
                ldr = _pct(cred, dep)
                ldr = ldr if 10 < ldr < 300 else None

        cout_risque = _pct(cdr, cred)
        cout_risque = cout_risque if cout_risque is not None and 0 <= cout_risque <= 15 else None
        solva = val["ratio_solvabilite_rapporte"]
        solva = solva if solva is not None and 2 <= solva <= 50 else None
        cp_actif = _pct(cp, bilan)

        def croissance(k):
            if val[k] is None or val1[k] in (None, 0):
                return None
            g = (val[k] - val1[k]) / abs(val1[k]) * 100
            return g if -90 <= g <= 500 else None

        if taux_npl is None:
            erreurs.append("Donnée manquante : Taux NPL.")
        if roe is None:
            erreurs.append("Donnée manquante : ROE (capitaux propres / résultat net).")
        if ldr is None:
            erreurs.append("Donnée manquante ou erronée : LDR (crédits / dépôts).")
        if couv is None:
            erreurs.append("Donnée manquante : Taux de couverture.")

        indicateurs = {
            "ROE": (roe, 15, lambda x: _interp(x, 0, 15)),
            "ROA": (roa, 10, lambda x: _interp(x, 0, 1.5)),
            "Coefficient d'exploitation": (coeff, 10, lambda x: _interp(x, 80, 50)),
            "Taux NPL": (taux_npl, 20, lambda x: _interp(x, 20, 4)),
            "Taux de couverture": (couv, 15, lambda x: _interp(x, 40, 80)),
            "LDR": (ldr, 10, _note_ldr),
            "Solvabilité": (solva, 10, lambda x: _interp(x, 8, 14)),
            "Coût du risque": (cout_risque, 10, lambda x: _interp(x, 3, 0.5)),
        }
        notes = {k: f(v) for k, (v, _, f) in indicateurs.items() if v is not None}
        poids_total = sum(p for _, p, _ in indicateurs.values())
        poids_dispo = sum(indicateurs[k][1] for k in notes)
        if poids_dispo == 0:
            return {"erreur": "Aucun indicateur exploitable : extraction insuffisante.",
                    "alertes": alertes, "erreurs_validation": erreurs}
        score = round(sum(indicateurs[k][1] * n for k, n in notes.items()) / poids_dispo)
        confiance = max(0, round(100 * poids_dispo / poids_total) - 7 * len([k for k in non_verifies if val[k] is not None]))
        if confiance < 50:
            alertes.append(f"Confiance faible ({confiance}%) : trop d'indicateurs manquants ou non vérifiés.")

        def r2(x):
            return round(x, 2) if x is not None else "N/A"

        return {
            "donnees_brutes": donnees,
            "erreurs_validation": erreurs,
            "alertes": alertes,
            "ratios": {
                "ROE (%)": r2(roe), "ROA (%)": r2(roa), "Marge Nette (%)": r2(marge),
                "Coeff. Exploitation (%)": r2(coeff), "Taux NPL (%)": r2(taux_npl),
                "Taux de Couverture (%)": r2(couv), "LDR (%)": r2(ldr),
                "Coût du Risque (% crédits)": r2(cout_risque), "Ratio de Solvabilité (%)": r2(solva),
                "Capitaux Propres / Actif (%)": r2(cp_actif),
                "Croissance Résultat Net (%)": r2(croissance("resultat_net")),
                "Croissance Total Bilan (%)": r2(croissance("total_bilan")),
                "Croissance Crédits (%)": r2(croissance("credits_clientele")),
                "Croissance Dépôts (%)": r2(croissance("depots_clientele")),
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
# 9. Résumé, cache et point d'entrée
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
        phrases.append(f"Ratio de solvabilité publié de {r['Ratio de Solvabilité (%)']:.1f}%.")
    if res["erreurs_validation"]:
        phrases.append("Limites : " + " ".join(res["erreurs_validation"]))
    elif res["alertes"]:
        phrases.append(f"{len(res['alertes'])} alerte(s) de contrôle à consulter.")
    return " ".join(phrases)


SYSTEM_RESUME = ("Tu es analyste bancaire. Rédige en français 4 à 6 phrases à partir UNIQUEMENT des indicateurs fournis. "
                 "N'invente aucun chiffre. Cite les points forts, les points de vigilance et les limites de l'analyse "
                 "(données manquantes, alertes). Pas de liste, pas de titre.")


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
        (CACHE_DIR / f"{cle}.json").write_text(json.dumps(contenu, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        _log(f"⚠️ Cache non écrit : {e}")


def generer_resume_ia(res: dict) -> str:
    payload = json.dumps({
        "ratios": {k: v for k, v in res["ratios"].items() if v != "N/A"},
        "score": res["score_fondamental"], "appreciation": res["appreciation"], "confiance": res["confiance"],
        "alertes": res["alertes"][:6], "donnees_manquantes": res["erreurs_validation"],
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
        if TOUS_CHAMPS[nom][1] == "montant" and c.get("valeur") is not None:
            c["valeur_milliers"] = c["valeur"] / 1000
            c["valeur_n1_milliers"] = c["valeur_n1"] / 1000 if c.get("valeur_n1") is not None else None


def analyser_document_financier(file_bytes: bytes, utiliser_cache: bool = True, avec_resume_ia: bool = False,
                                on_progress=None, perimetre: str = "auto") -> dict:
    """
    perimetre : "auto" (consolidé s'il existe, sinon individuel), "consolide" ou "individuel".
    on_progress(message, pourcentage) : rappel optionnel pour une barre de progression (ex. st.progress).

    Unités : toutes les valeurs (« valeur ») sont en UNITÉS de la devise (ex. TND). Utiliser
    « valeur_milliers » pour afficher des kTND.
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

    cle = hashlib.sha256(file_bytes + f"{PROMPT_VERSION}|{perimetre}".encode()).hexdigest()[:24]
    cache = _lire_cache(cle) if utiliser_cache else None
    alertes_amont: list = []
    if cache:
        etape("⚡ Extraction récupérée du cache (analyse instantanée).", 80)
        champs, alertes_norm, meta = cache["champs"], cache["alertes"], cache["meta"]
    else:
        etape("⏳ 1/4 Lecture du PDF et repérage des pages financières...", 5)
        t = time.perf_counter()
        lecture = lire_pdf(file_bytes)
        durees["lecture_pdf_s"] = round(time.perf_counter() - t, 2)
        if "erreur" in lecture:
            return lecture
        if not lecture["lignes"]:
            return {"erreur": "Aucune page contenant des tableaux financiers n'a été trouvée."}
        _log(f"   {lecture['nb_pages']} pages scannées en {durees['lecture_pdf_s']}s — "
             f"{len(lecture['lignes'])} pages retenues : {[i + 1 for i in sorted(lecture['lignes'])]}")

        etape("🔎 2/4 Lecture des postes financiers par libellés...", 25)
        t = time.perf_counter()
        cands = extraire_par_libelles(lecture)
        choisis, per_cible = choisir_candidats(cands, lecture, perimetre)
        alertes_norm: list = []
        champs = {nom: _champ_vide() for nom in TOUS_CHAMPS}
        for nom, c in choisis.items():
            champs[nom] = _champ_depuis_candidat(nom, c, lecture, alertes_norm)
        _reconstituer_charges(champs, alertes_norm)
        durees["libelles_s"] = round(time.perf_counter() - t, 2)
        manquants = champs_a_completer(champs)
        _log(f"   {sum(1 for c in champs.values() if c['valeur'] is not None)}/{len(TOUS_CHAMPS)} champs trouvés "
             f"en {durees['libelles_s']}s — à compléter par l'IA : {manquants or 'rien'}")

        fournisseurs: dict = {}
        if manquants:
            etape(f"⚡ 3/4 Complément IA ({len(manquants)} champ(s) manquant(s))...", 50)
            t = time.perf_counter()
            fournisseurs = completer_par_ia(champs, manquants, lecture, alertes_amont, alertes_norm)
            _reconstituer_charges(champs, alertes_norm)
            durees["ia_s"] = round(time.perf_counter() - t, 2)
        meta = {"annee": lecture["annee"], "perimetre": per_cible, "unite_document": lecture["unite_doc"],
                "pages_utilisees": [i + 1 for i in sorted(lecture["lignes"])], "fournisseurs": fournisseurs,
                "champs_manquants_apres_ia": [k for k in TOUS_CHAMPS if champs[k]["valeur"] is None]}
        if utiliser_cache and not alertes_amont:
            _ecrire_cache(cle, {"champs": champs, "alertes": alertes_norm, "meta": meta})

    etape("🔍 4/4 Contrôles de cohérence et calcul des ratios...", 90)
    _ajouter_milliers(champs)
    for nom, c in champs.items():
        if c["valeur"] is not None:
            marque = "✓" if c["verifie"] else "?"
            _log(f"   {marque} {nom:<28} {c['valeur']:>20,.2f}  [{c['unite']}, p.{c['page']}, {c['source']}]")
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

    parser = argparse.ArgumentParser(description="Analyse fondamentale d'un rapport annuel bancaire (PDF)")
    parser.add_argument("pdf", help="chemin du fichier PDF")
    parser.add_argument("--sans-cache", action="store_true", help="ignore le cache disque")
    parser.add_argument("--resume-ia", action="store_true", help="résumé rédigé par l'IA (un appel de plus)")
    parser.add_argument("--perimetre", choices=["auto", "consolide", "individuel"], default="auto")
    parser.add_argument("--court", action="store_true", help="n'affiche pas le détail des données brutes")
    args = parser.parse_args()
    sortie = analyser_document_financier(Path(args.pdf).read_bytes(), utiliser_cache=not args.sans_cache,
                                         avec_resume_ia=args.resume_ia, perimetre=args.perimetre)
    if args.court:
        sortie.pop("donnees_brutes", None)
    print(json.dumps(sortie, ensure_ascii=False, indent=2, default=str))