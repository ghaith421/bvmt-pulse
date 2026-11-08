"""
BVMT Pulse - Générateur de rapport PDF professionnel
Compatible avec l'ancienne signature : generer_rapport_pdf(...) -> bytes
"""
import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ----------------------------------------------------------------------------
# PALETTE
# ----------------------------------------------------------------------------
BLEU_FONCE = colors.HexColor("#12355b")
BLEU = colors.HexColor("#1f4e79")
BLEU_CLAIR = colors.HexColor("#e8f0f8")
GRIS_TEXTE = colors.HexColor("#333333")
GRIS_CLAIR = colors.HexColor("#f4f6f8")
GRIS_BORDURE = colors.HexColor("#c9d1d9")
VERT = colors.HexColor("#2e8b57")
VERT_CLAIR = colors.HexColor("#e3f4ea")
ORANGE = colors.HexColor("#e08e0b")
ORANGE_CLAIR = colors.HexColor("#fdf1dc")
ROUGE = colors.HexColor("#c0392b")
ROUGE_CLAIR = colors.HexColor("#fbe5e2")

LARGEUR_UTILE = A4[0] - 2 * 36  # marges de 36 pt


# ----------------------------------------------------------------------------
# OUTILS
# ----------------------------------------------------------------------------
def _couleur_score(score):
    if score >= 55:
        return VERT, VERT_CLAIR
    if score >= 40:
        return ORANGE, ORANGE_CLAIR
    return ROUGE, ROUGE_CLAIR


def _statut_pilier(score):
    """Retourne (libellé, couleur, fond) avec 3 niveaux."""
    if score >= 60:
        return "Positif", VERT, VERT_CLAIR
    if score >= 40:
        return "Neutre", ORANGE, ORANGE_CLAIR
    return "Négatif", ROUGE, ROUGE_CLAIR


def _num(v):
    """Convertit en float si possible, sinon None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", "").replace(",", ".").strip())
        except ValueError:
            return None
    return None


def _hex(c):
    return "#%02x%02x%02x" % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))


# ----------------------------------------------------------------------------
# INTERPRÉTATIONS (texte)
# ----------------------------------------------------------------------------
def get_interpretation_synthese(score):
    if score >= 70:
        return ("Le modèle d'IA préconise un <b>ACHAT FORT</b>. La convergence des signaux techniques, "
                "fondamentaux et de sentiment soutient une dynamique haussière à court et moyen terme.")
    if score >= 55:
        return ("Le modèle préconise un <b>ACHAT MODÉRÉ</b>. La tendance globale est positive, mais "
                "nécessite une surveillance accrue des indicateurs de risque.")
    if score >= 40:
        return ("Le modèle préconise de <b>CONSERVER</b> la position. Les signaux sont mitigés ou neutres. "
                "Il est recommandé d'attendre une tendance plus claire.")
    return ("Le modèle préconise la <b>VENTE</b>. Les indicateurs techniques et fondamentaux sont "
            "défavorables, suggérant un risque de baisse.")


def get_interpretation_piliers(score_tech, score_sent, score_fond, prob_ml):
    def niveau(s, pos, neu, neg):
        return pos if s >= 60 else (neu if s >= 40 else neg)

    interp = [
        niveau(score_tech, "l'analyse technique est haussière",
               "l'analyse technique est neutre", "l'analyse technique est baissière"),
        niveau(score_sent, "le sentiment des actualités est favorable",
               "le sentiment des actualités est mitigé", "le sentiment des actualités est défavorable"),
        niveau(score_fond, "les fondamentaux de l'entreprise sont solides",
               "les fondamentaux sont corrects mais sans signal fort", "les fondamentaux présentent des fragilités"),
    ]
    if prob_ml >= 50:
        interp.append(f"le modèle prédictif anticipe une hausse ({prob_ml}% de probabilité)")
    else:
        interp.append(f"le modèle prédictif anticipe une baisse à court terme ({prob_ml}% de probabilité de hausse)")

    return f"<b>Analyse croisée :</b> {', '.join(interp)}."


def _evaluer_ratio(nom, valeur):
    """
    Retourne (seuil_reference, appreciation, couleur, fond, phrase) ou None si pas de règle.
    """
    v = _num(valeur)
    if v is None:
        return None
    n = nom.lower()
    if "roe" in n:
        ref = "> 10 % correct, > 15 % excellent"
        if v > 15:
            return ref, "Excellent", VERT, VERT_CLAIR, f"une excellente rentabilité des fonds propres (ROE de {v}%)"
        if v > 10:
            return ref, "Correct", ORANGE, ORANGE_CLAIR, f"une rentabilité correcte des fonds propres (ROE de {v}%)"
        return ref, "Faible", ROUGE, ROUGE_CLAIR, f"une rentabilité faible des fonds propres (ROE de {v}%)"
    if "npl" in n:
        ref = "< 10 % maîtrisé"
        if v < 10:
            return ref, "Maîtrisé", VERT, VERT_CLAIR, f"une bonne maîtrise du risque de crédit (Taux NPL de {v}%)"
        return ref, "Préoccupant", ROUGE, ROUGE_CLAIR, f"un niveau de créances douteuses préoccupant (Taux NPL de {v}%)"
    if "exploitation" in n:
        ref = "< 60 % efficace, > 75 % élevé"
        if v < 60:
            return ref, "Efficace", VERT, VERT_CLAIR, f"une excellente efficacité opérationnelle (coefficient d'exploitation de {v}%)"
        if v > 75:
            return ref, "Élevé", ROUGE, ROUGE_CLAIR, f"des charges de structure élevées (coefficient d'exploitation de {v}%)"
        return ref, "Moyen", ORANGE, ORANGE_CLAIR, f"un coefficient d'exploitation intermédiaire ({v}%)"
    return None


def get_interpretation_ratios(ratios):
    if not ratios:
        return "Aucun ratio financier disponible pour interprétation."
    phrases = []
    for k, v in ratios.items():
        ev = _evaluer_ratio(str(k), v)
        if ev:
            phrases.append(ev[4])
    if phrases:
        return "<b>Analyse financière :</b> la banque affiche " + ", ".join(phrases) + "."
    return "Les ratios extraits ne permettent pas une interprétation qualitative approfondie."


def _points_forts_vigilance(score_tech, score_sent, score_fond, prob_ml, ratios):
    forts, vigilance = [], []
    for nom, s in (("Analyse technique", score_tech), ("Sentiment des actualités", score_sent),
                   ("Analyse fondamentale", score_fond)):
        if s >= 60:
            forts.append(f"{nom} favorable ({s}/100)")
        elif s < 40:
            vigilance.append(f"{nom} défavorable ({s}/100)")
    if prob_ml >= 60:
        forts.append(f"Probabilité de hausse élevée selon l'IA ({prob_ml}%)")
    elif prob_ml < 50:
        vigilance.append(f"Probabilité de hausse inférieure à 50% ({prob_ml}%)")
    for k, v in (ratios or {}).items():
        ev = _evaluer_ratio(str(k), v)
        if ev:
            if ev[1] in ("Excellent", "Maîtrisé", "Efficace"):
                forts.append(f"{k} : {ev[1].lower()} ({v}%)")
            elif ev[1] in ("Faible", "Préoccupant", "Élevé"):
                vigilance.append(f"{k} : {ev[1].lower()} ({v}%)")
    # divergence entre piliers
    scores = [score_tech, score_sent, score_fond]
    if max(scores) - min(scores) >= 30:
        vigilance.append("Forte divergence entre les piliers : signal moins fiable")
    return forts, vigilance


# ----------------------------------------------------------------------------
# GRAPHIQUES (reportlab.graphics, aucune dépendance externe)
# ----------------------------------------------------------------------------
def _jauge_score(score, largeur=LARGEUR_UTILE, hauteur=62):
    d = Drawing(largeur, hauteur)
    x0, y0, h = 0, 22, 16
    zones = [(0, 40, ROUGE), (40, 55, ORANGE), (55, 70, colors.HexColor("#7bc47f")), (70, 100, VERT)]
    for a, b, c in zones:
        d.add(Rect(x0 + largeur * a / 100, y0, largeur * (b - a) / 100, h,
                   fillColor=c, strokeColor=colors.white, strokeWidth=1))
    for a, b, lab in [(0, 40, "VENTE"), (40, 55, "CONSERVER"), (55, 70, "ACHAT MODÉRÉ"), (70, 100, "ACHAT FORT")]:
        d.add(String(largeur * (a + b) / 200, y0 + 4.5, lab, fontName="Helvetica-Bold",
                     fontSize=7, fillColor=colors.white, textAnchor="middle"))
    for t in (0, 40, 55, 70, 100):
        d.add(String(largeur * t / 100, y0 - 11, str(t), fontName="Helvetica", fontSize=7,
                     fillColor=GRIS_TEXTE, textAnchor="middle"))
    s = max(0, min(100, score))
    x = largeur * s / 100
    d.add(Polygon([x - 6, y0 + h + 12, x + 6, y0 + h + 12, x, y0 + h + 1],
                  fillColor=BLEU_FONCE, strokeColor=BLEU_FONCE))
    d.add(String(x, y0 + h + 15, f"{score}", fontName="Helvetica-Bold", fontSize=9,
                 fillColor=BLEU_FONCE, textAnchor="middle"))
    return d


def _barres_piliers(valeurs, largeur=LARGEUR_UTILE):
    """valeurs : liste de (libellé, score 0-100)."""
    ligne_h = 24
    hauteur = ligne_h * len(valeurs) + 26
    d = Drawing(largeur, hauteur)
    x_label, x_bar, w_bar = 0, 150, largeur - 150 - 45
    for i, (lab, val) in enumerate(valeurs):
        y = hauteur - 26 - i * ligne_h
        coul, _ = _couleur_score(val)
        d.add(String(x_label, y + 3, lab, fontName="Helvetica", fontSize=9, fillColor=GRIS_TEXTE))
        d.add(Rect(x_bar, y, w_bar, 12, fillColor=GRIS_CLAIR, strokeColor=GRIS_BORDURE, strokeWidth=0.5))
        d.add(Rect(x_bar, y, w_bar * max(0, min(100, val)) / 100, 12, fillColor=coul, strokeColor=None))
        d.add(String(x_bar + w_bar + 8, y + 3, f"{val}", fontName="Helvetica-Bold", fontSize=9,
                     fillColor=GRIS_TEXTE))
    # ligne de neutralité à 50
    x50 = x_bar + w_bar * 0.5
    d.add(Line(x50, 4, x50, hauteur - 12, strokeColor=GRIS_TEXTE, strokeDashArray=[2, 2], strokeWidth=0.6))
    d.add(String(x50, hauteur - 9, "50", fontName="Helvetica", fontSize=7,
                 fillColor=GRIS_TEXTE, textAnchor="middle"))
    return d


# ----------------------------------------------------------------------------
# EN-TÊTE / PIED DE PAGE
# ----------------------------------------------------------------------------
class _NumberedCanvas(canvas.Canvas):
    """Canvas permettant d'afficher 'Page X / Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._dessiner_pied(total)
            super().showPage()
        super().save()

    def _dessiner_pied(self, total):
        w, h = A4
        self.saveState()
        # bandeau supérieur
        self.setFillColor(BLEU_FONCE)
        self.rect(0, h - 10 * mm, w, 10 * mm, stroke=0, fill=1)
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(36, h - 6.5 * mm, "BVMT PULSE")
        self.setFont("Helvetica", 8)
        self.drawRightString(w - 36, h - 6.5 * mm, "Rapport d'analyse financière assistée par IA")
        # pied de page
        self.setStrokeColor(GRIS_BORDURE)
        self.setLineWidth(0.5)
        self.line(36, 18 * mm, w - 36, 18 * mm)
        self.setFillColor(colors.HexColor("#777777"))
        self.setFont("Helvetica", 7.5)
        self.drawString(36, 13 * mm, "Document confidentiel - Ne constitue pas un conseil en investissement.")
        self.drawRightString(w - 36, 13 * mm, f"Page {self._pageNumber} / {total}")
        self.restoreState()


# ----------------------------------------------------------------------------
# GÉNÉRATION DU RAPPORT
# ----------------------------------------------------------------------------
def generer_rapport_pdf(banque, cours, score_final, recommandation, score_tech, score_sent,
                        score_fond, prob_ml, ratios=None, poids=None, auteur="BVMT Pulse"):
    """
    Génère un rapport PDF complet en mémoire et le retourne en bytes.

    poids (optionnel) : dict ex. {"Technique": 0.3, "Sentiment": 0.2, "Fondamental": 0.3, "IA": 0.2}
    """
    buffer = io.BytesIO()
    maintenant = datetime.now()
    ref_rapport = f"BP-{maintenant.strftime('%Y%m%d-%H%M')}"

    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=16 * mm + 10, bottomMargin=24 * mm,
        title=f"Rapport d'analyse - {banque}", author=auteur, subject="Analyse boursière BVMT",
    )

    base = getSampleStyleSheet()
    st_titre = ParagraphStyle("Titre", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                              textColor=colors.white, alignment=TA_LEFT, leading=24, spaceAfter=2)
    st_sous_titre = ParagraphStyle("SousTitre", parent=base["Normal"], fontName="Helvetica", fontSize=11,
                                   textColor=colors.HexColor("#d6e4f2"), leading=14)
    st_meta = ParagraphStyle("Meta", parent=base["Normal"], fontName="Helvetica", fontSize=8.5,
                             textColor=colors.HexColor("#d6e4f2"), alignment=2, leading=12)
    st_h = ParagraphStyle("H", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                          textColor=BLEU, spaceBefore=14, spaceAfter=4)
    st_normal = ParagraphStyle("N", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
                               textColor=GRIS_TEXTE, leading=13.5, alignment=TA_JUSTIFY)
    st_cell = ParagraphStyle("Cell", parent=st_normal, alignment=TA_CENTER, fontSize=9.5, leading=12)
    st_cell_left = ParagraphStyle("CellL", parent=st_cell, alignment=TA_LEFT)
    st_head = ParagraphStyle("Head", parent=st_cell, fontName="Helvetica-Bold", textColor=colors.white)
    st_kpi_lab = ParagraphStyle("KpiL", parent=st_cell, fontSize=8, textColor=colors.HexColor("#666666"))
    st_interp = ParagraphStyle("Interp", parent=st_normal, fontName="Helvetica-Oblique", fontSize=9.5,
                               backColor=BLEU_CLAIR, borderPadding=(7, 8, 7, 8), spaceBefore=10, spaceAfter=14,
                               leftIndent=8, rightIndent=8)
    st_petit = ParagraphStyle("Petit", parent=st_normal, fontSize=7.5, leading=10,
                              textColor=colors.HexColor("#666666"))

    def titre_section(texte):
        return [Paragraph(texte, st_h),
                HRFlowable(width="100%", thickness=1.2, color=BLEU, spaceBefore=0, spaceAfter=6)]

    el = []

    # ---------------- Bandeau titre ----------------
    entete = Table(
        [[Paragraph("RAPPORT D'ANALYSE BOURSIÈRE", st_titre),
          Paragraph(f"Réf. {ref_rapport}<br/>{maintenant.strftime('%d/%m/%Y à %H:%M')}", st_meta)],
         [Paragraph(f"Valeur analysée : <b>{escape(str(banque))}</b>", st_sous_titre), ""]],
        colWidths=[LARGEUR_UTILE * 0.68, LARGEUR_UTILE * 0.32],
    )
    entete.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLEU),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, 0), 12), ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("SPAN", (0, 1), (1, 1)),
        ("LINEBELOW", (0, -1), (-1, -1), 3, ORANGE),
    ]))
    el += [entete, Spacer(1, 12)]

    # ---------------- 1. Synthèse ----------------
    el += titre_section("1. Synthèse de la décision")
    coul, fond = _couleur_score(score_final)
    cours_txt = f"{cours:.2f} TND" if _num(cours) is not None else "N/D"

    def kpi(valeur, libelle, couleur=BLEU_FONCE, bg=GRIS_CLAIR, taille=17):
        v = ParagraphStyle("kv", parent=st_cell, fontName="Helvetica-Bold", fontSize=taille,
                           leading=taille + 4, textColor=couleur)
        return [Paragraph(valeur, v), Paragraph(libelle, st_kpi_lab)], bg

    k1, bg1 = kpi(cours_txt, "DERNIER COURS")
    k2, bg2 = kpi(f"{score_final}<font size=10>/100</font>", "SCORE FINAL PONDÉRÉ", coul, fond)
    k3, bg3 = kpi(escape(str(recommandation)).upper(), "RECOMMANDATION", coul, fond, taille=13)
    w = LARGEUR_UTILE / 3
    t_kpi = Table([[k1, k2, k3]], colWidths=[w, w, w], rowHeights=[56])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), bg1), ("BACKGROUND", (1, 0), (1, 0), bg2), ("BACKGROUND", (2, 0), (2, 0), bg3),
        ("BOX", (0, 0), (0, 0), 0.6, GRIS_BORDURE), ("BOX", (1, 0), (1, 0), 0.6, coul), ("BOX", (2, 0), (2, 0), 0.6, coul),
        ("LINEABOVE", (0, 0), (0, 0), 3, BLEU), ("LINEABOVE", (1, 0), (2, 0), 3, coul),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    # séparation visuelle entre cartes
    t_kpi._argW = [w, w, w]
    el += [t_kpi, Spacer(1, 10), _jauge_score(score_final),
           Paragraph(get_interpretation_synthese(score_final), st_interp)]

    # ---------------- 2. Piliers ----------------
    el += titre_section("2. Détail des piliers d'analyse")
    piliers = [
        ("Analyse technique", score_tech),
        ("Analyse sentiment (news)", score_sent),
        ("Analyse fondamentale", score_fond),
    ]
    el.append(_barres_piliers(piliers + [("Probabilité IA (lendemain)", prob_ml)]))
    el.append(Spacer(1, 6))

    lignes = [[Paragraph(h, st_head) for h in ("Pilier", "Score", "Statut", "Lecture")]]
    styles_p = [
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRIS_BORDURE),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    lectures = {
        "Analyse technique": "Tendance, momentum et indicateurs de prix",
        "Analyse sentiment (news)": "Tonalité des actualités récentes",
        "Analyse fondamentale": "Solidité financière de l'entreprise",
    }
    for i, (nom, sc) in enumerate(piliers, start=1):
        lib, c, bg = _statut_pilier(sc)
        lignes.append([Paragraph(nom, st_cell_left), Paragraph(f"<b>{sc}/100</b>", st_cell),
                       Paragraph(f"<font color='{_hex(c)}'><b>{lib}</b></font>", st_cell),
                       Paragraph(lectures[nom], st_cell_left)])
        styles_p.append(("BACKGROUND", (2, i), (2, i), bg))
        if i % 2 == 0:
            styles_p.append(("BACKGROUND", (0, i), (1, i), GRIS_CLAIR))
            styles_p.append(("BACKGROUND", (3, i), (3, i), GRIS_CLAIR))
    i = len(lignes)
    c = VERT if prob_ml >= 50 else ROUGE
    bg = VERT_CLAIR if prob_ml >= 50 else ROUGE_CLAIR
    lignes.append([Paragraph("Probabilité IA (lendemain)", st_cell_left), Paragraph(f"<b>{prob_ml}%</b>", st_cell),
                   Paragraph(f"<font color='{_hex(c)}'><b>{'Hausse' if prob_ml >= 50 else 'Baisse'}</b></font>", st_cell),
                   Paragraph("Probabilité de hausse du cours à J+1", st_cell_left)])
    styles_p.append(("BACKGROUND", (2, i), (2, i), bg))
    t_p = Table(lignes, colWidths=[LARGEUR_UTILE * 0.30, LARGEUR_UTILE * 0.15,
                                   LARGEUR_UTILE * 0.15, LARGEUR_UTILE * 0.40], repeatRows=1)
    t_p.setStyle(TableStyle(styles_p))
    el.append(t_p)
    el.append(Paragraph(get_interpretation_piliers(score_tech, score_sent, score_fond, prob_ml), st_interp))

    # Pondération (optionnelle)
    if poids:
        txt = " &nbsp;|&nbsp; ".join(f"{escape(str(k))} : <b>{round(float(v) * 100)}%</b>" for k, v in poids.items())
        el.append(Paragraph(f"<b>Pondération du score final :</b> {txt}", st_petit))
        el.append(Spacer(1, 6))

    # ---------------- 3. Points forts / vigilance ----------------
    forts, vigilance = _points_forts_vigilance(score_tech, score_sent, score_fond, prob_ml, ratios)
    st_li = ParagraphStyle("Li", parent=st_normal, alignment=TA_LEFT, fontSize=9, leading=12.5)

    def colonne(titre, items, couleur, vide):
        contenu = [Paragraph(f"<font color='{_hex(couleur)}'><b>{titre}</b></font>", st_li)]
        if items:
            contenu += [Paragraph(f"&bull; {escape(x)}", st_li) for x in items]
        else:
            contenu.append(Paragraph(f"<i>{vide}</i>", st_li))
        return contenu

    bloc = Table([[colonne("POINTS FORTS", forts, VERT, "Aucun point fort marqué."),
                   colonne("POINTS DE VIGILANCE", vigilance, ROUGE, "Aucun point de vigilance majeur.")]],
                 colWidths=[LARGEUR_UTILE / 2] * 2)
    bloc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), VERT_CLAIR), ("BACKGROUND", (1, 0), (1, 0), ROUGE_CLAIR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (0, 0), 2, colors.white),
    ]))
    el.append(KeepTogether(titre_section("3. Points forts et points de vigilance") + [bloc, Spacer(1, 6)]))

    # ---------------- 4. Ratios ----------------
    if ratios:
        lignes = [[Paragraph(h, st_head) for h in ("Indicateur", "Valeur", "Référence", "Appréciation")]]
        styles_r = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a7ebb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, GRIS_BORDURE),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        for idx, (k, v) in enumerate(ratios.items(), start=1):
            nv = _num(v)
            val_aff = f"{nv:.2f} %" if nv is not None else "N/D"
            ev = _evaluer_ratio(str(k), v)
            if ev:
                ref, lib, c, bg, _ = ev
                appr = Paragraph(f"<font color='{_hex(c)}'><b>{lib}</b></font>", st_cell)
                styles_r.append(("BACKGROUND", (3, idx), (3, idx), bg))
            else:
                ref, appr = "-", Paragraph("-", st_cell)
            lignes.append([Paragraph(escape(str(k)), st_cell_left), Paragraph(f"<b>{val_aff}</b>", st_cell),
                           Paragraph(ref, st_cell), appr])
            if idx % 2 == 0:
                styles_r.append(("BACKGROUND", (0, idx), (2, idx), GRIS_CLAIR))
        t_r = Table(lignes, colWidths=[LARGEUR_UTILE * 0.34, LARGEUR_UTILE * 0.17,
                                       LARGEUR_UTILE * 0.29, LARGEUR_UTILE * 0.20], repeatRows=1)
        t_r.setStyle(TableStyle(styles_r))
        el.append(KeepTogether(titre_section("4. Ratios financiers (extraits des états financiers)") + [
            t_r, Paragraph(get_interpretation_ratios(ratios), st_interp)]))

    # ---------------- 5. Méthodologie & avertissement ----------------
    el.append(KeepTogether(titre_section("5. Méthodologie et avertissement") + [
        Paragraph(
            "Le score final est une moyenne pondérée de trois piliers (analyse technique, sentiment des actualités, "
            "analyse fondamentale) complétée par une prédiction issue d'un modèle d'apprentissage automatique. "
            "Barème : <b>&ge; 70</b> achat fort, <b>55-69</b> achat modéré, <b>40-54</b> conserver, <b>&lt; 40</b> vente. "
            "Les ratios sont comparés à des seuils de référence génériques pour le secteur bancaire.", st_normal),
        Spacer(1, 8),
        Table([[Paragraph(
            "<b>Avertissement :</b> ce rapport a été généré automatiquement par l'IA de BVMT Pulse à des fins "
            "d'information uniquement. Il ne constitue ni un conseil en investissement, ni une incitation à "
            "acheter ou vendre des instruments financiers. Les performances passées ne préjugent pas des "
            "performances futures. L'investisseur reste seul responsable de ses décisions.", st_petit)]],
              colWidths=[LARGEUR_UTILE],
              style=TableStyle([("BOX", (0, 0), (-1, -1), 0.6, GRIS_BORDURE), ("BACKGROUND", (0, 0), (-1, -1), GRIS_CLAIR),
                                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)])),
    ]))

    doc.build(el, canvasmaker=_NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    pdf = generer_rapport_pdf(
        "Attijari Bank", 52.40, 58, "Achat modéré", 65, 48, 72, 55,
        ratios={"ROE (%)": 16.27, "Taux NPL (%)": 7.8, "Coeff. Exploitation (%)": 48.5, "Ratio de solvabilité (%)": "N/A"},
        poids={"Technique": 0.30, "Sentiment": 0.20, "Fondamental": 0.30, "IA": 0.20},
    )
    with open("exemple_rapport.pdf", "wb") as f:
        f.write(pdf)
    print("PDF généré :", len(pdf), "octets")