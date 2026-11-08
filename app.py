import html
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

# --- SÉCURITÉ POUR STREAMLIT CLOUD ---
@st.cache_resource
def installer_navigateur():
    # On vérifie si le dossier de cache de Playwright existe
    if not os.path.exists(os.path.expanduser("~/.cache/ms-playwright")):
        print("⏳ Installation de Chromium pour Playwright...")
        os.system("playwright install chromium")
        print("✅ Installation terminée.")

installer_navigateur()
# --------------------------------------
# Importation des modules
from modules.market_data import get_historique_banque, BANQUES_ILBOURSA
from modules.technical_analysis import calculer_indicateurs, calculer_score_technique, predire_tendance_ml
from modules.news_sentiment import collecter_news, analyser_sentiment_ia, score_sentiment_global
from modules.document_ai import analyser_document_financier
from modules.report_generator import generer_rapport_pdf
from modules.backtesting import simuler_strategie

st.set_page_config(page_title="BVMT Pulse | Terminal", page_icon="⚡", layout="wide")

# ==========================================
# PALETTE
# ==========================================
MINT = "#00F5A0"
CYAN = "#00D9F5"
VIOLET = "#8B5CF6"
PINK = "#FF2E93"
AMBER = "#FFB020"
TEXT = "#EAF0FF"
MUTED = "#8B96B8"

# ==========================================
# 1. CSS PREMIUM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #EAF0FF; }
    h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }

    /* ---------- Fond animé ---------- */
    .stApp {
        background:
            radial-gradient(900px 500px at 8% -5%, rgba(139, 92, 246, 0.28), transparent 60%),
            radial-gradient(800px 500px at 95% 5%, rgba(0, 217, 245, 0.20), transparent 60%),
            radial-gradient(700px 500px at 50% 110%, rgba(255, 46, 147, 0.14), transparent 60%),
            #060914;
        background-attachment: fixed;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 34px 40px;
        margin-bottom: 22px;
        background: linear-gradient(120deg, rgba(139,92,246,0.25), rgba(0,217,245,0.12) 55%, rgba(0,245,160,0.14));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.07) 50%, transparent 70%);
        transform: translateX(-100%);
        animation: sweep 6s ease-in-out infinite;
        pointer-events: none;
    }
    @keyframes sweep { 0% { transform: translateX(-100%); } 55%, 100% { transform: translateX(100%); } }

    .hero-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 18px; }
    .hero h1 {
        margin: 0; padding: 0;
        font-size: 2.6rem; font-weight: 800; line-height: 1.1;
        background: linear-gradient(90deg, #00F5A0, #00D9F5 45%, #FF2E93);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 8px 0 0; color: #AEB8D8; font-size: 1.02rem; }

    .live-pill {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 999px;
        background: rgba(0,245,160,0.12); border: 1px solid rgba(0,245,160,0.45);
        color: #00F5A0; font-weight: 600; font-size: 0.82rem;
    }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #00F5A0; animation: pulse 1.6s infinite; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,245,160,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0,245,160,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,245,160,0); }
    }

    .hero-price { text-align: right; }
    .hero-price .asset { color: #AEB8D8; font-weight: 600; font-size: 0.95rem; }
    .hero-price .price { font-family: 'Sora', sans-serif; font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
    .chip { display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.88rem; }
    .chip.up { background: rgba(0,245,160,0.15); color: #00F5A0; }
    .chip.down { background: rgba(255,46,147,0.15); color: #FF2E93; }

    /* ---------- Cartes métriques ---------- */
    div[data-testid="stMetric"] {
        position: relative;
        background: linear-gradient(160deg, rgba(24,30,56,0.85), rgba(14,18,36,0.85));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        overflow: hidden;
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
        background: linear-gradient(180deg, #00F5A0, #00D9F5, #8B5CF6);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0,217,245,0.6);
        box-shadow: 0 14px 40px rgba(0,217,245,0.18);
    }
    div[data-testid="stMetricLabel"] p { color: #8B96B8; font-weight: 600; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.9rem; }

    /* ---------- Boutons ---------- */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #8B5CF6 0%, #00D9F5 55%, #00F5A0 100%);
        background-size: 200% 100%;
        color: #04101A;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        width: 100%;
        padding: 12px 16px;
        transition: background-position .5s ease, transform .2s ease, box-shadow .2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-position: 100% 0;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0,217,245,0.45);
        color: #04101A;
    }
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }

    /* ---------- Onglets en pilules ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; padding: 6px; border-radius: 16px;
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 18px; border-radius: 12px;
        background: transparent; color: #8B96B8; font-weight: 600;
        transition: all .25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #fff; background: rgba(255,255,255,0.06); }
    .stTabs [aria-selected="true"] {
        color: #04101A !important;
        background: linear-gradient(90deg, #00F5A0, #00D9F5) !important;
        box-shadow: 0 6px 20px rgba(0,245,160,0.35);
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0F24 0%, #070A1A 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .brand-logo {
        width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-size: 1.3rem;
        background: linear-gradient(135deg, #8B5CF6, #00D9F5); box-shadow: 0 6px 20px rgba(139,92,246,0.5);
    }
    .brand-name { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.15rem; }
    .brand-sub { color: #8B96B8; font-size: 0.78rem; }

    .weights { display: flex; height: 12px; border-radius: 999px; overflow: hidden; margin: 6px 0 10px; background: rgba(255,255,255,0.06); }
    .weights span { display: block; height: 100%; }
    .legend { display: flex; justify-content: space-between; font-size: 0.78rem; color: #AEB8D8; }

    /* ---------- Titres de section ---------- */
    .section-title { font-family: 'Sora', sans-serif; font-size: 1.35rem; font-weight: 700; margin: 6px 0 14px; }
    .section-title small { color: #8B96B8; font-weight: 500; font-size: 0.9rem; margin-left: 8px; }

    /* ---------- Cartes news ---------- */
    .news-card {
        border-radius: 16px; padding: 16px 20px; margin-bottom: 12px;
        background: rgba(18,24,46,0.85); border: 1px solid rgba(255,255,255,0.07);
        border-left: 5px solid var(--c);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .news-card:hover { transform: translateX(4px); box-shadow: 0 8px 28px rgba(0,0,0,0.4); }
    .news-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
    .news-title { font-weight: 600; font-size: 1.02rem; line-height: 1.4; }
    .news-score {
        flex-shrink: 0; padding: 4px 12px; border-radius: 999px; font-weight: 800; font-size: 0.85rem;
        color: #04101A; background: var(--c);
    }
    .news-just { color: #AEB8D8; font-size: 0.92rem; margin-top: 8px; }

    /* ---------- Bannières ---------- */
    .banner {
        border-radius: 18px; padding: 18px 22px; margin: 10px 0;
        border: 1px solid var(--c); background: linear-gradient(120deg, rgba(0,0,0,0.25), rgba(0,0,0,0));
        box-shadow: 0 0 30px color-mix(in srgb, var(--c) 25%, transparent);
    }
    .banner .b-label { color: #AEB8D8; font-size: 0.82rem; font-weight: 600; }
    .banner .b-value { font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 800; color: var(--c); }

    /* ---------- Divers ---------- */
    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(0,217,245,0.4); border-radius: 18px; background: rgba(0,217,245,0.04);
    }
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
    div[data-testid="stAlert"] { border-radius: 14px; }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(#8B5CF6, #00D9F5); border-radius: 8px; }

    @media (max-width: 768px) {
        .hero { padding: 22px; }
        .hero h1 { font-size: 1.8rem; }
        .hero-price { text-align: left; }
    }
    @media (prefers-reduced-motion: reduce) {
        .hero::after, .live-dot { animation: none; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. OUTILS DE DESIGN
# ==========================================
def couleur_score(s):
    if s >= 70: return MINT
    if s >= 55: return CYAN
    if s >= 40: return AMBER
    return PINK


def style_fig(fig, height=500):
    fig.update_layout(
        height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, family="Inter"), hovermode="x unified",
        hoverlabel=dict(bgcolor="#10162E", font_color=TEXT, bordercolor=CYAN),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                     showspikes=True, spikecolor=CYAN, spikethickness=1, spikedash="dot")
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    return fig


def jauge(valeur, titre, height=240):
    c = couleur_score(valeur)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=valeur,
        number=dict(font=dict(size=44, color=c, family="Sora"), suffix=""),
        title=dict(text=titre, font=dict(size=15, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=c, thickness=0.28),
            bgcolor="rgba(255,255,255,0.04)", borderwidth=0,
            steps=[dict(range=[0, 40], color="rgba(255,46,147,0.18)"),
                   dict(range=[40, 55], color="rgba(255,176,32,0.16)"),
                   dict(range=[55, 70], color="rgba(0,217,245,0.16)"),
                   dict(range=[70, 100], color="rgba(0,245,160,0.18)")],
        )))
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT),
                      margin=dict(l=20, r=20, t=50, b=10))
    return fig


def titre_section(texte, sous_titre=""):
    sub = f"<small>{html.escape(sous_titre)}</small>" if sous_titre else ""
    st.markdown(f"<div class='section-title'>{html.escape(texte)}{sub}</div>", unsafe_allow_html=True)


def banniere(label, valeur, couleur):
    st.markdown(
        f"<div class='banner' style='--c:{couleur}'><div class='b-label'>{html.escape(label)}</div>"
        f"<div class='b-value'>{html.escape(valeur)}</div></div>", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def charger_donnees(symbol):
    return get_historique_banque(symbol, length=180)


@st.cache_data(show_spinner=False)
def traiter_pdf_cached(contenu):
    return analyser_document_financier(contenu)


# ==========================================
# 3. ÉTAT DE SESSION
# ==========================================
defaults = {
    "score_tech": 50, "score_sent": 50, "score_fond": 50, "prob_ml": 50,
    "resultat_fond": None, "dernier_cours": 0.0, "current_bank": None, "news_res": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 4. BARRE LATÉRALE
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class='brand'>
        <div class='brand-logo'>⚡</div>
        <div><div class='brand-name'>BVMT Pulse</div><div class='brand-sub'>Terminal d'analyse IA</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### 🏦 Actif")
    banque_selectionnee = st.selectbox("Sélectionnez un actif :", options=list(BANQUES_ILBOURSA.keys()),
                                       label_visibility="collapsed")
    ticker = BANQUES_ILBOURSA[banque_selectionnee]

    st.markdown("---")
    st.markdown("##### ⚖️ Pondération des signaux")
    poids_tech = st.slider("Technique (%)", 0, 100, 40)
    if poids_tech < 100:
        poids_sent = st.slider("Sentiment (%)", 0, 100 - poids_tech, min(30, 100 - poids_tech))
    else:
        poids_sent = 0
    poids_fond = 100 - poids_tech - poids_sent

    st.markdown(f"""
    <div class='weights'>
        <span style='width:{poids_tech}%; background:{CYAN}'></span>
        <span style='width:{poids_sent}%; background:{VIOLET}'></span>
        <span style='width:{poids_fond}%; background:{MINT}'></span>
    </div>
    <div class='legend'>
        <span style='color:{CYAN}'>● Tech {poids_tech}%</span>
        <span style='color:{VIOLET}'>● Sent. {poids_sent}%</span>
        <span style='color:{MINT}'>● Fond. {poids_fond}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Actualiser les cours en direct"):
        st.cache_data.clear()
        st.rerun()

# ==========================================
# 5. DONNÉES + HERO
# ==========================================
with st.spinner("⏳ Connexion aux serveurs BVMT..."):
    df = charger_donnees(ticker)

if not df.empty:
    df = calculer_indicateurs(df)
    st.session_state.score_tech = calculer_score_technique(df)
    st.session_state.prob_ml = predire_tendance_ml(df)
    st.session_state.dernier_cours = float(df['Close'].iloc[-1])
    variation = float(df['Close'].pct_change().iloc[-1] * 100)
else:
    variation = 0.0

classe = "up" if variation >= 0 else "down"
fleche = "▲" if variation >= 0 else "▼"
prix_html = f"{st.session_state.dernier_cours:.2f} TND" if not df.empty else "—"

st.markdown(f"""
<div class='hero'>
    <div class='hero-top'>
        <div>
            <span class='live-pill'><span class='live-dot'></span>Marché BVMT en direct</span>
            <h1 style='margin-top:14px;'>BVMT Pulse Terminal</h1>
            <p>L'intelligence artificielle au service de la Bourse de Tunis.</p>
        </div>
        <div class='hero-price'>
            <div class='asset'>{html.escape(banque_selectionnee)}</div>
            <div class='price'>{prix_html}</div>
            <span class='chip {classe}'>{fleche} {abs(variation):.2f}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- ONGLETS ---
tab_technique, tab_news, tab_docs, tab_synthese, tab_backtest = st.tabs([
    "📊 Graphiques", "📰 News IA", "📄 Rapports & Fonda", "🎯 Synthèse", "⏱️ Backtesting"
])

# ==========================================
# ONGLET 1 : ANALYSE TECHNIQUE
# ==========================================
with tab_technique:
    titre_section("Analyse de marché", banque_selectionnee)

    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Score Technique", f"{st.session_state.score_tech}/100")
        c2.metric("Dernier Cours", f"{st.session_state.dernier_cours:.2f} TND", delta=f"{variation:.2f}%")
        c3.metric("🤖 Probabilité IA", f"{st.session_state.prob_ml}%")
        c4.metric("📉 RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")

        st.write("")
        a_volume = 'Volume' in df.columns
        nb_rows = 4 if a_volume else 3
        hauteurs = [0.5, 0.15, 0.17, 0.18] if a_volume else [0.6, 0.2, 0.2]
        fig = make_subplots(rows=nb_rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=hauteurs)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            increasing=dict(line=dict(color=MINT), fillcolor="rgba(0,245,160,0.55)"),
            decreasing=dict(line=dict(color=PINK), fillcolor="rgba(255,46,147,0.55)"),
            name="Cours"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color=AMBER, width=1.8), name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color=CYAN, width=1.8), name='SMA 50'), row=1, col=1)

        row = 2
        if a_volume:
            couleurs_vol = [MINT if c >= o else PINK for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=couleurs_vol, opacity=0.6, name='Volume'), row=row, col=1)
            row += 1

        # RSI avec zones colorées
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,46,147,0.08)", line_width=0, row=row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,245,160,0.08)", line_width=0, row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color=VIOLET, width=2), name='RSI'), row=row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=PINK, row=row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=MINT, row=row, col=1)
        row += 1

        # MACD + histogramme
        hist = df['MACD'] - df['Signal']
        fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=[MINT if h >= 0 else PINK for h in hist],
                             opacity=0.45, name='Histogramme'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color=CYAN, width=1.8), name='MACD'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color=AMBER, width=1.8), name='Signal'), row=row, col=1)

        style_fig(fig, height=820)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=nb_rows, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Aucune donnée disponible pour cet actif. Cliquez sur « Actualiser les cours en direct ».")

# ==========================================
# ONGLET 2 : NEWS & SENTIMENT
# ==========================================
with tab_news:
    titre_section("Flux d'actualités", banque_selectionnee)

    if st.button("🔍 Scanner les News en temps réel"):
        with st.spinner("L'IA analyse la presse..."):
            arts = collecter_news(banque_selectionnee)
            res = analyser_sentiment_ia(arts)
            st.session_state.score_sent = score_sentiment_global(res)
            st.session_state.news_res = res

    if st.session_state.news_res:
        col_g, col_n = st.columns([1, 2])
        with col_g:
            st.plotly_chart(jauge(st.session_state.score_sent, "Sentiment global (7 jours)"), use_container_width=True)
        with col_n:
            for a in st.session_state.news_res:
                score = a.get("score_sentiment", 50)
                c = MINT if score >= 60 else (PINK if score <= 40 else AMBER)
                st.markdown(f"""
                <div class='news-card' style='--c:{c}'>
                    <div class='news-head'>
                        <div class='news-title'>{html.escape(str(a.get('titre', '')))}</div>
                        <div class='news-score'>{score}</div>
                    </div>
                    <div class='news-just'>{html.escape(str(a.get('justification', '')))}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Lancez un scan pour obtenir le sentiment de la presse sur cet actif.")

# ==========================================
# ONGLET 3 : FONDAMENTAL (DOCUMENTS)
# ==========================================
with tab_docs:
    titre_section("Audit financier", "extraction des ratios par IA")

    if st.session_state.current_bank != banque_selectionnee:
        st.session_state.resultat_fond = None
        st.session_state.score_fond = 50
        st.session_state.current_bank = banque_selectionnee

    uploaded_file = st.file_uploader("Déposez un rapport annuel (PDF)", type=["pdf", "png", "jpg"])

    if uploaded_file is not None:
        if st.button("📄 Lancer l'extraction des ratios"):
            st.session_state.resultat_fond = None
            st.session_state.score_fond = 50

            with st.spinner("⚡ Lecture ultra-rapide du PDF..."):
                file_bytes = uploaded_file.read()
                resultat = traiter_pdf_cached(file_bytes)

                if "erreur" in resultat:
                    st.error(f"Erreur : {resultat['erreur']}")
                else:
                    st.session_state.score_fond = resultat["score_fondamental"]
                    st.session_state.resultat_fond = resultat

    if st.session_state.resultat_fond is not None:
        resultat = st.session_state.resultat_fond
        st.success("✅ Analyse terminée !")

        if resultat.get("resume_ia"):
            st.info(f"**🧠 Analyse qualitative de l'IA :**\n\n{resultat['resume_ia']}")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(jauge(st.session_state.score_fond, "Score fondamental", height=260), use_container_width=True)
            if st.session_state.score_fond >= 65: st.success("Solidité : **Bonne**")
            elif st.session_state.score_fond >= 45: st.warning("Solidité : **Moyenne**")
            else: st.error("Solidité : **Risquée**")

        with c2:
            st.markdown("#### 📊 Ratios clés extraits")
            ratios = resultat.get("ratios", {})

            def fmt(val): return f"{val}%" if val != "N/A" else "N/A"

            r1, r2, r3 = st.columns(3)
            r1.metric("ROE", fmt(ratios.get('ROE (%)', 'N/A')))
            r2.metric("ROA", fmt(ratios.get('ROA (%)', 'N/A')))
            r3.metric("Marge Nette", fmt(ratios.get('Marge Nette (%)', 'N/A')))

            st.write("")
            r4, r5, r6 = st.columns(3)
            r4.metric("Coeff. Exploit.", fmt(ratios.get('Coeff. Exploitation (%)', 'N/A')))
            r5.metric("Taux NPL", fmt(ratios.get('Taux NPL (%)', 'N/A')))
            r6.metric("Taux de Couverture", fmt(ratios.get('Taux de Couverture (%)', 'N/A')))

            st.write("")
            r7, _, _ = st.columns(3)
            r7.metric("LDR", fmt(ratios.get('LDR (%)', 'N/A')))

        if resultat.get("erreurs_validation"):
            st.markdown("### ⚠️ Alertes de cohérence")
            for err in resultat["erreurs_validation"]: st.warning(err)

        st.markdown("### 🔍 Détail des données extraites")
        donnees = resultat["donnees_brutes"]

        tableau_data = []
        for k, v in donnees.items():
            if k == "resume_ia": continue
            tableau_data.append({
                "Indicateur": k.replace("_", " ").capitalize(),
                "Valeur (kTND)": v.get("valeur") if isinstance(v, dict) else v,
                "Page PDF": v.get("page") if isinstance(v, dict) else "-"
            })

        st.dataframe(pd.DataFrame(tableau_data), use_container_width=True, hide_index=True)

# ==========================================
# ONGLET 4 : SYNTHÈSE FINALE
# ==========================================
with tab_synthese:
    titre_section("Verdict de l'IA", banque_selectionnee)

    score_final = int((st.session_state.score_tech * (poids_tech / 100)) +
                      (st.session_state.score_sent * (poids_sent / 100)) +
                      (st.session_state.score_fond * (poids_fond / 100)))

    if score_final >= 70: rec_txt, rec_col, rec_icon = "ACHAT FORT", MINT, "🟢"
    elif score_final >= 55: rec_txt, rec_col, rec_icon = "ACHAT MODÉRÉ", CYAN, "🟢"
    elif score_final >= 40: rec_txt, rec_col, rec_icon = "CONSERVER", AMBER, "🟡"
    else: rec_txt, rec_col, rec_icon = "VENTE", PINK, "🔴"

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.plotly_chart(jauge(score_final, "SCORE FINAL", height=300), use_container_width=True)
        banniere("Recommandation", f"{rec_icon} {rec_txt}", rec_col)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tech.", f"{st.session_state.score_tech}")
        m2.metric("Sent.", f"{st.session_state.score_sent}")
        m3.metric("Fond.", f"{st.session_state.score_fond}")

        st.write("")
        if st.button("📑 Générer le rapport d'investissement"):
            ratios_pour_pdf = st.session_state.resultat_fond.get("ratios") if st.session_state.resultat_fond else None
            pdf_bytes = generer_rapport_pdf(banque_selectionnee, st.session_state.dernier_cours, score_final, rec_txt,
                                            st.session_state.score_tech, st.session_state.score_sent,
                                            st.session_state.score_fond, st.session_state.prob_ml, ratios_pour_pdf)
            st.download_button("💾 Télécharger le PDF", data=pdf_bytes,
                               file_name=f"Rapport_{banque_selectionnee}.pdf", mime="application/pdf")

    with col2:
        categories = ['Technique', 'Sentiment', 'Fondamental', 'Probabilité IA', 'ROE']
        roe_val = 50
        if st.session_state.resultat_fond is not None:
            roe_raw = st.session_state.resultat_fond.get("ratios", {}).get("ROE (%)", 50)
            roe_val = roe_raw if isinstance(roe_raw, (int, float)) else 50
            roe_val = min(100, max(0, roe_val))

        valeurs_banque = [st.session_state.score_tech, st.session_state.score_sent,
                          st.session_state.score_fond, st.session_state.prob_ml, roe_val]
        valeurs_secteur = [55, 50, 55, 50, 50]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=valeurs_secteur + valeurs_secteur[:1], theta=categories + categories[:1],
                                            fill='toself', name='Moyenne secteur',
                                            line=dict(color=VIOLET, width=2), fillcolor="rgba(139,92,246,0.18)"))
        fig_radar.add_trace(go.Scatterpolar(r=valeurs_banque + valeurs_banque[:1], theta=categories + categories[:1],
                                            fill='toself', name=banque_selectionnee,
                                            line=dict(color=MINT, width=3), fillcolor="rgba(0,245,160,0.28)"))
        fig_radar.update_layout(
            polar=dict(bgcolor='rgba(20,26,41,0.35)',
                       radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)',
                                       tickfont=dict(color=MUTED)),
                       angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color=TEXT, size=13))),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color=TEXT), showlegend=True, height=520,
            margin=dict(l=60, r=60, t=40, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5))
        st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# ONGLET 5 : BACKTESTING
# ==========================================
with tab_backtest:
    titre_section("Preuve de performance", "stratégie IA vs Buy & Hold")

    if not df.empty:
        df_sim, capital = simuler_strategie(df, capital_initial=10000.0)

        if df_sim is not None:
            perf_strategie = (df_sim['Valeur_Strategie'].iloc[-1] / capital - 1) * 100
            perf_bh = (df_sim['Valeur_Buy_Hold'].iloc[-1] / capital - 1) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("📈 Performance IA", f"{perf_strategie:.2f}%",
                      f"{(df_sim['Valeur_Strategie'].iloc[-1] - capital):.2f} TND")
            c2.metric("📊 Buy & Hold", f"{perf_bh:.2f}%",
                      f"{(df_sim['Valeur_Buy_Hold'].iloc[-1] - capital):.2f} TND")
            c3.metric("⚡ Écart vs Buy & Hold", f"{perf_strategie - perf_bh:+.2f} pts")

            st.write("")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Valeur_Buy_Hold'], mode='lines', name='Buy & Hold',
                                        line=dict(color=MUTED, width=2, dash='dash')))
            fig_bt.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Valeur_Strategie'], mode='lines', name='Stratégie IA',
                                        line=dict(color=MINT, width=3.5),
                                        fill='tonexty', fillcolor="rgba(0,245,160,0.10)"))
            fig_bt.add_hline(y=capital, line_dash="dot", line_color=AMBER, opacity=0.6,
                             annotation_text="Capital initial", annotation_font_color=AMBER)
            style_fig(fig_bt, height=520)
            st.plotly_chart(fig_bt, use_container_width=True)

            if perf_strategie > perf_bh:
                banniere("Verdict", "🏆 La stratégie IA a surperformé la simple conservation", MINT)
            else:
                banniere("Verdict", "⚠️ Sur cette période, le Buy & Hold aurait été plus rentable", AMBER)
        else:
            st.error("Pas assez de données pour le backtesting.")