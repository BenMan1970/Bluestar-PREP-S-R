import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import concurrent.futures

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Scanner S/R Exhaustif",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
    <style>
    [data-testid="stDataFrame"] > div { overflow-x: visible !important; overflow-y: visible !important; }
    [data-testid="stDataFrame"] iframe { width: 100% !important; height: auto !important; }
    ::-webkit-scrollbar { width: 0px !important; height: 0px !important; }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #D32F2F !important; color: white !important;
        border: 1px solid #B71C1C !important; font-size: 18px !important;
        font-weight: 700 !important; padding: 14px 40px !important;
        border-radius: 8px !important; transition: all 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #B71C1C !important;
        box-shadow: 0 4px 16px rgba(211,47,47,0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Scanner S/R Exhaustif (H4, D1, W)")
st.markdown(
    "Zones S/R avec **distance ATR**, **score de pertinence**, **tendance multi-TF** "
    "et **résumé par actif** — optimisé pour l'analyse quantitative."
)

col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    scan_button = st.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "XAG_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

# ── CORRECTIF 3 : Largeur de zone élargie pour indices ───────────
ZONE_WIDTH_MAP = {
    "US30_USD":   0.30,   # élargi (était 0.15)
    "NAS100_USD": 0.30,   # élargi (était 0.15)
    "SPX500_USD": 0.20,   # élargi (était 0.15)
    "DE30_EUR":   0.20,   # élargi (était 0.15)
    "XAU_USD":    0.20,
    "XAG_USD":    0.25,
    "XPT_USD":    0.25,
}
DEFAULT_ZONE_WIDTH = 0.5

# Seuils de distance max POUR LE PDF — adaptatifs par type d'actif
PDF_DIST_THRESHOLDS = {
    "US30_USD": 5.0, "NAS100_USD": 5.0, "SPX500_USD": 5.0, "DE30_EUR": 5.0,
    "XAU_USD": 10.0, "XAG_USD":  10.0, "XPT_USD":   10.0,
}
DEFAULT_PDF_DIST = 8.0

# ── CORRECTIF 5 : Distance absolue max par instrument ────────────
ABSOLUTE_MAX_DIST = {
    "XAU_USD": 8.0,  "XAG_USD": 8.0,  "XPT_USD": 8.0,
    "US30_USD": 4.0, "NAS100_USD": 4.0, "SPX500_USD": 4.0, "DE30_EUR": 4.0,
}

# ── CORRECTIF 4 : Post-fusion réduite pour indices ───────────────
POST_MERGE_THRESHOLD = 0.30

POST_MERGE_MAP = {
    "US30_USD":   0.05,   # réduit (était 0.10)
    "NAS100_USD": 0.05,   # réduit (était 0.10)
    "SPX500_USD": 0.08,   # réduit (était 0.10)
    "DE30_EUR":   0.08,   # réduit (était 0.10)
    "XAU_USD":    0.15,
    "XAG_USD":    0.20,
    "XPT_USD":    0.20,
}

# ── CORRECTIF 3 : Seuil de confluence adaptatif par instrument ───
CONFLUENCE_THRESHOLD_MAP = {
    "US30_USD":   1.5,
    "NAS100_USD": 1.5,
    "SPX500_USD": 1.2,
    "DE30_EUR":   1.2,
    "XAU_USD":    1.5,
    "XAG_USD":    1.5,
    "XPT_USD":    1.5,
}


# ══════════════════════════════════════════════════════════════════
# API OANDA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    for name, url in [
        ("Practice (Démo)", "https://api-fxpractice.oanda.com"),
        ("Live (Réel)",     "https://api-fxtrade.oanda.com"),
    ]:
        try:
            r = requests.get(f"{url}/v3/accounts/{account_id}/summary",
                             headers=headers, timeout=5)
            if r.status_code == 200:
                return url, name
        except requests.RequestException:
            continue
    return None, None


@st.cache_data(ttl=600)
def get_oanda_data(base_url, access_token, symbol, timeframe="daily", limit=500):
    url     = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {
        "count":       limit,
        "granularity": {"h4": "H4", "daily": "D", "weekly": "W"}[timeframe],
        "price":       "M",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("candles"):
            return None
        candles = [
            {
                "date":   pd.to_datetime(c["time"]),
                "open":   float(c["mid"]["o"]),
                "high":   float(c["mid"]["h"]),
                "low":    float(c["mid"]["l"]),
                "close":  float(c["mid"]["c"]),
                "volume": int(c["volume"]),
            }
            for c in data.get("candles", []) if c.get("complete")
        ]
        return pd.DataFrame(candles).set_index("date")
    except requests.RequestException:
        return None


@st.cache_data(ttl=60)
def get_oanda_current_price(base_url, access_token, account_id, symbol):
    url     = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        r = requests.get(url, headers=headers, params={"instruments": symbol}, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "prices" in data and data["prices"]:
            bid = float(data["prices"][0]["closeoutBid"])
            ask = float(data["prices"][0]["closeoutAsk"])
            return (bid + ask) / 2
        return None
    except requests.RequestException:
        return None


# ══════════════════════════════════════════════════════════════════
# CORRECTIF 1 — Validation prix live vs dernier close H4
# ══════════════════════════════════════════════════════════════════
def validate_live_price(live_price, symbol, base_url, access_token):
    """
    Vérifie que le prix live ne s'écarte pas de plus de 15% du dernier
    close H4. Si c'est le cas (données aberrantes), on utilise le close
    comme fallback et on logue un avertissement.
    Retourne : (prix_validé, message_alerte_ou_None)
    """
    if live_price is None:
        return None, None

    df_check = get_oanda_data(base_url, access_token, symbol, "h4", limit=10)
    if df_check is None or df_check.empty:
        return live_price, None

    last_close = df_check["close"].iloc[-1]
    if last_close <= 0:
        return live_price, None

    deviation = abs(live_price - last_close) / last_close * 100
    if deviation > 15.0:
        msg = (
            f"Prix live {live_price:.5f} écarté de {deviation:.1f}% "
            f"du dernier close H4 ({last_close:.5f}) — fallback sur close H4"
        )
        return last_close, msg

    return live_price, None


# ══════════════════════════════════════════════════════════════════
# MOTEUR D'ANALYSE — FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def get_adaptive_distance(timeframe):
    return {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)


def compute_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return None
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def compute_trend(df, sma_period=20):
    """
    Détermine la tendance structurelle d'un timeframe.

    Logique :
    - Position du prix par rapport à la SMA(20)
    - Pente des 5 dernières bougies (momentum court terme)
    - Structure Higher Highs / Lower Lows sur 10 bougies

    Retourne : 'HAUSSIER', 'BAISSIER' ou 'NEUTRE'
    """
    if df is None or len(df) < sma_period + 10:
        return "NEUTRE"

    close   = df["close"]
    sma     = close.rolling(sma_period).mean().iloc[-1]
    current = close.iloc[-1]

    slope_pct = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100

    highs = df["high"].iloc[-10:]
    lows  = df["low"].iloc[-10:]
    hh    = highs.iloc[-1] > highs.iloc[0]
    ll    = lows.iloc[-1]  < lows.iloc[0]

    above_sma = current > sma
    below_sma = current < sma

    if above_sma and slope_pct > 0.05 and hh:
        return "HAUSSIER"
    elif below_sma and slope_pct < -0.05 and ll:
        return "BAISSIER"
    elif above_sma and slope_pct > 0.05:
        return "HAUSSIER"
    elif below_sma and slope_pct < -0.05:
        return "BAISSIER"
    elif above_sma:
        return "HAUSSIER"
    elif below_sma:
        return "BAISSIER"
    return "NEUTRE"


def compute_relevance_score(strength, dist_pct, nb_tf):
    """
    Score composite de pertinence opérationnelle d'une zone S/R.
    Formule : (Force × Nb_Timeframes) / Distance_%
    """
    d = max(dist_pct, 0.01)
    return round((strength * nb_tf) / d, 1)


def post_merge_zones(zones_list, threshold_pct=0.30):
    """
    Fusionne les zones dont les niveaux sont à moins de threshold_pct% d'écart.
    """
    if len(zones_list) <= 1:
        return zones_list

    changed = True
    while changed:
        changed = False
        new_zones, used = [], set()

        for i in range(len(zones_list)):
            if i in used:
                continue
            group = [zones_list[i]]

            for j in range(i + 1, len(zones_list)):
                if j in used:
                    continue
                ref = np.mean([z["level"] for z in group])
                if abs(zones_list[j]["level"] - ref) / ref * 100 <= threshold_pct:
                    group.append(zones_list[j])
                    used.add(j)
                    changed = True
            used.add(i)

            new_zones.append({
                "level":    np.mean([z["level"] for z in group]),
                "strength": sum(z["strength"] for z in group),
            })

        zones_list = new_zones

    return zones_list


# ══════════════════════════════════════════════════════════════════
# CORRECTIF 2 — Détection d'anomalie renforcée (seuil 3.0 → 1.8)
# ══════════════════════════════════════════════════════════════════
def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    """
    Détecte les prix aberrants par deux méthodes complémentaires :
    1. Ratio prix / médiane des supports > 1.8 (était 3.0)
    2. Écart prix live vs dernier close > 10%
    Retourne un message d'alerte ou None.
    """
    messages = []

    # Méthode 1 : ratio vs médiane des supports
    if len(support_levels) >= 3:
        median_sup = np.median(support_levels)
        if median_sup > 0:
            ratio = current_price / median_sup
            if ratio > 1.8:   # seuil abaissé de 3.0 à 1.8
                messages.append(
                    f"Prix {current_price:.2f} = {ratio:.1f}x la médiane des supports "
                    f"({median_sup:.2f}) — données à vérifier ou rally exceptionnel"
                )

    # Méthode 2 : écart prix live vs dernier close (nouveau)
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > 10.0:
            messages.append(
                f"Prix live {current_price:.2f} s'écarte de {dev:.1f}% "
                f"du dernier close ({last_candle_close:.2f})"
            )

    return " | ".join(messages) if messages else None


def get_price_context(current_price, supports, resistances):
    """
    Génère un résumé textuel court de la position du prix
    par rapport aux zones les plus proches.
    """
    parts = []

    if not supports.empty:
        nearest_sup = supports.loc[supports["level"].idxmax()]
        dist_s = abs(current_price - nearest_sup["level"]) / current_price * 100
        tag    = "SUR support" if dist_s < 0.5 else "S proche"
        parts.append(f"{tag}: {nearest_sup['level']:.5f} (-{dist_s:.2f}%)")

    if not resistances.empty:
        nearest_res = resistances.loc[resistances["level"].idxmin()]
        dist_r = abs(current_price - nearest_res["level"]) / current_price * 100
        tag    = "SUR resistance" if dist_r < 0.5 else "R proche"
        parts.append(f"{tag}: {nearest_res['level']:.5f} (+{dist_r:.2f}%)")

    return "  |  ".join(parts) if parts else "Zone intermédiaire"


# ══════════════════════════════════════════════════════════════════
# DÉTECTION DES ZONES S/R
# ══════════════════════════════════════════════════════════════════
def find_strong_sr_zones(df, current_price, zone_percentage_width=0.5,
                          min_touches=2, timeframe="daily",
                          post_merge_threshold=0.30):
    """
    Détecte les zones S/R par pivot clustering avec post-fusion.
    """
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None:
        current_price = df["close"].iloc[-1]

    distance   = get_adaptive_distance(timeframe)
    r_idx, _   = find_peaks(df["high"],  distance=distance)
    s_idx, _   = find_peaks(-df["low"],  distance=distance)
    pivots_h   = df.iloc[r_idx]["high"]
    pivots_l   = df.iloc[s_idx]["low"]
    all_pivots = pd.concat([pivots_h, pivots_l]).sort_values()

    if all_pivots.empty:
        return pd.DataFrame(), pd.DataFrame()

    zones, current_zone = [], [all_pivots.iloc[0]]
    for price in all_pivots.iloc[1:]:
        zone_avg = np.mean(current_zone)
        if abs(price - zone_avg) < (zone_avg * zone_percentage_width / 100):
            current_zone.append(price)
        else:
            zones.append(current_zone)
            current_zone = [price]
    zones.append(current_zone)

    strong = [
        {"level": np.mean(z), "strength": len(z)}
        for z in zones if len(z) >= min_touches
    ]
    if not strong:
        return pd.DataFrame(), pd.DataFrame()

    strong = post_merge_zones(strong, threshold_pct=post_merge_threshold)

    df_zones    = pd.DataFrame(strong).sort_values("level").reset_index(drop=True)
    supports    = df_zones[df_zones["level"] <  current_price].copy()
    resistances = df_zones[df_zones["level"] >= current_price].copy()
    return supports, resistances


# ══════════════════════════════════════════════════════════════════
# CONFLUENCES MULTI-TIMEFRAMES
# ══════════════════════════════════════════════════════════════════
def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0):
    if not zones_dict or current_price is None:
        return []

    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, z in supports.iterrows():
            all_zones.append({"tf": tf, "level": z["level"],
                               "strength": z["strength"], "type": "Support"})
        for _, z in resistances.iterrows():
            all_zones.append({"tf": tf, "level": z["level"],
                               "strength": z["strength"], "type": "Résistance"})

    if not all_zones:
        return []

    zones_df     = pd.DataFrame(all_zones).sort_values("level")
    used_indices = set()
    confluences  = []

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue

        similar = zones_df[
            (abs(zones_df["level"] - zone["level"]) / zone["level"] * 100 <= confluence_threshold) &
            (zones_df.index != i)
        ]

        if len(similar) >= 1:
            group          = pd.concat([zones_df.loc[[i]], similar])
            timeframes     = group["tf"].unique()

            if len(timeframes) >= 2:
                avg_level      = group["level"].mean()
                total_strength = int(group["strength"].sum())
                nb_tf          = len(timeframes)
                dist_pct       = abs(current_price - avg_level) / current_price * 100
                score          = compute_relevance_score(total_strength, dist_pct, nb_tf)
                zone_type      = group.iloc[0]["type"]
                signal         = "🟢 BUY ZONE" if zone_type == "Support" else "🔴 SELL ZONE"
                tf_label       = " + ".join(sorted(timeframes))
                alerte         = "🔥 ZONE CHAUDE" if dist_pct < 0.5 else ("⚠️ Proche" if dist_pct < 1.5 else "")

                confluences.append({
                    "Actif":        symbol,
                    "Signal":       signal,
                    "Niveau":       f"{avg_level:.5f}",
                    "Type":         zone_type,
                    "Timeframes":   tf_label,
                    "Nb TF":        nb_tf,
                    "Force Totale": total_strength,
                    "Score":        score,
                    "Distance %":   f"{dist_pct:.2f}%",
                    "Alerte":       alerte,
                })
                used_indices.update(group.index)

    return confluences


# ══════════════════════════════════════════════════════════════════
# SCAN PARALLÈLE PAR SYMBOLE
# ══════════════════════════════════════════════════════════════════
def scan_single_symbol(args):
    """
    Scan complet H4 + Daily + Weekly pour un symbole.
    Intègre les correctifs 1, 2, 3, 4, 5.
    """
    symbol, base_url, access_token, account_id, zone_width, min_touches = args

    # ── CORRECTIF 1 : Validation du prix live ────────────────────
    raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
    current_price, price_alert_msg = validate_live_price(
        raw_price, symbol, base_url, access_token
    )

    adaptive_width = ZONE_WIDTH_MAP.get(symbol, zone_width)
    merge_thresh   = POST_MERGE_MAP.get(symbol, POST_MERGE_THRESHOLD)
    pdf_dist_max   = PDF_DIST_THRESHOLDS.get(symbol, DEFAULT_PDF_DIST)
    abs_dist_max   = ABSOLUTE_MAX_DIST.get(symbol, 99.0)   # CORRECTIF 5

    rows           = {"H4": None, "Daily": None, "Weekly": None}
    zones_d        = {}
    trends         = {}
    all_sup_levels = []
    last_h4_close  = None   # CORRECTIF 2
    anomaly_msg    = None

    # Ajouter alerte prix aberrant s'il y en a une
    if price_alert_msg:
        anomaly_msg = price_alert_msg

    for tf_key, tf_cap in [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]:
        df = get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)
        if df is None or df.empty:
            continue

        cp = current_price if current_price is not None else df["close"].iloc[-1]

        # ── CORRECTIF 2 : Stocker le dernier close H4 ────────────
        if tf_key == "h4":
            last_h4_close = df["close"].iloc[-1]

        trends[tf_cap] = compute_trend(df)

        supports, resistances = find_strong_sr_zones(
            df, cp,
            zone_percentage_width=adaptive_width,
            min_touches=min_touches,
            timeframe=tf_key,
            post_merge_threshold=merge_thresh,
        )
        zones_d[tf_cap] = (supports, resistances)

        atr_val = compute_atr(df, period=14)

        if not supports.empty:
            all_sup_levels.extend(supports["level"].tolist())

        price_ctx = ""
        if tf_key == "daily":
            price_ctx = get_price_context(cp, supports, resistances)
            zones_d["_price_ctx"] = price_ctx

        sym_d = symbol.replace("_", "/")

        # ── CORRECTIF 5 : Distance absolue max dans make_row ─────
        def make_row(zone, ztype, _cp=cp, _atr=atr_val,
                     _pdf_max=pdf_dist_max, _abs_max=abs_dist_max):
            lvl      = zone["level"]
            strength = int(zone["strength"])
            dist_pct = abs(_cp - lvl) / _cp * 100
            dist_atr = round(abs(_cp - lvl) / _atr, 1) if (_atr and _atr > 0) else np.nan
            score    = compute_relevance_score(strength, dist_pct, 1)
            return {
                "Actif":       sym_d,
                "Prix Actuel": f"{_cp:.5f}",
                "Type":        ztype,
                "Niveau":      f"{lvl:.5f}",
                "Force":       f"{strength} touches",
                "Dist. %":     f"{dist_pct:.2f}%",
                "Dist. ATR":   f"{dist_atr:.1f}x" if not np.isnan(dist_atr) else "N/A",
                "Score":       score,
                "_dist_num":   dist_pct,
                # Filtre PDF : distance configurable ET distance absolue max
                "_in_pdf":     dist_pct <= _pdf_max and dist_pct <= _abs_max,
            }

        tf_rows = (
            [make_row(z, "Support")    for _, z in supports.iterrows()] +
            [make_row(z, "Résistance") for _, z in resistances.iterrows()]
        )
        if tf_rows:
            rows[tf_cap] = tf_rows

    # ── CORRECTIF 2 : Détection anomalie renforcée ───────────────
    if all_sup_levels and current_price:
        new_anomaly = flag_data_anomaly(
            symbol, current_price, all_sup_levels, last_h4_close
        )
        if new_anomaly:
            # Combiner avec l'alerte prix si elle existe déjà
            if anomaly_msg:
                anomaly_msg = f"{anomaly_msg} | {new_anomaly}"
            else:
                anomaly_msg = new_anomaly

    return symbol, rows, zones_d, current_price, trends, anomaly_msg


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION PDF
# ══════════════════════════════════════════════════════════════════
def strip_emojis_df(df):
    emoji_map = {
        '🟢': '[BUY]', '🔴': '[SELL]', '🔥': '[CHAUD]', '⚠️': '[PROCHE]',
        '📈': '', '📉': '', '↔️': '', '✅': '[OK]', '❌': '[X]',
        '⚡': '[!]', '📡': '', '📅': '',
    }
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        for emoji, replacement in emoji_map.items():
            clean[col] = clean[col].astype(str).str.replace(emoji, replacement, regex=False)
        clean[col] = clean[col].apply(
            lambda x: x.encode('latin-1', errors='ignore').decode('latin-1')
        )
    return clean


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Rapport de Scan Support/Resistance', border=0, align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 10, f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def chapter_summary(self, summaries):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, 'RESUME PAR ACTIF  (Tendances + Top Zones Confluentes)',
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

        for s in summaries:
            sym = s.get('symbol', '').encode('latin-1', errors='ignore').decode('latin-1')
            t_h4 = s.get('trend_h4',    'N/A')
            t_d  = s.get('trend_daily',  'N/A')
            t_w  = s.get('trend_weekly', 'N/A')
            ctx  = s.get('price_context', '').encode('latin-1', errors='ignore').decode('latin-1')

            self.set_font('Helvetica', 'B', 8)
            line = f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"
            self.cell(0, 5, line, border=0, new_x='LMARGIN', new_y='NEXT')

            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}",
                          border=0, new_x='LMARGIN', new_y='NEXT')

            top = s.get('top_zones', [])
            self.set_font('Helvetica', '', 7)
            if top:
                for z in top:
                    sig   = str(z.get('Signal',     '')).replace('🟢','[BUY]').replace('🔴','[SELL]')
                    sig   = sig.encode('latin-1', errors='ignore').decode('latin-1')
                    niv   = str(z.get('Niveau',     ''))
                    dist  = str(z.get('Distance %', ''))
                    score = str(z.get('Score',      ''))
                    tfs   = str(z.get('Timeframes', ''))
                    ale   = str(z.get('Alerte', '')).replace('🔥','[CHAUD]').replace('⚠️','[PROCHE]')
                    ale   = ale.encode('latin-1', errors='ignore').decode('latin-1')
                    txt   = f"  {sig}  Niv:{niv}  Dist:{dist}  Score:{score}  TF:{tfs}  {ale}"
                    self.cell(0, 4, txt[:130], border=0, new_x='LMARGIN', new_y='NEXT')
            else:
                self.cell(0, 4, "  Aucune confluence pour cet actif.",
                          border=0, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)

    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 10, "Aucune donnee a afficher.")
            self.ln()
            return

        if 'Timeframes' in df.columns:
            col_widths = {
                'Actif': 22, 'Signal': 28, 'Niveau': 24, 'Type': 24,
                'Timeframes': 60, 'Nb TF': 13, 'Force Totale': 22,
                'Score': 16, 'Distance %': 20, 'Alerte': 56,
            }
            font_size = 7
        else:
            col_widths = {
                'Actif': 28, 'Prix Actuel': 28, 'Type': 22,
                'Niveau': 28, 'Force': 24,
                'Dist. %': 18, 'Dist. ATR': 18, 'Score': 16,
            }
            font_size = 7

        cols    = [c for c in col_widths if c in df.columns]
        total_w = sum(col_widths[c] for c in cols)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start  = self.l_margin + max(0, (usable_w - total_w) / 2)

        self.set_font('Helvetica', 'B', font_size)
        self.set_x(x_start)
        for col_name in cols:
            w = col_widths[col_name]
            self.cell(w, 6, col_name, border=1, align='C',
                      new_x='RIGHT', new_y='TOP')
        self.ln()

        self.set_font('Helvetica', '', font_size)
        for i, (_, row) in enumerate(df.iterrows()):
            self.set_x(x_start)
            for col_name in cols:
                w        = col_widths[col_name]
                val      = str(row[col_name])
                max_chars = int(w / 1.25)
                if len(val) > max_chars:
                    val = val[:max_chars - 1] + '.'
                self.cell(w, 5, val, border=1, align='C',
                          new_x='RIGHT', new_y='TOP')
            self.ln()


def _apply_pdf_filter(df):
    if df.empty:
        return df
    if "_in_pdf" in df.columns:
        df = df[df["_in_pdf"] == True].copy()
    return df.drop(columns=["_in_pdf", "_dist_num"], errors="ignore").reset_index(drop=True)


def create_pdf_report(results_dict, confluences_df=None, summaries=None, anomalies=None):
    summaries = summaries or []
    anomalies = anomalies or []

    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if anomalies:
        pdf.chapter_title('ALERTES QUALITE DES DONNEES')
        pdf.set_font('Helvetica', 'I', 8)
        for a in anomalies:
            line = f"  {a['actif']} : {a['msg']}"
            line = line.encode('latin-1', errors='ignore').decode('latin-1')
            pdf.multi_cell(0, 5, line)
        pdf.ln(5)

    if summaries:
        pdf.chapter_summary(summaries)
        pdf.add_page()

    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title('*** ZONES DE CONFLUENCE MULTI-TIMEFRAMES ***')
        clean_conf = strip_emojis_df(confluences_df.copy())
        clean_conf = clean_conf.drop(columns=["_in_pdf", "_dist_num"], errors="ignore")
        if "Score" in clean_conf.columns:
            clean_conf = clean_conf.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_conf)
        pdf.ln(10)

    title_map = {
        'H4':     'Analyse 4 Heures (H4)',
        'Daily':  'Analyse Journaliere (Daily)',
        'Weekly': 'Analyse Hebdomadaire (Weekly)',
    }
    for tf_key, df in results_dict.items():
        if df is None or (hasattr(df, 'empty') and df.empty):
            continue
        pdf.chapter_title(title_map.get(tf_key, tf_key))
        clean_df = strip_emojis_df(df.copy())
        clean_df = clean_df.drop(columns=["_in_pdf", "_dist_num"], errors="ignore")
        if "Score" in clean_df.columns:
            clean_df = clean_df.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_df)
        pdf.ln(10)

    return bytes(pdf.output())


def create_csv_report(results_dict, confluences_df=None):
    all_dfs = []
    if confluences_df is not None and not confluences_df.empty:
        c = confluences_df.copy()
        c["Section"] = "CONFLUENCES"
        all_dfs.append(c)
    for tf, df in results_dict.items():
        if not df.empty:
            d = df.copy()
            d["Timeframe"] = tf
            all_dfs.append(d)
    if not all_dfs:
        return b""
    buf = BytesIO()
    pd.concat(all_dfs, ignore_index=True).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE STREAMLIT
# ══════════════════════════════════════════════════════════════════
def _display_results(sr, max_dist_filter):
    df_h4     = sr.get("df_h4",         pd.DataFrame())
    df_daily  = sr.get("df_daily",       pd.DataFrame())
    df_wk     = sr.get("df_weekly",      pd.DataFrame())
    conf_full = sr.get("conf_full",      pd.DataFrame())
    conf_filt = sr.get("conf_filtered",  pd.DataFrame())
    rep_dict  = sr.get("report_dict",    {})
    summaries = sr.get("summaries",      [])
    anomalies = sr.get("anomaly_flags",  [])

    tf_cfg = {
        "Actif":       st.column_config.TextColumn("Actif",       width="small"),
        "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"),
        "Type":        st.column_config.TextColumn("Type",        width="small"),
        "Niveau":      st.column_config.TextColumn("Niveau",      width="small"),
        "Force":       st.column_config.TextColumn("Force",       width="medium"),
        "Dist. %":     st.column_config.TextColumn("Dist. %",     width="small"),
        "Dist. ATR":   st.column_config.TextColumn("Dist. ATR",   width="small"),
        "Score":       st.column_config.NumberColumn("Score ▼",   width="small"),
    }

    if anomalies:
        with st.expander(f"⚡ {len(anomalies)} alerte(s) qualité des données", expanded=True):
            for a in anomalies:
                st.warning(f"**{a['actif']}** : {a['msg']}")

    if not conf_filt.empty:
        st.divider()
        st.subheader("🔥 ZONES DE CONFLUENCE MULTI-TIMEFRAMES")
        st.markdown("Zones validées sur plusieurs timeframes — triées par **Score de Pertinence**")

        disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total zones",       len(disp))
        c2.metric("🔥 Zones chaudes",  len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
        c3.metric("⚠️ Zones proches",  len(disp[disp["Alerte"] == "⚠️ Proche"]))
        c4.metric("🟢 BUY Zones",      len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
        c5.metric("🔴 SELL Zones",     len(disp[disp["Signal"] == "🔴 SELL ZONE"]))

        conf_cfg = {
            **{k: st.column_config.TextColumn(k, width="small")
               for k in ["Actif","Signal","Niveau","Type","Timeframes","Distance %","Alerte"]},
            "Nb TF":        st.column_config.NumberColumn("Nb TF",        width="small"),
            "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
            "Score":        st.column_config.NumberColumn("Score ▼",      width="small"),
        }
        st.dataframe(disp, column_config=conf_cfg, hide_index=True,
                     use_container_width=True, height=min(len(disp) * 35 + 38, 750))
    else:
        st.info("Aucune confluence détectée dans la plage de distance sélectionnée. "
                "Essayez d'augmenter le filtre ou le seuil de confluence.")

    st.subheader("📋 Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_pdf_report(rep_dict, conf_full, summaries, anomalies)
            st.download_button(
                "📄 Télécharger le Rapport (PDF)",
                data=pdf_bytes,
                file_name=f"rapport_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col2:
            csv_bytes = create_csv_report(rep_dict, conf_full)
            st.download_button(
                "📊 Télécharger les Données (CSV)",
                data=csv_bytes,
                file_name=f"donnees_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    def _filter_and_sort(df, max_pct):
        if df.empty:
            return df
        def to_float(s):
            try:
                return float(str(s).replace("%", ""))
            except Exception:
                return 999.0
        mask = df["Dist. %"].apply(to_float) <= max_pct
        out  = df[mask].drop(columns=["_in_pdf", "_dist_num"], errors="ignore")
        if "Score" in out.columns:
            out = out.sort_values("Score", ascending=False)
        return out.reset_index(drop=True)

    st.divider()
    st.subheader("📅 Analyse 4 Heures (H4)")
    fd = _filter_and_sort(df_h4, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(fd) * 35 + 38, 600))

    st.subheader("📅 Analyse Journalière (Daily)")
    fd = _filter_and_sort(df_daily, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(fd) * 35 + 38, 600))

    st.subheader("📅 Analyse Hebdomadaire (Weekly)")
    fd = _filter_and_sort(df_wk, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(fd) * 35 + 38, 600))


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("1. Connexion OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id   = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets chargés ✓")
    except Exception:
        access_token, account_id = None, None
        st.error("Secrets OANDA non trouvés.")

    st.header("2. Sélection des Actifs")
    select_all = st.checkbox(f"Scanner tous les actifs ({len(ALL_SYMBOLS)})", value=True)
    if select_all:
        symbols_to_scan = ALL_SYMBOLS
    else:
        default_sel = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY",
                       "AUD_USD", "EUR_JPY", "GBP_JPY"]
        symbols_to_scan = st.multiselect(
            "Actifs spécifiques :", options=ALL_SYMBOLS, default=default_sel
        )

    st.header("3. Paramètres de Détection")
    zone_width = st.slider(
        "Largeur de zone Forex (%)", 0.1, 2.0, 0.5, 0.1,
        help="Indices: 0.20-0.30% auto | Métaux: 0.20-0.25% auto",
    )
    min_touches = st.slider(
        "Force minimale (touches)", 2, 10, 3, 1,
        help="Nombre de contacts minimum pour valider une zone.",
    )
    confluence_threshold = st.slider(
        "Seuil de confluence Forex (%)", 0.3, 2.0, 1.0, 0.1,
        help="Indices/Métaux utilisent des seuils adaptatifs (1.2-1.5%) automatiquement.",
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) — filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
        help="Ne filtre que l'affichage Streamlit. Le PDF applique des seuils adaptatifs.",
    )

    st.divider()
    st.caption("**Score de Pertinence** = (Force × Nb TF) / Distance%")
    st.caption("🔴 > 200 : Zone critique")
    st.caption("🟠 100-200 : Zone prioritaire")
    st.caption("🟡 20-100 : Zone à surveiller")
    st.caption("⚪ < 20 : Zone secondaire")

    st.divider()
    st.caption("**Filtrage PDF adaptatif :**")
    st.caption("Forex : ≤ 8% | Indices : ≤ 4% abs | Métaux : ≤ 8% abs")
    st.caption("**Post-fusion :** 0.30% Forex | 0.05% Indices | 0.15-0.20% Métaux")
    st.caption("**Confluence :** 1.0% Forex | 1.2% Indices EU | 1.5% US/Métaux")


# ══════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE
# ══════════════════════════════════════════════════════════════════
if scan_button and symbols_to_scan:
    st.session_state.pop("scan_results", None)

    if not access_token or not account_id:
        st.warning("Configurez vos secrets OANDA avant de lancer le scan.")
    else:
        base_url, env_name = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.error("Identifiants OANDA invalides. Vérifiez vos secrets.")
        else:
            st.info(f"Environnement détecté : **{env_name}**")
            progress_bar = st.progress(0, text="Initialisation du scan parallèle…")

            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map, prices_map, trends_map = {}, {}, {}
            all_anomalies = []

            args_list = [
                (sym, base_url, access_token, account_id, zone_width, min_touches)
                for sym in symbols_to_scan
            ]
            total, completed = len(args_list), 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_map = {
                    executor.submit(scan_single_symbol, a): a[0] for a in args_list
                }
                for future in concurrent.futures.as_completed(future_map):
                    sym = future_map[future]
                    completed += 1
                    progress_bar.progress(
                        completed / total,
                        text=f"Scan… ({completed}/{total}) {sym.replace('_', '/')}",
                    )
                    try:
                        symbol, rows, zones_d, cp, trends, anomaly_msg = future.result()
                        all_zones_map[symbol] = zones_d
                        prices_map[symbol]    = cp
                        trends_map[symbol]    = trends

                        if anomaly_msg:
                            all_anomalies.append({"actif": symbol.replace("_", "/"),
                                                  "msg":   anomaly_msg})

                        for tf_cap, tf_rows in rows.items():
                            if tf_rows:
                                if tf_cap == "H4":
                                    results_h4.extend(tf_rows)
                                elif tf_cap == "Daily":
                                    results_daily.extend(tf_rows)
                                elif tf_cap == "Weekly":
                                    results_weekly.extend(tf_rows)
                    except Exception:
                        pass

            progress_bar.empty()
            st.success(f"✅ Scan terminé ! {len(symbols_to_scan)} actifs analysés.")

            # ── Confluences avec seuils adaptatifs (CORRECTIF 3) ──
            st.info("🔍 Analyse des confluences multi-timeframes…")
            all_confluences = []
            for sym in symbols_to_scan:
                cp = prices_map.get(sym)
                zones_clean = {k: v for k, v in all_zones_map.get(sym, {}).items()
                               if not k.startswith("_")}
                # Seuil adaptatif par instrument
                sym_threshold = CONFLUENCE_THRESHOLD_MAP.get(sym, confluence_threshold)
                confs = detect_confluences(
                    sym.replace("_", "/"), zones_clean, cp, sym_threshold
                )
                all_confluences.extend(confs)

            conf_full = pd.DataFrame(all_confluences)

            if not conf_full.empty:
                conf_full["_dist_num"] = (
                    conf_full["Distance %"].str.replace("%", "").astype(float)
                )
                conf_filtered = (
                    conf_full[conf_full["_dist_num"] <= max_dist_filter]
                    .drop(columns=["_dist_num"])
                    .reset_index(drop=True)
                )
                conf_full = conf_full.drop(columns=["_dist_num"])
            else:
                conf_filtered = pd.DataFrame()

            # ── Résumés par actif ──────────────────────────────────
            summaries = []
            for sym in symbols_to_scan:
                sym_d  = sym.replace("_", "/")
                trends = trends_map.get(sym, {})
                cp     = prices_map.get(sym)

                top_zones = []
                if not conf_full.empty and sym_d in conf_full["Actif"].values:
                    ac = conf_full[conf_full["Actif"] == sym_d].copy()
                    ac = ac.sort_values("Score", ascending=False)
                    top_zones = ac.head(3).to_dict("records")

                price_ctx = ""
                d_zones = all_zones_map.get(sym, {})
                if "_price_ctx" in d_zones:
                    price_ctx = d_zones["_price_ctx"]
                elif "Daily" in d_zones and cp:
                    sup_d, res_d = d_zones["Daily"]
                    price_ctx = get_price_context(cp, sup_d, res_d)

                summaries.append({
                    "symbol":        sym_d,
                    "trend_h4":      trends.get("H4",     "N/A"),
                    "trend_daily":   trends.get("Daily",  "N/A"),
                    "trend_weekly":  trends.get("Weekly", "N/A"),
                    "price_context": price_ctx,
                    "top_zones":     top_zones,
                })

            # ── DataFrames finaux ──────────────────────────────────
            df_h4     = pd.DataFrame(results_h4)
            df_daily  = pd.DataFrame(results_daily)
            df_weekly = pd.DataFrame(results_weekly)

            rep_dict = {
                "H4":     _apply_pdf_filter(df_h4),
                "Daily":  _apply_pdf_filter(df_daily),
                "Weekly": _apply_pdf_filter(df_weekly),
            }

            st.session_state["scan_results"] = {
                "df_h4":         df_h4,
                "df_daily":      df_daily,
                "df_weekly":     df_weekly,
                "conf_full":     conf_full,
                "conf_filtered": conf_filtered,
                "report_dict":   rep_dict,
                "summaries":     summaries,
                "anomaly_flags": all_anomalies,
                "max_dist":      max_dist_filter,
            }

            _display_results(st.session_state["scan_results"], max_dist_filter)

elif not symbols_to_scan:
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
else:
    st.info("Configurez les paramètres dans la barre latérale, puis cliquez sur **LANCER LE SCAN**.")

# ── Ré-affichage persistant après rerun Streamlit ────────────────
if "scan_results" in st.session_state and not scan_button:
    _display_results(
        st.session_state["scan_results"],
        st.session_state["scan_results"].get("max_dist", 3.0),
    )
