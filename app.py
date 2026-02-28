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
st.set_page_config(page_title="Scanner S/R Exhaustif", page_icon="📡", layout="wide")

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

st.title("📡 Scanner S/R Exhaustif - v4 (H4, D1, W)")
st.markdown(
    "Zones S/R avec **swing HH/LL confirmé**, **score pondéré TF+âge**, "
    "**statut Vierge/Testée/Consommée**, **plage prix valides**."
)

col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    scan_button = st.button("🚀 LANCER LE SCAN", type="primary", width='stretch')

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

ATR_ZONE_COEFF = {
    "XAU_USD": 0.5, "XAG_USD": 0.5, "XPT_USD": 0.5,
    "US30_USD": 0.3, "NAS100_USD": 0.3, "SPX500_USD": 0.3, "DE30_EUR": 0.3,
}
DEFAULT_ATR_COEFF = 0.4

PROMINENCE_COEFF = {
    "XAU_USD": 0.5, "XAG_USD": 0.5, "XPT_USD": 0.5,
    "US30_USD": 0.4, "NAS100_USD": 0.4, "SPX500_USD": 0.4, "DE30_EUR": 0.4,
}
DEFAULT_PROMINENCE_COEFF = 0.3

PRICE_SANITY_RANGE = {
    "XAU_USD":    (1500.0,  4000.0),
    "XAG_USD":    (15.0,    60.0),
    "XPT_USD":    (700.0,   2500.0),
    "US30_USD":   (20000.0, 60000.0),
    "NAS100_USD": (8000.0,  35000.0),
    "SPX500_USD": (3000.0,  8500.0),
    "DE30_EUR":   (8000.0,  30000.0),
}

ZONE_WIDTH_FALLBACK = {
    "US30_USD":   0.50,
    "NAS100_USD": 0.50,
    "SPX500_USD": 0.25,
    "DE30_EUR":   0.25,
    "XAU_USD":    0.20,
    "XAG_USD":    0.25,
    "XPT_USD":    0.25,
}
DEFAULT_ZONE_WIDTH = 0.5

PDF_DIST_THRESHOLDS = {
    "US30_USD": 5.0, "NAS100_USD": 5.0, "SPX500_USD": 5.0, "DE30_EUR": 5.0,
    "XAU_USD": 8.0, "XAG_USD": 8.0, "XPT_USD": 8.0,
}
DEFAULT_PDF_DIST = 8.0

ABSOLUTE_MAX_DIST = {
    "XAU_USD": 8.0,  "XAG_USD": 8.0,  "XPT_USD": 8.0,
    "US30_USD": 4.0, "NAS100_USD": 4.0, "SPX500_USD": 4.0, "DE30_EUR": 4.0,
}

POST_MERGE_THRESHOLD = 0.30
POST_MERGE_MAP = {
    "US30_USD": 0.05, "NAS100_USD": 0.05, "SPX500_USD": 0.08, "DE30_EUR": 0.08,
    "XAU_USD": 0.15,  "XAG_USD": 0.20,   "XPT_USD": 0.20,
}

CONFLUENCE_THRESHOLD_MAP = {
    "US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2,
    "XAU_USD": 1.5,  "XAG_USD": 1.5,   "XPT_USD": 1.5,
}


# ══════════════════════════════════════════════════════════════════
# UTILITAIRE PDF — CORRECTION BUG FPDFUnicodeEncodingException
# ══════════════════════════════════════════════════════════════════
# str.maketrans() n'accepte que des clés à 1 seul codepoint.
# Les emojis composés comme '⚠️' (2 codepoints) lèvent un ValueError.
# → On sépare : _ACCENT_MAP (str.maketrans, 1 char) + _EMOJI_MAP (str.replace).

# Accents uniquement — tous des caractères simples (1 codepoint)
_ACCENT_MAP = str.maketrans(
    'àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
    'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN'
)

# Emojis et séquences multi-codepoints — traités par str.replace()
_EMOJI_MAP = [
    ('🟢', '[BUY]'),
    ('🔴', '[SELL]'),
    ('🔥', '[CHAUD]'),
    ('⚠️', '[PROCHE]'),
    ('⚠',  '[PROCHE]'),   # variante sans sélecteur de variation
    ('📈', ''),
    ('📉', ''),
    ('↔️', ''),
    ('↔',  ''),
    ('✅', '[OK]'),
    ('❌', '[X]'),
    ('⚡', '[!]'),
    ('📡', ''),
    ('📅', ''),
]

def _safe_pdf_str(text: str) -> str:
    """
    Convertit une chaîne quelconque en chaîne sûre pour les polices
    core fpdf2 (latin-1). Translitère les accents français (str.translate),
    remplace les emojis (str.replace), puis encode en latin-1 avec
    remplacement des caractères restants.
    """
    if not isinstance(text, str):
        text = str(text)
    # 1. Translitérer les accents (clés single-char → safe pour maketrans)
    text = text.translate(_ACCENT_MAP)
    # 2. Remplacer les emojis / séquences multi-codepoints
    for emoji, replacement in _EMOJI_MAP:
        text = text.replace(emoji, replacement)
    # 3. Éliminer tout caractère restant hors latin-1
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ══════════════════════════════════════════════════════════════════
# API OANDA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    for name, url in [
        ("Practice (Demo)", "https://api-fxpractice.oanda.com"),
        ("Live (Reel)",     "https://api-fxtrade.oanda.com"),
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
# VALIDATION PRIX
# ══════════════════════════════════════════════════════════════════
def validate_live_price(live_price, symbol, base_url, access_token):
    if live_price is None:
        return None, None

    alerts = []

    if symbol in PRICE_SANITY_RANGE:
        lo, hi = PRICE_SANITY_RANGE[symbol]
        if not (lo <= live_price <= hi):
            alerts.append(
                f"Prix live {live_price:.2f} hors plage valide [{lo:.0f}-{hi:.0f}] "
                f"(prob. probleme cents/unites OANDA)"
            )
            live_price = None

    df_check = get_oanda_data(base_url, access_token, symbol, "h4", limit=10)
    if df_check is not None and not df_check.empty:
        last_close = df_check["close"].iloc[-1]
        if last_close > 0:
            if symbol in PRICE_SANITY_RANGE:
                lo, hi = PRICE_SANITY_RANGE[symbol]
                if not (lo <= last_close <= hi):
                    alerts.append(
                        f"Close H4 {last_close:.2f} aussi hors plage - "
                        f"donnees OANDA non fiables pour cet instrument"
                    )
                    return None, " | ".join(alerts) if alerts else None

            if live_price is not None:
                dev = abs(live_price - last_close) / last_close * 100
                if dev > 15.0:
                    alerts.append(
                        f"Prix live {live_price:.2f} ecarte de {dev:.1f}% "
                        f"du close H4 ({last_close:.2f}) - fallback close H4"
                    )
                    live_price = last_close
            else:
                live_price = last_close

    return live_price, " | ".join(alerts) if alerts else None


# ══════════════════════════════════════════════════════════════════
# MOTEUR D'ANALYSE
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
    if df is None or len(df) < 15:
        return "NEUTRE"

    actual_period = min(sma_period, len(df) - 5)
    close   = df["close"]
    sma     = close.rolling(actual_period).mean().iloc[-1]
    current = close.iloc[-1]

    slope_pct = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100

    n = min(10, len(df))
    highs = df["high"].iloc[-n:]
    lows  = df["low"].iloc[-n:]
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


TF_WEIGHT = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}

def compute_structural_score(strength, nb_tf, tf_name="H4", age_bars=0, total_bars=500):
    tf_w     = TF_WEIGHT.get(tf_name, 1.0)
    age_r    = age_bars / max(total_bars, 1)
    age_f    = float(np.exp(-1.5 * age_r))
    raw      = strength * tf_w * nb_tf
    return round(raw * age_f, 1)


def post_merge_zones(zones_list, threshold_pct=0.30):
    if len(zones_list) <= 1:
        return zones_list

    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Consommee": 2}

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

            best_age   = min(z.get("age_bars", 0) for z in group)
            best_status = min(
                (z.get("status", "Testee") for z in group),
                key=lambda s: STATUS_PRIORITY.get(s, 1)
            )
            new_zones.append({
                "level":    np.mean([z["level"] for z in group]),
                "strength": sum(z["strength"] for z in group),
                "age_bars": best_age,
                "status":   best_status,
            })
        zones_list = new_zones

    return zones_list


def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    messages = []
    if len(support_levels) >= 3:
        median_sup = np.median(support_levels)
        if median_sup > 0:
            ratio = current_price / median_sup
            if ratio > 1.8:
                messages.append(
                    f"Prix {current_price:.2f} = {ratio:.1f}x la mediane des supports "
                    f"({median_sup:.2f}) - donnees a verifier"
                )
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > 10.0:
            messages.append(
                f"Prix live {current_price:.2f} s'ecarte de {dev:.1f}% "
                f"du dernier close ({last_candle_close:.2f})"
            )
    return " | ".join(messages) if messages else None


def get_price_context(current_price, supports, resistances):
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
    return "  |  ".join(parts) if parts else "Zone intermediaire"


# ══════════════════════════════════════════════════════════════════
# DÉTECTION SWING HH/LL
# ══════════════════════════════════════════════════════════════════
def detect_swing_pivots(df, n=3, atr_val=None, prominence_coeff=0.3):
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n_bars = len(df)

    swing_high_idx, swing_low_idx = [], []

    for i in range(n, n_bars - n - 1):
        h = highs[i]
        l = lows[i]

        is_sh = (
            h > np.max(highs[i - n: i]) and
            h > np.max(highs[i + 1: i + n + 1]) and
            closes[i + 1] < h
        )
        if is_sh:
            if atr_val and atr_val > 0:
                local_low  = np.min(lows[max(0, i - n): i + n + 1])
                amplitude  = h - local_low
                if amplitude < atr_val * prominence_coeff:
                    is_sh = False
        if is_sh:
            swing_high_idx.append(i)

        is_sl = (
            l < np.min(lows[i - n: i]) and
            l < np.min(lows[i + 1: i + n + 1]) and
            closes[i + 1] > l
        )
        if is_sl:
            if atr_val and atr_val > 0:
                local_high = np.max(highs[max(0, i - n): i + n + 1])
                amplitude  = local_high - l
                if amplitude < atr_val * prominence_coeff:
                    is_sl = False
        if is_sl:
            swing_low_idx.append(i)

    pivot_highs = pd.Series(highs[swing_high_idx], index=swing_high_idx) if swing_high_idx else pd.Series(dtype=float)
    pivot_lows  = pd.Series(lows[swing_low_idx],   index=swing_low_idx)  if swing_low_idx  else pd.Series(dtype=float)
    return pivot_highs, pivot_lows


# ══════════════════════════════════════════════════════════════════
# STATUT DE ZONE
# ══════════════════════════════════════════════════════════════════
def classify_zone_status(level, zone_type, df, formation_idx,
                          atr_val=None, tolerance_coeff=0.25):
    if formation_idx >= len(df) - 1:
        return "Vierge"

    tolerance = (atr_val * tolerance_coeff) if (atr_val and atr_val > 0) else (level * 0.003)

    closes_after = df["close"].iloc[formation_idx + 1:]
    highs_after  = df["high"].iloc[formation_idx + 1:]
    lows_after   = df["low"].iloc[formation_idx + 1:]

    has_approach = False

    for i in range(len(closes_after)):
        c = closes_after.iloc[i]
        h = highs_after.iloc[i]
        l = lows_after.iloc[i]

        near_zone = abs(c - level) <= tolerance or (l <= level + tolerance and h >= level - tolerance)

        if near_zone:
            has_approach = True

        if zone_type == "Support" and c < level - tolerance:
            return "Consommee"
        if zone_type == "Resistance" and c > level + tolerance:
            return "Consommee"

    return "Testee" if has_approach else "Vierge"


# ══════════════════════════════════════════════════════════════════
# DÉTECTION S/R
# ══════════════════════════════════════════════════════════════════
def find_strong_sr_zones(df, current_price, atr_val=None,
                          zone_percentage_width=0.5,
                          atr_zone_coeff=0.4,
                          prominence_coeff=0.3,
                          min_touches=2, timeframe="daily",
                          post_merge_threshold=0.30,
                          swing_n=3):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None:
        current_price = df["close"].iloc[-1]

    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(
        df, n=swing_n, atr_val=atr_val, prominence_coeff=prominence_coeff
    )

    if len(pivot_highs) + len(pivot_lows) < 3:
        distance = get_adaptive_distance(timeframe)
        if atr_val and atr_val > 0:
            pk = {"distance": distance, "prominence": atr_val * prominence_coeff}
        else:
            pk = {"distance": distance}
        r_idx, _ = find_peaks( df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values,  **pk)
        pivot_highs = pd.Series(df["high"].values[r_idx], index=r_idx) if len(r_idx) else pd.Series(dtype=float)
        pivot_lows  = pd.Series(df["low"].values[s_idx],  index=s_idx) if len(s_idx) else pd.Series(dtype=float)

    if pivot_highs.empty and pivot_lows.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivots_with_idx = []
    for idx, price in pivot_highs.items():
        pivots_with_idx.append((float(price), int(idx)))
    for idx, price in pivot_lows.items():
        pivots_with_idx.append((float(price), int(idx)))

    pivots_with_idx.sort(key=lambda x: x[0])
    if not pivots_with_idx:
        return pd.DataFrame(), pd.DataFrame()

    zones_raw   = []
    cur_group   = [pivots_with_idx[0]]

    for price, idx in pivots_with_idx[1:]:
        avg = np.mean([p for p, _ in cur_group])
        zone_width_abs = (atr_val * atr_zone_coeff) if (atr_val and atr_val > 0) \
                         else (avg * zone_percentage_width / 100)
        if abs(price - avg) < zone_width_abs:
            cur_group.append((price, idx))
        else:
            zones_raw.append(cur_group)
            cur_group = [(price, idx)]
    zones_raw.append(cur_group)

    strong = []
    for grp in zones_raw:
        if len(grp) < min_touches:
            continue
        prices   = [p for p, _ in grp]
        indices  = [i for _, i in grp]
        lvl      = float(np.mean(prices))
        strength = len(grp)
        last_idx = max(indices)
        age_bars = n_total - 1 - last_idx

        zone_type_tmp = "Support" if lvl < current_price else "Resistance"
        status = classify_zone_status(
            lvl, zone_type_tmp, df, last_idx,
            atr_val=atr_val, tolerance_coeff=0.25
        )

        strong.append({
            "level":    lvl,
            "strength": strength,
            "age_bars": age_bars,
            "status":   status,
        })

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
            all_zones.append({
                "tf":       tf,
                "level":    z["level"],
                "strength": z["strength"],
                "age_bars": z.get("age_bars", 0),
                "status":   z.get("status", "Testee"),
                "type":     "Support",
            })
        for _, z in resistances.iterrows():
            all_zones.append({
                "tf":       tf,
                "level":    z["level"],
                "strength": z["strength"],
                "age_bars": z.get("age_bars", 0),
                "status":   z.get("status", "Testee"),
                "type":     "Resistance",
            })

    if not all_zones:
        return []

    zones_df     = pd.DataFrame(all_zones).sort_values("level")
    used_indices = set()
    confluences  = []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Consommee": 2}

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue

        similar = zones_df[
            (abs(zones_df["level"] - zone["level"]) / zone["level"] * 100 <= confluence_threshold) &
            (zones_df.index != i)
        ]

        if len(similar) >= 1:
            group      = pd.concat([zones_df.loc[[i]], similar])
            timeframes = group["tf"].unique()

            if len(timeframes) >= 2:
                avg_level  = group["level"].mean()
                nb_tf      = len(timeframes)
                dist_pct   = abs(current_price - avg_level) / current_price * 100

                total_score = 0.0
                for _, row in group.iterrows():
                    total_score += compute_structural_score(
                        int(row["strength"]),
                        nb_tf,
                        tf_name  = row["tf"],
                        age_bars = int(row.get("age_bars", 0)),
                        total_bars = 500,
                    )

                total_strength = int(group["strength"].sum())

                best_status = min(
                    group["status"].tolist(),
                    key=lambda s: STATUS_PRIORITY.get(s, 1)
                )

                zone_type = group.iloc[0]["type"]
                signal    = "🟢 BUY ZONE" if zone_type == "Support" else "🔴 SELL ZONE"
                tf_label  = " + ".join(sorted(timeframes))
                alerte    = "🔥 ZONE CHAUDE" if dist_pct < 0.5 else ("⚠️ Proche" if dist_pct < 1.5 else "")

                confluences.append({
                    "Actif":        symbol,
                    "Signal":       signal,
                    "Niveau":       f"{avg_level:.5f}",
                    "Type":         zone_type,
                    "Timeframes":   tf_label,
                    "Nb TF":        nb_tf,
                    "Force Totale": total_strength,
                    "Score":        round(total_score, 1),
                    "Statut":       best_status,
                    "Distance %":   f"{dist_pct:.2f}%",
                    "Alerte":       alerte,
                })
                used_indices.update(group.index)

    return confluences


# ══════════════════════════════════════════════════════════════════
# SCAN PARALLÈLE PAR SYMBOLE
# ══════════════════════════════════════════════════════════════════
def scan_single_symbol(args):
    symbol, base_url, access_token, account_id, zone_width, min_touches = args

    raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
    current_price, price_alert_msg = validate_live_price(
        raw_price, symbol, base_url, access_token
    )

    atr_zone_coeff    = ATR_ZONE_COEFF.get(symbol, DEFAULT_ATR_COEFF)
    prom_coeff        = PROMINENCE_COEFF.get(symbol, DEFAULT_PROMINENCE_COEFF)
    zone_w_fallback   = ZONE_WIDTH_FALLBACK.get(symbol, zone_width)
    merge_thresh      = POST_MERGE_MAP.get(symbol, POST_MERGE_THRESHOLD)
    pdf_dist_max      = PDF_DIST_THRESHOLDS.get(symbol, DEFAULT_PDF_DIST)
    abs_dist_max      = ABSOLUTE_MAX_DIST.get(symbol, 99.0)

    rows           = {"H4": None, "Daily": None, "Weekly": None}
    zones_d        = {}
    trends         = {}
    all_sup_levels = []
    last_h4_close  = None
    anomaly_msg    = price_alert_msg

    for tf_key, tf_cap in [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]:
        df = get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)
        if df is None or df.empty:
            continue

        cp = current_price if current_price is not None else df["close"].iloc[-1]

        if tf_key == "h4":
            last_h4_close = df["close"].iloc[-1]

        trends[tf_cap] = compute_trend(df)

        atr_val = compute_atr(df, period=14)

        supports, resistances = find_strong_sr_zones(
            df, cp,
            atr_val             = atr_val,
            zone_percentage_width = zone_w_fallback,
            atr_zone_coeff      = atr_zone_coeff,
            prominence_coeff    = prom_coeff,
            min_touches         = min_touches,
            timeframe           = tf_key,
            post_merge_threshold = merge_thresh,
        )
        zones_d[tf_cap] = (supports, resistances)

        if not supports.empty:
            all_sup_levels.extend(supports["level"].tolist())

        if tf_key == "daily":
            price_ctx = get_price_context(cp, supports, resistances)
            zones_d["_price_ctx"] = price_ctx

        sym_d = symbol.replace("_", "/")

        tf_weight_name = tf_cap
        n_total_bars   = len(df)

        def make_row(zone, ztype, _cp=cp, _atr=atr_val,
                     _pdf_max=pdf_dist_max, _abs_max=abs_dist_max,
                     _tf=tf_weight_name, _ntot=n_total_bars):
            lvl      = zone["level"]
            strength = int(zone["strength"])
            age_bars = int(zone.get("age_bars", 0))
            status   = zone.get("status", "Testee")
            dist_pct = abs(_cp - lvl) / _cp * 100
            dist_atr = round(abs(_cp - lvl) / _atr, 1) if (_atr and _atr > 0) else np.nan
            struct_score = compute_structural_score(strength, 1, _tf, age_bars, _ntot)
            return {
                "Actif":       sym_d,
                "Prix Actuel": f"{_cp:.5f}",
                "Type":        ztype,
                "Niveau":      f"{lvl:.5f}",
                "Force":       f"{strength} touches",
                "Score":       struct_score,
                "Statut":      status,
                "Dist. %":     f"{dist_pct:.2f}%",
                "Dist. ATR":   f"{dist_atr:.1f}x" if not np.isnan(dist_atr) else "N/A",
                "_dist_num":   dist_pct,
                "_in_pdf":     dist_pct <= _pdf_max and dist_pct <= _abs_max,
            }

        tf_rows = (
            [make_row(z, "Support")    for _, z in supports.iterrows()] +
            [make_row(z, "Resistance") for _, z in resistances.iterrows()]
        )
        if tf_rows:
            rows[tf_cap] = tf_rows

    if all_sup_levels and current_price:
        new_anomaly = flag_data_anomaly(
            symbol, current_price, all_sup_levels, last_h4_close
        )
        if new_anomaly:
            anomaly_msg = f"{anomaly_msg} | {new_anomaly}" if anomaly_msg else new_anomaly

    return symbol, rows, zones_d, current_price, trends, anomaly_msg


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION PDF  — entièrement reécrit avec _safe_pdf_str()
# ══════════════════════════════════════════════════════════════════
def strip_emojis_df(df):
    """Nettoie un DataFrame pour l'export PDF : emojis + accents."""
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        clean[col] = clean[col].astype(str).apply(_safe_pdf_str)
    return clean


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10,
                  _safe_pdf_str('Rapport de Scan Support/Resistance - v4'),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6,
                  _safe_pdf_str(
                      f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  |  "
                      "v4 - Score = (Force x Poids_TF x NbTF) x Facteur_Age | "
                      "Statut Vierge/Testee/Consommee"
                  ),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, _safe_pdf_str(title),
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def chapter_summary(self, summaries):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7,
                  _safe_pdf_str('RESUME PAR ACTIF  (Tendances + Top Zones Confluentes)'),
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

        for s in summaries:
            sym  = _safe_pdf_str(s.get('symbol', ''))
            t_h4 = _safe_pdf_str(s.get('trend_h4',    'N/A'))
            t_d  = _safe_pdf_str(s.get('trend_daily',  'N/A'))
            t_w  = _safe_pdf_str(s.get('trend_weekly', 'N/A'))
            ctx  = _safe_pdf_str(s.get('price_context', ''))

            self.set_font('Helvetica', 'B', 8)
            self.cell(0, 5,
                      _safe_pdf_str(f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"),
                      border=0, new_x='LMARGIN', new_y='NEXT')

            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}",
                          border=0, new_x='LMARGIN', new_y='NEXT')

            top = s.get('top_zones', [])
            self.set_font('Helvetica', '', 7)
            if top:
                for z in top:
                    sig  = _safe_pdf_str(str(z.get('Signal', '')))
                    niv  = _safe_pdf_str(str(z.get('Niveau',     '')))
                    dist = _safe_pdf_str(str(z.get('Distance %', '')))
                    sc   = _safe_pdf_str(str(z.get('Score',      '')))
                    tfs  = _safe_pdf_str(str(z.get('Timeframes', '')))
                    ale  = _safe_pdf_str(str(z.get('Alerte',     '')))
                    txt  = f"  {sig}  Niv:{niv}  Dist:{dist}  Score:{sc}  TF:{tfs}  {ale}"
                    self.cell(0, 4, txt[:130], border=0, new_x='LMARGIN', new_y='NEXT')
            else:
                self.cell(0, 4, "  Aucune confluence pour cet actif.",
                          border=0, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)

    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.set_x(self.l_margin)
            usable_w = self.w - self.l_margin - self.r_margin
            self.multi_cell(usable_w, 10, "Aucune donnee a afficher.")
            self.ln()
            return

        if 'Timeframes' in df.columns:
            col_widths = {
                'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22,
                'Timeframes': 50, 'Nb TF': 12, 'Force Totale': 20,
                'Score': 18, 'Statut': 22, 'Distance %': 18, 'Alerte': 55,
            }
        else:
            col_widths = {
                'Actif': 24, 'Prix Actuel': 24, 'Type': 20,
                'Niveau': 24, 'Force': 20,
                'Score': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16,
            }
        font_size = 7

        cols     = [c for c in col_widths if c in df.columns]
        total_w  = sum(col_widths[c] for c in cols)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start  = self.l_margin + max(0, (usable_w - total_w) / 2)

        self.set_font('Helvetica', 'B', font_size)
        self.set_x(x_start)
        for col_name in cols:
            self.cell(col_widths[col_name], 6,
                      _safe_pdf_str(col_name),
                      border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()

        self.set_font('Helvetica', '', font_size)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col_name in cols:
                w        = col_widths[col_name]
                val      = _safe_pdf_str(str(row[col_name]))  # ← FIX CENTRAL
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
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        for a in anomalies:
            line = _safe_pdf_str(f"  {a['actif']} : {a['msg']}")
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, 5, line)
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
        "Score":       st.column_config.NumberColumn("Score ▼",   width="small"),
        "Statut":      st.column_config.TextColumn("Statut",      width="small"),
        "Dist. %":     st.column_config.TextColumn("Dist. %",     width="small"),
        "Dist. ATR":   st.column_config.TextColumn("Dist. ATR",   width="small"),
    }

    if anomalies:
        with st.expander(f"⚡ {len(anomalies)} alerte(s) qualité des données", expanded=True):
            for a in anomalies:
                st.warning(f"**{a['actif']}** : {a['msg']}")

    if not conf_filt.empty:
        st.divider()
        st.subheader("🔥 ZONES DE CONFLUENCE MULTI-TIMEFRAMES")
        st.caption("Score = Force × Nb TF (structurel pur) | Distance affichée séparément")

        disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total zones",       len(disp))
        c2.metric("🔥 Zones chaudes",  len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
        c3.metric("⚠️ Zones proches",  len(disp[disp["Alerte"] == "⚠️ Proche"]))
        c4.metric("🟢 BUY Zones",      len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
        c5.metric("🔴 SELL Zones",     len(disp[disp["Signal"] == "🔴 SELL ZONE"]))

        conf_cfg = {
            **{k: st.column_config.TextColumn(k, width="small")
               for k in ["Actif","Signal","Niveau","Type","Timeframes","Statut","Distance %","Alerte"]},
            "Nb TF":        st.column_config.NumberColumn("Nb TF",        width="small"),
            "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
            "Score":        st.column_config.NumberColumn("Score ▼",      width="small"),
        }
        st.dataframe(disp, column_config=conf_cfg, hide_index=True,
                     width='stretch', height=min(len(disp) * 35 + 38, 750))
    else:
        st.info("Aucune confluence dans la plage sélectionnée. Augmentez le filtre ou le seuil.")

    st.subheader("📋 Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_pdf_report(rep_dict, conf_full, summaries, anomalies)
            st.download_button(
                "📄 Télécharger le Rapport (PDF)",
                data=pdf_bytes,
                file_name=f"rapport_sr_v4_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width='stretch',
            )
        with col2:
            csv_bytes = create_csv_report(rep_dict, conf_full)
            st.download_button(
                "📊 Télécharger les Données (CSV)",
                data=csv_bytes,
                file_name=f"donnees_sr_v4_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width='stretch',
            )

    def _filter_and_sort(df, max_pct):
        if df.empty:
            return df
        def to_float(s):
            try:   return float(str(s).replace("%", ""))
            except: return 999.0
        mask = df["Dist. %"].apply(to_float) <= max_pct
        out  = df[mask].drop(columns=["_in_pdf", "_dist_num"], errors="ignore")
        if "Score" in out.columns:
            out = out.sort_values("Score", ascending=False)
        return out.reset_index(drop=True)

    st.divider()
    st.subheader("📅 Analyse 4 Heures (H4)")
    fd = _filter_and_sort(df_h4, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 width='stretch', height=min(len(fd) * 35 + 38, 600))

    st.subheader("📅 Analyse Journalière (Daily)")
    fd = _filter_and_sort(df_daily, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 width='stretch', height=min(len(fd) * 35 + 38, 600))

    st.subheader("📅 Analyse Hebdomadaire (Weekly)")
    fd = _filter_and_sort(df_wk, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True,
                 width='stretch', height=min(len(fd) * 35 + 38, 600))


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
        "Largeur zone Forex (% fallback si ATR indispo)", 0.1, 2.0, 0.5, 0.1,
        help="Normalement remplacé par ATR × coeff. Utilisé si ATR non disponible.",
    )
    min_touches = st.slider(
        "Force minimale (touches)", 2, 10, 3, 1,
        help="Nombre de contacts minimum pour valider une zone.",
    )
    confluence_threshold = st.slider(
        "Seuil confluence Forex (%)", 0.3, 2.0, 1.0, 0.1,
        help="Indices/Métaux utilisent des seuils adaptatifs (1.2-1.5%) automatiquement.",
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) - filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
    )

    st.divider()
    st.caption("**Score v4 = (Force × Poids_TF × NbTF) × Facteur_Age**")
    st.caption("🔴 > 300 : Zone institutionnelle majeure")
    st.caption("🟠 100-300 : Zone structurelle forte")
    st.caption("🟡 30-100 : Zone technique valide")
    st.caption("⚪ < 30  : Zone secondaire")

    st.divider()
    st.caption("**v4 - Améliorations moteur :**")
    st.caption("✅ Swing HH/LL confirmé (clôture suivante)")
    st.caption("✅ Score pondéré TF (Weekly=3× H4) + âge")
    st.caption("✅ Statut : Vierge / Testée / Consommée")
    st.caption("✅ Prominence ATR + largeur zone ATR-based")
    st.caption("✅ Plage prix valides (XAU/XAG/XPT/Indices)")
    st.caption("✅ Fix PDF Unicode (accents + emojis)")


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
            progress_bar = st.progress(0, text="Initialisation du scan v4…")

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
                        text=f"Scan v4… ({completed}/{total}) {sym.replace('_', '/')}",
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
                                if tf_cap == "H4":      results_h4.extend(tf_rows)
                                elif tf_cap == "Daily": results_daily.extend(tf_rows)
                                elif tf_cap == "Weekly": results_weekly.extend(tf_rows)
                    except Exception:
                        pass

            progress_bar.empty()
            st.success(f"✅ Scan v4 terminé - {len(symbols_to_scan)} actifs analysés.")

            st.info("🔍 Analyse des confluences multi-timeframes…")
            all_confluences = []
            for sym in symbols_to_scan:
                cp = prices_map.get(sym)
                zones_clean = {k: v for k, v in all_zones_map.get(sym, {}).items()
                               if not k.startswith("_")}
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
                    "trend_h4":      trends.get("H4",     "NEUTRE"),
                    "trend_daily":   trends.get("Daily",  "NEUTRE"),
                    "trend_weekly":  trends.get("Weekly", "NEUTRE"),
                    "price_context": price_ctx,
                    "top_zones":     top_zones,
                })

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

if "scan_results" in st.session_state and not scan_button:
    _display_results(
        st.session_state["scan_results"],
        st.session_state["scan_results"].get("max_dist", 3.0),
    )
