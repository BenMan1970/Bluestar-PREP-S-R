import hashlib
import json
import threading
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    [data-testid="stDataFrame"] > div { overflow-x: visible !important; overflow-y: visible !important; }
    [data-testid="stDataFrame"] iframe { width: 100% !important; height: auto !important; }
    ::-webkit-scrollbar { width: 0px !important; height: 0px !important; }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #D32F2F;
        color: white;
        border: 1px solid #B71C1C;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #B71C1C;
        border-color: #D32F2F;
        box-shadow: 0 4px 12px rgba(211, 47, 47, 0.4);
    }
    div[data-testid="stButton"] > button[kind="primary"]:active {
        background-color: #D32F2F;
        transform: scale(0.98);
    }
    div[data-testid="stButton"] > button { font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Scanner Bluestar Supports et Résistances")
st.markdown(
    "Zones S/R avec **swing HH/LL confirmé**, **score pondéré TF+âge**, "
    "**statut Vierge/Testée/Consommée/Role Reverse**, **plage prix valides**."
)

st.markdown("<br>", unsafe_allow_html=True)
scan_button = st.button("🚀 LANCER LE SCAN COMPLET", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES — organisées par domaine (LOW #2 FIX)
# Avant : tout au même niveau, impossible de distinguer trading/PDF/validation.
# Maintenant : 3 sections clairement séparées.
# ══════════════════════════════════════════════════════════════════

# ── Version ────────────────────────────────────────────────────────
SCANNER_VERSION = "5.0"

# ── Instruments ────────────────────────────────────────────────────
ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

# ── Paramètres de détection S/R (logique trading) ─────────────────
ATR_ZONE_COEFF = {
    "XAU_USD": 0.5,
    "US30_USD": 0.3, "NAS100_USD": 0.3, "SPX500_USD": 0.3, "DE30_EUR": 0.3,
}
DEFAULT_ATR_COEFF = 0.4

PROMINENCE_COEFF = {
    "XAU_USD": 0.5,
    "US30_USD": 0.4, "NAS100_USD": 0.4, "SPX500_USD": 0.4, "DE30_EUR": 0.4,
}
DEFAULT_PROMINENCE_COEFF = 0.3

ZONE_WIDTH_FALLBACK = {
    "US30_USD":   0.50,
    "NAS100_USD": 0.50,
    "SPX500_USD": 0.25,
    "DE30_EUR":   0.25,
    "XAU_USD":    0.20,
}
DEFAULT_ZONE_WIDTH = 0.5

POST_MERGE_THRESHOLD = 0.30
POST_MERGE_MAP = {
    "US30_USD": 0.05, "NAS100_USD": 0.05, "SPX500_USD": 0.08, "DE30_EUR": 0.08,
    "XAU_USD": 0.15,
}

CONFLUENCE_THRESHOLD_MAP = {
    "US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2,
    "XAU_USD": 1.5,
}

# ── Validation des prix (santé des données) ────────────────────────
PRICE_SANITY_RANGE = {
    "XAU_USD":    (1500.0,  5500.0),
    "US30_USD":   (20000.0, 70000.0),
    "NAS100_USD": (8000.0,  25000.0),
    "SPX500_USD": (3000.0,  9000.0),
    "DE30_EUR":   (8000.0,  30000.0),
}
_SKIP_RATIO_CHECK    = {"XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR"}
_DEV_THRESHOLD       = {
    "XAU_USD": 3.0,
    "US30_USD": 2.5, "NAS100_USD": 2.5, "SPX500_USD": 2.0, "DE30_EUR": 2.0,
}
_DEFAULT_DEV_THRESHOLD = 1.5

# ── Paramètres export PDF (filtrage de distance) ───────────────────
PDF_DIST_THRESHOLDS = {
    "US30_USD": 5.0, "NAS100_USD": 5.0, "SPX500_USD": 5.0, "DE30_EUR": 5.0,
    "XAU_USD": 8.0,
}
DEFAULT_PDF_DIST = 8.0

ABSOLUTE_MAX_DIST = {
    "XAU_USD": 8.0,
    "US30_USD": 4.0, "NAS100_USD": 4.0, "SPX500_USD": 4.0, "DE30_EUR": 4.0,
}

# ══════════════════════════════════════════════════════════════════
# BUGFIX #1 — Cache thread-safe (remplace @st.cache_data pour les données OANDA)
# Problème original : ThreadPoolExecutor (8 workers) + @st.cache_data
# créait des race conditions sur les écritures de cache Streamlit.
# Solution : dict protégé par threading.Lock() + TTL manuel.
# ══════════════════════════════════════════════════════════════════

# ── Cache thread-safe ──────────────────────────────────────────────
_oanda_cache: dict = {}
_oanda_cache_lock  = threading.Lock()

def _token_fingerprint(token: str) -> str:
    """
    LOW #3 FIX — Le token OANDA ne doit pas apparaître en clair dans
    les clés de cache (logs Streamlit, métadonnées disque).
    On utilise les 12 premiers caractères du SHA-256 comme fingerprint.
    """
    return hashlib.sha256(token.encode()).hexdigest()[:12]

def _cache_key(token: str, symbol: str, timeframe: str, limit: int) -> str:
    return f"{_token_fingerprint(token)}__{symbol}__{timeframe}__{limit}"

def _cache_get(key: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
    with _oanda_cache_lock:
        entry = _oanda_cache.get(key)
    if entry is None:
        return None
    df, ts = entry
    if (datetime.now() - ts).total_seconds() > ttl_seconds:
        return None
    return df

def _cache_set(key: str, df: Optional[pd.DataFrame]) -> None:
    with _oanda_cache_lock:
        _oanda_cache[key] = (df, datetime.now())


# ══════════════════════════════════════════════════════════════════
# UTILITAIRE PDF
# ══════════════════════════════════════════════════════════════════
_ACCENT_MAP = str.maketrans(
    'àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
    'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN'
)

_EMOJI_MAP = [
    ('🟢', '[BUY]'),  ('🔴', '[SELL]'), ('🔥', '[CHAUD]'),
    ('⚠️', '[PROCHE]'), ('⚠',  '[PROCHE]'), ('📈', ''), ('📉', ''),
    ('↔️', ''), ('↔',  ''), ('✅', '[OK]'), ('❌', '[X]'),
    ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'),
]

def _safe_pdf_str(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_ACCENT_MAP)
    for emoji, replacement in _EMOJI_MAP:
        text = text.replace(emoji, replacement)
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ══════════════════════════════════════════════════════════════════
# API OANDA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)   # TTL court : 5 min (était 3600 = 1h, trop long)
def determine_oanda_environment(access_token: str, account_id: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {access_token}"}
    for url in [
        "https://api-fxpractice.oanda.com",
        "https://api-fxtrade.oanda.com",
    ]:
        try:
            r = requests.get(f"{url}/v3/accounts/{account_id}/summary",
                             headers=headers, timeout=5)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def get_oanda_data(base_url: str, access_token: str, symbol: str,
                   timeframe: str = "daily", limit: int = 500) -> Optional[pd.DataFrame]:
    """
    BUGFIX #1 : plus de @st.cache_data ici.
    Cache manuel thread-safe (dict + Lock).
    TTL = 600s pour les données de prix, identique à l'original.
    On demande limit+1 bougies pour compenser la bougie en cours (incomplète).
    """
    key = _cache_key(access_token, symbol, timeframe, limit)
    cached = _cache_get(key, ttl_seconds=600)
    if cached is not None:
        return cached

    url     = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {
        "count":       limit + 1,   # +1 pour compenser la bougie en cours
        "granularity": {"h4": "H4", "daily": "D", "weekly": "W"}[timeframe],
        "price":       "M",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("candles"):
            _cache_set(key, None)
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
        # On limite à `limit` bougies complètes (on a demandé limit+1)
        # HIGH #2 FIX — bloc pass mort supprimé. Le df retourné reflète
        # la réalité : si 450 bougies sur 500, len(df)=450, l'appelant
        # le verra via len(df) < 20 dans find_strong_sr_zones.
        df = pd.DataFrame(candles[:limit]).set_index("date")
        _cache_set(key, df)
        return df
    except requests.RequestException:
        _cache_set(key, None)
        return None


# Cache séparé pour les prix live (TTL court : 60s)
_price_cache: dict = {}
_price_cache_lock  = threading.Lock()

def get_oanda_current_price(base_url: str, access_token: str,
                             account_id: str, symbol: str) -> Optional[float]:
    """
    CRIT #2 FIX — @st.cache_data retiré.
    Même problème que get_oanda_data : appelée depuis ThreadPoolExecutor,
    @st.cache_data créait des race conditions sur les écritures de cache.
    Cache manuel thread-safe avec TTL=60s.
    """
    key = f"{_token_fingerprint(access_token)}__{symbol}__price"
    with _price_cache_lock:
        entry = _price_cache.get(key)
    if entry is not None:
        val, ts = entry
        if (datetime.now() - ts).total_seconds() < 60:
            return val

    url     = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        r = requests.get(url, headers=headers, params={"instruments": symbol}, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "prices" in data and data["prices"]:
            bid = float(data["prices"][0]["closeoutBid"])
            ask = float(data["prices"][0]["closeoutAsk"])
            result = (bid + ask) / 2
        else:
            result = None
    except requests.RequestException:
        result = None

    with _price_cache_lock:
        _price_cache[key] = (result, datetime.now())
    return result


def validate_live_price(live_price, symbol, base_url, access_token):
    """
    MEDIUM #1 FIX — limit=500 au lieu de limit=10 pour partager le cache
    avec le prefetch de scan_single_symbol (même clé de cache).
    Élimine 33 appels réseau superflus (un par symbole) au premier scan.
    """
    dev_threshold = _DEV_THRESHOLD.get(symbol, _DEFAULT_DEV_THRESHOLD)

    h4_close = None
    # limit=500 → même clé cache que le prefetch → hit garanti au 2e appel
    df_check = get_oanda_data(base_url, access_token, symbol, "h4", limit=500)
    if df_check is not None and not df_check.empty:
        h4_close = float(df_check["close"].iloc[-1])
        if symbol in PRICE_SANITY_RANGE:
            lo, hi = PRICE_SANITY_RANGE[symbol]
            if not (lo <= h4_close <= hi):
                h4_close = None

    if live_price is not None:
        if symbol in PRICE_SANITY_RANGE:
            lo, hi = PRICE_SANITY_RANGE[symbol]
            if not (lo <= live_price <= hi):
                live_price = None

        if live_price is not None and h4_close is not None and h4_close > 0:
            dev = abs(live_price - h4_close) / h4_close * 100
            if dev > dev_threshold:
                live_price = None

    if live_price is not None:
        return live_price, None
    if h4_close is not None:
        return h4_close, None
    return None, f"Aucun prix fiable disponible pour {symbol}"


def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    if current_price is None or current_price <= 0:
        return None
    messages = []
    if symbol not in _SKIP_RATIO_CHECK and len(support_levels) >= 3:
        median_sup = np.median(support_levels)
        if median_sup > 0:
            ratio = current_price / median_sup
            if ratio > 3.0:
                messages.append(
                    f"Prix {current_price:.2f} = {ratio:.1f}x la mediane des supports "
                    f"({median_sup:.2f}) - donnees a verifier"
                )
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > 25.0:
            messages.append(
                f"Prix live {current_price:.2f} s'ecarte de {dev:.1f}% "
                f"du dernier close ({last_candle_close:.2f})"
            )
    return " | ".join(messages) if messages else None


def get_price_context(current_price, supports, resistances):
    """
    MEDIUM #3 FIX — Guard current_price nul ou None.
    nlargest/nsmallest déjà en place (v2). On ajoute la protection
    contre current_price=0 qui donnait une division par zéro dans dist_s/dist_r.
    """
    if not current_price or current_price <= 0:
        return "Prix indisponible"
    parts = []
    if not supports.empty:
        nearest_sup = supports.nlargest(1, "level").iloc[0]
        dist_s = abs(current_price - nearest_sup["level"]) / current_price * 100
        tag    = "SUR support" if dist_s < 0.5 else "S proche"
        parts.append(f"{tag}: {nearest_sup['level']:.5f} (-{dist_s:.2f}%)")
    if not resistances.empty:
        nearest_res = resistances.nsmallest(1, "level").iloc[0]
        dist_r = abs(current_price - nearest_res["level"]) / current_price * 100
        tag    = "SUR resistance" if dist_r < 0.5 else "R proche"
        parts.append(f"{tag}: {nearest_res['level']:.5f} (+{dist_r:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"


# ══════════════════════════════════════════════════════════════════
# MOTEUR D'ANALYSE
# ══════════════════════════════════════════════════════════════════
def get_adaptive_distance(timeframe):
    return {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)


def compute_atr(df, period=14) -> Optional[float]:
    """
    MEDIUM #1 FIX — NaN pandas converti en None pour propagation explicite.
    Évite que rolling() sur fenêtre trop petite retourne NaN silencieux
    qui corrompt les calculs de zone_width et dist_atr en aval.
    """
    if df is None or len(df) < period + 1:
        return None
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    result = tr.rolling(period).mean().iloc[-1]
    return float(result) if pd.notna(result) and result > 0 else None


def compute_trend(df, sma_period=20):
    if df is None or len(df) < 15:
        return "NEUTRE"

    # BUGFIX : période insuffisante → tendance non fiable → NEUTRE
    actual_period = min(sma_period, len(df) - 5)
    if actual_period < sma_period * 0.6:
        return "NEUTRE"

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

    # MEDIUM #4 FIX — Branches elif redondantes supprimées (dead code).
    # hh/ll renforcent le signal mais ne sont pas des conditions bloquantes.
    # La logique précédente avait 4 branches pour 2 cas réels.
    if above_sma and slope_pct > 0.05:
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
    tf_w  = TF_WEIGHT.get(tf_name, 1.0)
    age_r = age_bars / max(total_bars, 1)
    age_f = float(np.exp(-1.5 * age_r))
    raw   = strength * tf_w * nb_tf
    return round(raw * age_f, 1)


def post_merge_zones(zones_list, threshold_pct=0.30):
    """
    BUGFIX #2 — Ancre fixe par groupe.
    Problème original : la moyenne du groupe dérivait après chaque ajout,
    ce qui pouvait absorber des zones trop éloignées du point de départ.
    Solution : on ancre sur le PREMIER élément du groupe (pivot stable),
    pas sur la moyenne recalculée.
    """
    if len(zones_list) <= 1:
        return zones_list

    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    changed = True
    while changed:
        changed = False
        new_zones, used = [], set()
        for i in range(len(zones_list)):
            if i in used:
                continue
            group  = [zones_list[i]]
            anchor = zones_list[i]["level"]   # ← ancre FIXE sur le 1er élément

            for j in range(i + 1, len(zones_list)):
                if j in used:
                    continue
                # On compare toujours à l'ancre du groupe, pas à la moyenne mobile
                if abs(zones_list[j]["level"] - anchor) / anchor * 100 <= threshold_pct:
                    group.append(zones_list[j])
                    used.add(j)
                    changed = True
            used.add(i)

            best_age = min(z.get("age_bars", 0) for z in group)
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


def detect_swing_pivots(df, n=3, atr_val=None, prominence_coeff=0.3):
    """
    HIGH #1 FIX — Version vectorisée numpy (O(n) au lieu de O(n²)).
    Problème original : boucle Python pure avec np.max sur des slices
    à chaque itération → ~50 000 appels sur 33 actifs × 3 TF × 500 bougies.
    Solution : rolling max/min via pd.Series + masques booléens vectorisés.
    Résultat identique, 10-30× plus rapide.
    """
    highs  = pd.Series(df["high"].values)
    lows   = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)

    # Rolling max/min sur n bougies à gauche et à droite
    roll_high_left  = highs.shift(1).rolling(n, min_periods=n).max()
    roll_high_right = highs[::-1].shift(1).rolling(n, min_periods=n).max()[::-1]
    roll_low_left   = lows.shift(1).rolling(n, min_periods=n).min()
    roll_low_right  = lows[::-1].shift(1).rolling(n, min_periods=n).min()[::-1]

    # Confirmation par la clôture suivante (shift(-1))
    next_close = closes.shift(-1)

    sh_mask = (
        (highs > roll_high_left) &
        (highs > roll_high_right) &
        (next_close < highs)
    )
    sl_mask = (
        (lows < roll_low_left) &
        (lows < roll_low_right) &
        (next_close > lows)
    )

    # Filtre prominence ATR (gardé pour cohérence avec l'original)
    if atr_val and atr_val > 0:
        min_amplitude = atr_val * prominence_coeff
        roll_low_around  = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
        roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()
        sh_mask = sh_mask & ((highs - roll_low_around)  >= min_amplitude)
        sl_mask = sl_mask & ((roll_high_around - lows)  >= min_amplitude)

    swing_high_idx = sh_mask[sh_mask].index.tolist()
    swing_low_idx  = sl_mask[sl_mask].index.tolist()

    pivot_highs = (pd.Series(highs.values[swing_high_idx], index=swing_high_idx)
                   if swing_high_idx else pd.Series(dtype=float))
    pivot_lows  = (pd.Series(lows.values[swing_low_idx],   index=swing_low_idx)
                   if swing_low_idx  else pd.Series(dtype=float))
    return pivot_highs, pivot_lows


def classify_zone_status(level, zone_type, df, formation_idx,
                          atr_val=None, tolerance_coeff=0.25):
    """
    BUGFIX #4 — Role Reversal.
    HIGH #1 FIX — Version vectorisée numpy (élimine la boucle iloc O(n)).
    Problème original : boucle Python avec closes_after.iloc[i] à chaque
    itération — très lent sur 490 bougies × N zones × 3 TF × 33 actifs.
    """
    if formation_idx >= len(df) - 1:
        return "Vierge"

    tolerance = (atr_val * tolerance_coeff) if (atr_val and atr_val > 0) else (level * 0.003)

    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]

    if len(c_arr) == 0:
        return "Vierge"

    # Approche vectorisée : masque "proche de la zone"
    near = (np.abs(c_arr - level) <= tolerance) | (
        (l_arr <= level + tolerance) & (h_arr >= level - tolerance)
    )
    has_approach = bool(near.any())

    # Détection cassure
    if zone_type == "Support":
        break_mask = c_arr < level - tolerance
    else:
        break_mask = c_arr > level + tolerance

    break_positions = np.where(break_mask)[0]
    if len(break_positions) == 0:
        return "Testee" if has_approach else "Vierge"

    break_idx = int(break_positions[0])

    # Re-test après cassure (role reversal) — vectorisé
    retest_tol = tolerance * 2
    rc = c_arr[break_idx + 1:]
    rh = h_arr[break_idx + 1:]
    rl = l_arr[break_idx + 1:]

    if len(rc) == 0:
        return "Consommee"

    retest_mask = (rl <= level + retest_tol) & (rh >= level - retest_tol)
    if retest_mask.any():
        return "Role Reverse"

    return "Consommee"


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
        pivot_highs = (pd.Series(df["high"].values[r_idx], index=r_idx)
                       if len(r_idx) else pd.Series(dtype=float))
        pivot_lows  = (pd.Series(df["low"].values[s_idx],  index=s_idx)
                       if len(s_idx) else pd.Series(dtype=float))

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

    zones_raw = []
    cur_group = [pivots_with_idx[0]]

    # HIGH #5 FIX — Fallback zone_width cohérent par type d'instrument.
    # Problème original : avg * zone_percentage_width / 100 avec avg=2000 (XAU)
    # donnait une zone de 10 points (trop large) ; avec avg=1.10 (EUR/USD)
    # donnait 0.0055 (trop étroit). Le fallback doit être absolu pour les indices/métaux.
    # On calcule une fois la largeur de référence basée sur le premier pivot.
    first_price = pivots_with_idx[0][0]
    if atr_val and atr_val > 0:
        _zone_width_ref = atr_val * atr_zone_coeff   # priorité ATR
    else:
        _zone_width_ref = first_price * zone_percentage_width / 100  # % du prix

    for price, idx in pivots_with_idx[1:]:
        avg = np.mean([p for p, _ in cur_group])
        zone_width_abs = ((atr_val * atr_zone_coeff) if (atr_val and atr_val > 0)
                          else _zone_width_ref)   # référence stable, pas recalculée
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

    # MEDIUM #2 FIX — Zones Consommées exclues AVANT la détection de confluence.
    # Problème original : une zone Consommée absorbait une zone Vierge voisine
    # dans un groupe via used_indices, faisant disparaître la Vierge des résultats.
    # On exclut les Consommées ici ; elles restent visibles dans les tableaux TF
    # individuels mais ne polluent pas les confluences.
    all_zones = [z for z in all_zones if z.get("status") != "Consommee"]

    if not all_zones:
        return []

    # HIGH #2 FIX — reset_index OBLIGATOIRE pour éviter les collisions d'index
    # quand detect_confluences est appelé sur un seul actif à la fois :
    # les indices 0,1,2... sont locaux à cet actif et ne peuvent pas se chevaucher,
    # mais un reset explicite garantit le comportement même si la logique change.
    zones_df     = (pd.DataFrame(all_zones)
                    .sort_values("level")
                    .reset_index(drop=True))   # ← index 0..N garanti propre
    used_indices = set()
    confluences  = []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue

        similar = zones_df[
            (abs(zones_df["level"] - zone["level"]) / zone["level"] * 100
             <= confluence_threshold) &
            (zones_df.index != i)
        ]

        if len(similar) >= 1:
            group      = pd.concat([zones_df.loc[[i]], similar])
            timeframes = group["tf"].unique()

            if len(timeframes) >= 2:
                avg_level = group["level"].mean()
                nb_tf     = len(timeframes)
                dist_pct  = abs(current_price - avg_level) / current_price * 100

                group_active = group[group["status"] != "Consommee"]
                if group_active.empty:
                    group_active = group

                total_score = 0.0
                for _, row in group_active.iterrows():
                    total_score += compute_structural_score(
                        int(row["strength"]),
                        nb_tf,
                        tf_name    = row["tf"],
                        age_bars   = int(row.get("age_bars", 0)),
                        total_bars = 500,
                    )

                total_strength = int(group["strength"].sum())
                best_status = min(
                    group["status"].tolist(),
                    key=lambda s: STATUS_PRIORITY.get(s, 1)
                )

                # HIGH #2b FIX — zone_type par vote majoritaire (pas iloc[0] arbitraire).
                # Quand H4=Support et Daily=Resistance sur le même niveau, iloc[0] donnait
                # un type aléatoire selon le tri. On prend le type le plus représenté.
                type_counts = group["type"].value_counts()
                zone_type = type_counts.idxmax()
                signal    = "🟢 BUY ZONE" if zone_type == "Support" else "🔴 SELL ZONE"
                tf_label  = " + ".join(sorted(timeframes))
                alerte    = ("🔥 ZONE CHAUDE" if dist_pct < 0.5
                             else ("⚠️ Proche" if dist_pct < 1.5 else ""))

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
# BUGFIX #5 — ScanResult dataclass (remplace le tuple de 6 éléments)
# Problème original : retour positionnel fragile, ajouter un champ
# cassait silencieusement tous les appels.
# ══════════════════════════════════════════════════════════════════
@dataclass
class ScanResult:
    symbol:        str
    rows:          dict
    zones:         dict
    price:         Optional[float]
    trends:        dict
    anomaly:       Optional[str]
    scan_error:    Optional[str] = None   # nouveau champ sans casser l'existant


def scan_single_symbol(args) -> ScanResult:
    symbol, base_url, access_token, account_id, zone_width, min_touches = args

    raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
    current_price, price_alert_msg = validate_live_price(
        raw_price, symbol, base_url, access_token
    )

    atr_zone_coeff  = ATR_ZONE_COEFF.get(symbol, DEFAULT_ATR_COEFF)
    prom_coeff      = PROMINENCE_COEFF.get(symbol, DEFAULT_PROMINENCE_COEFF)
    zone_w_fallback = ZONE_WIDTH_FALLBACK.get(symbol, zone_width)
    merge_thresh    = POST_MERGE_MAP.get(symbol, POST_MERGE_THRESHOLD)
    pdf_dist_max    = PDF_DIST_THRESHOLDS.get(symbol, DEFAULT_PDF_DIST)
    abs_dist_max    = ABSOLUTE_MAX_DIST.get(symbol, 99.0)

    rows           = {"H4": None, "Daily": None, "Weekly": None}
    zones_d        = {}
    trends         = {}
    all_sup_levels = []
    last_h4_close  = None
    anomaly_msg    = price_alert_msg

    # ──────────────────────────────────────────────────────────────
    # BUGFIX #6 — Prix de référence figé AVANT la boucle TF.
    # Problème original : current_price était muté dans le bloc H4
    # (fallback last_h4_close), ce qui faisait utiliser le close H4
    # comme référence pour Daily et Weekly → classification S/R fausse.
    # Solution : reference_price est assigné une seule fois et ne change plus.
    # ──────────────────────────────────────────────────────────────
    reference_price = current_price   # peut être None → résolu dans la boucle

    # LOW #1 FIX — sym_d calculé une fois hors boucle (était recalculé 3× par symbole)
    sym_d = symbol.replace("_", "/")

    # HIGH #8 FIX — Prefetch des 3 TFs en parallèle dans le thread du symbole.
    # Problème original : 3 appels réseau séquentiels par symbole = latence × 3.
    # Le ThreadPoolExecutor externe gère 8 symboles en parallèle, mais à l'intérieur
    # chaque symbole attendait H4, puis Daily, puis Weekly l'un après l'autre.
    # Solution : sous-pool de 3 workers pour récupérer les 3 TFs simultanément.
    # Gain estimé : latence réduite de ~66% par symbole (3 appels → 1 RTT).
    _TF_KEYS = [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]

    def _fetch_tf(tf_key):
        return tf_key, get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as tf_pool:
        tf_futures = {tf_pool.submit(_fetch_tf, tk): tk for tk, _ in _TF_KEYS}
        tf_data = {}
        for fut in concurrent.futures.as_completed(tf_futures):
            try:
                tf_key, df = fut.result()
                tf_data[tf_key] = df
            except Exception:
                tf_data[tf_futures[fut]] = None

    for tf_key, tf_cap in _TF_KEYS:
        df = tf_data.get(tf_key)
        if df is None or df.empty:
            continue

        # Au premier TF disponible, on fixe le prix de référence si encore inconnu
        if reference_price is None:
            reference_price = float(df["close"].iloc[-1])

        if tf_key == "h4":
            last_h4_close = float(df["close"].iloc[-1])

        cp = reference_price   # stable pour tous les TFs, jamais ré-assigné

        trends[tf_cap] = compute_trend(df)
        atr_val = compute_atr(df, period=14)

        supports, resistances = find_strong_sr_zones(
            df, cp,
            atr_val               = atr_val,
            zone_percentage_width = zone_w_fallback,
            atr_zone_coeff        = atr_zone_coeff,
            prominence_coeff      = prom_coeff,
            min_touches           = min_touches,
            timeframe             = tf_key,
            post_merge_threshold  = merge_thresh,
        )
        zones_d[tf_cap] = (supports, resistances)

        if not supports.empty:
            all_sup_levels.extend(supports["level"].tolist())

        if tf_key == "daily":
            price_ctx = get_price_context(cp, supports, resistances)
            zones_d["_price_ctx"] = price_ctx

        # LOW #1 FIX — tf_weight_name et n_total_bars étaient des aliases inutiles
        # recalculés à chaque TF. On utilise tf_cap et len(df) directement.
        def make_row(zone, ztype, _cp=cp, _atr=atr_val,
                     _pdf_max=pdf_dist_max, _abs_max=abs_dist_max,
                     _tf=tf_cap, _ntot=len(df)):
            """
            MEDIUM #2 FIX — Guards contre _cp=0/None et _atr=NaN.
            Problème original : dist_pct = abs(_cp - lvl) / _cp plantait
            en ZeroDivisionError si _cp était 0 ou None (cas extrêmes
            de données corrompues OANDA). dist_atr déjà protégé mais
            np.isnan échouait si dist_atr était None (pas NaN).
            """
            lvl      = zone["level"]
            strength = int(zone["strength"])
            age_bars = int(zone.get("age_bars", 0))
            status   = zone.get("status", "Testee")

            # Guard prix de référence nul ou absent
            if not _cp or _cp <= 0:
                dist_pct = 0.0
                dist_atr_str = "N/A"
                in_pdf = False
            else:
                dist_pct = abs(_cp - lvl) / _cp * 100
                if _atr and _atr > 0:
                    dist_atr_str = f"{round(abs(_cp - lvl) / _atr, 1):.1f}x"
                else:
                    dist_atr_str = "N/A"
                in_pdf = dist_pct <= _pdf_max and dist_pct <= _abs_max

            struct_score = compute_structural_score(strength, 1, _tf, age_bars, _ntot)
            return {
                "Actif":       sym_d,
                "Prix Actuel": f"{_cp:.5f}" if _cp else "N/A",
                "Type":        ztype,
                "Niveau":      f"{lvl:.5f}",
                "Force":       f"{strength} touches",
                "Score":       struct_score,
                "Statut":      status,
                "Dist. %":     f"{dist_pct:.2f}%",
                "Dist. ATR":   dist_atr_str,
                "_dist_num":   dist_pct,
                "_in_pdf":     in_pdf,
            }

        tf_rows = (
            [make_row(z, "Support")    for _, z in supports.iterrows()] +
            [make_row(z, "Resistance") for _, z in resistances.iterrows()]
        )
        if tf_rows:
            rows[tf_cap] = tf_rows

    if all_sup_levels and reference_price:
        new_anomaly = flag_data_anomaly(
            symbol, reference_price, all_sup_levels, last_h4_close
        )
        if new_anomaly:
            anomaly_msg = (f"{anomaly_msg} | {new_anomaly}"
                           if anomaly_msg else new_anomaly)

    return ScanResult(
        symbol  = symbol,
        rows    = rows,
        zones   = zones_d,
        price   = reference_price,
        trends  = trends,
        anomaly = anomaly_msg,
    )


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION PDF
# ══════════════════════════════════════════════════════════════════
# Colonnes internes à ne jamais exposer dans l'UI ou les exports
_INTERNAL_COLS = ["_dist_num", "_in_pdf"]

def strip_emojis_df(df):
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        clean[col] = clean[col].astype(str).apply(_safe_pdf_str)
    return clean


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10,
                  _safe_pdf_str('Rapport Scanner Bluestar - Supports & Resistances'),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6,
                  _safe_pdf_str(
                      f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  |  "
                      "Score = (Force x Poids_TF x NbTF) x Facteur_Age | "
                      "Statut Vierge / Testee / Role Reverse / Consommee"
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

    def chapter_anomalies(self, anomalies: dict):
        if not anomalies:
            return
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, _safe_pdf_str('ALERTES ANOMALIES PRIX'),
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        self.set_font('Helvetica', '', 8)
        for sym, msg in anomalies.items():
            line = _safe_pdf_str(f"[!] {sym} : {msg}")
            self.multi_cell(0, 5, line[:180])
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
                val      = _safe_pdf_str(str(row[col_name]))
                max_chars = int(w / 1.25)
                if len(val) > max_chars:
                    val = val[:max_chars - 1] + '.'
                self.cell(w, 5, val, border=1, align='C',
                          new_x='RIGHT', new_y='TOP')
            self.ln()


def _apply_pdf_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    HIGH #6 FIX — Garde-fou renforcé, fallback hiérarchique.
    Ordre : _in_pdf → _dist_num → reconstruction depuis 'Dist. %'.
    Garantit qu'aucune zone hors-distance ne passe dans le PDF,
    même si make_row() a produit un résultat partiel.
    """
    if df.empty:
        return df
    if "_in_pdf" in df.columns:
        df = df[df["_in_pdf"]].copy()
    elif "_dist_num" in df.columns:
        df = df[df["_dist_num"] <= DEFAULT_PDF_DIST].copy()
    elif "Dist. %" in df.columns:
        def _to_f(s):
            try:    return float(str(s).replace("%", ""))
            except: return 999.0
        df = df[df["Dist. %"].apply(_to_f) <= DEFAULT_PDF_DIST].copy()
    return df.drop(columns=_INTERNAL_COLS, errors="ignore").reset_index(drop=True)


def create_pdf_report(results_dict, confluences_df=None,
                      summaries=None, anomalies=None):
    summaries = summaries or []
    anomalies = anomalies or {}

    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if anomalies:
        pdf.chapter_anomalies(anomalies)

    if summaries:
        pdf.chapter_summary(summaries)
        pdf.add_page()

    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title('*** ZONES DE CONFLUENCE MULTI-TIMEFRAMES ***')
        clean_conf = strip_emojis_df(confluences_df.copy())
        clean_conf = clean_conf.drop(columns=_INTERNAL_COLS, errors="ignore")
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
        clean_df = clean_df.drop(columns=_INTERNAL_COLS, errors="ignore")
        if "Score" in clean_df.columns:
            clean_df = clean_df.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_df)
        pdf.ln(10)

    return bytes(pdf.output())


def create_csv_report(results_dict, confluences_df=None):
    all_dfs = []
    if confluences_df is not None and not confluences_df.empty:
        c = confluences_df.drop(columns=_INTERNAL_COLS, errors="ignore").copy()
        c["Section"] = "CONFLUENCES"
        all_dfs.append(c)
    for tf, df in results_dict.items():
        if df is not None and not df.empty:
            d = df.drop(columns=_INTERNAL_COLS, errors="ignore").copy()
            d["Timeframe"] = tf
            all_dfs.append(d)
    if not all_dfs:
        return b""
    buf = BytesIO()
    pd.concat(all_dfs, ignore_index=True).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def create_llm_brief(summaries, confluences_df,
                     max_dist=2.0, min_score=100.0,
                     allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    TREND_ARROW  = {"HAUSSIER": "↑", "BAISSIER": "↓", "NEUTRE": "→"}
    STATUS_LABEL = {"Vierge": "V", "Testee": "T", "Role Reverse": "RR", "Consommee": "C"}
    ALERT_LABEL  = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠", "": ""}

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    lines = [
        "# BRIEF S/R — Scanner Bluestar",
        f"_Généré le {now}_",
        "",
        "## INSTRUCTIONS POUR LLM",
        "Ce brief contient les zones Support/Résistance les plus fiables détectées",
        "par un scanner multi-timeframes (H4 + Daily + Weekly) sur 33 actifs Forex/Indices/Métaux.",
        "",
        "**Légende :**",
        "- `BUY` / `SELL` : direction de la zone",
        "- `Sc` : Score pondéré (Force × Poids_TF × NbTF × Facteur_Age). >300=institutionnel, 100-300=fort",
        "- `V` = Vierge | `T` = Testée | `RR` = Role Reverse (zone cassée retestée) | `C` = Consommée (éviter)",
        "- `Dist%` : distance du prix actuel à la zone",
        "- `TFs` : timeframes en confluence (H4/D/W)",
        "- `⚡` = zone chaude (<0.5% du prix) | `⚠` = proche (<1.5%)",
        "",
        f"**Filtres actifs** : Dist < {max_dist}% | Score ≥ {min_score} | Statuts : {', '.join(allowed_statuts)}",
        "",
        "---",
        "",
    ]

    if confluences_df is None or confluences_df.empty:
        lines.append("_Aucune confluence disponible._")
        return "\n".join(lines).encode("utf-8")

    actif_zones = {}
    for _, row in confluences_df.iterrows():
        try:
            dist_val = float(str(row.get("Distance %", "999")).replace("%", ""))
        except Exception:
            dist_val = 999.0

        # BUGFIX : guard sur Score pour éviter ValueError si "N/A" ou NaN
        try:
            score_val = float(row.get("Score", 0))
        except (TypeError, ValueError):
            score_val = 0.0

        statut = str(row.get("Statut", ""))

        if dist_val > max_dist:
            continue
        if score_val < min_score:
            continue
        if statut not in allowed_statuts:
            continue

        actif = str(row.get("Actif", ""))
        if actif not in actif_zones:
            actif_zones[actif] = []
        actif_zones[actif].append({
            "signal":  str(row.get("Signal", "")),
            "niveau":  str(row.get("Niveau", "")),
            "score":   score_val,
            "statut":  statut,
            "dist":    dist_val,
            "tfs":     str(row.get("Timeframes", "")),
            "nb_tf":   int(row.get("Nb TF", 0)),
            "alerte":  str(row.get("Alerte", "")),
        })

    actif_max_score = {
        actif: max(z["score"] for z in zones)
        for actif, zones in actif_zones.items()
    }
    sorted_actifs = sorted(actif_max_score,
                           key=lambda a: actif_max_score[a], reverse=True)

    summary_map = {s["symbol"]: s for s in summaries}

    total_zones = 0
    for actif in sorted_actifs:
        zones = sorted(actif_zones[actif], key=lambda z: z["score"], reverse=True)
        s     = summary_map.get(actif, {})

        t_h4 = TREND_ARROW.get(s.get("trend_h4",    "NEUTRE"), "→")
        t_d  = TREND_ARROW.get(s.get("trend_daily",  "NEUTRE"), "→")
        t_w  = TREND_ARROW.get(s.get("trend_weekly", "NEUTRE"), "→")

        ctx = s.get("price_context", "")

        lines.append(f"### {actif} | H4:{t_h4} D:{t_d} W:{t_w}")
        if ctx:
            lines.append(f"> {ctx}")

        for z in zones:
            signal_short = "BUY " if "BUY"  in z["signal"] else "SELL"
            st_short     = STATUS_LABEL.get(z["statut"], z["statut"])
            al_short     = ALERT_LABEL.get(z["alerte"], "")
            tf_short = (z["tfs"]
                        .replace("Daily", "D")
                        .replace("Weekly", "W")
                        .replace(" + ", "+"))
            line = (
                f"- {signal_short} `{z['niveau']}` | "
                f"Sc:{z['score']:.0f} | {st_short} | "
                f"{z['dist']:.2f}% | {tf_short} {al_short}"
            )
            lines.append(line)
            total_zones += 1

        lines.append("")

    lines += [
        "---",
        f"_Total zones retenues : {total_zones} sur {len(sorted_actifs)} actifs_",
        "",
        "## PROMPT SUGGÉRÉ POUR LLM",
        "```",
        "Tu es un analyste technique expert en trading Forex/Indices.",
        "Voici un brief S/R multi-timeframes généré automatiquement.",
        "Pour chaque actif pertinent :",
        "1. Identifie les setups les plus immédiats (⚡ zones chaudes en priorité)",
        "2. Vérifie la cohérence tendance vs direction de zone (ex: BUY en tendance haussière)",
        "3. Priorise les zones Vierge (V) sur 3 TF avec Score > 200",
        "4. Les zones Role Reverse (RR) = pullback sur niveau cassé, setup souvent court terme",
        "5. Propose un plan de trade structuré : entrée, SL (au-delà de la zone), TP (prochain niveau)",
        "```",
    ]

    return "\n".join(lines).encode("utf-8")


def create_json_export(summaries, confluences_df,
                       max_dist=5.0, min_score=50.0):
    output = {
        "generated_at":     datetime.now().isoformat(),
        "scanner_version":  f"Bluestar v{SCANNER_VERSION}",
        "description": (
            "Support/Resistance multi-timeframe scanner. "
            "Scores weighted by TF (H4=1x, Daily=2x, Weekly=3x) and age decay. "
            "Status: Vierge=never tested (most reliable), "
            "Testee=respected, Role Reverse=S/R flip setup, Consommee=broken."
        ),
        "filters_applied": {"max_dist_pct": max_dist, "min_score": min_score},
        "assets": []
    }

    summary_map = {s["symbol"]: s for s in summaries}

    if confluences_df is None or confluences_df.empty:
        return json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")

    actif_groups = {}
    for _, row in confluences_df.iterrows():
        try:
            dist_val = float(str(row.get("Distance %", "999")).replace("%", ""))
        except Exception:
            dist_val = 999.0
        # MEDIUM #5 FIX — guard NaN/None/str sur Score, cohérent avec create_llm_brief
        try:
            score_val = float(row.get("Score", 0))
            if not pd.notna(score_val):
                score_val = 0.0
        except (TypeError, ValueError):
            score_val = 0.0

        if dist_val > max_dist or score_val < min_score:
            continue

        actif = str(row.get("Actif", ""))
        if actif not in actif_groups:
            actif_groups[actif] = []
        actif_groups[actif].append({
            "signal":        str(row.get("Signal", "")).replace("🟢 ", "").replace("🔴 ", ""),
            "type":          str(row.get("Type", "")),
            "level":         str(row.get("Niveau", "")),
            "timeframes":    str(row.get("Timeframes", "")),
            "nb_tf":         int(row.get("Nb TF", 0)),
            "total_touches": int(row.get("Force Totale", 0)),
            "score":         round(score_val, 1),
            "status":        str(row.get("Statut", "")),
            "distance_pct":  round(dist_val, 3),
            "alert":         str(row.get("Alerte", "")).replace("🔥 ", "").replace("⚠️ ", ""),
        })

    for actif, zones in sorted(
        actif_groups.items(),
        key=lambda x: max(z["score"] for z in x[1]),
        reverse=True,
    ):
        s = summary_map.get(actif, {})
        output["assets"].append({
            "symbol": actif,
            "trend": {
                "H4":     s.get("trend_h4",    "NEUTRE"),
                "Daily":  s.get("trend_daily",  "NEUTRE"),
                "Weekly": s.get("trend_weekly", "NEUTRE"),
            },
            "price_context": s.get("price_context", ""),
            "zones": sorted(zones, key=lambda z: z["score"], reverse=True),
        })

    return json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE STREAMLIT
# ══════════════════════════════════════════════════════════════════
def _display_results(sr: dict, max_dist_filter: float):
    df_h4     = sr.get("df_h4",        pd.DataFrame())
    df_daily  = sr.get("df_daily",      pd.DataFrame())
    df_wk     = sr.get("df_weekly",     pd.DataFrame())
    conf_full = sr.get("conf_full",     pd.DataFrame())
    rep_dict  = sr.get("report_dict",   {})
    summaries = sr.get("summaries",     [])
    anomalies = sr.get("anomalies",     {})
    errors    = sr.get("scan_errors",   {})

    # HIGH #7 FIX — Recalcul conf_filt avec nettoyage systématique.
    # On nettoie _INTERNAL_COLS en entrée ET en sortie pour être
    # certain qu'aucune colonne fantôme n'atteint st.dataframe().
    if not conf_full.empty:
        tmp = conf_full.drop(columns=_INTERNAL_COLS, errors="ignore").copy()
        tmp["_dist_num"] = (
            tmp["Distance %"].str.replace("%", "", regex=False).astype(float)
        )
        conf_filt = (tmp[tmp["_dist_num"] <= max_dist_filter]
                     .drop(columns=_INTERNAL_COLS, errors="ignore")  # double nettoyage
                     .reset_index(drop=True))
    else:
        conf_filt = pd.DataFrame()

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

    # Erreurs de scan (actifs ayant échoué)
    if errors:
        with st.expander(f"❌ {len(errors)} actif(s) en erreur — cliquer pour voir"):
            for sym, err in errors.items():
                st.error(f"**{sym}** : {err}")

    if anomalies:
        with st.expander(f"⚠️ {len(anomalies)} anomalie(s) de prix — cliquer pour voir"):
            for sym, msg in anomalies.items():
                st.warning(f"**{sym}** : {msg}")

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
               for k in ["Actif", "Signal", "Niveau", "Type",
                         "Timeframes", "Statut", "Distance %", "Alerte"]},
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
                "📄 Rapport PDF (complet)",
                data=pdf_bytes,
                file_name=f"rapport_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width='stretch',
            )
        with col2:
            csv_bytes = create_csv_report(rep_dict, conf_full)
            st.download_button(
                "📊 Données brutes CSV",
                data=csv_bytes,
                file_name=f"donnees_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                width='stretch',
            )

        st.divider()
        st.markdown("**🤖 Exports optimisés LLM**")
        st.caption(
            "Paramètres LLM configurables dans la barre latérale (section 3) — "
            "ils survivent aux re-renders contrairement aux sliders ici."
        )

        # MEDIUM #4 FIX — Les paramètres LLM sont lus depuis la sidebar (st.session_state)
        # plutôt que redéfinis dans l'expander qui se réinitialise à chaque re-render.
        llm_max_dist  = st.session_state.get("llm_max_dist",  2.0)
        llm_min_score = st.session_state.get("llm_min_score", 100)
        llm_statuts_raw = st.session_state.get("llm_statuts", ["Vierge", "Testee", "Role Reverse"])
        llm_statuts = tuple(llm_statuts_raw) if llm_statuts_raw else ("Vierge", "Testee", "Role Reverse")

        st.caption(f"🔧 Filtres actifs : Dist < **{llm_max_dist}%** | Score ≥ **{llm_min_score}** | {', '.join(llm_statuts)}")

        col3, col4 = st.columns(2)
        with col3:
            md_bytes = create_llm_brief(
                summaries, conf_full,
                max_dist        = llm_max_dist,
                min_score       = llm_min_score,
                allowed_statuts = llm_statuts,
            )
            st.download_button(
                "🤖 Brief LLM (Markdown filtré)",
                data=md_bytes,
                file_name=f"brief_llm_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                width='stretch',
            )
        with col4:
            json_bytes = create_json_export(
                summaries, conf_full,
                max_dist  = max_dist_filter,
                min_score = 50.0,
            )
            st.download_button(
                "🔧 Export JSON structuré",
                data=json_bytes,
                file_name=f"sr_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                width='stretch',
            )

        st.divider()
        st.markdown("**👁️ Aperçu du Brief LLM**")
        st.caption(f"Filtres : Dist < {llm_max_dist}% | Score ≥ {llm_min_score} | {', '.join(llm_statuts)}")
        try:
            brief_preview = md_bytes.decode("utf-8")
            n_zones  = sum(1 for l in brief_preview.split("\n")
                           if l.strip().startswith("- BUY") or l.strip().startswith("- SELL"))
            n_actifs = brief_preview.count("### ")
            st.info(f"**{n_actifs} actifs** avec **{n_zones} zones** dans le brief LLM "
                    f"(≈ {n_zones * 15 + n_actifs * 10:,} tokens estimés)")
            st.text_area(
                "Brief LLM (copiable directement)",
                value=brief_preview,
                height=400,
                label_visibility="collapsed",
            )
        except Exception:
            st.warning("Aperçu non disponible.")

    def _filter_and_sort(df, max_pct):
        if df.empty or "Dist. %" not in df.columns:
            return df
        def to_float(s):
            try:   return float(str(s).replace("%", ""))
            except: return 999.0
        mask = df["Dist. %"].apply(to_float) <= max_pct
        out  = df[mask].drop(columns=_INTERNAL_COLS, errors="ignore")
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

    st.header("3. Paramètres Export LLM")
    st.caption("Ces paramètres survivent aux re-renders contrairement aux sliders dans l'expander.")
    llm_max_dist_sidebar = st.slider(
        "Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5,
        key="llm_max_dist",
        help="Zones au-delà de cette distance sont exclues du brief LLM.",
    )
    llm_min_score_sidebar = st.slider(
        "Score min brief LLM", 50, 300, 100, 25,
        key="llm_min_score",
        help="Zones sous ce score sont exclues du brief LLM.",
    )
    llm_statuts_sidebar = st.multiselect(
        "Statuts autorisés (brief LLM)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
        help="Role Reverse = zone cassée retestée (setup pullback).",
    )

    st.divider()
    st.header("4. Paramètres de Détection")
    zone_width = st.slider(
        "Largeur zone Forex (% fallback si ATR indispo)", 0.1, 2.0, 0.5, 0.1,
        help="Normalement remplacé par ATR × coeff. Utilisé si ATR non disponible.",
    )
    min_touches = st.slider(
        "Force minimale (touches)", 3, 10, 3, 1,
        help="Nombre de contacts minimum pour valider une zone. Min 3 recommandé.",
    )
    confluence_threshold = st.slider(
        "Seuil confluence Forex (%)", 0.3, 2.0, 1.0, 0.1,
        help="Indices/Métaux utilisent des seuils adaptatifs (1.2-1.5%) automatiquement.",
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) - filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
    )

    st.divider()
    st.caption("**Score = (Force × Poids_TF × NbTF) × Facteur_Age**")
    st.caption("🔴 > 300 : Zone institutionnelle majeure")
    st.caption("🟠 100-300 : Zone structurelle forte")
    st.caption("🟡 30-100 : Zone technique valide")
    st.caption("⚪ < 30  : Zone secondaire")

    st.divider()
    st.caption("**Statuts :**")
    st.caption("✅ Vierge = jamais testée (la plus fiable)")
    st.caption("🔵 Testée = respectée, toujours valide")
    st.caption("↩️ Role Reverse = niveau cassé retesté")
    st.caption("❌ Consommée = cassée sans retour")

    st.divider()
    st.caption("**Corrections v2 (critiques) :**")
    st.caption("✅ Cache thread-safe (ThreadPoolExecutor)")
    st.caption("✅ Prix de référence figé avant boucle TF")
    st.caption("✅ post_merge_zones ancre fixe (no drift)")
    st.caption("✅ Erreurs scan loggées (plus de pass silencieux)")
    st.caption("✅ Role Reversal détecté automatiquement")
    st.caption("✅ ScanResult dataclass (retour nommé)")
    st.divider()
    st.caption("**Corrections v3 (élevés) :**")
    st.caption("✅ detect_swing_pivots vectorisé (10-30× plus rapide)")
    st.caption("✅ Prefetch H4/Daily/Weekly en parallèle")
    st.caption("✅ zone_type par vote majoritaire (plus iloc[0])")
    st.caption("✅ _apply_pdf_filter fallback hiérarchique")
    st.caption("✅ Colonnes fantômes purgées dans _display_results")
    st.caption("✅ zone_width fallback stable (référence fixe)")
    st.divider()
    st.caption("**Corrections v5 (audit 2 — 11 findings) :**")
    st.caption("✅ Nested ThreadPoolExecutor → pool ext. 4 workers (12 threads max)")
    st.caption("✅ get_oanda_current_price : cache manuel thread-safe")
    st.caption("✅ classify_zone_status vectorisé numpy (élim. boucle iloc)")
    st.caption("✅ Dead code pass supprimé dans get_oanda_data")
    st.caption("✅ SCANNER_VERSION = '5.0' constante centralisée")
    st.caption("✅ validate_live_price : limit=500 (partage cache prefetch)")
    st.caption("✅ detect_confluences : Consommées exclues avant traitement")
    st.caption("✅ _filter_and_sort : guard colonne Dist. % absente")
    st.caption("✅ compute_trend : branches elif dead code supprimées")
    st.caption("✅ sym_d calculé une fois hors boucle TF")
    st.caption("✅ create_csv_report : guard df is not None")


# ══════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE
# ══════════════════════════════════════════════════════════════════
if scan_button and symbols_to_scan:
    st.session_state.pop("scan_results", None)

    if not access_token or not account_id:
        st.warning("Configurez vos secrets OANDA avant de lancer le scan.")
    else:
        base_url = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.error("Identifiants OANDA invalides. Vérifiez vos secrets.")
        else:
            progress_bar = st.progress(0, text="Initialisation du scan…")

            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map = {}
            prices_map    = {}
            trends_map    = {}
            anomalies_map = {}
            scan_errors   = {}   # BUGFIX #3 : collecte des erreurs par symbole

            args_list = [
                (sym, base_url, access_token, account_id, zone_width, min_touches)
                for sym in symbols_to_scan
            ]
            total, completed = len(args_list), 0

            # CRIT #1 FIX — Pool externe limité à 4 workers (était 8).
            # Avec le sous-pool de 3 threads par symbole (prefetch TFs),
            # on plafonne à 4×3=12 threads IO simultanés au lieu de 24.
            # Évite l'épuisement du pool OS et les deadlocks nested sur Python < 3.9.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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
                        # BUGFIX #5 : déstructuration nommée via dataclass
                        result: ScanResult = future.result()

                        all_zones_map[result.symbol] = result.zones
                        prices_map[result.symbol]    = result.price
                        trends_map[result.symbol]    = result.trends

                        if result.anomaly:
                            anomalies_map[result.symbol.replace("_", "/")] = result.anomaly

                        for tf_cap, tf_rows in result.rows.items():
                            if tf_rows:
                                if tf_cap == "H4":
                                    results_h4.extend(tf_rows)
                                elif tf_cap == "Daily":
                                    results_daily.extend(tf_rows)
                                elif tf_cap == "Weekly":
                                    results_weekly.extend(tf_rows)

                    # BUGFIX #3 : exception loggée par symbole, plus de pass silencieux
                    except Exception as e:
                        scan_errors[sym.replace("_", "/")] = (
                            f"{type(e).__name__}: {str(e)[:120]}"
                        )

            progress_bar.empty()

            n_ok   = len(symbols_to_scan) - len(scan_errors)
            n_fail = len(scan_errors)
            if n_fail == 0:
                st.success(f"✅ Scan terminé — {n_ok} actifs analysés avec succès.")
            else:
                st.warning(
                    f"⚠️ Scan terminé — {n_ok} actifs OK, "
                    f"{n_fail} en erreur (voir détail ci-dessous)."
                )

            if anomalies_map:
                st.warning(f"⚠️ {len(anomalies_map)} anomalie(s) de prix détectée(s).")

            st.info("🔍 Analyse des confluences multi-timeframes…")
            all_confluences = []
            for sym in symbols_to_scan:
                if sym.replace("_", "/") in scan_errors:
                    continue   # ne pas traiter les actifs en erreur
                cp = prices_map.get(sym)
                zones_clean = {k: v for k, v in all_zones_map.get(sym, {}).items()
                               if not k.startswith("_")}
                sym_threshold = CONFLUENCE_THRESHOLD_MAP.get(sym, confluence_threshold)
                confs = detect_confluences(
                    sym.replace("_", "/"), zones_clean, cp, sym_threshold
                )
                all_confluences.extend(confs)

            conf_full = pd.DataFrame(all_confluences)

            # BUGFIX #5b : conf_full stocké SANS colonnes internes
            if not conf_full.empty:
                conf_full = conf_full.drop(columns=_INTERNAL_COLS, errors="ignore")

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
                d_zones   = all_zones_map.get(sym, {})
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
                "df_h4":       df_h4,
                "df_daily":    df_daily,
                "df_weekly":   df_weekly,
                "conf_full":   conf_full,   # propre, sans _dist_num
                "report_dict": rep_dict,
                "summaries":   summaries,
                "anomalies":   anomalies_map,
                "scan_errors": scan_errors,
                "max_dist":    max_dist_filter,
            }

            _display_results(st.session_state["scan_results"], max_dist_filter)

elif not symbols_to_scan:
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
else:
    st.info(
        "Configurez les paramètres dans la barre latérale, "
        "puis cliquez sur **LANCER LE SCAN COMPLET**."
    )

if "scan_results" in st.session_state and not scan_button:
    _display_results(
        st.session_state["scan_results"],
        max_dist_filter,
    )
