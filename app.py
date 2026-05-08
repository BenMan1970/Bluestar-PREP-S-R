import hashlib
import json
import logging
import os
import threading
import time
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
# LOGGING STRUCTURÉ — Patch M7
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# LOG_LEVEL configurable sans toucher au code (ex. VPS : LOG_LEVEL=WARNING)
_log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level_str, logging.INFO))

# ══════════════════════════════════════════════════════════════════
# SESSION HTTP THREAD-LOCAL + RATE LIMITER — Patch C1 + M4
#
# AVANT : requests.Session() globale partagée entre threads → non thread-safe
# APRÈS : threading.local() → chaque thread a sa propre session isolée
#
# AVANT : aucun contrôle de débit → risque HTTP 429 OANDA
# APRÈS : sémaphore (max 10 requêtes concurrentes)
# ══════════════════════════════════════════════════════════════════
_thread_local = threading.local()
_api_semaphore = threading.Semaphore(10)  # max 10 requêtes OANDA simultanées


def _get_http_session() -> requests.Session:
    """Retourne la session HTTP propre au thread courant (thread-safe)."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=5)
        s.mount("https://", adapter)
        _thread_local.session = s
    return _thread_local.session


def _rate_limited_get(url: str, **kwargs) -> requests.Response:
    """Requête GET HTTP rate-limitée via sémaphore + session thread-locale."""
    with _api_semaphore:
        return _get_http_session().get(url, **kwargs)


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
        background-color: #D32F2F; color: white; border: 1px solid #B71C1C; transition: all 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #B71C1C; border-color: #D32F2F; box-shadow: 0 4px 12px rgba(211,47,47,0.4);
    }
    div[data-testid="stButton"] > button[kind="primary"]:active {
        background-color: #D32F2F; transform: scale(0.98);
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
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
SCANNER_VERSION = "5.8.1"

ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

ATR_ZONE_COEFF = {"XAU_USD": 0.5, "US30_USD": 0.3, "NAS100_USD": 0.3, "SPX500_USD": 0.3, "DE30_EUR": 0.3}
DEFAULT_ATR_COEFF = 0.4

PROMINENCE_COEFF = {"XAU_USD": 0.5, "US30_USD": 0.4, "NAS100_USD": 0.4, "SPX500_USD": 0.4, "DE30_EUR": 0.4}
DEFAULT_PROMINENCE_COEFF = 0.3

ZONE_WIDTH_FALLBACK = {"US30_USD": 0.50, "NAS100_USD": 0.50, "SPX500_USD": 0.25, "DE30_EUR": 0.25, "XAU_USD": 0.20}
DEFAULT_ZONE_WIDTH = 0.5

POST_MERGE_THRESHOLD = 0.30
POST_MERGE_MAP = {"US30_USD": 0.05, "NAS100_USD": 0.05, "SPX500_USD": 0.08, "DE30_EUR": 0.08, "XAU_USD": 0.15}
POST_MERGE_ATR_COEFF = 0.5

CONFLUENCE_THRESHOLD_MAP = {"US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2, "XAU_USD": 1.5}

# Patch m3 — Ajout des paires JPY (manquantes en v5.7)
PRICE_SANITY_RANGE = {
    "XAU_USD":    (1500.0, 5500.0),
    "US30_USD":   (20000.0, 70000.0),
    "NAS100_USD": (8000.0, 35000.0),
    "SPX500_USD": (3000.0, 9000.0),
    "DE30_EUR":   (8000.0, 30000.0),
    # Paires JPY — sans sanity check elles acceptaient tout prix aberrant
    "USD_JPY":    (80.0, 200.0),
    "EUR_JPY":    (100.0, 210.0),
    "GBP_JPY":    (120.0, 250.0),
    "AUD_JPY":    (50.0, 130.0),
    "CAD_JPY":    (70.0, 140.0),
    "CHF_JPY":    (100.0, 210.0),
    "NZD_JPY":    (50.0, 120.0),
}

_SKIP_RATIO_CHECK = {"XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR"}
_DEV_THRESHOLD = {"XAU_USD": 3.0, "US30_USD": 2.5, "NAS100_USD": 2.5, "SPX500_USD": 2.0, "DE30_EUR": 2.0}
_DEFAULT_DEV_THRESHOLD = 1.5

PDF_DIST_THRESHOLDS = {"US30_USD": 5.0, "NAS100_USD": 5.0, "SPX500_USD": 5.0, "DE30_EUR": 5.0, "XAU_USD": 8.0}
DEFAULT_PDF_DIST = 8.0

ABSOLUTE_MAX_DIST = {"XAU_USD": 8.0, "US30_USD": 4.0, "NAS100_USD": 4.0, "SPX500_USD": 4.0, "DE30_EUR": 4.0}

_GRANULARITY_MAP = {"h4": "H4", "daily": "D", "weekly": "W"}
_TREND_MIN_RATIO = {"h4": 0.6, "daily": 0.6, "weekly": 0.4}

PIVOT_BAND_MAP = {
    "US30_USD": 0.15, "NAS100_USD": 0.15,
    "SPX500_USD": 0.12, "DE30_EUR": 0.12,
    "XAU_USD": 0.20,
}
DEFAULT_PIVOT_BAND_PCT = 0.50
PIVOT_BAND_ATR_COEFF = 0.5

# ══════════════════════════════════════════════════════════════════
# CACHE THREAD-SAFE + LRU — Patch M1
#
# AVANT : éviction uniquement sur éléments expirés → si tout est frais,
#         le cache croît sans borne → OOM sur sessions longues
# APRÈS : fallback LRU (suppression des plus vieux) si TTL insuffisant
# ══════════════════════════════════════════════════════════════════
_oanda_cache: dict = {}
_oanda_cache_lock = threading.Lock()


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _cache_key(token: str, account_id: str, symbol: str, timeframe: str, limit: int) -> str:
    """Clé isolée par (token, account_id) — évite toute fuite entre comptes."""
    return f"{_token_fingerprint(token)}__{account_id}__{symbol}__{timeframe}__{limit}"


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
        if len(_oanda_cache) > 500:
            now = datetime.now()
            # Priorité 1 : supprimer les éléments expirés (> 600s)
            expired = [k for k, (_, ts) in _oanda_cache.items()
                       if (now - ts).total_seconds() > 600]
            for k in expired:
                del _oanda_cache[k]
            # Priorité 2 : si encore > 400, LRU sur les plus anciens
            if len(_oanda_cache) > 400:
                sorted_keys = sorted(_oanda_cache, key=lambda k: _oanda_cache[k][1])
                for k in sorted_keys[:100]:
                    del _oanda_cache[k]


# ══════════════════════════════════════════════════════════════════
# UTILITAIRE PDF — Patch m2 (encodage latin-1 complet)
# ══════════════════════════════════════════════════════════════════
_ACCENT_MAP = str.maketrans(
    'àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
    'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN'
)
_EMOJI_MAP = [
    ('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'),
    ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]'),
    ('📈', ''), ('📉', ''), ('✅', '[OK]'), ('❌', '[X]'),
    ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'),
    ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]'),
]


def _safe_pdf_str(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_ACCENT_MAP)
    for emoji, replacement in _EMOJI_MAP:
        text = text.replace(emoji, replacement)
    # Patch m2 : encodage latin-1 avec remplacement pour éviter UnicodeEncodeError FPDF
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


# ══════════════════════════════════════════════════════════════════
# UTILITAIRES MÉTIER — Patch m7 (DRY)
# ══════════════════════════════════════════════════════════════════
def _parse_dist_pct(val) -> float:
    """Parse '1.23%' ou float en float. Retourne 999.0 si invalide."""
    try:
        return float(str(val).replace("%", ""))
    except (ValueError, TypeError):
        return 999.0


def _determine_signal(signal_str: str) -> str:
    """Normalise un signal brut en 'BUY ZONE', 'SELL ZONE' ou 'PIVOT ZONE'."""
    s = str(signal_str)
    if "BUY" in s:
        return "BUY ZONE"
    if "SELL" in s:
        return "SELL ZONE"
    return "PIVOT ZONE"


# ══════════════════════════════════════════════════════════════════
# API OANDA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def determine_oanda_environment(access_token: str, account_id: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {access_token}"}
    for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
        try:
            r = _rate_limited_get(f"{url}/v3/accounts/{account_id}/summary",
                                  headers=headers, timeout=5)
            if r.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


def get_oanda_data(base_url: str, access_token: str, account_id: str, symbol: str,
                   timeframe: str = "daily", limit: int = 500) -> Optional[pd.DataFrame]:
    key = _cache_key(access_token, account_id, symbol, timeframe, limit)
    cached = _cache_get(key, ttl_seconds=600)
    if cached is not None:
        return cached
    gran = _GRANULARITY_MAP.get(timeframe)
    if gran is None:
        raise ValueError(f"Timeframe inconnu: {timeframe!r}")
    url = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"count": limit + 1, "granularity": gran, "price": "M"}
    try:
        r = _rate_limited_get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("candles"):
            return None
        candles = []
        for c in data.get("candles", []):
            if not c.get("complete"):
                continue
            # Patch C3 — accès sécurisé aux clés JSON (KeyError/TypeError si OANDA renvoie mal)
            mid = c.get("mid")
            if not mid:
                logger.warning(f"Bougie sans clé 'mid' ignorée pour {symbol}/{timeframe}")
                continue
            try:
                o = float(mid.get("o", 0) or 0)
                h = float(mid.get("h", 0) or 0)
                lo = float(mid.get("l", 0) or 0)
                cl = float(mid.get("c", 0) or 0)
                if cl == 0:
                    continue  # Bougie invalide
                candles.append({
                    "date": pd.to_datetime(c["time"]),
                    "open": o, "high": h, "low": lo, "close": cl,
                    "volume": int(c.get("volume", 0)),
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Erreur parsing bougie {symbol}/{timeframe}: {e}")
                continue
        if not candles:
            return None
        # Patch M2 — candles[-limit:] au lieu de candles[:limit]
        # AVANT : prenait les N PREMIÈRES bougies (les plus anciennes si count > limit)
        # APRÈS : prend les N DERNIÈRES (les plus récentes) — comportement attendu
        df = pd.DataFrame(candles[-limit:]).set_index("date")
        if df.empty:
            return None
        _cache_set(key, df)
        return df
    except requests.Timeout:
        logger.warning(f"Timeout OANDA pour {symbol}/{timeframe}")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error(f"HTTP {status} OANDA pour {symbol}/{timeframe}")
        if status == 401:
            logger.critical("Token OANDA invalide ou expiré")
        elif status == 429:
            logger.warning("Rate limit OANDA atteint")
        return None
    except requests.ConnectionError:
        logger.error(f"Erreur connexion pour {symbol}/{timeframe}")
        return None
    except Exception as e:
        logger.exception(f"Erreur inattendue {symbol}/{timeframe}: {type(e).__name__}: {e}")
        return None


_price_cache: dict = {}
_price_cache_lock = threading.Lock()


def get_oanda_current_price(base_url: str, access_token: str,
                             account_id: str, symbol: str) -> Optional[float]:
    key = f"{_token_fingerprint(access_token)}__{account_id}__{symbol}__price"
    now = datetime.now()
    # Lecture + validation atomique dans le même lock (Patch Qwen)
    with _price_cache_lock:
        entry = _price_cache.get(key)
        if entry is not None:
            val, ts = entry
            if (now - ts).total_seconds() < 60:
                return val
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    result = None
    try:
        r = _rate_limited_get(url, headers=headers, params={"instruments": symbol}, timeout=5)
        r.raise_for_status()
        data = r.json()
        if "prices" in data and data["prices"]:
            bid = float(data["prices"][0]["closeoutBid"])
            ask = float(data["prices"][0]["closeoutAsk"])
            result = (bid + ask) / 2
    except requests.RequestException as e:
        logger.warning(f"Échec fetch prix {symbol}: {e}")
        result = None
    # Patch M1 — éviction cache prix
    with _price_cache_lock:
        if len(_price_cache) > 200:
            expired = [k for k, (_, ts) in _price_cache.items()
                       if (now - ts).total_seconds() > 120]
            for k in expired:
                del _price_cache[k]
        _price_cache[key] = (result, now)
    return result


def validate_live_price(live_price, symbol, h4_df=None):
    """
    Patch M3 — suppression du re-fetch H4 redondant.
    AVANT : re-fetchait get_oanda_data(base_url, access_token, symbol, "h4") même si déjà téléchargé.
    APRÈS : reçoit directement h4_df depuis scan_single_symbol.
    """
    dev_threshold = _DEV_THRESHOLD.get(symbol, _DEFAULT_DEV_THRESHOLD)
    h4_close = None
    if h4_df is not None and not h4_df.empty:
        h4_close = float(h4_df["close"].iloc[-1])
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
                messages.append(f"Prix {current_price:.2f} = {ratio:.1f}x mediane supports ({median_sup:.2f})")
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > 25.0:
            messages.append(f"Prix live {current_price:.2f} ecart {dev:.1f}% du close ({last_candle_close:.2f})")
    return " | ".join(messages) if messages else None


def get_price_context(current_price, supports, resistances,
                      max_dist_pct: float = 5.0, min_strength_anchor: int = 3):
    if not current_price or current_price <= 0:
        return "Prix indisponible"
    parts = []
    if not supports.empty:
        sup_nearby = supports[
            (supports["level"] < current_price) &
            (abs(supports["level"] - current_price) / current_price * 100 <= max_dist_pct) &
            (supports["strength"] >= min_strength_anchor)
        ]
        if not sup_nearby.empty:
            nearest_sup = sup_nearby.nlargest(1, "level").iloc[0]
            dist_s = abs(current_price - nearest_sup["level"]) / current_price * 100
            tag = "SUR support" if dist_s < 0.5 else "S proche"
            parts.append(f"{tag}: {nearest_sup['level']:.5f} (-{dist_s:.2f}%)")
    if not resistances.empty:
        res_nearby = resistances[
            (resistances["level"] > current_price) &
            (abs(resistances["level"] - current_price) / current_price * 100 <= max_dist_pct) &
            (resistances["strength"] >= min_strength_anchor)
        ]
        if not res_nearby.empty:
            nearest_res = res_nearby.nsmallest(1, "level").iloc[0]
            dist_r = abs(current_price - nearest_res["level"]) / current_price * 100
            tag = "SUR resistance" if dist_r < 0.5 else "R proche"
            parts.append(f"{tag}: {nearest_res['level']:.5f} (+{dist_r:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"


# ══════════════════════════════════════════════════════════════════
# MOTEUR D'ANALYSE
# ══════════════════════════════════════════════════════════════════
def get_adaptive_distance(timeframe):
    return {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)


def compute_atr(df, period=14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        return None
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    result = tr.rolling(period).mean().iloc[-1]
    return float(result) if pd.notna(result) and result > 0 else None


def compute_trend(df, sma_period=20, timeframe_key: str = "daily"):
    if df is None or len(df) < 15:
        return "NEUTRE"
    actual_period = min(sma_period, len(df) - 5)
    min_ratio = _TREND_MIN_RATIO.get(timeframe_key, 0.6)
    if actual_period < sma_period * min_ratio:
        return "NEUTRE"
    close   = df["close"]
    sma     = close.rolling(actual_period).mean().iloc[-1]
    current = close.iloc[-1]
    # Patch m1 — ZeroDivisionError si close.iloc[-6] == 0
    ref_idx = min(6, len(close) - 1)
    ref_val = close.iloc[-ref_idx] if ref_idx > 0 else close.iloc[-1]
    if abs(ref_val) < 1e-10:
        return "NEUTRE"
    slope_pct = (close.iloc[-1] - ref_val) / ref_val * 100
    n     = min(10, len(df))
    highs = df["high"].iloc[-n:]
    lows  = df["low"].iloc[-n:]
    hh = highs.iloc[-1] > highs.iloc[0]
    ll = lows.iloc[-1]  < lows.iloc[0]
    lh = highs.iloc[-1] < highs.iloc[0]
    hl = lows.iloc[-1]  > lows.iloc[0]
    above_sma = current > sma
    below_sma = current < sma
    if above_sma and slope_pct > 0.05 and hh and hl:
        return "HAUSSIER"
    if below_sma and slope_pct < -0.05 and lh and ll:
        return "BAISSIER"
    if above_sma and slope_pct > 0.05:
        return "HAUSSIER"
    if below_sma and slope_pct < -0.05:
        return "BAISSIER"
    if above_sma and hh:
        return "HAUSSIER"
    if below_sma and ll:
        return "BAISSIER"
    return "NEUTRE"


TF_WEIGHT = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}


def compute_structural_score(strength, nb_tf, tf_name="H4", age_bars=0, total_bars=500):
    tf_w  = TF_WEIGHT.get(tf_name, 1.0)
    age_r = age_bars / max(total_bars, 1)
    age_f = float(np.exp(-1.5 * age_r))
    return round(strength * tf_w * nb_tf * age_f, 1)


def post_merge_zones(zones_list, threshold_pct=0.30, atr_val=None):
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
            group = [zones_list[i]]
            for j in range(i + 1, len(zones_list)):
                if j in used:
                    continue
                current_centroid = np.mean([z["level"] for z in group])
                if atr_val and atr_val > 0:
                    merge_ok = abs(zones_list[j]["level"] - current_centroid) <= atr_val * POST_MERGE_ATR_COEFF
                else:
                    merge_ok = (abs(zones_list[j]["level"] - current_centroid) / current_centroid * 100
                                <= threshold_pct) if current_centroid > 0 else False
                if merge_ok:
                    group.append(zones_list[j])
                    used.add(j)
                    changed = True
            used.add(i)
            best_age = min(z.get("age_bars", 0) for z in group)
            best_status = min((z.get("status", "Testee") for z in group),
                              key=lambda s: STATUS_PRIORITY.get(s, 1))
            new_zones.append({"level": np.mean([z["level"] for z in group]),
                               "strength": sum(z["strength"] for z in group),
                               "age_bars": best_age, "status": best_status})
        zones_list = new_zones
    return zones_list


def detect_swing_pivots(df, n=3, atr_val=None, prominence_coeff=0.3):
    highs  = pd.Series(df["high"].values)
    lows   = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)
    roll_high_left  = highs.shift(1).rolling(n, min_periods=n).max()
    roll_high_right = highs[::-1].shift(1).rolling(n, min_periods=n).max()[::-1]
    roll_low_left   = lows.shift(1).rolling(n, min_periods=n).min()
    roll_low_right  = lows[::-1].shift(1).rolling(n, min_periods=n).min()[::-1]
    next_close = closes.shift(-1).fillna(closes)
    sh_mask = (highs > roll_high_left) & (highs > roll_high_right) & (next_close < highs)
    sl_mask = (lows < roll_low_left)   & (lows < roll_low_right)   & (next_close > lows)
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
                         atr_val=None, tolerance_coeff=None):
    if tolerance_coeff is None:
        tolerance_coeff = DEFAULT_ATR_COEFF
    if formation_idx >= len(df) - 1:
        return "Vierge"
    tolerance = (atr_val * tolerance_coeff) if (atr_val and atr_val > 0) else (level * 0.003)
    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]
    if len(c_arr) == 0:
        return "Vierge"
    near = (np.abs(c_arr - level) <= tolerance) | (
        (l_arr <= level + tolerance) & (h_arr >= level - tolerance))
    has_approach = bool(near.any())
    if zone_type == "Support":
        break_mask = c_arr < level - tolerance
    else:
        break_mask = c_arr > level + tolerance
    break_positions = np.where(break_mask)[0]
    if len(break_positions) == 0:
        return "Testee" if has_approach else "Vierge"
    break_idx = int(break_positions[0])
    retest_tol = tolerance * 2
    rc = c_arr[break_idx + 1:]
    rh = h_arr[break_idx + 1:]
    rl = l_arr[break_idx + 1:]
    if len(rc) == 0:
        return "Consommee"
    retest_mask = (rl <= level + retest_tol) & (rh >= level - retest_tol)
    if not retest_mask.any():
        return "Consommee"
    retest_idx = int(np.where(retest_mask)[0][0])
    rc_after = rc[retest_idx + 1:]
    if len(rc_after) == 0:
        return "Role Reverse"
    if zone_type == "Support":
        second_break = rc_after < level - tolerance
    else:
        second_break = rc_after > level + tolerance
    if second_break.any():
        return "Consommee"
    return "Role Reverse"


def find_strong_sr_zones(df, current_price, symbol=None, atr_val=None,
                          zone_percentage_width=0.5, atr_zone_coeff=0.4,
                          prominence_coeff=0.3, min_touches=2, timeframe="daily",
                          post_merge_threshold=0.30, swing_n=3):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None:
        current_price = df["close"].iloc[-1]
    n_total = len(df)
    pivot_highs, pivot_lows = detect_swing_pivots(df, n=swing_n, atr_val=atr_val,
                                                   prominence_coeff=prominence_coeff)
    if len(pivot_highs) + len(pivot_lows) < 3:
        distance = get_adaptive_distance(timeframe)
        pk = {"distance": distance, "prominence": atr_val * prominence_coeff} if (atr_val and atr_val > 0) else {"distance": distance}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values,  **pk)
        pivot_highs = pd.Series(df["high"].values[r_idx], index=r_idx) if len(r_idx) else pd.Series(dtype=float)
        pivot_lows  = pd.Series(df["low"].values[s_idx],  index=s_idx) if len(s_idx) else pd.Series(dtype=float)
    if pivot_highs.empty and pivot_lows.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivots_with_idx = [(float(p), int(i)) for i, p in pivot_highs.items()]
    pivots_with_idx += [(float(p), int(i)) for i, p in pivot_lows.items()]
    pivots_with_idx.sort(key=lambda x: x[0])
    if not pivots_with_idx:
        return pd.DataFrame(), pd.DataFrame()
    ref_price = current_price if current_price and current_price > 0 else pivots_with_idx[0][0]

    # Patch m8 — zone_width_abs calculé UNE FOIS avant la boucle (était recalculé N fois)
    zone_width_abs = (atr_val * atr_zone_coeff) if (atr_val and atr_val > 0) \
        else (ref_price * zone_percentage_width / 100)

    zones_raw = []
    cur_group = [pivots_with_idx[0]]
    for price, idx in pivots_with_idx[1:]:
        anchor = cur_group[0][0]
        if abs(price - anchor) < zone_width_abs:
            cur_group.append((price, idx))
        else:
            zones_raw.append(cur_group)
            cur_group = [(price, idx)]
    zones_raw.append(cur_group)
    strong = []
    for grp in zones_raw:
        if len(grp) < min_touches:
            continue
        prices  = [p for p, _ in grp]
        indices = [i for _, i in grp]
        lvl     = float(np.mean(prices))
        strength = len(grp)
        last_idx = max(indices)
        age_bars = n_total - 1 - last_idx
        zone_type_tmp = "Support" if lvl < current_price else "Resistance"
        status = classify_zone_status(lvl, zone_type_tmp, df, last_idx,
                                       atr_val=atr_val, tolerance_coeff=atr_zone_coeff)
        strong.append({"level": lvl, "strength": strength, "age_bars": age_bars, "status": status})
    if not strong:
        return pd.DataFrame(), pd.DataFrame()
    strong = post_merge_zones(strong, threshold_pct=post_merge_threshold, atr_val=atr_val)
    df_zones = pd.DataFrame(strong).sort_values("level").reset_index(drop=True)

    if atr_val and atr_val > 0:
        pivot_band_abs = atr_val * PIVOT_BAND_ATR_COEFF
        pivot_mask = abs(df_zones["level"] - current_price) <= pivot_band_abs \
            if current_price and current_price > 0 else pd.Series([False] * len(df_zones))
    else:
        band_pct = PIVOT_BAND_MAP.get(symbol, DEFAULT_PIVOT_BAND_PCT) if symbol else DEFAULT_PIVOT_BAND_PCT
        pivot_mask = abs(df_zones["level"] - current_price) / current_price * 100 <= band_pct \
            if current_price and current_price > 0 else pd.Series([False] * len(df_zones))

    df_zones["is_pivot"] = pivot_mask
    supports    = df_zones[(df_zones["level"] <  current_price) | pivot_mask].copy()
    resistances = df_zones[(df_zones["level"] >= current_price) & ~pivot_mask].copy()
    return supports, resistances


def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0,
                       atr_map=None, total_bars_map=None):
    """
    v5.8 — Corrections appliquées :
    Patch C2  : used_indices filtre la requête `similar` → fin du double-comptage
    Patch m4  : filtre 'Consommee' AVANT déduplication → priorité aux zones vierges
    Patch m6  : total_bars_map par TF au lieu de 500 hardcodé
    FIX A/B/2/5 (v5.6/5.7) conservés intacts.
    """
    _ref_atr = None
    if atr_map:
        _ref_atr = atr_map.get("h4") or atr_map.get("daily") or atr_map.get("weekly")
    if not zones_dict or current_price is None:
        return []
    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, z in supports.iterrows():
            is_piv = bool(z.get("is_pivot", False))
            all_zones.append({"tf": tf, "level": z["level"], "strength": z["strength"],
                               "age_bars": z.get("age_bars", 0), "status": z.get("status", "Testee"),
                               "type": "Pivot" if is_piv else "Support", "is_pivot": is_piv})
        for _, z in resistances.iterrows():
            is_piv = bool(z.get("is_pivot", False))
            all_zones.append({"tf": tf, "level": z["level"], "strength": z["strength"],
                               "age_bars": z.get("age_bars", 0), "status": z.get("status", "Testee"),
                               "type": "Pivot" if is_piv else "Resistance", "is_pivot": is_piv})
    if not all_zones:
        return []

    # Patch m4 — filtrer Consommées AVANT déduplication
    # AVANT : dédup puis filtre → une zone Consommée pouvait écraser une Vierge au même niveau
    # APRÈS : filtre d'abord → Vierge/Testée priment dans la dédup
    all_zones = [z for z in all_zones if z.get("status") != "Consommee"]
    if not all_zones:
        return []

    # FIX B (v5.6) — déduplication par (level, tf)
    seen_keys: set = set()
    deduped: list  = []
    for z in all_zones:
        key = (round(z["level"], 6), z["tf"])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(z)
    all_zones = deduped
    if not all_zones:
        return []

    zones_df     = pd.DataFrame(all_zones).sort_values("level").reset_index(drop=True)
    used_indices = set()
    confluences  = []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue
        if zone["level"] <= 0:
            used_indices.add(i)
            continue

        # Patch C2 — exclure les zones déjà utilisées de la recherche de `similar`
        # AVANT : zones déjà dans un groupe pouvaient apparaître dans d'autres groupes
        # → double-comptage silencieux, scores gonflés artificiellement
        # APRÈS : ~zones_df.index.isin(used_indices) garantit l'exclusivité
        available_mask = ~zones_df.index.isin(used_indices)
        similar = zones_df[
            available_mask &
            (abs(zones_df["level"] - zone["level"]) / zone["level"] * 100 <= confluence_threshold) &
            (zones_df.index != i)
        ]

        if len(similar) >= 1:
            group      = pd.concat([zones_df.loc[[i]], similar])
            timeframes = group["tf"].unique()
            if len(timeframes) >= 2:
                used_indices.update(group.index)
                group_sup = group[group["level"] < current_price]
                group_res = group[group["level"] >= current_price]
                is_mixed  = (not group_sup.empty) and (not group_res.empty)
                subgroups = ([(group_sup, "Support"), (group_res, "Resistance")]
                             if is_mixed else [(group, None)])
                for subgroup, forced_type in subgroups:
                    if subgroup.empty:
                        continue
                    sub_tfs = subgroup["tf"].unique()
                    if not is_mixed and len(sub_tfs) < 2:
                        continue
                    sub_avg   = subgroup["level"].mean()
                    sub_nb_tf = len(sub_tfs)
                    sub_dist  = abs(current_price - sub_avg) / current_price * 100
                    sub_active = subgroup[subgroup["status"] != "Consommee"]
                    if sub_active.empty:
                        sub_active = subgroup
                    # Patch m6 — total_bars réel par TF au lieu de 500 hardcodé
                    sub_score = sum(
                        compute_structural_score(
                            int(r["strength"]), sub_nb_tf,
                            tf_name=r["tf"], age_bars=int(r.get("age_bars", 0)),
                            total_bars=total_bars_map.get(r["tf"], 500) if total_bars_map else 500,
                        )
                        for _, r in sub_active.iterrows()
                    )
                    sub_strength = int(subgroup["strength"].sum())
                    sub_status   = min(subgroup["status"].tolist(),
                                       key=lambda s: STATUS_PRIORITY.get(s, 1))
                    # FIX 5 — pivot band ATR-based
                    if _ref_atr and _ref_atr > 0 and current_price > 0:
                        is_pivot = abs(sub_avg - current_price) <= _ref_atr * PIVOT_BAND_ATR_COEFF
                    else:
                        is_pivot = sub_dist <= 0.50
                    if is_pivot:
                        sub_type, sub_signal = "Pivot", "↔ PIVOT ZONE"
                    elif forced_type:
                        sub_type   = forced_type
                        sub_signal = "🟢 BUY ZONE" if forced_type == "Support" else "🔴 SELL ZONE"
                    else:
                        n_sup = (subgroup["level"] < current_price).sum()
                        n_res = (subgroup["level"] >= current_price).sum()
                        if n_sup > n_res:
                            sub_type, sub_signal = "Support", "🟢 BUY ZONE"
                        elif n_res > n_sup:
                            sub_type, sub_signal = "Resistance", "🔴 SELL ZONE"
                        else:
                            sub_type   = "Support" if sub_avg < current_price else "Resistance"
                            sub_signal = "🟢 BUY ZONE" if sub_type == "Support" else "🔴 SELL ZONE"
                    sub_tf_label = " + ".join(sorted(sub_tfs))
                    sub_alerte   = ("🔥 ZONE CHAUDE" if sub_dist < 0.5
                                    else ("⚠️ Proche" if sub_dist < 1.5 else ""))
                    confluences.append({
                        "Actif": symbol, "Signal": sub_signal,
                        "Niveau": f"{sub_avg:.5f}", "Type": sub_type,
                        "Timeframes": sub_tf_label, "Nb TF": sub_nb_tf,
                        "Force Totale": sub_strength, "Score": round(sub_score, 1),
                        "Statut": sub_status, "Distance %": f"{sub_dist:.2f}%",
                        "Alerte": sub_alerte,
                    })
            else:
                used_indices.add(i)
        else:
            used_indices.add(i)
    return confluences


# ══════════════════════════════════════════════════════════════════
# SCAN RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════
@dataclass
class ScanResult:
    symbol:     str
    rows:       dict
    zones:      dict
    price:      Optional[float]
    trends:     dict
    anomaly:    Optional[str]
    scan_error: Optional[str] = None


def scan_single_symbol(args) -> ScanResult:
    symbol, base_url, access_token, account_id, zone_width, min_touches = args
    atr_zone_coeff  = ATR_ZONE_COEFF.get(symbol, DEFAULT_ATR_COEFF)
    prom_coeff      = PROMINENCE_COEFF.get(symbol, DEFAULT_PROMINENCE_COEFF)
    zone_w_fallback = ZONE_WIDTH_FALLBACK.get(symbol, zone_width)
    merge_thresh    = POST_MERGE_MAP.get(symbol, POST_MERGE_THRESHOLD)
    pdf_dist_max    = PDF_DIST_THRESHOLDS.get(symbol, DEFAULT_PDF_DIST)
    abs_dist_max    = ABSOLUTE_MAX_DIST.get(symbol, 99.0)

    rows          = {"H4": None, "Daily": None, "Weekly": None}
    zones_d       = {}
    trends        = {}
    all_sup_levels = []
    last_h4_close  = None
    anomaly_msg    = None
    total_bars_map = {}  # Patch m6 — longueur réelle par TF
    sym_d = symbol.replace("_", "/")
    _TF_KEYS = [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]

    # Patch M6 — Suppression du ThreadPoolExecutor imbriqué
    # AVANT : ThreadPoolExecutor(max_workers=3) à l'intérieur d'un autre pool(max_workers=4)
    #         → jusqu'à 12 threads + contention + nested thread overhead
    # APRÈS : fetch séquentiel des TF par symbole, parallèle entre symboles (pool externe)
    #         → max 4 threads simultanés, trafic OANDA maîtrisé, code plus lisible
    tf_data = {}
    for tk, _ in _TF_KEYS:
        try:
            df = get_oanda_data(base_url, access_token, account_id, symbol, tk, limit=500)
            tf_data[tk] = df
        except Exception as e:
            logger.error(f"Fetch {symbol}/{tk}: {type(e).__name__}: {str(e)[:100]}")
            tf_data[tk] = None

    raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
    # Patch M3 — passe h4_df directement, plus de re-fetch dans validate_live_price
    current_price, price_alert_msg = validate_live_price(raw_price, symbol, tf_data.get("h4"))
    anomaly_msg = price_alert_msg
    reference_price = current_price

    for tf_key, tf_cap in _TF_KEYS:
        df = tf_data.get(tf_key)
        if df is None or df.empty:
            continue
        # Patch M7 (GLM) — try/except par TF : un crash sur un TF ne tue plus tout le symbole
        try:
            if reference_price is None:
                reference_price = float(df["close"].iloc[-1])
            if tf_key == "h4":
                last_h4_close = float(df["close"].iloc[-1])
            cp = reference_price
            trends[tf_cap] = compute_trend(df, timeframe_key=tf_key)
            atr_val = compute_atr(df, period=14)
            total_bars_map[tf_cap] = len(df)  # Patch m6
            supports, resistances = find_strong_sr_zones(
                df, cp, symbol=symbol, atr_val=atr_val,
                zone_percentage_width=zone_w_fallback,
                atr_zone_coeff=atr_zone_coeff, prominence_coeff=prom_coeff,
                min_touches=min_touches, timeframe=tf_key,
                post_merge_threshold=merge_thresh,
            )
            zones_d[tf_cap] = (supports, resistances)
            zones_d[f"_atr_{tf_key}"] = atr_val
            if not supports.empty:
                all_sup_levels.extend(supports["level"].tolist())
            if tf_key == "daily":
                zones_d["_price_ctx"] = get_price_context(cp, supports, resistances,
                                                           min_strength_anchor=min_touches)
            elif tf_key == "h4":
                zones_d["_price_ctx_h4"] = get_price_context(cp, supports, resistances,
                                                               min_strength_anchor=min_touches)

            def make_row(zone, ztype, _cp=cp, _atr=atr_val,
                         _pdf_max=pdf_dist_max, _abs_max=abs_dist_max,
                         _tf=tf_cap, _ntot=len(df)):
                lvl      = zone["level"]
                strength = int(zone["strength"])
                age_bars = int(zone.get("age_bars", 0))
                status   = zone.get("status", "Testee")
                if not _cp or _cp <= 0:
                    dist_pct, dist_atr_str, in_pdf = 0.0, "N/A", False
                else:
                    dist_pct = abs(_cp - lvl) / _cp * 100
                    dist_atr_str = f"{round(abs(_cp - lvl) / _atr, 1):.1f}x" if (_atr and _atr > 0) else "N/A"
                    in_pdf = dist_pct <= _pdf_max and dist_pct <= _abs_max
                struct_score = compute_structural_score(strength, 1, _tf, age_bars, _ntot)
                return {
                    "Actif": sym_d, "Prix Actuel": f"{_cp:.5f}" if _cp else "N/A",
                    "Type": ztype, "Niveau": f"{lvl:.5f}", "Force": f"{strength} touches",
                    "Score (1TF)": struct_score, "Statut": status,
                    "Dist. %": f"{dist_pct:.2f}%", "Dist. ATR": dist_atr_str,
                    "_dist_num": dist_pct, "_in_pdf": in_pdf,
                }

            tf_rows = (
                [make_row(z, "PIVOT" if z.get("is_pivot") else "Support") for _, z in supports.iterrows()] +
                [make_row(z, "PIVOT" if z.get("is_pivot") else "Resistance") for _, z in resistances.iterrows()]
            )
            seen_levels = set()
            unique_rows = []
            for r in tf_rows:
                key = (r["Niveau"], r["Type"])
                if key not in seen_levels:
                    seen_levels.add(key)
                    unique_rows.append(r)
            if unique_rows:
                rows[tf_cap] = unique_rows

        except Exception as e:
            logger.error(f"Erreur traitement TF {tf_cap} pour {symbol}: {type(e).__name__}: {str(e)[:150]}")
            continue

    # Patch m6 — stocke total_bars_map dans zones_d pour detect_confluences
    zones_d["_total_bars"] = total_bars_map

    if all_sup_levels and reference_price:
        new_anomaly = flag_data_anomaly(symbol, reference_price, all_sup_levels, last_h4_close)
        if new_anomaly:
            anomaly_msg = f"{anomaly_msg} | {new_anomaly}" if anomaly_msg else new_anomaly

    return ScanResult(symbol=symbol, rows=rows, zones=zones_d,
                      price=reference_price, trends=trends, anomaly=anomaly_msg)


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION PDF / CSV / EXPORTS
# ══════════════════════════════════════════════════════════════════
_INTERNAL_COLS = ["_dist_num", "_in_pdf"]


def strip_emojis_df(df):
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        clean[col] = clean[col].astype(str).apply(_safe_pdf_str)
    return clean


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, _safe_pdf_str('Rapport Scanner Bluestar - Supports & Resistances'),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6, _safe_pdf_str(
            f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  |  "
            "Score confluence = Somme(Force_i x Poids_TF_i x NbTF x Age_i) | "
            "Statut Vierge / Testee / Role Reverse / Consommee"),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def chapter_anomalies(self, anomalies: dict):
        if not anomalies:
            return
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, _safe_pdf_str('ALERTES ANOMALIES PRIX'), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        self.set_font('Helvetica', '', 8)
        for sym, msg in anomalies.items():
            self.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}")[:180])
        self.ln(4)

    def chapter_summary(self, summaries):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, _safe_pdf_str('RESUME PAR ACTIF  (Tendances + Top Zones Confluentes)'),
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        for s in summaries:
            sym  = _safe_pdf_str(s.get('symbol', ''))
            t_h4 = _safe_pdf_str(s.get('trend_h4', 'N/A'))
            t_d  = _safe_pdf_str(s.get('trend_daily', 'N/A'))
            t_w  = _safe_pdf_str(s.get('trend_weekly', 'N/A'))
            ctx  = _safe_pdf_str(s.get('price_context', ''))
            self.set_font('Helvetica', 'B', 8)
            self.cell(0, 5, _safe_pdf_str(f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"),
                      border=0, new_x='LMARGIN', new_y='NEXT')
            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}", border=0, new_x='LMARGIN', new_y='NEXT')
            top = s.get('top_zones', [])
            self.set_font('Helvetica', '', 7)
            if top:
                for z in top:
                    txt = (f"  {_safe_pdf_str(str(z.get('Signal','')))}  "
                           f"Niv:{_safe_pdf_str(str(z.get('Niveau','')))}  "
                           f"Dist:{_safe_pdf_str(str(z.get('Distance %','')))}  "
                           f"Score:{_safe_pdf_str(str(z.get('Score','')))}  "
                           f"TF:{_safe_pdf_str(str(z.get('Timeframes','')))}  "
                           f"{_safe_pdf_str(str(z.get('Alerte','')))}")
                    self.cell(0, 4, txt[:130], border=0, new_x='LMARGIN', new_y='NEXT')
            else:
                self.cell(0, 4, "  Aucune confluence pour cet actif.", border=0, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)

    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.multi_cell(self.w - self.l_margin - self.r_margin, 10, "Aucune donnee a afficher.")
            self.ln()
            return
        if 'Timeframes' in df.columns:
            col_widths = {'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22,
                          'Timeframes': 50, 'Nb TF': 12, 'Force Totale': 20,
                          'Score': 18, 'Statut': 22, 'Distance %': 18, 'Alerte': 55}
        else:
            col_widths = {'Actif': 24, 'Prix Actuel': 24, 'Type': 20, 'Niveau': 24,
                          'Force': 20, 'Score (1TF)': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16}
        font_size = 7
        cols     = [c for c in col_widths if c in df.columns]
        total_w  = sum(col_widths[c] for c in cols)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start  = self.l_margin + max(0, (usable_w - total_w) / 2)
        self.set_font('Helvetica', 'B', font_size)
        self.set_x(x_start)
        for col_name in cols:
            self.cell(col_widths[col_name], 6, _safe_pdf_str(col_name),
                      border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()
        self.set_font('Helvetica', '', font_size)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col_name in cols:
                w = col_widths[col_name]
                val = _safe_pdf_str(str(row[col_name]))
                max_chars = int(w / 1.25)
                if len(val) > max_chars:
                    val = val[:max_chars - 1] + '.'
                self.cell(w, 5, val, border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()


def _apply_pdf_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "_in_pdf" in df.columns:
        df = df[df["_in_pdf"]].copy()
    elif "_dist_num" in df.columns:
        if "Actif" in df.columns:
            def _max_for_row(row):
                sym_key = str(row.get("Actif", "")).replace("/", "_")
                return min(DEFAULT_PDF_DIST, ABSOLUTE_MAX_DIST.get(sym_key, DEFAULT_PDF_DIST))
            mask = df.apply(lambda r: r["_dist_num"] <= _max_for_row(r), axis=1)
            df = df[mask].copy()
        else:
            df = df[df["_dist_num"] <= DEFAULT_PDF_DIST].copy()
    elif "Dist. %" in df.columns:
        df = df[df["Dist. %"].apply(_parse_dist_pct) <= DEFAULT_PDF_DIST].copy()
    return df.drop(columns=_INTERNAL_COLS, errors="ignore").reset_index(drop=True)


def create_pdf_report(results_dict, confluences_df=None, summaries=None, anomalies=None):
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
        clean_conf = strip_emojis_df(confluences_df.copy()).drop(columns=_INTERNAL_COLS, errors="ignore")
        if "Score" in clean_conf.columns:
            clean_conf = clean_conf.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_conf)
        pdf.ln(10)
    title_map = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journaliere (Daily)',
                 'Weekly': 'Analyse Hebdomadaire (Weekly)'}
    for tf_key, df in results_dict.items():
        if df is None or (hasattr(df, 'empty') and df.empty):
            continue
        pdf.chapter_title(title_map.get(tf_key, tf_key))
        clean_df = strip_emojis_df(df.copy()).drop(columns=_INTERNAL_COLS, errors="ignore")
        if "Score (1TF)" in clean_df.columns:
            clean_df = clean_df.sort_values("Score (1TF)", ascending=False)
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


def _normalize_tf_label(tf_str: str) -> str:
    return tf_str.replace("Daily", "D").replace("Weekly", "W").replace(" + ", "+")


def create_llm_brief(summaries, confluences_df, max_dist=2.0, min_score=100.0,
                     allowed_statuts=("Vierge", "Testee", "Role Reverse"),
                     session_ts: Optional[datetime] = None):
    TREND_ARROW  = {"HAUSSIER": "↑", "BAISSIER": "↓", "NEUTRE": "→"}
    STATUS_LABEL = {"Vierge": "V", "Testee": "T", "Role Reverse": "RR", "Consommee": "C"}
    ALERT_LABEL  = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠"}
    ts  = session_ts if session_ts is not None else datetime.now()
    now = ts.strftime("%d/%m/%Y %H:%M")
    lines = [
        "# BRIEF S/R — Scanner Bluestar",
        f"_Généré le {now} (session ID: {ts.strftime('%Y%m%d_%H%M%S')})_",
        "",
        "## INSTRUCTIONS POUR LLM",
        "Ce brief contient les zones Support/Résistance les plus fiables détectées",
        "par un scanner multi-timeframes (H4 + Daily + Weekly) sur 33 actifs Forex/Indices/Métaux.",
        "",
        "**⚠️ IMPORTANT — Tendances (H4/D/W dans les en-têtes) :**",
        "Ces flèches sont calculées par le scanner S/R (SMA20 + structure swing).",
        "Elles sont INDICATIVES UNIQUEMENT. Pour le biais directionnel, se référer",
        "EXCLUSIVEMENT au fichier GPS MTF qui utilise un algorithme multi-timeframes dédié.",
        "En cas de conflit flèche ↑/↓ vs GPS : le GPS a la priorité absolue.",
        "",
        "**Légende :**",
        "- `BUY` / `SELL` / `PIVOT` : direction de la zone",
        "- `Sc` : Score pondéré = Σ(Force_i × Poids_TF_i × NbTF × Âge_i). >300=institutionnel, 100-300=fort",
        "- `V` = Vierge | `T` = Testée | `RR` = Role Reverse | `C` = Consommée (éviter)",
        "- `Dist%` : distance du prix actuel à la zone",
        "- `TFs` : timeframes en confluence (H4/D/W)",
        "- `⚡` = zone chaude (<0.5% du prix) | `⚠` = proche (<1.5%)",
        "- `PIVOT` = prix sur la zone — inclus dans brief ET JSON depuis v5.6",
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
        # Patch m7 — _parse_dist_pct DRY
        dist_val  = _parse_dist_pct(row.get("Distance %", "999"))
        try:    score_val = float(row.get("Score", 0))
        except: score_val = 0.0
        statut = str(row.get("Statut", ""))
        if dist_val > max_dist or score_val < min_score or statut not in allowed_statuts:
            continue
        actif = str(row.get("Actif", ""))
        if actif not in actif_zones:
            actif_zones[actif] = []
        actif_zones[actif].append({
            "signal": str(row.get("Signal", "")), "niveau": str(row.get("Niveau", "")),
            "score": score_val, "statut": statut, "dist": dist_val,
            "tfs": str(row.get("Timeframes", "")), "nb_tf": int(row.get("Nb TF", 0)),
            "alerte": str(row.get("Alerte", "")),
        })
    actif_max_score = {a: max(z["score"] for z in zones) for a, zones in actif_zones.items()}
    sorted_actifs   = sorted(actif_max_score, key=lambda a: actif_max_score[a], reverse=True)
    summary_map     = {s["symbol"]: s for s in summaries}
    total_zones = 0
    for actif in sorted_actifs:
        zones = sorted(actif_zones[actif], key=lambda z: z["score"], reverse=True)
        s     = summary_map.get(actif, {})
        t_h4  = TREND_ARROW.get(s.get("trend_h4",   "NEUTRE"), "→")
        t_d   = TREND_ARROW.get(s.get("trend_daily", "NEUTRE"), "→")
        t_w   = TREND_ARROW.get(s.get("trend_weekly","NEUTRE"), "→")
        ctx   = s.get("price_context", "")
        lines.append(f"### {actif} | H4:{t_h4} D:{t_d} W:{t_w}")
        if ctx:
            lines.append(f"> {ctx}")
        for z in zones:
            # Patch m7 — _determine_signal DRY
            signal_short = _determine_signal(z["signal"]).split()[0]
            st_short = STATUS_LABEL.get(z["statut"], z["statut"])
            al_short = ALERT_LABEL.get(z["alerte"], "")
            tf_short = _normalize_tf_label(z["tfs"])
            lines.append(f"- {signal_short} `{z['niveau']}` | Sc:{z['score']:.0f} | {st_short} | {z['dist']:.2f}% | {tf_short} {al_short}")
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
        "2. Vérifie la cohérence tendance vs direction de zone — utilise le GPS MTF pour le biais",
        "3. Priorise les zones Vierge (V) sur 3 TF avec Score > 200",
        "4. Les zones Role Reverse (RR) = pullback sur niveau cassé, setup souvent court terme",
        "5. Les zones PIVOT = prix actuel dessus, attendre confirmation avant d'agir",
        "6. Propose un plan de trade structuré : entrée, SL (au-delà de la zone), TP (prochain niveau)",
        "```",
    ]
    return "\n".join(lines).encode("utf-8")


def create_json_export(summaries, confluences_df, max_dist=5.0, min_score=50.0,
                       session_ts: Optional[datetime] = None):
    ts = session_ts if session_ts is not None else datetime.now()
    output = {
        "generated_at": ts.isoformat(),
        "session_id":   ts.strftime("%Y%m%d_%H%M%S"),
        "filters": {
            "max_dist_pct":   max_dist,
            "min_score":      min_score,
            "statuts_exclus": ["Consommee"],
            "pivots_inclus":  True,
            "note": (
                "v5.8 : C1 session thread-local, C2 detect_confluences dedup fix, "
                "C3 JSON parsing sécurisé, M1 cache LRU, M2 candles[-limit:], "
                "M3 validate sans re-fetch, M4 rate limiter, M5 timeout futures, "
                "M6 total_bars réel, M7 logging, m1-m8 mineurs."
            ),
        },
        "assets": []
    }
    summary_map = {s["symbol"]: s for s in summaries}
    if confluences_df is None or confluences_df.empty:
        return json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")
    actif_groups = {}
    for _, row in confluences_df.iterrows():
        dist_val  = _parse_dist_pct(row.get("Distance %", "999"))  # Patch m7 DRY
        try:
            score_val = float(row.get("Score", 0))
            if not pd.notna(score_val): score_val = 0.0
        except: score_val = 0.0
        if dist_val > max_dist or score_val < min_score:
            continue
        if str(row.get("Statut", "")) == "Consommee":
            continue
        signal_clean = _determine_signal(str(row.get("Signal", "")))  # Patch m7 DRY
        actif = str(row.get("Actif", ""))
        if actif not in actif_groups:
            actif_groups[actif] = []
        alert_raw   = str(row.get("Alerte", ""))
        alert_clean = alert_raw.replace("🔥 ", "").replace("⚠️ ", "").replace("🔥", "").replace("⚠️", "").strip()
        actif_groups[actif].append({
            "signal":       signal_clean,
            "level":        str(row.get("Niveau", "")),
            "score":        round(score_val, 1),
            "status":       str(row.get("Statut", "")),
            "distance_pct": round(dist_val, 3),
            "alert":        alert_clean,
            "timeframes":   _normalize_tf_label(str(row.get("Timeframes", ""))),
        })
    for actif, zones in sorted(actif_groups.items(),
                                key=lambda x: max(z["score"] for z in x[1]), reverse=True):
        s = summary_map.get(actif, {})
        output["assets"].append({
            "symbol":        actif,
            "price_context": s.get("price_context", ""),
            "zones":         sorted(zones, key=lambda z: z["score"], reverse=True),
        })
    return json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE STREAMLIT
# ══════════════════════════════════════════════════════════════════
def _display_results(sr: dict, max_dist_filter: float):
    df_h4     = sr.get("df_h4",       pd.DataFrame())
    df_daily  = sr.get("df_daily",     pd.DataFrame())
    df_wk     = sr.get("df_weekly",    pd.DataFrame())
    conf_full = sr.get("conf_full",    pd.DataFrame())
    rep_dict  = sr.get("report_dict",  {})
    summaries = sr.get("summaries",    [])
    anomalies = sr.get("anomalies",    {})
    errors    = sr.get("scan_errors",  {})
    session_ts = sr.get("session_ts",  datetime.now())

    if not conf_full.empty:
        tmp = conf_full.drop(columns=_INTERNAL_COLS, errors="ignore").copy()
        tmp["_dist_num"] = tmp["Distance %"].apply(_parse_dist_pct)  # Patch m7 DRY
        conf_filt = tmp[tmp["_dist_num"] <= max_dist_filter].drop(columns=["_dist_num"], errors="ignore").reset_index(drop=True)
    else:
        conf_filt = pd.DataFrame()

    tf_cfg = {
        "Actif": st.column_config.TextColumn("Actif", width="small"),
        "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"),
        "Type": st.column_config.TextColumn("Type", width="small"),
        "Niveau": st.column_config.TextColumn("Niveau", width="small"),
        "Force": st.column_config.TextColumn("Force", width="medium"),
        "Score (1TF)": st.column_config.NumberColumn("Score (1TF) ▼", width="small"),
        "Statut": st.column_config.TextColumn("Statut", width="small"),
        "Dist. %": st.column_config.TextColumn("Dist. %", width="small"),
        "Dist. ATR": st.column_config.TextColumn("Dist. ATR", width="small"),
    }

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
        st.caption("Score confluence = Σ(Force_i × Poids_TF_i × NbTF × Âge_i) | Score (1TF) = mono-TF brut")
        disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total zones",      len(disp))
        c2.metric("🔥 Zones chaudes", len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
        c3.metric("⚠️ Zones proches", len(disp[disp["Alerte"] == "⚠️ Proche"]))
        c4.metric("🟢 BUY Zones",     len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
        c5.metric("🔴 SELL Zones",    len(disp[disp["Signal"] == "🔴 SELL ZONE"]))
        c6.metric("↔ PIVOT Zones",    len(disp[disp["Signal"] == "↔ PIVOT ZONE"]))
        conf_cfg = {
            **{k: st.column_config.TextColumn(k, width="small")
               for k in ["Actif","Signal","Niveau","Type","Timeframes","Statut","Distance %","Alerte"]},
            "Nb TF":        st.column_config.NumberColumn("Nb TF",        width="small"),
            "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
            "Score":        st.column_config.NumberColumn("Score ▼",      width="small"),
        }
        st.dataframe(disp, column_config=conf_cfg, hide_index=True,
                     use_container_width=True, height=min(len(disp) * 35 + 38, 750))
    else:
        st.info("Aucune confluence dans la plage sélectionnée. Augmentez le filtre ou le seuil.")

    st.subheader("📋 Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_pdf_report(rep_dict, conf_full, summaries, anomalies)
            st.download_button("📄 Rapport PDF (complet)", data=pdf_bytes,
                               file_name=f"rapport_bluestar_{session_ts.strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf")
        with col2:
            csv_bytes = create_csv_report(rep_dict, conf_full)
            st.download_button("📊 Données brutes CSV", data=csv_bytes,
                               file_name=f"donnees_bluestar_{session_ts.strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")
        st.divider()
        st.markdown("**🤖 Exports optimisés LLM**")
        st.caption("Paramètres LLM configurables dans la barre latérale (section 3).")
        llm_max_dist    = st.session_state.get("llm_max_dist",  2.0)
        llm_min_score   = st.session_state.get("llm_min_score", 100)
        llm_statuts_raw = st.session_state.get("llm_statuts",   ["Vierge", "Testee", "Role Reverse"])
        llm_statuts     = tuple(llm_statuts_raw) if llm_statuts_raw else ("Vierge", "Testee", "Role Reverse")
        st.caption(f"🔧 Filtres actifs : Dist < **{llm_max_dist}%** | Score ≥ **{llm_min_score}** | {', '.join(llm_statuts)}")
        col3, col4 = st.columns(2)
        with col3:
            md_bytes = create_llm_brief(summaries, conf_full, max_dist=llm_max_dist,
                                         min_score=llm_min_score, allowed_statuts=llm_statuts,
                                         session_ts=session_ts)
            st.download_button("🤖 Brief LLM (Markdown filtré)", data=md_bytes,
                               file_name=f"brief_llm_{session_ts.strftime('%Y%m%d_%H%M')}.md",
                               mime="text/markdown")
        with col4:
            # Patch m5 — JSON utilise maintenant llm_max_dist + llm_min_score (cohérence avec brief LLM)
            # AVANT : min_score=50.0 hardcodé, max_dist=max_dist_filter → incohérent avec la sidebar
            json_bytes = create_json_export(summaries, conf_full,
                                             max_dist=llm_max_dist,
                                             min_score=float(llm_min_score),
                                             session_ts=session_ts)
            st.download_button("🔧 Export JSON structuré", data=json_bytes,
                               file_name=f"sr_bluestar_{session_ts.strftime('%Y%m%d_%H%M')}.json",
                               mime="application/json")
        st.divider()
        st.markdown("**👁️ Aperçu du Brief LLM**")
        st.caption(f"Filtres : Dist < {llm_max_dist}% | Score ≥ {llm_min_score} | {', '.join(llm_statuts)}")
        try:
            brief_preview = md_bytes.decode("utf-8")
            n_zones  = sum(1 for l in brief_preview.split("\n")
                           if l.strip().startswith(("- BUY", "- SELL", "- PIVOT")))
            n_actifs = brief_preview.count("### ")
            st.info(f"**{n_actifs} actifs** avec **{n_zones} zones** dans le brief LLM "
                    f"(≈ {n_zones * 15 + n_actifs * 10:,} tokens estimés)")
            st.text_area("Brief LLM (copiable directement)", value=brief_preview,
                         height=400, label_visibility="collapsed")
        except Exception:
            st.warning("Aperçu non disponible.")

    def _filter_and_sort(df, max_pct):
        if df.empty or "Dist. %" not in df.columns:
            return df
        mask     = df["Dist. %"].apply(_parse_dist_pct) <= max_pct  # Patch m7 DRY
        out      = df[mask].drop(columns=_INTERNAL_COLS, errors="ignore")
        sort_col = "Score (1TF)" if "Score (1TF)" in out.columns else "Score"
        if sort_col in out.columns:
            out = out.sort_values(sort_col, ascending=False)
        return out.reset_index(drop=True)

    st.divider()
    st.subheader("📅 Analyse 4 Heures (H4)")
    fd = _filter_and_sort(df_h4, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True, use_container_width=True,
                 height=min(len(fd) * 35 + 38, 600))
    st.subheader("📅 Analyse Journalière (Daily)")
    fd = _filter_and_sort(df_daily, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True, use_container_width=True,
                 height=min(len(fd) * 35 + 38, 600))
    st.subheader("📅 Analyse Hebdomadaire (Weekly)")
    fd = _filter_and_sort(df_wk, max_dist_filter)
    st.dataframe(fd, column_config=tf_cfg, hide_index=True, use_container_width=True,
                 height=min(len(fd) * 35 + 38, 600))


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
        default_sel = ["XAU_USD","EUR_USD","GBP_USD","USD_JPY","AUD_USD","EUR_JPY","GBP_JPY"]
        symbols_to_scan = st.multiselect("Actifs spécifiques :", options=ALL_SYMBOLS, default=default_sel)

    st.header("3. Paramètres Export LLM")
    st.caption("Ces paramètres survivent aux re-renders.")
    st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    st.slider("Score min brief LLM", 50, 300, 100, 25, key="llm_min_score")
    st.multiselect("Statuts autorisés (brief LLM)",
                   options=["Vierge","Testee","Role Reverse","Consommee"],
                   default=["Vierge","Testee","Role Reverse"], key="llm_statuts")

    st.divider()
    st.header("4. Paramètres de Détection")
    zone_width = st.slider("Largeur zone Forex (% fallback ATR)", 0.1, 2.0, 0.5, 0.1)
    min_touches = st.slider("Force minimale (touches)", 3, 10, 3, 1)
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 1.0, 0.1)
    max_dist_filter = st.slider("Afficher zones < (%) - filtre visuel", 1.0, 15.0, 3.0, 0.5)

    st.divider()
    st.caption("**Score = Σ(Force_i × Poids_TF_i × NbTF × Âge_i)**")
    st.caption("🔴 >300 : Institutionnel | 🟠 100-300 : Fort | 🟡 30-100 : Valide | ⚪ <30 : Secondaire")
    st.caption("✅ Vierge | 🔵 Testée | ↩️ Role Reverse | ❌ Consommée")
    st.divider()
    st.caption(f"**v{SCANNER_VERSION} — Fixes appliqués (audit 4 sources) :**")
    st.caption("🔴 C1 — Session HTTP thread-local (non thread-safe corrigé)")
    st.caption("🔴 C2 — detect_confluences : fin du double-comptage used_indices")
    st.caption("🔴 C3 — Parsing JSON OANDA sécurisé (.get() défensif)")
    st.caption("🟠 M1 — Cache LRU (_oanda_cache + _price_cache bornés)")
    st.caption("🟠 M2 — candles[-limit:] (off-by-one bougies récentes)")
    st.caption("🟠 M3 — validate_live_price sans re-fetch H4")
    st.caption("🟠 M4 — Rate limiter sémaphore (max 10 req. concurrentes)")
    st.caption("🟠 M5 — as_completed(timeout=180s) anti-freeze UI")
    st.caption("🟠 M6 — Nested ThreadPool supprimé → fetch séquentiel par TF")
    st.caption("🟠 M7 — Logging structuré (WARNING/ERROR/CRITICAL par type)")
    st.caption("🟡 m1 — ZeroDivisionError compute_trend protégé")
    st.caption("🟡 m2 — _safe_pdf_str encodage latin-1 complet")
    st.caption("🟡 m3 — PRICE_SANITY_RANGE : 7 paires JPY ajoutées")
    st.caption("🟡 m4 — Filtre Consommée avant dédup dans detect_confluences")
    st.caption("🟡 m5 — Export JSON min_score = sidebar llm_min_score")
    st.caption("🟡 m6 — total_bars réel par TF (plus 500 hardcodé)")
    st.caption("🟡 m7 — _parse_dist_pct + _determine_signal DRY")
    st.caption("🟡 m8 — zone_width_abs hors boucle (calc. unique)")
    st.caption("🟡 m9 — account_id dans clé cache (isolation multi-compte)")
    st.caption("🟡 m10 — width=stretch → use_container_width=True (Streamlit compat)")
    st.caption("🟡 m11 — LOG_LEVEL via os.getenv (configurable VPS)")
    st.divider()
    st.caption("**Fixes v5.7 :** FIX 1-5 (ATR cohérence, confluences mixtes, merge, pivot band)")
    st.caption("**Fixes v5.6 :** FIX A-F (mono-TF, PIVOT, JSON, trend, PDF, price_ctx)")

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
            session_ts   = datetime.now()
            progress_bar = st.progress(0, text="Initialisation du scan…")
            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map, prices_map, trends_map, anomalies_map, scan_errors = {}, {}, {}, {}, {}
            args_list = [(sym, base_url, access_token, account_id, zone_width, min_touches)
                         for sym in symbols_to_scan]
            total, completed = len(args_list), 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_map = {executor.submit(scan_single_symbol, a): a[0] for a in args_list}
                # Patch M5 — timeout=180s sur as_completed : évite le freeze UI indéfini
                try:
                    for future in concurrent.futures.as_completed(future_map, timeout=180):
                        sym = future_map[future]
                        completed += 1
                        progress_bar.progress(completed / total,
                                              text=f"Scan… ({completed}/{total}) {sym.replace('_','/')}")
                        try:
                            result: ScanResult = future.result()
                            all_zones_map[result.symbol] = result.zones
                            prices_map[result.symbol]    = result.price
                            trends_map[result.symbol]    = result.trends
                            if result.anomaly:
                                anomalies_map[result.symbol.replace("_", "/")] = result.anomaly
                            for tf_cap, tf_rows in result.rows.items():
                                if tf_rows:
                                    if tf_cap == "H4":     results_h4.extend(tf_rows)
                                    elif tf_cap == "Daily":  results_daily.extend(tf_rows)
                                    elif tf_cap == "Weekly": results_weekly.extend(tf_rows)
                        except Exception as e:
                            scan_errors[sym.replace("_", "/")] = f"{type(e).__name__}: {str(e)[:120]}"
                            logger.error(f"Scan {sym}: {type(e).__name__}: {str(e)[:200]}")
                except concurrent.futures.TimeoutError:
                    scan_errors["__TIMEOUT__"] = (
                        f"Timeout global 180s atteint — {completed}/{total} actifs traités. "
                        "Résultats partiels affichés."
                    )
                    logger.warning(f"Scan timeout : {completed}/{total} actifs traités")

            progress_bar.empty()
            n_ok   = len([s for s in symbols_to_scan if s.replace("_", "/") not in scan_errors])
            n_fail = len(scan_errors)
            if n_fail == 0:
                st.success(f"✅ Scan terminé — {n_ok} actifs analysés avec succès.")
            else:
                st.warning(f"⚠️ Scan terminé — {n_ok} OK, {n_fail} en erreur.")
            if anomalies_map:
                st.warning(f"⚠️ {len(anomalies_map)} anomalie(s) de prix détectée(s).")

            st.info("🔍 Analyse des confluences multi-timeframes…")
            all_confluences = []
            for sym in symbols_to_scan:
                if sym.replace("_", "/") in scan_errors:
                    continue
                cp          = prices_map.get(sym)
                sym_raw     = all_zones_map.get(sym, {})
                zones_clean = {k: v for k, v in sym_raw.items() if not k.startswith("_")}
                atr_map     = {k.replace("_atr_", ""): v for k, v in sym_raw.items()
                               if k.startswith("_atr_") and v is not None}
                # Patch m6 — total_bars_map réel transmis à detect_confluences
                total_bars_map = sym_raw.get("_total_bars", {})
                sym_threshold  = CONFLUENCE_THRESHOLD_MAP.get(sym, confluence_threshold)
                all_confluences.extend(detect_confluences(
                    sym.replace("_", "/"), zones_clean, cp, sym_threshold,
                    atr_map=atr_map, total_bars_map=total_bars_map,
                ))

            conf_full = pd.DataFrame(all_confluences)
            if not conf_full.empty:
                conf_full = conf_full.drop(columns=_INTERNAL_COLS, errors="ignore")

            summaries = []
            for sym in symbols_to_scan:
                sym_d  = sym.replace("_", "/")
                trends = trends_map.get(sym, {})
                cp     = prices_map.get(sym)
                top_zones = []
                if not conf_full.empty and sym_d in conf_full["Actif"].values:
                    ac = conf_full[conf_full["Actif"] == sym_d].sort_values("Score", ascending=False)
                    top_zones = ac.head(3).to_dict("records")
                d_zones   = all_zones_map.get(sym, {})
                ctx_daily = d_zones.get("_price_ctx", "")
                ctx_h4    = d_zones.get("_price_ctx_h4", "")
                if ctx_daily and ctx_h4 and ctx_h4 != ctx_daily:
                    price_ctx = f"{ctx_daily}  ||  H4: {ctx_h4}"
                elif ctx_daily:
                    price_ctx = ctx_daily
                elif ctx_h4:
                    price_ctx = ctx_h4
                elif "Daily" in d_zones and cp:
                    sup_d, res_d = d_zones["Daily"]
                    price_ctx = get_price_context(cp, sup_d, res_d, min_strength_anchor=min_touches)
                else:
                    price_ctx = ""
                summaries.append({
                    "symbol": sym_d, "trend_h4": trends.get("H4", "NEUTRE"),
                    "trend_daily": trends.get("Daily", "NEUTRE"),
                    "trend_weekly": trends.get("Weekly", "NEUTRE"),
                    "price_context": price_ctx, "top_zones": top_zones,
                })

            df_h4    = pd.DataFrame(results_h4)
            df_daily = pd.DataFrame(results_daily)
            df_wk    = pd.DataFrame(results_weekly)
            rep_dict = {
                "H4":     _apply_pdf_filter(df_h4),
                "Daily":  _apply_pdf_filter(df_daily),
                "Weekly": _apply_pdf_filter(df_wk),
            }
            st.session_state["scan_results"] = {
                "df_h4": df_h4, "df_daily": df_daily, "df_weekly": df_wk,
                "conf_full": conf_full, "report_dict": rep_dict,
                "summaries": summaries, "anomalies": anomalies_map,
                "scan_errors": scan_errors, "max_dist": max_dist_filter,
                "session_ts": session_ts,
            }
            _display_results(st.session_state["scan_results"], max_dist_filter)
            logger.info(f"Scan complet | OK={n_ok} | Fail={n_fail} | "
                        f"Confluences={len(all_confluences)} | Durée estimée v5.8")

elif not symbols_to_scan:
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
else:
    st.info("Configurez les paramètres dans la barre latérale, puis cliquez sur **LANCER LE SCAN COMPLET**.")

if "scan_results" in st.session_state and not scan_button:
    _display_results(st.session_state["scan_results"], max_dist_filter)
