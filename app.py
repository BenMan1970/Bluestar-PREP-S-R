import hashlib
import json
import threading
import traceback
import concurrent.futures
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════
# SESSION HTTP — BUG-012 CORRIGÉ : threading.local() par thread
# ══════════════════════════════════════════════════════════════════
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Retourne la session HTTP du thread courant (créée à la demande)."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
        ))
        _thread_local.session = s
    return _thread_local.session

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

scan_button = st.button(
    "🚀 LANCER LE SCAN COMPLET",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.get("scanning", False),
)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════

SCANNER_VERSION = "6.0"

ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

CONFLUENCE_THRESHOLD_MAP = {
    "US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2,
    "XAU_USD": 1.5,
}

PDF_DIST_THRESHOLDS = {
    "US30_USD": 5.0, "NAS100_USD": 5.0, "SPX500_USD": 5.0, "DE30_EUR": 5.0,
    "XAU_USD": 8.0,
}
DEFAULT_PDF_DIST = 8.0

ABSOLUTE_MAX_DIST = {
    "XAU_USD":    8.0,
    "US30_USD":   5.0,
    "NAS100_USD": 5.0,
    "SPX500_USD": 5.0,
    "DE30_EUR":   5.0,
}

_GRANULARITY_MAP = {"h4": "H4", "daily": "D", "weekly": "W"}

# ══════════════════════════════════════════════════════════════════
# PROFILS D'INSTRUMENTS (MOTEUR v6.0)
# ══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    pip_value: float
    tick_size: float
    volatility_class: str      # LOW, MEDIUM, HIGH, EXTREME
    avg_spread_pips: float
    noise_factor: float        # rapport bruit / ATR
    cluster_radius_atr: float  # multiple d'ATR pour le clustering
    merge_threshold_atr: float
    pivot_prominence_atr: float
    price_sanity_range: Optional[Tuple[float, float]] = None
    dev_threshold_pct: float = 1.5
    skip_ratio_check: bool = False
    pip_floor_abs: float = 0.0  # plancher de tolérance en unités de prix

_PROFILES = {
    "EUR_USD": InstrumentProfile("EUR_USD", 0.0001, 0.00001, "LOW", 0.2, 0.3, 1.2, 0.8, 0.6, pip_floor_abs=0.00010),
    "GBP_USD": InstrumentProfile("GBP_USD", 0.0001, 0.00001, "MEDIUM", 0.4, 0.4, 1.3, 0.85, 0.65, pip_floor_abs=0.00010),
    "USD_JPY": InstrumentProfile("USD_JPY", 0.01, 0.001, "LOW", 0.3, 0.25, 0.9, 0.5, 0.5, pip_floor_abs=0.10),
    "XAU_USD": InstrumentProfile("XAU_USD", 0.01, 0.01, "HIGH", 20.0, 0.6, 2.0, 1.2, 1.0, price_sanity_range=(1500.0,6500.0), dev_threshold_pct=3.0, skip_ratio_check=True),
    "US30_USD": InstrumentProfile("US30_USD", 1.0, 1.0, "MEDIUM", 2.0, 0.3, 1.5, 0.9, 0.7, price_sanity_range=(20000.0,70000.0), dev_threshold_pct=2.5, skip_ratio_check=True),
    "NAS100_USD": InstrumentProfile("NAS100_USD", 1.0, 1.0, "HIGH", 3.0, 0.35, 1.5, 1.0, 0.8, price_sanity_range=(8000.0,35000.0), dev_threshold_pct=2.5, skip_ratio_check=True),
    "SPX500_USD": InstrumentProfile("SPX500_USD", 0.1, 0.1, "MEDIUM", 0.5, 0.3, 1.3, 0.8, 0.65, price_sanity_range=(3000.0,9000.0), dev_threshold_pct=2.0, skip_ratio_check=True),
    "DE30_EUR": InstrumentProfile("DE30_EUR", 0.1, 0.1, "MEDIUM", 0.5, 0.3, 1.3, 0.8, 0.65, price_sanity_range=(8000.0,30000.0), dev_threshold_pct=2.0, skip_ratio_check=True),
}
_DEFAULT_PROFILE = InstrumentProfile("DEFAULT", 0.0001, 0.00001, "MEDIUM", 0.5, 0.4, 1.2, 0.8, 0.6, pip_floor_abs=0.00010)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    base = symbol.split("_")[0]
    if base in ("EUR","GBP","AUD","NZD","CAD","CHF"):
        return InstrumentProfile(symbol, 0.0001, 0.00001, "MEDIUM", 0.5, 0.4, 1.2, 0.8, 0.6, pip_floor_abs=0.00010)
    if base == "JPY" or symbol.endswith("_JPY"):
        return InstrumentProfile(symbol, 0.01, 0.001, "LOW", 0.3, 0.25, 0.9, 0.5, 0.5, pip_floor_abs=0.10)
    return _DEFAULT_PROFILE


# ══════════════════════════════════════════════════════════════════
# RATE LIMITING API OANDA
# ══════════════════════════════════════════════════════════════════
_api_semaphore = threading.BoundedSemaphore(5)


# ══════════════════════════════════════════════════════════════════
# CACHE THREAD-SAFE BORNÉ
# ══════════════════════════════════════════════════════════════════
class _BoundedTTLCache:
    """Cache TTL à taille bornée avec éviction LRU. Thread-safe via verrou externe."""
    def __init__(self, maxsize: int, ttl: int):
        self._cache   = OrderedDict()
        self._maxsize = maxsize
        self._ttl     = ttl

    def get(self, key):
        entry = self._cache.get(key)
        if entry is None:
            return None
        val, ts = entry
        if (datetime.now() - ts).total_seconds() > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return val

    def set(self, key, val):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (val, datetime.now())
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def __len__(self):
        return len(self._cache)

_oanda_cache = _BoundedTTLCache(maxsize=300, ttl=600)
_price_cache = _BoundedTTLCache(maxsize=100, ttl=60)
_cache_lock  = threading.RLock()

def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]

def _cache_key(token: str, symbol: str, timeframe: str, limit: int) -> str:
    return f"{_token_fingerprint(token)}__{symbol}__{timeframe}__{limit}"

def _cache_get(key: str) -> Optional[pd.DataFrame]:
    with _cache_lock:
        df = _oanda_cache.get(key)
        if df is None:
            return None
        return df.copy()

def _cache_set(key: str, df: Optional[pd.DataFrame]) -> None:
    with _cache_lock:
        _oanda_cache.set(key, df)


# ══════════════════════════════════════════════════════════════════
# UTILITAIRE PDF
# ══════════════════════════════════════════════════════════════════
_ACCENT_MAP = str.maketrans(
    'àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
    'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN'
)

_EMOJI_MAP = [
    ('🟢', '[BUY]'),  ('🔴', '[SELL]'), ('🔥', '[CHAUD]'),
    ('↔️', '[PIVOT]'), ('↔',  '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠',  '[PROCHE]'),
    ('📈', ''), ('📉', ''),
    ('✅', '[OK]'), ('❌', '[X]'),
    ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'),
    ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]'),
]

def _safe_pdf_str(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_ACCENT_MAP)
    for emoji, replacement in _EMOJI_MAP:
        text = text.replace(emoji, replacement)
    return text

def _sanitize_traceback(tb: str, sensitive_values: list) -> str:
    for val in sensitive_values:
        if val and isinstance(val, str) and len(val) > 4:
            tb = tb.replace(val, f"***{val[-4:]}")
    return tb

_INTERNAL_COLS = ["_dist_num", "_in_pdf"]

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=_INTERNAL_COLS, errors="ignore")

def _sym_display(sym: str) -> str:
    return sym.replace("_", "/")


# ══════════════════════════════════════════════════════════════════
# API OANDA
# ══════════════════════════════════════════════════════════════════
def determine_oanda_environment(access_token: str, account_id: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {access_token}"}
    for url in [
        "https://api-fxpractice.oanda.com",
        "https://api-fxtrade.oanda.com",
    ]:
        try:
            with _api_semaphore:
                r = _get_session().get(
                    f"{url}/v3/accounts/{account_id}/summary",
                    headers=headers,
                    timeout=(3, 5),
                )
            if r.status_code == 200:
                return url
        except (requests.RequestException, ValueError):
            continue
    return None

def get_oanda_data(base_url: str, access_token: str, symbol: str,
                   timeframe: str = "daily", limit: int = 500) -> Optional[pd.DataFrame]:
    key = _cache_key(access_token, symbol, timeframe, limit)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    gran = _GRANULARITY_MAP.get(timeframe)
    if gran is None:
        raise ValueError(f"Timeframe inconnu: {timeframe!r}.")

    url     = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {"count": limit + 1, "granularity": gran, "price": "M"}
    try:
        with _api_semaphore:
            r = _get_session().get(url, headers=headers, params=params, timeout=(3, 8))
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
        df = pd.DataFrame(candles).tail(limit).set_index("date")
        if df.empty:
            return None
        _cache_set(key, df)
        return df
    except (requests.RequestException, ValueError):
        return None

def get_oanda_current_price(base_url: str, access_token: str,
                             account_id: str, symbol: str) -> Optional[float]:
    key = f"{_token_fingerprint(access_token)}__{symbol}__price"
    with _cache_lock:
        val = _price_cache.get(key)
        if val is not None:
            return val

    url     = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        with _api_semaphore:
            r = _get_session().get(
                url, headers=headers,
                params={"instruments": symbol},
                timeout=(3, 5),
            )
            r.raise_for_status()
            data = r.json()

        if "prices" in data and data["prices"]:
            bid = float(data["prices"][0]["closeoutBid"])
            ask = float(data["prices"][0]["closeoutAsk"])
            result = (bid + ask) / 2
        else:
            result = None
    except (requests.RequestException, ValueError, KeyError, IndexError):
        result = None

    if result is not None:
        with _cache_lock:
            _price_cache.set(key, result)
    return result


def validate_live_price(live_price, symbol, base_url, access_token):
    profile = get_profile(symbol)
    dev_threshold = profile.dev_threshold_pct

    h4_close = None
    df_check = get_oanda_data(base_url, access_token, symbol, "h4", limit=500)
    if df_check is not None and not df_check.empty:
        h4_close = float(df_check["close"].iloc[-1])
        if profile.price_sanity_range:
            lo, hi = profile.price_sanity_range
            if not (lo <= h4_close <= hi):
                h4_close = None

    if live_price is not None:
        if profile.price_sanity_range:
            lo, hi = profile.price_sanity_range
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
    profile = get_profile(symbol)
    messages = []
    if not profile.skip_ratio_check and len(support_levels) >= 3:
        median_sup = np.median(support_levels)
        if median_sup > 0 and median_sup > 0.01 * current_price:
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


def get_price_context(current_price, supports, resistances, max_dist_pct: float = 5.0):
    if not current_price or current_price <= 0:
        return "Prix indisponible"
    parts = []
    if not supports.empty:
        sup_nearby = supports[
            (supports["level"] < current_price) &
            (abs(supports["level"] - current_price) / current_price * 100 <= max_dist_pct)
        ]
        if not sup_nearby.empty:
            nearest_sup = sup_nearby.nlargest(1, "level").iloc[0]
            dist_s = abs(current_price - nearest_sup["level"]) / current_price * 100
            tag    = "SUR support" if dist_s < 0.5 else "S proche"
            parts.append(f"{tag}: {nearest_sup['level']:.5f} (-{dist_s:.2f}%)")
    if not resistances.empty:
        res_nearby = resistances[
            (resistances["level"] > current_price) &
            (abs(resistances["level"] - current_price) / current_price * 100 <= max_dist_pct)
        ]
        if not res_nearby.empty:
            nearest_res = res_nearby.nsmallest(1, "level").iloc[0]
            dist_r = abs(current_price - nearest_res["level"]) / current_price * 100
            tag    = "SUR resistance" if dist_r < 0.5 else "R proche"
            parts.append(f"{tag}: {nearest_res['level']:.5f} (+{dist_r:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"


def _parse_price_context_obstacles(ctx_str: str, current_price: float) -> dict:
    result: dict = {"nearest_support": None, "nearest_resistance": None}
    if not ctx_str or ctx_str == "Zone intermediaire" or not current_price:
        return result
    import re
    pat = re.compile(
        r"(SUR support|S proche|SUR resistance|R proche):\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
    )
    for m in pat.finditer(ctx_str):
        tag, level_str, dist_str = m.group(1), m.group(2), m.group(3)
        try:
            lvl  = float(level_str)
            dist = float(dist_str)
        except ValueError:
            continue
        entry = {
            "level":        lvl,
            "distance_pct": dist,
            "on_level":     abs(dist) < 0.5,
        }
        if tag in ("SUR support", "S proche"):
            result["nearest_support"] = entry
        else:
            result["nearest_resistance"] = entry
    return result


# ══════════════════════════════════════════════════════════════════
# MOTEUR D'ANALYSE v6.0
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

def compute_trend(df, sma_period=20):
    if df is None or len(df) < 15:
        return "NEUTRE"

    actual_period = min(sma_period, len(df) - 5)
    _MIN_SMA_BARS = 10
    if actual_period < _MIN_SMA_BARS:
        return "NEUTRE"

    close   = df["close"]
    sma     = close.rolling(actual_period).mean().iloc[-1]
    current = close.iloc[-1]

    base = close.iloc[-6]
    if not np.isfinite(base) or base <= 0:
        return "NEUTRE"
    slope_pct = (close.iloc[-1] - base) / base * 100

    n      = min(10, len(df))
    highs  = df["high"].iloc[-n:]
    lows   = df["low"].iloc[-n:]
    hh     = highs.iloc[-1] > highs.iloc[0]
    ll     = lows.iloc[-1]  < lows.iloc[0]
    lh     = highs.iloc[-1] < highs.iloc[0]
    hl     = lows.iloc[-1]  > lows.iloc[0]

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
    age_r = max(int(age_bars), 0) / max(int(total_bars), 1)
    age_f = float(np.exp(-1.5 * age_r))
    raw   = int(strength) * tf_w * nb_tf
    return round(raw * age_f, 1)


def detect_swing_pivots(df, profile, atr_val, timeframe="daily"):
    if not atr_val or atr_val <= 0:
        prominence = 0.001
    else:
        prominence = atr_val * profile.pivot_prominence_atr

    n = 3
    highs  = pd.Series(df["high"].values)
    lows   = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)
    opens  = pd.Series(df["open"].values)

    roll_high_left  = highs.shift(1).rolling(n, min_periods=n).max()
    roll_high_right = highs[::-1].shift(1).rolling(n, min_periods=n).max()[::-1]
    roll_low_left   = lows.shift(1).rolling(n, min_periods=n).min()
    roll_low_right  = lows[::-1].shift(1).rolling(n, min_periods=n).min()[::-1]

    next_close = closes.shift(-1).fillna(closes)

    candle_range  = (highs - lows).clip(lower=1e-10)
    body_top      = pd.Series(np.maximum(opens.values, closes.values))
    upper_wick_pct = (highs - body_top) / candle_range

    body_bottom   = pd.Series(np.minimum(opens.values, closes.values))
    lower_wick_pct = (body_bottom - lows) / candle_range

    _WICK_THRESHOLD = 0.20 if timeframe.lower() in ("h4", "m15", "m30") else 0.30

    sh_mask = (
        (highs > roll_high_left) &
        (highs > roll_high_right) &
        (next_close < highs) &
        (upper_wick_pct >= _WICK_THRESHOLD)
    )
    sl_mask = (
        (lows < roll_low_left) &
        (lows < roll_low_right) &
        (next_close > lows) &
        (lower_wick_pct >= _WICK_THRESHOLD)
    )

    roll_low_around  = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
    roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()
    sh_mask = sh_mask & ((highs - roll_low_around)  >= prominence)
    sl_mask = sl_mask & ((roll_high_around - lows)  >= prominence)

    swing_high_idx = sh_mask[sh_mask].index.tolist()
    swing_low_idx  = sl_mask[sl_mask].index.tolist()

    pivot_highs = (pd.Series(highs.values[swing_high_idx], index=swing_high_idx)
                   if swing_high_idx else pd.Series(dtype=float))
    pivot_lows  = (pd.Series(lows.values[swing_low_idx],   index=swing_low_idx)
                   if swing_low_idx  else pd.Series(dtype=float))
    return pivot_highs, pivot_lows


def cluster_zones(pivots_with_idx, atr_val, profile):
    if not pivots_with_idx:
        return []
    
    if atr_val and atr_val > 0:
        radius = atr_val * profile.cluster_radius_atr
    else:
        radius = 0.001
        
    zones_raw = []
    cur_group = [pivots_with_idx[0]]
    
    for price, idx in pivots_with_idx[1:]:
        anchor = cur_group[0][0]
        if abs(price - anchor) <= radius:
            cur_group.append((price, idx))
        else:
            zones_raw.append(cur_group)
            cur_group = [(price, idx)]
    zones_raw.append(cur_group)
    return zones_raw


def classify_zone_status(level, zone_type, df, formation_idx, atr_val, profile):
    if formation_idx >= len(df) - 1:
        return "Vierge"

    if atr_val and atr_val > 0:
        tolerance = atr_val * 0.25
    else:
        tolerance = max(level * 0.001, profile.pip_floor_abs)

    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]

    if len(c_arr) == 0:
        return "Vierge"

    near = (np.abs(c_arr - level) <= tolerance) | (
        (l_arr <= level + tolerance) & (h_arr >= level - tolerance)
    )
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
        second_break = rc_after > level + tolerance
    else:
        second_break = rc_after < level - tolerance

    if second_break.any():
        return "Consommee"

    return "Role Reverse"


def find_strong_sr_zones(df, current_price, symbol, atr_val, timeframe="daily", min_touches=2):
    profile = get_profile(symbol)

    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None:
        current_price = df["close"].iloc[-1]

    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(
        df, profile=profile, atr_val=atr_val, timeframe=timeframe
    )

    if len(pivot_highs) + len(pivot_lows) < 3:
        distance = get_adaptive_distance(timeframe)
        if atr_val and atr_val > 0:
            pk = {"distance": distance, "prominence": atr_val * profile.pivot_prominence_atr}
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

    pivots_with_idx.sort(key=lambda x: (x[0], x[1]))

    if not pivots_with_idx:
        return pd.DataFrame(), pd.DataFrame()

    zones_raw = cluster_zones(pivots_with_idx, atr_val, profile)

    strong = []
    for grp in zones_raw:
        if len(grp) < min_touches:
            continue
        prices   = [p for p, _ in grp]
        indices  = [i for _, i in grp]
        lvl      = float(np.mean(prices))
        strength = len(grp)
        last_idx = max(indices)
        age_bars = max(0, n_total - 1 - last_idx)

        if lvl <= 0:
            continue

        zone_type_tmp = "Support" if lvl < current_price else "Resistance"
        status = classify_zone_status(
            lvl, zone_type_tmp, df, last_idx, atr_val, profile
        )

        strong.append({
            "level":    lvl,
            "strength": strength,
            "age_bars": age_bars,
            "status":   status,
        })

    if not strong:
        return pd.DataFrame(), pd.DataFrame()

    threshold_abs = atr_val * profile.merge_threshold_atr if (atr_val and atr_val > 0) else current_price * 0.003
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}
    
    strong.sort(key=lambda x: x["level"])
    merged = []
    for z in strong:
        if not merged:
            merged.append(z)
        else:
            prev = merged[-1]
            if abs(z["level"] - prev["level"]) <= threshold_abs:
                new_level = (prev["level"] * prev["strength"] + z["level"] * z["strength"]) / (prev["strength"] + z["strength"])
                new_strength = prev["strength"] + z["strength"]
                new_age = min(prev["age_bars"], z["age_bars"])
                new_status = max(
                    [prev["status"], z["status"]],
                    key=lambda s: STATUS_PRIORITY.get(s, 1)
                )
                merged[-1] = {
                    "level": new_level,
                    "strength": new_strength,
                    "age_bars": new_age,
                    "status": new_status,
                }
            else:
                merged.append(z)

    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)

    PIVOT_BAND_PCT = 0.50

    if current_price and current_price > 0:
        pivot_mask = (
            abs(df_zones["level"] - current_price) / current_price * 100
            <= PIVOT_BAND_PCT
        )
    else:
        pivot_mask = pd.Series([False] * len(df_zones))

    df_zones["is_pivot"] = pivot_mask

    supports    = df_zones[df_zones["level"] <  current_price].copy()
    resistances = df_zones[df_zones["level"] >= current_price].copy()

    return supports, resistances


def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0,
                        bars_map=None):
    bars_map = bars_map or {}

    if not zones_dict or current_price is None:
        return []

    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, z in supports.iterrows():
            is_piv = bool(z.get("is_pivot", False))
            all_zones.append({
                "tf":       tf,
                "level":    z["level"],
                "strength": z["strength"],
                "age_bars": z.get("age_bars", 0),
                "status":   z.get("status", "Testee"),
                "type":     "Pivot" if is_piv else "Support",
                "is_pivot": is_piv,
            })
        for _, z in resistances.iterrows():
            is_piv = bool(z.get("is_pivot", False))
            all_zones.append({
                "tf":       tf,
                "level":    z["level"],
                "strength": z["strength"],
                "age_bars": z.get("age_bars", 0),
                "status":   z.get("status", "Testee"),
                "type":     "Pivot" if is_piv else "Resistance",
                "is_pivot": is_piv,
            })

    if not all_zones:
        return []

    all_zones = [z for z in all_zones if z.get("status") != "Consommee"]
    if not all_zones:
        return []

    zones_df     = (pd.DataFrame(all_zones)
                    .sort_values("level")
                    .reset_index(drop=True))
    used_indices = set()
    confluences  = []

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue

        if zone["level"] <= 0:
            used_indices.add(i)
            continue

        similar = zones_df[
            (abs(zones_df["level"] - zone["level"]) / zone["level"] * 100
             <= confluence_threshold) &
            (zones_df.index != i) &
            (~zones_df.index.isin(used_indices))
        ]

        if len(similar) >= 1:
            group_indices = [i] + similar.index.tolist()
            group         = zones_df.loc[group_indices]
            timeframes    = group["tf"].unique()

            if len(timeframes) >= 2:
                used_indices.update(group.index)

                group_sup = group[group["level"] < current_price]
                group_res = group[group["level"] >= current_price]
                is_mixed  = (not group_sup.empty) and (not group_res.empty)

                subgroups = (
                    [(group_sup, "Support"), (group_res, "Resistance")]
                    if is_mixed
                    else [(group, None)]
                )

                for subgroup, forced_type in subgroups:
                    if subgroup.empty:
                        continue

                    sub_tfs = subgroup["tf"].unique()
                    if is_mixed and len(sub_tfs) < 2:
                        continue

                    sub_avg   = subgroup["level"].mean()
                    sub_nb_tf = len(sub_tfs)
                    sub_dist  = abs(current_price - sub_avg) / current_price * 100

                    sub_active = subgroup[subgroup["status"] != "Consommee"]
                    if sub_active.empty:
                        sub_active = subgroup

                    sub_score = sum(
                        compute_structural_score(
                            int(r["strength"]), sub_nb_tf,
                            tf_name    = r["tf"],
                            age_bars   = int(r.get("age_bars", 0)),
                            total_bars = bars_map.get(r["tf"], 500),
                        )
                        for _, r in sub_active.iterrows()
                    )

                    sub_strength = int(subgroup["strength"].sum())

                    sub_status = max(
                        subgroup["status"].tolist(),
                        key=lambda s: STATUS_PRIORITY.get(s, 1)
                    )

                    is_pivot = sub_dist <= 0.50
                    if is_pivot:
                        sub_type   = "Pivot"
                        sub_signal = "↔ PIVOT ZONE"
                    elif forced_type:
                        sub_type   = forced_type
                        sub_signal = "🟢 BUY ZONE" if forced_type == "Support" else "🔴 SELL ZONE"
                    else:
                        n_sup = (subgroup["level"] < current_price).sum()
                        n_res = (subgroup["level"] >= current_price).sum()
                        if n_sup > n_res:
                            sub_type, sub_signal = "Support",    "🟢 BUY ZONE"
                        elif n_res > n_sup:
                            sub_type, sub_signal = "Resistance", "🔴 SELL ZONE"
                        else:
                            sub_type   = "Support" if sub_avg < current_price else "Resistance"
                            sub_signal = "🟢 BUY ZONE" if sub_type == "Support" else "🔴 SELL ZONE"

                    sub_tf_label = " + ".join(sorted(sub_tfs))
                    sub_alerte   = ("🔥 ZONE CHAUDE" if sub_dist < 0.5
                                    else ("⚠️ Proche" if sub_dist < 1.5 else ""))

                    confluences.append({
                        "Actif":        symbol,
                        "Signal":       sub_signal,
                        "Niveau":       round(sub_avg, 5),
                        "Type":         sub_type,
                        "Timeframes":   sub_tf_label,
                        "Nb TF":        sub_nb_tf,
                        "Force Totale": sub_strength,
                        "Score":        round(sub_score, 1),
                        "Statut":       sub_status,
                        "Distance %":   round(sub_dist, 3),
                        "Alerte":       sub_alerte,
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
    symbol:        str
    rows:          dict
    zones:         dict
    price:         Optional[float]
    trends:        dict
    anomaly:       Optional[str]
    bars_map:      dict = field(default_factory=dict)
    scan_error:    Optional[str] = None


def scan_single_symbol(args) -> ScanResult:
    symbol, base_url, access_token, account_id, zone_width, min_touches_ui = args
    profile = get_profile(symbol)

    pdf_dist_max    = PDF_DIST_THRESHOLDS.get(symbol, DEFAULT_PDF_DIST)
    abs_dist_max    = ABSOLUTE_MAX_DIST.get(symbol, 99.0)

    rows           = {"H4": None, "Daily": None, "Weekly": None}
    zones_d        = {}
    trends         = {}
    bars_map       = {}
    all_sup_levels = []
    last_h4_close  = None
    anomaly_msg    = None
    internal_error = None
    reference_price = None

    sym_d = _sym_display(symbol)

    _TF_KEYS = [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]

    def _fetch_tf(tf_key):
        return tf_key, get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as tf_pool:
            tf_futures = {tf_pool.submit(_fetch_tf, tk): tk for tk, _ in _TF_KEYS}
            tf_data = {}
            for fut in concurrent.futures.as_completed(tf_futures, timeout=30):
                try:
                    tf_key, df = fut.result()
                    tf_data[tf_key] = df
                except Exception:
                    tf_data[tf_futures[fut]] = None

        raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
        current_price, price_alert_msg = validate_live_price(
            raw_price, symbol, base_url, access_token
        )
        anomaly_msg = price_alert_msg

        reference_price = current_price

        for tf_key, tf_cap in _TF_KEYS:
            df = tf_data.get(tf_key)
            if df is None or df.empty:
                continue

            if reference_price is None:
                reference_price = float(df["close"].iloc[-1])

            if tf_key == "h4":
                last_h4_close = float(df["close"].iloc[-1])

            bars_map[tf_cap] = len(df)

            cp = reference_price

            trends[tf_cap] = compute_trend(df)
            atr_val = compute_atr(df, period=14)

            min_touches = {"h4": max(3, min_touches_ui)}.get(tf_key, max(2, min_touches_ui))

            supports, resistances = find_strong_sr_zones(
                df, cp,
                symbol      = symbol,
                atr_val     = atr_val,
                timeframe   = tf_key,
                min_touches = min_touches,
            )
            zones_d[tf_cap] = (supports, resistances)

            if not supports.empty:
                all_sup_levels.extend(supports["level"].tolist())

            if tf_key == "daily":
                price_ctx = get_price_context(cp, supports, resistances)
                zones_d["_price_ctx"] = price_ctx

            def make_row(zone, ztype, _cp=cp, _atr=atr_val,
                         _pdf_max=pdf_dist_max, _abs_max=abs_dist_max,
                         _tf=tf_cap, _ntot=len(df)):
                lvl      = zone["level"]
                strength = int(zone["strength"])
                age_bars = int(zone.get("age_bars", 0))
                status   = zone.get("status", "Testee")

                if not _cp or _cp <= 0:
                    dist_pct = 0.0
                    dist_atr_str = "N/A"
                    in_pdf = False
                else:
                    dist_pct = abs(_cp - lvl) / _cp * 100
                    if _atr and _atr > 0:
                        dist_atr_str = f"{round(abs(_cp - lvl) / _atr, 1):.1f}x"
                        in_pdf = dist_pct <= (profile.merge_threshold_atr * 200)
                        in_pdf = in_pdf or (dist_pct <= _pdf_max and dist_pct <= _abs_max)
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
                    "Score (1TF)": struct_score,
                    "Statut":      status,
                    "Dist. %":     f"{dist_pct:.2f}%",
                    "Dist. ATR":   dist_atr_str,
                    "_dist_num":   dist_pct,
                    "_in_pdf":     in_pdf,
                }

            tf_rows = (
                [make_row(z, "PIVOT" if z.get("is_pivot") else "Support")
                 for _, z in supports.iterrows()] +
                [make_row(z, "PIVOT" if z.get("is_pivot") else "Resistance")
                 for _, z in resistances.iterrows()]
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

        if all_sup_levels and reference_price:
            new_anomaly = flag_data_anomaly(
                symbol, reference_price, all_sup_levels, last_h4_close
            )
            if new_anomaly:
                anomaly_msg = (f"{anomaly_msg} | {new_anomaly}"
                               if anomaly_msg else new_anomaly)

    except concurrent.futures.TimeoutError:
        internal_error = "TimeoutError: données TF non reçues dans les 30s"
    except Exception as e:
        internal_error = f"{type(e).__name__}: {str(e)[:300]}"

    return ScanResult(
        symbol     = symbol,
        rows       = rows,
        zones      = zones_d,
        price      = reference_price,
        trends     = trends,
        anomaly    = anomaly_msg,
        bars_map   = bars_map,
        scan_error = internal_error,
    )


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATION PDF & EXPORTS
# ══════════════════════════════════════════════════════════════════
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
                      f"v{SCANNER_VERSION}  |  "
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
                    sig  = str(z.get('Signal', ''))
                    niv  = str(z.get('Niveau',     ''))
                    dist = str(z.get('Distance %', ''))
                    sc   = str(z.get('Score',      ''))
                    tfs  = str(z.get('Timeframes', ''))
                    ale  = str(z.get('Alerte',     ''))
                    txt  = _safe_pdf_str(
                        f"  {sig}  Niv:{niv}  Dist:{dist}  Score:{sc}  TF:{tfs}  {ale}"
                    )
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
                'Score (1TF)': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16,
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
    if df.empty:
        return df
    if "_in_pdf" in df.columns:
        df = df[df["_in_pdf"]].copy()
    elif "_dist_num" in df.columns:
        df = df[df["_dist_num"] <= DEFAULT_PDF_DIST].copy()
    elif "Dist. %" in df.columns:
        def _to_f(s):
            try:
                return float(str(s).replace("%", ""))
            except (ValueError, TypeError):
                return 999.0
        df = df[df["Dist. %"].apply(_to_f) <= DEFAULT_PDF_DIST].copy()
    return _clean_df(df).reset_index(drop=True)

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
        clean_conf = strip_emojis_df(_clean_df(confluences_df.copy()))
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
        clean_df = strip_emojis_df(_clean_df(df.copy()))
        if "Score (1TF)" in clean_df.columns:
            clean_df = clean_df.sort_values("Score (1TF)", ascending=False)
        pdf.chapter_body(clean_df)
        pdf.ln(10)

    return bytes(pdf.output())

def create_csv_report(results_dict, confluences_df=None):
    all_dfs = []
    if confluences_df is not None and not confluences_df.empty:
        c = _clean_df(confluences_df).copy()
        c["Section"] = "CONFLUENCES"
        all_dfs.append(c)
    for tf, df in results_dict.items():
        if df is not None and not df.empty:
            d = _clean_df(df).copy()
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
    ALERT_LABEL  = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠"}

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
            "niveau":  str(row.get("Niveau",     "")),
            "score":   score_val,
            "statut":  statut,
            "dist":    dist_val,
            "tfs":     str(row.get("Timeframes", "")),
            "nb_tf":   int(row.get("Nb TF", 0)),
            "alerte":  str(row.get("Alerte",     "")),
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
            sig = z["signal"]
            if "BUY"   in sig: signal_short = "BUY  "
            elif "SELL" in sig: signal_short = "SELL "
            else:               signal_short = "PIVOT"
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
        "1. Identifie les setups les plus immédiats (zones chaudes en priorité)",
        "2. Vérifie la cohérence tendance vs direction de zone (ex: BUY en tendance haussière)",
        "3. Priorise les zones Vierge (V) sur 3 TF avec Score > 200",
        "4. Les zones Role Reverse (RR) = pullback sur niveau cassé, setup souvent court terme",
        "5. Propose un plan de trade structuré : entrée, SL (au-delà de la zone), TP (prochain niveau)",
        "```",
    ]

    return "\n".join(lines).encode("utf-8")

def create_json_export(summaries, confluences_df,
                       max_dist=5.0, min_score=60.0,
                       allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    output = {
        "generated_at":     datetime.now().isoformat(),
        "scanner_version":  SCANNER_VERSION,
        "filters": {
            "max_dist_pct": max_dist,
            "min_score":    min_score,
        },
        "assets": [],
    }

    summary_map = {s["symbol"]: s for s in summaries}

    def _normalize_signal(raw: str) -> str:
        r = raw.replace("🟢", "").replace("🔴", "").replace("↔️", "").replace("↔", "")
        r = r.replace("ZONE", "").strip()
        if "PIVOT" in r: return "PIVOT"
        if "BUY"   in r: return "BUY"
        if "SELL"  in r: return "SELL"
        return r.strip()

    def _normalize_alert(raw: str) -> str:
        r = raw.replace("🔥", "").replace("⚠️", "").replace("⚠", "").strip()
        if "CHAUD" in r.upper() or "HOT" in r.upper():
            return "HOT"
        if "PROCHE" in r.upper() or "CLOSE" in r.upper():
            return "CLOSE"
        return ""

    def _parse_timeframes(tf_str: str) -> list:
        _order = {"Weekly": 0, "Daily": 1, "H4": 2}
        parts  = [p.strip() for p in tf_str.replace("+", ",").split(",") if p.strip()]
        return sorted(parts, key=lambda t: _order.get(t, 99))

    actif_groups: dict = {}

    if confluences_df is not None and not confluences_df.empty:
        for _, row in confluences_df.iterrows():
            try:
                dist_raw = row.get("Distance %", 999)
                dist_val = float(str(dist_raw).replace("%", ""))
            except Exception:
                dist_val = 999.0
            try:
                score_val = float(row.get("Score", 0))
                if not pd.notna(score_val):
                    score_val = 0.0
            except (TypeError, ValueError):
                score_val = 0.0

            if dist_val > max_dist or score_val < min_score:
                continue

            signal_raw = str(row.get("Signal", ""))
            signal_norm = _normalize_signal(signal_raw)
            if signal_norm not in ("BUY", "SELL", "PIVOT"):
                continue

            statut = str(row.get("Statut", ""))
            if statut not in allowed_statuts:
                continue

            actif = str(row.get("Actif", ""))
            if actif not in actif_groups:
                actif_groups[actif] = []

            niveau_raw = row.get("Niveau", "")
            try:
                level_float = round(float(niveau_raw), 5)
            except (TypeError, ValueError):
                level_float = None

            if level_float is None:
                continue

            tf_str    = str(row.get("Timeframes", ""))
            alert_raw = str(row.get("Alerte", ""))
            tfs_parsed = _parse_timeframes(tf_str)

            actif_groups[actif].append({
                "signal":        signal_norm,
                "type":          str(row.get("Type", "")),
                "level":         level_float,
                "score":         round(score_val, 1),
                "status":        statut,
                "distance_pct":  round(dist_val, 3),
                "alert":         _normalize_alert(alert_raw),
                "timeframes":    tfs_parsed,
                "nb_tf":         int(row.get("Nb TF", len(tfs_parsed))),
            })

    all_actifs = set(summary_map.keys())
    all_actifs.update(actif_groups.keys())

    sorted_actifs = sorted(
        all_actifs,
        key=lambda a: max(
            (z["score"] for z in actif_groups.get(a, [])),
            default=0.0
        ),
        reverse=True,
    )

    def _get_ict_session(dt) -> str:
        h = dt.hour
        if 22 <= h or h < 7:   return "ASIAN"
        if 7  <= h < 12:        return "LONDON"
        if 12 <= h < 16:        return "OVERLAP_LDN_NY"
        return "NEW_YORK"

    def _trend_alignment(h4: str, daily: str, weekly: str) -> tuple:
        bias_map = {"HAUSSIER": "BULLISH", "BAISSIER": "BEARISH", "NEUTRE": "NEUTRAL"}
        b_h4 = bias_map.get(h4,     "NEUTRAL")
        b_d  = bias_map.get(daily,  "NEUTRAL")
        b_w  = bias_map.get(weekly, "NEUTRAL")

        if b_d == b_w and b_d != "NEUTRAL":
            dominant  = b_d
            alignment = "ALIGNED" if b_h4 == dominant else "CONFLICTED"
        elif b_d == "NEUTRAL" and b_w != "NEUTRAL":
            dominant  = b_w
            alignment = "ALIGNED" if b_h4 == dominant else "CONFLICTED"
        elif b_w == "NEUTRAL" and b_d != "NEUTRAL":
            dominant  = b_d
            alignment = "ALIGNED" if b_h4 == dominant else "CONFLICTED"
        else:
            dominant  = "NEUTRAL"
            alignment = "MIXED"

        return alignment, dominant

    try:
        scan_dt = datetime.fromisoformat(output["generated_at"])
    except Exception:
        scan_dt = datetime.now()
    output["session"] = _get_ict_session(scan_dt)

    for actif in sorted_actifs:
        s     = summary_map.get(actif, {})

        zones = sorted(
            actif_groups.get(actif, []),
            key=lambda z: z["score"],
            reverse=True,
        )
        cp_val  = s.get("current_price")
        ctx_str = s.get("price_context", "")
        t_h4    = s.get("trend_h4",     "NEUTRE")
        t_d     = s.get("trend_daily",  "NEUTRE")
        t_w     = s.get("trend_weekly", "NEUTRE")

        alignment, dominant = _trend_alignment(t_h4, t_d, t_w)
        obstacles = _parse_price_context_obstacles(ctx_str, cp_val or 0)

        output["assets"].append({
            "symbol":               actif,
            "current_price":        round(cp_val, 5) if cp_val else None,
            "trends": {
                "h4":     t_h4,
                "daily":  t_d,
                "weekly": t_w,
            },
            "trend_alignment":      alignment,
            "dominant_bias":        dominant,
            "price_context":        ctx_str,
            "nearest_support":      obstacles["nearest_support"],
            "nearest_resistance":   obstacles["nearest_resistance"],
            "nb_zones":             len(zones),
            "zones":                zones,
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

    if not conf_full.empty:
        tmp = _clean_df(conf_full).copy()
        tmp["_dist_num"] = pd.to_numeric(
            tmp["Distance %"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        ).fillna(999.0)
        conf_filt = (tmp[tmp["_dist_num"] <= max_dist_filter]
                     .drop(columns=["_dist_num"], errors="ignore")
                     .reset_index(drop=True))
    else:
        conf_filt = pd.DataFrame()

    tf_cfg = {
        "Actif":       st.column_config.TextColumn("Actif",       width="small"),
        "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"),
        "Type":        st.column_config.TextColumn("Type",        width="small"),
        "Niveau":      st.column_config.TextColumn("Niveau",      width="small"),
        "Force":       st.column_config.TextColumn("Force",       width="medium"),
        "Score (1TF)": st.column_config.NumberColumn("Score (1TF) ▼", width="small"),
        "Statut":      st.column_config.TextColumn("Statut",      width="small"),
        "Dist. %":     st.column_config.TextColumn("Dist. %",     width="small"),
        "Dist. ATR":   st.column_config.TextColumn("Dist. ATR",   width="small"),
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
        st.caption(
            "Score confluence = Force × Nb TF × Poids_TF × Facteur_Age  |  "
            "Score (1TF) dans les tableaux ci-dessous = mono-timeframe brut"
        )

        disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total zones",       len(disp))
        c2.metric("🔥 Zones chaudes",  len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
        c3.metric("⚠️ Zones proches",  len(disp[disp["Alerte"] == "⚠️ Proche"]))
        c4.metric("🟢 BUY Zones",      len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
        c5.metric("🔴 SELL Zones",     len(disp[disp["Signal"] == "🔴 SELL ZONE"]))
        c6.metric("↔ PIVOT Zones",     len(disp[disp["Signal"] == "↔ PIVOT ZONE"]))

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
        st.caption("Paramètres LLM configurables dans la barre latérale (section 3).")

        llm_max_dist    = st.session_state.get("llm_max_dist",  2.0)
        llm_min_score   = st.session_state.get("llm_min_score", 100)
        llm_statuts_raw = st.session_state.get("llm_statuts", ["Vierge", "Testee", "Role Reverse"])
        llm_statuts     = tuple(llm_statuts_raw) if llm_statuts_raw else ("Vierge", "Testee", "Role Reverse")

        st.caption(
            f"🔧 Filtres actifs : Dist < **{llm_max_dist}%** | "
            f"Score ≥ **{llm_min_score}** | {', '.join(llm_statuts)}"
        )

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
                max_dist        = llm_max_dist,
                min_score       = float(llm_min_score),
                allowed_statuts = tuple(llm_statuts),
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
            n_zones  = sum(1 for line in brief_preview.split("\n")
                           if line.strip().startswith("- BUY") or line.strip().startswith("- SELL"))
            n_actifs = brief_preview.count("### ")
            st.info(
                f"**{n_actifs} actifs** avec **{n_zones} zones** dans le brief LLM "
                f"(≈ {n_zones * 15 + n_actifs * 10:,} tokens estimés)"
            )
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
            try:
                return float(str(s).replace("%", ""))
            except (ValueError, TypeError):
                return 999.0
        mask = df["Dist. %"].apply(to_float) <= max_pct
        out  = _clean_df(df[mask])
        sort_col = "Score (1TF)" if "Score (1TF)" in out.columns else "Score"
        if sort_col in out.columns:
            out = out.sort_values(sort_col, ascending=False)
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
        if not access_token or not str(access_token).strip():
            raise ValueError("OANDA_ACCESS_TOKEN est vide")
        if not account_id or not str(account_id).strip():
            raise ValueError("OANDA_ACCOUNT_ID est vide")
        st.success("Secrets chargés ✓")
    except (KeyError, ValueError) as e:
        access_token, account_id = None, None
        st.error(f"Secrets OANDA invalides : {e}")
    except Exception as e:
        access_token, account_id = None, None
        st.error(f"Erreur lecture secrets : {type(e).__name__}")

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
    )
    llm_min_score_sidebar = st.slider(
        "Score min JSON/LLM", 40, 300, 60, 10,
        key="llm_min_score",
    )
    llm_statuts_sidebar = st.multiselect(
        "Statuts autorisés (brief LLM)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    st.divider()
    st.header("4. Paramètres de Détection")
    zone_width = st.slider(
        "Largeur zone Forex (% fallback si ATR indispo)", 0.1, 2.0, 0.5, 0.1,
    )
    min_touches = st.slider(
        "Force minimale (touches)", 3, 10, 3, 1,
    )
    confluence_threshold = st.slider(
        "Seuil confluence Forex (%)", 0.3, 2.0, 1.0, 0.1,
    )
    _overridden = [s.replace("_USD","").replace("_EUR","")
                   for s in CONFLUENCE_THRESHOLD_MAP]
    st.caption(
        f"⚠️ Seuil ignoré pour : {', '.join(_overridden)} "
        f"(valeurs fixes : {list(CONFLUENCE_THRESHOLD_MAP.values())})"
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) - filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
    )

    st.divider()
    st.caption("**Score confluence = (Force × Poids_TF × NbTF) × Facteur_Age**")
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
    st.caption(f"**v{SCANNER_VERSION} — Audit 3e passe : 86 corrections (→ MAJ-048)**")
    st.caption("— v5.11 (9) — v5.12 (8) — v5.13 (13 : couverture + 3 bugs v5.12 corrigés) —")
    st.caption("✅ CRIT-001 — Backslash SyntaxError (introduit v5.12)")
    st.caption("✅ MAJ-034/035 — Shadowing _get_session + code mort _trend_alignment")
    st.caption("✅ MAJ-005 — .tail(limit) : bougies récentes conservées (pas les anciennes)")
    st.caption("✅ MAJ-012 — min_periods=n côté droit : pivots non-confirmés éliminés")
    st.caption("✅ MAJ-048 — min_touches adaptatif : H4=3, Daily/Weekly=2")
    st.caption("✅ MAJ-019 — used_indices filtré dans similar : fin du double-comptage")
    st.caption("✅ MAJ-024 — PIVOT ≠ SELL dans create_llm_brief")
    st.caption("✅ MAJ-025/026 — JSON statuts + llm_max_dist cohérents avec Markdown")


# ══════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE
# ══════════════════════════════════════════════════════════════════

# Déclenchement : premier clic sur le bouton
if scan_button and symbols_to_scan and not st.session_state.get("scanning", False):
    st.session_state.pop("scan_results", None)
    st.session_state["scanning"]    = True
    st.session_state["pending_scan"] = True
    st.rerun()

# Exécution du scan (render suivant, scanning=True, pending_scan=True)
if st.session_state.get("pending_scan", False) and symbols_to_scan:
    st.session_state.pop("pending_scan", None)

    if not access_token or not account_id:
        st.session_state["scanning"] = False
        st.warning("Configurez vos secrets OANDA avant de lancer le scan.")
    else:
        base_url = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.session_state["scanning"] = False
            st.error("Identifiants OANDA invalides. Vérifiez vos secrets.")
        else:
            progress_bar = st.progress(0, text="Initialisation du scan…")

            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map   = {}
            prices_map      = {}
            trends_map      = {}
            anomalies_map   = {}
            scan_errors     = {}
            bars_map_global = {}

            args_list = [
                (sym, base_url, access_token, account_id, zone_width, min_touches)
                for sym in symbols_to_scan
            ]
            total, completed = len(args_list), 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_map = {
                    executor.submit(scan_single_symbol, a): a[0] for a in args_list
                }
                try:
                    for future in concurrent.futures.as_completed(future_map, timeout=120):
                        sym = future_map[future]
                        completed += 1
                        progress_bar.progress(
                            completed / total,
                            text=f"Scan… ({completed}/{total}) {sym.replace('_', '/')}",
                        )
                        try:
                            result: ScanResult = future.result()

                            all_zones_map[result.symbol]   = result.zones
                            prices_map[result.symbol]      = result.price
                            trends_map[result.symbol]      = result.trends
                            bars_map_global[result.symbol] = result.bars_map

                            if result.anomaly:
                                anomalies_map[_sym_display(result.symbol)] = result.anomaly

                            if result.scan_error:
                                scan_errors[_sym_display(result.symbol)] = (
                                    f"[interne] {result.scan_error}"
                                )

                            for tf_cap, tf_rows in result.rows.items():
                                if tf_rows:
                                    if tf_cap == "H4":
                                        results_h4.extend(tf_rows)
                                    elif tf_cap == "Daily":
                                        results_daily.extend(tf_rows)
                                    elif tf_cap == "Weekly":
                                        results_weekly.extend(tf_rows)

                        except (requests.RequestException, ValueError,
                                KeyError, pd.errors.EmptyDataError) as e:
                            scan_errors[_sym_display(sym)] = (
                                f"{type(e).__name__}: {str(e)[:200]}"
                            )
                        except Exception as e:
                            tb = traceback.format_exc()
                            tb = _sanitize_traceback(tb, [access_token, account_id, base_url])
                            scan_errors[_sym_display(sym)] = (
                                f"INATTENDU {type(e).__name__}: {tb[-400:]}"
                            )
                except concurrent.futures.TimeoutError:
                    completed_raw = set(all_zones_map.keys())
                    remaining = set(future_map.values()) - completed_raw
                    for sym in remaining:
                        scan_errors[_sym_display(sym)] = "Timeout global scan (120s)"

            progress_bar.empty()
            st.session_state["scanning"] = False

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
                if _sym_display(sym) in scan_errors:
                    continue
                cp = prices_map.get(sym)
                zones_clean = {k: v for k, v in all_zones_map.get(sym, {}).items()
                               if not k.startswith("_")}
                sym_threshold = CONFLUENCE_THRESHOLD_MAP.get(sym, confluence_threshold)
                confs = detect_confluences(
                    _sym_display(sym),
                    zones_clean,
                    cp,
                    sym_threshold,
                    bars_map=bars_map_global.get(sym, {}),
                )
                all_confluences.extend(confs)

            conf_full = pd.DataFrame(all_confluences)
            if not conf_full.empty:
                conf_full = _clean_df(conf_full)

            summaries = []
            for sym in symbols_to_scan:
                sym_d  = _sym_display(sym)
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
                    "current_price": cp,
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
                "conf_full":   conf_full,
                "report_dict": rep_dict,
                "summaries":   summaries,
                "anomalies":   anomalies_map,
                "scan_errors": scan_errors,
                "max_dist":    max_dist_filter,
            }

            _display_results(st.session_state["scan_results"], max_dist_filter)

elif not symbols_to_scan and not st.session_state.get("scanning", False):
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
elif not st.session_state.get("pending_scan", False) and not st.session_state.get("scanning", False):
    st.info(
        "Configurez les paramètres dans la barre latérale, "
        "puis cliquez sur **LANCER LE SCAN COMPLET**."
    )

if "scan_results" in st.session_state and not st.session_state.get("pending_scan", False):
    _display_results(
        st.session_state["scan_results"],
        max_dist_filter,
    )
