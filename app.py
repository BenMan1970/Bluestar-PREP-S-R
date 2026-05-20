import hashlib
import json
import threading
import traceback
import concurrent.futures
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════════════════
# SESSION HTTP (thread-safe)
# ══════════════════════════════════════════════════════════════════
_thread_local = threading.local()

def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4))
        _thread_local.session = s
    return _thread_local.session

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS (inchangé)
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
        background-color: #B71C1C; border-color: #D32F2F; box-shadow: 0 4px 12px rgba(211, 47, 47, 0.4);
    }
    div[data-testid="stButton"] > button[kind="primary"]:active { background-color: #D32F2F; transform: scale(0.98); }
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
# CONSTANTES & PROFIL D'INSTRUMENT (nouvelle architecture)
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
    # Seuils de validation des prix
    price_sanity_range: Optional[Tuple[float, float]] = None
    dev_threshold_pct: float = 1.5
    skip_ratio_check: bool = False
    pip_floor_abs: float = 0.0  # plancher de tolérance en unités de prix

# Base de profils calibrés (tous les seuils en multiples d'ATR)
_PROFILES = {
    "EUR_USD": InstrumentProfile("EUR_USD", 0.0001, 0.00001, "LOW", 0.2, 0.3, 1.2, 0.8, 0.6,
                                 pip_floor_abs=0.00010),
    "GBP_USD": InstrumentProfile("GBP_USD", 0.0001, 0.00001, "MEDIUM", 0.4, 0.4, 1.3, 0.85, 0.65,
                                 pip_floor_abs=0.00010),
    "USD_JPY": InstrumentProfile("USD_JPY", 0.01, 0.001, "LOW", 0.3, 0.25, 0.9, 0.5, 0.5,
                                 pip_floor_abs=0.10),
    "XAU_USD": InstrumentProfile("XAU_USD", 0.01, 0.01, "HIGH", 20.0, 0.6, 2.0, 1.2, 1.0,
                                 price_sanity_range=(1500.0, 6500.0), dev_threshold_pct=3.0, skip_ratio_check=True),
    "US30_USD": InstrumentProfile("US30_USD", 1.0, 1.0, "MEDIUM", 2.0, 0.3, 1.5, 0.9, 0.7,
                                  price_sanity_range=(20000.0, 70000.0), dev_threshold_pct=2.5, skip_ratio_check=True),
    "NAS100_USD": InstrumentProfile("NAS100_USD", 1.0, 1.0, "HIGH", 3.0, 0.35, 1.5, 1.0, 0.8,
                                    price_sanity_range=(8000.0, 35000.0), dev_threshold_pct=2.5, skip_ratio_check=True),
    "SPX500_USD": InstrumentProfile("SPX500_USD", 0.1, 0.1, "MEDIUM", 0.5, 0.3, 1.3, 0.8, 0.65,
                                    price_sanity_range=(3000.0, 9000.0), dev_threshold_pct=2.0, skip_ratio_check=True),
    "DE30_EUR": InstrumentProfile("DE30_EUR", 0.1, 0.1, "MEDIUM", 0.5, 0.3, 1.3, 0.8, 0.65,
                                  price_sanity_range=(8000.0, 30000.0), dev_threshold_pct=2.0, skip_ratio_check=True),
}

# Profil générique pour les symboles non listés
_DEFAULT_PROFILE = InstrumentProfile("DEFAULT", 0.0001, 0.00001, "MEDIUM", 0.5, 0.4, 1.2, 0.8, 0.6,
                                     pip_floor_abs=0.00010)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    # Héritage automatique pour les croisés
    base = symbol.split("_")[0]
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF"):
        return InstrumentProfile(symbol, 0.0001, 0.00001, "MEDIUM", 0.5, 0.4, 1.2, 0.8, 0.6,
                                 pip_floor_abs=0.00010)
    if base == "JPY" or symbol.endswith("_JPY"):
        return InstrumentProfile(symbol, 0.01, 0.001, "LOW", 0.3, 0.25, 0.9, 0.5, 0.5,
                                 pip_floor_abs=0.10)
    return _DEFAULT_PROFILE

# ══════════════════════════════════════════════════════════════════
# CACHE THREAD-SAFE (inchangé)
# ══════════════════════════════════════════════════════════════════
class _BoundedTTLCache:
    def __init__(self, maxsize: int, ttl: int):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
    def get(self, key):
        entry = self._cache.get(key)
        if entry is None: return None
        val, ts = entry
        if (datetime.now() - ts).total_seconds() > self._ttl:
            del self._cache[key]; return None
        self._cache.move_to_end(key); return val
    def set(self, key, val):
        if key in self._cache: self._cache.move_to_end(key)
        self._cache[key] = (val, datetime.now())
        while len(self._cache) > self._maxsize: self._cache.popitem(last=False)

_oanda_cache = _BoundedTTLCache(maxsize=300, ttl=600)
_price_cache = _BoundedTTLCache(maxsize=100, ttl=60)
_cache_lock = threading.RLock()

def _token_fingerprint(token: str) -> str: return hashlib.sha256(token.encode()).hexdigest()[:12]
def _cache_key(token, symbol, timeframe, limit): return f"{_token_fingerprint(token)}__{symbol}__{timeframe}__{limit}"
def _cache_get(key): 
    with _cache_lock:
        df = _oanda_cache.get(key)
        return df.copy() if df is not None else None
def _cache_set(key, df):
    with _cache_lock: _oanda_cache.set(key, df)

# ══════════════════════════════════════════════════════════════════
# API OANDA (inchangée)
# ══════════════════════════════════════════════════════════════════
_api_semaphore = threading.BoundedSemaphore(5)
_GRANULARITY_MAP = {"h4": "H4", "daily": "D", "weekly": "W"}

def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
        try:
            with _api_semaphore:
                r = _get_session().get(f"{url}/v3/accounts/{account_id}/summary", headers=headers, timeout=(3,5))
            if r.status_code == 200: return url
        except: continue
    return None

def get_oanda_data(base_url, access_token, symbol, timeframe="daily", limit=500):
    key = _cache_key(access_token, symbol, timeframe, limit)
    cached = _cache_get(key)
    if cached is not None: return cached
    gran = _GRANULARITY_MAP.get(timeframe)
    if gran is None: raise ValueError(f"Timeframe inconnu: {timeframe!r}")
    url = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"count": limit+1, "granularity": gran, "price": "M"}
    try:
        with _api_semaphore:
            r = _get_session().get(url, headers=headers, params=params, timeout=(3,8))
            r.raise_for_status()
            data = r.json()
        if not data.get("candles"): return None
        candles = [{"date": pd.to_datetime(c["time"]), "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]), "volume": int(c["volume"])}
                   for c in data["candles"] if c.get("complete")]
        df = pd.DataFrame(candles).tail(limit).set_index("date")
        if df.empty: return None
        _cache_set(key, df)
        return df
    except: return None

def get_oanda_current_price(base_url, access_token, account_id, symbol):
    key = f"{_token_fingerprint(access_token)}__{symbol}__price"
    with _cache_lock:
        val = _price_cache.get(key)
        if val is not None: return val
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        with _api_semaphore:
            r = _get_session().get(url, headers=headers, params={"instruments": symbol}, timeout=(3,5))
            r.raise_for_status()
            data = r.json()
        if "prices" in data and data["prices"]:
            bid, ask = float(data["prices"][0]["closeoutBid"]), float(data["prices"][0]["closeoutAsk"])
            result = (bid + ask)/2
        else: result = None
    except: result = None
    if result is not None:
        with _cache_lock: _price_cache.set(key, result)
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
            if not (lo <= h4_close <= hi): h4_close = None
    if live_price is not None:
        if profile.price_sanity_range:
            lo, hi = profile.price_sanity_range
            if not (lo <= live_price <= hi): live_price = None
        if live_price is not None and h4_close is not None and h4_close > 0:
            dev = abs(live_price - h4_close)/h4_close*100
            if dev > dev_threshold: live_price = None
    if live_price is not None: return live_price, None
    if h4_close is not None: return h4_close, None
    return None, f"Aucun prix fiable disponible pour {symbol}"

# ══════════════════════════════════════════════════════════════════
# UTILITAIRES (inchangés)
# ══════════════════════════════════════════════════════════════════
_ACCENT_MAP = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
                           'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
_EMOJI_MAP = [('🟢','[BUY]'),('🔴','[SELL]'),('🔥','[CHAUD]'),('↔️','[PIVOT]'),('↔','[PIVOT]'),
              ('⚠️','[PROCHE]'),('⚠','[PROCHE]'),('📈',''),('📉',''),('✅','[OK]'),('❌','[X]'),
              ('⚡','[!]'),('📡',''),('📅',''),('↩️','[RR]'),('↑','[HAUSSE]'),('↓','[BAISSE]'),('→','[NEUTRE]')]
def _safe_pdf_str(text):
    if not isinstance(text, str): text = str(text)
    text = text.translate(_ACCENT_MAP)
    for emoji, repl in _EMOJI_MAP: text = text.replace(emoji, repl)
    return text
def _sanitize_traceback(tb, sensitive_values):
    for val in sensitive_values:
        if val and isinstance(val, str) and len(val)>4: tb = tb.replace(val, f"***{val[-4:]}")
    return tb
_INTERNAL_COLS = ["_dist_num", "_in_pdf"]
def _clean_df(df): return df.drop(columns=_INTERNAL_COLS, errors="ignore")
def _sym_display(sym): return sym.replace("_", "/")

# ══════════════════════════════════════════════════════════════════
# NOUVEAU MOTEUR DE DÉTECTION CALIBRÉ
# ══════════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    if df is None or len(df) < period+1: return None
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift(1)).abs(),
                    (df["low"]-df["close"].shift(1)).abs()], axis=1).max(axis=1)
    result = tr.rolling(period).mean().iloc[-1]
    return float(result) if pd.notna(result) and result > 0 else None

def compute_trend(df, sma_period=20):
    if df is None or len(df) < 15: return "NEUTRE"
    actual_period = min(sma_period, len(df)-5)
    if actual_period < 10: return "NEUTRE"
    close = df["close"]
    sma = close.rolling(actual_period).mean().iloc[-1]
    current = close.iloc[-1]
    base = close.iloc[-6]
    if not np.isfinite(base) or base <= 0: return "NEUTRE"
    slope_pct = (close.iloc[-1] - base)/base*100
    n = min(10, len(df))
    highs, lows = df["high"].iloc[-n:], df["low"].iloc[-n:]
    hh, ll = highs.iloc[-1] > highs.iloc[0], lows.iloc[-1] < lows.iloc[0]
    lh, hl = highs.iloc[-1] < highs.iloc[0], lows.iloc[-1] > lows.iloc[0]
    above_sma, below_sma = current > sma, current < sma
    if above_sma and slope_pct > 0.05 and hh and hl: return "HAUSSIER"
    if below_sma and slope_pct < -0.05 and lh and ll: return "BAISSIER"
    if above_sma and slope_pct > 0.05: return "HAUSSIER"
    if below_sma and slope_pct < -0.05: return "BAISSIER"
    if above_sma and hh: return "HAUSSIER"
    if below_sma and ll: return "BAISSIER"
    return "NEUTRE"

TF_WEIGHT = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}

def compute_structural_score(strength, nb_tf, tf_name="H4", age_bars=0, total_bars=500):
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(int(age_bars),0)/max(int(total_bars),1)
    age_f = float(np.exp(-1.5*age_r))
    raw = int(strength)*tf_w*nb_tf
    return round(raw*age_f, 1)

def detect_swing_pivots(df, profile, atr_val, timeframe):
    """Détection de pivots avec validation par rejet (mèche/ATR)."""
    n = 3
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    opens = df["open"].values
    roll_high_left = pd.Series(highs).shift(1).rolling(n, min_periods=n).max().values
    roll_high_right = pd.Series(highs[::-1]).shift(1).rolling(n, min_periods=n).max()[::-1].values
    roll_low_left = pd.Series(lows).shift(1).rolling(n, min_periods=n).min().values
    roll_low_right = pd.Series(lows[::-1]).shift(1).rolling(n, min_periods=n).min()[::-1].values
    next_close = pd.Series(closes).shift(-1).fillna(closes).values
    candle_range = (highs - lows).clip(1e-10)
    upper_wick_pct = (highs - np.maximum(opens, closes)) / candle_range
    lower_wick_pct = (np.minimum(opens, closes) - lows) / candle_range
    wick_threshold = 0.20 if timeframe in ("h4","m15","m30") else 0.30
    sh_mask = (highs > roll_high_left) & (highs > roll_high_right) & (next_close < highs) & (upper_wick_pct >= wick_threshold)
    sl_mask = (lows < roll_low_left) & (lows < roll_low_right) & (next_close > lows) & (lower_wick_pct >= wick_threshold)
    if atr_val and atr_val > 0:
        min_amp = atr_val * profile.pivot_prominence_atr
        roll_low_around = pd.Series(lows).rolling(2*n+1, center=True, min_periods=1).min().values
        roll_high_around = pd.Series(highs).rolling(2*n+1, center=True, min_periods=1).max().values
        sh_mask &= (highs - roll_low_around) >= min_amp
        sl_mask &= (roll_high_around - lows) >= min_amp
    sh_idx = np.where(sh_mask)[0]
    sl_idx = np.where(sl_mask)[0]
    pivot_highs = pd.Series(highs[sh_idx], index=sh_idx) if len(sh_idx) else pd.Series(dtype=float)
    pivot_lows  = pd.Series(lows[sl_idx],  index=sl_idx)  if len(sl_idx) else pd.Series(dtype=float)
    return pivot_highs, pivot_lows

def cluster_zones(pivots, atr_val, profile):
    """Regroupement basé sur un rayon proportionnel à l'ATR."""
    if not pivots: return []
    radius = atr_val * profile.cluster_radius_atr if atr_val and atr_val>0 else 0.001
    pivots_sorted = sorted(pivots, key=lambda x: (x[0], x[1]))
    clusters = []
    current_cluster = [pivots_sorted[0]]
    for price, idx in pivots_sorted[1:]:
        if abs(price - current_cluster[0][0]) < radius:
            current_cluster.append((price, idx))
        else:
            clusters.append(current_cluster)
            current_cluster = [(price, idx)]
    clusters.append(current_cluster)
    return clusters

def classify_zone_status(level, zone_type, df, formation_idx, atr_val, profile):
    """Détermine Vierge/Testée/Consommée/Role Reverse avec tolérance adaptative."""
    if formation_idx >= len(df)-1: return "Vierge"
    if atr_val and atr_val > 0:
        tolerance = atr_val * 0.25
    else:
        tolerance = max(level * 0.001, profile.pip_floor_abs)
    c_arr = df["close"].values[formation_idx+1:]
    h_arr = df["high"].values[formation_idx+1:]
    l_arr = df["low"].values[formation_idx+1:]
    if len(c_arr) == 0: return "Vierge"
    near = (np.abs(c_arr - level) <= tolerance) | ((l_arr <= level+tolerance) & (h_arr >= level-tolerance))
    has_approach = bool(near.any())
    if zone_type == "Support":
        break_mask = c_arr < level - tolerance
    else:
        break_mask = c_arr > level + tolerance
    break_positions = np.where(break_mask)[0]
    if len(break_positions) == 0:
        return "Testee" if has_approach else "Vierge"
    break_idx = break_positions[0]
    retest_tol = tolerance*2
    rc = c_arr[break_idx+1:]
    rh = h_arr[break_idx+1:]
    rl = l_arr[break_idx+1:]
    if len(rc) == 0: return "Consommee"
    retest_mask = (rl <= level+retest_tol) & (rh >= level-retest_tol)
    if not retest_mask.any(): return "Consommee"
    retest_idx = np.where(retest_mask)[0][0]
    rc_after = rc[retest_idx+1:]
    if len(rc_after) == 0: return "Role Reverse"
    if zone_type == "Support":
        second_break = rc_after > level + tolerance
    else:
        second_break = rc_after < level - tolerance
    return "Consommee" if second_break.any() else "Role Reverse"

def find_strong_sr_zones(df, current_price, symbol, atr_val, timeframe, min_touches=2):
    """Détection et clustering calibrés par l'instrument."""
    profile = get_profile(symbol)
    if df is None or df.empty or len(df) < 20: return pd.DataFrame(), pd.DataFrame()
    if current_price is None: current_price = df["close"].iloc[-1]
    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)
    if len(pivot_highs)+len(pivot_lows) < 3:
        # fallback avec find_peaks (distance adaptative)
        distance = {"h4":5,"daily":8,"weekly":10}.get(timeframe,5)
        pk = {"distance": distance, "prominence": atr_val*profile.pivot_prominence_atr if atr_val else None}
        if atr_val is None: pk = {"distance": distance}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        pivot_highs = pd.Series(df["high"].values[r_idx], index=r_idx) if len(r_idx) else pd.Series(dtype=float)
        pivot_lows  = pd.Series(df["low"].values[s_idx],  index=s_idx)  if len(s_idx) else pd.Series(dtype=float)

    pivots_with_idx = [(float(p), int(i)) for i,p in pivot_highs.items()] + \
                      [(float(p), int(i)) for i,p in pivot_lows.items()]
    pivots_with_idx.sort(key=lambda x: (x[0], x[1]))
    if not pivots_with_idx: return pd.DataFrame(), pd.DataFrame()

    if atr_val and atr_val>0:
        zone_width_abs = atr_val * profile.cluster_radius_atr
    else:
        zone_width_abs = current_price * 0.005  # fallback 0.5% du prix

    clusters = cluster_zones(pivots_with_idx, atr_val, profile)
    strong = []
    for grp in clusters:
        if len(grp) < min_touches: continue
        prices = [p for p,_ in grp]
        indices = [i for _,i in grp]
        lvl = float(np.mean(prices))
        if lvl <= 0: continue
        strength = len(grp)
        last_idx = max(indices)
        age_bars = max(0, n_total-1-last_idx)
        zone_type_tmp = "Support" if lvl < current_price else "Resistance"
        status = classify_zone_status(lvl, zone_type_tmp, df, last_idx, atr_val, profile)
        strong.append({"level": lvl, "strength": strength, "age_bars": age_bars, "status": status})

    if not strong: return pd.DataFrame(), pd.DataFrame()

    # Post-merge basé ATR
    df_zones = pd.DataFrame(strong).sort_values("level").reset_index(drop=True)
    if atr_val and atr_val>0:
        merge_radius = atr_val * profile.merge_threshold_atr
        # on fusionne les zones trop proches (en prix)
        i = 0
        while i < len(df_zones)-1:
            if abs(df_zones.at[i+1,"level"] - df_zones.at[i,"level"]) < merge_radius:
                # fusionner
                avg_level = (df_zones.at[i,"level"]*df_zones.at[i,"strength"] + df_zones.at[i+1,"level"]*df_zones.at[i+1,"strength"]) / (df_zones.at[i,"strength"]+df_zones.at[i+1,"strength"])
                new_strength = df_zones.at[i,"strength"] + df_zones.at[i+1,"strength"]
                new_age = min(df_zones.at[i,"age_bars"], df_zones.at[i+1,"age_bars"])
                # statut: max selon priorité
                prior = {"Vierge":0, "Testee":1, "Role Reverse":1, "Consommee":2}
                new_status = max(df_zones.at[i,"status"], df_zones.at[i+1,"status"], key=lambda s: prior.get(s,1))
                df_zones.at[i, "level"] = avg_level
                df_zones.at[i, "strength"] = new_strength
                df_zones.at[i, "age_bars"] = new_age
                df_zones.at[i, "status"] = new_status
                df_zones.drop(i+1, inplace=True)
                df_zones.reset_index(drop=True, inplace=True)
            else:
                i += 1

    PIVOT_BAND = 0.50
    if current_price and current_price > 0:
        df_zones["is_pivot"] = abs(df_zones["level"] - current_price)/current_price*100 <= PIVOT_BAND
    else:
        df_zones["is_pivot"] = False

    supports = df_zones[df_zones["level"] < current_price].copy() if current_price else pd.DataFrame()
    resistances = df_zones[df_zones["level"] >= current_price].copy() if current_price else pd.DataFrame()
    return supports, resistances

def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0, bars_map=None):
    """Détection de confluences multi-TF avec scoring amélioré."""
    bars_map = bars_map or {}
    if not zones_dict or current_price is None: return []

    STATUS_PRIORITY = {"Vierge":0, "Testee":1, "Role Reverse":1, "Consommee":2}
    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, z in supports.iterrows():
            all_zones.append({"tf":tf, "level":z["level"], "strength":z["strength"],
                              "age_bars":z.get("age_bars",0), "status":z.get("status","Testee"),
                              "type":"Support"})
        for _, z in resistances.iterrows():
            all_zones.append({"tf":tf, "level":z["level"], "strength":z["strength"],
                              "age_bars":z.get("age_bars",0), "status":z.get("status","Testee"),
                              "type":"Resistance"})
    all_zones = [z for z in all_zones if z.get("status") != "Consommee"]
    if not all_zones: return []

    zones_df = pd.DataFrame(all_zones).sort_values("level").reset_index(drop=True)
    used_indices = set()
    confluences = []

    for i, zone in zones_df.iterrows():
        if i in used_indices: continue
        if zone["level"] <= 0: continue
        similar = zones_df[(abs(zones_df["level"]-zone["level"])/zone["level"]*100 <= confluence_threshold) &
                           (zones_df.index != i) & (~zones_df.index.isin(used_indices))]
        if len(similar) >= 1:
            group_indices = [i] + similar.index.tolist()
            group = zones_df.loc[group_indices]
            timeframes = group["tf"].unique()
            if len(timeframes) >= 2:
                used_indices.update(group.index)
                # ... (logique de construction des confluences inchangée, mais on pourrait l'enrichir)
                # Pour rester compatible avec l'affichage, on conserve le format existant.
                # Le calcul du score utilise maintenant le nouveau compute_structural_score avec nb_tf basé sur len(unique tfs)
                sub_tfs = timeframes
                sub_avg = group["level"].mean()
                sub_nb_tf = len(sub_tfs)
                sub_dist = abs(current_price - sub_avg)/current_price*100
                sub_active = group[group["status"] != "Consommee"]
                if sub_active.empty: sub_active = group
                sub_score = sum(compute_structural_score(int(r["strength"]), sub_nb_tf, r["tf"],
                                                         int(r.get("age_bars",0)), bars_map.get(r["tf"],500))
                                for _, r in sub_active.iterrows())
                sub_strength = int(group["strength"].sum())
                sub_status = max(group["status"].tolist(), key=lambda s: STATUS_PRIORITY.get(s,1))
                is_pivot = sub_dist <= 0.50
                if is_pivot:
                    sub_type, sub_signal = "Pivot", "↔ PIVOT ZONE"
                else:
                    n_sup = (group["level"] < current_price).sum()
                    n_res = (group["level"] >= current_price).sum()
                    if n_sup > n_res:
                        sub_type, sub_signal = "Support", "🟢 BUY ZONE"
                    elif n_res > n_sup:
                        sub_type, sub_signal = "Resistance", "🔴 SELL ZONE"
                    else:
                        sub_type = "Support" if sub_avg < current_price else "Resistance"
                        sub_signal = "🟢 BUY ZONE" if sub_type=="Support" else "🔴 SELL ZONE"
                sub_alerte = "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else "")
                confluences.append({
                    "Actif": symbol, "Signal": sub_signal, "Niveau": round(sub_avg,5),
                    "Type": sub_type, "Timeframes": " + ".join(sorted(sub_tfs)), "Nb TF": sub_nb_tf,
                    "Force Totale": sub_strength, "Score": round(sub_score,1), "Statut": sub_status,
                    "Distance %": round(sub_dist,3), "Alerte": sub_alerte,
                })
            else:
                used_indices.add(i)
        else:
            used_indices.add(i)
    return confluences

# ══════════════════════════════════════════════════════════════════
# SCAN RESULT & SCAN LOGIC (inchangé mais adapté)
# ══════════════════════════════════════════════════════════════════
@dataclass
class ScanResult:
    symbol: str; rows: dict; zones: dict; price: Optional[float]
    trends: dict; anomaly: Optional[str]; bars_map: dict = field(default_factory=dict)
    scan_error: Optional[str] = None

def scan_single_symbol(args):
    symbol, base_url, access_token, account_id, zone_width, min_touches = args
    profile = get_profile(symbol)
    rows = {"H4": None, "Daily": None, "Weekly": None}
    zones_d, trends, bars_map = {}, {}, {}
    all_sup_levels, last_h4_close, anomaly_msg, internal_error = [], None, None, None
    reference_price = None
    sym_d = _sym_display(symbol)
    _TF_KEYS = [("h4","H4"), ("daily","Daily"), ("weekly","Weekly")]

    def _fetch_tf(tf_key):
        return tf_key, get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as tf_pool:
            tf_futures = {tf_pool.submit(_fetch_tf, tk): tk for tk,_ in _TF_KEYS}
            tf_data = {}
            for fut in concurrent.futures.as_completed(tf_futures, timeout=30):
                try: tf_key, df = fut.result(); tf_data[tf_key] = df
                except: tf_data[tf_futures[fut]] = None

        raw_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
        current_price, price_alert = validate_live_price(raw_price, symbol, base_url, access_token)
        anomaly_msg = price_alert
        reference_price = current_price

        for tf_key, tf_cap in _TF_KEYS:
            df = tf_data.get(tf_key)
            if df is None or df.empty: continue
            if reference_price is None: reference_price = float(df["close"].iloc[-1])
            if tf_key == "h4": last_h4_close = float(df["close"].iloc[-1])
            bars_map[tf_cap] = len(df)
            cp = reference_price
            trends[tf_cap] = compute_trend(df)
            atr_val = compute_atr(df)
            # Nouvel appel avec le profil
            supports, resistances = find_strong_sr_zones(df, cp, symbol, atr_val, tf_key,
                                                         min_touches={"h4":3}.get(tf_key,2))
            zones_d[tf_cap] = (supports, resistances)
            if not supports.empty: all_sup_levels.extend(supports["level"].tolist())
            if tf_key == "daily":
                price_ctx = get_price_context(cp, supports, resistances)
                zones_d["_price_ctx"] = price_ctx

            def make_row(zone, ztype, _cp=cp, _atr=atr_val, _tf=tf_cap, _ntot=len(df)):
                lvl = zone["level"]; strength = int(zone["strength"])
                age_bars = int(zone.get("age_bars",0)); status = zone.get("status","Testee")
                if not _cp or _cp <= 0:
                    dist_pct = 0.0; dist_atr_str = "N/A"; in_pdf = False
                else:
                    dist_pct = abs(_cp-lvl)/_cp*100
                    dist_atr_str = f"{round(abs(_cp-lvl)/_atr,1):.1f}x" if _atr and _atr>0 else "N/A"
                    in_pdf = dist_pct <= profile.merge_threshold_atr*200  # seuil pdf simplifié
                struct_score = compute_structural_score(strength, 1, _tf, age_bars, _ntot)
                return {"Actif": sym_d, "Prix Actuel": f"{_cp:.5f}" if _cp else "N/A",
                        "Type": ztype, "Niveau": f"{lvl:.5f}", "Force": f"{strength} touches",
                        "Score (1TF)": struct_score, "Statut": status,
                        "Dist. %": f"{dist_pct:.2f}%", "Dist. ATR": dist_atr_str,
                        "_dist_num": dist_pct, "_in_pdf": in_pdf}

            tf_rows = []
            for _, z in supports.iterrows():
                tf_rows.append(make_row(z, "PIVOT" if z.get("is_pivot") else "Support"))
            for _, z in resistances.iterrows():
                tf_rows.append(make_row(z, "PIVOT" if z.get("is_pivot") else "Resistance"))
            seen = set(); unique_rows = []
            for r in tf_rows:
                key = (r["Niveau"], r["Type"])
                if key not in seen: seen.add(key); unique_rows.append(r)
            if unique_rows: rows[tf_cap] = unique_rows

        if all_sup_levels and reference_price:
            new_anomaly = flag_data_anomaly(symbol, reference_price, all_sup_levels, last_h4_close)
            if new_anomaly: anomaly_msg = (f"{anomaly_msg} | {new_anomaly}" if anomaly_msg else new_anomaly)
    except concurrent.futures.TimeoutError: internal_error = "TimeoutError données TF"
    except Exception as e: internal_error = f"{type(e).__name__}: {str(e)[:300]}"

    return ScanResult(symbol=symbol, rows=rows, zones=zones_d, price=reference_price,
                      trends=trends, anomaly=anomaly_msg, bars_map=bars_map, scan_error=internal_error)

# ══════════════════════════════════════════════════════════════════
# PDF / CSV / JSON (inchangés)
# ══════════════════════════════════════════════════════════════════
# [Le code de génération de rapports est strictement identique à celui fourni précédemment]
# ... (coller ici tout le code des fonctions create_pdf_report, create_csv_report, create_llm_brief, create_json_export,
#      ainsi que la classe PDF, strip_emojis_df, etc. exactement comme dans le précédent fichier)

# Pour ne pas alourdir cette réponse, je les omets mais ils doivent être inclus.
# ══════════════════════════════════════════════════════════════════
# SIDEBAR (inchangée)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # ... (identique au code précédent)
    pass

# ══════════════════════════════════════════════════════════════════
# LOGIQUE PRINCIPALE (inchangée, sauf appel à _display_results corrigé)
# ══════════════════════════════════════════════════════════════════
# ... (comme dans la version corrigée, avec un seul appel final)
