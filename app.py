import asyncio
import time
import nest_asyncio
import hashlib
import json
import logging
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any

import aiohttp
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ==============================================================================
# [ LAYER 0: GLOBAL CONFIG & LOGGING ]
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Patch asyncio to allow nested event loops (required for Streamlit compatibility)
nest_asyncio.apply()

# ==============================================================================
# [ LAYER 0b: ASYNC DATA CACHE ]
# ==============================================================================
# Cache candles par bucket temporel de 10 min.
# Clé : (symbol, tf, bucket) — le bucket change toutes les 600s,
# ce qui invalide automatiquement les entrées sans TTL check explicite.
# Avantage : 0 verrou, 0 thread, compatible asyncio single-threaded.
_OANDA_CACHE: Dict[tuple, Optional[pd.DataFrame]] = {}
_CACHE_TTL_SECS = 600  # 10 minutes

def _cache_bucket() -> int:
    """Retourne le bucket temporel courant (change toutes les 600s)."""
    return int(time.time()) // _CACHE_TTL_SECS

def _cache_purge_old() -> None:
    """Supprime les entrées de plus de 2 buckets (~20 min) pour éviter les fuites mémoire."""
    current = _cache_bucket()
    stale = [k for k in _OANDA_CACHE if k[2] < current - 1]
    for k in stale:
        del _OANDA_CACHE[k]


SCANNER_VERSION = "7.0-INSTITUTIONAL"

ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

_GRANULARITY_MAP = {"h4": "H4", "daily": "D", "weekly": "W"}
TF_WEIGHT = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}

# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES ]
# ==============================================================================
@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    asset_class: str           # FOREX, INDEX, METAL
    pip_value: float
    cluster_radius_atr: float  # Radius for 1D Agglomerative Clustering
    merge_threshold_atr: float # Distance for post-cluster merging
    pivot_prominence_atr: float# Rejection significance
    dev_threshold_pct: float   # Price anomaly threshold
    skip_ratio_check: bool     # Skip median vs price sanity check
    # Wick thresholds : proportion minimale de mèche sur la range de la bougie
    # pour valider un swing. Varie selon la volatilité de l'actif ET le TF.
    # Intraday (H4 et inférieur) : mèches plus courtes acceptées
    # HTF (Daily, Weekly) : on exige une mèche plus marquée = signal plus fort
    wick_threshold_intraday: float = 0.20  # H4 / M15
    wick_threshold_htf: float      = 0.30  # Daily / Weekly
    # Seuil de confluence inter-TF en % du prix.
    # Deux zones de TFs différents sont en confluence si leur écart <= ce seuil.
    # Indices volatils : seuil plus large. Forex : seuil serré.
    confluence_threshold_pct: float = 1.0

_PROFILES = {
    # ── FOREX Majors ── wick_intraday=0.20 / wick_htf=0.30 (baseline retail validé)
    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2,  0.8,  0.6,  1.5, False, wick_threshold_intraday=0.20, wick_threshold_htf=0.30, confluence_threshold_pct=1.0),
    "GBP_USD":    InstrumentProfile("GBP_USD",    "FOREX", 0.0001, 1.3,  0.85, 0.65, 1.5, False, wick_threshold_intraday=0.20, wick_threshold_htf=0.30, confluence_threshold_pct=1.0),
    "USD_JPY":    InstrumentProfile("USD_JPY",    "FOREX", 0.01,   0.9,  0.5,  0.5,  1.5, False, wick_threshold_intraday=0.20, wick_threshold_htf=0.30, confluence_threshold_pct=1.0),
    # ── Métal ── Gold : price action wick-heavy → seuil abaissé pour capturer plus de swings
    "XAU_USD":    InstrumentProfile("XAU_USD",    "METAL", 0.01,   2.0,  1.2,  1.0,  3.0, True,  wick_threshold_intraday=0.18, wick_threshold_htf=0.28, confluence_threshold_pct=1.5),
    # ── Indices ── corps larges, mèches proportionnellement réduites → seuil relevé
    "US30_USD":   InstrumentProfile("US30_USD",   "INDEX", 1.0,    1.5,  0.9,  0.7,  2.5, True,  wick_threshold_intraday=0.22, wick_threshold_htf=0.32, confluence_threshold_pct=1.5),
    "NAS100_USD": InstrumentProfile("NAS100_USD", "INDEX", 1.0,    1.5,  1.0,  0.8,  2.5, True,  wick_threshold_intraday=0.22, wick_threshold_htf=0.32, confluence_threshold_pct=1.5),
    "SPX500_USD": InstrumentProfile("SPX500_USD", "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  wick_threshold_intraday=0.22, wick_threshold_htf=0.32, confluence_threshold_pct=1.2),
    "DE30_EUR":   InstrumentProfile("DE30_EUR",   "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  wick_threshold_intraday=0.22, wick_threshold_htf=0.32, confluence_threshold_pct=1.2),
}
_DEFAULT_PROFILE = InstrumentProfile("DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
                                     wick_threshold_intraday=0.20, wick_threshold_htf=0.30)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    base = symbol.split("_")[0]
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF"):
        return InstrumentProfile(symbol, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
                                 wick_threshold_intraday=0.20, wick_threshold_htf=0.30)
    if base == "JPY" or symbol.endswith("_JPY"):
        return InstrumentProfile(symbol, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False,
                                 wick_threshold_intraday=0.20, wick_threshold_htf=0.30)
    return _DEFAULT_PROFILE

# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE (OANDA) ]
# ==============================================================================
class OandaAuthError(Exception):
    """Levée quand l'authentification OANDA échoue. Catchée par la couche UI."""
    pass


class AsyncOandaClient:
    """Institutional Async Client handling rapid parallel data fetching with connection pooling."""
    def __init__(self, token: str, account_id: str):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = account_id
        self.env_url = None
        
    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        """Determines Live vs Practice environment dynamically."""
        for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
            try:
                async with session.get(f"{url}/v3/accounts/{self.account_id}/summary", headers=self.headers, timeout=5) as r:
                    if r.status == 200:
                        self.env_url = url
                        return True
            except Exception:
                continue
        return False

    async def fetch_candles(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbol: str, tf: str, limit: int = 500) -> Tuple[str, str, Optional[pd.DataFrame]]:
        # Cache hit : évite une requête OANDA si les données sont < 10 min
        _ck = (symbol, tf, _cache_bucket())
        if _ck in _OANDA_CACHE:
            logging.debug(f"Cache HIT: {symbol}/{tf}")
            return symbol, tf, _OANDA_CACHE[_ck]

        gran = _GRANULARITY_MAP.get(tf)
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}
        
        async with sem:
            try:
                async with session.get(url, headers=self.headers, params=params, timeout=10) as r:
                    if r.status != 200: return symbol, tf, None
                    data = await r.json()
                    candles = [
                        {
                            "date": pd.to_datetime(c["time"]),
                            "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
                            "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
                            "volume": int(c["volume"])
                        } for c in data.get("candles", []) if c.get("complete")
                    ]
                    df = pd.DataFrame(candles).tail(limit).set_index("date")
                    result_df = df if not df.empty else None
                    _OANDA_CACHE[_ck] = result_df
                    _cache_purge_old()
                    return symbol, tf, result_df
            except Exception:
                return symbol, tf, None

    async def fetch_price(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbol: str) -> Tuple[str, Optional[float]]:
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            try:
                async with session.get(url, headers=self.headers, params={"instruments": symbol}, timeout=5) as r:
                    if r.status != 200: return symbol, None
                    data = await r.json()
                    if "prices" in data and data["prices"]:
                        bid = float(data["prices"][0]["closeoutBid"])
                        ask = float(data["prices"][0]["closeoutAsk"])
                        return symbol, (bid + ask) / 2
            except Exception:
                pass
        return symbol, None

# ==============================================================================
# [ LAYER 3: QUANTITATIVE ENGINE (VECTORIZED) ]
# ==============================================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    res = tr.rolling(period).mean().iloc[-1]
    return float(res) if pd.notna(res) and res > 0 else 0.001

def compute_institutional_trend(closes: pd.Series, lookback: int = 20) -> str:
    """Z-Score Normalized Linear Regression (Institutional Standard)."""
    if len(closes) < lookback: return "NEUTRE"
    y = closes.tail(lookback).values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    std_dev = np.std(y)
    if std_dev == 0: return "NEUTRE"
    z_score = slope / std_dev
    if z_score > 0.15: return "HAUSSIER"
    if z_score < -0.15: return "BAISSIER"
    return "NEUTRE"

def detect_swing_pivots(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float, timeframe: str) -> Tuple[pd.Series, pd.Series]:
    """Strictly ATR-based swing detection.

    Toutes les Series partagent le même RangeIndex 0…n-1 (créées depuis .values
    UNE SEULE FOIS). Évite le bug "Can only compare identically-labeled Series"
    causé par pd.Series(arr[::-1]) qui brise l'alignement d'index.
    """
    prominence = atr_val * profile.pivot_prominence_atr if atr_val and atr_val > 0 else 0.001
    n = 3

    # Series créées UNE FOIS — index 0…n-1 partagé, jamais recréé
    highs  = pd.Series(df["high"].values)
    lows   = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)
    opens  = pd.Series(df["open"].values)

    # [::-1] sur une Series héritée préserve l'index — le second [::-1] remet 0…n-1
    roll_high_left  = highs.shift(1).rolling(n, min_periods=n).max()
    roll_high_right = highs[::-1].shift(1).rolling(n, min_periods=n).max()[::-1]
    roll_low_left   = lows.shift(1).rolling(n, min_periods=n).min()
    roll_low_right  = lows[::-1].shift(1).rolling(n, min_periods=n).min()[::-1]

    next_close   = closes.shift(-1).fillna(closes)
    candle_range = (highs - lows).clip(lower=1e-10)
    body_top     = pd.Series(np.maximum(opens.values, closes.values))
    body_bottom  = pd.Series(np.minimum(opens.values, closes.values))

    upper_wick_pct = (highs - body_top)   / candle_range
    lower_wick_pct = (body_bottom - lows) / candle_range

    # Seuil de mèche issu du profil instrument — zéro nombre magique
    _WICK_THRESHOLD = (
        profile.wick_threshold_intraday if timeframe.lower() in ("h4", "m15")
        else profile.wick_threshold_htf
    )

    sh_mask = (
        (highs > roll_high_left) & (highs > roll_high_right) &
        (next_close < highs) & (upper_wick_pct >= _WICK_THRESHOLD)
    )
    sl_mask = (
        (lows < roll_low_left) & (lows < roll_low_right) &
        (next_close > lows) & (lower_wick_pct >= _WICK_THRESHOLD)
    )

    roll_low_around  = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
    roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()
    sh_mask = sh_mask & ((highs - roll_low_around)  >= prominence)
    sl_mask = sl_mask & ((roll_high_around - lows)   >= prominence)

    idx_highs = sh_mask[sh_mask].index.tolist()
    idx_lows  = sl_mask[sl_mask].index.tolist()

    return (
        pd.Series(highs.values[idx_highs], index=idx_highs) if idx_highs else pd.Series(dtype=float),
        pd.Series(lows.values[idx_lows],   index=idx_lows)  if idx_lows  else pd.Series(dtype=float),
    )

def agglomerative_1d_clustering(
    price_weight_pairs: List[tuple],
    bandwidth: float
) -> List[List[tuple]]:
    """Deterministic 1D Agglomerative Clustering sur (price, weight) pairs.

    Trie par prix puis regroupe les paires dont l'écart consécutif <= bandwidth.
    Transporte les poids pour permettre un centroïde pondéré par récence.

    Args:
        price_weight_pairs: liste de (price: float, weight: float)
        bandwidth: seuil d'agglomération en prix (ATR * cluster_radius)
    Returns:
        Liste de groupes, chaque groupe étant une liste de (price, weight)
    """
    if not price_weight_pairs:
        return []
    sorted_pw = sorted(price_weight_pairs, key=lambda x: x[0])
    clusters: List[List[tuple]] = []
    curr_cluster = [sorted_pw[0]]

    for i in range(1, len(sorted_pw)):
        if sorted_pw[i][0] - sorted_pw[i - 1][0] <= bandwidth:
            curr_cluster.append(sorted_pw[i])
        else:
            clusters.append(curr_cluster)
            curr_cluster = [sorted_pw[i]]
    clusters.append(curr_cluster)
    return clusters

def classify_zone_status(level: float, zone_type: str, df: pd.DataFrame, formation_idx: int, atr_val: float) -> str:
    """Dynamic ATR tolerance for Status Classification."""
    if formation_idx >= len(df) - 1: return "Vierge"
    tolerance = atr_val * 0.25

    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]
    if len(c_arr) == 0: return "Vierge"

    near = (np.abs(c_arr - level) <= tolerance) | ((l_arr <= level + tolerance) & (h_arr >= level - tolerance))
    has_approach = bool(near.any())

    break_mask = (c_arr < level - tolerance) if zone_type == "Support" else (c_arr > level + tolerance)
    break_positions = np.where(break_mask)[0]
    
    if len(break_positions) == 0: return "Testee" if has_approach else "Vierge"

    break_idx = int(break_positions[0])
    retest_tol = tolerance * 2
    rc = c_arr[break_idx + 1:]
    rh = h_arr[break_idx + 1:]
    rl = l_arr[break_idx + 1:]

    if len(rc) == 0: return "Consommee"
    retest_mask = (rl <= level + retest_tol) & (rh >= level - retest_tol)
    if not retest_mask.any(): return "Consommee"

    retest_idx = int(np.where(retest_mask)[0][0])
    rc_after = rc[retest_idx + 1:]
    if len(rc_after) == 0: return "Role Reverse"

    second_break = (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level - tolerance)
    return "Consommee" if second_break.any() else "Role Reverse"

def compute_structural_score(strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int) -> float:
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    age_f = np.exp(-1.5 * age_r)
    return round((strength * tf_w * nb_tf) * age_f, 1)

def find_strong_sr_zones(df: pd.DataFrame, current_price: float, symbol: str, atr_val: float, timeframe: str, min_touches: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profile = get_profile(symbol)
    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)

    # Fallback to pure prominent peaks if standard logic fails (low vol market)
    if len(pivot_highs) + len(pivot_lows) < 3:
        dist = {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)
        pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        pivot_highs = pd.Series(df["high"].values[r_idx], index=r_idx) if len(r_idx) else pd.Series(dtype=float)
        pivot_lows = pd.Series(df["low"].values[s_idx], index=s_idx) if len(s_idx) else pd.Series(dtype=float)

    # Recency weight : un pivot récent influence davantage le centroïde.
    # weight = (index + 1e-6) / n_total → ~0 vieux, ~1 récent
    all_pivots = (
        [(float(p), int(i), (int(i) + 1e-6) / n_total) for i, p in pivot_highs.items()] +
        [(float(p), int(i), (int(i) + 1e-6) / n_total) for i, p in pivot_lows.items()]
    )
    if not all_pivots: return pd.DataFrame(), pd.DataFrame()

    # Deterministic Agglomerative Clustering
    bandwidth = atr_val * profile.cluster_radius_atr
    price_weight_pairs = [(p, w) for p, _, w in all_pivots]
    clusters_raw = agglomerative_1d_clustering(price_weight_pairs, bandwidth)

    strong = []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    for grp_pw in clusters_raw:
        if len(grp_pw) < min_touches:
            continue
        
        grp_prices_arr = np.array([p for p, _ in grp_pw])
        grp_weights_arr = np.array([w for _, w in grp_pw])

        # Centroïde pondéré par recency : les niveaux récents priment
        lvl = float(np.average(grp_prices_arr, weights=grp_weights_arr))
        if lvl <= 0:
            continue

        grp_price_set = set(grp_prices_arr.tolist())
        grp_indices = [idx for p, idx, _ in all_pivots if p in grp_price_set]

        last_idx = max(grp_indices)
        age = max(0, n_total - 1 - last_idx)
        ztype = "Support" if lvl < current_price else "Resistance"
        status = classify_zone_status(lvl, ztype, df, last_idx, atr_val)

        strong.append({
            "level": float(lvl), "strength": len(grp_pw), "age_bars": age, "status": status
        })

    if not strong: return pd.DataFrame(), pd.DataFrame()

    # Post-merge cleanup
    strong.sort(key=lambda x: x["level"])
    merge_thresh = atr_val * profile.merge_threshold_atr
    merged = []
    
    for z in strong:
        if not merged:
            merged.append(z)
        else:
            prev = merged[-1]
            if abs(z["level"] - prev["level"]) <= merge_thresh:
                new_str = prev["strength"] + z["strength"]
                new_lvl = (prev["level"] * prev["strength"] + z["level"] * z["strength"]) / new_str
                new_status = max([prev["status"], z["status"]], key=lambda s: STATUS_PRIORITY.get(s, 1))
                merged[-1] = {
                    "level": new_lvl, "strength": new_str,
                    "age_bars": min(prev["age_bars"], z["age_bars"]), "status": new_status
                }
            else:
                merged.append(z)

    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)
    df_zones["is_pivot"] = (np.abs(df_zones["level"] - current_price) / current_price * 100) <= 0.50

    return df_zones[df_zones["level"] < current_price].copy(), df_zones[df_zones["level"] >= current_price].copy()


def detect_confluences(symbol: str, zones_dict: dict, current_price: float, bars_map: dict) -> list:
    """Vectorized-friendly Confluence Detection."""
    if not zones_dict or not current_price: return []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}
    
    # ── Collecte vectorisée des zones (remplace iterrows) ──────────────────────
    frames = []
    for tf, (sup, res) in zones_dict.items():
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            if df_z.empty:
                continue
            tmp = df_z[df_z["status"] != "Consommee"].copy()
            if tmp.empty:
                continue
            tmp = tmp.assign(
                tf=tf,
                type=tmp["is_pivot"].map({True: "Pivot", False: ztype})
            )
            frames.append(tmp[["tf", "level", "strength", "age_bars", "status", "type", "is_pivot"]])

    if not frames:
        return []
    z_df = pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
    z_df = pd.DataFrame(all_zones).sort_values("level").reset_index(drop=True)
    used = set()
    confluences = []

    # Seuil de confluence issu du profil instrument — plus de dict hardcodé
    profile = get_profile(symbol)
    threshold = profile.confluence_threshold_pct

    # itertuples : 5-10x plus rapide que iterrows (accès par attribut, pas dict)
    for z in z_df.itertuples():
        if z.Index in used or z.level <= 0:
            used.add(z.Index)
            continue
            
        similar = z_df[
            (np.abs(z_df["level"] - z.level) / z.level * 100 <= threshold) &
            (~z_df.index.isin(used))
        ]
        if len(similar) > 0:  # includes self
            group_idx = [z.Index] + similar[similar.index != z.Index].index.tolist()
            group = z_df.loc[group_idx]
            tfs = group["tf"].unique()
            
            if len(tfs) >= 2:
                used.update(group.index)
                sub_avg = group["level"].mean()
                sub_nb_tf = len(tfs)
                sub_dist = abs(current_price - sub_avg) / current_price * 100
                
                # Score vectorisé numpy — remplace for _, r in group.iterrows()
                _tf_w   = group["tf"].map(TF_WEIGHT).fillna(1.0).values
                _totals = group["tf"].map(lambda t: bars_map.get(t, 500)).values.astype(float)
                _age_r  = np.clip(group["age_bars"].values / np.maximum(_totals, 1), 0, 1)
                _age_f  = np.exp(-1.5 * _age_r)
                score   = round(float((group["strength"].values * _tf_w * sub_nb_tf * _age_f).sum()), 1)
                status = max(group["status"].tolist(), key=lambda s: STATUS_PRIORITY.get(s, 1))
                
                is_pivot = sub_dist <= 0.50
                if is_pivot:
                    ctype, sig = "Pivot", "↔ PIVOT ZONE"
                else:
                    n_sup = (group["level"] < current_price).sum()
                    n_res = (group["level"] >= current_price).sum()
                    ctype = "Support" if n_sup >= n_res else "Resistance"
                    sig = "🟢 BUY ZONE" if ctype == "Support" else "🔴 SELL ZONE"

                confluences.append({
                    "Actif": symbol, "Signal": sig, "Niveau": round(sub_avg, 5), "Type": ctype,
                    "Timeframes": " + ".join(sorted(tfs)), "Nb TF": sub_nb_tf,
                    "Force Totale": int(group["strength"].sum()), "Score": round(score, 1),
                    "Statut": status, "Distance %": round(sub_dist, 3),
                    "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else "")
                })
            else:
                used.add(z.Index)

    return confluences

# ==============================================================================
# [ LAYER 4: PIPELINE ORCHESTRATOR ]
# ==============================================================================
@dataclass
class ScanResult:
    symbol: str
    rows: dict
    zones: dict
    price: Optional[float]
    trends: dict
    bars_map: dict
    scan_error: Optional[str] = None
    price_context: str = ""

async def run_institutional_scan(symbols: List[str], token: str, account_id: str, min_touches_ui: int) -> List[ScanResult]:
    """Async Orchestrator for Maximum Performance."""
    client = AsyncOandaClient(token, account_id)
    async with aiohttp.ClientSession() as session:
        if not await client.initialize(session):
            raise OandaAuthError("Impossible de s'authentifier sur OANDA. Vérifiez vos secrets API.")

        sem = asyncio.Semaphore(15) # Rate limit protection
        
        # 1. Fetch Prices
        price_tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
        prices_res = await asyncio.gather(*price_tasks)
        live_prices = {sym: p for sym, p in prices_res}

        # 2. Fetch Candles (All symbols, All TFs)
        candle_tasks = []
        for sym in symbols:
            for tf in _GRANULARITY_MAP.keys():
                candle_tasks.append(client.fetch_candles(session, sem, sym, tf))
        
        candles_res = await asyncio.gather(*candle_tasks)
        data_cube = {}
        for sym, tf, df in candles_res:
            if sym not in data_cube: data_cube[sym] = {}
            data_cube[sym][tf] = df

    # 3. Process Quant Engine Locally (CPU Bound, but fast enough natively)
    results = []
    for sym in symbols:
        try:
            profile = get_profile(sym)
            cp = live_prices.get(sym)
            sym_d = sym.replace("_", "/")
            
            rows = {"H4": None, "Daily": None, "Weekly": None}
            zones_d = {}
            trends = {}
            bars_map = {}
            price_ctx = ""
            
            for tf_k, tf_name in [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]:
                df = data_cube.get(sym, {}).get(tf_k)
                if df is None or df.empty: continue
                
                if not cp: cp = float(df["close"].iloc[-1])
                bars_map[tf_name] = len(df)
                trends[tf_name] = compute_institutional_trend(df["close"])
                atr_val = compute_atr(df)
                
                min_t = max(3, min_touches_ui) if tf_k == "h4" else max(2, min_touches_ui)
                sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
                zones_d[tf_name] = (sup, res)
                
                if tf_k == "daily":
                    # Quick context string builder
                    parts = []
                    if not sup.empty:
                        s_near = sup[(sup["level"] < cp) & (abs(sup["level"] - cp)/cp*100 <= 5.0)]
                        if not s_near.empty:
                            n_s = s_near.nlargest(1, "level").iloc[0]
                            d_s = abs(cp - n_s["level"])/cp*100
                            parts.append(f"{'SUR support' if d_s<0.5 else 'S proche'}: {n_s['level']:.5f} (-{d_s:.2f}%)")
                    if not res.empty:
                        r_near = res[(res["level"] > cp) & (abs(res["level"] - cp)/cp*100 <= 5.0)]
                        if not r_near.empty:
                            n_r = r_near.nsmallest(1, "level").iloc[0]
                            d_r = abs(cp - n_r["level"])/cp*100
                            parts.append(f"{'SUR resistance' if d_r<0.5 else 'R proche'}: {n_r['level']:.5f} (+{d_r:.2f}%)")
                    price_ctx = "  |  ".join(parts) if parts else "Zone intermediaire"

                def mk_row(z, zt):
                    dist = abs(cp - z["level"])/cp*100 if cp else 0
                    dist_atr = f"{round(abs(cp - z['level'])/atr_val,1)}x" if atr_val>0 else "N/A"
                    in_pdf = (dist <= (profile.merge_threshold_atr * 200)) or (dist <= 8.0)
                    return {
                        "Actif": sym_d, "Prix Actuel": f"{cp:.5f}" if cp else "N/A", "Type": zt,
                        "Niveau": f"{z['level']:.5f}", "Force": f"{z['strength']} touches",
                        "Score (1TF)": compute_structural_score(z["strength"], 1, tf_name, z["age_bars"], len(df)),
                        "Statut": z["status"], "Dist. %": f"{dist:.2f}%", "Dist. ATR": dist_atr,
                        "_dist_num": dist, "_in_pdf": in_pdf
                    }
                
                tf_r = [mk_row(z, "PIVOT" if z.get("is_pivot") else "Support") for _, z in sup.iterrows()] + \
                       [mk_row(z, "PIVOT" if z.get("is_pivot") else "Resistance") for _, z in res.iterrows()]
                
                seen = set()
                uniq = []
                for r in tf_r:
                    if (r["Niveau"], r["Type"]) not in seen:
                        seen.add((r["Niveau"], r["Type"]))
                        uniq.append(r)
                if uniq: rows[tf_name] = uniq

            results.append(ScanResult(sym, rows, zones_d, cp, trends, bars_map, price_context=price_ctx))
        except Exception as e:
            results.append(ScanResult(sym, {}, {}, None, {}, {}, scan_error=str(e)))
            
    return results

# ==============================================================================
# [ LAYER 5: EXPORTERS & UTILS ]
# ==============================================================================
# (Preserved exact legacy formatting to ensure UI/PDF/JSON compatibility)
_ACCENT_MAP = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ', 'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
_EMOJI_MAP = [('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'), ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]'), ('📈', ''), ('📉', ''), ('✅', '[OK]'), ('❌', '[X]'), ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'), ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]')]

def _safe_pdf_str(text: str) -> str:
    text = str(text).translate(_ACCENT_MAP)
    for e, r in _EMOJI_MAP: text = text.replace(e, r)
    return text

def strip_emojis_df(df):
    cln = df.copy()
    for col in cln.select_dtypes(include='object').columns: cln[col] = cln[col].apply(_safe_pdf_str)
    return cln

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["_dist_num", "_in_pdf"], errors="ignore")

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, _safe_pdf_str('Rapport Scanner Bluestar - Supports & Resistances'), border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6, _safe_pdf_str(f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  |  v{SCANNER_VERSION}  |  Score = (Force x Poids_TF x NbTF) x Facteur_Age | Statut Vierge / Testee / Role Reverse / Consommee"), border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)
    def chapter_summary(self, summaries):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, _safe_pdf_str('RESUME PAR ACTIF  (Tendances + Top Zones Confluentes)'), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        for s in summaries:
            sym, t_h4, t_d, t_w, ctx = _safe_pdf_str(s.get('symbol','')), _safe_pdf_str(s.get('trend_h4','N/A')), _safe_pdf_str(s.get('trend_daily','N/A')), _safe_pdf_str(s.get('trend_weekly','N/A')), _safe_pdf_str(s.get('price_context',''))
            self.set_font('Helvetica', 'B', 8)
            self.cell(0, 5, _safe_pdf_str(f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"), border=0, new_x='LMARGIN', new_y='NEXT')
            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}", border=0, new_x='LMARGIN', new_y='NEXT')
            top = s.get('top_zones', [])
            self.set_font('Helvetica', '', 7)
            if top:
                for z in top:
                    txt = _safe_pdf_str(f"  {z.get('Signal','')}  Niv:{z.get('Niveau','')}  Dist:{z.get('Distance %','')}  Score:{z.get('Score','')}  TF:{z.get('Timeframes','')}  {z.get('Alerte','')}")
                    self.cell(0, 4, txt[:130], border=0, new_x='LMARGIN', new_y='NEXT')
            else:
                self.cell(0, 4, "  Aucune confluence pour cet actif.", border=0, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)
    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.set_x(self.l_margin)
            self.multi_cell(self.w - self.l_margin - self.r_margin, 10, "Aucune donnee a afficher.")
            self.ln()
            return
        col_w = {'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22, 'Timeframes': 50, 'Nb TF': 12, 'Force Totale': 20, 'Score': 18, 'Statut': 22, 'Distance %': 18, 'Alerte': 55} if 'Timeframes' in df.columns else {'Actif': 24, 'Prix Actuel': 24, 'Type': 20, 'Niveau': 24, 'Force': 20, 'Score (1TF)': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16}
        cols = [c for c in col_w if c in df.columns]
        x_start = self.l_margin + max(0, ((self.w - self.l_margin - self.r_margin) - sum(col_w[c] for c in cols)) / 2)
        self.set_font('Helvetica', 'B', 7)
        self.set_x(x_start)
        for c in cols: self.cell(col_w[c], 6, _safe_pdf_str(c), border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()
        self.set_font('Helvetica', '', 7)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for c in cols:
                val = _safe_pdf_str(str(row[c]))
                mx = int(col_w[c] / 1.25)
                self.cell(col_w[c], 5, val[:mx-1]+'.' if len(val)>mx else val, border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()

def create_pdf_report(rep_dict, conf_df, summaries):
    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    if summaries:
        pdf.chapter_summary(summaries)
        pdf.add_page()
    if conf_df is not None and not conf_df.empty:
        pdf.chapter_title('*** ZONES DE CONFLUENCE MULTI-TIMEFRAMES ***')
        cln = strip_emojis_df(_clean_df(conf_df.copy()))
        if "Score" in cln.columns: cln = cln.sort_values("Score", ascending=False)
        pdf.chapter_body(cln)
        pdf.ln(10)
    tmap = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journaliere (Daily)', 'Weekly': 'Analyse Hebdomadaire (Weekly)'}
    for tf, df in rep_dict.items():
        if df is None or df.empty: continue
        pdf.chapter_title(tmap.get(tf, tf))
        cln = strip_emojis_df(_clean_df(df.copy()))
        if "Score (1TF)" in cln.columns: cln = cln.sort_values("Score (1TF)", ascending=False)
        pdf.chapter_body(cln)
        pdf.ln(10)
    return bytes(pdf.output())

def create_csv_report(rep_dict, conf_df):
    dfs = []
    if conf_df is not None and not conf_df.empty:
        c = _clean_df(conf_df).copy()
        c["Section"] = "CONFLUENCES"
        dfs.append(c)
    for tf, df in rep_dict.items():
        if df is not None and not df.empty:
            d = _clean_df(df).copy()
            d["Timeframe"] = tf
            dfs.append(d)
    if not dfs: return b""
    buf = BytesIO()
    pd.concat(dfs, ignore_index=True).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()

def create_json_export(summaries, conf_df, max_dist=5.0, min_score=60.0, allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    out = {"generated_at": datetime.now().isoformat(), "scanner_version": SCANNER_VERSION, "filters": {"max_dist_pct": max_dist, "min_score": min_score}, "assets": []}
    smap = {s["symbol"]: s for s in summaries}
    
    agrp = {}
    if conf_df is not None and not conf_df.empty:
        for _, r in conf_df.iterrows():
            try: dist = float(str(r.get("Distance %", 999)).replace("%", ""))
            except: dist = 999.0
            try: score = float(r.get("Score", 0))
            except: score = 0.0
            
            sig = str(r.get("Signal", "")).replace("🟢","").replace("🔴","").replace("↔️","").replace("↔","").replace("ZONE","").strip()
            if "PIVOT" in sig: sig = "PIVOT"
            elif "BUY" in sig: sig = "BUY"
            elif "SELL" in sig: sig = "SELL"
            
            stat = str(r.get("Statut", ""))
            if dist > max_dist or score < min_score or sig not in ("BUY","SELL","PIVOT") or stat not in allowed_statuts: continue
            
            sym = str(r.get("Actif", ""))
            if sym not in agrp: agrp[sym] = []
            
            try: lvl = round(float(r.get("Niveau", 0)), 5)
            except: continue
            
            tfs = sorted([p.strip() for p in str(r.get("Timeframes", "")).replace("+",",").split(",") if p.strip()], key=lambda t: {"Weekly":0,"Daily":1,"H4":2}.get(t,99))
            alrt = str(r.get("Alerte", "")).replace("🔥","").replace("⚠️","").replace("⚠","").strip()
            alrt = "HOT" if "CHAUD" in alrt.upper() or "HOT" in alrt.upper() else ("CLOSE" if "PROCHE" in alrt.upper() or "CLOSE" in alrt.upper() else "")
            
            agrp[sym].append({"signal": sig, "type": str(r.get("Type", "")), "level": lvl, "score": round(score, 1), "status": stat, "distance_pct": round(dist, 3), "alert": alrt, "timeframes": tfs, "nb_tf": int(r.get("Nb TF", len(tfs)))})

    all_syms = set(smap.keys()).union(agrp.keys())
    for sym in sorted(all_syms, key=lambda a: max((z["score"] for z in agrp.get(a, [])), default=0.0), reverse=True):
        s = smap.get(sym, {})
        cp = s.get("current_price")
        
        bmap = {"HAUSSIER": "BULLISH", "BAISSIER": "BEARISH", "NEUTRE": "NEUTRAL"}
        bh4, bd, bw = bmap.get(s.get("trend_h4","NEUTRE"),"NEUTRAL"), bmap.get(s.get("trend_daily","NEUTRE"),"NEUTRAL"), bmap.get(s.get("trend_weekly","NEUTRE"),"NEUTRAL")
        
        if bd == bw and bd != "NEUTRAL": dom, align = bd, "ALIGNED" if bh4 == bd else "CONFLICTED"
        elif bd == "NEUTRAL" and bw != "NEUTRAL": dom, align = bw, "ALIGNED" if bh4 == bw else "CONFLICTED"
        elif bw == "NEUTRAL" and bd != "NEUTRAL": dom, align = bd, "ALIGNED" if bh4 == bd else "CONFLICTED"
        else: dom, align = "NEUTRAL", "MIXED"
        
        out["assets"].append({"symbol": sym, "current_price": round(cp, 5) if cp else None, "trends": {"h4": s.get("trend_h4","NEUTRE"), "daily": s.get("trend_daily","NEUTRE"), "weekly": s.get("trend_weekly","NEUTRE")}, "trend_alignment": align, "dominant_bias": dom, "price_context": s.get("price_context", ""), "nb_zones": len(agrp.get(sym, [])), "zones": sorted(agrp.get(sym, []), key=lambda z: z["score"], reverse=True)})
    
    return json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8")

def create_llm_brief(summaries, conf_df, max_dist=2.0, min_score=100.0, allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    lines = [f"# BRIEF INSTITUTIONNEL S/R — v{SCANNER_VERSION}", f"_Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}_", "", "## LEGEND", "- Sc > 300: Institutional / 100-300: Strong / < 100: Standard", ""]
    if conf_df is None or conf_df.empty: return "\n".join(lines).encode("utf-8")
    
    agrp = {}
    for _, r in conf_df.iterrows():
        try: dist = float(str(r.get("Distance %", 999)).replace("%", ""))
        except: dist = 999.0
        try: score = float(r.get("Score", 0))
        except: score = 0.0
        stat = str(r.get("Statut", ""))
        
        if dist <= max_dist and score >= min_score and stat in allowed_statuts:
            sym = str(r.get("Actif", ""))
            if sym not in agrp: agrp[sym] = []
            agrp[sym].append({"sig": str(r.get("Signal","")), "lvl": str(r.get("Niveau","")), "sc": score, "st": stat, "d": dist, "tf": str(r.get("Timeframes","")).replace("Daily","D").replace("Weekly","W").replace(" + ","+"), "al": str(r.get("Alerte",""))})

    smap = {s["symbol"]: s for s in summaries}
    tarr = {"HAUSSIER": "↑", "BAISSIER": "↓", "NEUTRE": "→"}
    stlb = {"Vierge": "V", "Testee": "T", "Role Reverse": "RR", "Consommee": "C"}
    allb = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠"}

    for sym in sorted(agrp.keys(), key=lambda a: max(z["sc"] for z in agrp[a]), reverse=True):
        s = smap.get(sym, {})
        ctx = s.get("price_context", "")
        lines.append(f"### {sym} | H4:{tarr.get(s.get('trend_h4','NEUTRE'),'→')} D:{tarr.get(s.get('trend_daily','NEUTRE'),'→')} W:{tarr.get(s.get('trend_weekly','NEUTRE'),'→')}")
        if ctx: lines.append(f"> {ctx}")
        for z in sorted(agrp[sym], key=lambda x: x["sc"], reverse=True):
            sg = "BUY  " if "BUY" in z["sig"] else ("SELL " if "SELL" in z["sig"] else "PIVOT")
            lines.append(f"- {sg} `{z['lvl']}` | Sc:{z['sc']:.0f} | {stlb.get(z['st'], z['st'])} | {z['d']:.2f}% | {z['tf']} {allb.get(z['al'], '')}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")

# ==============================================================================
# [ LAYER 6: PRESENTATION (STREAMLIT UI) ]
# ==============================================================================
st.set_page_config(page_title="Institutional S/R Quant Engine", page_icon="🏦", layout="wide")
st.markdown("""
    <style>
    [data-testid="stDataFrame"] > div { overflow-x: visible !important; overflow-y: visible !important; }
    ::-webkit-scrollbar { width: 0px !important; height: 0px !important; }
    div[data-testid="stButton"] > button[kind="primary"] { background-color: #0A2540; color: white; border: 1px solid #000; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #113A63; }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Institutional S/R Quant Engine v7.0")
st.markdown("Prop-Firm Grade. **Async I/O**, **Deterministic 1D Agglomerative Clustering**, **Z-Score Trend Normalization**.")

with st.sidebar:
    st.header("1. OANDA API (Live/Practice)")
    token = st.secrets.get("OANDA_ACCESS_TOKEN", "")
    acc_id = st.secrets.get("OANDA_ACCOUNT_ID", "")
    if token and acc_id: st.success("Secrets OK ✓")
    else: st.error("Missing Secrets")

    st.header("2. Assets")
    sel_all = st.checkbox(f"All Assets ({len(ALL_SYMBOLS)})", value=True)
    syms_to_scan = ALL_SYMBOLS if sel_all else st.multiselect("Select:", ALL_SYMBOLS, default=["XAU_USD", "EUR_USD"])

    st.header("3. LLM / Quant Export Filters")
    llm_max_dist = st.slider("Max Dist (%)", 0.5, 5.0, 2.0, 0.5)
    llm_min_score = st.slider("Min Score", 40, 300, 60, 10)
    llm_statuts = st.multiselect("Valid Status", ["Vierge", "Testee", "Role Reverse", "Consommee"], default=["Vierge", "Testee", "Role Reverse"])

    st.header("4. Display Filters")
    min_touches = st.slider("Min Touches", 2, 10, 2)
    max_dist_filter = st.slider("UI Max Dist (%)", 1.0, 15.0, 3.0, 0.5)

def _display_results(sr: dict, max_dist_filter: float):
    df_h4, df_daily, df_wk = sr.get("df_h4", pd.DataFrame()), sr.get("df_daily", pd.DataFrame()), sr.get("df_weekly", pd.DataFrame())
    conf_full = sr.get("conf_full", pd.DataFrame())
    errors = sr.get("scan_errors", {})
    
    if errors:
        with st.expander(f"❌ {len(errors)} errors"):
            for k, v in errors.items(): st.error(f"{k}: {v}")

    if not conf_full.empty:
        c_filt = conf_full[pd.to_numeric(conf_full["Distance %"], errors="coerce").fillna(999) <= max_dist_filter].reset_index(drop=True)
        if not c_filt.empty:
            st.subheader("🔥 HIGH PROBABILITY CONFLUENCES (MULTI-TF)")
            st.dataframe(c_filt.sort_values("Score", ascending=False), hide_index=True, use_container_width=True, height=400)
    
    st.subheader("📋 Exports")
    c1, c2, c3, c4 = st.columns(4)
    if not conf_full.empty:
        c1.download_button("📄 PDF", data=create_pdf_report(sr["report_dict"], conf_full, sr["summaries"]), file_name="report.pdf", use_container_width=True)
        c2.download_button("📊 CSV", data=create_csv_report(sr["report_dict"], conf_full), file_name="data.csv", use_container_width=True)
        c3.download_button("🤖 LLM MD", data=create_llm_brief(sr["summaries"], conf_full, llm_max_dist, llm_min_score, llm_statuts), file_name="llm.md", use_container_width=True)
        c4.download_button("🔧 JSON", data=create_json_export(sr["summaries"], conf_full, llm_max_dist, llm_min_score, tuple(llm_statuts)), file_name="quant.json", use_container_width=True)

    def _f(df):
        if df.empty or "Dist. %" not in df.columns: return df
        return df[df["Dist. %"].astype(str).str.replace("%","").astype(float) <= max_dist_filter].sort_values("Score (1TF)", ascending=False).reset_index(drop=True)

    st.divider()
    c_h4, c_d, c_w = st.columns(3)
    with c_h4:
        st.write("📅 H4")
        st.dataframe(_f(df_h4), hide_index=True, use_container_width=True)
    with c_d:
        st.write("📅 Daily")
        st.dataframe(_f(df_daily), hide_index=True, use_container_width=True)
    with c_w:
        st.write("📅 Weekly")
        st.dataframe(_f(df_wk), hide_index=True, use_container_width=True)

if st.button("🚀 RUN INSTITUTIONAL QUANT SCAN", type="primary", use_container_width=True):
    if not token or not acc_id:
        st.error("Missing OANDA Secrets.")
    else:
        with st.spinner("Executing High-Frequency Async I/O & Quant Agglomeration..."):
            # nest_asyncio.apply() (en haut du fichier) permet l'utilisation d'asyncio.run()
            # dans l'event loop active de Streamlit, sans RuntimeError.
            try:
                raw_results = asyncio.run(run_institutional_scan(syms_to_scan, token, acc_id, min_touches))
            except OandaAuthError as e:
                st.error(str(e))
                st.stop()
            
            # Post-Process for UI State
            h4_r, d_r, w_r, confs, summaries, errs = [], [], [], [], [], {}
            for r in raw_results:
                if r.scan_error: errs[r.symbol] = r.scan_error
                else:
                    if r.rows.get("H4"): h4_r.extend(r.rows["H4"])
                    if r.rows.get("Daily"): d_r.extend(r.rows["Daily"])
                    if r.rows.get("Weekly"): w_r.extend(r.rows["Weekly"])
                    
                    c = detect_confluences(r.symbol.replace("_","/"), r.zones, r.price, r.bars_map)
                    confs.extend(c)
                    
                    summaries.append({
                        "symbol": r.symbol.replace("_","/"),
                        "trend_h4": r.trends.get("H4", "NEUTRE"),
                        "trend_daily": r.trends.get("Daily", "NEUTRE"),
                        "trend_weekly": r.trends.get("Weekly", "NEUTRE"),
                        "price_context": r.price_context,
                        "current_price": r.price,
                        "top_zones": sorted(c, key=lambda x: x["Score"], reverse=True)[:3] if c else []
                    })
            
            df_h4, df_d, df_w = pd.DataFrame(h4_r), pd.DataFrame(d_r), pd.DataFrame(w_r)
            
            # Filter internal columns for PDF rep
            r_dict = {
                "H4": _clean_df(df_h4[df_h4["_in_pdf"]]) if not df_h4.empty else pd.DataFrame(),
                "Daily": _clean_df(df_d[df_d["_in_pdf"]]) if not df_d.empty else pd.DataFrame(),
                "Weekly": _clean_df(df_w[df_w["_in_pdf"]]) if not df_w.empty else pd.DataFrame(),
            }

            st.session_state["scan_results"] = {
                "df_h4": _clean_df(df_h4), "df_daily": _clean_df(df_d), "df_weekly": _clean_df(df_w),
                "conf_full": pd.DataFrame(confs), "report_dict": r_dict,
                "summaries": summaries, "scan_errors": errs
            }
            
if "scan_results" in st.session_state:
    _display_results(st.session_state["scan_results"], max_dist_filter)
    
