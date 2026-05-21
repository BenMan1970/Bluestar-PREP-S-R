import asyncio
import time
import nest_asyncio
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple, List, Dict

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
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
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
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, KeyError, ValueError):
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
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, KeyError, ValueError):
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
    """Z-Score Normalized Linear Regression — scale-invariant multi-asset.

    La série est normalisée par closes[0] avant polyfit : slope et std_dev sont
    en %/bougie, indépendants du niveau de prix absolu.
    → Le seuil 0.10 est universel : EURUSD (1.16), XAU (4500), US30 (50 000).

    Ancien bug : sans normalisation, slope/std_dev donnait un z_score 2-3x plus
    faible sur les actifs à prix élevé (XAU, indices) → biais NEUTRE systématique.
    """
    if len(closes) < lookback:
        return "NEUTRE"
    y = closes.tail(lookback).values
    base = y[0]
    if base == 0:
        return "NEUTRE"
    # Normalisation : série en ratio → slope en %/bougie, comparable entre actifs
    y_norm = y / base
    x = np.arange(len(y_norm))
    slope, _ = np.polyfit(x, y_norm, 1)
    std_dev = np.std(y_norm)
    if std_dev == 0:
        return "NEUTRE"
    z_score = slope / std_dev
    if z_score >  0.10: return "HAUSSIER"
    if z_score < -0.10: return "BAISSIER"
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


def detect_confluences(symbol: str, zones_dict: dict, current_price: float, bars_map: dict, confluence_threshold: Optional[float] = None) -> list:
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
    used = set()
    confluences = []

    # Seuil de confluence issu du profil instrument — plus de dict hardcodé
    profile = get_profile(symbol)
    threshold = confluence_threshold if confluence_threshold is not None else profile.confluence_threshold_pct

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
    anomaly: Optional[str] = None
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

# ==============================================================================
# EXPORT FUNCTIONS (portées depuis v6.0)
# ==============================================================================

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
        except (ValueError, TypeError, AttributeError):
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
            except (ValueError, TypeError, AttributeError):
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
    except (ValueError, KeyError):
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
        except (UnicodeDecodeError, AttributeError, TypeError):
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




# ==============================================================================
# CONSTANTES UI
# ==============================================================================

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
    "XAU_USD":    8.0, "US30_USD": 5.0, "NAS100_USD": 5.0,
    "SPX500_USD": 5.0, "DE30_EUR": 5.0,
}

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




# ==============================================================================
# LOGIQUE PRINCIPALE
# ==============================================================================

if scan_button and symbols_to_scan and not st.session_state.get("scanning", False):
    st.session_state.pop("scan_results", None)
    st.session_state["scanning"]     = True
    st.session_state["pending_scan"] = True
    st.rerun()

if st.session_state.get("pending_scan", False) and symbols_to_scan:
    st.session_state.pop("pending_scan", None)

    if not access_token or not account_id:
        st.session_state["scanning"] = False
        st.warning("Configurez vos secrets OANDA avant de lancer le scan.")
    else:
        progress_bar = st.progress(0, text="Initialisation du scan async…")

        results_h4, results_daily, results_weekly = [], [], []
        all_zones_map   = {}
        prices_map      = {}
        trends_map      = {}
        anomalies_map   = {}
        scan_errors     = {}
        bars_map_global = {}

        try:
            with st.spinner("Pipeline async I/O en cours…"):
                raw_results = asyncio.run(
                    run_institutional_scan(symbols_to_scan, access_token, account_id, min_touches)
                )

            total = len(raw_results)
            for idx, result in enumerate(raw_results):
                sym_label = result.symbol.replace("_", "/")
                progress_bar.progress(
                    (idx + 1) / max(total, 1),
                    text=f"Post-traitement… ({idx+1}/{total}) {sym_label}",
                )

                if result.scan_error:
                    scan_errors[_sym_display(result.symbol)] = result.scan_error
                    continue

                all_zones_map[result.symbol]   = result.zones
                prices_map[result.symbol]      = result.price
                trends_map[result.symbol]      = result.trends
                bars_map_global[result.symbol] = result.bars_map

                if result.anomaly:
                    anomalies_map[_sym_display(result.symbol)] = result.anomaly

                for tf_cap, tf_rows in result.rows.items():
                    if tf_rows:
                        if tf_cap == "H4":       results_h4.extend(tf_rows)
                        elif tf_cap == "Daily":  results_daily.extend(tf_rows)
                        elif tf_cap == "Weekly": results_weekly.extend(tf_rows)

        except OandaAuthError as e:
            st.error(str(e))
            st.session_state["scanning"] = False
            st.stop()
        except Exception as e:
            tb = traceback.format_exc()
            tb = _sanitize_traceback(tb, [access_token, account_id])
            st.error(f"Erreur inattendue : {type(e).__name__} — {tb[-400:]}")
            st.session_state["scanning"] = False
            st.stop()

        progress_bar.empty()
        st.session_state["scanning"] = False

        n_ok   = len(symbols_to_scan) - len(scan_errors)
        n_fail = len(scan_errors)
        if n_fail == 0:
            st.success(f"✅ Scan terminé — {n_ok} actifs analysés avec succès.")
        else:
            st.warning(f"⚠️ Scan terminé — {n_ok} actifs OK, {n_fail} en erreur.")

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
                _sym_display(sym), zones_clean, cp,
                bars_map_global.get(sym, {}),
                confluence_threshold=sym_threshold,
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
            if not conf_full.empty and "Actif" in conf_full.columns and sym_d in conf_full["Actif"].values:
                ac = conf_full[conf_full["Actif"] == sym_d].copy()
                top_zones = ac.sort_values("Score", ascending=False).head(3).to_dict("records")
            price_ctx = ""
            d_zones = all_zones_map.get(sym, {})
            if "_price_ctx" in d_zones:
                price_ctx = d_zones["_price_ctx"]
            elif "Daily" in d_zones and cp:
                sup_d, res_d = d_zones["Daily"]
                price_ctx = get_price_context(cp, sup_d, res_d)
            summaries.append({
                "symbol": sym_d,
                "trend_h4":      trends.get("H4",     "NEUTRE"),
                "trend_daily":   trends.get("Daily",  "NEUTRE"),
                "trend_weekly":  trends.get("Weekly", "NEUTRE"),
                "price_context": price_ctx,
                "top_zones":     top_zones,
                "current_price": cp,
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
            "df_h4":       df_h4,
            "df_daily":    df_daily,
            "df_weekly":   df_wk,
            "conf_full":   conf_full,
            "report_dict": rep_dict,
            "summaries":   summaries,
            "anomalies":   anomalies_map,
            "scan_errors": scan_errors,
            "max_dist":    max_dist_filter,
        }

elif not symbols_to_scan and not st.session_state.get("scanning", False):
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
elif not st.session_state.get("pending_scan", False) and not st.session_state.get("scanning", False):
    st.info("Configurez les paramètres dans la barre latérale, puis cliquez sur **LANCER LE SCAN COMPLET**.")

if "scan_results" in st.session_state and not st.session_state.get("pending_scan", False):
    _display_results(
        st.session_state["scan_results"],
        max_dist_filter,
    )
