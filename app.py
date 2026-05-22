import asyncio
import re
import threading
import time
try:
    import nest_asyncio
    _NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    _NEST_ASYNCIO_AVAILABLE = False
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional, Tuple, List, Dict

try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except ImportError:
    _NY_TZ = None

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

if _NEST_ASYNCIO_AVAILABLE:
    nest_asyncio.apply()

# ==============================================================================
# [ LAYER 0b: THREAD-SAFE DATA CACHE ]
# ==============================================================================
# FIX BUG-011 : threading.Lock au lieu d'asyncio.Lock (immune aux event-loops Streamlit)
# FIX BUG-005 : TTL relatif par TF (H4=60s, Daily=300s, Weekly=600s)
# FIX CONC-001 : Cache borné (LRU avec purge systématique à la lecture)
_CACHE_TTL_BY_TF: Dict[str, int] = {"h4": 60, "daily": 300, "weekly": 600}
_CACHE_TTL_DEFAULT = 300
_CACHE_MAX_ENTRIES = 256
_CACHE_LOCK = threading.Lock()
_OANDA_CACHE: Dict[tuple, tuple] = {}

def _cache_ttl(tf: str) -> int:
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)

def _cache_is_fresh(fetched_at: float, tf: str) -> bool:
    return (time.time() - fetched_at) <= _cache_ttl(tf)

def _cache_purge_stale_locked() -> None:
    now = time.time()
    stale = [
        k for k, (ts, _) in _OANDA_CACHE.items()
        if (now - ts) > _cache_ttl(k[1])
    ]
    for k in stale:
        del _OANDA_CACHE[k]
    if len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        items = sorted(_OANDA_CACHE.items(), key=lambda kv: kv[1][0], reverse=True)
        keep = dict(items[:_CACHE_MAX_ENTRIES])
        _OANDA_CACHE.clear()
        _OANDA_CACHE.update(keep)

def _cache_get(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    with _CACHE_LOCK:
        _cache_purge_stale_locked()
        entry = _OANDA_CACHE.get((symbol, tf))
        if entry is None:
            return None
        fetched_at, df = entry
        if not _cache_is_fresh(fetched_at, tf):
            del _OANDA_CACHE[(symbol, tf)]
            return None
        return df

def _cache_set(symbol: str, tf: str, df: Optional[pd.DataFrame]) -> None:
    with _CACHE_LOCK:
        _OANDA_CACHE[(symbol, tf)] = (time.time(), df)
        _cache_purge_stale_locked()

SCANNER_VERSION = "7.9-AUDITED"

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
_TF_LAMBDA = {"H4": 2.0, "Daily": 1.0, "Weekly": 0.5}

# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES ]
# ==============================================================================
@dataclass(frozen=True)
class InstrumentProfile:
    symbol: str
    asset_class: str
    pip_value: float
    cluster_radius_atr: float
    merge_threshold_atr: float
    pivot_prominence_atr: float
    dev_threshold_pct: float
    skip_ratio_check: bool
    wick_threshold_intraday: float = 0.20
    wick_threshold_htf: float = 0.30
    confluence_threshold_pct: float = 1.0
    max_live_vs_close_pct: float = 5.0
    pdf_max_dist_pct: float = 5.0

_PROFILES = {
    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2,  0.8,  0.6,  1.5, False, 0.20, 0.30, 1.0, 5.0, 5.0),
    "GBP_USD":    InstrumentProfile("GBP_USD",    "FOREX", 0.0001, 1.3,  0.85, 0.65, 1.5, False, 0.20, 0.30, 1.0, 5.0, 5.0),
    "USD_JPY":    InstrumentProfile("USD_JPY",    "FOREX", 0.01,   0.9,  0.5,  0.5,  1.5, False, 0.20, 0.30, 1.0, 5.0, 5.0),
    "XAU_USD":    InstrumentProfile("XAU_USD",    "METAL", 0.01,   2.0,  1.2,  1.0,  3.0, True,  0.18, 0.28, 1.5, 10.0, 8.0),
    "US30_USD":   InstrumentProfile("US30_USD",   "INDEX", 1.0,    1.5,  0.9,  0.7,  2.5, True,  0.22, 0.32, 1.5, 8.0, 5.0),
    "NAS100_USD": InstrumentProfile("NAS100_USD", "INDEX", 1.0,    1.5,  1.0,  0.8,  2.5, True,  0.22, 0.32, 1.5, 8.0, 5.0),
    "SPX500_USD": InstrumentProfile("SPX500_USD", "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  0.22, 0.32, 1.2, 8.0, 5.0),
    "DE30_EUR":   InstrumentProfile("DE30_EUR",   "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  0.22, 0.32, 1.2, 8.0, 5.0),
}
_DEFAULT_PROFILE = InstrumentProfile("DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    base = symbol.split("_")[0]
    if symbol.endswith("_JPY") or base == "JPY":
        return InstrumentProfile(symbol, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False)
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"):
        return InstrumentProfile(symbol, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)
    return _DEFAULT_PROFILE

# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE (OANDA) ]
# ==============================================================================
class OandaAuthError(Exception):
    """Levée quand l'authentification OANDA échoue."""

class AsyncOandaClient:
    def __init__(self, token: str, account_id: str):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = account_id
        self.env_url: Optional[str] = None

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
            try:
                async with session.get(
                    f"{url}/v3/accounts/{self.account_id}/summary",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        self.env_url = url
                        return True
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                continue
        return False

    async def fetch_candles(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
        tf: str,
        limit: int = 500,
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
        cached = _cache_get(symbol, tf)
        if cached is not None:
            return symbol, tf, cached

        gran = _GRANULARITY_MAP.get(tf)
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}

        async with sem:
            try:
                async with session.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status != 200:
                        return symbol, tf, None
                    data = await r.json()
                    candles = [
                        {
                            "date": pd.to_datetime(c["time"]),
                            "open": float(c["mid"]["o"]),
                            "high": float(c["mid"]["h"]),
                            "low": float(c["mid"]["l"]),
                            "close": float(c["mid"]["c"]),
                            "volume": int(c["volume"]),
                        }
                        for c in data.get("candles", [])
                        if c.get("complete")
                    ]
                    if not candles:
                        _cache_set(symbol, tf, None)
                        return symbol, tf, None
                    df = pd.DataFrame(candles).tail(limit).set_index("date")
                    result_df = df if not df.empty else None
                    _cache_set(symbol, tf, result_df)
                    return symbol, tf, result_df
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, KeyError, ValueError):
                return symbol, tf, None

    async def fetch_price(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
    ) -> Tuple[str, Optional[float]]:
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            try:
                async with session.get(
                    url,
                    headers=self.headers,
                    params={"instruments": symbol},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status != 200:
                        return symbol, None
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
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        if df is not None and len(df) >= 2:
            fb = (df["high"] - df["low"]).mean()
            return float(fb) if pd.notna(fb) and fb > 0 else None
        return None
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    res = tr.rolling(period).mean().iloc[-1]
    if pd.notna(res) and res > 0:
        return float(res)
    fb = (df["high"] - df["low"]).mean()
    return float(fb) if pd.notna(fb) and fb > 0 else None

def compute_institutional_trend(
    closes: pd.Series,
    lookback: int = 20,
    threshold: float = 2.0,
) -> str:
    if closes is None or len(closes) < lookback:
        return "NEUTRE"
    y = closes.tail(lookback).values.astype(float)
    base = y[0]
    if base == 0 or not np.isfinite(base):
        return "NEUTRE"
    y_norm = y / base
    x = np.arange(len(y_norm), dtype=float)
    try:
        slope, intercept = np.polyfit(x, y_norm, 1)
    except (np.linalg.LinAlgError, ValueError):
        return "NEUTRE"
    residuals = y_norm - (slope * x + intercept)
    std_resid = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    if std_resid <= 0:
        return "NEUTRE"
    t_stat = slope / (std_resid / np.sqrt(len(x)))
    if t_stat > threshold:
        return "HAUSSIER"
    if t_stat < -threshold:
        return "BAISSIER"
    return "NEUTRE"

def detect_swing_pivots(
    df: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> Tuple[pd.Series, pd.Series]:
    if df is None or len(df) < 2 * 3 + 2 or atr_val is None or atr_val <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    n = 3
    prominence = atr_val * profile.pivot_prominence_atr

    highs = pd.Series(df["high"].values)
    lows = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)
    opens = pd.Series(df["open"].values)

    roll_high_left = highs.shift(1).rolling(n, min_periods=n).max()
    roll_low_left = lows.shift(1).rolling(n, min_periods=n).min()

    rev_high = highs.iloc[::-1].reset_index(drop=True)
    rev_low = lows.iloc[::-1].reset_index(drop=True)
    roll_high_right = (
        rev_high.shift(1).rolling(n, min_periods=n).max().iloc[::-1].reset_index(drop=True)
    )
    roll_low_right = (
        rev_low.shift(1).rolling(n, min_periods=n).min().iloc[::-1].reset_index(drop=True)
    )

    candle_range = (highs - lows).clip(lower=1e-10)
    body_top = pd.Series(np.maximum(opens.values, closes.values))
    body_bottom = pd.Series(np.minimum(opens.values, closes.values))
    upper_wick_pct = (highs - body_top) / candle_range
    lower_wick_pct = (body_bottom - lows) / candle_range

    wick_threshold = (
        profile.wick_threshold_intraday
        if timeframe.lower() in ("h4", "m15")
        else profile.wick_threshold_htf
    )

    sh_mask = (
        (highs > roll_high_left)
        & (highs > roll_high_right)
        & (upper_wick_pct >= wick_threshold)
    ).fillna(False)
    sl_mask = (
        (lows < roll_low_left)
        & (lows < roll_low_right)
        & (lower_wick_pct >= wick_threshold)
    ).fillna(False)

    roll_low_around = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
    roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()
    sh_mask = sh_mask & ((highs - roll_low_around) >= prominence)
    sl_mask = sl_mask & ((roll_high_around - lows) >= prominence)

    idx_highs = sh_mask[sh_mask].index.tolist()
    idx_lows = sl_mask[sl_mask].index.tolist()

    return (
        pd.Series(highs.values[idx_highs], index=idx_highs) if idx_highs else pd.Series(dtype=float),
        pd.Series(lows.values[idx_lows], index=idx_lows) if idx_lows else pd.Series(dtype=float),
    )

def agglomerative_1d_clustering(
    price_weight_pairs: List[tuple],
    bandwidth: float,
) -> List[List[tuple]]:
    if not price_weight_pairs or bandwidth <= 0:
        return [[pw] for pw in price_weight_pairs] if price_weight_pairs else []
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

def classify_zone_status(
    level: float,
    zone_type: str,
    df: pd.DataFrame,
    formation_idx: int,
    atr_val: float,
) -> str:
    if formation_idx >= len(df) - 1 or atr_val is None or atr_val <= 0:
        return "Vierge"
    tolerance = atr_val * 0.25

    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]
    if len(c_arr) == 0:
        return "Vierge"

    if zone_type == "Support":
        test_mask = (l_arr <= level + tolerance) & (c_arr > level - tolerance)
        break_mask = c_arr < level - tolerance
    else:
        test_mask = (h_arr >= level - tolerance) & (c_arr < level + tolerance)
        break_mask = c_arr > level + tolerance

    has_approach = bool(test_mask.any())
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

    second_break = (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level - tolerance)
    return "Consommee" if second_break.any() else "Role Reverse"

def compute_structural_score(strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int) -> float:
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    age_f = float(np.exp(-lam * age_r))
    return round((strength * tf_w * nb_tf) * age_f, 1)

def find_strong_sr_zones(
    df: pd.DataFrame,
    current_price: float,
    symbol: str,
    atr_val: Optional[float],
    timeframe: str,
    min_touches: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if atr_val is None or atr_val <= 0 or df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    profile = get_profile(symbol)
    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)

    if len(pivot_highs) + len(pivot_lows) < 3:
        dist = {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)
        pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        safe_cutoff = n_total - 3
        r_idx = [i for i in r_idx if i < safe_cutoff]
        s_idx = [i for i in s_idx if i < safe_cutoff]
        pivot_highs = (
            pd.Series(df["high"].values[r_idx], index=r_idx) if len(r_idx) else pd.Series(dtype=float)
        )
        pivot_lows = (
            pd.Series(df["low"].values[s_idx], index=s_idx) if len(s_idx) else pd.Series(dtype=float)
        )

    all_pivots = (
        [(float(p), int(i), (int(i) + 1e-6) / n_total) for i, p in pivot_highs.items()]
        + [(float(p), int(i), (int(i) + 1e-6) / n_total) for i, p in pivot_lows.items()]
    )
    if not all_pivots:
        return pd.DataFrame(), pd.DataFrame()

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
        if grp_weights_arr.sum() <= 0:
            continue
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
            "level": float(lvl), "strength": len(grp_pw),
            "age_bars": age, "status": status,
        })

    if not strong:
        return pd.DataFrame(), pd.DataFrame()

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
                new_status = max(
                    [prev["status"], z["status"]],
                    key=lambda s: STATUS_PRIORITY.get(s, 1),
                )
                merged[-1] = {
                    "level": new_lvl,
                    "strength": new_str,
                    "age_bars": min(prev["age_bars"], z["age_bars"]),
                    "status": new_status,
                }
            else:
                merged.append(z)

    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)
    df_zones["is_pivot"] = (np.abs(df_zones["level"] - current_price) / current_price * 100) <= 0.50
    return (
        df_zones[df_zones["level"] < current_price].copy(),
        df_zones[df_zones["level"] >= current_price].copy(),
    )

def detect_confluences(
    symbol: str,
    zones_dict: dict,
    current_price: float,
    bars_map: dict,
    confluence_threshold: Optional[float] = None,
) -> list:
    if not zones_dict or not current_price:
        return []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 1, "Consommee": 2}

    frames = []
    for tf, (sup, res) in zones_dict.items():
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            if df_z is None or df_z.empty:
                continue
            tmp = df_z[df_z["status"] != "Consommee"].copy()
            if tmp.empty:
                continue
            tmp = tmp.assign(
                tf=tf,
                type=tmp["is_pivot"].map({True: "Pivot", False: ztype}),
            )
            frames.append(tmp[["tf", "level", "strength", "age_bars", "status", "type", "is_pivot"]])

    if not frames:
        return []
    z_df = pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
    used = set()
    confluences = []

    profile = get_profile(symbol.replace("/", "_"))
    threshold = confluence_threshold if confluence_threshold is not None else profile.confluence_threshold_pct

    for z in z_df.itertuples():
        if z.Index in used or z.level <= 0:
            used.add(z.Index)
            continue

        similar = z_df[
            (np.abs(z_df["level"] - z.level) / z.level * 100 <= threshold)
            & (~z_df.index.isin(used))
        ]
        if len(similar) == 0:
            continue

        group_idx = similar.index.tolist()
        group_full = z_df.loc[group_idx]
        tfs = group_full["tf"].unique()

        if len(tfs) >= 2:
            used.update(group_full.index)
            sub_avg = group_full["level"].mean()

            group_full = group_full.assign(_dist_to_center=(group_full["level"] - sub_avg).abs())
            keep_idx = group_full.groupby("tf")["_dist_to_center"].idxmin().values
            group = group_full.loc[keep_idx].drop(columns=["_dist_to_center"])

            sub_nb_tf = group["tf"].nunique()
            sub_dist = abs(current_price - sub_avg) / current_price * 100

            _tf_w = group["tf"].map(TF_WEIGHT).fillna(1.0).values
            _totals = group["tf"].map(lambda t: bars_map.get(t, 500)).values.astype(float)
            _age_r = np.clip(group["age_bars"].values / np.maximum(_totals, 1), 0, 1)
            _lams = group["tf"].map(_TF_LAMBDA).fillna(1.5).values
            _age_f = np.exp(-_lams * _age_r)
            score = round(
                float((group["strength"].values * _tf_w * sub_nb_tf * _age_f).sum()),
                1,
            )
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
                "Actif": symbol,
                "Signal": sig,
                "Niveau": round(sub_avg, 5),
                "Type": ctype,
                "Timeframes": " + ".join(sorted(tfs)),
                "Nb TF": int(sub_nb_tf),
                "Force Totale": int(group["strength"].sum()),
                "Score": round(score, 1),
                "Statut": status,
                "Distance %": round(sub_dist, 3),
                "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else ""),
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
    missing_tfs: List[str] = field(default_factory=list)
    price_is_fallback: bool = False

def _make_row(z: dict, ztype: str, cp: float, atr_val: float, sym_d: str,
              tf_name: str, df_len: int, profile: InstrumentProfile) -> dict:
    dist = abs(cp - z["level"]) / cp * 100 if cp else 0.0
    dist_atr = f"{round(abs(cp - z['level']) / atr_val, 1)}x" if (atr_val and atr_val > 0) else "N/A"
    in_pdf = dist <= profile.pdf_max_dist_pct
    return {
        "Actif": sym_d,
        "Prix Actuel": f"{cp:.5f}" if cp else "N/A",
        "Type": ztype,
        "Niveau": f"{z['level']:.5f}",
        "Force": f"{z['strength']} touches",
        "Score (1TF)": compute_structural_score(z["strength"], 1, tf_name, z["age_bars"], df_len),
        "Statut": z["status"],
        "Dist. %": f"{dist:.2f}%",
        "Dist. ATR": dist_atr,
        "_dist_num": dist,
        "_in_pdf": in_pdf,
    }

async def run_institutional_scan(
    symbols: List[str],
    token: str,
    account_id: str,
    min_touches_ui: int,
) -> List[ScanResult]:
    client = AsyncOandaClient(token, account_id)
    async with aiohttp.ClientSession() as session:
        if not await client.initialize(session):
            raise OandaAuthError("Impossible de s'authentifier sur OANDA. Vérifiez vos secrets API.")

        sem = asyncio.Semaphore(15)

        price_tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
        prices_res = await asyncio.gather(*price_tasks)
        live_prices = {sym: p for sym, p in prices_res}

        candle_tasks = []
        for sym in symbols:
            for tf in _GRANULARITY_MAP.keys():
                candle_tasks.append(client.fetch_candles(session, sem, sym, tf))
        candles_res = await asyncio.gather(*candle_tasks)
        data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
        for sym, tf, df in candles_res:
            data_cube.setdefault(sym, {})[tf] = df

    results: List[ScanResult] = []
    for sym in symbols:
        try:
            profile = get_profile(sym)
            cp_live = live_prices.get(sym)
            cp = cp_live
            price_is_fallback = False
            sym_d = sym.replace("_", "/")

            rows = {"H4": None, "Daily": None, "Weekly": None}
            zones_d: Dict[str, tuple] = {}
            trends: Dict[str, str] = {}
            bars_map: Dict[str, int] = {}
            price_ctx = ""
            missing_tfs: List[str] = []

            for tf_k, tf_name in [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]:
                df = data_cube.get(sym, {}).get(tf_k)
                if df is None or df.empty:
                    missing_tfs.append(tf_name)
                    continue

                if not cp:
                    cp = float(df["close"].iloc[-1])
                    price_is_fallback = True

                bars_map[tf_name] = len(df)
                _lb, _th = {
                    "H4": (30, 2.0),
                    "Daily": (20, 2.0),
                    "Weekly": (10, 2.0),
                }.get(tf_name, (20, 2.0))
                trends[tf_name] = compute_institutional_trend(df["close"], lookback=_lb, threshold=_th)
                atr_val = compute_atr(df)

                if atr_val is None:
                    continue

                min_t = max(3, min_touches_ui) if tf_k == "h4" else max(2, min_touches_ui)
                sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
                zones_d[tf_name] = (sup, res)

                if tf_k == "daily":
                    parts = []
                    if not sup.empty:
                        s_near = sup[(sup["level"] < cp) & (abs(sup["level"] - cp) / cp * 100 <= 5.0)]
                        if not s_near.empty:
                            n_s = s_near.nlargest(1, "level").iloc[0]
                            d_s = abs(cp - n_s["level"]) / cp * 100
                            parts.append(f"{'SUR support' if d_s < 0.5 else 'S proche'}: {n_s['level']:.5f} (-{d_s:.2f}%)")
                    if not res.empty:
                        r_near = res[(res["level"] > cp) & (abs(res["level"] - cp) / cp * 100 <= 5.0)]
                        if not r_near.empty:
                            n_r = r_near.nsmallest(1, "level").iloc[0]
                            d_r = abs(cp - n_r["level"]) / cp * 100
                            parts.append(f"{'SUR resistance' if d_r < 0.5 else 'R proche'}: {n_r['level']:.5f} (+{d_r:.2f}%)")
                    price_ctx = "  |  ".join(parts) if parts else "Zone intermediaire"

                tf_r = (
                    [_make_row(z, "PIVOT" if z.get("is_pivot") else "Support", cp, atr_val, sym_d, tf_name, len(df), profile)
                     for _, z in sup.iterrows()]
                    + [_make_row(z, "PIVOT" if z.get("is_pivot") else "Resistance", cp, atr_val, sym_d, tf_name, len(df), profile)
                       for _, z in res.iterrows()]
                )

                seen = set()
                uniq = []
                for r in tf_r:
                    key = (r["Niveau"], r["Type"])
                    if key not in seen:
                        seen.add(key)
                        uniq.append(r)
                if uniq:
                    rows[tf_name] = uniq

            _sup_levels: List[float] = []
            for _tf_n, (_s, _r) in zones_d.items():
                if _s is not None and not _s.empty and "level" in _s.columns:
                    _sup_levels.extend(_s["level"].tolist())
            _daily_df = data_cube.get(sym, {}).get("daily")
            _last_close = (
                float(_daily_df["close"].iloc[-1]) if (_daily_df is not None and not _daily_df.empty) else None
            )
            _anomaly = flag_data_anomaly(sym, cp, _sup_levels, last_candle_close=_last_close)

            if price_is_fallback and cp is not None:
                pf_msg = f"Prix live indisponible — utilisation du dernier close ({cp:.5f})"
                _anomaly = f"{_anomaly} | {pf_msg}" if _anomaly else pf_msg

            results.append(ScanResult(
                sym, rows, zones_d, cp, trends, bars_map,
                price_context=price_ctx, anomaly=_anomaly,
                missing_tfs=missing_tfs, price_is_fallback=price_is_fallback,
            ))
        except Exception as e:
            logging.exception("Scan error for %s", sym)
            results.append(ScanResult(sym, {}, {}, None, {}, {}, scan_error=f"{type(e).__name__}: {e}"))

    return results

# ==============================================================================
# [ LAYER 5: EXPORTERS & UTILS ]
# ==============================================================================
_ACCENT_MAP = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ', 'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
_EMOJI_MAP = [('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'), ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'),
              ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]'), ('📈', ''), ('📉', ''), ('✅', '[OK]'),
              ('❌', '[X]'), ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'),
              ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]')]

def _safe_pdf_str(text: str) -> str:
    text = str(text).translate(_ACCENT_MAP)
    for e, r in _EMOJI_MAP:
        text = text.replace(e, r)
    return text

def _sanitize_traceback(tb: str, sensitive_values: list) -> str:
    if not tb:
        return tb
    for val in sensitive_values:
        if not val or not isinstance(val, str) or len(val) < 4:
            continue
        pattern = re.escape(val)
        tb = re.sub(pattern, "***REDACTED***", tb)
    return tb

_INTERNAL_COLS = ["_dist_num", "_in_pdf"]

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=_INTERNAL_COLS, errors="ignore")

def _sym_display(sym: str) -> str:
    return sym.replace("_", "/")

def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    if current_price is None or current_price <= 0:
        return "Prix indisponible ou non valide"
    profile = get_profile(symbol)
    messages = []
    if not profile.skip_ratio_check and len(support_levels) >= 3:
        median_sup = float(np.median(support_levels))
        if median_sup > 0 and median_sup > 0.01 * current_price:
            ratio = current_price / median_sup
            if ratio > 3.0:
                messages.append(
                    f"Prix {current_price:.2f} = {ratio:.1f}x la mediane des supports "
                    f"({median_sup:.2f}) - donnees a verifier"
                )
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > profile.max_live_vs_close_pct:
            messages.append(
                f"Prix live {current_price:.2f} s'ecarte de {dev:.1f}% "
                f"du dernier close ({last_candle_close:.2f}) — seuil profil {profile.max_live_vs_close_pct}%"
            )
    return " | ".join(messages) if messages else None

def get_price_context(current_price, supports, resistances, max_dist_pct: float = 5.0):
    if not current_price or current_price <= 0:
        return "Prix indisponible"
    parts = []
    if supports is not None and not supports.empty:
        sup_nearby = supports[
            (supports["level"] < current_price)
            & (abs(supports["level"] - current_price) / current_price * 100 <= max_dist_pct)
        ]
        if not sup_nearby.empty:
            nearest_sup = sup_nearby.nlargest(1, "level").iloc[0]
            dist_s = abs(current_price - nearest_sup["level"]) / current_price * 100
            tag = "SUR support" if dist_s < 0.5 else "S proche"
            parts.append(f"{tag}: {nearest_sup['level']:.5f} (-{dist_s:.2f}%)")
    if resistances is not None and not resistances.empty:
        res_nearby = resistances[
            (resistances["level"] > current_price)
            & (abs(resistances["level"] - current_price) / current_price * 100 <= max_dist_pct)
        ]
        if not res_nearby.empty:
            nearest_res = res_nearby.nsmallest(1, "level").iloc[0]
            dist_r = abs(current_price - nearest_res["level"]) / current_price * 100
            tag = "SUR resistance" if dist_r < 0.5 else "R proche"
            parts.append(f"{tag}: {nearest_res['level']:.5f} (+{dist_r:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"

def _parse_price_context_obstacles(ctx_str: str, current_price: float) -> dict:
    result: dict = {"nearest_support": None, "nearest_resistance": None}
    if not ctx_str or ctx_str == "Zone intermediaire" or not current_price:
        return result
    pat = re.compile(r"(SUR support|S proche|SUR resistance|R proche):\s*([\d.]+)\s*\(([+-][\d.]+)%\)")
    for m in pat.finditer(ctx_str):
        tag, level_str, dist_str = m.group(1), m.group(2), m.group(3)
        try:
            lvl = float(level_str)
            dist = float(dist_str)
        except ValueError:
            continue
        entry = {"level": lvl, "distance_pct": dist, "on_level": abs(dist) < 0.5}
        if tag in ("SUR support", "S proche"):
            result["nearest_support"] = entry
        else:
            result["nearest_resistance"] = entry
    return result

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
        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
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
            sym = _safe_pdf_str(s.get('symbol', ''))
            t_h4 = _safe_pdf_str(s.get('trend_h4', 'N/A'))
            t_d = _safe_pdf_str(s.get('trend_daily', 'N/A'))
            t_w = _safe_pdf_str(s.get('trend_weekly', 'N/A'))
            ctx = _safe_pdf_str(s.get('price_context', ''))
            self.set_font('Helvetica', 'B', 8)
            self.cell(0, 5,
                      _safe_pdf_str(f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"),
                      border=0, new_x='LMARGIN', new_y='NEXT')
            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}", border=0, new_x='LMARGIN', new_y='NEXT')
            top = s.get('top_zones', [])
            self.set_font('Helvetica', '', 7)
            if top:
                for z in top:
                    sig = str(z.get('Signal', ''))
                    niv = str(z.get('Niveau', ''))
                    dist = str(z.get('Distance %', ''))
                    sc = str(z.get('Score', ''))
                    tfs = str(z.get('Timeframes', ''))
                    ale = str(z.get('Alerte', ''))
                    txt = _safe_pdf_str(
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
        cols = [c for c in col_widths if c in df.columns]
        total_w = sum(col_widths[c] for c in cols)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start = self.l_margin + max(0, (usable_w - total_w) / 2)

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
        return _clean_df(df[df["_in_pdf"]].copy()).reset_index(drop=True)
    if "Actif" in df.columns and "_dist_num" in df.columns:
        def _threshold_for(actif: str) -> float:
            return get_profile(actif.replace("/", "_")).pdf_max_dist_pct
        thresholds = df["Actif"].apply(_threshold_for)
        mask = df["_dist_num"] <= thresholds
        return _clean_df(df[mask].copy()).reset_index(drop=True)
    if "Dist. %" in df.columns:
        def _to_f(s):
            try:
                return float(str(s).replace("%", ""))
            except (ValueError, TypeError):
                return 999.0
        df = df[df["Dist. %"].apply(_to_f) <= 8.0].copy()
    return _clean_df(df).reset_index(drop=True)

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
        clean_conf = strip_emojis_df(_clean_df(confluences_df.copy()))
        if "Score" in clean_conf.columns:
            clean_conf = clean_conf.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_conf)
        pdf.ln(10)
    title_map = {
        'H4': 'Analyse 4 Heures (H4)',
        'Daily': 'Analyse Journaliere (Daily)',
        'Weekly': 'Analyse Hebdomadaire (Weekly)',
    }
    for tf_key, df in results_dict.items():
        if df is None or (hasattr(df, 'empty') and df.empty):
            continue
        pdf.chapter_title(title_map.get(tf_key, tf_key))
        clean_df = strip_emojis_df(df.copy())
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
            d["Section"] = "TF_ROWS"
            all_dfs.append(d)
    if not all_dfs:
        return b""
    buf = BytesIO()
    pd.concat(all_dfs, ignore_index=True).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()

# ==============================================================================
# [ LAYER 6: STREAMLIT UI LAYER ]
# ==============================================================================
CONFLUENCE_THRESHOLD_MAP = {
    "US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2,
    "XAU_USD": 1.5,
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
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        if not access_token or not str(access_token).strip():
            raise ValueError("OANDA_ACCESS_TOKEN est vide")
        if not account_id or not str(account_id).strip():
            raise ValueError("OANDA_ACCOUNT_ID est vide")
        st.success("Secrets chargés ✓")
    except (KeyError, ValueError) as e:
        access_token, account_id = None, None
        st.error(f"Secrets OANDA invalides : {e}")
    except (FileNotFoundError, AttributeError) as e:
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
            "Actifs spécifiques :", options=ALL_SYMBOLS, default=default_sel,
        )

    st.header("3. Paramètres Export LLM")
    llm_max_dist_sidebar = st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    llm_min_score_sidebar = st.slider("Score min JSON/LLM", 40, 300, 60, 10, key="llm_min_score")
    llm_statuts_sidebar = st.multiselect(
        "Statuts autorisés (brief LLM)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    st.divider()
    st.header("4. Paramètres de Détection")
    min_touches = st.slider("Force minimale H4 (touches)", 2, 10, 3, 1)
    st.caption("H4 utilise ce seuil | Daily/Weekly utilisent max(2, valeur)")
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 1.0, 0.1)
    _overridden = [s.replace("_USD", "").replace("_EUR", "") for s in CONFLUENCE_THRESHOLD_MAP]
    st.caption(
        f"⚠️ Seuil ignoré pour : {', '.join(_overridden)} "
        f"(valeurs fixes : {list(CONFLUENCE_THRESHOLD_MAP.values())})"
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) - filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
    )

if scan_button and symbols_to_scan and not st.session_state.get("scanning", False):
    st.session_state.pop("scan_results", None)
    st.session_state["scanning"] = True
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
        all_zones_map = {}
        prices_map = {}
        trends_map = {}
        anomalies_map = {}
        scan_errors = {}
        bars_map_global = {}
        missing_tfs_map: Dict[str, List[str]] = {}
        price_fallback_map: Dict[str, bool] = {}

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
                    text=f"Post-traitement… ({idx + 1}/{total}) {sym_label}",
                )
                if result.scan_error:
                    scan_errors[_sym_display(result.symbol)] = result.scan_error
                    continue
                all_zones_map[result.symbol] = result.zones
                prices_map[result.symbol] = result.price
                trends_map[result.symbol] = result.trends
                bars_map_global[result.symbol] = result.bars_map
                price_fallback_map[result.symbol] = result.price_is_fallback
                if result.anomaly:
                    anomalies_map[_sym_display(result.symbol)] = result.anomaly
                if result.missing_tfs:
                    missing_tfs_map[_sym_display(result.symbol)] = result.missing_tfs
                for tf_cap, tf_rows in result.rows.items():
                    if tf_rows:
                        if tf_cap == "H4":
                            results_h4.extend(tf_rows)
                        elif tf_cap == "Daily":
                            results_daily.extend(tf_rows)
                        elif tf_cap == "Weekly":
                            results_weekly.extend(tf_rows)

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

        st.info("Analyse des confluences…")
        all_confluences = []
        for sym in symbols_to_scan:
            if _sym_display(sym) in scan_errors:
                continue
            cp = prices_map.get(sym)
            zones_clean = {
                k: v for k, v in all_zones_map.get(sym, {}).items()
                if not k.startswith("_")
            }
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
            sym_d = _sym_display(sym)
            trends = trends_map.get(sym, {})
            cp = prices_map.get(sym)
            top_zones = []
            if not conf_full.empty and "Actif" in conf_full.columns and sym_d in conf_full["Actif"].values:
                ac = conf_full[conf_full["Actif"] == sym_d].copy()
                top_zones = ac.sort_values("Score", ascending=False).head(3).to_dict("records")
            price_ctx = ""
            d_zones = all_zones_map.get(sym, {})
            if "Daily" in d_zones and cp:
                sup_d, res_d = d_zones["Daily"]
                price_ctx = get_price_context(cp, sup_d, res_d)
            summaries.append({
                "symbol": sym_d,
                "trend_h4": trends.get("H4", "NEUTRE"),
                "trend_daily": trends.get("Daily", "NEUTRE"),
                "trend_weekly": trends.get("Weekly", "NEUTRE"),
                "price_context": price_ctx,
                "top_zones": top_zones,
                "current_price": cp,
                "missing_tfs": missing_tfs_map.get(sym_d, []),
                "price_is_fallback": price_fallback_map.get(sym, False),
            })

        df_h4 = pd.DataFrame(results_h4)
        df_daily = pd.DataFrame(results_daily)
        df_wk = pd.DataFrame(results_weekly)
        rep_dict = {
            "H4": _apply_pdf_filter(df_h4),
            "Daily": _apply_pdf_filter(df_daily),
            "Weekly": _apply_pdf_filter(df_wk),
        }

        st.session_state["scan_results"] = {
            "df_h4": df_h4, "df_daily": df_daily, "df_weekly": df_wk,
            "conf_full": conf_full, "report_dict": rep_dict,
            "summaries": summaries, "anomalies": anomalies_map,
            "scan_errors": scan_errors, "max_dist": max_dist_filter,
            "missing_tfs_map": missing_tfs_map,
        }
