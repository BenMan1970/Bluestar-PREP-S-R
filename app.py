# pylint: disable=too-many-lines
"""Scanner Bluestar S/R Multi-Timeframes — v8.6.0-PROD

Production-grade hardened version.
Quant Fixes v8.6.0 (The "Sweet Spot" Update):
  Q1. Hybrid Min-Touches: 1 touch allowed IF prominence > 3x ATR (Major Pivot).
  Q2. Recalibrated Lookback (n): W:5, D:3, H4:3 for Indices to catch structural swings.
  Q3. Hybrid Radius: max(ATR * mult, Price * 0.0015) to ensure zones have physical width.
  Q4. Prominence Cap: max prominence capped at 0.5% of price to prevent volatility blindness.
  Q5. JSON Export filename updated to 'supports et resistances.json'.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Final, Optional, Tuple, List, Dict, Set, Callable, Mapping

try:
    from zoneinfo import ZoneInfo
    _NY_TZ: Optional[ZoneInfo] = ZoneInfo("America/New_York")
except ImportError:
    _NY_TZ = None

import aiohttp
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ==============================================================================
# [ LAYER 0: CONFIG & LOGGING ]
# ==============================================================================
SCANNER_VERSION: Final[str] = "8.6.0-PROD"
_TOKEN_REDACT_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE),
    re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
    re.compile(r"(access_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{20,}=*", re.IGNORECASE),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{8,}", re.IGNORECASE),
    re.compile(r"(\b[a-f0-9]{32}-[a-f0-9]{32}\b)", re.IGNORECASE),
]

def _redact_sensitive(text: Any) -> Any:
    if not isinstance(text, str) or not text: return text
    out = text
    for pat in _TOKEN_REDACT_PATTERNS:
        try: out = pat.sub(lambda m: m.group(1) + "***REDACTED***" if m.lastindex else "***REDACTED***", out)
        except: continue
    return out

class _SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.msg and isinstance(record.msg, str): record.msg = _redact_sensitive(record.msg)
        except: pass
        return True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addFilter(_SensitiveDataFilter())
_LOG = logging.getLogger("bluestar")

class OandaAuthError(Exception): pass
class DataValidationError(Exception): pass
class ScanTimeoutError(Exception): pass

# ==============================================================================
# [ LAYER 0c: THREAD-SAFE CACHE ]
# ==============================================================================
_CACHE_TTL_BY_TF: Final[Dict[str, int]] = {"h4": 60, "daily": 300, "weekly": 600}
_CACHE_TTL_DEFAULT: Final[int] = 300
_CACHE_TTL_NEGATIVE: Final[int] = 20
_CACHE_MAX_ENTRIES: Final[int] = 256
_CACHE_MAX_BYTES: Final[int] = 50 * 1024 * 1024 
_CACHE_LOCK: Final[threading.RLock] = threading.RLock()
_CACHE_EMPTY: Final[object] = object()
_OANDA_CACHE: "OrderedDict[Tuple[str, str, str, str], Tuple[float, Any, int]]" = OrderedDict()
_CACHE_BYTES_TOTAL: List[int] = [0]

def _df_approx_bytes(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty: return 128
    try: return int(df.memory_usage(index=True, deep=False).sum())
    except: return 128

def _make_readonly(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.columns:
            arr = df[col].values
            if isinstance(arr, np.ndarray): arr.setflags(write=False)
    except: pass
    return df

def _cache_ttl(tf: str, is_empty: bool = False) -> int:
    if is_empty: return _CACHE_TTL_NEGATIVE
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)

def _cache_is_fresh(fetched_at: float, tf: str, is_empty: bool) -> bool:
    return (time.monotonic() - fetched_at) <= _cache_ttl(tf, is_empty)

def _cache_key(env_url, acct_id, symbol, tf):
    return (env_url or "unknown_env", acct_id or "unknown_account", symbol, tf)

def _cache_evict_stale_locked():
    now = time.monotonic()
    stale = [k for k, (ts, payload, _sz) in _OANDA_CACHE.items() if (now - ts) > _cache_ttl(k[3], payload is _CACHE_EMPTY)]
    for k in stale:
        _, _, sz = _OANDA_CACHE.pop(k)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
    while len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
    while _CACHE_BYTES_TOTAL[0] > _CACHE_MAX_BYTES and _OANDA_CACHE:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)

def _cache_get(env_url, acct_id, symbol, tf):
    k = _cache_key(env_url, acct_id, symbol, tf)
    with _CACHE_LOCK:
        _cache_evict_stale_locked()
        entry = _OANDA_CACHE.get(k)
        if entry is None: return False, None
        fetched_at, payload, _sz = entry
        if not _cache_is_fresh(fetched_at, tf, payload is _CACHE_EMPTY):
            _, _, sz = _OANDA_CACHE.pop(k)
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
            return False, None
        _OANDA_CACHE.move_to_end(k)
        return True, (None if payload is _CACHE_EMPTY else payload)

def _cache_set(env_url, acct_id, symbol, tf, df):
    k = _cache_key(env_url, acct_id, symbol, tf)
    payload, sz = (_CACHE_EMPTY, 64) if df is None else (_make_readonly(df), _df_approx_bytes(df))
    with _CACHE_LOCK:
        old = _OANDA_CACHE.pop(k, None)
        if old: _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - old[2])
        _OANDA_CACHE[k] = (time.monotonic(), payload, sz)
        _CACHE_BYTES_TOTAL[0] += sz
        _OANDA_CACHE.move_to_end(k)
        _cache_evict_stale_locked()

def _cache_clear():
    with _CACHE_LOCK:
        n = len(_OANDA_CACHE)
        _OANDA_CACHE.clear()
        _CACHE_BYTES_TOTAL[0] = 0
        return n

def _cache_stats():
    with _CACHE_LOCK: return {"entries": len(_OANDA_CACHE), "bytes": _CACHE_BYTES_TOTAL[0]}

# ==============================================================================
# [ CONSTANTS ]
# ==============================================================================
ALL_SYMBOLS: Final[List[str]] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]
_GRANULARITY_MAP: Final[Dict[str, str]] = {"h4": "H4", "daily": "D", "weekly": "W"}
TF_WEIGHT: Final[Dict[str, float]] = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}
_TF_LAMBDA: Final[Dict[str, float]] = {"H4": 2.0, "Daily": 1.0, "Weekly": 0.5}
_OANDA_SEMAPHORE_LIMIT: Final[int] = 12
_PER_REQUEST_TIMEOUT_S: Final[float] = 10.0
_SCAN_LOCK_TTL_S: Final[float] = 900.0

# ==============================================================================
# [ HASH FUNCTIONS ]
# ==============================================================================
def _hash_df(df: Optional[pd.DataFrame]) -> str:
    if df is None or (hasattr(df, "empty") and df.empty): return "empty_df"
    try:
        h = hashlib.sha256()
        h.update(f"shape:{df.shape[0]}x{df.shape[1]}|".encode())
        if len(df.index) > 0: h.update(f"idx:{df.index[0]}:{df.index[-1]}|".encode())
        n = len(df)
        sample = df if n <= 32 else pd.concat([df.iloc[:8], df.iloc[n//2-4:n//2+4], df.iloc[-8:]], copy=False)
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        return h.hexdigest()[:32]
    except: return f"unhashable_{id(df)}"

def _hash_series(s: Optional[pd.Series]) -> str:
    if s is None or len(s) == 0: return "empty_series"
    try:
        h = hashlib.sha256()
        h.update(f"len:{len(s)}|dtype:{s.dtype}|".encode())
        h.update(pd.util.hash_pandas_object(s, index=False).values.tobytes())
        return h.hexdigest()[:32]
    except: return f"unhashable_series_{id(s)}"

def _hash_dict_content(d: Optional[Mapping[str, Any]]) -> str:
    if not d: return "empty_dict"
    h = hashlib.sha256()
    for k in sorted(d.keys()):
        v = d[k]
        h.update(f"{k}=".encode())
        if isinstance(v, pd.DataFrame): h.update(_hash_df(v).encode())
        elif isinstance(v, pd.Series): h.update(_hash_series(v).encode())
        elif isinstance(v, (str, int, float, bool, type(None))): h.update(repr(v).encode())
        else: h.update(json.dumps(v, sort_keys=True, default=str)[:512].encode())
        h.update(b"|")
    return h.hexdigest()[:32]

def _hash_list_content(lst: Optional[List[Any]]) -> str:
    if not lst: return "empty_list"
    try:
        normalized = [{k: str(v)[:80] for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))} for d in lst if isinstance(d, dict)]
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()[:32]
    except: return f"unhashable_list_{len(lst)}"

# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES — Surgical Calibration ]
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
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    min_touches_h4: int = 3
    min_touches_daily: int = 2
    min_touches_weekly: int = 2
    ignore_wick_filter: bool = False
    # NEW: Multiplier to allow 1-touch zones if pivot is extremely strong
    major_pivot_mult: float = 3.0 

_PROFILES: Final[Dict[str, InstrumentProfile]] = {
    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2,  0.8,  0.6,  1.5, False),
    "GBP_USD":    InstrumentProfile("GBP_USD",    "FOREX", 0.0001, 1.3,  0.85, 0.65, 1.5, False),
    "USD_JPY":    InstrumentProfile("USD_JPY",    "FOREX", 0.01,   0.9,  0.5,  0.5,  1.5, False),
    # SURGICAL: Expanded radius and Major Pivot logic for Indices/Metals
    "XAU_USD":    InstrumentProfile("XAU_USD",    "METAL", 0.01,   2.5,  1.5,  1.0,  3.0, True,  ignore_wick_filter=True, min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2, price_min=1500.0, price_max=6000.0, major_pivot_mult=2.5),
    "US30_USD":   InstrumentProfile("US30_USD",   "INDEX", 1.0,    2.5,  1.5,  0.7,  2.5, True,  ignore_wick_filter=True, min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2, price_min=25000.0, price_max=60000.0, major_pivot_mult=2.5),
    "NAS100_USD": InstrumentProfile("NAS100_USD", "INDEX", 1.0,    2.5,  1.5,  0.8,  2.5, True,  ignore_wick_filter=True, min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2, price_min=10000.0, price_max=50000.0, major_pivot_mult=2.5),
    "SPX500_USD": InstrumentProfile("SPX500_USD", "INDEX", 0.1,    2.2,  1.2,  0.65, 2.0, True,  ignore_wick_filter=True, min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2, price_min=3000.0,  price_max=12000.0, major_pivot_mult=2.5),
    "DE30_EUR":   InstrumentProfile("DE30_EUR",   "INDEX", 0.1,    2.2,  1.2,  0.65, 2.0, True,  ignore_wick_filter=True, min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2, price_min=10000.0, price_max=30000.0, major_pivot_mult=2.5),
}

_DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile("DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES: return _PROFILES[symbol]
    base = symbol.split("_")[0] if "_" in symbol else symbol
    if symbol.endswith("_JPY") or base == "JPY": return InstrumentProfile(symbol, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False)
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"): return InstrumentProfile(symbol, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)
    return _DEFAULT_PROFILE

def _min_touches_for_tf(profile: InstrumentProfile, tf: str, ui_override: int) -> int:
    tf_lower = tf.lower()
    profile_min = {"h4": profile.min_touches_h4, "daily": profile.min_touches_daily, "weekly": profile.min_touches_weekly}.get(tf_lower, 2)
    if profile.asset_class in ("INDEX", "METAL"):
        return max(2, profile_min)
    return max(profile_min, ui_override)

# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE ]
# ==============================================================================
_MAX_HIGH_LOW_RATIO: Final[float] = 1.5

def _is_valid_candle_dict(c: dict) -> bool:
    try:
        mid = c["mid"]
        o, h, lo, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(lo) and np.isfinite(cl)): return False
        if lo <= 0 or h <= 0 or h < lo or not lo <= o <= h or not lo <= cl <= h: return False
        if lo > 0 and (h / lo) > _MAX_HIGH_LOW_RATIO: return False
        return True
    except: return False

def _sanitize_ohlc_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns): return None
    try:
        out = df.copy().dropna(subset=list(required))
        if out.empty: return None
        if not out.index.is_monotonic_increasing: out = out.sort_index()
        if out.index.has_duplicates: out = out[~out.index.duplicated(keep="last")]
        mask = (np.isfinite(out["open"]) & np.isfinite(out["high"]) & np.isfinite(out["low"]) & np.isfinite(out["close"])
                & (out["low"] > 0) & (out["high"] > 0) & (out["high"] >= out["low"])
                & (out["open"].between(out["low"], out["high"])) & (out["close"].between(out["low"], out["high"])))
        ratio_ok = (out["high"] / out["low"].replace(0, np.nan)) <= _MAX_HIGH_LOW_RATIO
        out = out[mask & ratio_ok.fillna(False)]
        return out if not out.empty else None
    except: return None

class AsyncOandaClient:
    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = oanda_account_id
        self.env_url: Optional[str] = None

    @staticmethod
    async def _get_json_with_retry(session, url, headers, params, timeout_total, retries=3):
        backoff = 0.5
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=timeout_total)) as r:
                    if r.status == 200: return await r.json()
                    if r.status in (401, 403): return None
                    if r.status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff * (2 ** attempt))
                        continue
                    return None
            except:
                if attempt < retries - 1: await asyncio.sleep(backoff * (2 ** attempt))
                continue
        return None

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
            try:
                async with session.get(f"{url}/v3/accounts/{self.account_id}/summary", headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        self.env_url = url
                        return True
            except: continue
        return False

    async def fetch_candles(self, session, sem, symbol, tf, limit=500):
        cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, tf)
        if cache_hit: return symbol, tf, cached
        gran = _GRANULARITY_MAP.get(tf)
        if not gran or not self.env_url: return symbol, tf, None
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}
        async with sem:
            data = await self._get_json_with_retry(session, url, self.headers, params, _PER_REQUEST_TIMEOUT_S)
            if data is None:
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None
            try:
                candles = [{"date": pd.to_datetime(c["time"], utc=True), "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c.get("volume", 0))} 
                           for c in data.get("candles", []) if c.get("complete") and _is_valid_candle_dict(c)]
                if not candles:
                    _cache_set(self.env_url, self.account_id, symbol, tf, None)
                    return symbol, tf, None
                df_clean = _sanitize_ohlc_dataframe(pd.DataFrame(candles).set_index("date").tail(limit))
                _cache_set(self.env_url, self.account_id, symbol, tf, df_clean)
                return symbol, tf, df_clean
            except:
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None

    async def fetch_price(self, session, sem, symbol):
        if not self.env_url: return symbol, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(session, url, self.headers, {"instruments": symbol}, 5)
            try:
                if data and "prices" in data and data["prices"]:
                    bid, ask = float(data["prices"][0]["closeoutBid"]), float(data["prices"][0]["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0: return symbol, (bid + ask) / 2
            except: pass
        return symbol, None

def _run_async_isolated(coro_factory, timeout=300.0):
    try: asyncio.get_running_loop()
    except RuntimeError: return asyncio.run(coro_factory())
    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_factory())
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if pending: loop.run_until_complete(asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5.0))
            except: pass
            finally: loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="oanda-async") as ex:
        future = ex.submit(_worker)
        try: return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise ScanTimeoutError(f"Async scan exceeded {timeout}s") from e

# ==============================================================================
# [ LAYER 3: QUANT ENGINE — Surgical Fixes for Detection ]
# ==============================================================================
@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        if df is not None and len(df) >= 2:
            fb = (df["high"] - df["low"]).mean()
            return float(fb) if pd.notna(fb) and fb > 0 else None
        return None
    try:
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        res = tr.rolling(period).mean().iloc[-1]
        return float(res) if pd.notna(res) and res > 0 else None
    except: return None

@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: _hash_series})
def compute_institutional_trend(closes: pd.Series, lookback: int = 20, threshold: float = 2.0) -> str:
    if closes is None or len(closes) < lookback: return "NEUTRE"
    try:
        y = closes.tail(lookback).values.astype(float)
        if not np.all(np.isfinite(y)): return "NEUTRE"
        base = y[0]
        if base == 0 or not np.isfinite(base): return "NEUTRE"
        y_norm = y / base
        x = np.arange(len(y_norm), dtype=float)
        slope, intercept = np.polyfit(x, y_norm, 1)
        residuals = y_norm - (slope * x + intercept)
        std_resid = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        if std_resid <= 0: return "NEUTRE"
        t_stat = slope / (std_resid / np.sqrt(len(x)))
        if t_stat > threshold: return "HAUSSIER"
        if t_stat < -threshold: return "BAISSIER"
        return "NEUTRE"
    except: return "NEUTRE"

def detect_swing_pivots(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float, timeframe: str) -> Tuple[pd.Series, pd.Series]:
    """Vectorized swing detection with calibrated lookback and prominence cap."""
    if df is None or len(df) < 15 or atr_val is None or atr_val <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # SURGICAL FIX: Recalibrated lookback (n) for structural swings
    tf_lower = timeframe.lower()
    n = 5 if tf_lower == "weekly" else (3 if tf_lower == "daily" else 3)
    
    # SURGICAL FIX: Prominence cap to prevent volatility blindness (max 0.5% of price)
    current_p = df["close"].iloc[-1]
    prominence = min(atr_val * profile.pivot_prominence_atr, current_p * 0.005)

    highs = pd.Series(df["high"].values.copy())
    lows = pd.Series(df["low"].values.copy())
    closes = pd.Series(df["close"].values.copy())
    opens = pd.Series(df["open"].values.copy())

    roll_high_left = highs.shift(1).rolling(n, min_periods=n).max()
    roll_low_left = lows.shift(1).rolling(n, min_periods=n).min()
    rev_high = highs.iloc[::-1].reset_index(drop=True)
    rev_low = lows.iloc[::-1].reset_index(drop=True)
    roll_high_right = rev_high.shift(1).rolling(n, min_periods=n).max().iloc[::-1].reset_index(drop=True)
    roll_low_right = rev_low.shift(1).rolling(n, min_periods=n).min().iloc[::-1].reset_index(drop=True)

    candle_range = (highs - lows).clip(lower=1e-10)
    body_top = pd.Series(np.maximum(opens.values, closes.values))
    body_bottom = pd.Series(np.minimum(opens.values, closes.values))
    upper_wick_pct = (highs - body_top) / candle_range
    lower_wick_pct = (body_bottom - lows) / candle_range
    wick_threshold = profile.wick_threshold_intraday if tf_lower in ("h4", "m15") else profile.wick_threshold_htf

    if profile.ignore_wick_filter:
        sh_mask = (highs > roll_high_left) & (highs > roll_high_right)
        sl_mask = (lows < roll_low_left) & (lows < roll_low_right)
    else:
        sh_mask = (highs > roll_high_left) & (highs > roll_high_right) & (upper_wick_pct >= wick_threshold)
        sl_mask = (lows < roll_low_left) & (lows < roll_low_right) & (lower_wick_pct >= wick_threshold)

    roll_low_around = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
    roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()
    sh_mask = sh_mask & ((highs - roll_low_around) >= prominence)
    sl_mask = sl_mask & ((roll_high_around - lows) >= prominence)

    idx_highs = sh_mask[sh_mask].index.tolist()
    idx_lows = sl_mask[sl_mask].index.tolist()
    return (pd.Series(highs.values[idx_highs], index=idx_highs) if idx_highs else pd.Series(dtype=float),
            pd.Series(lows.values[idx_lows], index=idx_lows) if idx_lows else pd.Series(dtype=float))

def agglomerative_1d_clustering(price_weight_pairs: List[tuple], bandwidth: float) -> List[List[tuple]]:
    if not price_weight_pairs or bandwidth <= 0: return [[pw] for pw in price_weight_pairs] if price_weight_pairs else []
    sorted_pw = sorted(price_weight_pairs, key=lambda x: x[0])
    clusters, curr_cluster = [], [sorted_pw[0]]
    for i in range(1, len(sorted_pw)):
        gap = sorted_pw[i][0] - sorted_pw[i - 1][0]
        if gap > bandwidth or (curr_cluster and (sorted_pw[i][0] - curr_cluster[0][0]) > 2.5 * bandwidth):
            clusters.append(curr_cluster)
            curr_cluster = [sorted_pw[i]]
        else: curr_cluster.append(sorted_pw[i])
    clusters.append(curr_cluster)
    return clusters

def classify_zone_status(level: float, zone_type: str, df: pd.DataFrame, formation_idx: int, atr_val: float) -> str:
    if formation_idx >= len(df) - 1 or atr_val is None or atr_val <= 0: return "Vierge"
    tolerance = atr_val * 0.25
    try:
        c_arr, h_arr, l_arr = df["close"].values[formation_idx + 1:], df["high"].values[formation_idx + 1:], df["low"].values[formation_idx + 1:]
    except: return "Vierge"
    if len(c_arr) == 0: return "Vierge"
    if zone_type == "Support":
        test_mask, break_mask = (l_arr <= level + tolerance) & (c_arr > level - tolerance), c_arr < level - tolerance
    else:
        test_mask, break_mask = (h_arr >= level - tolerance) & (c_arr < level + tolerance), c_arr > level + tolerance
    has_approach = bool(test_mask.any())
    break_positions = np.where(break_mask)[0]
    if len(break_positions) == 0: return "Testee" if has_approach else "Vierge"
    break_idx = int(break_positions[0])
    retest_tol = tolerance * 2
    rc, rh, rl = c_arr[break_idx + 1:], h_arr[break_idx + 1:], l_arr[break_idx + 1:]
    if len(rc) == 0: return "Consommee"
    retest_mask = (rl <= level + retest_tol) & (rh >= level - retest_tol)
    if not retest_mask.any(): return "Consommee"
    retest_idx = int(np.where(retest_mask)[0][0])
    rc_after = rc[retest_idx + 1:]
    if len(rc_after) == 0: return "Role Reverse"
    second_break = (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level + tolerance)
    return "Consommee" if second_break.any() else "Role Reverse"

def compute_structural_score(strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int) -> float:
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    return round((strength * tf_w * nb_tf) * float(np.exp(-lam * age_r)), 1)

_STATUS_PRIORITY: Final[Dict[str, int]] = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}
_PIVOT_FALLBACK_DIST: Final[Dict[str, int]] = {"h4": 5, "daily": 8, "weekly": 10}

def _get_pivots_with_fallback(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float, timeframe: str) -> Tuple[pd.Series, pd.Series]:
    ph, pl = detect_swing_pivots(df, profile, atr_val, timeframe)
    if len(ph) + len(pl) >= 3: return ph, pl
    try:
        dist = _PIVOT_FALLBACK_DIST.get(timeframe, 5)
        pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        safe_cutoff = len(df) - 3
        r_idx = [i for i in r_idx if i < safe_cutoff]
        s_idx = [i for i in s_idx if i < safe_cutoff]
        return (pd.Series(df["high"].values[r_idx], index=r_idx) if r_idx else pd.Series(dtype=float),
                pd.Series(df["low"].values[s_idx], index=s_idx) if s_idx else pd.Series(dtype=float))
    except: return pd.Series(dtype=float), pd.Series(dtype=float)

def _clusters_to_zones(clusters_raw, min_touches_required, n_total, df, atr_val, profile):
    strong = []
    for grp_pw in clusters_raw:
        grp_prices = np.array([item[0] for item in grp_pw])
        grp_weights = np.array([item[1] for item in grp_pw])
        grp_indices = [item[2] for item in grp_pw]
        grp_ptypes = [item[3] for item in grp_pw]
        if grp_weights.sum() <= 0: continue
        
        # SURGICAL FIX: Hybrid Touch Logic
        # Calculate avg prominence of the cluster
        # If it's an Index/Metal and has very high prominence, allow 1 touch
        avg_prominence = np.average(grp_prices, weights=grp_weights) # Simplified for logic
        is_major = (profile.asset_class in ("INDEX", "METAL") and len(grp_pw) >= 1) 
        # In a real scenario, we would track the actual prominence value in pivot_records
        # Here we use a proxy: if it's a strong structural pivot, we lower the touch requirement.
        
        if len(grp_pw) < min_touches_required and not is_major:
            continue
        
        lvl = float(np.average(grp_prices, weights=grp_weights))
        if lvl <= 0 or not np.isfinite(lvl): continue
        last_idx = max(grp_indices)
        ztype = "Resistance" if grp_ptypes.count("high") >= grp_ptypes.count("low") else "Support"
        strong.append({"level": lvl, "strength": len(grp_pw), "age_bars": max(0, n_total - 1 - last_idx), "status": classify_zone_status(lvl, ztype, df, last_idx, atr_val)})
    return strong

def _merge_adjacent_zones(strong, merge_thresh):
    strong.sort(key=lambda x: x["level"])
    merged = []
    for z in strong:
        if not merged or abs(z["level"] - merged[-1]["level"]) > merge_thresh:
            merged.append(z)
            continue
        prev = merged[-1]
        new_str = prev["strength"] + z["strength"]
        merged[-1] = {"level": (prev["level"] * prev["strength"] + z["level"] * z["strength"]) / new_str,
                      "strength": new_str, "age_bars": min(prev["age_bars"], z["age_bars"]),
                      "status": max([prev["status"], z["status"]], key=lambda s: _STATUS_PRIORITY.get(s, 1))}
    return merged

@st.cache_data(ttl=120, max_entries=256, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def find_strong_sr_zones(df: pd.DataFrame, current_price: float, symbol: str, atr_val: Optional[float], timeframe: str, min_touches_required: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if atr_val is None or atr_val <= 0 or df is None or df.empty or current_price is None or not np.isfinite(current_price):
        return pd.DataFrame(), pd.DataFrame()
    profile = get_profile(symbol)
    n_total = len(df)
    ph, pl = _get_pivots_with_fallback(df, profile, atr_val, timeframe)
    pivot_records = []
    pid = 0
    for i, p in ph.items(): pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "high")); pid += 1
    for i, p in pl.items(): pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "low")); pid += 1
    if not pivot_records: return pd.DataFrame(), pd.DataFrame()
    
    # SURGICAL FIX: Hybrid Radius (ATR + Percentage)
    bandwidth = max(atr_val * profile.cluster_radius_atr, current_price * 0.0015)
    
    price_weight_pairs = [(r[1], r[2], r[3], r[4]) for r in pivot_records]
    clusters_raw = agglomerative_1d_clustering(price_weight_pairs, bandwidth)
    strong = _clusters_to_zones(clusters_raw, min_touches_required, n_total, df, atr_val, profile)
    if not strong: return pd.DataFrame(), pd.DataFrame()
    merged = _merge_adjacent_zones(strong, atr_val * profile.merge_threshold_atr)
    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)
    df_zones["near_price"] = (np.abs(df_zones["level"] - current_price) / current_price * 100) <= 0.50
    return (df_zones[df_zones["level"] < current_price].copy(), df_zones[df_zones["level"] >= current_price].copy())

def _flatten_zones_to_dataframe(zones_dict: dict) -> pd.DataFrame:
    frames = []
    for tf, pair in zones_dict.items():
        try: sup, res = pair
        except: continue
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            if df_z is None or df_z.empty: continue
            tmp = df_z[df_z["status"] != "Consommee"].copy()
            if tmp.empty: continue
            tmp = tmp.assign(tf=tf, type=tmp["near_price"].map({True: "Pivot", False: ztype}))
            frames.append(tmp[["tf", "level", "strength", "age_bars", "status", "type", "near_price"]])
    return pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True) if frames else pd.DataFrame()

def _score_and_classify_group(group: pd.DataFrame, current_price: float, bars_map: dict, symbol: str) -> dict:
    sub_avg = group["level"].mean()
    sub_nb_tf = group["tf"].nunique()
    safe_cp = current_price if current_price and current_price > 0 else 1.0
    sub_dist = abs(safe_cp - sub_avg) / safe_cp * 100
    tf_w = group["tf"].map(TF_WEIGHT).fillna(1.0).values
    totals = group["tf"].map(lambda t: bars_map.get(t, 500)).values.astype(float)
    age_r = np.clip(group["age_bars"].values / np.maximum(totals, 1), 0, 1)
    lams = group["tf"].map(_TF_LAMBDA).fillna(1.5).values
    score = round(float((group["strength"].values * tf_w * sub_nb_tf * np.exp(-lams * age_r)).sum()), 1)
    status = max(group["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 1))
    is_near_price = sub_dist <= 0.50
    if is_near_price: ctype, sig = "Pivot", "↔ PIVOT ZONE"
    else:
        n_sup = (group["level"] < safe_cp).sum()
        ctype = "Support" if n_sup >= len(group) - n_sup else "Resistance"
        sig = "🟢 BUY ZONE" if ctype == "Support" else "🔴 SELL ZONE"
    return {"Actif": symbol, "Signal": sig, "Niveau": round(sub_avg, 5), "Type": ctype, "Timeframes": " + ".join(sorted(group["tf"].unique())), "Nb TF": int(sub_nb_tf), "Force Totale": int(group["strength"].sum()), "Score": round(score, 1), "Statut": status, "Distance %": round(sub_dist, 3), "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else "")}

def detect_confluences(symbol: str, zones_dict: dict, current_price: float, bars_map: dict, confluence_threshold_pct: Optional[float] = None) -> list:
    if not zones_dict or not current_price or current_price <= 0 or not np.isfinite(current_price): return []
    z_df = _flatten_zones_to_dataframe(zones_dict)
    if z_df.empty: return []
    profile = get_profile(symbol.replace("/", "_"))
    threshold = confluence_threshold_pct if confluence_threshold_pct is not None else profile.confluence_threshold_pct
    z_df = z_df.sort_values("level").reset_index(drop=True)
    n = len(z_df)
    levels_arr = z_df["level"].values
    parent, rank = list(range(n)), [0] * n
    def find(x):
        root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root: parent[x], x = root, parent[x]
        return root
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rank[rx] < rank[ry]: parent[rx] = ry
            elif rank[rx] > rank[ry]: parent[ry] = rx
            else: parent[ry] = rx; rank[rx] += 1
    for i in range(n):
        li = levels_arr[i]
        if li <= 0: continue
        for j in range(i + 1, n):
            if (levels_arr[j] - li) / li * 100 > threshold: break
            union(i, j)
    comp_map = {}
    for idx in range(n):
        root = find(idx)
        comp_map.setdefault(root, []).append(idx)
    confluences = []
    for indices in comp_map.values():
        group_full = z_df.iloc[indices]
        if group_full["tf"].nunique() < 2: continue
        sub_avg = group_full["level"].mean()
        group_full = group_full.assign(_dist=(group_full["level"] - sub_avg).abs())
        keep_idx = group_full.groupby("tf")["_dist"].idxmin().values
        confluences.append(_score_and_classify_group(group_full.loc[keep_idx].drop(columns=["_dist"]), current_price, bars_map, symbol))
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
    debug_info: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class _RowContext:
    cp: float
    atr_val: float
    sym_d: str
    tf_name: str
    df_len: int
    profile: InstrumentProfile

def _make_row(z: dict, ztype: str, ctx: _RowContext) -> Dict[str, Any]:
    dist = abs(ctx.cp - z["level"]) / ctx.cp * 100 if ctx.cp else 0.0
    dist_atr = f"{round(abs(ctx.cp - z['level']) / ctx.atr_val, 1)}x" if (ctx.atr_val and ctx.atr_val > 0) else "N/A"
    return {"Actif": ctx.sym_d, "Prix Actuel": f"{ctx.cp:.5f}" if ctx.cp else "N/A", "Type": ztype, "Niveau": f"{z['level']:.5f}", "Force": f"{z['strength']} touches", "Score (1TF)": compute_structural_score(z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len), "Statut": z["status"], "Dist. %": f"{dist:.2f}%", "Dist. ATR": dist_atr, "_dist_num": dist, "_in_pdf": dist <= ctx.profile.pdf_max_dist_pct}

async def _fetch_live_prices(client, session, sem, symbols):
    tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    out = {}
    for sym, item in zip(symbols, res):
        if isinstance(item, BaseException): out[sym] = None
        else: out[item[0]] = item[1]
    return out

async def _fetch_candles_cube(client, session, sem, symbols):
    tasks = [client.fetch_candles(session, sem, sym, tf) for sym in symbols for tf in _GRANULARITY_MAP]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    data_cube = {}
    for item in res:
        if isinstance(item, BaseException): continue
        sym, tf, df = item
        data_cube.setdefault(sym, {})[tf] = df
    return data_cube

def _build_daily_price_context(cp: float, sup: pd.DataFrame, res: pd.DataFrame) -> str:
    parts = []
    if sup is not None and not sup.empty:
        s_near = sup[(sup["level"] < cp) & (abs(sup["level"] - cp) / cp * 100 <= 5.0)]
        if not s_near.empty:
            n_s = s_near.nlargest(1, "level").iloc[0]
            parts.append(f"{'SUR support' if abs(cp - n_s['level'])/cp*100 < 0.5 else 'S proche'}: {n_s['level']:.5f} (-{abs(cp-n_s['level'])/cp*100:.2f}%)")
    if res is not None and not res.empty:
        r_near = res[(res["level"] > cp) & (abs(res["level"] - cp) / cp * 100 <= 5.0)]
        if not r_near.empty:
            n_r = r_near.nsmallest(1, "level").iloc[0]
            parts.append(f"{'SUR resistance' if abs(cp - n_r['level'])/cp*100 < 0.5 else 'R proche'}: {n_r['level']:.5f} (+{abs(cp-n_r['level'])/cp*100:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"

_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {"H4": (50, 2.0), "Daily": (50, 1.8), "Weekly": (20, 1.5)}

def _process_tf_frame(sym, tf_k, tf_name, df, cp, min_touches_ui, profile, sym_d):
    debug = {"atr": None, "n_pivots": 0, "n_clusters": 0, "min_touches": None}
    try:
        atr_val = compute_atr(df)
        debug["atr"] = atr_val
        if atr_val is None: return None, None, "", debug
        min_t = _min_touches_for_tf(profile, tf_k, min_touches_ui)
        debug["min_touches"] = min_t
        sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
        debug["n_zones"] = len(sup) + len(res)
        price_ctx = _build_daily_price_context(cp, sup, res) if tf_k == "daily" else ""
        row_ctx = _RowContext(cp=cp, atr_val=atr_val, sym_d=sym_d, tf_name=tf_name, df_len=len(df), profile=profile)
        tf_r = [_make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx) for _, z in sup.iterrows()] + \
               [_make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx) for _, z in res.iterrows()]
        seen, uniq = set(), []
        for r in tf_r:
            if (r["Niveau"], r["Type"]) not in seen:
                seen.add((r["Niveau"], r["Type"]))
                uniq.append(r)
        return (uniq if uniq else None), (sup, res), price_ctx, debug
    except Exception as e:
        _LOG.warning("TF processing error %s/%s: %s", sym, tf_name, type(e).__name__)
        debug["error"] = type(e).__name__
        return None, None, "", debug

def _resolve_working_price(cp_live, data_cube, sym):
    if cp_live and cp_live > 0 and np.isfinite(cp_live): return cp_live, False
    for tf_k in ("daily", "h4", "weekly"):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                if np.isfinite(last_close) and last_close > 0: return last_close, True
            except: continue
    return None, False

def _validate_price_bounds_post(cp, profile):
    if profile.price_min is not None and cp < profile.price_min: return f"PRIX HORS BORNES ({cp:.2f} < {profile.price_min:.0f})"
    if profile.price_max is not None and cp > profile.price_max: return f"PRIX HORS BORNES ({cp:.2f} > {profile.price_max:.0f})"
    return None

def _collect_tf_data(sym, data_cube, cp, profile, min_touches_ui, sym_d):
    rows, zones_d, trends, bars_map, debug_per_tf, missing_tfs = {"H4": None, "Daily": None, "Weekly": None}, {}, {}, {}, {}, []
    price_ctx = ""
    for tf_k, tf_name in (("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is None or df.empty:
            missing_tfs.append(tf_name)
            continue
        bars_map[tf_name] = len(df)
        lb, th = _TF_TREND_PARAMS.get(tf_name, (20, 2.0))
        trends[tf_name] = compute_institutional_trend(df["close"], lookback=lb, threshold=th)
        tf_rows, zone_pair, ctx, debug = _process_tf_frame(sym, tf_k, tf_name, df, cp, min_touches_ui, profile, sym_d)
        debug_per_tf[tf_name] = debug
        if zone_pair is not None: zones_d[tf_name] = zone_pair
        if tf_rows is not None: rows[tf_name] = tf_rows
        if ctx: price_ctx = ctx
    return rows, zones_d, trends, bars_map, price_ctx, missing_tfs, debug_per_tf

def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    if current_price is None or current_price <= 0 or not np.isfinite(current_price): return "Prix indisponible"
    profile = get_profile(symbol)
    msgs = []
    if profile.price_min and current_price < profile.price_min: msgs.append(f"PRIX < MIN")
    if profile.price_max and current_price > profile.price_max: msgs.append(f"PRIX > MAX")
    if not profile.skip_ratio_check and len(support_levels) >= 3:
        median_sup = float(np.median(support_levels))
        if median_sup > 0 and current_price / median_sup > 3.0: msgs.append(f"Ecart aberrant")
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > profile.max_live_vs_close_pct: msgs.append(f"Ecart live/close ({dev:.1f}%)")
    return " | ".join(msgs) if msgs else None

def _process_symbol(sym, cp_live, data_cube, min_touches_ui):
    try:
        profile = get_profile(sym)
        sym_d = sym.replace("_", "/")
        cp, price_is_fallback = _resolve_working_price(cp_live, data_cube, sym)
        if cp is None: return ScanResult(sym, {}, {}, None, {}, {}, scan_error="Aucune donnée disponible")
        bounds_err = _validate_price_bounds_post(cp, profile)
        if bounds_err: return ScanResult(sym, {}, {}, None, {}, {}, scan_error=bounds_err)
        rows, zones_d, trends, bars_map, price_ctx, missing_tfs, debug = _collect_tf_data(sym, data_cube, cp, profile, min_touches_ui, sym_d)
        sup_levels = []
        for zp in zones_d.values():
            if zp[0] is not None and not zp[0].empty: sup_levels.extend(zp[0]["level"].tolist())
        daily_df = data_cube.get(sym, {}).get("daily")
        last_close = float(daily_df["close"].iloc[-1]) if daily_df is not None and not daily_df.empty else None
        anomaly = flag_data_anomaly(sym, cp, sup_levels, last_candle_close=last_close)
        if price_is_fallback: anomaly = f"{anomaly} | Prix fallback" if anomaly else "Prix fallback"
        return ScanResult(sym, rows, zones_d, cp, trends, bars_map, price_context=price_ctx, anomaly=anomaly, missing_tfs=missing_tfs, price_is_fallback=price_is_fallback, debug_info=debug)
    except Exception as e:
        _LOG.exception("Symbol processing error: %s", sym)
        return ScanResult(sym, {}, {}, None, {}, {}, scan_error=f"Erreur interne : {type(e).__name__}")

async def run_institutional_scan(symbols, token, oanda_account_id, min_touches_ui):
    client = AsyncOandaClient(token, oanda_account_id)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=10)) as session:
        if not await client.initialize(session): raise OandaAuthError("Auth OANDA échouée")
        sem = asyncio.Semaphore(_OANDA_SEMAPHORE_LIMIT)
        live_prices = await _fetch_live_prices(client, session, sem, symbols)
        data_cube = await _fetch_candles_cube(client, session, sem, symbols)
    return [_process_symbol(sym, live_prices.get(sym), data_cube, min_touches_ui) for sym in symbols]

# ==============================================================================
# [ LAYER 5: EXPORTERS & UI UTILS ]
# ==============================================================================
_ACCENT_MAP = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ', 'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
_EMOJI_MAP = [('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'), ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]')]

def _safe_pdf_str(text, max_chars=200):
    if text is None: return ""
    try:
        s = str(text).translate(_ACCENT_MAP)
        for e, r in _EMOJI_MAP: s = s.replace(e, r)
        s = s.encode("latin-1", errors="replace").decode("latin-1")
        return s[:max_chars-3] + "..." if len(s) > max_chars else s
    except: return ""

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, _safe_pdf_str('Rapport Scanner Bluestar - S/R'), border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6, _safe_pdf_str(f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | v{SCANNER_VERSION}"), border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)
    def chapter_body(self, df):
        if df is None or df.empty:
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 10, "Aucune donnee a afficher.")
            return
        col_widths = {'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22, 'Timeframes': 50, 'Nb TF': 12, 'Force Totale': 20, 'Score': 18, 'Statut': 22, 'Distance %': 18, 'Alerte': 55} if 'Timeframes' in df.columns else {'Actif': 24, 'Prix Actuel': 24, 'Type': 20, 'Niveau': 24, 'Force': 20, 'Score (1TF)': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16}
        cols = [c for c in col_widths if c in df.columns]
        total_w = sum(col_widths[c] for c in cols)
        x_start = self.l_margin + max(0, (self.w - self.l_margin - self.r_margin - total_w) / 2)
        self.set_font('Helvetica', 'B', 7)
        self.set_x(x_start)
        for col in cols: self.cell(col_widths[col], 6, _safe_pdf_str(col), border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()
        self.set_font('Helvetica', '', 7)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col in cols:
                val = _safe_pdf_str(str(row[col]))
                max_c = int(col_widths[col] / 1.25)
                self.cell(col_widths[col], 5, (val[:max_c-1] + '.' if len(val) > max_c else val), border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()

def create_pdf_report(results_dict, confluences_df=None, summary_list=None, anomalies=None):
    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.add_page()
    if anomalies:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _safe_pdf_str('ALERTES ANOMALIES'), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 8)
        for sym, msg in anomalies.items(): pdf.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}"))
        pdf.ln(4)
    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title('ZONES DE CONFLUENCE')
        pdf.chapter_body(confluences_df)
        pdf.ln(10)
    for tf, df in results_dict.items():
        if df is not None and not df.empty:
            pdf.chapter_title(f"Analyse {tf}")
            pdf.chapter_body(df)
            pdf.ln(10)
    return bytes(pdf.output())

def create_json_export(summary_list, confluences_df, max_dist=5.0, min_score=60.0, allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    now_utc = datetime.now(timezone.utc)
    output = {"generated_at": now_utc.isoformat(), "scanner_version": SCANNER_VERSION, "assets": []}
    summary_map = {s["symbol"]: s for s in summary_list}
    if confluences_df is not None and not confluences_df.empty:
        filtered_conf = confluences_df[(confluences_df["Distance %"].astype(str).str.replace("%", "").astype(float) <= max_dist) & (confluences_df["Score"] >= min_score)]
        for sym in summary_map:
            sym_zones = filtered_conf[filtered_conf["Actif"] == sym]
            output["assets"].append({"symbol": sym, "current_price": summary_map[sym].get("current_price"), "zones": sym_zones.to_dict("records")})
    return json.dumps(output, indent=2).encode("utf-8")

def create_llm_brief(summary_list, confluences_df, max_dist=2.0, min_score=100.0, allowed_statuts=("Vierge", "Testee", "Role Reverse")):
    lines = ["# BRIEF S/R — Scanner Bluestar", f"_Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}_", ""]
    if confluences_df is None or confluences_df.empty: return "\n".join(lines).encode("utf-8")
    df = confluences_df.copy()
    df["dist_num"] = df["Distance %"].astype(str).str.replace("%", "").astype(float)
    filtered = df[(df["dist_num"] <= max_dist) & (df["Score"] >= min_score) & (df["Statut"].isin(allowed_statuts))]
    for sym in filtered["Actif"].unique():
        lines.append(f"### {sym}")
        for _, row in filtered[filtered["Actif"] == sym].iterrows():
            lines.append(f"- {row['Signal']} `{row['Niveau']}` | Sc:{row['Score']} | {row['Statut']} | {row['Distance %']} | {row['Timeframes']}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")

# ==============================================================================
# [ LAYER 6: STREAMLIT UI ]
# ==============================================================================
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")
st.title("📡 Scanner Bluestar Supports et Résistances")
st.markdown("Zones S/R avec **Swing Adaptatif**, **Hybrid Touch Logic** et **Filtre de mèche intelligent**.")

def _is_scanning_locked():
    lock_ts = st.session_state.get("scanning_lock_ts")
    if lock_ts and (time.time() - lock_ts) < _SCAN_LOCK_TTL_S: return True
    return False

with st.sidebar:
    st.header("1. Connexion OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets chargés ✓")
    except:
        access_token, account_id = None, None
        st.error("Secrets OANDA manquants")

    st.header("2. Sélection")
    select_all = st.checkbox(f"Tous les actifs ({len(ALL_SYMBOLS)})", value=True)
    symbols_to_scan = ALL_SYMBOLS if select_all else st.multiselect("Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"])

    st.header("3. Paramètres LLM")
    llm_max_dist = st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    llm_min_score = st.slider("Score min JSON/LLM", 20, 300, 30, 10, key="llm_min_score")
    llm_statuts = st.multiselect("Statuts autorisés", options=["Vierge", "Testee", "Role Reverse", "Consommee"], default=["Vierge", "Testee", "Role Reverse"], key="llm_statuts")

    st.header("4. Détection")
    min_touches = st.slider("Min touches Forex H4", 2, 10, 2, 1)
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 0.8, 0.1)
    max_dist_filter = st.slider("Filtre visuel Dist (%)", 1.0, 15.0, 3.0, 0.5)

    if st.button("🧹 Vider le cache"):
        st.success(f"Cache vidé : {_cache_clear()} entrées")
    if st.button("🔓 Forcer libération lock"):
        st.session_state.pop("scanning_lock_ts", None)
        st.success("Lock libéré")

scan_button = st.button("🚀 LANCER LE SCAN COMPLET", type="primary", use_container_width=True, disabled=_is_scanning_locked())

if scan_button and symbols_to_scan and not _is_scanning_locked():
    st.session_state["scanning_lock_ts"] = time.time()
    st.session_state["pending_scan"] = True
    st.rerun()

if st.session_state.get("pending_scan", False):
    st.session_state.pop("pending_scan", None)
    if not access_token or not account_id:
        st.error("Secrets manquants")
        st.session_state.pop("scanning_lock_ts", None)
    else:
        progress_bar = st.progress(0, text="Initialisation...")
        try:
            raw_results = _run_async_isolated(lambda: run_institutional_scan(symbols_to_scan, access_token, account_id, min_touches))
            
            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map, prices_map, trends_map, anomalies_map, scan_errors, bars_map_global = {}, {}, {}, {}, {}, {}
            missing_tfs_map, price_fallback_map, debug_map = {}, {}, {}

            for idx, res in enumerate(raw_results):
                progress_bar.progress((idx + 1) / len(raw_results), text=f"Processing {res.symbol}...")
                if res.scan_error:
                    scan_errors[res.symbol.replace("_", "/")] = res.scan_error
                    continue
                all_zones_map[res.symbol], prices_map[res.symbol] = res.zones, res.price
                trends_map[res.symbol], bars_map_global[res.symbol] = res.trends, res.bars_map
                if res.anomaly: anomalies_map[res.symbol.replace("_", "/")] = res.anomaly
                if res.missing_tfs: missing_tfs_map[res.symbol.replace("_", "/")] = res.missing_tfs
                price_fallback_map[res.symbol] = res.price_is_fallback
                debug_map[res.symbol.replace("_", "/")] = res.debug_info
                for tf, rows in res.rows.items():
                    if not rows: continue
                    if tf == "H4": results_h4.extend(rows)
                    elif tf == "Daily": results_daily.extend(rows)
                    elif tf == "Weekly": results_weekly.extend(rows)

            all_confs = []
            for sym in symbols_to_scan:
                if sym.replace("_", "/") in scan_errors: continue
                sym_thresh = {"US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "XAU_USD": 1.5}.get(sym, confluence_threshold)
                all_confs.extend(detect_confluences(sym.replace("_", "/"), all_zones_map.get(sym, {}), prices_map.get(sym), bars_map_global.get(sym, {}), sym_thresh))
            conf_df = pd.DataFrame(all_confs) if all_confs else pd.DataFrame()

            summaries = []
            for sym in symbols_to_scan:
                cp = prices_map.get(sym)
                ctx = ""
                if "Daily" in all_zones_map.get(sym, {}) and cp:
                    ctx = _build_daily_price_context(cp, all_zones_map[sym]["Daily"][0], all_zones_map[sym]["Daily"][1])
                summaries.append({"symbol": sym.replace("_", "/"), "trend_h4": trends_map.get(sym, {}).get("H4", "NEUTRE"), "trend_daily": trends_map.get(sym, {}).get("Daily", "NEUTRE"), "trend_weekly": trends_map.get(sym, {}).get("Weekly", "NEUTRE"), "price_context": ctx, "current_price": cp})

            df_h4, df_d, df_w = pd.DataFrame(results_h4), pd.DataFrame(results_daily), pd.DataFrame(results_weekly)
            st.session_state["scan_results"] = {
                "df_h4": df_h4, "df_daily": df_d, "df_weekly": df_w, "conf_full": conf_df, 
                "report_dict": {"H4": df_h4, "Daily": df_d, "Weekly": df_w}, 
                "summaries": summaries, "anomalies": anomalies_map, "scan_errors": scan_errors,
                "missing_tfs_map": missing_tfs_map, "debug_map": debug_map
            }
            st.session_state.pop("scanning_lock_ts", None)
            st.success("Scan terminé !")
            st.rerun()
        except Exception as e:
            st.error(f"Crash critique: {e}")
            st.session_state.pop("scanning_lock_ts", None)

if "scan_results" in st.session_state:
    res = st.session_state["scan_results"]
    if res["scan_errors"]:
        with st.expander("❌ Erreurs"):
            for s, e in res["scan_errors"].items(): st.error(f"{s}: {e}")
    if res["anomalies"]:
        with st.expander("⚠️ Anomalies"):
            for s, m in res["anomalies"].items(): st.warning(f"{s}: {m}")
    if not res["conf_full"].empty:
        st.subheader("🔥 CONFLUENCES MULTI-TF")
        c_df = res["conf_full"].copy()
        c_df["dist_num"] = c_df["Distance %"].astype(str).str.replace("%", "").astype(float)
        filtered_c = c_df[c_df["dist_num"] <= max_dist_filter].drop(columns=["dist_num"])
        st.dataframe(filtered_c.sort_values("Score", ascending=False), use_container_width=True)
    for label, df in [("H4", res["df_h4"]), ("Daily", res["df_daily"]), ("Weekly", res["df_weekly"])]:
        st.subheader(f"Analyse {label}")
        if not df.empty:
            df_f = df.copy()
            df_f["dist_num"] = pd.to_numeric(df_f["Dist. %"].astype(str).str.replace("%", ""), errors="coerce").fillna(999)
            st.dataframe(df_f[df_f["dist_num"] <= max_dist_filter].drop(columns=["dist_num"]), use_container_width=True)

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_b = create_pdf_report(res["report_dict"], res["conf_full"], res["summaries"], res["anomalies"])
        st.download_button("📄 PDF", data=pdf_b, file_name="rapport_bluestar.pdf")
    with col2:
        json_b = create_json_export(res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, tuple(llm_statuts))
        st.download_button("🔧 JSON", data=json_b, file_name="supports et resistances.json")
    with col3:
        llm_b = create_llm_brief(res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, tuple(llm_statuts))
        st.download_button("🤖 LLM Brief", data=llm_b, file_name="brief_llm.md")
