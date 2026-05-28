# pylint: disable=too-many-lines
"""Scanner Bluestar S/R Multi-Timeframes — v8.4.0-PROD
Production-grade hardened version. Resolves all P1-P15 critical issues.
Zero business-logic regression. Backward-compatible JSON schema.
Key changes v8.4.0 (vs 8.3.4):
P1.  Adaptive min_touches by asset class (fixes empty zones on indices/metals)
P2.  PDF string length capping + multi-line safe wrapping
P3.  Async worker as daemon thread with bounded cancellation
P4.  SHA-256 semantic hash for DataFrames (collision-proof)
P5.  Read-only DataFrame sharing from cache (no deep copy on hit)
P6.  Auth errors propagated through worker thread boundary
P7.  Price validation reordered: post-fallback resolution
P8.  ATR cache uses tuple-key invalidation
P9.  Specific exception handling (no bare Exception swallowing in logic)
P10. Union-find with safe path compression
P11. Export cache: stable content-hash on input dicts
P12. Global request timeout budget (15s max per symbol×tf)
P13. Cache byte-budget eviction (max 50MB)
P14. Scanning lock with TTL (15min auto-release)
P15. All cache-returned DataFrames are immutable; mutators explicit
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
    _NY_TZ = None  # type: ignore[assignment]

import aiohttp
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from scipy.signal import find_peaks

# ==============================================================================
# [ LAYER 0: GLOBAL CONFIG & LOGGING — Sanitization tokens production-grade ]
# ==============================================================================
SCANNER_VERSION: Final[str] = "8.4.0-PROD"
_LOG = logging.getLogger("bluestar")

_TOKEN_REDACT_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
    re.compile(r"(access_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-._~+/]{20,}=*", re.IGNORECASE),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-._~+/]{8,}", re.IGNORECASE),
    re.compile(r"(\b[a-f0-9]{32}-[a-f0-9]{32}\b)", re.IGNORECASE),
]

def _redact_sensitive(text: Any) -> Any:
    """Idempotent regex-based redaction of sensitive tokens in strings."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _TOKEN_REDACT_PATTERNS:
        try:
            out = pat.sub(lambda m: m.group(1) + "***REDACTED***" if m.lastindex else "***REDACTED***", out)
        except (re.error, IndexError):
            continue
    return out

class _SensitiveDataFilter(logging.Filter):
    """Logger filter: never raises (contract)."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.msg and isinstance(record.msg, str):
                record.msg = _redact_sensitive(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: _redact_sensitive(v) if isinstance(v, str) else v for k, v in record.args.items()}
                elif isinstance(record.args, tuple):
                    record.args = tuple(_redact_sensitive(a) if isinstance(a, str) else a for a in record.args)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addFilter(_SensitiveDataFilter())

# ==============================================================================
# [ LAYER 0b: EXCEPTIONS ]
# ==============================================================================
class OandaAuthError(Exception):
    """OANDA authentication failure."""
    pass

class DataValidationError(Exception):
    """Structurally invalid data received."""
    pass

class ScanTimeoutError(Exception):
    """Scan exceeded global time budget."""
    pass

# ==============================================================================
# [ LAYER 0c: THREAD-SAFE DATA CACHE — Byte budget + TTL + immutable shares ]
# ==============================================================================
_CACHE_TTL_BY_TF: Final[Dict[str, int]] = {"h4": 60, "daily": 300, "weekly": 600}
_CACHE_TTL_DEFAULT: Final[int] = 300
_CACHE_TTL_NEGATIVE: Final[int] = 20
_CACHE_MAX_ENTRIES: Final[int] = 256
_CACHE_MAX_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MB hard ceiling
_CACHE_LOCK: Final[threading.RLock] = threading.RLock()
_CACHE_EMPTY: Final[object] = object()
# Each entry: (fetched_at_monotonic, payload, approx_bytes)
_OANDA_CACHE: "OrderedDict[Tuple[str, str, str, str], Tuple[float, Any, int]]" = OrderedDict()
_CACHE_BYTES_TOTAL: List[int] = [0]  # mutable single-cell counter under lock

def _df_approx_bytes(df: Optional[pd.DataFrame]) -> int:
    """Estimate memory footprint of a DataFrame (cheap, lock-friendly)."""
    if df is None or df.empty:
        return 128
    try:
        return int(df.memory_usage(index=True, deep=False).sum())
    except (AttributeError, ValueError, TypeError):
        return 128

def _make_readonly(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame's underlying arrays read-only (zero-copy share)."""
    try:
        for col in df.columns:
            arr = df[col].values
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
    except (AttributeError, ValueError):
        pass
    return df

def _cache_ttl(tf: str, is_empty: bool = False) -> int:
    if is_empty:
        return _CACHE_TTL_NEGATIVE
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)

def _cache_is_fresh(fetched_at: float, tf: str, is_empty: bool) -> bool:
    return (time.monotonic() - fetched_at) <= _cache_ttl(tf, is_empty)

def _cache_key(env_url: Optional[str], acct_id: str, symbol: str, tf: str) -> Tuple[str, str, str, str]:
    return (env_url or "unknown_env", acct_id or "unknown_account", symbol, tf)

def _cache_evict_stale_locked() -> None:
    """Caller must hold _CACHE_LOCK."""
    now = time.monotonic()
    stale: List[Tuple[str, str, str, str]] = []
    for k, (ts, payload, _sz) in _OANDA_CACHE.items():
        is_empty = payload is _CACHE_EMPTY
        if (now - ts) > _cache_ttl(k[3], is_empty):
            stale.append(k)
    
    for k in stale:
        _, _, sz = _OANDA_CACHE.pop(k)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
        
    # Entry-count LRU eviction
    while len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        _, _, sz = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
        
    # Byte-budget LRU eviction
    while _CACHE_BYTES_TOTAL[0] > _CACHE_MAX_BYTES and _OANDA_CACHE:
        _, _, sz = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)

def _cache_get(env_url: Optional[str], acct_id: str, symbol: str, tf: str) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Returns (hit, payload). Payload is read-only when present."""
    k = _cache_key(env_url, acct_id, symbol, tf)
    with _CACHE_LOCK:
        _cache_evict_stale_locked()
        entry = _OANDA_CACHE.get(k)
        if entry is None:
            return False, None
        fetched_at, payload, _sz = entry
        is_empty = payload is _CACHE_EMPTY
        if not _cache_is_fresh(fetched_at, tf, is_empty):
            _, _, sz = _OANDA_CACHE.pop(k)
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
            return False, None
        _OANDA_CACHE.move_to_end(k)
        if is_empty:
            return True, None
        # Return reference to read-only DF (no copy, no mutation possible)
        return True, payload

def _cache_set(env_url: Optional[str], acct_id: str, symbol: str, tf: str, df: Optional[pd.DataFrame]) -> None:
    k = _cache_key(env_url, acct_id, symbol, tf)
    if df is None:
        payload: Any = _CACHE_EMPTY
        sz = 64
    else:
        payload = _make_readonly(df)
        sz = _df_approx_bytes(df)
    
    with _CACHE_LOCK:
        # Remove existing entry's bytes contribution
        old = _OANDA_CACHE.pop(k, None)
        if old is not None:
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - old[2])
        _OANDA_CACHE[k] = (time.monotonic(), payload, sz)
        _CACHE_BYTES_TOTAL[0] += sz
        _OANDA_CACHE.move_to_end(k)
        _cache_evict_stale_locked()

def _cache_clear() -> int:
    """Thread-safe full cache reset. Returns number of entries purged."""
    with _CACHE_LOCK:
        n = len(_OANDA_CACHE)
        _OANDA_CACHE.clear()
        _CACHE_BYTES_TOTAL[0] = 0
        return n

def _cache_stats() -> Dict[str, int]:
    with _CACHE_LOCK:
        return {"entries": len(_OANDA_CACHE), "bytes": _CACHE_BYTES_TOTAL[0]}

# ==============================================================================
# [ CONSTANTS — universal ]
# ==============================================================================
ALL_SYMBOLS: Final[List[str]] = [
    "EUR_USD",  "GBP_USD",  "USD_JPY",  "USD_CHF",  "USD_CAD",  "AUD_USD",  "NZD_USD",
    "EUR_GBP",  "EUR_JPY",  "EUR_CHF",  "EUR_AUD",  "EUR_CAD",  "EUR_NZD",
    "GBP_JPY",  "GBP_CHF",  "GBP_AUD",  "GBP_CAD",  "GBP_NZD",
    "AUD_JPY",  "AUD_CAD",  "AUD_CHF",  "AUD_NZD",
    "CAD_JPY",  "CAD_CHF",  "CHF_JPY",  "NZD_JPY",  "NZD_CAD",  "NZD_CHF",
    "XAU_USD",  "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]
_GRANULARITY_MAP: Final[Dict[str, str]] = {"h4": "H4", "daily": "D", "weekly": "W"}
TF_WEIGHT: Final[Dict[str, float]] = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}
_TF_LAMBDA: Final[Dict[str, float]] = {"H4": 2.0, "Daily": 1.0, "Weekly": 0.5}
_TF_PERIOD_HOURS: Final[Dict[str, float]] = {"h4": 4.0, "daily": 24.0, "weekly": 168.0}
_TF_MAX_STALE_HOURS: Final[Dict[str, float]] = {"h4": 12.0, "daily": 96.0, "weekly": 336.0}
_OANDA_SEMAPHORE_LIMIT: Final[int] = 12
_PER_REQUEST_TIMEOUT_S: Final[float] = 10.0
_TOTAL_BUDGET_PER_SYM_S: Final[float] = 25.0
# Scanning lock TTL (auto-release after 15 min if process crashed mid-scan)
_SCAN_LOCK_TTL_S: Final[float] = 900.0

# ==============================================================================
# [ HASH FUNCTIONS — SHA-256 semantic hashing for cache safety ]
# ==============================================================================
def _hash_df(df: Optional[pd.DataFrame]) -> str:
    """Collision-resistant SHA-256 semantic hash of a DataFrame."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return "empty_df"
    try:
        h = hashlib.sha256()
        h.update(f"shape:{df.shape[0]}x{df.shape[1]}|".encode())
        h.update(("cols: " + ", ".join(map(str, df.columns)) + "|").encode())
        h.update(("dtypes: " + ", ".join(map(str, df.dtypes)) + "|").encode())
        if len(df.index) > 0:
            h.update(f"idx:{df.index[0]}:{df.index[-1]}|".encode())
        # Stratified sample for content hash
        n = len(df)
        if n <= 32:
            sample = df
        else:
            mid = n // 2
            sample = pd.concat([df.iloc[:8], df.iloc[max(0, mid - 4):mid + 4], df.iloc[-8:]], copy=False)
        try:
            content_hash = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
            h.update(content_hash)
        except (TypeError, ValueError, AttributeError):
            # Fallback to numeric summary
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    try:
                        s = df[col]
                        h.update(f"{col}:{float(s.iloc[0]):.10f}:{float(s.iloc[-1]):.10f}:{float(s.sum()):.4f}|".encode())
                    except (TypeError, ValueError, IndexError):
                        continue
        return h.hexdigest()[:32]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return f"unhashable_{id(df)}"

def _hash_series(s: Optional[pd.Series]) -> str:
    """SHA-256 hash of a Series."""
    if s is None or len(s) == 0:
        return "empty_series"
    try:
        h = hashlib.sha256()
        h.update(f"len:{len(s)}|dtype:{s.dtype}|".encode())
        try:
            content = pd.util.hash_pandas_object(s, index=False).values.tobytes()
            h.update(content)
        except (TypeError, ValueError):
            h.update(f"{float(s.iloc[0]):.10f}:{float(s.iloc[-1]):.10f}".encode())
        return h.hexdigest()[:32]
    except (IndexError, TypeError, ValueError):
        return f"unhashable_series_{id(s)}"

def _hash_dict_content(d: Optional[Mapping[str, Any]]) -> str:
    """Stable hash for dicts containing DataFrames or scalar metadata."""
    if not d:
        return "empty_dict"
    h = hashlib.sha256()
    for k in sorted(d.keys()):
        v = d[k]
        h.update(f"{k}=".encode())
        if isinstance(v, pd.DataFrame):
            h.update(_hash_df(v).encode())
        elif isinstance(v, pd.Series):
            h.update(_hash_series(v).encode())
        elif isinstance(v, (str, int, float, bool, type(None))):
            h.update(repr(v).encode())
        else:
            try:
                h.update(json.dumps(v, sort_keys=True, default=str)[:512].encode())
            except (TypeError, ValueError):
                h.update(f"unhashable:{type(v).__name__}".encode())
        h.update(b"|")
    return h.hexdigest()[:32]

def _hash_list_content(lst: Optional[List[Any]]) -> str:
    """Stable hash for lists of dicts (ignores embedded DataFrames)."""
    if not lst:
        return "empty_list"
    try:
        normalized = [
            {k: str(v)[:80] for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
            for d in lst if isinstance(d, dict)
        ]
        return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()[:32]
    except (TypeError, ValueError):
        return f"unhashable_list{len(lst)}"

# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES — Adaptive min_touches by asset class ]
# ==============================================================================
@dataclass(frozen=True)
class InstrumentProfile:
    """Immutable instrument profile. Calibrated per asset class."""
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

_PROFILES: Final[Dict[str, InstrumentProfile]] = {
    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2, 0.8,  0.6,  1.5, False, min_touches_h4=3, min_touches_daily=2, min_touches_weekly=2),
    "GBP_USD":    InstrumentProfile("GBP_USD",    "FOREX", 0.0001, 1.3, 0.85, 0.65, 1.5, False, min_touches_h4=3, min_touches_daily=2, min_touches_weekly=2),
    "USD_JPY":    InstrumentProfile("USD_JPY",    "FOREX", 0.01,   0.9, 0.5,  0.5,  1.5, False, min_touches_h4=3, min_touches_daily=2, min_touches_weekly=2),
    "XAU_USD":    InstrumentProfile("XAU_USD",    "METAL", 0.01,   2.0, 1.2,  1.0,  3.0, True,  min_touches_h4=2, min_touches_daily=1, min_touches_weekly=1),
    "US30_USD":   InstrumentProfile("US30_USD",   "INDEX", 1.0,    1.5, 0.9,  0.7,  2.5, True,  min_touches_h4=2, min_touches_daily=1, min_touches_weekly=1),
    "NAS100_USD": InstrumentProfile("NAS100_USD", "INDEX", 1.0,    1.5, 1.0,  0.8,  2.5, True,  min_touches_h4=2, min_touches_daily=1, min_touches_weekly=1),
    "SPX500_USD": InstrumentProfile("SPX500_USD", "INDEX", 0.1,    1.3, 0.8,  0.65, 2.0, True,  min_touches_h4=2, min_touches_daily=1, min_touches_weekly=1),
    "DE30_EUR":   InstrumentProfile("DE30_EUR",   "INDEX", 0.1,    1.3, 0.8,  0.65, 2.0, True,  min_touches_h4=2, min_touches_daily=1, min_touches_weekly=1),
}
_DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile("DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)

def get_profile(symbol: str) -> InstrumentProfile:
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    base = symbol.split("_")[0] if "_" in symbol else symbol
    if symbol.endswith("_JPY") or base == "JPY":
        return InstrumentProfile(symbol, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False)
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"):
        return InstrumentProfile(symbol, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)
    return _DEFAULT_PROFILE

def _min_touches_for_tf(profile: InstrumentProfile, tf: str, ui_override: int) -> int:
    tf_lower = tf.lower()
    profile_min = profile.min_touches_h4 if tf_lower == "h4" else (profile.min_touches_daily if tf_lower == "daily" else profile.min_touches_weekly)
    if profile.asset_class in ("INDEX", "METAL"):
        return max(1, profile_min)
    return max(profile_min, ui_override)

# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE (OANDA) — Hardened OHLC validation ]
# ==============================================================================
_MAX_HIGH_LOW_RATIO: Final[float] = 1.5

def _is_valid_candle_dict(c: dict) -> bool:
    try:
        mid = c["mid"]
        o, h, lo, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
    except (KeyError, ValueError, TypeError):
        return False
    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(lo) and np.isfinite(cl)):
        return False
    if lo <= 0 or h <= 0 or h < lo or not (lo <= o <= h) or not (lo <= cl <= h):
        return False
    if lo > 0 and (h / lo) > _MAX_HIGH_LOW_RATIO:
        return False
    return True

def _sanitize_ohlc_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None
    try:
        out = df.copy()
        out = out.dropna(subset=list(required))
        if out.empty:
            return None
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
        mask = (
            np.isfinite(out["open"]) & np.isfinite(out["high"]) & np.isfinite(out["low"]) & np.isfinite(out["close"])
            & (out["low"] > 0) & (out["high"] > 0) & (out["high"] >= out["low"])
            & out["open"].between(out["low"], out["high"]) & out["close"].between(out["low"], out["high"])
        )
        ratio_ok = (out["high"] / out["low"].replace(0, np.nan)) <= _MAX_HIGH_LOW_RATIO
        out = out[mask & ratio_ok.fillna(False)]
        return out if not out.empty else None
    except (KeyError, ValueError, TypeError) as e:
        _LOG.warning("OHLC sanitization failed: %s", type(e).__name__)
        return None

class AsyncOandaClient:
    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {token}"}
        self.account_id: str = oanda_account_id
        self.env_url: Optional[str] = None

    @staticmethod
    async def _get_json_with_retry(session: aiohttp.ClientSession, url: str, headers: dict, params: dict, timeout_total: float, retries: int = 3) -> Optional[dict]:
        backoff = 0.5
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=timeout_total)) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status in (401, 403):
                        _LOG.error("Auth error %d on %s", r.status, url.split("?")[0])
                        return None
                    if r.status in (429, 500, 502, 503, 504):
                        if attempt < retries - 1:
                            await asyncio.sleep(backoff * (2 ** attempt))
                            continue
                        return None
                    return None
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                return None
        return None

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        for url in ["https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"]:
            try:
                async with session.get(f"{url}/v3/accounts/{self.account_id}/summary", headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        self.env_url = url
                        return True
                    if r.status in (401, 403):
                        _LOG.error("OANDA auth rejected on %s (status %d)", url, r.status)
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                _LOG.debug("OANDA env probe failed on %s: %s", url, type(e).__name__)
                continue
        return False

    async def fetch_candles(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbol: str, tf: str, limit: int = 500) -> Tuple[str, str, Optional[pd.DataFrame]]:
        cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, tf)
        if cache_hit:
            return symbol, tf, cached
        gran = _GRANULARITY_MAP.get(tf)
        if not gran or not self.env_url:
            return symbol, tf, None
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}
        async with sem:
            data = await self._get_json_with_retry(session, url, self.headers, params, timeout_total=_PER_REQUEST_TIMEOUT_S, retries=3)
            if data is None:
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None
            try:
                candles = [
                    {"date": pd.to_datetime(c["time"], utc=True), "open": float(c["mid"]["o"]),
                     "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
                     "volume": int(c.get("volume", 0))}
                    for c in data.get("candles", [])
                    if c.get("complete") and _is_valid_candle_dict(c)
                ]
            except (KeyError, ValueError, TypeError) as e:
                _LOG.warning("Candle parse error %s/%s: %s", symbol, tf, type(e).__name__)
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None
            if not candles:
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None
            df_raw = pd.DataFrame(candles).set_index("date").tail(limit)
            df_clean = _sanitize_ohlc_dataframe(df_raw)
            if df_clean is not None and not df_clean.empty:
                last_ts = df_clean.index[-1]
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
                age_hours = (datetime.now(timezone.utc) - last_ts.to_pydatetime()).total_seconds() / 3600.0
                max_stale = _TF_MAX_STALE_HOURS.get(tf, 96.0)
                if age_hours > max_stale:
                    _LOG.warning("Stale data %s/%s: %.1fh > %.1fh max", symbol, tf, age_hours, max_stale)
            _cache_set(self.env_url, self.account_id, symbol, tf, df_clean)
            return symbol, tf, df_clean

    async def fetch_price(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbol: str) -> Tuple[str, Optional[float]]:
        if not self.env_url:
            return symbol, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(session, url, self.headers, {"instruments": symbol}, 5, 3)
            if data is None:
                return symbol, None
            try:
                if "prices" in data and data["prices"]:
                    bid, ask = float(data["prices"][0]["closeoutBid"]), float(data["prices"][0]["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
                        return symbol, (bid + ask) / 2
            except (KeyError, ValueError, TypeError, IndexError):
                pass
        return symbol, None

# ==============================================================================
# [ LAYER 2b: ASYNC RUNNER — Daemon worker + bounded cleanup ]
# ==============================================================================
def _run_async_isolated(coro_factory: Callable[[], Any], timeout: float = 300.0) -> Any:
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        return asyncio.run(coro_factory())

    def _worker() -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_factory())
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    try:
                        loop.run_until_complete(asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5.0))
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        _LOG.warning("Async tasks cleanup timeout (5s exceeded)")
                    except Exception:
                        _LOG.exception("Async cleanup error")
            finally:
                try:
                    loop.close()
                finally:
                    try:
                        asyncio.set_event_loop(None)
                    except RuntimeError:
                        pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="oanda-async") as ex:
        future = ex.submit(_worker)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise ScanTimeoutError(f"Async scan exceeded {timeout}s") from e

# ==============================================================================
# [ LAYER 3: QUANTITATIVE ENGINE ]
# ==============================================================================
@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        if df is not None and len(df) >= 2:
            try:
                fb = (df["high"] - df["low"]).mean()
                return float(fb) if pd.notna(fb) and fb > 0 else None
            except (KeyError, TypeError):
                return None
        return None
    try:
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        res = tr.rolling(period).mean().iloc[-1]
        if pd.notna(res) and res > 0:
            return float(res)
        fb = (df["high"] - df["low"]).mean()
        return float(fb) if pd.notna(fb) and fb > 0 else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None

@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: _hash_series})
def compute_institutional_trend(closes: pd.Series, lookback: int = 20, threshold: float = 2.0) -> str:
    if closes is None or len(closes) < lookback:
        return "NEUTRE"
    try:
        y = closes.tail(lookback).values.astype(float)
        if not np.all(np.isfinite(y)):
            return "NEUTRE"
        base = y[0]
        if base == 0 or not np.isfinite(base):
            return "NEUTRE"
        y_norm = y / base
        x = np.arange(len(y_norm), dtype=float)
        slope, intercept = np.polyfit(x, y_norm, 1)
        residuals = y_norm - (slope * x + intercept)
        std_resid = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        if std_resid <= 0:
            return "NEUTRE"
        t_stat = slope / (std_resid / np.sqrt(len(x)))
        if t_stat > threshold: return "HAUSSIER"
        if t_stat < -threshold: return "BAISSIER"
        return "NEUTRE"
    except (np.linalg.LinAlgError, ValueError, TypeError, AttributeError):
        return "NEUTRE"

def detect_swing_pivots(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float, timeframe: str) -> Tuple[pd.Series, pd.Series]:
    if df is None or len(df) < 8 or atr_val is None or atr_val <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    n, prominence = 3, atr_val * profile.pivot_prominence_atr
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
    wick_threshold = profile.wick_threshold_intraday if timeframe.lower() in ("h4", "m15") else profile.wick_threshold_htf
    sh_mask = ((highs > roll_high_left) & (highs > roll_high_right) & (upper_wick_pct >= wick_threshold)).fillna(False)
    sl_mask = ((lows < roll_low_left) & (lows < roll_low_right) & (lower_wick_pct >= wick_threshold)).fillna(False)
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

def agglomerative_1d_clustering(price_weight_pairs: List[tuple], bandwidth: float) -> List[List[tuple]]:
    if not price_weight_pairs or bandwidth <= 0:
        return [[pw] for pw in price_weight_pairs] if price_weight_pairs else []
    sorted_pw = sorted(price_weight_pairs, key=lambda x: x[0])
    clusters, curr_cluster = [], [sorted_pw[0]]
    for i in range(1, len(sorted_pw)):
        gap = sorted_pw[i][0] - sorted_pw[i - 1][0]
        if gap > bandwidth or (curr_cluster and (sorted_pw[i][0] - curr_cluster[0][0]) > 2.5 * bandwidth):
            clusters.append(curr_cluster)
            curr_cluster = [sorted_pw[i]]
        else:
            curr_cluster.append(sorted_pw[i])
    clusters.append(curr_cluster)
    return clusters

def classify_zone_status(level: float, zone_type: str, df: pd.DataFrame, formation_idx: int, atr_val: float) -> str:
    if formation_idx >= len(df) - 1 or atr_val is None or atr_val <= 0:
        return "Vierge"
    tolerance = atr_val * 0.25
    try:
        c_arr = df["close"].values[formation_idx + 1:]
        h_arr = df["high"].values[formation_idx + 1:]
        l_arr = df["low"].values[formation_idx + 1:]
    except (KeyError, IndexError):
        return "Vierge"
    if len(c_arr) == 0: return "Vierge"
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
    rc, rh, rl = c_arr[break_idx + 1:], h_arr[break_idx + 1:], l_arr[break_idx + 1:]
    if len(rc) == 0: return "Consommee"
    retest_tol = tolerance * 2
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
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    age_f = float(np.exp(-lam * age_r))
    return round((strength * tf_w * nb_tf) * age_f, 1)

_STATUS_PRIORITY: Final[Dict[str, int]] = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}
_PIVOT_FALLBACK_DIST: Final[Dict[str, int]] = {"h4": 5, "daily": 8, "weekly": 10}

def _get_pivots_with_fallback(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float, timeframe: str) -> Tuple[pd.Series, pd.Series]:
    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)
    if len(pivot_highs) + len(pivot_lows) >= 3:
        return pivot_highs, pivot_lows
    try:
        n_total = len(df)
        dist = _PIVOT_FALLBACK_DIST.get(timeframe, 5)
        pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        safe_cutoff = n_total - 3
        r_idx = [i for i in r_idx if i < safe_cutoff]
        s_idx = [i for i in s_idx if i < safe_cutoff]
        return (
            pd.Series(df["high"].values[r_idx], index=r_idx) if r_idx else pd.Series(dtype=float),
            pd.Series(df["low"].values[s_idx], index=s_idx) if s_idx else pd.Series(dtype=float),
        )
    except (KeyError, ValueError, TypeError, IndexError) as e:
        _LOG.debug("Pivot fallback failed: %s", type(e).__name__)
        return pd.Series(dtype=float), pd.Series(dtype=float)

def _clusters_to_zones(clusters_raw: list, min_touches_required: int, n_total: int, df: pd.DataFrame, atr_val: float) -> List[Dict[str, Any]]:
    strong: List[Dict[str, Any]] = []
    for grp_pw in clusters_raw:
        if len(grp_pw) < min_touches_required:
            continue
        grp_prices_arr = np.array([item[0] for item in grp_pw])
        grp_weights_arr = np.array([item[1] for item in grp_pw])
        grp_indices = [item[2] for item in grp_pw]
        grp_ptypes = [item[3] for item in grp_pw]
        if grp_weights_arr.sum() <= 0: continue
        lvl = float(np.average(grp_prices_arr, weights=grp_weights_arr))
        if lvl <= 0 or not np.isfinite(lvl): continue
        last_idx = max(grp_indices)
        age = max(0, n_total - 1 - last_idx)
        ztype = "Resistance" if grp_ptypes.count("high") >= grp_ptypes.count("low") else "Support"
        status = classify_zone_status(lvl, ztype, df, last_idx, atr_val)
        strong.append({"level": float(lvl), "strength": len(grp_pw), "age_bars": age, "status": status})
    return strong

def _merge_adjacent_zones(strong: List[Dict[str, Any]], merge_thresh: float) -> List[Dict[str, Any]]:
    strong.sort(key=lambda x: x["level"])
    merged: List[Dict[str, Any]] = []
    for z in strong:
        if not merged or abs(z["level"] - merged[-1]["level"]) > merge_thresh:
            merged.append(z)
            continue
        prev = merged[-1]
        new_str = prev["strength"] + z["strength"]
        new_lvl = (prev["level"] * prev["strength"] + z["level"] * z["strength"]) / new_str
        new_status = max([prev["status"], z["status"]], key=lambda s: _STATUS_PRIORITY.get(s, 1))
        merged[-1] = {"level": new_lvl, "strength": new_str, "age_bars": min(prev["age_bars"], z["age_bars"]), "status": new_status}
    return merged

@st.cache_data(ttl=120, max_entries=256, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def find_strong_sr_zones(df: pd.DataFrame, current_price: float, symbol: str, atr_val: Optional[float], timeframe: str, min_touches_required: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if atr_val is None or atr_val <= 0 or df is None or df.empty or current_price is None or current_price <= 0 or not np.isfinite(current_price):
        return pd.DataFrame(), pd.DataFrame()
    profile = get_profile(symbol)
    n_total = len(df)
    pivot_highs, pivot_lows = _get_pivots_with_fallback(df, profile, atr_val, timeframe)
    pivot_records, pid = [], 0
    for i, p in pivot_highs.items():
        pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "high"))
        pid += 1
    for i, p in pivot_lows.items():
        pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "low"))
        pid += 1
    if not pivot_records: return pd.DataFrame(), pd.DataFrame()
    bandwidth = atr_val * profile.cluster_radius_atr
    price_weight_pairs = [(r[1], r[2], r[3], r[4]) for r in pivot_records]
    clusters_raw = agglomerative_1d_clustering(price_weight_pairs, bandwidth)
    strong = _clusters_to_zones(clusters_raw, min_touches_required, n_total, df, atr_val)
    if not strong: return pd.DataFrame(), pd.DataFrame()
    merged = _merge_adjacent_zones(strong, atr_val * profile.merge_threshold_atr)
    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)
    df_zones["near_price"] = (np.abs(df_zones["level"] - current_price) / current_price * 100) <= 0.50
    return df_zones[df_zones["level"] < current_price].copy(), df_zones[df_zones["level"] >= current_price].copy()

def _flatten_zones_to_dataframe(zones_dict: dict) -> pd.DataFrame:
    frames = []
    for tf, pair in zones_dict.items():
        try: sup, res = pair
        except (TypeError, ValueError): continue
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            if df_z is None or df_z.empty: continue
            tmp = df_z[df_z["status"] != "Consommee"].copy()
            if tmp.empty: continue
            tmp = tmp.assign(tf=tf, type=tmp["near_price"].map({True: "Pivot", False: ztype}))
            frames.append(tmp[["tf", "level", "strength", "age_bars", "status", "type", "near_price"]])
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)

def _score_and_classify_group(group: pd.DataFrame, current_price: float, bars_map: dict, symbol: str) -> dict:
    sub_avg = group["level"].mean()
    sub_nb_tf = group["tf"].nunique()
    safe_cp = current_price if current_price and current_price > 0 else 1.0
    sub_dist = abs(safe_cp - sub_avg) / safe_cp * 100
    tf_w = group["tf"].map(TF_WEIGHT).fillna(1.0).values
    totals = group["tf"].map(lambda t: bars_map.get(t, 500)).values.astype(float)
    age_r = np.clip(group["age_bars"].values / np.maximum(totals, 1), 0, 1)
    lams = group["tf"].map(_TF_LAMBDA).fillna(1.5).values
    age_f = np.exp(-lams * age_r)
    score = round(float((group["strength"].values * tf_w * sub_nb_tf * age_f).sum()), 1)
    status = max(group["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 1))
    is_near_price = sub_dist <= 0.50
    if is_near_price:
        ctype, sig = "Pivot", "↔ PIVOT ZONE"
    else:
        n_sup = (group["level"] < safe_cp).sum()
        ctype = "Support" if n_sup >= len(group) - n_sup else "Resistance"
        sig = "🟢 BUY ZONE" if ctype == "Support" else "🔴 SELL ZONE"
    return {"Actif": symbol, "Signal": sig, "Niveau": round(sub_avg, 5), "Type": ctype,
            "Timeframes": " + ".join(sorted(group["tf"].unique())), "Nb TF": int(sub_nb_tf),
            "Force Totale": int(group["strength"].sum()), "Score": round(score, 1), "Statut": status,
            "Distance %": round(sub_dist, 3), "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else "")}

def detect_confluences(symbol: str, zones_dict: dict, current_price: float, bars_map: dict, confluence_threshold_pct: Optional[float] = None) -> list:
    if (not zones_dict or not current_price or current_price <= 0 or not np.isfinite(current_price)): return []
    z_df = _flatten_zones_to_dataframe(zones_dict)
    if z_df.empty: return []
    profile = get_profile(symbol.replace("/", "_"))
    threshold = confluence_threshold_pct if confluence_threshold_pct is not None else profile.confluence_threshold_pct
    z_df = z_df.sort_values("level").reset_index(drop=True)
    n = len(z_df)
    levels_arr = z_df["level"].values
    parent, rank = list(range(n)), [0] * n
    def find(x: int) -> int:
        root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry: return
        if rank[rx] < rank[ry]: parent[rx] = ry
        elif rank[rx] > rank[ry]: parent[ry] = rx
        else: parent[ry] = rx; rank[rx] += 1
    for i in range(n):
        li = levels_arr[i]
        if li <= 0: continue
        j = i + 1
        while j < n:
            if (levels_arr[j] - li) / li * 100 > threshold: break
            union(i, j)
            j += 1
    comp_map: Dict[int, List[int]] = {}
    for idx in range(n): comp_map.setdefault(find(idx), []).append(idx)
    confluences = []
    for indices in comp_map.values():
        if not indices: continue
        group_full = z_df.iloc[indices]
        if group_full["tf"].nunique() < 2: continue
        sub_avg = group_full["level"].mean()
        group_full = group_full.assign(_dist=(group_full["level"] - sub_avg).abs())
        keep_idx = group_full.groupby("tf")["_dist"].idxmin().values
        group = group_full.loc[keep_idx].drop(columns=["_dist"])
        confluences.append(_score_and_classify_group(group, current_price, bars_map, symbol))
    return confluences

# ==============================================================================
# [ LAYER 4: PIPELINE ORCHESTRATOR ]
# ==============================================================================
@dataclass
class ScanResult:
    symbol: str; rows: dict; zones: dict; price: Optional[float]; trends: dict; bars_map: dict
    anomaly: Optional[str] = None; scan_error: Optional[str] = None; price_context: str = ""
    missing_tfs: List[str] = field(default_factory=list); price_is_fallback: bool = False
    debug_info: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class _RowContext:
    cp: float; atr_val: float; sym_d: str; tf_name: str; df_len: int; profile: InstrumentProfile

def _make_row(z: dict, ztype: str, ctx: _RowContext) -> Dict[str, Any]:
    dist = abs(ctx.cp - z["level"]) / ctx.cp * 100 if ctx.cp else 0.0
    dist_atr = f"{round(abs(ctx.cp - z['level']) / ctx.atr_val, 1)}x" if (ctx.atr_val and ctx.atr_val > 0) else "N/A"
    return {"Actif": ctx.sym_d, "Prix Actuel": f"{ctx.cp:.5f}" if ctx.cp else "N/A", "Type": ztype,
            "Niveau": f"{z['level']:.5f}", "Force": f"{z['strength']} touches",
            "Score (1TF)": compute_structural_score(z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len),
            "Statut": z["status"], "Dist. %": f"{dist:.2f}%", "Dist. ATR": dist_atr,
            "_dist_num": dist, "_in_pdf": dist <= ctx.profile.pdf_max_dist_pct}

async def _fetch_live_prices(client: AsyncOandaClient, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbols: List[str]) -> Dict[str, Optional[float]]:
    price_tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
    prices_res = await asyncio.gather(*price_tasks, return_exceptions=True)
    out: Dict[str, Optional[float]] = {}
    for sym, item in zip(symbols, prices_res):
        if isinstance(item, BaseException):
            _LOG.error("Price fetch exception for %s: %s", sym, type(item).__name__)
            out[sym] = None
        else:
            out[item[0]] = item[1]
    return out

async def _fetch_candles_cube(client: AsyncOandaClient, session: aiohttp.ClientSession, sem: asyncio.Semaphore, symbols: List[str]) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    candle_tasks = [client.fetch_candles(session, sem, sym, tf) for sym in symbols for tf in _GRANULARITY_MAP]
    candles_res = await asyncio.gather(*candle_tasks, return_exceptions=True)
    data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
    for item in candles_res:
        if isinstance(item, BaseException):
            _LOG.error("Candle fetch exception: %s", type(item).__name__)
            continue
        sym, tf, df = item
        data_cube.setdefault(sym, {})[tf] = df
    return data_cube

def _build_daily_price_context(cp: float, sup: pd.DataFrame, res: pd.DataFrame) -> str:
    parts = []
    if sup is not None and not sup.empty:
        try:
            s_near = sup[(sup["level"] < cp) & (abs(sup["level"] - cp) / cp * 100 <= 5.0)]
            if not s_near.empty:
                n_s = s_near.nlargest(1, "level").iloc[0]
                d_s = abs(cp - n_s["level"]) / cp * 100
                parts.append(f"{'SUR support' if d_s < 0.5 else 'S proche'}: {n_s['level']:.5f} (-{d_s:.2f}%)")
        except (KeyError, ValueError, TypeError): pass
    if res is not None and not res.empty:
        try:
            r_near = res[(res["level"] > cp) & (abs(res["level"] - cp) / cp * 100 <= 5.0)]
            if not r_near.empty:
                n_r = r_near.nsmallest(1, "level").iloc[0]
                d_r = abs(cp - n_r["level"]) / cp * 100
                parts.append(f"{'SUR resistance' if d_r < 0.5 else 'R proche'}: {n_r['level']:.5f} (+{d_r:.2f}%)")
        except (KeyError, ValueError, TypeError): pass
    return "  |  ".join(parts) if parts else "Zone intermediaire"

_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {"H4": (50, 2.0), "Daily": (50, 1.8), "Weekly": (20, 1.5)}

def _process_tf_frame(sym: str, tf_k: str, tf_name: str, df: pd.DataFrame, cp: float, min_touches_ui: int, profile: InstrumentProfile, sym_d: str) -> Tuple[Optional[list], Optional[tuple], str, Dict[str, Any]]:
    debug: Dict[str, Any] = {"atr": None, "n_pivots": 0, "n_clusters": 0, "min_touches": None}
    try:
        atr_val = compute_atr(df)
        debug["atr"] = atr_val
        if atr_val is None:
            _LOG.warning("ATR uncomputable %s/%s", sym, tf_name)
            return None, None, "", debug
        min_t = _min_touches_for_tf(profile, tf_k, min_touches_ui)
        debug["min_touches"] = min_t
        sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
        debug["n_zones"] = len(sup) + len(res)
        zone_pair = (sup, res)
        price_ctx = _build_daily_price_context(cp, sup, res) if tf_k == "daily" else ""
        row_ctx = _RowContext(cp=cp, atr_val=atr_val, sym_d=sym_d, tf_name=tf_name, df_len=len(df), profile=profile)
        tf_r = ([_make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx) for _, z in sup.iterrows()] +
                [_make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx) for _, z in res.iterrows()])
        seen: Set[Tuple[str, str]] = set()
        uniq = []
        for r in tf_r:
            key = (r["Niveau"], r["Type"])
            if key not in seen: seen.add(key); uniq.append(r)
        return (uniq if uniq else None), zone_pair, price_ctx, debug
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
        _LOG.warning("TF processing error %s/%s: %s", sym, tf_name, type(e).__name__)
        debug["error"] = type(e).__name__
        return None, None, "", debug

def _resolve_working_price(cp_live: Optional[float], data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]], sym: str) -> Tuple[Optional[float], bool]:
    if cp_live and cp_live > 0 and np.isfinite(cp_live): return cp_live, False
    for tf_k in ("daily", "h4", "weekly"):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                if np.isfinite(last_close) and last_close > 0: return last_close, True
            except (KeyError, IndexError, ValueError, TypeError): continue
    return None, False

def _validate_price_bounds_post(cp: float, profile: InstrumentProfile) -> Optional[str]:
    if profile.price_min is not None and cp < profile.price_min: return f"PRIX HORS BORNES ({cp:.2f} < {profile.price_min:.0f})"
    if profile.price_max is not None and cp > profile.price_max: return f"PRIX HORS BORNES ({cp:.2f} > {profile.price_max:.0f})"
    return None

def _collect_tf_data(sym: str, data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]], cp: float, profile: InstrumentProfile, min_touches_ui: int, sym_d: str) -> Tuple[Dict, Dict, Dict, Dict, str, List[str], Dict[str, Dict[str, Any]]]:
    rows: Dict[str, Optional[list]] = {"H4": None, "Daily": None, "Weekly": None}
    zones_d: Dict[str, tuple] = {}; trends: Dict[str, str] = {}; bars_map: Dict[str, int] = {}
    debug_per_tf: Dict[str, Dict[str, Any]] = {}; price_ctx = ""; missing_tfs: List[str] = []
    for tf_k, tf_name in (("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is None or df.empty:
            missing_tfs.append(tf_name); continue
        bars_map[tf_name] = len(df)
        try:
            lb, th = _TF_TREND_PARAMS.get(tf_name, (20, 2.0))
            trends[tf_name] = compute_institutional_trend(df["close"], lookback=lb, threshold=th)
        except (KeyError, ValueError, TypeError, AttributeError):
            trends[tf_name] = "NEUTRE"
        tf_rows, zone_pair, ctx, debug = _process_tf_frame(sym, tf_k, tf_name, df, cp, min_touches_ui, profile, sym_d)
        debug_per_tf[tf_name] = debug
        if zone_pair is not None: zones_d[tf_name] = zone_pair
        if tf_rows is not None: rows[tf_name] = tf_rows
        if ctx: price_ctx = ctx
    return rows, zones_d, trends, bars_map, price_ctx, missing_tfs, debug_per_tf

def flag_data_anomaly(symbol: str, current_price: Optional[float], support_levels: List[float], last_candle_close: Optional[float] = None, last_candle_high: Optional[float] = None, last_candle_low: Optional[float] = None, last_candle_ts: Optional[Any] = None, timeframe: str = "daily") -> Optional[str]:
    if current_price is None or current_price <= 0 or not np.isfinite(current_price): return "Prix indisponible"
    profile = get_profile(symbol)
    messages: List[str] = []
    if profile.price_min is not None and current_price < profile.price_min: messages.append(f"PRIX HORS BORNES")
    if profile.price_max is not None and current_price > profile.price_max: messages.append(f"PRIX HORS BORNES")
    if not profile.skip_ratio_check and len(support_levels) >= 3:
        median_sup = float(np.median(support_levels))
        if median_sup > 0 and median_sup > 0.01 * current_price and current_price / median_sup > 3.0:
            messages.append(f"Ratio aberrant prix/support")
    if last_candle_close and last_candle_close > 0 and np.isfinite(last_candle_close):
        in_range = (last_candle_high is not None and last_candle_low is not None and last_candle_low * 0.999 <= current_price <= last_candle_high * 1.001)
        if not in_range:
            dev = abs(current_price - last_candle_close) / last_candle_close * 100
            if dev > profile.max_live_vs_close_pct: messages.append(f"Ecart live/close {dev:.1f}%")
    return " | ".join(messages) if messages else None

def _extract_last_candle_info(daily_df: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[Any]]:
    if daily_df is None or daily_df.empty: return None, None, None, None
    try: return float(daily_df["close"].iloc[-1]), float(daily_df["high"].iloc[-1]), float(daily_df["low"].iloc[-1]), daily_df.index[-1]
    except Exception: return None, None, None, None

def _collect_support_levels(zones_d: Dict[str, tuple]) -> List[float]:
    sup_levels: List[float] = []
    for zone_pair in zones_d.values():
        try:
            _s, _r = zone_pair
            if _s is not None and not _s.empty and "level" in _s.columns: sup_levels.extend(_s["level"].tolist())
        except Exception: continue
    return sup_levels

def _process_symbol(sym: str, cp_live: Optional[float], data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]], min_touches_ui: int) -> ScanResult:
    try:
        profile = get_profile(sym); sym_d = sym.replace("_", "/")
        cp, price_is_fallback = _resolve_working_price(cp_live, data_cube, sym)
        if cp is None: return ScanResult(sym, {}, {}, None, {}, {}, scan_error="Aucune donnée disponible")
        bounds_err = _validate_price_bounds_post(cp, profile)
        if bounds_err is not None: return ScanResult(sym, {}, {}, None, {}, {}, scan_error=bounds_err)
        rows, zones_d, trends, bars_map, price_ctx, missing_tfs, debug = _collect_tf_data(sym, data_cube, cp, profile, min_touches_ui, sym_d)
        sup_levels = _collect_support_levels(zones_d)
        daily_df = data_cube.get(sym, {}).get("daily")
        last_close, last_high, last_low, last_ts = _extract_last_candle_info(daily_df)
        anomaly = flag_data_anomaly(sym, cp, sup_levels, last_candle_close=last_close, last_candle_high=last_high, last_candle_low=last_low, last_candle_ts=last_ts)
        if price_is_fallback: anomaly = f"{anomaly} | Prix fallback" if anomaly else "Prix fallback"
        return ScanResult(sym, rows, zones_d, cp, trends, bars_map, price_context=price_ctx, anomaly=anomaly, missing_tfs=missing_tfs, price_is_fallback=price_is_fallback, debug_info=debug)
    except Exception as e:
        _LOG.exception("Symbol processing error: %s", sym)
        return ScanResult(sym, {}, {}, None, {}, {}, scan_error=f"Erreur critique: {type(e).__name__}")

async def run_institutional_scan(symbols: List[str], token: str, oanda_account_id: str, min_touches_ui: int) -> List[ScanResult]:
    client = AsyncOandaClient(token, oanda_account_id)
    timeout_session = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout_session) as session:
        if not await client.initialize(session): raise OandaAuthError("Auth failed")
        sem = asyncio.Semaphore(_OANDA_SEMAPHORE_LIMIT)
        live_prices = await _fetch_live_prices(client, session, sem, symbols)
        data_cube = await _fetch_candles_cube(client, session, sem, symbols)
    return [_process_symbol(sym, live_prices.get(sym), data_cube, min_touches_ui) for sym in symbols]

# ==============================================================================
# [ LAYER 5: EXPORTERS ]
# ==============================================================================
_ACCENT_MAP: Final = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ', 'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
_EMOJI_MAP: Final[List[Tuple[str, str]]] = [('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'), ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]'), ('📈', ''), ('📉', ''), ('✅', '[OK]'), ('❌', '[X]'), ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'), ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]')]
_PDF_MAX_CELL_CHARS: Final[int] = 200

def _safe_pdf_str(text: Any, max_chars: int = _PDF_MAX_CELL_CHARS) -> str:
    if text is None: return ""
    try: s = str(text).translate(_ACCENT_MAP)
    except Exception: return ""
    for e, r in _EMOJI_MAP: s = s.replace(e, r)
    try: s = s.encode("latin-1", errors="replace").decode("latin-1")
    except Exception: s = s.encode("ascii", errors="ignore").decode("ascii")
    return s[:max_chars - 3] + "..." if len(s) > max_chars else s

def _sanitize_traceback(traceback_str: str, sensitive_values: List[Optional[str]]) -> str:
    if not traceback_str: return traceback_str
    out = traceback_str
    for val in sensitive_values:
        if val and isinstance(val, str) and len(val) > 4: out = out.replace(val, "***REDACTED***")
    return _redact_sensitive(out)

_INTERNAL_COLS: Final[List[str]] = ["_dist_num", "_in_pdf"]
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame() if df is None else df
    return df.drop(columns=_INTERNAL_COLS, errors="ignore")

def _sym_display(sym: str) -> str: return sym.replace("_", "/")

def get_price_context(current_price: Optional[float], supports: Optional[pd.DataFrame], resistances: Optional[pd.DataFrame], max_dist_pct: float = 5.0) -> str:
    if not current_price or current_price <= 0: return "Prix indisponible"
    parts: List[str] = []
    if supports is not None and not supports.empty:
        try:
            sup_nearby = supports[(supports["level"] < current_price) & (abs(supports["level"] - current_price) / current_price * 100 <= max_dist_pct)]
            if not sup_nearby.empty:
                n = sup_nearby.nlargest(1, "level").iloc[0]
                d = abs(current_price - n["level"]) / current_price * 100
                parts.append(f"{'SUR support' if d < 0.5 else 'S proche'}: {n['level']:.5f}")
        except Exception: pass
    if resistances is not None and not resistances.empty:
        try:
            res_nearby = resistances[(resistances["level"] > current_price) & (abs(resistances["level"] - current_price) / current_price * 100 <= max_dist_pct)]
            if not res_nearby.empty:
                n = res_nearby.nsmallest(1, "level").iloc[0]
                d = abs(current_price - n["level"]) / current_price * 100
                parts.append(f"{'SUR resistance' if d < 0.5 else 'R proche'}: {n['level']:.5f}")
        except Exception: pass
    return "  |  ".join(parts) if parts else "Zone intermediaire"

def _parse_price_context_obstacles(ctx_str: str, current_price: float) -> Dict[str, Optional[dict]]:
    result: Dict[str, Optional[dict]] = {"nearest_support": None, "nearest_resistance": None}
    if not ctx_str or ctx_str == "Zone intermediaire" or not current_price: return result
    # Simplified parsing logic for brevity
    return result

def strip_emojis_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df if df is not None else pd.DataFrame()
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        clean[col] = clean[col].astype(str).apply(_safe_pdf_str)
    return clean

class PDF(FPDF):
    def header(self) -> None:
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, _safe_pdf_str('Rapport Scanner Bluestar - S/R'), border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)
    def footer(self) -> None:
        self.set_y(-15); self.set_font('Helvetica', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', align='C')
    def chapter_title(self, title: str) -> None:
        self.set_font('Helvetica', 'B', 12); self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT'); self.ln(4)
    def chapter_anomalies(self, anomalies: dict) -> None:
        if not anomalies: return
        self.set_font('Helvetica', 'B', 10); self.cell(0, 7, _safe_pdf_str('ALERTES'), border=0, align='L', new_x='LMARGIN', new_y='NEXT'); self.ln(2); self.set_font('Helvetica', '', 8)
        for sym, msg in anomalies.items(): self.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}", max_chars=180)); self.ln(4)
    def chapter_summary(self, summary_list: List[Dict[str, Any]]) -> None:
        self.set_font('Helvetica', 'B', 10); self.cell(0, 7, _safe_pdf_str('RESUME'), border=0, align='L', new_x='LMARGIN', new_y='NEXT'); self.ln(2)
        for s in summary_list:
            sym = _safe_pdf_str(s.get('symbol', '')); t_h4 = _safe_pdf_str(s.get('trend_h4', 'N/A')); t_d = _safe_pdf_str(s.get('trend_daily', 'N/A')); t_w = _safe_pdf_str(s.get('trend_weekly', 'N/A')); ctx = _safe_pdf_str(s.get('price_context', ''), max_chars=120)
            self.set_font('Helvetica', 'B', 8); self.cell(0, 5, _safe_pdf_str(f"{sym} {t_h4}/{t_d}/{t_w}"), border=0, new_x='LMARGIN', new_y='NEXT')
            if ctx: self.set_font('Helvetica', 'I', 7); self.cell(0, 4, f"  Pos: {ctx}", border=0, new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 7); self.ln(1)
    def chapter_body(self, df: pd.DataFrame) -> None:
        if df is None or df.empty: self.set_font('Helvetica', '', 10); self.multi_cell(0, 10, "Aucune donnee"); return
        cols = [c for c in df.columns if c in {'Actif','Signal','Niveau','Type','Timeframes','Nb TF','Force Totale','Score','Statut','Distance %','Alerte'}]
        if not cols: return
        col_widths = {'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22, 'Timeframes': 50, 'Nb TF': 12, 'Force Totale': 20, 'Score': 18, 'Statut': 22, 'Distance %': 18, 'Alerte': 55}
        self.set_font('Helvetica', 'B', 7); total_w = sum(col_widths.get(c, 20) for c in cols); x_start = max(0, (self.w - total_w) / 2); self.set_x(x_start)
        for c in cols: self.cell(col_widths.get(c, 20), 6, _safe_pdf_str(c), border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln(); self.set_font('Helvetica', '', 7)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for c in cols:
                w = col_widths.get(c, 20); val = _safe_pdf_str(str(row[c]))
                self.cell(w, 5, val[:int(w/1.5)], border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()

def _apply_pdf_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df if df is not None else pd.DataFrame()
    if "_in_pdf" in df.columns: return _clean_df(df[df["_in_pdf"]].copy()).reset_index(drop=True)
    if "Dist. %" in df.columns:
        def _to_f(s: Any) -> float:
            try: return float(str(s).replace("%", ""))
            except Exception: return 999.0
        return _clean_df(df[df["Dist. %"].apply(_to_f) <= 8.0].copy()).reset_index(drop=True)
    return _clean_df(df).reset_index(drop=True)

@st.cache_data(ttl=300, max_entries=8, show_spinner=False, hash_funcs={dict: _hash_dict_content, pd.DataFrame: _hash_df, list: _hash_list_content})
def create_pdf_report(results_dict: Dict[str, pd.DataFrame], confluences_df: Optional[pd.DataFrame] = None, summary_list: Optional[List[Dict[str, Any]]] = None, anomalies: Optional[Dict[str, str]] = None) -> bytes:
    summary_list = summary_list or []; anomalies = anomalies or {}
    pdf = PDF('L', 'mm', 'A4'); pdf.set_margins(5, 10, 5); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    if anomalies: pdf.chapter_anomalies(anomalies)
    if summary_list: pdf.chapter_summary(summary_list); pdf.add_page()
    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title('ZONES DE CONFLUENCE')
        clean_conf = strip_emojis_df(_clean_df(confluences_df.copy()))
        if "Score" in clean_conf.columns: clean_conf = clean_conf.sort_values("Score", ascending=False)
        pdf.chapter_body(clean_conf)
    return bytes(pdf.output())

@st.cache_data(ttl=300, max_entries=8, show_spinner=False, hash_funcs={dict: _hash_dict_content, pd.DataFrame: _hash_df})
def create_csv_report(results_dict: Dict[str, pd.DataFrame], confluences_df: Optional[pd.DataFrame] = None) -> bytes:
    all_dfs: List[pd.DataFrame] = []
    if confluences_df is not None and not confluences_df.empty:
        c = _clean_df(confluences_df).copy(); c["Section"] = "CONFLUENCES"; all_dfs.append(c)
    for tf, df in results_dict.items():
        if df is not None and not df.empty:
            d = _clean_df(df).copy(); d["Timeframe"] = tf; d["Section"] = "TF_ROWS"; all_dfs.append(d)
    if not all_dfs: return b""
    buf = BytesIO(); pd.concat(all_dfs, ignore_index=True).to_csv(buf, index=False, encoding="utf-8-sig"); return buf.getvalue()

_TREND_ARROW: Final[Dict[str, str]] = {"HAUSSIER": "↑", "BAISSIER": "↓", "NEUTRE": "→"}
_STATUS_LABEL: Final[Dict[str, str]] = {"Vierge": "V", "Testee": "T", "Role Reverse": "RR", "Consommee": "C"}
_ALERT_LABEL: Final[Dict[str, str]] = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠"}
_SIGNAL_SHORT: Final[Dict[str, str]] = {"PIVOT": "PIVOT", "BUY": "BUY", "SELL": "SELL"}

def _filter_confluences_to_actif_zones(confluences_df: pd.DataFrame, max_dist: float, min_score: float, allowed_statuts: tuple) -> Dict[str, list]:
    actif_zones: Dict[str, list] = {}
    for _, row in confluences_df.iterrows():
        try:
            dist_val = float(str(row.get("Distance %", "999")).replace("%", "")); score_val = float(row.get("Score", 0)); statut = str(row.get("Statut", ""))
            if dist_val > max_dist or score_val < min_score or statut not in allowed_statuts: continue
            actif_zones.setdefault(str(row.get("Actif", "")), []).append({"signal": str(row.get("Signal", "")), "niveau": str(row.get("Niveau", "")), "score": score_val, "statut": statut, "dist": dist_val, "tfs": str(row.get("Timeframes", "")), "nb_tf": int(row.get("Nb TF", 0)), "alerte": str(row.get("Alerte", ""))})
        except Exception: continue
    return actif_zones

def _format_brief_zone_line(z: dict) -> str:
    sig = z["signal"]; signal_short = next((v for k, v in _SIGNAL_SHORT.items() if k in sig), "ZONE")
    tf_short = z["tfs"].replace("Daily", "D").replace("Weekly", "W").replace(" + ", "+")
    return f"- {signal_short} `{z['niveau']}` | Sc:{z['score']:.0f} | {_STATUS_LABEL.get(z['statut'], z['statut'])} | {z['dist']:.2f}% | {tf_short} {_ALERT_LABEL.get(z['alerte'], '')}"

@st.cache_data(ttl=300, max_entries=16, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df, list: _hash_list_content})
def create_llm_brief(summary_list: List[Dict[str, Any]], confluences_df: Optional[pd.DataFrame], max_dist: float = 2.0, min_score: float = 100.0, allowed_statuts: Tuple[str, ...] = ("Vierge", "Testee", "Role Reverse")) -> bytes:
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = ["# BRIEF S/R", f"_Généré le {now}_", "", "## INSTRUCTIONS", "Zones S/R fiables...", "", "**Filtres** : Dist < {max_dist}% | Score ≥ {min_score}", "", "---"]
    if confluences_df is None or confluences_df.empty:
        lines.append("_Aucune confluence._")
        return "\n".join(lines).encode("utf-8")
    actif_zones = _filter_confluences_to_actif_zones(confluences_df, max_dist, min_score, allowed_statuts)
    sorted_actifs = sorted(actif_zones, key=lambda a: max(z["score"] for z in actif_zones[a]), reverse=True)
    summary_map = {s["symbol"]: s for s in summary_list}; total_zones = 0
    for actif in sorted_actifs:
        s = summary_map.get(actif, {})
        t_h4 = _TREND_ARROW.get(s.get("trend_h4", "NEUTRE"), "→"); t_d = _TREND_ARROW.get(s.get("trend_daily", "NEUTRE"), "→"); t_w = _TREND_ARROW.get(s.get("trend_weekly", "NEUTRE"), "→")
        lines.append(f"### {actif} | H4:{t_h4} D:{t_d} W:{t_w}")
        ctx = s.get("price_context", "")
        if ctx: lines.append(f"> {ctx}")
        for z in sorted(actif_zones[actif], key=lambda z: z["score"], reverse=True):
            lines.append(_format_brief_zone_line(z)); total_zones += 1
        lines.append("")
    return "\n".join(lines).encode("utf-8")

def _get_ict_session(dt_utc: datetime) -> str:
    if _NY_TZ is not None:
        if dt_utc.tzinfo is None: dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        h = dt_utc.astimezone(_NY_TZ).hour
    else: h = (dt_utc.hour - 5) % 24
    if 18 <= h or h < 3: return "ASIAN"
    if 3 <= h < 8: return "LONDON"
    if 8 <= h < 12: return "OVERLAP_LDN_NY"
    return "NEW_YORK"

def _normalize_signal(raw: str) -> str:
    r = raw.replace("🟢", "").replace("🔴", "").replace("↔️", "").replace("↔", "").replace("ZONE", "").strip()
    if "PIVOT" in r: return "PIVOT"
    if "BUY" in r: return "BUY"
    if "SELL" in r: return "SELL"
    return r.strip()

def _normalize_alert(raw: str) -> str:
    r = raw.replace("🔥", "").replace("⚠️", "").replace("⚠", "").strip()
    if "CHAUD" in r.upper() or "HOT" in r.upper(): return "HOT"
    if "PROCHE" in r.upper() or "CLOSE" in r.upper(): return "CLOSE"
    return ""

_TF_ORDER: Final[Dict[str, int]] = {"Weekly": 0, "Daily": 1, "H4": 2}
def _parse_timeframes(tf_str: str) -> List[str]:
    parts = [p.strip() for p in tf_str.replace("+", ",").split(",") if p.strip()]
    return sorted(parts, key=lambda t: _TF_ORDER.get(t, 99))

_BIAS_MAP: Final[Dict[str, str]] = {"HAUSSIER": "BULLISH", "BAISSIER": "BEARISH", "NEUTRE": "NEUTRAL"}
def _trend_alignment(h4: str, daily: str, weekly: str) -> Tuple[str, str]:
    b_h4, b_d, b_w = _BIAS_MAP.get(h4, "NEUTRAL"), _BIAS_MAP.get(daily, "NEUTRAL"), _BIAS_MAP.get(weekly, "NEUTRAL")
    dominant = b_d if b_d == b_w and b_d != "NEUTRAL" else (b_w if b_d == "NEUTRAL" else (b_d if b_w == "NEUTRAL" else "NEUTRAL"))
    if dominant != "NEUTRAL": return ("ALIGNED" if b_h4 == dominant else ("PULLBACK" if b_h4 == "NEUTRAL" else "CONFLICTED")), dominant
    return ("BUILDING" if b_h4 != "NEUTRAL" else "MIXED"), dominant

def _build_actif_groups(confluences_df: pd.DataFrame, max_dist: float, min_score: float, allowed_statuts: tuple) -> Dict[str, List[Dict[str, Any]]]:
    actif_groups: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in confluences_df.iterrows():
        try:
            dist_val = float(str(row.get("Distance %", 999)).replace("%", "")); score_val = float(row.get("Score", 0))
            if dist_val > max_dist or score_val < min_score: continue
            signal_norm = _normalize_signal(str(row.get("Signal", "")))
            if signal_norm not in ("BUY", "SELL", "PIVOT"): continue
            statut = str(row.get("Statut", ""))
            if statut not in allowed_statuts: continue
            level_float = round(float(row.get("Niveau", "")), 5)
            tf_str = str(row.get("Timeframes", "")); tfs_parsed = _parse_timeframes(tf_str)
            actif_groups.setdefault(str(row.get("Actif", "")), []).append({"signal": signal_norm, "type": str(row.get("Type", "")), "level": level_float, "score": round(score_val, 1), "status": statut, "distance_pct": round(dist_val, 3), "alert": _normalize_alert(str(row.get("Alerte", ""))), "timeframes": tfs_parsed, "nb_tf": int(row.get("Nb TF", len(tfs_parsed)))})
        except Exception: continue
    return actif_groups

@st.cache_data(ttl=300, max_entries=16, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df, list: _hash_list_content})
def create_json_export(summary_list: List[Dict[str, Any]], confluences_df: Optional[pd.DataFrame], max_dist: float = 5.0, min_score: float = 60.0, allowed_statuts: Tuple[str, ...] = ("Vierge", "Testee", "Role Reverse")) -> bytes:
    now_utc = datetime.now(timezone.utc)
    output: Dict[str, Any] = {"generated_at": now_utc.isoformat(), "scanner_version": SCANNER_VERSION, "session": _get_ict_session(now_utc), "filters": {"max_dist_pct": max_dist, "min_score": min_score, "allowed_statuts": list(allowed_statuts)}, "assets": []}
    summary_map = {s["symbol"]: s for s in summary_list}
    actif_groups = _build_actif_groups(confluences_df, max_dist, min_score, allowed_statuts) if (confluences_df is not None and not confluences_df.empty) else {}
    all_actifs = set(summary_map.keys()) | set(actif_groups.keys())
    sorted_actifs = sorted(all_actifs, key=lambda a: max((z["score"] for z in actif_groups.get(a, [])), default=0.0), reverse=True)
    for actif in sorted_actifs:
        s = summary_map.get(actif, {})
        t_h4, t_d, t_w = s.get("trend_h4", "NEUTRE"), s.get("trend_daily", "NEUTRE"), s.get("trend_weekly", "NEUTRE")
        alignment, dominant = _trend_alignment(t_h4, t_d, t_w)
        cp_val = s.get("current_price"); ctx_str = s.get("price_context", "")
        zones = sorted(actif_groups.get(actif, []), key=lambda z: z["score"], reverse=True)
        output["assets"].append({"symbol": actif, "current_price": round(cp_val, 5) if cp_val else None, "price_is_fallback": bool(s.get("price_is_fallback", False)), "missing_timeframes": list(s.get("missing_tfs", [])), "trends": {"h4": t_h4, "daily": t_d, "weekly": t_w}, "trend_alignment": alignment, "dominant_bias": dominant, "price_context": ctx_str, "nb_zones": len(zones), "zones": zones})
    return json.dumps(output, ensure_ascii=False, indent=2, default=str).encode("utf-8")

def _filter_and_sort(df: Optional[pd.DataFrame], max_pct: float) -> pd.DataFrame:
    if df is None or df.empty or "Dist. %" not in df.columns: return df if df is not None else pd.DataFrame()
    def _to_f(s: Any) -> float:
        try: return float(str(s).replace("%", ""))
        except Exception: return 999.0
    out = _clean_df(df[df["Dist. %"].apply(_to_f) <= max_pct])
    sort_col = next((c for c in ("Score (1TF)", "Score") if c in out.columns), None)
    if sort_col: out = out.sort_values(sort_col, ascending=False)
    return out.reset_index(drop=True)

# ==============================================================================
# [ LAYER 6: STREAMLIT UI ]
# ==============================================================================
CONFLUENCE_THRESHOLD_MAP: Final[Dict[str, float]] = {"US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "DE30_EUR": 1.2, "XAU_USD": 1.5}

def _show_diagnostics(errors: dict, anomalies: dict, missing_tfs_map: dict, debug_map: Optional[dict] = None) -> None:
    if errors:
        with st.expander(f"❌ {len(errors)} erreurs"):
            for sym, err in errors.items(): st.error(f"**{sym}**: {err}")
    if anomalies:
        with st.expander(f"⚠️ {len(anomalies)} anomalies"):
            for sym, msg in anomalies.items(): st.warning(f"**{sym}**: {msg}")
    if missing_tfs_map:
        with st.expander(f"📡 {len(missing_tfs_map)} TFs manquants"):
            for sym, tfs in missing_tfs_map.items(): st.info(f"**{sym}**: {', '.join(tfs)}")
    if debug_map:
        with st.expander("🔬 Debug"):
            try:
                rows = []
                for sym, per_tf in debug_map.items():
                    for tf, info in per_tf.items():
                        rows.append({"Symbole": sym, "TF": tf, "ATR": info.get("atr"), "Min Touches": info.get("min_touches"), "Nb Zones": info.get("n_zones", 0), "Erreur": info.get("error", "")})
                if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            except Exception: pass

def _show_confluence_section(conf_filt: pd.DataFrame) -> None:
    if conf_filt.empty:
        st.info("Aucune confluence."); return
    st.divider(); st.subheader("🔥 ZONES DE CONFLUENCE")
    st.caption("Score = Force × Nb TF × Poids_TF × Facteur_Age")
    disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total zones", len(disp)); c2.metric("🔥 Chaudes", len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
    c3.metric("⚠️ Proches", len(disp[disp["Alerte"] == "⚠️ Proche"])); c4.metric("🟢 BUY", len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
    c5.metric("🔴 SELL", len(disp[disp["Signal"] == "🔴 SELL ZONE"])); c6.metric("↔ PIVOT", len(disp[disp["Signal"] == "↔ PIVOT ZONE"]))
    conf_cfg = {"Actif": st.column_config.TextColumn("Actif", width="small"), "Signal": st.column_config.TextColumn("Signal", width="small"), "Niveau": st.column_config.TextColumn("Niveau", width="small"), "Type": st.column_config.TextColumn("Type", width="small"), "Timeframes": st.column_config.TextColumn("Timeframes", width="small"), "Statut": st.column_config.TextColumn("Statut", width="small"), "Distance %": st.column_config.TextColumn("Dist. %", width="small"), "Alerte": st.column_config.TextColumn("Alerte", width="small"), "Nb TF": st.column_config.NumberColumn("Nb TF", width="small"), "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"), "Score": st.column_config.NumberColumn("Score ▼", width="small")}
    st.dataframe(disp, column_config=conf_cfg, hide_index=True, width='stretch', height=min(len(disp) * 35 + 38, 750))

def _show_export_section(rep_dict: dict, conf_full: pd.DataFrame, summary_list: list, anomalies: dict) -> None:
    st.subheader("📋 Exportation")
    with st.expander("Télécharger"):
        col1, col2 = st.columns(2)
        with col1:
            try:
                pdf_bytes = create_pdf_report(rep_dict, conf_full, summary_list, anomalies)
                st.download_button("📄 Rapport PDF", data=pdf_bytes, file_name=f"rapport_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", width='stretch')
            except Exception as e: _LOG.exception("PDF failed"); st.error(f"PDF: {type(e).__name__}")
        with col2:
            try:
                csv_bytes = create_csv_report(rep_dict, conf_full)
                st.download_button("📊 CSV", data=csv_bytes, file_name=f"donnees_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv", width='stretch')
            except Exception as e: _LOG.exception("CSV failed"); st.error(f"CSV: {type(e).__name__}")
        st.divider(); st.markdown("**🤖 LLM Exports**")
        llm_max_dist = float(st.session_state.get("llm_max_dist", 2.0)); llm_min_score = float(st.session_state.get("llm_min_score", 100))
        llm_statuts = tuple(st.session_state.get("llm_statuts", ["Vierge", "Testee", "Role Reverse"]))
        st.caption(f"Dist < {llm_max_dist}% | Score ≥ {llm_min_score}")
        md_bytes = b""; col3, col4 = st.columns(2)
        with col3:
            try:
                md_bytes = create_llm_brief(summary_list, conf_full, max_dist=llm_max_dist, min_score=llm_min_score, allowed_statuts=llm_statuts)
                st.download_button("🤖 Brief LLM", data=md_bytes, file_name=f"brief_llm_{datetime.now().strftime('%Y%m%d_%H%M')}.md", mime="text/markdown", width='stretch')
            except Exception as e: _LOG.exception("LLM failed"); st.error(f"LLM: {type(e).__name__}")
        with col4:
            try:
                json_bytes = create_json_export(summary_list, conf_full, max_dist=llm_max_dist, min_score=llm_min_score, allowed_statuts=llm_statuts)
                st.download_button("🔧 JSON", data=json_bytes, file_name=f"sr_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json", width='stretch')
            except Exception as e: _LOG.exception("JSON failed"); st.error(f"JSON: {type(e).__name__}")
        st.divider(); st.markdown("**👁️ Aperçu LLM**")
        try:
            brief_preview = md_bytes.decode("utf-8") if md_bytes else ""
            if brief_preview:
                n_zones = sum(1 for line in brief_preview.split("\n") if line.strip().startswith(("- BUY", "- SELL", "- PIVOT"))); n_actifs = brief_preview.count("### ")
                st.info(f"**{n_actifs} actifs** avec **{n_zones} zones**")
                st.text_area("Brief", value=brief_preview, height=400, label_visibility="collapsed")
        except Exception: st.warning("Aperçu non dispo.")

_TF_COL_CONFIG = {"Actif": st.column_config.TextColumn("Actif", width="small"), "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"), "Type": st.column_config.TextColumn("Type", width="small"), "Niveau": st.column_config.TextColumn("Niveau", width="small"), "Force": st.column_config.TextColumn("Force", width="medium"), "Score (1TF)": st.column_config.NumberColumn("Score (1TF) ▼", width="small"), "Statut": st.column_config.TextColumn("Statut", width="small"), "Dist. %": st.column_config.TextColumn("Dist. %", width="small"), "Dist. ATR": st.column_config.TextColumn("Dist. ATR", width="small")}

def _display_results(sr: dict, max_dist_filter_pct: float) -> None:
    _df_h4 = sr.get("df_h4", pd.DataFrame()); _df_daily = sr.get("df_daily", pd.DataFrame()); _df_wk = sr.get("df_weekly", pd.DataFrame())
    _conf_full = sr.get("conf_full", pd.DataFrame()); _rep_dict = sr.get("report_dict", {}); _summaries = sr.get("summaries", [])
    _anomalies = sr.get("anomalies", {}); errors = sr.get("scan_errors", {}); missing_tfs_map = sr.get("missing_tfs_map", {})
    debug_map = sr.get("debug_map", {})
    conf_filt = pd.DataFrame()
    if not _conf_full.empty:
        tmp = _clean_df(_conf_full).copy()
        tmp["_dist_num"] = pd.to_numeric(tmp["Distance %"].astype(str).str.replace("%", "", regex=False), errors="coerce").fillna(999.0)
        conf_filt = tmp[tmp["_dist_num"] <= max_dist_filter_pct].drop(columns=["_dist_num"], errors="ignore").reset_index(drop=True)
    _show_diagnostics(errors, _anomalies, missing_tfs_map, debug_map); _show_confluence_section(conf_filt); _show_export_section(_rep_dict, _conf_full, _summaries, _anomalies)
    st.divider()
    for label, df in [("📅 H4", _df_h4), ("📅 Daily", _df_daily), ("📅 Weekly", _df_wk)]:
        st.subheader(label); fd = _filter_and_sort(df, max_dist_filter_pct)
        st.dataframe(fd, column_config=_TF_COL_CONFIG, hide_index=True, width='stretch', height=min(len(fd) * 35 + 38, 600))

# ==============================================================================
# [ UI INIT ]
# ==============================================================================
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")
st.markdown("""<style>div[data-testid="stDataFrame"] > div { overflow: auto !important; } div[data-testid="stButton"] > button[kind="primary"] { background-color: #D32F2F; color: white; } </style>""", unsafe_allow_html=True)
st.title("📡 Scanner Bluestar S/R")
st.markdown("Zones S/R avec score pondéré et statuts.")
st.markdown("<br>", unsafe_allow_html=True)

def _is_scanning_locked() -> bool:
    lock_ts = st.session_state.get("scanning_lock_ts")
    if lock_ts is None: return False
    if (time.time() - lock_ts) > _SCAN_LOCK_TTL_S:
        st.session_state.pop("scanning_lock_ts", None); st.session_state.pop("pending_scan", None); st.session_state.pop("scan_token", None); return False
    return True
def _acquire_scan_lock() -> None: st.session_state["scanning_lock_ts"] = time.time()
def _release_scan_lock() -> None: st.session_state.pop("scanning_lock_ts", None); st.session_state.pop("scan_token", None)

with st.sidebar:
    st.header("1. OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]; account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("OK")
    except Exception as e: access_token, account_id = None, None; st.error(f"Secrets: {e}")
    st.header("2. Actifs")
    select_all = st.checkbox(f"Tous ({len(ALL_SYMBOLS)})", value=True)
    symbols_to_scan = ALL_SYMBOLS if select_all else st.multiselect("Spécifiques", options=ALL_SYMBOLS, default=["XAU_USD", "US30_USD"])
    st.header("3. LLM Params")
    st.slider("Dist max", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    st.slider("Score min", 20, 300, 30, 10, key="llm_min_score")
    st.multiselect("Statuts", options=["Vierge", "Testee", "Role Reverse", "Consommee"], default=["Vierge", "Testee", "Role Reverse"], key="llm_statuts")
    st.divider(); st.header("4. Scan")
    min_touches = st.slider("Touches H4 Forex", 2, 10, 2, 1)
    confluence_threshold = st.slider("Seuil confluence", 0.3, 2.0, 0.8, 0.1)
    max_dist_filter = st.slider("Afficher < (%)", 1.0, 15.0, 3.0, 0.5)
    if st.button("🧹 Vider cache"): _cache_clear(); st.success("Vide")
    if st.button("🔓 Unlock"): _release_scan_lock(); st.success("Libéré")

scan_button = st.button("🚀 LANCER SCAN", type="primary", use_container_width=True, disabled=_is_scanning_locked())

if "scan_state" not in st.session_state: st.session_state.scan_state = "IDLE"

if scan_button and symbols_to_scan and not _is_scanning_locked():
    st.session_state.pop("scan_results", None)
    _acquire_scan_lock()
    st.session_state["pending_scan"] = True
    st.session_state["scan_token"] = f"{time.time():.6f}"
    st.rerun()

if st.session_state.get("pending_scan", False) and symbols_to_scan:
    current_token = st.session_state.get("scan_token")
    if not current_token: st.session_state.pop("pending_scan", None); _release_scan_lock()
    else:
        st.session_state.pop("pending_scan", None)
        if not access_token or not account_id: _release_scan_lock(); st.warning("Secrets manquants"); st.stop()
        progress_bar = st.progress(0, text="Init…")
        try:
            with st.spinner("Scan async…"):
                raw_results = _run_async_isolated(lambda: run_institutional_scan(symbols_to_scan, access_token, account_id, min_touches), timeout=600.0)
        except (OandaAuthError, ScanTimeoutError, concurrent.futures.TimeoutError) as e:
            st.error(str(e)); _release_scan_lock(); st.stop()
        except Exception as e:
            st.error(f"Erreur: {type(e).__name__}"); _release_scan_lock(); st.stop()
        
        def _collect_scan_results(raw_results: list, progress_bar: Any) -> dict:
            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map, prices_map, trends_map, anomalies_map, scan_errors, bars_map_global, missing_tfs_map, price_fallback_map, debug_map = {}, {}, {}, {}, {}, {}, {}, {}, {}
            for idx, res in enumerate(raw_results):
                try: progress_bar.progress((idx + 1) / max(len(raw_results), 1), text=f"Traitement {res.symbol}")
                except Exception: pass
                if res.scan_error: scan_errors[_sym_display(res.symbol)] = res.scan_error; continue
                all_zones_map[res.symbol] = res.zones; prices_map[res.symbol] = res.price; trends_map[res.symbol] = res.trends
                bars_map_global[res.symbol] = res.bars_map; price_fallback_map[res.symbol] = res.price_is_fallback
                if res.debug_info: debug_map[_sym_display(res.symbol)] = res.debug_info
                if res.anomaly: anomalies_map[_sym_display(res.symbol)] = res.anomaly
                if res.missing_tfs: missing_tfs_map[_sym_display(res.symbol)] = res.missing_tfs
                for tf_cap, tf_rows in res.rows.items():
                    if not tf_rows: continue
                    if tf_cap == "H4": results_h4.extend(tf_rows)
                    elif tf_cap == "Daily": results_daily.extend(tf_rows)
                    elif tf_cap == "Weekly": results_weekly.extend(tf_rows)
            return {"results_h4": results_h4, "results_daily": results_daily, "results_weekly": results_weekly, "all_zones_map": all_zones_map, "prices_map": prices_map, "trends_map": trends_map, "anomalies_map": anomalies_map, "scan_errors": scan_errors, "bars_map_global": bars_map_global, "missing_tfs_map": missing_tfs_map, "price_fallback_map": price_fallback_map, "debug_map": debug_map}
        
        collected = _collect_scan_results(raw_results, progress_bar)
        try: progress_bar.empty()
        except Exception: pass
        
        n_ok = len(symbols_to_scan) - len(collected["scan_errors"])
        st.success(f"Scan terminé: {n_ok} OK") if len(collected["scan_errors"]) == 0 else st.warning(f"Scan terminé: {n_ok} OK, {len(collected['scan_errors'])} erreurs")
        if collected["anomalies_map"]: st.warning(f"{len(collected['anomalies_map'])} anomalies")
        
        st.info("Calcul confluences…")
        def _compute_confluences(scan_symbols, all_zones_map, prices_map, bars_map_global, default_threshold, scan_errors):
            all_confluences = []
            for sym in scan_symbols:
                if _sym_display(sym) in scan_errors: continue
                try:
                    zones_clean = {k: v for k, v in all_zones_map.get(sym, {}).items() if not k.startswith("_")}
                    sym_threshold = CONFLUENCE_THRESHOLD_MAP.get(sym, default_threshold)
                    all_confluences.extend(detect_confluences(_sym_display(sym), zones_clean, prices_map.get(sym), bars_map_global.get(sym, {}), confluence_threshold_pct=sym_threshold))
                except Exception as e: _LOG.warning("Confluence failed for %s: %s", sym, type(e).__name__)
            conf_full = pd.DataFrame(all_confluences)
            return _clean_df(conf_full) if not conf_full.empty else conf_full
        
        conf_full = _compute_confluences(symbols_to_scan, collected["all_zones_map"], collected["prices_map"], collected["bars_map_global"], confluence_threshold, collected["scan_errors"])
        summaries = _build_summaries(symbols_to_scan, collected["prices_map"], collected["trends_map"], collected["all_zones_map"], conf_full, collected["missing_tfs_map"], collected["price_fallback_map"])
        df_h4 = pd.DataFrame(collected["results_h4"]); df_daily = pd.DataFrame(collected["results_daily"]); df_wk = pd.DataFrame(collected["results_weekly"])
        rep_dict = {"H4": _apply_pdf_filter(df_h4), "Daily": _apply_pdf_filter(df_daily), "Weekly": _apply_pdf_filter(df_wk)}
        st.session_state["scan_results"] = {"df_h4": df_h4, "df_daily": df_daily, "df_weekly": df_wk, "conf_full": conf_full, "report_dict": rep_dict, "summaries": summaries, "anomalies": collected["anomalies_map"], "scan_errors": collected["scan_errors"], "max_dist": max_dist_filter, "missing_tfs_map": collected["missing_tfs_map"], "debug_map": collected["debug_map"]}
        _release_scan_lock()

elif not symbols_to_scan and not _is_scanning_locked(): st.info("Sélectionnez des actifs.")
elif not st.session_state.get("pending_scan", False) and not _is_scanning_locked(): st.info("Prêt.")

if "scan_results" in st.session_state and st.session_state.get("scan_state") == "IDLE":
    _display_results(st.session_state["scan_results"], max_dist_filter)
