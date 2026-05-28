# pylint: disable=too-many-lines
"""
Scanner Bluestar Supports/Résistances - v8.6.2-PROD-INSTITUTIONAL

Hotfix vs 8.6.1:
  H1. PDF rendering hardened against FPDFException "Not enough horizontal
      space to render a single character":
      - explicit set_x(l_margin) before every multi_cell
      - explicit effective-page-width (epw) instead of width=0
      - margins widened from 5mm to 8mm
      - safe_multi_cell() wrapper with cell() fallback
      - control-char stripping in _safe_pdf_str
      - aggressive truncation of anomaly messages
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
from fpdf.errors import FPDFException
from scipy.signal import find_peaks

# =============================================================================
# LAYER 0a - VERSION & CONSTANTS
# =============================================================================
SCANNER_VERSION: Final[str] = "8.6.2-PROD-INSTITUTIONAL"

# =============================================================================
# LAYER 0b - SECURE LOGGING (C6)
# =============================================================================
_TOKEN_REDACT_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE),
    re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(access_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{12,}=*", re.IGNORECASE),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{8,}", re.IGNORECASE),
    re.compile(r"\b[a-f0-9]{32}-[a-f0-9]{32}\b", re.IGNORECASE),
]


def _redact_sensitive(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _TOKEN_REDACT_PATTERNS:
        try:
            def _repl(m: re.Match) -> str:
                if m.lastindex and m.lastindex >= 1:
                    prefix = m.group(1)
                    if prefix:
                        return prefix + "***REDACTED***"
                return "***REDACTED***"
            out = pat.sub(_repl, out)
        except Exception:
            continue
    return out


def _sanitize_log_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _redact_sensitive(obj)
    if isinstance(obj, tuple):
        return tuple(_sanitize_log_obj(x) for x in obj)
    if isinstance(obj, list):
        return [_sanitize_log_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_log_obj(v) for k, v in obj.items()}
    return obj


class _SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = _redact_sensitive(record.msg)
            if record.args:
                record.args = _sanitize_log_obj(record.args)
            if record.exc_info:
                try:
                    exc_text = "".join(traceback.format_exception(*record.exc_info))
                    record.exc_text = _redact_sensitive(exc_text)
                except Exception:
                    record.exc_text = "<exception formatting failed>"
                record.exc_info = None
            if record.exc_text:
                record.exc_text = _redact_sensitive(record.exc_text)
        except Exception:
            pass
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger().addFilter(_SensitiveDataFilter())
_LOG = logging.getLogger("bluestar")


class OandaAuthError(Exception):
    pass


class DataValidationError(Exception):
    pass


class ScanTimeoutError(Exception):
    pass


class CoverageViolationError(Exception):
    pass


# =============================================================================
# LAYER 0c - THREAD-SAFE BOUNDED CACHE
# =============================================================================
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
    if df is None or df.empty:
        return 128
    try:
        return int(df.memory_usage(index=True, deep=False).sum())
    except Exception:
        return 128


def _make_readonly(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for col in df.columns:
            arr = df[col].values
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
    except Exception:
        pass
    return df


def _cache_ttl(tf: str, is_empty: bool = False) -> int:
    if is_empty:
        return _CACHE_TTL_NEGATIVE
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)


def _cache_is_fresh(fetched_at: float, tf: str, is_empty: bool) -> bool:
    return (time.monotonic() - fetched_at) <= _cache_ttl(tf, is_empty)


def _cache_key(env_url: Optional[str], acct_id: Optional[str], symbol: str, tf: str) -> Tuple[str, str, str, str]:
    return (env_url or "unknown_env", acct_id or "unknown_account", symbol, tf)


def _cache_evict_stale_locked() -> None:
    now = time.monotonic()
    stale_keys = [
        k for k, (ts, payload, _sz) in _OANDA_CACHE.items()
        if (now - ts) > _cache_ttl(k[3], payload is _CACHE_EMPTY)
    ]
    for k in stale_keys:
        _, _, sz = _OANDA_CACHE.pop(k)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)

    while len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)

    while _CACHE_BYTES_TOTAL[0] > _CACHE_MAX_BYTES and _OANDA_CACHE:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)


def _cache_get(env_url: Optional[str], acct_id: Optional[str], symbol: str, tf: str) -> Tuple[bool, Any]:
    k = _cache_key(env_url, acct_id, symbol, tf)
    with _CACHE_LOCK:
        _cache_evict_stale_locked()
        entry = _OANDA_CACHE.get(k)
        if entry is None:
            return False, None
        fetched_at, payload, _sz = entry
        if not _cache_is_fresh(fetched_at, tf, payload is _CACHE_EMPTY):
            _, _, sz = _OANDA_CACHE.pop(k)
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
            return False, None
        _OANDA_CACHE.move_to_end(k)
        return True, (None if payload is _CACHE_EMPTY else payload)


def _cache_set(env_url: Optional[str], acct_id: Optional[str], symbol: str, tf: str, df: Optional[pd.DataFrame]) -> None:
    k = _cache_key(env_url, acct_id, symbol, tf)
    if df is None:
        payload, sz = _CACHE_EMPTY, 64
    else:
        payload, sz = _make_readonly(df), _df_approx_bytes(df)

    with _CACHE_LOCK:
        old = _OANDA_CACHE.pop(k, None)
        if old:
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - old[2])
        _OANDA_CACHE[k] = (time.monotonic(), payload, sz)
        _CACHE_BYTES_TOTAL[0] += sz
        _OANDA_CACHE.move_to_end(k)
        _cache_evict_stale_locked()


def _cache_clear() -> int:
    with _CACHE_LOCK:
        n = len(_OANDA_CACHE)
        _OANDA_CACHE.clear()
        _CACHE_BYTES_TOTAL[0] = 0
        return n


def _cache_stats() -> Dict[str, int]:
    with _CACHE_LOCK:
        return {"entries": len(_OANDA_CACHE), "bytes": _CACHE_BYTES_TOTAL[0]}


# =============================================================================
# LAYER 0d - GLOBAL CONSTANTS
# =============================================================================
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
_MAX_HIGH_LOW_RATIO_DEFAULT: Final[float] = 1.8


def _to_internal_symbol(sym: str) -> str:
    return str(sym).upper().replace("/", "_").strip()


def _to_display_symbol(sym: str) -> str:
    return str(sym).upper().replace("_", "/").strip()


# =============================================================================
# LAYER 0e - HASH HELPERS
# =============================================================================
def _hash_df(df: Optional[pd.DataFrame]) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return "empty_df"
    try:
        h = hashlib.sha256()
        h.update(f"shape:{df.shape[0]}x{df.shape[1]}|".encode())
        if len(df.index) > 0:
            h.update(f"idx:{df.index[0]}:{df.index[-1]}|".encode())
        n = len(df)
        if n <= 32:
            sample = df
        else:
            sample = pd.concat([df.iloc[:8], df.iloc[n // 2 - 4:n // 2 + 4], df.iloc[-8:]], copy=False)
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        return h.hexdigest()[:32]
    except Exception:
        return f"unhashable_{id(df)}"


def _hash_series(s: Optional[pd.Series]) -> str:
    if s is None or len(s) == 0:
        return "empty_series"
    try:
        h = hashlib.sha256()
        h.update(f"len:{len(s)}|dtype:{s.dtype}|".encode())
        h.update(pd.util.hash_pandas_object(s, index=False).values.tobytes())
        return h.hexdigest()[:32]
    except Exception:
        return f"unhashable_series_{id(s)}"


# =============================================================================
# LAYER 1 - INSTRUMENT PROFILES
# =============================================================================
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

    min_touches_h4: int = 2
    min_touches_daily: int = 2
    min_touches_weekly: int = 2

    ignore_wick_filter: bool = False
    major_pivot_mult: float = 3.0
    max_high_low_ratio: float = 1.8
    max_cluster_width_pct: float = 1.0


_DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile(
    "DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
)


_PROFILES: Final[Dict[str, InstrumentProfile]] = {
    "EUR_USD": InstrumentProfile("EUR_USD", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False),
    "GBP_USD": InstrumentProfile("GBP_USD", "FOREX", 0.0001, 1.3, 0.85, 0.65, 1.5, False),
    "USD_JPY": InstrumentProfile("USD_JPY", "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False),
    "XAU_USD": InstrumentProfile(
        "XAU_USD", "METAL", 0.01, 2.5, 1.5, 1.0, 3.0, True,
        ignore_wick_filter=True,
        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2,
        price_min=1500.0, price_max=6000.0,
        major_pivot_mult=2.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.25,
    ),
    "US30_USD": InstrumentProfile(
        "US30_USD", "INDEX", 1.0, 2.5, 1.5, 0.7, 2.5, True,
        ignore_wick_filter=True,
        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2,
        price_min=25000.0, price_max=60000.0,
        major_pivot_mult=2.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.25,
    ),
    "NAS100_USD": InstrumentProfile(
        "NAS100_USD", "INDEX", 1.0, 2.5, 1.5, 0.8, 2.5, True,
        ignore_wick_filter=True,
        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2,
        price_min=10000.0, price_max=50000.0,
        major_pivot_mult=2.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.25,
    ),
    "SPX500_USD": InstrumentProfile(
        "SPX500_USD", "INDEX", 0.1, 2.2, 1.2, 0.65, 2.0, True,
        ignore_wick_filter=True,
        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2,
        price_min=3000.0, price_max=12000.0,
        major_pivot_mult=2.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.25,
    ),
    "DE30_EUR": InstrumentProfile(
        "DE30_EUR", "INDEX", 0.1, 2.2, 1.2, 0.65, 2.0, True,
        ignore_wick_filter=True,
        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=2,
        price_min=10000.0, price_max=30000.0,
        major_pivot_mult=2.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.25,
    ),
}


def get_profile(symbol: str) -> InstrumentProfile:
    sym = _to_internal_symbol(symbol)
    if sym in _PROFILES:
        return _PROFILES[sym]
    parts = sym.split("_")
    base = parts[0] if len(parts) >= 1 else sym
    quote = parts[1] if len(parts) >= 2 else ""
    if quote == "JPY":
        return InstrumentProfile(
            sym, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False,
            max_high_low_ratio=1.8,
        )
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"):
        return InstrumentProfile(
            sym, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
            max_high_low_ratio=1.8,
        )
    return replace(_DEFAULT_PROFILE, symbol=sym)


def _min_touches_for_tf(profile: InstrumentProfile, tf: str, ui_override: int) -> int:
    tf_lower = tf.lower()
    profile_min = {
        "h4": profile.min_touches_h4,
        "daily": profile.min_touches_daily,
        "weekly": profile.min_touches_weekly,
    }.get(tf_lower, 2)
    if profile.asset_class in ("INDEX", "METAL"):
        return max(2, profile_min)
    return max(profile_min, int(ui_override))


# =============================================================================
# LAYER 2 - DATA PIPELINE
# =============================================================================
def _is_valid_candle_dict(c: dict, profile: Optional[InstrumentProfile] = None) -> bool:
    try:
        prof = profile or _DEFAULT_PROFILE
        mid = c["mid"]
        o = float(mid["o"])
        h = float(mid["h"])
        lo = float(mid["l"])
        cl = float(mid["c"])
        if not all(np.isfinite(x) for x in (o, h, lo, cl)):
            return False
        if lo <= 0 or h <= 0:
            return False
        if h < lo:
            return False
        if not lo <= o <= h:
            return False
        if not lo <= cl <= h:
            return False
        max_ratio = getattr(prof, "max_high_low_ratio", _MAX_HIGH_LOW_RATIO_DEFAULT)
        if lo > 0 and (h / lo) > max_ratio:
            return False
        return True
    except Exception:
        return False


def _sanitize_ohlc_dataframe(
    df: Optional[pd.DataFrame],
    profile: Optional[InstrumentProfile] = None,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None
    prof = profile or _DEFAULT_PROFILE
    max_ratio = getattr(prof, "max_high_low_ratio", _MAX_HIGH_LOW_RATIO_DEFAULT)
    try:
        out = df.copy()
        out = out.dropna(subset=list(required))
        if out.empty:
            return None
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
        for col in required:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=list(required))
        if out.empty:
            return None
        mask = (
            np.isfinite(out["open"])
            & np.isfinite(out["high"])
            & np.isfinite(out["low"])
            & np.isfinite(out["close"])
            & (out["low"] > 0)
            & (out["high"] > 0)
            & (out["high"] >= out["low"])
            & (out["open"].between(out["low"], out["high"]))
            & (out["close"].between(out["low"], out["high"]))
        )
        ratio_ok = (out["high"] / out["low"].replace(0, np.nan)) <= max_ratio
        out = out[mask & ratio_ok.fillna(False)]
        return out if not out.empty else None
    except Exception:
        return None


class AsyncOandaClient:
    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = oanda_account_id
        self.env_url: Optional[str] = None

    @staticmethod
    async def _get_json_with_retry(
        session: aiohttp.ClientSession,
        url: str,
        headers: dict,
        params: dict,
        timeout_total: float,
        retries: int = 3,
    ) -> Optional[dict]:
        backoff = 0.5
        for attempt in range(retries):
            try:
                async with session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_total),
                ) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status in (401, 403):
                        return None
                    if r.status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff * (2 ** attempt))
                        continue
                    return None
            except Exception as e:
                _LOG.debug("HTTP attempt %d failed: %s", attempt, type(e).__name__)
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                continue
        return None

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
            except Exception:
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
        cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, tf)
        if cache_hit:
            return symbol, tf, cached
        gran = _GRANULARITY_MAP.get(tf)
        if not gran or not self.env_url:
            return symbol, tf, None
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}
        profile = get_profile(symbol)
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, params, _PER_REQUEST_TIMEOUT_S,
            )
            if data is None:
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None
            try:
                candles = [
                    {
                        "date": pd.to_datetime(c["time"], utc=True),
                        "open": float(c["mid"]["o"]),
                        "high": float(c["mid"]["h"]),
                        "low": float(c["mid"]["l"]),
                        "close": float(c["mid"]["c"]),
                        "volume": int(c.get("volume", 0)),
                    }
                    for c in data.get("candles", [])
                    if c.get("complete") and _is_valid_candle_dict(c, profile)
                ]
                if not candles:
                    _cache_set(self.env_url, self.account_id, symbol, tf, None)
                    return symbol, tf, None
                df_clean = _sanitize_ohlc_dataframe(
                    pd.DataFrame(candles).set_index("date").tail(limit),
                    profile,
                )
                _cache_set(self.env_url, self.account_id, symbol, tf, df_clean)
                return symbol, tf, df_clean
            except Exception as e:
                _LOG.warning("Candle parsing failed for %s/%s: %s", symbol, tf, type(e).__name__)
                _cache_set(self.env_url, self.account_id, symbol, tf, None)
                return symbol, tf, None

    async def fetch_price(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
    ) -> Tuple[str, Optional[float]]:
        if not self.env_url:
            return symbol, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, {"instruments": symbol}, 5,
            )
            try:
                if data and "prices" in data and data["prices"]:
                    bid = float(data["prices"][0]["closeoutBid"])
                    ask = float(data["prices"][0]["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0:
                        return symbol, (bid + ask) / 2
            except Exception:
                pass
        return symbol, None


def _run_async_isolated(coro_factory: Callable, timeout: float = 300.0) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro_factory())
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if pending:
                    loop.run_until_complete(
                        asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=5.0)
                    )
            except Exception:
                pass
            finally:
                loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="oanda-async") as ex:
        future = ex.submit(_worker)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise ScanTimeoutError(f"Async scan exceeded {timeout}s") from e


# =============================================================================
# LAYER 3 - QUANT ENGINE
# =============================================================================
@dataclass(frozen=True)
class PivotPoint:
    price: float
    weight: float
    index: int
    kind: str
    prominence: float


@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        if df is not None and len(df) >= 2:
            fb = (df["high"] - df["low"]).mean()
            return float(fb) if pd.notna(fb) and fb > 0 else None
        return None
    try:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        res = tr.rolling(period).mean().iloc[-1]
        return float(res) if pd.notna(res) and res > 0 else None
    except Exception:
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
        if t_stat > threshold:
            return "HAUSSIER"
        if t_stat < -threshold:
            return "BAISSIER"
        return "NEUTRE"
    except Exception:
        return "NEUTRE"


def _time_decay_weight(index: int, n_total: int) -> float:
    if n_total <= 1:
        return 1.0
    raw = index / max(n_total - 1, 1)
    return float(0.35 + 0.65 * np.clip(raw, 0.0, 1.0))


def _pivot_lookback_for_tf(timeframe: str) -> int:
    tf = timeframe.lower()
    if tf == "weekly":
        return 5
    if tf == "daily":
        return 3
    return 3


def _pivot_prominence_threshold(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float) -> float:
    try:
        current_p = float(df["close"].iloc[-1])
    except Exception:
        return atr_val * profile.pivot_prominence_atr
    if current_p <= 0 or not np.isfinite(current_p):
        return atr_val * profile.pivot_prominence_atr
    return float(min(atr_val * profile.pivot_prominence_atr, current_p * 0.005))


_PIVOT_FALLBACK_DIST: Final[Dict[str, int]] = {"h4": 5, "daily": 8, "weekly": 10}


def detect_swing_pivots_meta(
    df: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> List[PivotPoint]:
    if df is None or len(df) < 15 or atr_val is None or atr_val <= 0:
        return []
    try:
        n_total = len(df)
        n = _pivot_lookback_for_tf(timeframe)
        prominence_min = _pivot_prominence_threshold(df, profile, atr_val)

        highs = pd.Series(df["high"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        lows = pd.Series(df["low"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        closes = pd.Series(df["close"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        opens = pd.Series(df["open"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))

        roll_high_left = highs.shift(1).rolling(n, min_periods=n).max()
        roll_low_left = lows.shift(1).rolling(n, min_periods=n).min()

        rev_high = highs.iloc[::-1].reset_index(drop=True)
        rev_low = lows.iloc[::-1].reset_index(drop=True)

        roll_high_right = rev_high.shift(1).rolling(n, min_periods=n).max().iloc[::-1].reset_index(drop=True)
        roll_low_right = rev_low.shift(1).rolling(n, min_periods=n).min().iloc[::-1].reset_index(drop=True)

        candle_range = (highs - lows).clip(lower=1e-10)
        body_top = pd.Series(np.maximum(opens.to_numpy(), closes.to_numpy()), index=highs.index)
        body_bottom = pd.Series(np.minimum(opens.to_numpy(), closes.to_numpy()), index=highs.index)

        upper_wick_pct = (highs - body_top) / candle_range
        lower_wick_pct = (body_bottom - lows) / candle_range

        tf_lower = timeframe.lower()
        wick_threshold = (
            profile.wick_threshold_intraday if tf_lower in ("h4", "m15")
            else profile.wick_threshold_htf
        )

        if profile.ignore_wick_filter:
            sh_mask = (highs > roll_high_left) & (highs > roll_high_right)
            sl_mask = (lows < roll_low_left) & (lows < roll_low_right)
        else:
            sh_mask = (highs > roll_high_left) & (highs > roll_high_right) & (upper_wick_pct >= wick_threshold)
            sl_mask = (lows < roll_low_left) & (lows < roll_low_right) & (lower_wick_pct >= wick_threshold)

        roll_low_around = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
        roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()

        high_prom = highs - roll_low_around
        low_prom = roll_high_around - lows

        sh_mask = sh_mask & (high_prom >= prominence_min)
        sl_mask = sl_mask & (low_prom >= prominence_min)

        pivots: List[PivotPoint] = []

        for idx in sh_mask[sh_mask].index.tolist():
            pivots.append(PivotPoint(
                price=float(highs.iloc[idx]),
                weight=_time_decay_weight(int(idx), n_total),
                index=int(idx),
                kind="high",
                prominence=float(high_prom.iloc[idx]),
            ))

        for idx in sl_mask[sl_mask].index.tolist():
            pivots.append(PivotPoint(
                price=float(lows.iloc[idx]),
                weight=_time_decay_weight(int(idx), n_total),
                index=int(idx),
                kind="low",
                prominence=float(low_prom.iloc[idx]),
            ))

        pivots.sort(key=lambda p: (p.index, p.kind, p.price))
        return pivots
    except Exception as e:
        _LOG.warning("Pivot detection failed: %s", type(e).__name__)
        return []


def _get_pivots_with_fallback_meta(
    df: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> List[PivotPoint]:
    pivots = detect_swing_pivots_meta(df, profile, atr_val, timeframe)
    if len(pivots) >= 3:
        return pivots
    try:
        n_total = len(df)
        dist = _PIVOT_FALLBACK_DIST.get(timeframe.lower(), 5)
        prominence_min = _pivot_prominence_threshold(df, profile, atr_val)
        peak_kwargs = {"distance": dist, "prominence": prominence_min}
        high_idx, high_props = find_peaks(df["high"].to_numpy(dtype=float), **peak_kwargs)
        low_idx, low_props = find_peaks(-df["low"].to_numpy(dtype=float), **peak_kwargs)
        safe_cutoff = n_total - 3
        extra: List[PivotPoint] = []
        high_proms = high_props.get("prominences", np.full(len(high_idx), prominence_min))
        for k, idx in enumerate(high_idx):
            if idx >= safe_cutoff:
                continue
            extra.append(PivotPoint(
                price=float(df["high"].iloc[idx]),
                weight=_time_decay_weight(int(idx), n_total),
                index=int(idx),
                kind="high",
                prominence=float(high_proms[k]),
            ))
        low_proms = low_props.get("prominences", np.full(len(low_idx), prominence_min))
        for k, idx in enumerate(low_idx):
            if idx >= safe_cutoff:
                continue
            extra.append(PivotPoint(
                price=float(df["low"].iloc[idx]),
                weight=_time_decay_weight(int(idx), n_total),
                index=int(idx),
                kind="low",
                prominence=float(low_proms[k]),
            ))
        merged: Dict[Tuple[int, str], PivotPoint] = {}
        for p in pivots + extra:
            key = (p.index, p.kind)
            old = merged.get(key)
            if old is None or p.prominence > old.prominence:
                merged[key] = p
        out = list(merged.values())
        out.sort(key=lambda p: (p.index, p.kind, p.price))
        return out
    except Exception:
        return pivots


def agglomerative_1d_clustering(pivots: List[PivotPoint], bandwidth: float) -> List[List[PivotPoint]]:
    if not pivots:
        return []
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        return [[p] for p in sorted(pivots, key=lambda x: (x.price, x.index, x.kind))]
    ordered = sorted(pivots, key=lambda x: (x.price, x.index, x.kind))
    clusters: List[List[PivotPoint]] = []
    current: List[PivotPoint] = [ordered[0]]
    for p in ordered[1:]:
        prev = current[-1]
        cluster_first = current[0]
        gap = p.price - prev.price
        span = p.price - cluster_first.price
        if gap > bandwidth or span > 2.5 * bandwidth:
            clusters.append(current)
            current = [p]
        else:
            current.append(p)
    clusters.append(current)
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
    try:
        c_arr = df["close"].values[formation_idx + 1:]
        h_arr = df["high"].values[formation_idx + 1:]
        l_arr = df["low"].values[formation_idx + 1:]
    except Exception:
        return "Vierge"
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
    second_break = (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level + tolerance)
    return "Consommee" if second_break.any() else "Role Reverse"


def compute_structural_score(strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int) -> float:
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    return round((strength * tf_w * nb_tf) * float(np.exp(-lam * age_r)), 1)


_STATUS_PRIORITY: Final[Dict[str, int]] = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}


def _clusters_to_zones(
    clusters_raw: List[List[PivotPoint]],
    min_touches_required: int,
    n_total: int,
    df: pd.DataFrame,
    atr_val: float,
    profile: InstrumentProfile,
    zone_type: str,
) -> List[dict]:
    strong: List[dict] = []
    if atr_val is None or atr_val <= 0:
        return strong
    major_threshold = atr_val * profile.major_pivot_mult
    for grp in clusters_raw:
        if not grp:
            continue
        prices = np.array([p.price for p in grp], d
