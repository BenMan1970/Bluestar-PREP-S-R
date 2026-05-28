"""Scanner Bluestar S/R Multi-Timeframes — v8.7.2-AUDIT100

Corrections pour score d'audit 100/100 :
  - Complexité cyclomatique réduite par extraction de helpers.
  - Tous les paramètres renommés pour éviter le shadowing.
  - Captures d'exceptions rendues spécifiques (plus de except Exception).
  - Docstrings ajoutées à toutes les fonctions.
  - Regroupement des paramètres via dataclasses.
  - Nettoyage cosmétique (longueur de ligne, trailing space).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Final, Optional, Tuple, List, Dict, Set, Mapping

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
SCANNER_VERSION: Final[str] = "8.7.2-AUDIT100"

_TOKEN_REDACT_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE),
    re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(access_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{12,}=*", re.IGNORECASE),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{8,}", re.IGNORECASE),
    re.compile(r"\b[a-f0-9]{32}-[a-f0-9]{32}\b", re.IGNORECASE),
]


def _redact_sensitive(text: Any) -> Any:
    """Supprime les tokens sensibles d'une chaîne."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _TOKEN_REDACT_PATTERNS:
        try:

            def _repl(match: re.Match) -> str:
                if match.lastindex and match.lastindex >= 1:
                    prefix = match.group(1)
                    if prefix:
                        return prefix + "***REDACTED***"
                return "***REDACTED***"

            out = pat.sub(_repl, out)
        except re.error:  # redaction best-effort, ne doit jamais crasher le logging
            continue
    return out


def _sanitize_log_obj(obj: Any) -> Any:
    """Nettoie récursivement un objet pour le logging."""
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
    """Filtre de logging qui masque les données sensibles."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = _redact_sensitive(record.msg)
            if record.args:
                record.args = _sanitize_log_obj(record.args)
            if record.exc_info:
                exc_text = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = _redact_sensitive(exc_text)
                record.exc_info = None
        except (AttributeError, TypeError, ValueError):
            pass
        return True


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_sensitive_filter = _SensitiveDataFilter()
root_logger = logging.getLogger()
root_logger.addFilter(_sensitive_filter)
for handler in root_logger.handlers:
    handler.addFilter(_sensitive_filter)
_LOG = logging.getLogger("bluestar")
_LOG.addFilter(_sensitive_filter)


class OandaAuthError(Exception):
    """Erreur d'authentification OANDA."""


class DataValidationError(Exception):
    """Erreur de validation des données."""


class ScanTimeoutError(Exception):
    """Le scan a dépassé le délai imparti."""


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


def _df_approx_bytes(frame: Optional[pd.DataFrame]) -> int:
    """Estimation de la mémoire occupée par un DataFrame."""
    if frame is None or frame.empty:
        return 128
    try:
        return int(frame.memory_usage(index=True, deep=False).sum())
    except (ValueError, AttributeError):
        return 128


def _make_readonly(frame: pd.DataFrame) -> pd.DataFrame:
    """Rend les tableaux numpy du DataFrame non modifiables."""
    try:
        for col in frame.columns:
            arr = frame[col].values
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
    except (ValueError, AttributeError):
        pass
    return frame


def _cache_ttl(timeframe: str, is_empty: bool = False) -> int:
    """Retourne la durée de vie du cache pour une timeframe."""
    if is_empty:
        return _CACHE_TTL_NEGATIVE
    return _CACHE_TTL_BY_TF.get(timeframe, _CACHE_TTL_DEFAULT)


def _cache_is_fresh(fetched_at: float, timeframe: str, is_empty: bool) -> bool:
    """Vérifie si une entrée du cache est encore fraîche."""
    return (time.monotonic() - fetched_at) <= _cache_ttl(timeframe, is_empty)


def _cache_key(env_url, acct_id, symbol, timeframe):
    """Construit la clé de cache."""
    return (env_url or "unknown_env", acct_id or "unknown_account", symbol, timeframe)


def _cache_evict_stale_locked():
    """Supprime les entrées périmées du cache (doit être appelé sous verrou)."""
    now = time.monotonic()
    stale = [
        k
        for k, (ts, payload, _sz) in _OANDA_CACHE.items()
        if (now - ts) > _cache_ttl(k[3], payload is _CACHE_EMPTY)
    ]
    for k in stale:
        _, _, sz = _OANDA_CACHE.pop(k)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
    while len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
    while _CACHE_BYTES_TOTAL[0] > _CACHE_MAX_BYTES and _OANDA_CACHE:
        _, (_, _, sz) = _OANDA_CACHE.popitem(last=False)
        _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)


def _cache_get(env_url, acct_id, symbol, timeframe):
    """Récupère une entrée du cache si elle est fraîche."""
    k = _cache_key(env_url, acct_id, symbol, timeframe)
    with _CACHE_LOCK:
        _cache_evict_stale_locked()
        entry = _OANDA_CACHE.get(k)
        if entry is None:
            return False, None
        fetched_at, payload, _sz = entry
        if not _cache_is_fresh(fetched_at, timeframe, payload is _CACHE_EMPTY):
            _, _, sz = _OANDA_CACHE.pop(k)
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - sz)
            return False, None
        _OANDA_CACHE.move_to_end(k)
        return True, (None if payload is _CACHE_EMPTY else payload)


def _cache_set(env_url, acct_id, symbol, timeframe, frame):
    """Insère ou met à jour une entrée dans le cache."""
    k = _cache_key(env_url, acct_id, symbol, timeframe)
    payload, sz = (
        (_CACHE_EMPTY, 64) if frame is None else (_make_readonly(frame), _df_approx_bytes(frame))
    )
    with _CACHE_LOCK:
        old = _OANDA_CACHE.pop(k, None)
        if old:
            _CACHE_BYTES_TOTAL[0] = max(0, _CACHE_BYTES_TOTAL[0] - old[2])
        _OANDA_CACHE[k] = (time.monotonic(), payload, sz)
        _CACHE_BYTES_TOTAL[0] += sz
        _OANDA_CACHE.move_to_end(k)
        _cache_evict_stale_locked()


def _cache_clear():
    """Vide complètement le cache."""
    with _CACHE_LOCK:
        n = len(_OANDA_CACHE)
        _OANDA_CACHE.clear()
        _CACHE_BYTES_TOTAL[0] = 0
        return n


def _cache_stats():
    """Retourne les statistiques du cache."""
    with _CACHE_LOCK:
        return {"entries": len(_OANDA_CACHE), "bytes": _CACHE_BYTES_TOTAL[0]}


# ==============================================================================
# [ CONSTANTS ]
# ==============================================================================
ALL_SYMBOLS: Final[List[str]] = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_CHF",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_NZD",
    "GBP_JPY",
    "GBP_CHF",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_NZD",
    "AUD_JPY",
    "AUD_CAD",
    "AUD_CHF",
    "AUD_NZD",
    "CAD_JPY",
    "CAD_CHF",
    "CHF_JPY",
    "NZD_JPY",
    "NZD_CAD",
    "NZD_CHF",
    "XAU_USD",
    "US30_USD",
    "NAS100_USD",
    "SPX500_USD",
    "DE30_EUR",
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
def _hash_df(frame: Optional[pd.DataFrame]) -> str:
    """Calcule un hash déterministe pour un DataFrame."""
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        return "empty_df"
    try:
        h = hashlib.sha256()
        h.update(f"shape:{frame.shape[0]}x{frame.shape[1]}|".encode())
        if len(frame.index) > 0:
            h.update(f"idx:{frame.index[0]}:{frame.index[-1]}|".encode())
        n = len(frame)
        sample = (
            frame
            if n <= 32
            else pd.concat(
                [frame.iloc[:8], frame.iloc[n // 2 - 4 : n // 2 + 4], frame.iloc[-8:]], copy=False
            )
        )
        h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        return h.hexdigest()[:32]
    except (ValueError, TypeError, AttributeError):
        return f"unhashable_{id(frame)}"


def _hash_series(series: Optional[pd.Series]) -> str:
    """Calcule un hash pour une Series."""
    if series is None or len(series) == 0:
        return "empty_series"
    try:
        h = hashlib.sha256()
        h.update(f"len:{len(series)}|dtype:{series.dtype}|".encode())
        h.update(pd.util.hash_pandas_object(series, index=False).values.tobytes())
        return h.hexdigest()[:32]
    except (ValueError, TypeError, AttributeError):
        return f"unhashable_series_{id(series)}"


def _hash_dict_content(d: Optional[Mapping[str, Any]]) -> str:
    """Hash d'un dictionnaire (pour le cache Streamlit)."""
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
            h.update(json.dumps(v, sort_keys=True, default=str)[:512].encode())
        h.update(b"|")
    return h.hexdigest()[:32]


def _hash_list_content(lst: Optional[List[Any]]) -> str:
    """Hash d'une liste de dictionnaires (pour le cache Streamlit)."""
    if not lst:
        return "empty_list"
    try:
        normalized = [
            {k: str(v)[:80] for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
            for d in lst
            if isinstance(d, dict)
        ]
        return hashlib.sha256(
            json.dumps(normalized, sort_keys=True, default=str).encode()
        ).hexdigest()[:32]
    except (TypeError, ValueError, AttributeError):
        return f"unhashable_list_{len(lst)}"


# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES — Surgical Calibration ]
# ==============================================================================
@dataclass(frozen=True)
class InstrumentProfile:
    """Profil de calibration pour un instrument."""

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


_PROFILES: Final[Dict[str, InstrumentProfile]] = {
    "EUR_USD": InstrumentProfile("EUR_USD", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False),
    "GBP_USD": InstrumentProfile(
        "GBP_USD",
        "FOREX",
        0.0001,
        1.3,
        0.85,
        0.65,
        1.5,
        False,
        min_touches_h4=2,
        min_touches_daily=2,
        min_touches_weekly=1,
    ),
    "USD_JPY": InstrumentProfile(
        "USD_JPY",
        "FOREX",
        0.01,
        0.9,
        0.5,
        0.2,
        1.5,
        False,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=1,
        min_touches_weekly=1,
        major_pivot_mult=0.5,
        max_high_low_ratio=1.8,
    ),
    "XAU_USD": InstrumentProfile(
        "XAU_USD",
        "METAL",
        0.01,
        2.5,
        1.5,
        0.3,
        3.0,
        True,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=2,
        min_touches_weekly=2,
        price_min=1500.0,
        price_max=6000.0,
        major_pivot_mult=0.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.5,
    ),
    "US30_USD": InstrumentProfile(
        "US30_USD",
        "INDEX",
        1.0,
        2.5,
        1.5,
        0.4,
        2.5,
        True,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=2,
        min_touches_weekly=2,
        price_min=25000.0,
        price_max=60000.0,
        major_pivot_mult=0.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.5,
    ),
    "NAS100_USD": InstrumentProfile(
        "NAS100_USD",
        "INDEX",
        1.0,
        2.5,
        1.5,
        0.4,
        2.5,
        True,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=2,
        min_touches_weekly=2,
        price_min=10000.0,
        price_max=50000.0,
        major_pivot_mult=0.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.5,
    ),
    "SPX500_USD": InstrumentProfile(
        "SPX500_USD",
        "INDEX",
        0.1,
        2.2,
        1.2,
        0.35,
        2.0,
        True,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=2,
        min_touches_weekly=2,
        price_min=3000.0,
        price_max=12000.0,
        major_pivot_mult=0.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.5,
    ),
    "DE30_EUR": InstrumentProfile(
        "DE30_EUR",
        "INDEX",
        0.1,
        2.2,
        1.2,
        0.35,
        2.0,
        True,
        ignore_wick_filter=True,
        min_touches_h4=1,
        min_touches_daily=2,
        min_touches_weekly=2,
        price_min=10000.0,
        price_max=30000.0,
        major_pivot_mult=0.5,
        max_high_low_ratio=2.5,
        max_cluster_width_pct=1.5,
    ),
}

_DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile(
    "DEFAULT",
    "FOREX",
    0.0001,
    1.2,
    0.8,
    0.6,
    1.5,
    False,
    max_high_low_ratio=1.8,
)


def get_profile(symbol: str) -> InstrumentProfile:
    """Retourne le profil d'un instrument (créé à la volée si nécessaire)."""
    sym_clean = str(symbol).upper().replace("/", "_").strip()
    if sym_clean in _PROFILES:
        return _PROFILES[sym_clean]
    parts = sym_clean.split("_")
    base = parts[0] if len(parts) >= 1 else sym_clean
    quote = parts[1] if len(parts) >= 2 else ""
    if quote == "JPY":
        return InstrumentProfile(
            sym_clean,
            "FOREX",
            0.01,
            0.9,
            0.5,
            0.2,
            1.5,
            False,
            max_high_low_ratio=1.8,
            ignore_wick_filter=True,
            min_touches_h4=1,
            min_touches_daily=1,
            min_touches_weekly=1,
            major_pivot_mult=0.5,
        )
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"):
        return InstrumentProfile(
            sym_clean,
            "FOREX",
            0.0001,
            1.2,
            0.8,
            0.5,
            1.5,
            False,
            max_high_low_ratio=1.8,
        )
    return _DEFAULT_PROFILE


def _min_touches_for_tf(profile: InstrumentProfile, timeframe: str, ui_override: int) -> int:
    """Nombre minimum de touches pour une timeframe donnée."""
    tf_lower = timeframe.lower()
    profile_min = {
        "h4": profile.min_touches_h4,
        "daily": profile.min_touches_daily,
        "weekly": profile.min_touches_weekly,
    }.get(tf_lower, 2)
    is_jpy_cross = profile.symbol.upper().endswith("_JPY")
    if profile.asset_class in ("INDEX", "METAL") or is_jpy_cross:
        return profile_min
    return max(profile_min, ui_override)


# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE ]
# ==============================================================================
def _is_valid_candle_dict(c: dict, profile: Optional[InstrumentProfile] = None) -> bool:
    """Vérifie qu'un dictionnaire de chandelle OANDA est valide."""
    try:
        prof = profile or _DEFAULT_PROFILE
        mid = c["mid"]
        o, h, lo, cl = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
        if not all(np.isfinite(x) for x in (o, h, lo, cl)):
            return False
        if lo <= 0 or h <= 0 or h < lo or not lo <= o <= h or not lo <= cl <= h:
            return False
        max_ratio = getattr(prof, "max_high_low_ratio", 1.8)
        if lo > 0 and (h / lo) > max_ratio:
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def _sanitize_ohlc_dataframe(
    frame: Optional[pd.DataFrame], profile: Optional[InstrumentProfile] = None
) -> Optional[pd.DataFrame]:
    """Nettoie et valide un DataFrame OHLC."""
    if frame is None or frame.empty:
        return None
    required = {"open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return None
    try:
        prof = profile or _DEFAULT_PROFILE
        max_ratio = getattr(prof, "max_high_low_ratio", 1.8)
        out = frame.copy().dropna(subset=list(required))
        if out.empty:
            return None
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        if out.index.has_duplicates:
            out = out[~out.index.duplicated(keep="last")]
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
    except (KeyError, ValueError, TypeError):
        return None


class AsyncOandaClient:
    """Client asynchrone pour l'API OANDA."""

    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = oanda_account_id
        self.env_url: Optional[str] = None

    @staticmethod
    async def _get_json_with_retry(session, url, headers, params, timeout_total, retries=3):
        """Effectue une requête GET avec retry."""
        backoff = 0.5
        for attempt in range(retries):
            try:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_total),
                ) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status in (401, 403):
                        return None
                    if r.status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                        await asyncio.sleep(backoff * (2**attempt))
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2**attempt))
                continue
        return None

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        """Détermine l'environnement OANDA (practice ou trade)."""
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
            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue
        return False

    async def fetch_candles(self, session, sem, symbol, timeframe, limit=500):
        """Récupère les chandelles pour un symbole/timeframe."""
        cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, timeframe)
        if cache_hit:
            return symbol, timeframe, cached
        gran = _GRANULARITY_MAP.get(timeframe)
        if not gran or not self.env_url:
            return symbol, timeframe, None
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, params, _PER_REQUEST_TIMEOUT_S
            )
            if data is None:
                _cache_set(self.env_url, self.account_id, symbol, timeframe, None)
                return symbol, timeframe, None
            try:
                profile = get_profile(symbol)
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
                    _cache_set(self.env_url, self.account_id, symbol, timeframe, None)
                    return symbol, timeframe, None
                df_clean = _sanitize_ohlc_dataframe(
                    pd.DataFrame(candles).set_index("date").tail(limit),
                    profile,
                )
                _cache_set(self.env_url, self.account_id, symbol, timeframe, df_clean)
                return symbol, timeframe, df_clean
            except (KeyError, ValueError, TypeError):
                _cache_set(self.env_url, self.account_id, symbol, timeframe, None)
                return symbol, timeframe, None

    async def fetch_price(self, session, sem, symbol):
        """Récupère le prix courant (mid) pour un symbole."""
        if not self.env_url:
            return symbol, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, {"instruments": symbol}, 5
            )
            try:
                if data and "prices" in data and data["prices"]:
                    bid = float(data["prices"][0]["closeoutBid"])
                    ask = float(data["prices"][0]["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0:
                        return symbol, (bid + ask) / 2
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        return symbol, None


def _run_async_isolated(coro_factory, timeout=300.0):
    """Exécute une coroutine dans un thread séparé si nécessaire."""
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
                        asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True), timeout=5.0
                        )
                    )
            except (asyncio.TimeoutError, RuntimeError):
                pass
            finally:
                loop.close()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="oanda-async"
    ) as ex:
        future = ex.submit(_worker)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise ScanTimeoutError(f"Async scan exceeded {timeout}s") from e


# ==============================================================================
# [ LAYER 3: QUANT ENGINE — Surgical Fixes for Detection ]
# ==============================================================================
@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def compute_atr(dataframe: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calcule l'ATR (Average True Range) sur la période donnée."""
    if dataframe is None or len(dataframe) < period + 1:
        if dataframe is not None and len(dataframe) >= 2:
            fb = (dataframe["high"] - dataframe["low"]).mean()
            return float(fb) if pd.notna(fb) and fb > 0 else None
        return None
    try:
        tr = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - dataframe["close"].shift(1)).abs(),
                (dataframe["low"] - dataframe["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_result = tr.rolling(period).mean().iloc[-1]
        return float(atr_result) if pd.notna(atr_result) and atr_result > 0 else None
    except (KeyError, ValueError):
        return None


@st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: _hash_series})
def compute_institutional_trend(
    closes: pd.Series, lookback: int = 20, threshold: float = 2.0
) -> str:
    """Détermine la tendance institutionnelle (HAUSSIER/BAISSIER/NEUTRE)."""
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
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return "NEUTRE"


def _trend_regression_fit(
    recent: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Ajuste une régression log-linéaire sur les prix récents."""
    n = len(recent)
    close = recent["close"].to_numpy(dtype=float)
    highs = recent["high"].to_numpy(dtype=float)
    lows = recent["low"].to_numpy(dtype=float)

    if not np.all(np.isfinite(close)) or np.any(close <= 0):
        return None
    if np.any(highs <= 0) or np.any(lows <= 0):
        return None

    x = np.arange(n, dtype=float)
    y = np.log(close)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    last_fit = float(fitted[-1])
    return close, highs, lows, fitted, last_fit


def _make_trend_zone(
    lvl: float, mult: float, current_price: float, atr_val: float, zone_type: str
) -> dict:
    """Construit un dictionnaire représentant une zone de tendance."""
    return {
        "level": round(lvl, 5),
        "strength": int(mult * 10 + 5),
        "age_bars": 0,
        "status": "Vierge",
        "zone_type": zone_type,
        "prominence": round(abs(current_price - lvl), 8),
        "prominence_atr": round(abs(current_price - lvl) / atr_val, 3),
        "is_major": True,
    }


def _trend_support_levels(
    lows: np.ndarray,
    fitted: np.ndarray,
    last_fit: float,
    current_price: float,
    atr_val: float,
    max_dist_pct: float,
) -> List[dict]:
    """Calcule les niveaux de support à partir de la régression."""
    resid = np.log(lows) - fitted
    if not np.all(np.isfinite(resid)):
        return []
    resid_anchor = float(np.quantile(resid, 0.10))
    band = float(np.std(resid))
    if not np.isfinite(band) or band <= 0:
        band = abs(resid_anchor) if abs(resid_anchor) > 0 else (atr_val / max(current_price, 1e-9))

    zones: List[dict] = []
    for mult in (1.0, 2.0):
        lvl = float(np.exp(last_fit + resid_anchor - mult * band))
        if lvl <= 0 or not np.isfinite(lvl):
            continue
        if lvl >= current_price:
            lvl = float(current_price * (1.0 - (0.004 * mult)))
            if lvl <= 0 or lvl >= current_price:
                continue
        dist_pct = (current_price - lvl) / current_price * 100
        if dist_pct > max_dist_pct:
            continue
        zones.append(_make_trend_zone(lvl, mult, current_price, atr_val, "Support"))
    return zones


def _trend_resistance_levels(
    highs: np.ndarray,
    fitted: np.ndarray,
    last_fit: float,
    current_price: float,
    atr_val: float,
    max_dist_pct: float,
) -> List[dict]:
    """Calcule les niveaux de résistance à partir de la régression."""
    resid = np.log(highs) - fitted
    if not np.all(np.isfinite(resid)):
        return []
    resid_anchor = float(np.quantile(resid, 0.90))
    band = float(np.std(resid))
    if not np.isfinite(band) or band <= 0:
        band = abs(resid_anchor) if abs(resid_anchor) > 0 else (atr_val / max(current_price, 1e-9))

    zones: List[dict] = []
    for mult in (1.0, 2.0):
        lvl = float(np.exp(last_fit + resid_anchor + mult * band))
        if lvl <= 0 or not np.isfinite(lvl):
            continue
        if lvl <= current_price:
            lvl = float(current_price * (1.0 + (0.004 * mult)))
            if lvl <= current_price:
                continue
        dist_pct = (lvl - current_price) / current_price * 100
        if dist_pct > max_dist_pct:
            continue
        zones.append(_make_trend_zone(lvl, mult, current_price, atr_val, "Resistance"))
    return zones


def _dedupe_zones_by_level(zones: List[dict]) -> List[dict]:
    """Supprime les zones en double (même niveau à 5 décimales)."""
    seen_levels: Set[float] = set()
    deduped: List[dict] = []
    for z in zones:
        key = round(z["level"], 5)
        if key in seen_levels:
            continue
        seen_levels.add(key)
        deduped.append(z)
    return deduped


def _detect_trend_structure_zones(
    dataframe: pd.DataFrame,
    current_price: float,
    profile: InstrumentProfile,
    atr_val: float,
    zone_type: str,
) -> List[dict]:
    """Détection alternative de zones par régression log-prix (marchés en tendance)."""
    if dataframe is None or len(dataframe) < 30 or atr_val is None or atr_val <= 0:
        return []
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        return []
    try:
        n = min(len(dataframe), 100)
        recent = dataframe.tail(n)

        fit = _trend_regression_fit(recent)
        if fit is None:
            return []
        _close, highs, lows, fitted, last_fit = fit

        max_dist_pct = 15.0 if profile.asset_class == "INDEX" else 8.0

        if zone_type == "Support":
            zones = _trend_support_levels(
                lows, fitted, last_fit, current_price, atr_val, max_dist_pct
            )
        else:
            zones = _trend_resistance_levels(
                highs, fitted, last_fit, current_price, atr_val, max_dist_pct
            )

        return _dedupe_zones_by_level(zones)
    except (ValueError, np.linalg.LinAlgError):
        return []


@dataclass(frozen=True)
class PivotPoint:
    """Point pivot détecté."""

    price: float
    weight: float
    index: int
    kind: str  # "high" or "low"
    prominence: float


def _time_decay_weight(index: int, n_total: int) -> float:
    """Pondération temporelle décroissante."""
    if n_total <= 1:
        return 1.0
    raw = index / max(n_total - 1, 1)
    return float(0.35 + 0.65 * np.clip(raw, 0.0, 1.0))


def _pivot_lookback_for_tf(timeframe: str) -> int:
    """Fenêtre de regard arrière pour les pivots."""
    tf_lower = timeframe.lower()
    if tf_lower == "weekly":
        return 5
    if tf_lower == "daily":
        return 3
    return 3


def _pivot_prominence_threshold(
    dataframe: pd.DataFrame, profile: InstrumentProfile, atr_val: float
) -> float:
    """Seuil de proéminence adaptatif pour les pivots."""
    current_p = float(dataframe["close"].iloc[-1])
    if current_p <= 0 or not np.isfinite(current_p):
        return atr_val * profile.pivot_prominence_atr
    cap_pct = 0.012 if profile.asset_class in ("INDEX", "METAL") else 0.008
    return float(min(atr_val * profile.pivot_prominence_atr, current_p * cap_pct))


def detect_swing_pivots_meta(
    dataframe: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> List[PivotPoint]:
    """Détection des pivots de swing avec filtres adaptatifs."""
    if dataframe is None or len(dataframe) < 15 or atr_val is None or atr_val <= 0:
        return []
    try:
        n_total = len(dataframe)
        n = _pivot_lookback_for_tf(timeframe)
        prominence_min = _pivot_prominence_threshold(dataframe, profile, atr_val)

        highs = pd.Series(dataframe["high"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        lows = pd.Series(dataframe["low"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        closes = pd.Series(dataframe["close"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))
        opens = pd.Series(dataframe["open"].to_numpy(dtype=float), index=pd.RangeIndex(n_total))

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
        body_top = pd.Series(np.maximum(opens.to_numpy(), closes.to_numpy()), index=highs.index)
        body_bottom = pd.Series(np.minimum(opens.to_numpy(), closes.to_numpy()), index=highs.index)

        upper_wick_pct = (highs - body_top) / candle_range
        lower_wick_pct = (body_bottom - lows) / candle_range

        tf_lower = timeframe.lower()
        wick_threshold = (
            profile.wick_threshold_intraday
            if tf_lower in ("h4", "m15")
            else profile.wick_threshold_htf
        )

        if profile.ignore_wick_filter:
            sh_mask = (highs > roll_high_left) & (highs > roll_high_right)
            sl_mask = (lows < roll_low_left) & (lows < roll_low_right)
        else:
            sh_mask = (
                (highs > roll_high_left)
                & (highs > roll_high_right)
                & (upper_wick_pct >= wick_threshold)
            )
            sl_mask = (
                (lows < roll_low_left)
                & (lows < roll_low_right)
                & (lower_wick_pct >= wick_threshold)
            )

        roll_low_around = lows.rolling(2 * n + 1, center=True, min_periods=1).min()
        roll_high_around = highs.rolling(2 * n + 1, center=True, min_periods=1).max()

        high_prom = highs - roll_low_around
        low_prom = roll_high_around - lows

        sh_mask = sh_mask & (high_prom >= prominence_min)
        sl_mask = sl_mask & (low_prom >= prominence_min)

        safe_cutoff = n_total - n
        sh_idx = [i for i in sh_mask[sh_mask].index.tolist() if i < safe_cutoff]
        sl_idx = [i for i in sl_mask[sl_mask].index.tolist() if i < safe_cutoff]

        pivots: List[PivotPoint] = []
        for sh_index in sh_idx:
            pivots.append(
                PivotPoint(
                    price=float(highs.iloc[sh_index]),
                    weight=_time_decay_weight(int(sh_index), n_total),
                    index=int(sh_index),
                    kind="high",
                    prominence=float(high_prom.iloc[sh_index]),
                )
            )
        for sl_index in sl_idx:
            pivots.append(
                PivotPoint(
                    price=float(lows.iloc[sl_index]),
                    weight=_time_decay_weight(int(sl_index), n_total),
                    index=int(sl_index),
                    kind="low",
                    prominence=float(low_prom.iloc[sl_index]),
                )
            )

        pivots.sort(key=lambda p: (p.index, p.kind, p.price))
        return pivots
    except (ValueError, KeyError):
        return []


_PIVOT_FALLBACK_DIST: Final[Dict[str, int]] = {"h4": 5, "daily": 8, "weekly": 10}


def _fallback_peak_detection(
    dataframe: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
    n_total: int,
) -> List[PivotPoint]:
    """Détection de pics avec scipy.find_peaks en fallback."""
    n = _pivot_lookback_for_tf(timeframe)
    dist = _PIVOT_FALLBACK_DIST.get(timeframe.lower(), 5)
    eff_profile = replace(profile, pivot_prominence_atr=profile.pivot_prominence_atr * 0.5)
    prominence_min = _pivot_prominence_threshold(dataframe, eff_profile, atr_val)
    wlen = min(2 * n + 1, n_total if n_total % 2 == 1 else n_total - 1)
    wlen = max(3, wlen)
    peak_kwargs = {"distance": max(1, dist), "prominence": prominence_min, "wlen": wlen}

    high_arr = dataframe["high"].to_numpy(dtype=float)
    low_arr = dataframe["low"].to_numpy(dtype=float)

    high_idx, high_props = find_peaks(high_arr, **peak_kwargs)
    low_idx, low_props = find_peaks(-low_arr, **peak_kwargs)

    safe_cutoff = n_total - n
    extra: List[PivotPoint] = []
    for k, peak_idx in enumerate(high_idx):
        if peak_idx >= safe_cutoff:
            continue
        if peak_idx < n or peak_idx + n >= len(high_arr):
            continue
        if not (
            high_arr[peak_idx] > np.nanmax(high_arr[peak_idx - n : peak_idx])
            and high_arr[peak_idx] > np.nanmax(high_arr[peak_idx + 1 : peak_idx + n + 1])
        ):
            continue
        extra.append(
            PivotPoint(
                price=float(high_arr[peak_idx]),
                weight=_time_decay_weight(int(peak_idx), n_total),
                index=int(peak_idx),
                kind="high",
                prominence=float(high_props["prominences"][k]),
            )
        )
    for k, peak_idx in enumerate(low_idx):
        if peak_idx >= safe_cutoff:
            continue
        if peak_idx < n or peak_idx + n >= len(low_arr):
            continue
        if not (
            low_arr[peak_idx] < np.nanmin(low_arr[peak_idx - n : peak_idx])
            and low_arr[peak_idx] < np.nanmin(low_arr[peak_idx + 1 : peak_idx + n + 1])
        ):
            continue
        extra.append(
            PivotPoint(
                price=float(low_arr[peak_idx]),
                weight=_time_decay_weight(int(peak_idx), n_total),
                index=int(peak_idx),
                kind="low",
                prominence=float(low_props["prominences"][k]),
            )
        )
    return extra


def _merge_pivots_fallback(
    pivots: List[PivotPoint], extra: List[PivotPoint]
) -> List[PivotPoint]:
    """Fusionne les pivots existants avec ceux du fallback, en gardant la meilleure proéminence."""
    merged: Dict[Tuple[int, str], PivotPoint] = {}
    for p in pivots + extra:
        key = (p.index, p.kind)
        old = merged.get(key)
        if old is None or p.prominence > old.prominence:
            merged[key] = p
    out = list(merged.values())
    out.sort(key=lambda p: (p.index, p.kind, p.price))
    return out


def _get_pivots_with_fallback_meta(
    dataframe: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> List[PivotPoint]:
    """Récupère les pivots avec un mécanisme de fallback si trop peu détectés."""
    pivots = detect_swing_pivots_meta(dataframe, profile, atr_val, timeframe)
    if len(pivots) >= 3:
        return pivots

    # Soft-fallback: réduire la proéminence de 50 %
    if len(pivots) < 3 and profile.asset_class in ("INDEX", "METAL", "FOREX"):
        fallback_profile = replace(profile, pivot_prominence_atr=profile.pivot_prominence_atr * 0.5)
        fp = detect_swing_pivots_meta(dataframe, fallback_profile, atr_val, timeframe)
        if len(fp) > len(pivots):
            pivots = fp
            _LOG.info(
                "Soft-fallback pivot recovery: %s %s -> %d pivots",
                profile.symbol,
                timeframe,
                len(pivots),
            )

    if len(pivots) >= 3:
        return pivots

    # Hard fallback avec scipy.find_peaks
    try:
        n_total = len(dataframe)
        extra = _fallback_peak_detection(dataframe, profile, atr_val, timeframe, n_total)
        return _merge_pivots_fallback(pivots, extra)
    except (ValueError, TypeError):
        return pivots


def agglomerative_1d_clustering(
    pivots: List[PivotPoint], bandwidth: float
) -> List[List[PivotPoint]]:
    """Regroupe les pivots par proximité de prix."""
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
    dataframe: pd.DataFrame,
    formation_idx: int,
    atr_val: float,
) -> str:
    """Détermine le statut d'une zone (Vierge, Testée, Consommée, Role Reverse)."""
    if formation_idx >= len(dataframe) - 1 or atr_val is None or atr_val <= 0:
        return "Vierge"
    tolerance = atr_val * 0.25
    try:
        c_arr = dataframe["close"].values[formation_idx + 1 :]
        h_arr = dataframe["high"].values[formation_idx + 1 :]
        l_arr = dataframe["low"].values[formation_idx + 1 :]
    except (IndexError, KeyError):
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
    rc = c_arr[break_idx + 1 :]
    rh = h_arr[break_idx + 1 :]
    rl = l_arr[break_idx + 1 :]
    if len(rc) == 0:
        return "Consommee"
    retest_mask = (rl <= level + retest_tol) & (rh >= level - retest_tol)
    if not retest_mask.any():
        return "Consommee"
    retest_idx = int(np.where(retest_mask)[0][0])
    rc_after = rc[retest_idx + 1 :]
    if len(rc_after) == 0:
        return "Role Reverse"
    second_break = (
        (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level - tolerance)
    )
    return "Consommee" if second_break.any() else "Role Reverse"


def compute_structural_score(
    strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int
) -> float:
    """Score structurel d'une zone."""
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    return round((strength * tf_w * nb_tf) * float(np.exp(-lam * age_r)), 1)


_STATUS_PRIORITY: Final[Dict[str, int]] = {
    "Vierge": 0,
    "Testee": 1,
    "Role Reverse": 2,
    "Consommee": 3,
}


def _cluster_to_zone_dict(
    grp: List[PivotPoint],
    min_touches_required: int,
    n_total: int,
    dataframe: pd.DataFrame,
    atr_val: float,
    profile: InstrumentProfile,
    zone_type: str,
) -> Optional[dict]:
    """Convertit un cluster de pivots en dictionnaire de zone (ou None si rejeté)."""
    prices = np.array([p.price for p in grp], dtype=float)
    weights = np.array([p.weight for p in grp], dtype=float)
    prominences = np.array([p.prominence for p in grp], dtype=float)
    if not np.all(np.isfinite(prices)):
        return None
    if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(prices)
    unique_touch_indices = {p.index for p in grp}
    touches = len(unique_touch_indices)
    max_prominence = float(np.nanmax(prominences)) if len(prominences) else 0.0
    avg_prominence = (
        float(np.average(prominences, weights=weights)) if weights.sum() > 0 else max_prominence
    )
    major_threshold = atr_val * profile.major_pivot_mult
    is_major = touches == 1 and max_prominence >= major_threshold
    if touches < min_touches_required and not is_major:
        return None
    lvl = float(np.average(prices, weights=weights))
    if lvl <= 0 or not np.isfinite(lvl):
        return None
    last_idx = max(p.index for p in grp)
    status = classify_zone_status(
        level=lvl, zone_type=zone_type, dataframe=dataframe, formation_idx=last_idx, atr_val=atr_val
    )
    return {
        "level": lvl,
        "strength": int(touches),
        "age_bars": int(max(0, n_total - 1 - last_idx)),
        "status": status,
        "zone_type": zone_type,
        "prominence": round(avg_prominence, 8),
        "prominence_atr": round(avg_prominence / atr_val, 3) if atr_val else None,
        "is_major": bool(is_major),
    }


def _clusters_to_zones(
    clusters_raw: List[List[PivotPoint]],
    min_touches_required: int,
    n_total: int,
    dataframe: pd.DataFrame,
    atr_val: float,
    profile: InstrumentProfile,
    zone_type: str,
) -> List[dict]:
    """Transforme les clusters bruts en zones consolidées."""
    strong: List[dict] = []
    if atr_val is None or atr_val <= 0:
        return strong
    for grp in clusters_raw:
        if not grp:
            continue
        zone = _cluster_to_zone_dict(
            grp, min_touches_required, n_total, dataframe, atr_val, profile, zone_type
        )
        if zone is not None:
            strong.append(zone)
    strong.sort(key=lambda z: (z["level"], z["zone_type"], z["age_bars"]))
    return strong


def _merge_adjacent_zones(strong: List[dict], merge_thresh: float) -> List[dict]:
    """Fusionne les zones adjacentes de même type."""
    if not strong:
        return []
    if merge_thresh <= 0 or not np.isfinite(merge_thresh):
        return sorted(strong, key=lambda x: (x["level"], x.get("zone_type", "")))
    ordered = sorted(strong, key=lambda x: (x["level"], x.get("zone_type", "")))
    merged: List[dict] = []
    for z in ordered:
        if not merged:
            merged.append(z)
            continue
        prev = merged[-1]
        same_type = prev.get("zone_type") == z.get("zone_type")
        close_enough = abs(z["level"] - prev["level"]) <= merge_thresh
        both_major = bool(prev.get("is_major")) and bool(z.get("is_major"))
        if not close_enough or (not same_type and not both_major):
            merged.append(z)
            continue
        prev_strength = max(int(prev.get("strength", 1)), 1)
        z_strength = max(int(z.get("strength", 1)), 1)
        new_strength = prev_strength + z_strength
        new_level = (prev["level"] * prev_strength + z["level"] * z_strength) / new_strength
        merged[-1] = {
            "level": float(new_level),
            "strength": int(new_strength),
            "age_bars": int(min(prev["age_bars"], z["age_bars"])),
            "status": max(
                [prev["status"], z["status"]],
                key=lambda s: _STATUS_PRIORITY.get(s, 1),
            ),
            "zone_type": prev.get("zone_type") if same_type else "Pivot",
            "prominence": max(float(prev.get("prominence", 0.0)), float(z.get("prominence", 0.0))),
            "prominence_atr": max(
                float(prev.get("prominence_atr") or 0.0),
                float(z.get("prominence_atr") or 0.0),
            ),
            "is_major": bool(prev.get("is_major")) or bool(z.get("is_major")),
        }
    return merged


@dataclass(frozen=True)
class _SwingZoneContext:
    """Contexte pour le calcul des zones de swing."""

    dataframe: pd.DataFrame
    current_price: float
    symbol: str
    atr_val: float
    timeframe: str
    min_touches_required: int
    profile: InstrumentProfile


def _compute_swing_zones(ctx: _SwingZoneContext) -> Tuple[List[dict], List[dict]]:
    """Phase 1+2 : clustering swing + union trend-structure."""
    n_total = len(ctx.dataframe)
    pivots = _get_pivots_with_fallback_meta(ctx.dataframe, ctx.profile, ctx.atr_val, ctx.timeframe)

    raw_bandwidth = max(ctx.atr_val * ctx.profile.cluster_radius_atr, ctx.current_price * 0.0015)
    max_bandwidth = ctx.current_price * (ctx.profile.max_cluster_width_pct / 100.0)
    bandwidth = float(min(raw_bandwidth, max_bandwidth))

    resistance_zones: List[dict] = []
    support_zones: List[dict] = []

    if pivots and bandwidth > 0 and np.isfinite(bandwidth):
        highs = [p for p in pivots if p.kind == "high"]
        lows = [p for p in pivots if p.kind == "low"]

        high_clusters = agglomerative_1d_clustering(highs, bandwidth)
        low_clusters = agglomerative_1d_clustering(lows, bandwidth)

        resistance_zones = _clusters_to_zones(
            clusters_raw=high_clusters,
            min_touches_required=ctx.min_touches_required,
            n_total=n_total,
            dataframe=ctx.dataframe,
            atr_val=ctx.atr_val,
            profile=ctx.profile,
            zone_type="Resistance",
        )
        support_zones = _clusters_to_zones(
            clusters_raw=low_clusters,
            min_touches_required=ctx.min_touches_required,
            n_total=n_total,
            dataframe=ctx.dataframe,
            atr_val=ctx.atr_val,
            profile=ctx.profile,
            zone_type="Support",
        )

    # Phase 2: MODE TREND-STRUCTURE (union, pas fallback)
    if ctx.profile.asset_class in ("INDEX", "METAL"):
        tr_res = _detect_trend_structure_zones(
            ctx.dataframe, ctx.current_price, ctx.profile, ctx.atr_val, "Resistance"
        )
        tr_sup = _detect_trend_structure_zones(
            ctx.dataframe, ctx.current_price, ctx.profile, ctx.atr_val, "Support"
        )
        if tr_res:
            resistance_zones.extend(tr_res)
            _LOG.info(
                "Trend-structure union: %s %s added %d resistance zones",
                ctx.symbol,
                ctx.timeframe,
                len(tr_res),
            )
        if tr_sup:
            support_zones.extend(tr_sup)
            _LOG.info(
                "Trend-structure union: %s %s added %d support zones",
                ctx.symbol,
                ctx.timeframe,
                len(tr_sup),
            )

    return support_zones, resistance_zones


def _reclassify_orphan_zones(
    df_zones: pd.DataFrame, current_price: float
) -> pd.DataFrame:
    """Reclassifie les zones dont le type est incompatible avec la position du prix."""
    for zone_idx, row in df_zones.iterrows():
        lvl = row["level"]
        ztype = row["zone_type"]
        if ztype == "Resistance" and lvl < current_price:
            df_zones.at[zone_idx, "zone_type"] = "Pivot"
        elif ztype == "Support" and lvl >= current_price:
            df_zones.at[zone_idx, "zone_type"] = "Pivot"
    return df_zones


def _split_supports_resistances(
    df_zones: pd.DataFrame, current_price: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les supports et résistances en deux DataFrames."""
    supports = df_zones[
        (df_zones["level"] < current_price) & (df_zones["zone_type"].isin(["Support", "Pivot"]))
    ].copy()
    resistances = df_zones[
        (df_zones["level"] >= current_price) & (df_zones["zone_type"].isin(["Resistance", "Pivot"]))
    ].copy()
    return supports, resistances


@st.cache_data(ttl=120, max_entries=256, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df})
def find_strong_sr_zones(
    dataframe: pd.DataFrame,
    current_price: float,
    symbol: str,
    atr_val: Optional[float],
    timeframe: str,
    min_touches_required: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fonction principale de détection des zones de support/résistance."""
    if (
        atr_val is None
        or atr_val <= 0
        or dataframe is None
        or dataframe.empty
        or current_price is None
        or not np.isfinite(current_price)
        or current_price <= 0
    ):
        return pd.DataFrame(), pd.DataFrame()
    profile = get_profile(symbol)

    ctx = _SwingZoneContext(
        dataframe=dataframe,
        current_price=current_price,
        symbol=symbol,
        atr_val=atr_val,
        timeframe=timeframe,
        min_touches_required=min_touches_required,
        profile=profile,
    )
    support_zones, resistance_zones = _compute_swing_zones(ctx)

    if not resistance_zones and not support_zones:
        return pd.DataFrame(), pd.DataFrame()

    merge_thresh_raw = atr_val * profile.merge_threshold_atr
    merge_thresh_cap = current_price * 0.0075
    merge_thresh = float(min(merge_thresh_raw, merge_thresh_cap))

    support_zones = _merge_adjacent_zones(support_zones, merge_thresh)
    resistance_zones = _merge_adjacent_zones(resistance_zones, merge_thresh)

    all_zones = support_zones + resistance_zones
    if not all_zones:
        return pd.DataFrame(), pd.DataFrame()

    df_zones = pd.DataFrame(all_zones)
    if df_zones.empty or "level" not in df_zones.columns:
        return pd.DataFrame(), pd.DataFrame()

    df_zones = df_zones.sort_values(
        ["level", "zone_type", "age_bars"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    df_zones = _reclassify_orphan_zones(df_zones, current_price)

    df_zones["near_price"] = (
        np.abs(df_zones["level"] - current_price) / current_price * 100
    ) <= 0.50

    supports, resistances = _split_supports_resistances(df_zones, current_price)

    if supports.empty and resistances.empty and profile.asset_class in ("INDEX", "METAL"):
        _LOG.info(
            "Trend-mode safety net for %s %s (orphan zones blocked output)", symbol, timeframe
        )
        tr_zones = _detect_trend_structure_zones(
            dataframe, current_price, profile, atr_val, "Support"
        ) + _detect_trend_structure_zones(
            dataframe, current_price, profile, atr_val, "Resistance"
        )
        if tr_zones:
            tr_df = pd.DataFrame(tr_zones)
            tr_df["near_price"] = (
                np.abs(tr_df["level"] - current_price) / current_price * 100
            ) <= 0.50
            supports, resistances = _split_supports_resistances(tr_df, current_price)

    return supports, resistances


def _flatten_zones_to_dataframe(zones_dict: dict) -> pd.DataFrame:
    """Aplatit le dictionnaire de zones en un DataFrame unique."""
    frames = []
    for timeframe_key, pair in zones_dict.items():
        try:
            sup, res = pair
        except (ValueError, TypeError):
            continue
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            if df_z is None or df_z.empty:
                continue
            tmp = df_z[df_z["status"] != "Consommee"].copy()
            if tmp.empty:
                continue
            tmp = tmp.assign(
                tf=timeframe_key, type=tmp["near_price"].map({True: "Pivot", False: ztype})
            )
            cols = ["tf", "level", "strength", "age_bars", "status", "type", "near_price"]
            for c in ["prominence_atr", "is_major"]:
                if c in tmp.columns:
                    cols.append(c)
            frames.append(tmp[[c for c in cols if c in tmp.columns]])
    return (
        pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )


def _score_and_classify_group(
    group: pd.DataFrame, current_price: float, bars_map: dict, symbol: str
) -> dict:
    """Score et classification d'un groupe de confluences."""
    sub_avg = group["level"].mean()
    sub_nb_tf = group["tf"].nunique()
    safe_cp = current_price if current_price and current_price > 0 else 1.0
    sub_dist = abs(safe_cp - sub_avg) / safe_cp * 100
    tf_w = group["tf"].map(TF_WEIGHT).fillna(1.0).values
    totals = group["tf"].map(lambda t: bars_map.get(t, 500)).values.astype(float)
    age_r = np.clip(group["age_bars"].values / np.maximum(totals, 1), 0, 1)
    lams = group["tf"].map(_TF_LAMBDA).fillna(1.5).values
    major_bonus = 1.0
    if "is_major" in group.columns:
        major_bonus = 1.0 + 0.3 * group["is_major"].sum()
    score = round(
        float(
            (group["strength"].values * tf_w * sub_nb_tf * np.exp(-lams * age_r)).sum()
            * major_bonus
        ),
        1,
    )
    status = max(group["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 1))
    is_near_price = sub_dist <= 0.50
    if is_near_price:
        ctype, sig = "Pivot", "↔ PIVOT ZONE"
    else:
        n_sup = (group["level"] < safe_cp).sum()
        ctype = "Support" if n_sup >= len(group) - n_sup else "Resistance"
        sig = "🟢 BUY ZONE" if ctype == "Support" else "🔴 SELL ZONE"
    return {
        "Actif": symbol,
        "Signal": sig,
        "Niveau": round(sub_avg, 5),
        "Type": ctype,
        "Timeframes": " + ".join(sorted(group["tf"].unique())),
        "Nb TF": int(sub_nb_tf),
        "Force Totale": int(group["strength"].sum()),
        "Score": round(score, 1),
        "Statut": status,
        "Distance %": round(sub_dist, 3),
        "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else ""),
    }


def detect_confluences(
    symbol: str,
    zones_dict: dict,
    current_price: float,
    bars_map: dict,
    confluence_threshold_pct: Optional[float] = None,
) -> list:
    """Détecte les confluences multi‑timeframes."""
    if not zones_dict or not current_price or current_price <= 0 or not np.isfinite(current_price):
        return []
    z_df = _flatten_zones_to_dataframe(zones_dict)
    if z_df.empty:
        return []
    profile = get_profile(symbol.replace("/", "_"))
    threshold = (
        confluence_threshold_pct
        if confluence_threshold_pct is not None
        else profile.confluence_threshold_pct
    )
    z_df = z_df.sort_values("level").reset_index(drop=True)
    n = len(z_df)
    levels_arr = z_df["level"].values
    parent, rank = list(range(n)), [0] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

    for i in range(n):
        li = levels_arr[i]
        if li <= 0:
            continue
        for j in range(i + 1, n):
            if (levels_arr[j] - li) / li * 100 > threshold:
                break
            union(i, j)
    comp_map = {}
    for union_idx in range(n):
        root = find(union_idx)
        comp_map.setdefault(root, []).append(union_idx)
    confluences = []
    for indices in comp_map.values():
        group_full = z_df.iloc[indices]
        nb_tf = group_full["tf"].nunique()
        has_trend_zone = False
        if "is_major" in group_full.columns:
            has_trend_zone = group_full["is_major"].any()
        if nb_tf < 2 and not has_trend_zone:
            continue
        sub_avg = group_full["level"].mean()
        group_full = group_full.assign(_dist=(group_full["level"] - sub_avg).abs())
        keep_idx = group_full.groupby("tf")["_dist"].idxmin().values
        confluences.append(
            _score_and_classify_group(
                group_full.loc[keep_idx].drop(columns=["_dist"]),
                current_price,
                bars_map,
                symbol,
            )
        )
    return confluences


# ==============================================================================
# [ LAYER 4: PIPELINE ORCHESTRATOR ]
# ==============================================================================
@dataclass
class ScanResult:
    """Résultat du scan pour un symbole."""

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
    """Contexte pour la construction des lignes d'affichage."""

    current_price: float
    atr_val: float
    sym_d: str
    tf_name: str
    df_len: int
    profile: InstrumentProfile


def _make_row(z: dict, ztype: str, ctx: _RowContext) -> Dict[str, Any]:
    """Crée une ligne formatée pour l'interface."""
    dist = abs(ctx.current_price - z["level"]) / ctx.current_price * 100 if ctx.current_price else 0.0
    dist_atr = (
        f"{round(abs(ctx.current_price - z['level']) / ctx.atr_val, 1)}x"
        if (ctx.atr_val and ctx.atr_val > 0)
        else "N/A"
    )
    return {
        "Actif": ctx.sym_d,
        "Prix Actuel": f"{ctx.current_price:.5f}" if ctx.current_price else "N/A",
        "Type": ztype,
        "Niveau": f"{z['level']:.5f}",
        "Force": f"{z['strength']} touches",
        "Score (1TF)": compute_structural_score(
            z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len
        ),
        "Statut": z["status"],
        "Dist. %": f"{dist:.2f}%",
        "Dist. ATR": dist_atr,
        "_dist_num": dist,
        "_in_pdf": dist <= ctx.profile.pdf_max_dist_pct,
    }


async def _fetch_live_prices(client, session, sem, symbols):
    """Récupère les prix live pour tous les symboles."""
    tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    out = {}
    for sym, item in zip(symbols, res):
        if isinstance(item, BaseException):
            out[sym] = None
        else:
            out[item[0]] = item[1]
    return out


async def _fetch_candles_cube(client, session, sem, symbols):
    """Récupère les chandelles pour toutes les timeframes."""
    data_cube = {sym: {tf: None for tf in _GRANULARITY_MAP} for sym in symbols}
    tasks = [
        client.fetch_candles(session, sem, sym, tf) for sym in symbols for tf in _GRANULARITY_MAP
    ]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    for item in res:
        if isinstance(item, BaseException):
            _LOG.warning("Candle fetch task failed: %s", type(item).__name__)
            continue
        try:
            sym, tf, frame = item
            if sym in data_cube and tf in data_cube[sym]:
                data_cube[sym][tf] = frame
        except (ValueError, TypeError):
            _LOG.warning("Malformed candle fetch result ignored")
    return data_cube


def _build_daily_price_context(
    current_price: float, sup: pd.DataFrame, res: pd.DataFrame
) -> str:
    """Construit le contexte de prix quotidien."""
    parts = []
    if sup is not None and not sup.empty:
        s_near = sup[
            (sup["level"] < current_price) & (abs(sup["level"] - current_price) / current_price * 100 <= 5.0)
        ]
        if not s_near.empty:
            n_s = s_near.nlargest(1, "level").iloc[0]
            price_label = (
                "SUR support" if abs(current_price - n_s["level"]) / current_price * 100 < 0.5 else "S proche"
            )
            parts.append(
                f"{price_label}: {n_s['level']:.5f} (-{abs(current_price - n_s['level']) / current_price * 100:.2f}%)"
            )
    if res is not None and not res.empty:
        r_near = res[
            (res["level"] > current_price) & (abs(res["level"] - current_price) / current_price * 100 <= 5.0)
        ]
        if not r_near.empty:
            n_r = r_near.nsmallest(1, "level").iloc[0]
            price_label = (
                "SUR resistance"
                if abs(current_price - n_r["level"]) / current_price * 100 < 0.5
                else "R proche"
            )
            parts.append(
                f"{price_label}: {n_r['level']:.5f} (+{abs(current_price - n_r['level']) / current_price * 100:.2f}%)"
            )
    return "  |  ".join(parts) if parts else "Zone intermediaire"


_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {
    "H4": (50, 2.0),
    "Daily": (50, 1.8),
    "Weekly": (20, 1.5),
}


@dataclass(frozen=True)
class _TFProcessingContext:
    """Contexte pour le traitement d'une timeframe."""

    symbol: str
    tf_key: str
    tf_name: str
    dataframe: pd.DataFrame
    current_price: float
    min_touches_ui: int
    profile: InstrumentProfile
    sym_d: str


def _process_tf_frame(ctx: _TFProcessingContext):
    """Traite une timeframe unique et retourne les lignes, zones, contexte prix et debug."""
    debug = {
        "atr": None,
        "n_pivots": 0,
        "n_zones": 0,
        "n_trend_zones": 0,
        "min_touches": None,
        "tf": ctx.tf_name,
    }
    try:
        atr_val = compute_atr(ctx.dataframe)
        debug["atr"] = atr_val
        if atr_val is None:
            return None, None, "", debug
        min_t = _min_touches_for_tf(ctx.profile, ctx.tf_key, ctx.min_touches_ui)
        debug["min_touches"] = min_t

        # Debug: compter les pivots
        try:
            pivots_meta = _get_pivots_with_fallback_meta(
                ctx.dataframe, ctx.profile, atr_val, ctx.tf_key
            )
            debug["n_pivots"] = len(pivots_meta)
        except (ValueError, TypeError):
            debug["n_pivots"] = None

        # Debug: compter les zones trend-structure
        if ctx.profile.asset_class in ("INDEX", "METAL"):
            try:
                tr_count = len(
                    _detect_trend_structure_zones(
                        ctx.dataframe, ctx.current_price, ctx.profile, atr_val, "Support"
                    )
                ) + len(
                    _detect_trend_structure_zones(
                        ctx.dataframe, ctx.current_price, ctx.profile, atr_val, "Resistance"
                    )
                )
                debug["n_trend_zones"] = tr_count
            except (ValueError, TypeError):
                debug["n_trend_zones"] = None

        sup, res = find_strong_sr_zones(
            ctx.dataframe, ctx.current_price, ctx.symbol, atr_val, ctx.tf_key, min_t
        )
        debug["n_zones"] = int(len(sup) + len(res))

        price_ctx = (
            _build_daily_price_context(ctx.current_price, sup, res) if ctx.tf_key == "daily" else ""
        )
        row_ctx = _RowContext(
            current_price=ctx.current_price,
            atr_val=atr_val,
            sym_d=ctx.sym_d,
            tf_name=ctx.tf_name,
            df_len=len(ctx.dataframe),
            profile=ctx.profile,
        )
        tf_r = [
            _make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx)
            for _, z in sup.iterrows()
        ] + [
            _make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx)
            for _, z in res.iterrows()
        ]
        seen, uniq = set(), []
        for r in tf_r:
            if (r["Niveau"], r["Type"]) not in seen:
                seen.add((r["Niveau"], r["Type"]))
                uniq.append(r)
        return (uniq if uniq else None), (sup, res), price_ctx, debug
    except (ValueError, KeyError, TypeError) as e:
        _LOG.warning("TF processing error %s/%s: %s", ctx.symbol, ctx.tf_name, type(e).__name__)
        debug["error"] = type(e).__name__
        return None, None, "", debug


def _resolve_working_price(cp_live, data_cube, symbol):
    """Détermine le prix de travail (live ou fallback)."""
    if cp_live and cp_live > 0 and np.isfinite(cp_live):
        return cp_live, False
    for tf_k in ("daily", "h4", "weekly"):
        df = data_cube.get(symbol, {}).get(tf_k)
        if df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                if np.isfinite(last_close) and last_close > 0:
                    return last_close, True
            except (ValueError, KeyError, IndexError):
                continue
    return None, False


def _validate_price_bounds_post(current_price, profile):
    """Vérifie que le prix est dans les bornes autorisées."""
    if profile.price_min is not None and current_price < profile.price_min:
        return f"PRIX HORS BORNES ({current_price:.2f} < {profile.price_min:.0f})"
    if profile.price_max is not None and current_price > profile.price_max:
        return f"PRIX HORS BORNES ({current_price:.2f} > {profile.price_max:.0f})"
    return None


def _collect_tf_data(
    symbol: str,
    data_cube: dict,
    current_price: float,
    profile: InstrumentProfile,
    min_touches_ui: int,
    sym_d: str,
):
    """Collecte les données pour toutes les timeframes d'un symbole."""
    rows = {"H4": None, "Daily": None, "Weekly": None}
    zones_d, trends, bars_map = {}, {}, {}
    debug_per_tf, missing_tfs = {}, []
    price_ctx = ""
    for tf_k, tf_name in (("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")):
        df = data_cube.get(symbol, {}).get(tf_k)
        if df is None or df.empty:
            missing_tfs.append(tf_name)
            continue
        bars_map[tf_name] = len(df)
        lb, th = _TF_TREND_PARAMS.get(tf_name, (20, 2.0))
        trends[tf_name] = compute_institutional_trend(df["close"], lookback=lb, threshold=th)
        ctx = _TFProcessingContext(
            symbol=symbol,
            tf_key=tf_k,
            tf_name=tf_name,
            dataframe=df,
            current_price=current_price,
            min_touches_ui=min_touches_ui,
            profile=profile,
            sym_d=sym_d,
        )
        tf_rows, zone_pair, ctx_str, debug = _process_tf_frame(ctx)
        debug_per_tf[tf_name] = debug
        if zone_pair is not None:
            zones_d[tf_name] = zone_pair
        if tf_rows is not None:
            rows[tf_name] = tf_rows
        if ctx_str:
            price_ctx = ctx_str
    return rows, zones_d, trends, bars_map, price_ctx, missing_tfs, debug_per_tf


def flag_data_anomaly(
    symbol: str, current_price: float, support_levels: list, last_candle_close: Optional[float] = None
) -> Optional[str]:
    """Détecte les anomalies sur les données d'un symbole."""
    if current_price is None or current_price <= 0 or not np.isfinite(current_price):
        return "Prix indisponible"
    profile = get_profile(symbol)
    msgs = []
    if profile.price_min and current_price < profile.price_min:
        msgs.append("PRIX < MIN")
    if profile.price_max and current_price > profile.price_max:
        msgs.append("PRIX > MAX")
    if not profile.skip_ratio_check and len(support_levels) >= 3:
        median_sup = float(np.median(support_levels))
        if median_sup > 0 and current_price / median_sup > 3.0:
            msgs.append("Ecart aberrant")
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        if dev > profile.max_live_vs_close_pct:
            msgs.append(f"Ecart live/close ({dev:.1f}%)")
    return " | ".join(msgs) if msgs else None


def _process_symbol(symbol: str, cp_live: Optional[float], data_cube: dict, min_touches_ui: int):
    """Traite un symbole complet."""
    try:
        profile = get_profile(symbol)
        sym_d = symbol.replace("_", "/")
        current_price, price_is_fallback = _resolve_working_price(cp_live, data_cube, symbol)
        if current_price is None:
            return ScanResult(symbol, {}, {}, None, {}, {}, scan_error="Aucune donnee disponible")
        bounds_err = _validate_price_bounds_post(current_price, profile)
        if bounds_err:
            return ScanResult(symbol, {}, {}, None, {}, {}, scan_error=bounds_err)
        (
            rows,
            zones_d,
            trends,
            bars_map,
            price_ctx,
            missing_tfs,
            debug,
        ) = _collect_tf_data(symbol, data_cube, current_price, profile, min_touches_ui, sym_d)
        sup_levels = []
        for zp in zones_d.values():
            if zp[0] is not None and not zp[0].empty:
                sup_levels.extend(zp[0]["level"].tolist())
        daily_df = data_cube.get(symbol, {}).get("daily")
        last_close = (
            float(daily_df["close"].iloc[-1])
            if daily_df is not None and not daily_df.empty
            else None
        )
        anomaly = flag_data_anomaly(symbol, current_price, sup_levels, last_candle_close=last_close)
        if price_is_fallback:
            anomaly = f"{anomaly} | Prix fallback" if anomaly else "Prix fallback"
        return ScanResult(
            symbol,
            rows,
            zones_d,
            current_price,
            trends,
            bars_map,
            price_context=price_ctx,
            anomaly=anomaly,
            missing_tfs=missing_tfs,
            price_is_fallback=price_is_fallback,
            debug_info=debug,
        )
    except (ValueError, KeyError, TypeError) as e:
        _LOG.exception("Symbol processing error: %s", symbol)
        return ScanResult(
            symbol, {}, {}, None, {}, {}, scan_error=f"Erreur interne : {type(e).__name__}"
        )


def _validate_symbol_coverage(
    requested_symbols: List[str], results: List[ScanResult]
) -> Dict[str, Any]:
    """Vérifie la couverture des symboles demandés."""
    requested = set(requested_symbols)
    returned = {r.symbol for r in results if isinstance(r, ScanResult)}
    missing = sorted(requested - returned)
    extra = sorted(returned - requested)
    return {
        "requested": len(requested),
        "returned": len(returned),
        "missing": missing,
        "extra": extra,
        "ok": not missing and not extra,
    }


async def run_institutional_scan(symbols, token, oanda_account_id, min_touches_ui):
    """Exécute le scan institutionnel complet."""
    client = AsyncOandaClient(token, oanda_account_id)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=10)
    ) as session:
        if not await client.initialize(session):
            raise OandaAuthError("Auth OANDA echouee")
        sem = asyncio.Semaphore(_OANDA_SEMAPHORE_LIMIT)
        live_prices = await _fetch_live_prices(client, session, sem, symbols)
        data_cube = await _fetch_candles_cube(client, session, sem, symbols)

    results: List[ScanResult] = []
    for sym in symbols:
        res = _process_symbol(sym, live_prices.get(sym), data_cube, min_touches_ui)
        results.append(res)

    coverage = _validate_symbol_coverage(symbols, results)
    if not coverage["ok"]:
        _LOG.error("Coverage violation: %s", coverage)
        existing = {r.symbol for r in results}
        for missing_sym in coverage["missing"]:
            if missing_sym not in existing:
                results.append(
                    ScanResult(
                        symbol=missing_sym,
                        rows={},
                        zones={},
                        price=None,
                        trends={},
                        bars_map={},
                        scan_error="Coverage violation: symbole non traite",
                    )
                )
    return results


# ==============================================================================
# [ LAYER 5: EXPORTERS & UI UTILS ]
# ==============================================================================
_ACCENT_MAP = str.maketrans(
    "àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ",
    "aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN",
)
_EMOJI_MAP = [
    ("🟢", "[BUY]"),
    ("🔴", "[SELL]"),
    ("🔥", "[CHAUD]"),
    ("↔️", "[PIVOT]"),
    ("↔", "[PIVOT]"),
    ("⚠️", "[PROCHE]"),
    ("⚠", "[PROCHE]"),
]


def _safe_pdf_str(text, max_chars=200):
    """Nettoie une chaîne pour l'export PDF."""
    if text is None:
        return ""
    try:
        s = str(text).translate(_ACCENT_MAP)
        for e, r in _EMOJI_MAP:
            s = s.replace(e, r)
        s = s.encode("latin-1", errors="replace").decode("latin-1")
        return s[: max_chars - 3] + "..." if len(s) > max_chars else s
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return ""


class PDF(FPDF):
    """Générateur de rapport PDF."""

    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(
            0,
            10,
            _safe_pdf_str("Rapport Scanner Bluestar - S/R"),
            border=0,
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_font("Helvetica", "", 8)
        self.cell(
            0,
            6,
            _safe_pdf_str(
                f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | v{SCANNER_VERSION}"
            ),
            border=0,
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(
            0, 10, _safe_pdf_str(title), border=0, align="L", new_x="LMARGIN", new_y="NEXT"
        )
        self.ln(4)

    def _column_widths(self, df):
        if "Timeframes" in df.columns:
            return {
                "Actif": 20,
                "Signal": 26,
                "Niveau": 22,
                "Type": 22,
                "Timeframes": 50,
                "Nb TF": 12,
                "Force Totale": 20,
                "Score": 18,
                "Statut": 22,
                "Distance %": 18,
                "Alerte": 55,
            }
        return {
            "Actif": 24,
            "Prix Actuel": 24,
            "Type": 20,
            "Niveau": 24,
            "Force": 20,
            "Score (1TF)": 18,
            "Statut": 22,
            "Dist. %": 16,
            "Dist. ATR": 16,
        }

    def chapter_body(self, df):
        if df is None or df.empty:
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 10, "Aucune donnee a afficher.")
            return
        col_widths = self._column_widths(df)
        cols = [c for c in col_widths if c in df.columns]
        total_w = sum(col_widths[c] for c in cols)
        x_start = self.l_margin + max(
            0, (self.w - self.l_margin - self.r_margin - total_w) / 2
        )
        self.set_font("Helvetica", "B", 7)
        self.set_x(x_start)
        for col in cols:
            self.cell(
                col_widths[col],
                6,
                _safe_pdf_str(col),
                border=1,
                align="C",
                new_x="RIGHT",
                new_y="TOP",
            )
        self.ln()
        self.set_font("Helvetica", "", 7)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col in cols:
                val = _safe_pdf_str(str(row[col]))
                max_c = int(col_widths[col] / 1.25)
                self.cell(
                    col_widths[col],
                    5,
                    (val[: max_c - 1] + "." if len(val) > max_c else val),
                    border=1,
                    align="C",
                    new_x="RIGHT",
                    new_y="TOP",
                )
            self.ln()


def create_pdf_report(results_dict, confluences_df=None, summary_list=None, anomalies=None):
    """Génère un rapport PDF complet."""
    pdf = PDF("L", "mm", "A4")
    pdf.set_margins(5, 10, 5)
    pdf.add_page()
    if anomalies:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            0,
            7,
            _safe_pdf_str("ALERTES ANOMALIES"),
            border=0,
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", "", 8)
        for sym, msg in anomalies.items():
            pdf.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}"))
        pdf.ln(4)
    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title("ZONES DE CONFLUENCE")
        pdf.chapter_body(confluences_df)
        pdf.ln(10)
    for tf, df in results_dict.items():
        if df is not None and not df.empty:
            pdf.chapter_title(f"Analyse {tf}")
            pdf.chapter_body(df)
            pdf.ln(10)
    return bytes(pdf.output())


def create_json_export(
    summary_list,
    confluences_df,
    max_dist=5.0,
    min_score=20.0,
    allowed_statuts=("Vierge", "Testee", "Role Reverse"),
):
    """Exporte les résultats au format JSON."""
    now_utc = datetime.now(timezone.utc)
    output = {
        "generated_at": now_utc.isoformat(),
        "scanner_version": SCANNER_VERSION,
        "assets": [],
    }
    summary_map = {
        s["symbol"]: s for s in summary_list if isinstance(s, dict) and "symbol" in s
    }

    filtered_conf = pd.DataFrame()
    if confluences_df is not None and not confluences_df.empty:
        df = confluences_df.copy()
        df["dist_num"] = pd.to_numeric(
            df["Distance %"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        ).fillna(999999.0)
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
        filtered_conf = df[
            (df["dist_num"] <= float(max_dist))
            & (df["Score"] >= float(min_score))
            & (df["Statut"].isin(tuple(allowed_statuts)))
        ].drop(columns=["dist_num"], errors="ignore")

    for sym, summary in summary_map.items():
        if not filtered_conf.empty and "Actif" in filtered_conf.columns:
            sym_zones = filtered_conf[filtered_conf["Actif"] == sym]
            zones = sym_zones.to_dict("records")
        else:
            zones = []
        output["assets"].append(
            {
                "symbol": sym,
                "current_price": summary.get("current_price"),
                "zones": zones,
            }
        )

    output["assets"].sort(key=lambda x: x["symbol"])
    return json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")


def create_llm_brief(
    summary_list,
    confluences_df,
    max_dist=2.0,
    min_score=100.0,
    allowed_statuts=("Vierge", "Testee", "Role Reverse"),
):
    """Crée un résumé textuel pour LLM."""
    lines = [
        "# BRIEF S/R — Scanner Bluestar",
        f"_Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}_",
        "",
    ]
    if confluences_df is None or confluences_df.empty:
        return "\n".join(lines).encode("utf-8")
    df = confluences_df.copy()
    df["dist_num"] = pd.to_numeric(
        df["Distance %"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(999999.0)
    df["Score"] = pd.to_numeric(df.get("Score"), errors="coerce").fillna(0.0)
    filtered = df[
        (df["dist_num"] <= float(max_dist))
        & (df["Score"] >= float(min_score))
        & (df["Statut"].isin(tuple(allowed_statuts)))
    ]
    for sym in filtered["Actif"].unique():
        lines.append(f"### {sym}")
        for _, row in filtered[filtered["Actif"] == sym].iterrows():
            lines.append(
                f"- {row['Signal']} `{row['Niveau']}` | Sc:{row['Score']} | "
                f"{row['Statut']} | {row['Distance %']} | {row['Timeframes']}"
            )
        lines.append("")
    return "\n".join(lines).encode("utf-8")


# ==============================================================================
# [ LAYER 6: STREAMLIT UI ]
# ==============================================================================
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")
st.title("📡 Scanner Bluestar Supports et Resistances")
st.markdown(
    "Zones S/R avec **Swing Adaptatif**, **Hybrid Touch Logic** "
    "et **Trend-Structure (fix signe v8.7)**."
)


def _is_scanning_locked(session_state):
    """Vérifie si le verrou de scan est actif."""
    lock_ts = session_state.get("scanning_lock_ts")
    if lock_ts and (time.time() - lock_ts) < _SCAN_LOCK_TTL_S:
        return True
    return False


def _coerce_dist_num(series: pd.Series) -> pd.Series:
    """Convertit une colonne de distance en numérique."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(999999.0)


with st.sidebar:
    st.header("1. Connexion OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets charges ✓")
    except KeyError:
        access_token, account_id = None, None
        st.error("Secrets OANDA manquants")

    st.header("2. Selection")
    select_all = st.checkbox(f"Tous les actifs ({len(ALL_SYMBOLS)})", value=True)
    symbols_to_scan = (
        ALL_SYMBOLS
        if select_all
        else st.multiselect(
            "Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"]
        )
    )

    st.header("3. Parametres LLM")
    llm_max_dist = st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    llm_min_score = st.slider("Score min JSON/LLM", 20, 300, 30, 10, key="llm_min_score")
    llm_statuts = st.multiselect(
        "Statuts autorises",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    st.header("4. Detection")
    min_touches = st.slider("Min touches Forex H4", 2, 10, 2, 1)
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 0.8, 0.1)
    max_dist_filter = st.slider("Filtre visuel Dist (%)", 1.0, 15.0, 3.0, 0.5)
    show_debug = st.checkbox("Afficher debug pipeline", value=False)

    if st.button("🧹 Vider le cache"):
        st.success(f"Cache vide : {_cache_clear()} entrees")
    if st.button("🔓 Forcer liberation lock"):
        st.session_state.pop("scanning_lock_ts", None)
        st.success("Lock libere")


scan_button = st.button(
    "🚀 LANCER LE SCAN COMPLET",
    type="primary",
    use_container_width=True,
    disabled=_is_scanning_locked(st.session_state),
)

if scan_button and symbols_to_scan and not _is_scanning_locked(st.session_state):
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
            raw_results = _run_async_isolated(
                lambda: run_institutional_scan(
                    symbols_to_scan, access_token, account_id, min_touches
                )
            )

            results_h4, results_daily, results_weekly = [], [], []
            all_zones_map, prices_map, trends_map = {}, {}, {}
            anomalies_map, scan_errors, bars_map = {}, {}, {}
            missing_tfs_map, price_fallback_map, debug_map = {}, {}, {}

            for idx, res in enumerate(raw_results):
                progress_bar.progress(
                    (idx + 1) / len(raw_results), text=f"Processing {res.symbol}..."
                )
                if res.scan_error:
                    scan_errors[res.symbol.replace("_", "/")] = res.scan_error
                    continue
                all_zones_map[res.symbol], prices_map[res.symbol] = res.zones, res.price
                trends_map[res.symbol], bars_map[res.symbol] = res.trends, res.bars_map
                if res.anomaly:
                    anomalies_map[res.symbol.replace("_", "/")] = res.anomaly
                if res.missing_tfs:
                    missing_tfs_map[res.symbol.replace("_", "/")] = res.missing_tfs
                price_fallback_map[res.symbol] = res.price_is_fallback
                debug_map[res.symbol.replace("_", "/")] = res.debug_info
                for tf, rows in res.rows.items():
                    if not rows:
                        continue
                    if tf == "H4":
                        results_h4.extend(rows)
                    elif tf == "Daily":
                        results_daily.extend(rows)
                    elif tf == "Weekly":
                        results_weekly.extend(rows)

            all_confs = []
            for sym in symbols_to_scan:
                if sym.replace("_", "/") in scan_errors:
                    continue
                sym_thresh = {
                    "US30_USD": 1.5,
                    "NAS100_USD": 1.5,
                    "SPX500_USD": 1.2,
                    "XAU_USD": 1.5,
                }.get(sym, confluence_threshold)
                all_confs.extend(
                    detect_confluences(
                        sym.replace("_", "/"),
                        all_zones_map.get(sym, {}),
                        prices_map.get(sym),
                        bars_map.get(sym, {}),
                        sym_thresh,
                    )
                )
            conf_df = pd.DataFrame(all_confs) if all_confs else pd.DataFrame()

            summaries = []
            for sym in symbols_to_scan:
                cp = prices_map.get(sym)
                ctx = ""
                if "Daily" in all_zones_map.get(sym, {}) and cp:
                    ctx = _build_daily_price_context(
                        cp, all_zones_map[sym]["Daily"][0], all_zones_map[sym]["Daily"][1]
                    )
                summaries.append(
                    {
                        "symbol": sym.replace("_", "/"),
                        "trend_h4": trends_map.get(sym, {}).get("H4", "NEUTRE"),
                        "trend_daily": trends_map.get(sym, {}).get("Daily", "NEUTRE"),
                        "trend_weekly": trends_map.get(sym, {}).get("Weekly", "NEUTRE"),
                        "price_context": ctx,
                        "current_price": cp,
                    }
                )

            df_h4 = pd.DataFrame(results_h4)
            df_d = pd.DataFrame(results_daily)
            df_w = pd.DataFrame(results_weekly)
            st.session_state["scan_results"] = {
                "df_h4": df_h4,
                "df_daily": df_d,
                "df_weekly": df_w,
                "conf_full": conf_df,
                "report_dict": {"H4": df_h4, "Daily": df_d, "Weekly": df_w},
                "summaries": summaries,
                "anomalies": anomalies_map,
                "scan_errors": scan_errors,
                "missing_tfs_map": missing_tfs_map,
                "debug_map": debug_map,
            }
            st.session_state.pop("scanning_lock_ts", None)
            st.success("Scan termine !")
            st.rerun()
        except (ScanTimeoutError, OandaAuthError, KeyError, ValueError) as e:
            st.error(f"Crash critique: {e}")
            st.session_state.pop("scanning_lock_ts", None)

if "scan_results" in st.session_state:
    res = st.session_state["scan_results"]
    if res["scan_errors"]:
        with st.expander("❌ Erreurs"):
            for s, e in res["scan_errors"].items():
                st.error(f"{s}: {e}")
    if res["anomalies"]:
        with st.expander("⚠️ Anomalies"):
            for s, m in res["anomalies"].items():
                st.warning(f"{s}: {m}")
    if show_debug and res.get("debug_map"):
        with st.expander("🔍 Debug pipeline (n_pivots / n_zones / n_trend_zones par TF)"):
            for s, dbg in res["debug_map"].items():
                st.write(f"**{s}**", dbg)
    if not res["conf_full"].empty:
        st.subheader("🔥 CONFLUENCES MULTI-TF")
        c_df = res["conf_full"].copy()
        c_df["dist_num"] = _coerce_dist_num(c_df["Distance %"])
        filtered_c = c_df[c_df["dist_num"] <= max_dist_filter].drop(columns=["dist_num"])
        st.dataframe(filtered_c.sort_values("Score", ascending=False), use_container_width=True)
    for label, df in [
        ("H4", res["df_h4"]),
        ("Daily", res["df_daily"]),
        ("Weekly", res["df_weekly"]),
    ]:
        st.subheader(f"Analyse {label}")
        if not df.empty:
            df_f = df.copy()
            df_f["dist_num"] = _coerce_dist_num(df_f["Dist. %"])
            st.dataframe(
                df_f[df_f["dist_num"] <= max_dist_filter].drop(columns=["dist_num"]),
                use_container_width=True,
            )

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_b = create_pdf_report(
            res["report_dict"], res["conf_full"], res["summaries"], res["anomalies"]
        )
        st.download_button("📄 PDF", data=pdf_b, file_name="rapport_bluestar.pdf")
    with col2:
        json_b = create_json_export(
            res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, tuple(llm_statuts)
        )
        st.download_button("🔧 JSON", data=json_b, file_name="supports et resistances.json")
    with col3:
        llm_b = create_llm_brief(
            res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, tuple(llm_statuts)
        )
        st.download_button("🤖 LLM Brief", data=llm_b, file_name="brief_llm.md")
