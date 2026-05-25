# pylint: disable=too-many-lines
"""Scanner Bluestar S/R Multi-Timeframes — v8.3.4-PROD-HARDENED

Durcissement chirurgical production-grade du fichier 8.3.3-HARDENED.
Compatibilité fonctionnelle stricte préservée. Aucune régression métier.

Changements v8.3.4 (vs 8.3.3) :
  C1.  flag_data_anomaly décomposé (CC 24 → 3 sous-fonctions CC ≤ 6).
  C2.  _process_symbol décomposé en helpers (_extract_last_candle_info,
       _collect_support_levels).
  C3.  Renommage paramètres pour éliminer collisions W0621 (account_id,
       min_touches, confluence_threshold, tb, summaries).
  C4.  _validate_price_bounds : price=None retourné si hors bornes
       (pas d'exposition prix corrompu).
  C5.  _hash_df enrichi : shape + volume sum + first/last open
       (réduction risque collision cache).
  C6.  Pattern token court ajouté dans _TOKEN_REDACT_PATTERNS.
  C7.  Logging debug dans except silencieux (observabilité prod).
  C8.  _run_async_isolated : timeout sur gather de fermeture.
  C9.  _apply_pdf_filter : fillna explicite (pas de NaN silencieux).
  C10. _cache_clear thread-safe exposé.
  C11. Guard atomique anti-double-scan via session_state.
  C12. Docstrings densifiés sur fonctions critiques.
  C13. Type hints renforcés (Final, Tuple, Optional cohérents).
  C14. _safe_pdf_str traite None et types non-str sans crash.
  C15. _filter_and_sort robuste à df=None.

Dépendances figées (testées) :
  - fpdf2 >= 2.7.0 (signature new_x/new_y)
  - pandas >= 1.5
  - aiohttp >= 3.8
  - streamlit >= 1.28
  - scipy >= 1.10
  - numpy >= 1.24
"""
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Final, Optional, Tuple, List, Dict, Set, Callable

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
# [ LAYER 0: GLOBAL CONFIG & LOGGING — filtre sanitisation tokens ]
# ==============================================================================
_TOKEN_REDACT_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE),
    re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s]+", re.IGNORECASE),
    re.compile(r"(access_token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE),
    re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{20,}=*", re.IGNORECASE),
    # C6: pattern court avec contexte explicite OANDA
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9\-\._~\+\/]{8,}", re.IGNORECASE),
    re.compile(r"(\b[a-f0-9]{32}-[a-f0-9]{32}\b)", re.IGNORECASE),  # OANDA-like token format
]


def _redact_sensitive(text: Any) -> Any:
    """Sanitisation regex appliquée systématiquement aux strings sensibles.

    Idempotente, tolérante à None et non-str (passthrough).
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for pat in _TOKEN_REDACT_PATTERNS:
        try:
            out = pat.sub(lambda m: m.group(1) + "***REDACTED***" if m.lastindex else "***REDACTED***", out)
        except (re.error, IndexError):
            continue
    return out


class _SensitiveDataFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Filtre logger : sanitise message + args avant émission.

    Un filtre logger NE DOIT JAMAIS lever — toute exception est avalée
    silencieusement pour ne pas casser le logger lui-même.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.msg and isinstance(record.msg, str):
                record.msg = _redact_sensitive(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: _redact_sensitive(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        _redact_sensitive(a) if isinstance(a, str) else a
                        for a in record.args
                    )
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # contrat : un filtre ne lève jamais
        return True


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addFilter(_SensitiveDataFilter())


# ==============================================================================
# [ LAYER 0b: EXCEPTIONS PERSONNALISÉES ]
# ==============================================================================
class OandaAuthError(Exception):
    """Levée quand l'authentification OANDA échoue."""


class DataValidationError(Exception):
    """Levée quand les données reçues sont structurellement invalides."""


# ==============================================================================
# [ LAYER 0c: THREAD-SAFE DATA CACHE — TTL négatif dédié, reset exposé ]
# ==============================================================================
_CACHE_TTL_BY_TF: Final[Dict[str, int]] = {"h4": 60, "daily": 300, "weekly": 600}
_CACHE_TTL_DEFAULT: Final[int] = 300
_CACHE_TTL_NEGATIVE: Final[int] = 20
_CACHE_MAX_ENTRIES: Final[int] = 256
_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_CACHE_EMPTY: Final[object] = object()
_OANDA_CACHE: "OrderedDict[Tuple[str, str, str, str], Tuple[float, Any]]" = OrderedDict()


def _cache_ttl(tf: str, is_empty: bool = False) -> int:
    """TTL distinct succès vs échecs (cache négatif court)."""
    if is_empty:
        return _CACHE_TTL_NEGATIVE
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)


def _cache_is_fresh(fetched_at: float, tf: str, is_empty: bool) -> bool:
    return (time.monotonic() - fetched_at) <= _cache_ttl(tf, is_empty)


def _cache_key(env_url: Optional[str], acct_id: str, symbol: str, tf: str) -> Tuple[str, str, str, str]:
    return (env_url or "unknown_env", acct_id or "unknown_account", symbol, tf)


def _cache_purge_stale_locked() -> None:
    """Doit être appelé sous _CACHE_LOCK acquis."""
    now = time.monotonic()
    stale: List[Tuple[str, str, str, str]] = []
    for k, (ts, payload) in _OANDA_CACHE.items():
        is_empty = payload is _CACHE_EMPTY
        if (now - ts) > _cache_ttl(k[3], is_empty):
            stale.append(k)
    for k in stale:
        _OANDA_CACHE.pop(k, None)
    while len(_OANDA_CACHE) > _CACHE_MAX_ENTRIES:
        _OANDA_CACHE.popitem(last=False)


def _cache_get(
    env_url: Optional[str], acct_id: str, symbol: str, tf: str,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Retourne (hit, payload).

    - (False, None) : miss.
    - (True, None)  : hit cache négatif (échec mémorisé).
    - (True, df)    : hit succès.
    """
    k = _cache_key(env_url, acct_id, symbol, tf)
    with _CACHE_LOCK:
        _cache_purge_stale_locked()
        entry = _OANDA_CACHE.get(k)
        if entry is None:
            return False, None
        fetched_at, payload = entry
        is_empty = payload is _CACHE_EMPTY
        if not _cache_is_fresh(fetched_at, tf, is_empty):
            _OANDA_CACHE.pop(k, None)
            return False, None
        _OANDA_CACHE.move_to_end(k)
        if is_empty:
            return True, None
        return True, payload.copy(deep=True)


def _cache_set(
    env_url: Optional[str], acct_id: str, symbol: str, tf: str,
    df: Optional[pd.DataFrame],
) -> None:
    k = _cache_key(env_url, acct_id, symbol, tf)
    payload: Any = _CACHE_EMPTY if df is None else df
    with _CACHE_LOCK:
        _OANDA_CACHE[k] = (time.monotonic(), payload)
        _OANDA_CACHE.move_to_end(k)
        _cache_purge_stale_locked()


def _cache_clear() -> int:
    """Reset thread-safe du cache OANDA. Retourne le nombre d'entrées purgées."""
    with _CACHE_LOCK:
        n = len(_OANDA_CACHE)
        _OANDA_CACHE.clear()
        return n


SCANNER_VERSION: Final[str] = "8.3.4-PROD-HARDENED"

ALL_SYMBOLS: Final[List[str]] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

_GRANULARITY_MAP: Final[Dict[str, str]] = {"h4": "H4", "daily": "D", "weekly": "W"}
TF_WEIGHT: Final[Dict[str, float]] = {"H4": 1.0, "Daily": 2.0, "Weekly": 3.0}
_TF_LAMBDA: Final[Dict[str, float]] = {"H4": 2.0, "Daily": 1.0, "Weekly": 0.5}
_TF_PERIOD_HOURS: Final[Dict[str, float]] = {"h4": 4.0, "daily": 24.0, "weekly": 168.0}
_TF_MAX_STALE_HOURS: Final[Dict[str, float]] = {"h4": 12.0, "daily": 72.0, "weekly": 336.0}

_OANDA_SEMAPHORE_LIMIT: Final[int] = 15


# ==============================================================================
# [ HELPER : HASH STABLE ET SÉMANTIQUE POUR @st.cache_data ]
# ==============================================================================
def _hash_df(df: Optional[pd.DataFrame]) -> str:
    """Hash sémantiquement stable d'un DataFrame OHLCV.

    Compose : shape, first/last timestamp, first/last open/close,
    volume sum, et hash pandas d'un échantillon stratifié (head+middle+tail).

    Réduit drastiquement le risque de collision vs (len, last_ts, last_close).
    """
    if df is None or df.empty:
        return "empty"
    try:
        n = len(df)
        ncols = len(df.columns)
        cols_present = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if not cols_present:
            return f"no_ohlc_{n}_{ncols}"

        first_ts = str(df.index[0])
        last_ts = str(df.index[-1])
        first_close = float(df["close"].iloc[0]) if "close" in df.columns else 0.0
        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        first_open = float(df["open"].iloc[0]) if "open" in df.columns else 0.0

        # Volume sum si présent (faible coût, fort discriminant)
        vol_sig: float = 0.0
        if "volume" in df.columns:
            try:
                vol_sig = float(df["volume"].sum())
            except (TypeError, ValueError):
                vol_sig = 0.0

        # Échantillon stratifié borné (max 24 lignes)
        if n <= 24:
            sample = df[cols_present]
        else:
            mid_start = max(0, n // 2 - 4)
            sample = pd.concat([
                df[cols_present].iloc[:8],
                df[cols_present].iloc[mid_start:mid_start + 8],
                df[cols_present].iloc[-8:],
            ])
        try:
            sample_hash = int(pd.util.hash_pandas_object(sample, index=False).sum()) & 0xFFFFFFFFFFFFFFFF
        except (TypeError, ValueError, AttributeError):
            sample_hash = 0

        return (f"{n}|{ncols}|{first_ts}|{last_ts}|"
                f"{first_open:.8f}|{first_close:.8f}|{last_close:.8f}|"
                f"{vol_sig:.2f}|{sample_hash:016x}")
    except (KeyError, IndexError, TypeError, ValueError):
        return f"unhashable_{len(df) if df is not None else 0}"


def _hash_series(s: Optional[pd.Series]) -> str:
    """Hash stable d'une Series. Échantillonnage head+mid+tail."""
    if s is None or len(s) == 0:
        return "empty_series"
    try:
        n = len(s)
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        mid = float(s.iloc[n // 2]) if n > 2 else last
        return f"{n}|{first:.8f}|{mid:.8f}|{last:.8f}"
    except (IndexError, TypeError, ValueError):
        return f"unhashable_series_{len(s) if s is not None else 0}"


# ==============================================================================
# [ LAYER 1: INSTRUMENT PROFILES ]
# ==============================================================================
@dataclass(frozen=True)
class InstrumentProfile:  # pylint: disable=too-many-instance-attributes
    """Profil d'instrument figé (immutable) : seuils calibrés par classe d'actif."""
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


# pylint: disable=line-too-long
_PROFILES: Final[Dict[str, InstrumentProfile]] = {
    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2,  0.8,  0.6,  1.5, False, 0.20, 0.30, 1.0,  5.0, 5.0),
    "GBP_USD":    InstrumentProfile("GBP_USD",    "FOREX", 0.0001, 1.3,  0.85, 0.65, 1.5, False, 0.20, 0.30, 1.0,  5.0, 5.0),
    "USD_JPY":    InstrumentProfile("USD_JPY",    "FOREX", 0.01,   0.9,  0.5,  0.5,  1.5, False, 0.20, 0.30, 1.0,  5.0, 5.0),
    "XAU_USD":    InstrumentProfile("XAU_USD",    "METAL", 0.01,   2.0,  1.2,  1.0,  3.0, True,  0.18, 0.28, 1.5, 10.0, 8.0, price_min=1500.0, price_max=6000.0),
    "US30_USD":   InstrumentProfile("US30_USD",   "INDEX", 1.0,    1.5,  0.9,  0.7,  2.5, True,  0.22, 0.32, 1.5,  8.0, 5.0, price_min=25000.0, price_max=60000.0),
    "NAS100_USD": InstrumentProfile("NAS100_USD", "INDEX", 1.0,    1.5,  1.0,  0.8,  2.5, True,  0.22, 0.32, 1.5,  8.0, 5.0, price_min=10000.0, price_max=50000.0),
    "SPX500_USD": InstrumentProfile("SPX500_USD", "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  0.22, 0.32, 1.2,  8.0, 5.0, price_min=3000.0,  price_max=12000.0),
    "DE30_EUR":   InstrumentProfile("DE30_EUR",   "INDEX", 0.1,    1.3,  0.8,  0.65, 2.0, True,  0.22, 0.32, 1.2,  8.0, 5.0, price_min=10000.0, price_max=30000.0),
}
# pylint: enable=line-too-long
_DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile(
    "DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
)


def get_profile(symbol: str) -> InstrumentProfile:
    """Résolution profil par symbole avec fallback heuristique."""
    if symbol in _PROFILES:
        return _PROFILES[symbol]
    base = symbol.split("_")[0] if "_" in symbol else symbol
    if symbol.endswith("_JPY") or base == "JPY":
        return InstrumentProfile(symbol, "FOREX", 0.01, 0.9, 0.5, 0.5, 1.5, False)
    if base in ("EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "USD"):
        return InstrumentProfile(symbol, "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False)
    return _DEFAULT_PROFILE


# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE (OANDA) — sanitization OHLC renforcée ]
# ==============================================================================
_MAX_HIGH_LOW_RATIO: Final[float] = 1.5


def _is_valid_candle_dict(c: dict) -> bool:
    """Validation OHLC stricte d'un dict de bougie OANDA. Retourne False pour tout doute."""
    try:
        mid = c["mid"]
        o = float(mid["o"])
        h = float(mid["h"])
        lo = float(mid["l"])
        cl = float(mid["c"])
    except (KeyError, ValueError, TypeError):
        return False
    if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(lo) and np.isfinite(cl)):
        return False
    if lo <= 0 or h <= 0:
        return False
    if h < lo:
        return False
    if not lo <= o <= h:
        return False
    if not lo <= cl <= h:
        return False
    if lo > 0 and (h / lo) > _MAX_HIGH_LOW_RATIO:
        return False
    return True


def _sanitize_ohlc_dataframe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Sanitization défensive d'un DataFrame OHLC.

    Étapes : drop NaN OHLC, sort_index, dédoublonnage, validation vectorisée
    (cohérence OHLC + ratio high/low borné).
    """
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
            np.isfinite(out["open"]) & np.isfinite(out["high"])
            & np.isfinite(out["low"]) & np.isfinite(out["close"])
            & (out["low"] > 0) & (out["high"] > 0)
            & (out["high"] >= out["low"])
            & (out["open"].between(out["low"], out["high"]))
            & (out["close"].between(out["low"], out["high"]))
        )
        ratio_ok = (out["high"] / out["low"].replace(0, np.nan)) <= _MAX_HIGH_LOW_RATIO
        mask = mask & ratio_ok.fillna(False)
        out = out[mask]
        return out if not out.empty else None
    except (KeyError, ValueError, TypeError) as e:
        logging.warning("OHLC sanitization failed: %s", type(e).__name__)
        return None


class AsyncOandaClient:
    """Client OANDA async avec retry exponential backoff et cache thread-safe."""

    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers: Dict[str, str] = {"Authorization": f"Bearer {token}"}
        self.account_id: str = oanda_account_id
        self.env_url: Optional[str] = None

    @staticmethod
    async def _get_json_with_retry(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        session: aiohttp.ClientSession,
        url: str,
        headers: dict,
        params: dict,
        timeout_total: float,
        retries: int = 3,
    ) -> Optional[dict]:
        """GET JSON avec retry exponentiel sur erreurs 429/5xx et erreurs réseau."""
        backoff = 0.5
        for attempt in range(retries):
            try:
                async with session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_total),
                ) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status in (429, 500, 502, 503, 504):
                        if attempt < retries - 1:
                            await asyncio.sleep(backoff * (2 ** attempt))
                            continue
                        return None
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                return None
        return None

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        """Détecte l'env OANDA (practice puis live). Retourne True si auth OK."""
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

    async def fetch_candles(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
        tf: str,
        limit: int = 500,
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """Fetch bougies pour (symbol, tf) avec cache, retry, et sanitization OHLC."""
        cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, tf)
        if cache_hit:
            logging.debug("Cache HIT: %s/%s", symbol, tf)
            return symbol, tf, cached

        gran = _GRANULARITY_MAP.get(tf)
        if not gran or not self.env_url:
            return symbol, tf, None

        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        params = {"count": limit + 1, "granularity": gran, "price": "M"}

        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, params, timeout_total=10, retries=3,
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
                    if c.get("complete") and _is_valid_candle_dict(c)
                ]
            except (KeyError, ValueError, TypeError) as e:
                logging.warning("Candle parse error for %s/%s: %s", symbol, tf, type(e).__name__)
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
                age_hours = (
                    datetime.now(timezone.utc) - last_ts.to_pydatetime()
                ).total_seconds() / 3600.0
                max_stale = _TF_MAX_STALE_HOURS.get(tf, 72.0)
                if age_hours > max_stale:
                    logging.warning(
                        "Stale data for %s/%s: last candle %.1fh old (max %.1fh)",
                        symbol, tf, age_hours, max_stale,
                    )

            _cache_set(self.env_url, self.account_id, symbol, tf, df_clean)
            return symbol, tf, df_clean

    async def fetch_price(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
    ) -> Tuple[str, Optional[float]]:
        """Fetch prix mid live (bid+ask)/2 avec validation finitude."""
        if not self.env_url:
            return symbol, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, {"instruments": symbol},
                timeout_total=5, retries=3,
            )
            if data is None:
                return symbol, None
            try:
                if "prices" in data and data["prices"]:
                    bid = float(data["prices"][0]["closeoutBid"])
                    ask = float(data["prices"][0]["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
                        return symbol, (bid + ask) / 2
            except (KeyError, ValueError, TypeError, IndexError):
                pass
        return symbol, None


# ==============================================================================
# [ LAYER 2b: ASYNC RUNNER — worker thread isolé, timeout fermeture ]
# ==============================================================================
def _run_async_isolated(coro_factory: Callable[[], Any], timeout: float = 300.0) -> Any:
    """Exécute une coroutine dans un event loop isolé (thread dédié).

    - Si pas de loop courant : `asyncio.run()`.
    - Si loop déjà actif (Streamlit + jupyter) : worker thread isolé, loop neuf.
    - Annulation propre des tâches en suspens à la fermeture (timeout borné).

    `coro_factory` doit retourner une coroutine fraîche (non réutilisable).
    """
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
                        # C8: timeout borné sur la fermeture pour éviter un hang
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=5.0,
                            )
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        logging.warning("Some async tasks did not cancel within 5s")
                    except Exception:  # pylint: disable=broad-exception-caught
                        logging.exception("Error during async task cleanup")
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(None)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="oanda-async",
    ) as ex:
        future = ex.submit(_worker)
        return future.result(timeout=timeout)


# ==============================================================================
# [ LAYER 3: QUANTITATIVE ENGINE (VECTORIZED + CACHED) ]
# ==============================================================================
@st.cache_data(ttl=120, max_entries=512, show_spinner=False,
               hash_funcs={pd.DataFrame: _hash_df})
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """ATR classique avec fallback range moyen si insuffisant pour rolling."""
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


@st.cache_data(ttl=120, max_entries=512, show_spinner=False,
               hash_funcs={pd.Series: _hash_series})
def compute_institutional_trend(
    closes: pd.Series, lookback: int = 20, threshold: float = 2.0,
) -> str:
    """Trend par régression linéaire t-test sur série normalisée."""
    if closes is None or len(closes) < lookback:
        return "NEUTRE"
    y = closes.tail(lookback).values.astype(float)
    if not np.all(np.isfinite(y)):
        return "NEUTRE"
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


def detect_swing_pivots(  # pylint: disable=too-many-locals
    df: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> Tuple[pd.Series, pd.Series]:
    """Détection de swing highs/lows vectorisée avec filtre wick + prominence ATR."""
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
        pd.Series(highs.values[idx_highs], index=idx_highs)
        if idx_highs else pd.Series(dtype=float),
        pd.Series(lows.values[idx_lows], index=idx_lows)
        if idx_lows else pd.Series(dtype=float),
    )


def agglomerative_1d_clustering(
    price_weight_pairs: List[tuple],
    bandwidth: float,
) -> List[List[tuple]]:
    """Clustering 1D agglomératif simple sur prix avec bandwidth = k × ATR."""
    if not price_weight_pairs or bandwidth <= 0:
        return [[pw] for pw in price_weight_pairs] if price_weight_pairs else []
    sorted_pw = sorted(price_weight_pairs, key=lambda x: x[0])
    clusters: List[List[tuple]] = []
    curr_cluster = [sorted_pw[0]]
    for i in range(1, len(sorted_pw)):
        gap = sorted_pw[i][0] - sorted_pw[i - 1][0]
        if gap > bandwidth or (
            curr_cluster and (sorted_pw[i][0] - curr_cluster[0][0]) > 2.5 * bandwidth
        ):
            clusters.append(curr_cluster)
            curr_cluster = [sorted_pw[i]]
        else:
            curr_cluster.append(sorted_pw[i])
    clusters.append(curr_cluster)
    return clusters


def classify_zone_status(  # pylint: disable=too-many-locals,too-many-return-statements
    level: float,
    zone_type: str,
    df: pd.DataFrame,
    formation_idx: int,
    atr_val: float,
) -> str:
    """Classifie une zone : Vierge / Testee / Role Reverse / Consommee."""
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

    second_break = (
        (rc_after > level + tolerance) if zone_type == "Support"
        else (rc_after < level - tolerance)
    )
    return "Consommee" if second_break.any() else "Role Reverse"


def compute_structural_score(
    strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int,
) -> float:
    """Score pondéré : Force × Poids_TF × NbTF × exp(-λ × age_relatif)."""
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    age_r = max(age_bars, 0) / max(total_bars, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    age_f = float(np.exp(-lam * age_r))
    return round((strength * tf_w * nb_tf) * age_f, 1)


_STATUS_PRIORITY: Final[Dict[str, int]] = {
    "Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3,
}
_PIVOT_FALLBACK_DIST: Final[Dict[str, int]] = {"h4": 5, "daily": 8, "weekly": 10}


def _get_pivots_with_fallback(
    df: pd.DataFrame,
    profile: InstrumentProfile,
    atr_val: float,
    timeframe: str,
) -> Tuple[pd.Series, pd.Series]:
    """Détection swing avec fallback scipy.find_peaks si <3 pivots."""
    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)
    if len(pivot_highs) + len(pivot_lows) >= 3:
        return pivot_highs, pivot_lows

    n_total = len(df)
    dist = _PIVOT_FALLBACK_DIST.get(timeframe, 5)
    pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
    r_idx, _ = find_peaks(df["high"].values, **pk)
    s_idx, _ = find_peaks(-df["low"].values, **pk)
    safe_cutoff = n_total - 3
    r_idx = [i for i in r_idx if i < safe_cutoff]
    s_idx = [i for i in s_idx if i < safe_cutoff]
    pivot_highs = (
        pd.Series(df["high"].values[r_idx], index=r_idx) if r_idx else pd.Series(dtype=float)
    )
    pivot_lows = (
        pd.Series(df["low"].values[s_idx], index=s_idx) if s_idx else pd.Series(dtype=float)
    )
    return pivot_highs, pivot_lows


def _clusters_to_zones(  # pylint: disable=too-many-locals
    clusters_raw: list,
    min_touches_required: int,
    n_total: int,
    df: pd.DataFrame,
    atr_val: float,
) -> List[Dict[str, Any]]:
    """Convertit clusters de pivots en zones structurées avec status."""
    strong: List[Dict[str, Any]] = []
    for grp_pw in clusters_raw:
        if len(grp_pw) < min_touches_required:
            continue
        grp_prices_arr = np.array([item[0] for item in grp_pw])
        grp_weights_arr = np.array([item[1] for item in grp_pw])
        grp_indices = [item[2] for item in grp_pw]
        grp_ptypes = [item[3] for item in grp_pw]
        if grp_weights_arr.sum() <= 0:
            continue
        lvl = float(np.average(grp_prices_arr, weights=grp_weights_arr))
        if lvl <= 0 or not np.isfinite(lvl):
            continue
        last_idx = max(grp_indices)
        age = max(0, n_total - 1 - last_idx)
        ztype = (
            "Resistance" if grp_ptypes.count("high") >= grp_ptypes.count("low")
            else "Support"
        )
        status = classify_zone_status(lvl, ztype, df, last_idx, atr_val)
        strong.append({
            "level": float(lvl), "strength": len(grp_pw),
            "age_bars": age, "status": status,
        })
    return strong


def _merge_adjacent_zones(
    strong: List[Dict[str, Any]], merge_thresh: float,
) -> List[Dict[str, Any]]:
    """Merge zones adjacentes par seuil ATR. Score conservé par priorité de statut."""
    strong.sort(key=lambda x: x["level"])
    merged: List[Dict[str, Any]] = []
    for z in strong:
        if not merged or abs(z["level"] - merged[-1]["level"]) > merge_thresh:
            merged.append(z)
            continue
        prev = merged[-1]
        new_str = prev["strength"] + z["strength"]
        new_lvl = (
            prev["level"] * prev["strength"] + z["level"] * z["strength"]
        ) / new_str
        new_status = max(
            [prev["status"], z["status"]],
            key=lambda s: _STATUS_PRIORITY.get(s, 1),
        )
        merged[-1] = {
            "level": new_lvl,
            "strength": new_str,
            "age_bars": min(prev["age_bars"], z["age_bars"]),
            "status": new_status,
        }
    return merged


@st.cache_data(ttl=120, max_entries=256, show_spinner=False,
               hash_funcs={pd.DataFrame: _hash_df})
def find_strong_sr_zones(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    df: pd.DataFrame,
    current_price: float,
    symbol: str,
    atr_val: Optional[float],
    timeframe: str,
    min_touches_required: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Détection zones S/R fortes : pivots → clustering → status → merge."""
    if atr_val is None or atr_val <= 0 or df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None or current_price <= 0 or not np.isfinite(current_price):
        return pd.DataFrame(), pd.DataFrame()

    profile = get_profile(symbol)
    n_total = len(df)

    pivot_highs, pivot_lows = _get_pivots_with_fallback(df, profile, atr_val, timeframe)

    pivot_records = []
    pid = 0
    for i, p in pivot_highs.items():
        pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "high"))
        pid += 1
    for i, p in pivot_lows.items():
        pivot_records.append((pid, float(p), (int(i) + 1e-6) / n_total, int(i), "low"))
        pid += 1

    if not pivot_records:
        return pd.DataFrame(), pd.DataFrame()

    bandwidth = atr_val * profile.cluster_radius_atr
    price_weight_pairs = [(r[1], r[2], r[3], r[4]) for r in pivot_records]
    clusters_raw = agglomerative_1d_clustering(price_weight_pairs, bandwidth)

    strong = _clusters_to_zones(clusters_raw, min_touches_required, n_total, df, atr_val)
    if not strong:
        return pd.DataFrame(), pd.DataFrame()

    merged = _merge_adjacent_zones(strong, atr_val * profile.merge_threshold_atr)

    df_zones = pd.DataFrame(merged).sort_values("level").reset_index(drop=True)
    df_zones["near_price"] = (
        np.abs(df_zones["level"] - current_price) / current_price * 100
    ) <= 0.50
    return (
        df_zones[df_zones["level"] < current_price].copy(),
        df_zones[df_zones["level"] >= current_price].copy(),
    )


def _flatten_zones_to_dataframe(zones_dict: dict) -> pd.DataFrame:
    """Aplatit le dict {tf: (sup_df, res_df)} en un DataFrame unique typé."""
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
                type=tmp["near_price"].map({True: "Pivot", False: ztype}),
            )
            frames.append(
                tmp[["tf", "level", "strength", "age_bars", "status", "type", "near_price"]]
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)


def _score_and_classify_group(  # pylint: disable=too-many-locals
    group: pd.DataFrame,
    current_price: float,
    bars_map: dict,
    symbol: str,
) -> dict:
    """Score un groupe de zones confluentes et classifie (BUY/SELL/PIVOT)."""
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

    tfs = group["tf"].unique()
    return {
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
        "Alerte": (
            "🔥 ZONE CHAUDE" if sub_dist < 0.5
            else ("⚠️ Proche" if sub_dist < 1.5 else "")
        ),
    }


def detect_confluences(  # pylint: disable=too-many-locals
    symbol: str,
    zones_dict: dict,
    current_price: float,
    bars_map: dict,
    confluence_threshold_pct: Optional[float] = None,
) -> list:
    """Détecte confluences multi-TF par union-find sur niveaux proches."""
    if (not zones_dict or not current_price or current_price <= 0
            or not np.isfinite(current_price)):
        return []

    z_df = _flatten_zones_to_dataframe(zones_dict)
    if z_df.empty:
        return []

    profile = get_profile(symbol.replace("/", "_"))
    threshold = (
        confluence_threshold_pct if confluence_threshold_pct is not None
        else profile.confluence_threshold_pct
    )

    z_df = z_df.sort_values("level").reset_index(drop=True)
    n = len(z_df)
    levels_arr = z_df["level"].values

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
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
        j = i + 1
        while j < n:
            dist_pct = (levels_arr[j] - li) / li * 100
            if dist_pct > threshold:
                break
            union(i, j)
            j += 1

    comp_map: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        comp_map.setdefault(root, []).append(idx)

    confluences = []
    for indices in comp_map.values():
        if len(indices) < 1:
            continue
        group_full = z_df.iloc[indices]
        if group_full["tf"].nunique() < 2:
            continue
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
class ScanResult:  # pylint: disable=too-many-instance-attributes
    """Résultat d'analyse complet pour un symbole. Tous les champs sont remplis
    même en cas d'erreur partielle (anomaly/scan_error documentent les soucis)."""
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


@dataclass(frozen=True)
class _RowContext:
    """Contexte immutable passé à _make_row (frozen pour empêcher la mutation)."""
    cp: float
    atr_val: float
    sym_d: str
    tf_name: str
    df_len: int
    profile: InstrumentProfile


def _make_row(z: dict, ztype: str, ctx: _RowContext) -> Dict[str, Any]:
    """Construit une ligne d'affichage normalisée pour un niveau S/R."""
    dist = abs(ctx.cp - z["level"]) / ctx.cp * 100 if ctx.cp else 0.0
    dist_atr = (
        f"{round(abs(ctx.cp - z['level']) / ctx.atr_val, 1)}x"
        if (ctx.atr_val and ctx.atr_val > 0) else "N/A"
    )
    in_pdf = dist <= ctx.profile.pdf_max_dist_pct
    return {
        "Actif": ctx.sym_d,
        "Prix Actuel": f"{ctx.cp:.5f}" if ctx.cp else "N/A",
        "Type": ztype,
        "Niveau": f"{z['level']:.5f}",
        "Force": f"{z['strength']} touches",
        "Score (1TF)": compute_structural_score(
            z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len,
        ),
        "Statut": z["status"],
        "Dist. %": f"{dist:.2f}%",
        "Dist. ATR": dist_atr,
        "_dist_num": dist,
        "_in_pdf": in_pdf,
    }


async def _fetch_live_prices(
    client: AsyncOandaClient,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbols: List[str],
) -> Dict[str, Optional[float]]:
    """Fetch prices live en parallèle. Erreurs isolées par symbole."""
    price_tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
    prices_res = await asyncio.gather(*price_tasks, return_exceptions=True)
    out: Dict[str, Optional[float]] = {}
    for sym, item in zip(symbols, prices_res):
        if isinstance(item, Exception):
            logging.error("Price fetch exception for %s: %s", sym, type(item).__name__)
            out[sym] = None
        else:
            out[item[0]] = item[1]
    return out


async def _fetch_candles_cube(
    client: AsyncOandaClient,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbols: List[str],
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """Fetch toutes les bougies (sym × tf) en parallèle. Erreurs isolées."""
    candle_tasks = [
        client.fetch_candles(session, sem, sym, tf)
        for sym in symbols
        for tf in _GRANULARITY_MAP
    ]
    candles_res = await asyncio.gather(*candle_tasks, return_exceptions=True)
    data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
    for item in candles_res:
        if isinstance(item, Exception):
            logging.error("Candle fetch exception: %s", type(item).__name__)
            continue
        sym, tf, df = item
        data_cube.setdefault(sym, {})[tf] = df
    return data_cube


def _build_daily_price_context(cp: float, sup: pd.DataFrame, res: pd.DataFrame) -> str:
    """Contexte texte position prix vs supports/résistances Daily (top 1 chacun)."""
    parts = []
    if not sup.empty:
        s_near = sup[(sup["level"] < cp) & (abs(sup["level"] - cp) / cp * 100 <= 5.0)]
        if not s_near.empty:
            n_s = s_near.nlargest(1, "level").iloc[0]
            d_s = abs(cp - n_s["level"]) / cp * 100
            label = "SUR support" if d_s < 0.5 else "S proche"
            parts.append(f"{label}: {n_s['level']:.5f} (-{d_s:.2f}%)")
    if not res.empty:
        r_near = res[(res["level"] > cp) & (abs(res["level"] - cp) / cp * 100 <= 5.0)]
        if not r_near.empty:
            n_r = r_near.nsmallest(1, "level").iloc[0]
            d_r = abs(cp - n_r["level"]) / cp * 100
            label = "SUR resistance" if d_r < 0.5 else "R proche"
            parts.append(f"{label}: {n_r['level']:.5f} (+{d_r:.2f}%)")
    return "  |  ".join(parts) if parts else "Zone intermediaire"


_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {
    "H4": (50, 2.0), "Daily": (50, 1.8), "Weekly": (20, 1.5),
}


def _process_tf_frame(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sym: str,
    tf_k: str,
    tf_name: str,
    df: pd.DataFrame,
    cp: float,
    min_touches_ui: int,
    profile: InstrumentProfile,
    sym_d: str,
) -> Tuple[Optional[list], Optional[tuple], str]:
    """Traite un TF pour un symbole avec isolation totale des erreurs."""
    try:
        atr_val = compute_atr(df)
        if atr_val is None:
            logging.warning("ATR non calculable pour %s/%s — zones ignorées.", sym, tf_name)
            return None, None, ""

        min_t = max(3, min_touches_ui) if tf_k == "h4" else max(2, min_touches_ui)
        sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
        zone_pair = (sup, res)
        price_ctx = _build_daily_price_context(cp, sup, res) if tf_k == "daily" else ""

        row_ctx = _RowContext(
            cp=cp, atr_val=atr_val, sym_d=sym_d,
            tf_name=tf_name, df_len=len(df), profile=profile,
        )
        tf_r = (
            [_make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx)
             for _, z in sup.iterrows()]
            + [_make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx)
               for _, z in res.iterrows()]
        )
        seen: Set[Tuple[str, str]] = set()
        uniq = [
            r for r in tf_r
            if (key := (r["Niveau"], r["Type"])) not in seen and not seen.add(key)
        ]
        return (uniq if uniq else None), zone_pair, price_ctx
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
        logging.warning("TF processing error %s/%s: %s", sym, tf_name, type(e).__name__)
        return None, None, ""


def _validate_price_bounds(
    sym: str, cp_live: float, profile: InstrumentProfile,
) -> Optional[ScanResult]:
    """Valide bornes profil. Retourne ScanResult d'erreur (price=None) si hors bornes."""
    if profile.price_min is not None and cp_live < profile.price_min:
        return ScanResult(
            sym, {}, {}, None, {}, {},  # C4: price=None (pas d'exposition prix corrompu)
            scan_error=(
                f"PRIX HORS BORNES ({cp_live:.2f} < {profile.price_min:.0f})"
                " — instrument OANDA mal configuré"
            ),
        )
    if profile.price_max is not None and cp_live > profile.price_max:
        return ScanResult(
            sym, {}, {}, None, {}, {},
            scan_error=(
                f"PRIX HORS BORNES ({cp_live:.2f} > {profile.price_max:.0f})"
                " — instrument OANDA mal configuré"
            ),
        )
    return None


def _resolve_working_price(
    cp_live: Optional[float],
    data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    sym: str,
) -> Tuple[Optional[float], bool]:
    """Résout prix utilisable : live OK ou fallback sur dernier close."""
    if cp_live and cp_live > 0 and np.isfinite(cp_live):
        return cp_live, False
    for tf_k in ("daily", "h4", "weekly"):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                if np.isfinite(last_close) and last_close > 0:
                    return last_close, True
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    return None, False


def _collect_tf_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    sym: str,
    data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    cp: float,
    profile: InstrumentProfile,
    min_touches_ui: int,
    sym_d: str,
) -> Tuple[Dict, Dict, Dict, Dict, str, List[str]]:
    """Collecte données par TF avec isolation par TF."""
    rows: Dict[str, Optional[list]] = {"H4": None, "Daily": None, "Weekly": None}
    zones_d: Dict[str, tuple] = {}
    trends: Dict[str, str] = {}
    bars_map: Dict[str, int] = {}
    price_ctx = ""
    missing_tfs: List[str] = []

    for tf_k, tf_name in (("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")):
        df = data_cube.get(sym, {}).get(tf_k)
        if df is None or df.empty:
            missing_tfs.append(tf_name)
            continue
        bars_map[tf_name] = len(df)
        try:
            lb, th = _TF_TREND_PARAMS.get(tf_name, (20, 2.0))
            trends[tf_name] = compute_institutional_trend(
                df["close"], lookback=lb, threshold=th,
            )
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logging.debug("Trend compute fail %s/%s: %s", sym, tf_name, type(e).__name__)
            trends[tf_name] = "NEUTRE"

        tf_rows, zone_pair, ctx = _process_tf_frame(
            sym, tf_k, tf_name, df, cp, min_touches_ui, profile, sym_d,
        )
        if zone_pair is not None:
            zones_d[tf_name] = zone_pair
        if tf_rows is not None:
            rows[tf_name] = tf_rows
        if ctx:
            price_ctx = ctx

    return rows, zones_d, trends, bars_map, price_ctx, missing_tfs


# ============================================================
# C1 : flag_data_anomaly décomposé (CC 24 → 3 sous-fonctions CC ≤ 6)
# ============================================================
def _anomaly_check_bounds(
    current_price: float, profile: InstrumentProfile,
) -> List[str]:
    """Vérifie bornes prix du profil."""
    msgs: List[str] = []
    if profile.price_min is not None and current_price < profile.price_min:
        msgs.append(
            f"PRIX HORS BORNES : {current_price:.2f} < min attendu {profile.price_min:.0f}"
            " — instrument OANDA mal mappé ou donnée corrompue"
        )
    if profile.price_max is not None and current_price > profile.price_max:
        msgs.append(
            f"PRIX HORS BORNES : {current_price:.2f} > max attendu {profile.price_max:.0f}"
            " — instrument OANDA mal mappé ou donnée corrompue"
        )
    return msgs


def _anomaly_check_support_ratio(
    current_price: float, support_levels: List[float], profile: InstrumentProfile,
) -> List[str]:
    """Détecte écart aberrant prix vs médiane des supports."""
    msgs: List[str] = []
    if profile.skip_ratio_check or len(support_levels) < 3:
        return msgs
    median_sup = float(np.median(support_levels))
    if median_sup > 0 and median_sup > 0.01 * current_price:
        ratio = current_price / median_sup
        if ratio > 3.0:
            msgs.append(
                f"Prix {current_price:.2f} = {ratio:.1f}x la mediane des supports "
                f"({median_sup:.2f}) — donnees a verifier"
            )
    return msgs


def _anomaly_check_live_vs_close(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    current_price: float,
    profile: InstrumentProfile,
    last_candle_close: Optional[float],
    last_candle_high: Optional[float],
    last_candle_low: Optional[float],
    last_candle_ts: Optional[Any],
    timeframe: str,
) -> List[str]:
    """Détecte écart anormal prix live vs dernier close, avec tolérance frontière."""
    msgs: List[str] = []
    if not (last_candle_close and last_candle_close > 0 and np.isfinite(last_candle_close)):
        return msgs

    in_range = (
        last_candle_high is not None and last_candle_low is not None
        and last_candle_low * 0.999 <= current_price <= last_candle_high * 1.001
    )
    if in_range:
        return msgs

    dev = abs(current_price - last_candle_close) / last_candle_close * 100
    threshold_pct = profile.max_live_vs_close_pct
    try:
        if last_candle_ts is not None:
            ts = pd.to_datetime(last_candle_ts, utc=True)
            age_hours = (
                datetime.now(timezone.utc) - ts.to_pydatetime()
            ).total_seconds() / 3600.0
            period_hours = _TF_PERIOD_HOURS.get(timeframe, 24.0)
            if age_hours < 1.5 * period_hours:
                threshold_pct = profile.max_live_vs_close_pct * 1.5
    except (ValueError, TypeError, AttributeError) as e:
        logging.debug("Anomaly TS parsing fail: %s", type(e).__name__)

    if dev > threshold_pct:
        msgs.append(
            f"Prix live {current_price:.5f} s'ecarte de {dev:.1f}% "
            f"du dernier close ({last_candle_close:.5f}) — seuil {threshold_pct:.1f}%"
        )
    return msgs


def flag_data_anomaly(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol: str,
    current_price: Optional[float],
    support_levels: List[float],
    last_candle_close: Optional[float] = None,
    last_candle_high: Optional[float] = None,
    last_candle_low: Optional[float] = None,
    last_candle_ts: Optional[Any] = None,
    timeframe: str = "daily",
) -> Optional[str]:
    """Détection d'anomalies prix avec tolérance frontière de bougie.

    Composition : bounds + support-ratio + live-vs-close.
    Retourne None si tout est cohérent, sinon string descriptif.
    """
    if current_price is None or current_price <= 0 or not np.isfinite(current_price):
        return "Prix indisponible ou non valide"

    profile = get_profile(symbol)
    messages: List[str] = []
    messages += _anomaly_check_bounds(current_price, profile)
    messages += _anomaly_check_support_ratio(current_price, support_levels, profile)
    messages += _anomaly_check_live_vs_close(
        current_price, profile,
        last_candle_close, last_candle_high, last_candle_low,
        last_candle_ts, timeframe,
    )

    return " | ".join(messages) if messages else None


# ============================================================
# C2 : _process_symbol décomposé
# ============================================================
def _extract_last_candle_info(
    daily_df: Optional[pd.DataFrame],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[Any]]:
    """Extrait (close, high, low, ts) de la dernière bougie daily si disponible."""
    if daily_df is None or daily_df.empty:
        return None, None, None, None
    try:
        return (
            float(daily_df["close"].iloc[-1]),
            float(daily_df["high"].iloc[-1]),
            float(daily_df["low"].iloc[-1]),
            daily_df.index[-1],
        )
    except (KeyError, IndexError, ValueError, TypeError) as e:
        logging.debug("Last candle extraction fail: %s", type(e).__name__)
        return None, None, None, None


def _collect_support_levels(zones_d: Dict[str, tuple]) -> List[float]:
    """Aplatit tous les niveaux de support de tous les TFs."""
    sup_levels: List[float] = []
    for zone_pair in zones_d.values():
        try:
            _s, _r = zone_pair
            if _s is not None and not _s.empty and "level" in _s.columns:
                sup_levels.extend(_s["level"].tolist())
        except (ValueError, TypeError, AttributeError) as e:
            logging.debug("Support collection fail: %s", type(e).__name__)
            continue
    return sup_levels


def _process_symbol(  # pylint: disable=too-many-locals
    sym: str,
    cp_live: Optional[float],
    data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    min_touches_ui: int,
) -> ScanResult:
    """Orchestre l'analyse complète d'un symbole avec isolation totale des erreurs.

    Filet ultime : aucun symbole ne peut faire crasher le scan global.
    """
    try:
        profile = get_profile(sym)

        if cp_live is not None and cp_live > 0 and np.isfinite(cp_live):
            err = _validate_price_bounds(sym, cp_live, profile)
            if err is not None:
                return err

        sym_d = sym.replace("_", "/")

        cp, price_is_fallback = _resolve_working_price(cp_live, data_cube, sym)
        if cp is None:
            return ScanResult(
                sym, {}, {}, None, {}, {},
                scan_error="Aucune donnée disponible (prix + bougies)",
            )

        rows, zones_d, trends, bars_map, price_ctx, missing_tfs = _collect_tf_data(
            sym, data_cube, cp, profile, min_touches_ui, sym_d,
        )

        sup_levels = _collect_support_levels(zones_d)
        daily_df = data_cube.get(sym, {}).get("daily")
        last_close, last_high, last_low, last_ts = _extract_last_candle_info(daily_df)

        anomaly = flag_data_anomaly(
            sym, cp, sup_levels,
            last_candle_close=last_close,
            last_candle_high=last_high,
            last_candle_low=last_low,
            last_candle_ts=last_ts,
            timeframe="daily",
        )

        if price_is_fallback:
            pf_msg = f"Prix live indisponible — utilisation du dernier close ({cp:.5f})"
            anomaly = f"{anomaly} | {pf_msg}" if anomaly else pf_msg

        return ScanResult(
            sym, rows, zones_d, cp, trends, bars_map,
            price_context=price_ctx, anomaly=anomaly,
            missing_tfs=missing_tfs, price_is_fallback=price_is_fallback,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Unexpected error processing %s", sym)
        return ScanResult(
            sym, {}, {}, None, {}, {},
            scan_error=f"Erreur interne : {type(e).__name__}",
        )


async def run_institutional_scan(
    symbols: List[str],
    token: str,
    oanda_account_id: str,
    min_touches_ui: int,
) -> List[ScanResult]:
    """Pipeline scan complet : auth → prices → candles → analyse parallèle."""
    client = AsyncOandaClient(token, oanda_account_id)
    async with aiohttp.ClientSession() as session:
        if not await client.initialize(session):
            raise OandaAuthError(
                "Impossible de s'authentifier sur OANDA. Vérifiez vos secrets API."
            )

        sem = asyncio.Semaphore(_OANDA_SEMAPHORE_LIMIT)
        live_prices = await _fetch_live_prices(client, session, sem, symbols)
        data_cube = await _fetch_candles_cube(client, session, sem, symbols)

    return [
        _process_symbol(sym, live_prices.get(sym), data_cube, min_touches_ui)
        for sym in symbols
    ]


# ==============================================================================
# [ LAYER 5: EXPORTERS & UTILS ]
# ==============================================================================
# pylint: disable=line-too-long
_ACCENT_MAP: Final = str.maketrans('àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ', 'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN')
# pylint: enable=line-too-long
_EMOJI_MAP: Final[List[Tuple[str, str]]] = [
    ('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'), ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'),
    ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]'), ('📈', ''), ('📉', ''), ('✅', '[OK]'),
    ('❌', '[X]'), ('⚡', '[!]'), ('📡', ''), ('📅', ''), ('↩️', '[RR]'),
    ('↑', '[HAUSSE]'), ('↓', '[BAISSE]'), ('→', '[NEUTRE]'),
]


def _safe_pdf_str(text: Any) -> str:
    """Sanitise pour FPDF latin-1 : accents + emoji + None safe (C14)."""
    if text is None:
        return ""
    try:
        s = str(text).translate(_ACCENT_MAP)
    except (TypeError, ValueError, AttributeError):
        return ""
    for e, r in _EMOJI_MAP:
        s = s.replace(e, r)
    return s


def _sanitize_traceback(traceback_str: str, sensitive_values: List[Optional[str]]) -> str:
    """Sanitisation triple : valeurs explicites + regex token + redact_sensitive."""
    if not traceback_str:
        return traceback_str
    out = traceback_str
    for val in sensitive_values:
        if not val or not isinstance(val, str) or len(val) < 4:
            continue
        out = out.replace(val, "***REDACTED***")
    out = _redact_sensitive(out)
    return out


_INTERNAL_COLS: Final[List[str]] = ["_dist_num", "_in_pdf"]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=_INTERNAL_COLS, errors="ignore")


def _sym_display(sym: str) -> str:
    return sym.replace("_", "/")


def get_price_context(  # pylint: disable=too-many-locals
    current_price: Optional[float],
    supports: Optional[pd.DataFrame],
    resistances: Optional[pd.DataFrame],
    max_dist_pct: float = 5.0,
) -> str:
    """Contexte prix vs S/R les plus proches (filtre max_dist_pct)."""
    if not current_price or current_price <= 0:
        return "Prix indisponible"
    parts: List[str] = []
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


def _parse_price_context_obstacles(
    ctx_str: str, current_price: float,
) -> Dict[str, Optional[dict]]:
    """Parse string contexte → dict obstacles structuré pour JSON export."""
    result: Dict[str, Optional[dict]] = {
        "nearest_support": None, "nearest_resistance": None,
    }
    if not ctx_str or ctx_str == "Zone intermediaire" or not current_price:
        return result
    pat = re.compile(
        r"(SUR support|S proche|SUR resistance|R proche):\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
    )
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


def strip_emojis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip emojis/accents pour PDF latin-1."""
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        clean[col] = clean[col].astype(str).apply(_safe_pdf_str)
    return clean


class PDF(FPDF):
    """Rapport PDF formaté avec sections : anomalies, summary, confluences, TFs."""

    def header(self) -> None:
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10,  # pylint: disable=unexpected-keyword-arg
                  _safe_pdf_str('Rapport Scanner Bluestar - Supports & Resistances'),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6,  # pylint: disable=unexpected-keyword-arg
                  _safe_pdf_str(
                      f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  |  "
                      f"v{SCANNER_VERSION}  |  "
                      "Score = (Force x Poids_TF x NbTF) x Facteur_Age | "
                      "Statut Vierge / Testee / Role Reverse / Consommee"
                  ),
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def chapter_title(self, title: str) -> None:
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L',  # pylint: disable=unexpected-keyword-arg
                  new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def chapter_anomalies(self, anomalies: dict) -> None:
        if not anomalies:
            return
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, _safe_pdf_str('ALERTES ANOMALIES PRIX'),  # pylint: disable=unexpected-keyword-arg
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        self.set_font('Helvetica', '', 8)
        for sym, msg in anomalies.items():
            line = _safe_pdf_str(f"[!] {sym} : {msg}")
            self.multi_cell(0, 5, line[:180])
        self.ln(4)

    def chapter_summary(self, summary_list: List[Dict[str, Any]]) -> None:  # pylint: disable=too-many-locals
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7,  # pylint: disable=unexpected-keyword-arg
                  _safe_pdf_str('RESUME PAR ACTIF  (Tendances + Top Zones Confluentes)'),
                  border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(2)
        for s in summary_list:
            sym = _safe_pdf_str(s.get('symbol', ''))
            t_h4 = _safe_pdf_str(s.get('trend_h4', 'N/A'))
            t_d = _safe_pdf_str(s.get('trend_daily', 'N/A'))
            t_w = _safe_pdf_str(s.get('trend_weekly', 'N/A'))
            ctx = _safe_pdf_str(s.get('price_context', ''))
            self.set_font('Helvetica', 'B', 8)
            self.cell(0, 5,  # pylint: disable=unexpected-keyword-arg
                      _safe_pdf_str(f"{sym}   H4:{t_h4}  Daily:{t_d}  Weekly:{t_w}"),
                      border=0, new_x='LMARGIN', new_y='NEXT')
            if ctx:
                self.set_font('Helvetica', 'I', 7)
                self.cell(0, 4, f"  Position : {ctx[:120]}",  # pylint: disable=unexpected-keyword-arg
                          border=0, new_x='LMARGIN', new_y='NEXT')
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
                    self.cell(0, 4, txt[:130],  # pylint: disable=unexpected-keyword-arg
                              border=0, new_x='LMARGIN', new_y='NEXT')
            else:
                self.cell(0, 4, "  Aucune confluence pour cet actif.",  # pylint: disable=unexpected-keyword-arg
                          border=0, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)

    def chapter_body(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
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
            self.cell(col_widths[col_name], 6, _safe_pdf_str(col_name),  # pylint: disable=unexpected-keyword-arg
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
                self.cell(w, 5, val, border=1, align='C',  # pylint: disable=unexpected-keyword-arg
                          new_x='RIGHT', new_y='TOP')
            self.ln()


def _apply_pdf_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre lignes selon pdf_max_dist_pct du profil. C9: fillna explicite."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    if "_in_pdf" in df.columns:
        return _clean_df(df[df["_in_pdf"]].copy()).reset_index(drop=True)
    if "Actif" in df.columns and "_dist_num" in df.columns:
        unique_actifs = df["Actif"].unique()
        thresh_map = {a: get_profile(a.replace("/", "_")).pdf_max_dist_pct for a in unique_actifs}
        thresholds = df["Actif"].map(thresh_map).fillna(8.0)  # C9
        dist_num = pd.to_numeric(df["_dist_num"], errors="coerce").fillna(999.0)
        mask = dist_num <= thresholds
        return _clean_df(df[mask].copy()).reset_index(drop=True)
    if "Dist. %" in df.columns:
        def _to_f(s: Any) -> float:
            try:
                return float(str(s).replace("%", ""))
            except (ValueError, TypeError):
                return 999.0
        df = df[df["Dist. %"].apply(_to_f) <= 8.0].copy()
    return _clean_df(df).reset_index(drop=True)


# ============================================================
# EXPORTS — hash sémantique correct
# ============================================================
def _hash_results_dict(d: Dict[str, pd.DataFrame]) -> str:
    """Hash sémantique réel d'un dict de DataFrames."""
    if not d:
        return "empty_dict"
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        if v is None or (hasattr(v, 'empty') and v.empty):
            parts.append(f"{k}:empty")
        else:
            parts.append(f"{k}:{_hash_df(v)}")
    return hashlib.md5("|".join(parts).encode(), usedforsecurity=False).hexdigest()


def _hash_list_of_dicts(lst: List[Dict[str, Any]]) -> str:
    """Hash stable d'une liste de dicts (ignore les DataFrames imbriqués)."""
    if not lst:
        return "empty_list"
    try:
        return hashlib.md5(
            json.dumps(
                [
                    {k: str(v)[:80] for k, v in d.items()
                     if not isinstance(v, (pd.DataFrame, pd.Series))}
                    for d in lst
                ],
                sort_keys=True, default=str,
            ).encode(),
            usedforsecurity=False,
        ).hexdigest()
    except (TypeError, ValueError):
        return f"unhashable_list_{len(lst)}"


def _hash_anomalies(d: Dict[str, str]) -> str:
    if not d:
        return "no_anom"
    try:
        return hashlib.md5(
            json.dumps(d, sort_keys=True).encode(),
            usedforsecurity=False,
        ).hexdigest()
    except (TypeError, ValueError):
        return f"unhashable_anom_{len(d)}"


@st.cache_data(ttl=300, max_entries=8, show_spinner=False,
               hash_funcs={
                   dict: _hash_results_dict,
                   pd.DataFrame: _hash_df,
                   list: _hash_list_of_dicts,
               })
def create_pdf_report(  # pylint: disable=too-many-locals
    results_dict: Dict[str, pd.DataFrame],
    confluences_df: Optional[pd.DataFrame] = None,
    summary_list: Optional[List[Dict[str, Any]]] = None,
    anomalies: Optional[Dict[str, str]] = None,
) -> bytes:
    """Génère le rapport PDF complet. Cache 300s."""
    summary_list = summary_list or []
    anomalies = anomalies or {}
    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    if anomalies:
        pdf.chapter_anomalies(anomalies)
    if summary_list:
        pdf.chapter_summary(summary_list)
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
        clean_d = strip_emojis_df(_clean_df(df.copy()))
        if "Score (1TF)" in clean_d.columns:
            clean_d = clean_d.sort_values("Score (1TF)", ascending=False)
        pdf.chapter_body(clean_d)
        pdf.ln(10)
    return bytes(pdf.output())


@st.cache_data(ttl=300, max_entries=8, show_spinner=False,
               hash_funcs={dict: _hash_results_dict, pd.DataFrame: _hash_df})
def create_csv_report(
    results_dict: Dict[str, pd.DataFrame],
    confluences_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """Export CSV consolidé (confluences + rows par TF)."""
    all_dfs: List[pd.DataFrame] = []
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


_TREND_ARROW: Final[Dict[str, str]] = {"HAUSSIER": "↑", "BAISSIER": "↓", "NEUTRE": "→"}
_STATUS_LABEL: Final[Dict[str, str]] = {
    "Vierge": "V", "Testee": "T", "Role Reverse": "RR", "Consommee": "C",
}
_ALERT_LABEL: Final[Dict[str, str]] = {"🔥 ZONE CHAUDE": "⚡", "⚠️ Proche": "⚠"}
_SIGNAL_SHORT: Final[Dict[str, str]] = {
    "PIVOT": "PIVOT", "BUY": "BUY  ", "SELL": "SELL ",
}


def _filter_confluences_to_actif_zones(
    confluences_df: pd.DataFrame,
    max_dist: float,
    min_score: float,
    allowed_statuts: tuple,
) -> Dict[str, list]:
    """Filtre + groupe confluences par actif pour brief LLM."""
    actif_zones: Dict[str, list] = {}
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
        if dist_val > max_dist or score_val < min_score or statut not in allowed_statuts:
            continue
        actif_zones.setdefault(str(row.get("Actif", "")), []).append({
            "signal": str(row.get("Signal", "")),
            "niveau": str(row.get("Niveau", "")),
            "score": score_val,
            "statut": statut,
            "dist": dist_val,
            "tfs": str(row.get("Timeframes", "")),
            "nb_tf": int(row.get("Nb TF", 0)),
            "alerte": str(row.get("Alerte", "")),
        })
    return actif_zones


def _format_brief_zone_line(z: dict) -> str:
    """Formate une ligne de zone pour brief LLM markdown."""
    sig = z["signal"]
    signal_short = next(
        (v for k, v in _SIGNAL_SHORT.items() if k in sig),
        "ZONE ",
    )
    tf_short = z["tfs"].replace("Daily", "D").replace("Weekly", "W").replace(" + ", "+")
    return (
        f"- {signal_short} `{z['niveau']}` | "
        f"Sc:{z['score']:.0f} | {_STATUS_LABEL.get(z['statut'], z['statut'])} | "
        f"{z['dist']:.2f}% | {tf_short} {_ALERT_LABEL.get(z['alerte'], '')}"
    )


@st.cache_data(ttl=300, max_entries=16, show_spinner=False,
               hash_funcs={pd.DataFrame: _hash_df, list: _hash_list_of_dicts})
def create_llm_brief(
    summary_list: List[Dict[str, Any]],
    confluences_df: Optional[pd.DataFrame],
    max_dist: float = 2.0,
    min_score: float = 100.0,
    allowed_statuts: Tuple[str, ...] = ("Vierge", "Testee", "Role Reverse"),
) -> bytes:
    """Génère brief markdown optimisé pour LLM."""
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
        ("- `Sc` : Score pondéré (Force × Poids_TF × NbTF × Facteur_Age)."
         " >300=institutionnel, 100-300=fort"),
        ("- `V` = Vierge | `T` = Testée | `RR` = Role Reverse (zone cassée retestée)"
         " | `C` = Consommée (éviter)"),
        "- `Dist%` : distance du prix actuel à la zone",
        "- `TFs` : timeframes en confluence (H4/D/W)",
        "- `⚡` = zone chaude (<0.5% du prix) | `⚠` = proche (<1.5%)",
        "",
        (f"**Filtres actifs** : Dist < {max_dist}% | Score ≥ {min_score} | "
         f"Statuts : {', '.join(allowed_statuts)}"),
        "",
        "---",
        "",
    ]
    if confluences_df is None or confluences_df.empty:
        lines.append("_Aucune confluence disponible._")
        return "\n".join(lines).encode("utf-8")

    actif_zones = _filter_confluences_to_actif_zones(
        confluences_df, max_dist, min_score, allowed_statuts,
    )
    sorted_actifs = sorted(
        actif_zones,
        key=lambda a: max(z["score"] for z in actif_zones[a]),
        reverse=True,
    )
    summary_map = {s["symbol"]: s for s in summary_list}
    total_zones = 0

    for actif in sorted_actifs:
        s = summary_map.get(actif, {})
        t_h4 = _TREND_ARROW.get(s.get("trend_h4", "NEUTRE"), "→")
        t_d = _TREND_ARROW.get(s.get("trend_daily", "NEUTRE"), "→")
        t_w = _TREND_ARROW.get(s.get("trend_weekly", "NEUTRE"), "→")
        lines.append(f"### {actif} | H4:{t_h4} D:{t_d} W:{t_w}")
        if ctx := s.get("price_context", ""):
            lines.append(f"> {ctx}")
        for z in sorted(actif_zones[actif], key=lambda z: z["score"], reverse=True):
            lines.append(_format_brief_zone_line(z))
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


def _get_ict_session(dt_utc: datetime) -> str:
    """Identifie la session ICT (Asian/London/Overlap/NewYork)."""
    if _NY_TZ is not None:
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        ny = dt_utc.astimezone(_NY_TZ)
        h = ny.hour
    else:
        h = (dt_utc.hour - 5) % 24
    if 18 <= h or h < 3:
        return "ASIAN"
    if 3 <= h < 8:
        return "LONDON"
    if 8 <= h < 12:
        return "OVERLAP_LDN_NY"
    return "NEW_YORK"


def _normalize_signal(raw: str) -> str:
    """Normalise signal vers BUY/SELL/PIVOT/<other>."""
    r = raw.replace("🟢", "").replace("🔴", "").replace("↔️", "").replace("↔", "")
    r = r.replace("ZONE", "").strip()
    if "PIVOT" in r:
        return "PIVOT"
    if "BUY" in r:
        return "BUY"
    if "SELL" in r:
        return "SELL"
    return r.strip()


def _normalize_alert(raw: str) -> str:
    """Normalise alerte vers HOT/CLOSE/''."""
    r = raw.replace("🔥", "").replace("⚠️", "").replace("⚠", "").strip()
    if "CHAUD" in r.upper() or "HOT" in r.upper():
        return "HOT"
    if "PROCHE" in r.upper() or "CLOSE" in r.upper():
        return "CLOSE"
    return ""


_TF_ORDER: Final[Dict[str, int]] = {"Weekly": 0, "Daily": 1, "H4": 2}


def _parse_timeframes(tf_str: str) -> List[str]:
    parts = [p.strip() for p in tf_str.replace("+", ",").split(",") if p.strip()]
    return sorted(parts, key=lambda t: _TF_ORDER.get(t, 99))


_BIAS_MAP: Final[Dict[str, str]] = {
    "HAUSSIER": "BULLISH", "BAISSIER": "BEARISH", "NEUTRE": "NEUTRAL",
}


def _trend_alignment(h4: str, daily: str, weekly: str) -> Tuple[str, str]:
    """Calcule alignment + bias dominant. Préserve sémantique métier existante."""
    b_h4 = _BIAS_MAP.get(h4, "NEUTRAL")
    b_d = _BIAS_MAP.get(daily, "NEUTRAL")
    b_w = _BIAS_MAP.get(weekly, "NEUTRAL")
    if b_d == b_w and b_d != "NEUTRAL":
        dominant = b_d
    elif b_d == "NEUTRAL":
        dominant = b_w
    else:
        dominant = b_d if b_w == "NEUTRAL" else "NEUTRAL"

    if dominant != "NEUTRAL":
        alignment = (
            "ALIGNED" if b_h4 == dominant
            else ("PULLBACK" if b_h4 == "NEUTRAL" else "CONFLICTED")
        )
    else:
        alignment = "BUILDING" if b_h4 != "NEUTRAL" else "MIXED"
    return alignment, dominant


def _build_actif_groups(  # pylint: disable=too-many-locals
    confluences_df: pd.DataFrame,
    max_dist: float,
    min_score: float,
    allowed_statuts: tuple,
) -> Dict[str, List[Dict[str, Any]]]:
    """Filtre + groupe confluences par actif pour export JSON."""
    actif_groups: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in confluences_df.iterrows():
        try:
            dist_val = float(str(row.get("Distance %", 999)).replace("%", ""))
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
        signal_norm = _normalize_signal(str(row.get("Signal", "")))
        if signal_norm not in ("BUY", "SELL", "PIVOT"):
            continue
        statut = str(row.get("Statut", ""))
        if statut not in allowed_statuts:
            continue
        try:
            level_float = round(float(row.get("Niveau", "")), 5)
        except (TypeError, ValueError):
            continue
        tf_str = str(row.get("Timeframes", ""))
        tfs_parsed = _parse_timeframes(tf_str)
        actif_groups.setdefault(str(row.get("Actif", "")), []).append({
            "signal": signal_norm,
            "type": str(row.get("Type", "")),
            "level": level_float,
            "score": round(score_val, 1),
            "status": statut,
            "distance_pct": round(dist_val, 3),
            "alert": _normalize_alert(str(row.get("Alerte", ""))),
            "timeframes": tfs_parsed,
            "nb_tf": int(row.get("Nb TF", len(tfs_parsed))),
        })
    return actif_groups


@st.cache_data(ttl=300, max_entries=16, show_spinner=False,
               hash_funcs={pd.DataFrame: _hash_df, list: _hash_list_of_dicts})
def create_json_export(  # pylint: disable=too-many-locals
    summary_list: List[Dict[str, Any]],
    confluences_df: Optional[pd.DataFrame],
    max_dist: float = 5.0,
    min_score: float = 60.0,
    allowed_statuts: Tuple[str, ...] = ("Vierge", "Testee", "Role Reverse"),
) -> bytes:
    """Export JSON structuré (schéma stable, compatibilité préservée)."""
    now_utc = datetime.now(timezone.utc)
    output: Dict[str, Any] = {
        "generated_at": now_utc.isoformat(),
        "scanner_version": SCANNER_VERSION,
        "session": _get_ict_session(now_utc),
        "filters": {
            "max_dist_pct": max_dist, "min_score": min_score,
            "allowed_statuts": list(allowed_statuts),
        },
        "assets": [],
    }
    summary_map = {s["symbol"]: s for s in summary_list}
    actif_groups = (
        _build_actif_groups(confluences_df, max_dist, min_score, allowed_statuts)
        if (confluences_df is not None and not confluences_df.empty)
        else {}
    )

    all_actifs = set(summary_map.keys()) | set(actif_groups.keys())
    sorted_actifs = sorted(
        all_actifs,
        key=lambda a: max((z["score"] for z in actif_groups.get(a, [])), default=0.0),
        reverse=True,
    )

    for actif in sorted_actifs:
        s = summary_map.get(actif, {})
        t_h4 = s.get("trend_h4", "NEUTRE")
        t_d = s.get("trend_daily", "NEUTRE")
        t_w = s.get("trend_weekly", "NEUTRE")
        alignment, dominant = _trend_alignment(t_h4, t_d, t_w)
        cp_val = s.get("current_price")
        ctx_str = s.get("price_context", "")
        obstacles = _parse_price_context_obstacles(ctx_str, cp_val or 0)
        zones = sorted(actif_groups.get(actif, []), key=lambda z: z["score"], reverse=True)
        output["assets"].append({
            "symbol": actif,
            "current_price": round(cp_val, 5) if cp_val else None,
            "price_is_fallback": bool(s.get("price_is_fallback", False)),
            "missing_timeframes": list(s.get("missing_tfs", [])),
            "trends": {"h4": t_h4, "daily": t_d, "weekly": t_w},
            "trend_alignment": alignment,
            "dominant_bias": dominant,
            "price_context": ctx_str,
            "nearest_support": obstacles["nearest_support"],
            "nearest_resistance": obstacles["nearest_resistance"],
            "nb_zones": len(zones),
            "zones": zones,
        })
    return json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")


def _filter_and_sort(df: Optional[pd.DataFrame], max_pct: float) -> pd.DataFrame:
    """Filtre lignes par Dist. % et trie par score (C15: robuste à None)."""
    if df is None or df.empty or "Dist. %" not in df.columns:
        return df if df is not None else pd.DataFrame()

    def _to_float(s: Any) -> float:
        try:
            return float(str(s).replace("%", ""))
        except (ValueError, TypeError):
            return 999.0

    out = _clean_df(df[df["Dist. %"].apply(_to_float) <= max_pct])
    sort_col = next((c for c in ("Score (1TF)", "Score") if c in out.columns), None)
    if sort_col:
        out = out.sort_values(sort_col, ascending=False)
    return out.reset_index(drop=True)


def _show_diagnostics(errors: dict, anomalies: dict, missing_tfs_map: dict) -> None:
    """Sections d'expanders pour diagnostics scan."""
    if errors:
        with st.expander(f"❌ {len(errors)} actif(s) en erreur — cliquer pour voir"):
            for sym, err in errors.items():
                st.error(f"**{sym}** : {err}")
    if anomalies:
        with st.expander(f"⚠️ {len(anomalies)} anomalie(s) de prix — cliquer pour voir"):
            for sym, msg in anomalies.items():
                st.warning(f"**{sym}** : {msg}")
    if missing_tfs_map:
        with st.expander(f"📡 {len(missing_tfs_map)} actif(s) avec TFs manquants — cliquer pour voir"):
            for sym, tfs in missing_tfs_map.items():
                st.info(f"**{sym}** : TFs absents → {', '.join(tfs)}")


def _show_confluence_section(conf_filt: pd.DataFrame) -> None:
    """Section principale d'affichage confluences."""
    if conf_filt.empty:
        st.info("Aucune confluence dans la plage sélectionnée. Augmentez le filtre ou le seuil.")
        return
    st.divider()
    st.subheader("🔥 ZONES DE CONFLUENCE MULTI-TIMEFRAMES")
    st.caption(
        "Score confluence = Force × Nb TF × Poids_TF × Facteur_Age  |  "
        "Score (1TF) dans les tableaux ci-dessous = mono-timeframe brut"
    )
    disp = conf_filt.sort_values("Score", ascending=False).reset_index(drop=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total zones", len(disp))
    c2.metric("🔥 Zones chaudes", len(disp[disp["Alerte"] == "🔥 ZONE CHAUDE"]))
    c3.metric("⚠️ Zones proches", len(disp[disp["Alerte"] == "⚠️ Proche"]))
    c4.metric("🟢 BUY Zones", len(disp[disp["Signal"] == "🟢 BUY ZONE"]))
    c5.metric("🔴 SELL Zones", len(disp[disp["Signal"] == "🔴 SELL ZONE"]))
    c6.metric("↔ PIVOT Zones", len(disp[disp["Signal"] == "↔ PIVOT ZONE"]))
    text_cols = ["Actif", "Signal", "Niveau", "Type", "Timeframes",
                 "Statut", "Distance %", "Alerte"]
    conf_cfg = {
        **{k: st.column_config.TextColumn(k, width="small") for k in text_cols},
        "Nb TF": st.column_config.NumberColumn("Nb TF", width="small"),
        "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
        "Score": st.column_config.NumberColumn("Score ▼", width="small"),
    }
    st.dataframe(
        disp, column_config=conf_cfg, hide_index=True,
        width='stretch', height=min(len(disp) * 35 + 38, 750),
    )


def _show_export_section(  # pylint: disable=too-many-locals
    rep_dict: dict, conf_full: pd.DataFrame,
    summary_list: list, anomalies: dict,
) -> None:
    """Section exports PDF/CSV/LLM/JSON avec gestion d'erreur granulaire."""
    st.subheader("📋 Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            try:
                pdf_bytes = create_pdf_report(rep_dict, conf_full, summary_list, anomalies)
                st.download_button(
                    "📄 Rapport PDF (complet)", data=pdf_bytes,
                    file_name=f"rapport_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", width='stretch',
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception("PDF generation failed")
                st.error(f"Génération PDF impossible : {type(e).__name__}")
        with col2:
            try:
                csv_bytes = create_csv_report(rep_dict, conf_full)
                st.download_button(
                    "📊 Données brutes CSV", data=csv_bytes,
                    file_name=f"donnees_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", width='stretch',
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception("CSV generation failed")
                st.error(f"Génération CSV impossible : {type(e).__name__}")

        st.divider()
        st.markdown("**🤖 Exports optimisés LLM**")
        st.caption("Paramètres LLM configurables dans la barre latérale (section 3).")
        llm_max_dist = float(st.session_state.get("llm_max_dist", 2.0))
        llm_min_score = float(st.session_state.get("llm_min_score", 100))
        llm_statuts_raw = st.session_state.get("llm_statuts", ["Vierge", "Testee", "Role Reverse"])
        llm_statuts = (
            tuple(llm_statuts_raw) if llm_statuts_raw
            else ("Vierge", "Testee", "Role Reverse")
        )
        st.caption(
            f"🔧 Filtres actifs : Dist < **{llm_max_dist}%** | "
            f"Score ≥ **{llm_min_score}** | {', '.join(llm_statuts)}"
        )
        md_bytes = b""
        col3, col4 = st.columns(2)
        with col3:
            try:
                md_bytes = create_llm_brief(
                    summary_list, conf_full,
                    max_dist=llm_max_dist, min_score=llm_min_score,
                    allowed_statuts=llm_statuts,
                )
                st.download_button(
                    "🤖 Brief LLM (Markdown filtré)", data=md_bytes,
                    file_name=f"brief_llm_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown", width='stretch',
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception("LLM brief generation failed")
                st.error(f"Génération brief LLM impossible : {type(e).__name__}")
        with col4:
            try:
                json_bytes = create_json_export(
                    summary_list, conf_full,
                    max_dist=llm_max_dist, min_score=llm_min_score,
                    allowed_statuts=llm_statuts,
                )
                st.download_button(
                    "🔧 Export JSON structuré", data=json_bytes,
                    file_name=f"sr_bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json", width='stretch',
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.exception("JSON export failed")
                st.error(f"Génération JSON impossible : {type(e).__name__}")

        st.divider()
        st.markdown("**👁️ Aperçu du Brief LLM**")
        st.caption(
            f"Filtres : Dist < {llm_max_dist}% | Score ≥ {llm_min_score} | "
            f"{', '.join(llm_statuts)}"
        )
        try:
            brief_preview = md_bytes.decode("utf-8") if md_bytes else ""
            if brief_preview:
                n_zones = sum(
                    1 for line in brief_preview.split("\n")
                    if line.strip().startswith(("- BUY", "- SELL", "- PIVOT"))
                )
                n_actifs = brief_preview.count("### ")
                st.info(
                    f"**{n_actifs} actifs** avec **{n_zones} zones** dans le brief LLM "
                    f"(≈ {n_zones * 15 + n_actifs * 10:,} tokens estimés)"
                )
                st.text_area(
                    "Brief LLM (copiable directement)", value=brief_preview,
                    height=400, label_visibility="collapsed",
                )
            else:
                st.warning("Aperçu non disponible.")
        except (UnicodeDecodeError, AttributeError, TypeError, MemoryError):
            st.warning("Aperçu non disponible.")


_TF_COL_CONFIG = {
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


def _display_results(sr: dict, max_dist_filter_pct: float) -> None:
    """Affichage final résultats avec filtres."""
    _df_h4 = sr.get("df_h4", pd.DataFrame())
    _df_daily = sr.get("df_daily", pd.DataFrame())
    _df_wk = sr.get("df_weekly", pd.DataFrame())
    _conf_full = sr.get("conf_full", pd.DataFrame())
    _rep_dict = sr.get("report_dict", {})
    _summaries = sr.get("summaries", [])
    _anomalies = sr.get("anomalies", {})
    errors = sr.get("scan_errors", {})
    missing_tfs_map = sr.get("missing_tfs_map", {})

    if not _conf_full.empty:
        tmp = _clean_df(_conf_full).copy()
        tmp["_dist_num"] = pd.to_numeric(
            tmp["Distance %"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        ).fillna(999.0)
        conf_filt = (
            tmp[tmp["_dist_num"] <= max_dist_filter_pct]
            .drop(columns=["_dist_num"], errors="ignore")
            .reset_index(drop=True)
        )
    else:
        conf_filt = pd.DataFrame()

    _show_diagnostics(errors, _anomalies, missing_tfs_map)
    _show_confluence_section(conf_filt)
    _show_export_section(_rep_dict, _conf_full, _summaries, _anomalies)

    st.divider()
    for label, df in [
        ("📅 Analyse 4 Heures (H4)", _df_h4),
        ("📅 Analyse Journalière (Daily)", _df_daily),
        ("📅 Analyse Hebdomadaire (Weekly)", _df_wk),
    ]:
        st.subheader(label)
        fd = _filter_and_sort(df, max_dist_filter_pct)
        st.dataframe(
            fd, column_config=_TF_COL_CONFIG, hide_index=True,
            width='stretch', height=min(len(fd) * 35 + 38, 600),
        )


# ==============================================================================
# CONSTANTES UI
# ==============================================================================
CONFLUENCE_THRESHOLD_MAP: Final[Dict[str, float]] = {
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
    st.caption("Ces paramètres survivent aux re-renders contrairement aux sliders dans l'expander.")
    st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    st.slider("Score min JSON/LLM", 20, 300, 30, 10, key="llm_min_score")
    st.multiselect(
        "Statuts autorisés (brief LLM)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    st.divider()
    st.header("4. Paramètres de Détection")
    min_touches = st.slider("Force minimale H4 (touches)", 2, 10, 2, 1)
    st.caption("H4 utilise ce seuil | Daily/Weekly utilisent max(2, valeur)")
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 0.8, 0.1)
    _overridden = [s.replace("_USD", "").replace("_EUR", "") for s in CONFLUENCE_THRESHOLD_MAP]
    st.caption(
        f"⚠️ Seuil ignoré pour : {', '.join(_overridden)} "
        f"(valeurs fixes : {list(CONFLUENCE_THRESHOLD_MAP.values())})"
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) - filtre visuel uniquement", 1.0, 15.0, 3.0, 0.5,
    )

    st.divider()
    if st.button("🧹 Vider le cache OANDA", help="Reset thread-safe du cache mémoire"):
        n_cleared = _cache_clear()
        st.success(f"Cache vidé : {n_cleared} entrée(s)")

    st.divider()
    st.caption("**Score confluence = (Force × Poids_TF × NbTF) × Facteur_Age**")
    st.caption("🔴 > 300 : Zone institutionnelle majeure")
    st.caption("🟠 100–300 : Zone structurelle forte")
    st.caption("🟡 30–100 : Zone technique valide")
    st.caption("⚪ < 30   : Zone secondaire (filtrée par défaut)")

    st.divider()
    st.caption("**Statuts :**")
    st.caption("✅ Vierge = jamais testée (la plus fiable)")
    st.caption("🔵 Testée = respectée, toujours valide")
    st.caption("↩️ Role Reverse = niveau cassé retesté")
    st.caption("❌ Consommée = cassée sans retour")

    st.divider()
    st.caption(f"**v{SCANNER_VERSION} — Production Hardened**")
    st.caption("✅ Hash DataFrame sémantique enrichi (shape+vol+sample)")
    st.caption("✅ Anomalies prix décomposées (CC 24 → 6)")
    st.caption("✅ Cache négatif TTL court dédié + reset thread-safe")
    st.caption("✅ Async isolé : worker thread + timeout fermeture")
    st.caption("✅ Sanitization tokens : regex + filtre logger global")
    st.caption("✅ Sanitization OHLC stricte (drop NaN, dédup, ratio)")
    st.caption("✅ Isolation totale par symbole (filet ultime)")
    st.caption("✅ Borne fraîcheur absolue + tolérance frontière")
    st.caption("✅ Anti-double-scan via state atomique")


# ==============================================================================
# POST-SCAN AGGREGATORS
# ==============================================================================
def _collect_scan_results(raw_results: list, progress_bar: Any) -> dict:
    """Aggrégation post-scan : éclate ScanResult en maps utilisables UI."""
    results_h4: list = []
    results_daily: list = []
    results_weekly: list = []
    all_zones_map: dict = {}
    prices_map: dict = {}
    trends_map: dict = {}
    anomalies_map: dict = {}
    scan_errors: dict = {}
    bars_map_global: dict = {}
    missing_tfs_map: Dict[str, List[str]] = {}
    price_fallback_map: Dict[str, bool] = {}

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
            if not tf_rows:
                continue
            if tf_cap == "H4":
                results_h4.extend(tf_rows)
            elif tf_cap == "Daily":
                results_daily.extend(tf_rows)
            elif tf_cap == "Weekly":
                results_weekly.extend(tf_rows)

    return {
        "results_h4": results_h4, "results_daily": results_daily,
        "results_weekly": results_weekly,
        "all_zones_map": all_zones_map, "prices_map": prices_map,
        "trends_map": trends_map,
        "anomalies_map": anomalies_map, "scan_errors": scan_errors,
        "bars_map_global": bars_map_global, "missing_tfs_map": missing_tfs_map,
        "price_fallback_map": price_fallback_map,
    }


def _compute_confluences(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    scan_symbols: list,
    all_zones_map: dict,
    prices_map: dict,
    bars_map_global: dict,
    default_threshold: float,
    scan_errors: dict,
) -> pd.DataFrame:
    """Calcule toutes les confluences multi-TF avec seuil par profil."""
    all_confluences = []
    for sym in scan_symbols:
        if _sym_display(sym) in scan_errors:
            continue
        try:
            zones_clean = {
                k: v for k, v in all_zones_map.get(sym, {}).items()
                if not k.startswith("_")
            }
            sym_threshold = CONFLUENCE_THRESHOLD_MAP.get(sym, default_threshold)
            confs = detect_confluences(
                _sym_display(sym), zones_clean, prices_map.get(sym),
                bars_map_global.get(sym, {}),
                confluence_threshold_pct=sym_threshold,
            )
            all_confluences.extend(confs)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logging.warning("Confluence computation failed for %s: %s", sym, type(e).__name__)
            continue

    conf_full = pd.DataFrame(all_confluences)
    return _clean_df(conf_full) if not conf_full.empty else conf_full


def _build_summaries(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    scan_symbols: list,
    prices_map: dict,
    trends_map: dict,
    all_zones_map: dict,
    conf_full: pd.DataFrame,
    missing_tfs_map: dict,
    price_fallback_map: dict,
) -> List[Dict[str, Any]]:
    """Construit résumés par actif pour PDF/LLM/JSON."""
    summary_list: List[Dict[str, Any]] = []
    for sym in scan_symbols:
        sym_d = _sym_display(sym)
        trends = trends_map.get(sym, {})
        cp = prices_map.get(sym)
        top_zones: list = []
        if (not conf_full.empty and "Actif" in conf_full.columns
                and sym_d in conf_full["Actif"].values):
            try:
                top_zones = (
                    conf_full[conf_full["Actif"] == sym_d]
                    .sort_values("Score", ascending=False)
                    .head(3)
                    .to_dict("records")
                )
            except (KeyError, ValueError, TypeError):
                top_zones = []
        price_ctx = ""
        if "Daily" in all_zones_map.get(sym, {}) and cp:
            try:
                sup_d, res_d = all_zones_map[sym]["Daily"]
                price_ctx = get_price_context(cp, sup_d, res_d)
            except (ValueError, TypeError, AttributeError):
                price_ctx = ""
        summary_list.append({
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
    return summary_list


# ==============================================================================
# LOGIQUE PRINCIPALE — C11 : guard atomique anti-double-scan
# ==============================================================================
if scan_button and symbols_to_scan and not st.session_state.get("scanning", False):
    # C11: lock atomique avant rerun pour empêcher double déclenchement
    st.session_state.pop("scan_results", None)
    st.session_state["scanning"] = True
    st.session_state["pending_scan"] = True
    st.session_state["scan_token"] = f"{time.time():.6f}"
    st.rerun()

if st.session_state.get("pending_scan", False) and symbols_to_scan:
    current_token = st.session_state.get("scan_token")
    if not current_token:
        st.session_state.pop("pending_scan", None)
        st.session_state["scanning"] = False
    else:
        st.session_state.pop("pending_scan", None)

        if not access_token or not account_id:
            st.session_state["scanning"] = False
            st.session_state.pop("scan_token", None)
            st.warning("Configurez vos secrets OANDA avant de lancer le scan.")
        else:
            progress_bar = st.progress(0, text="Initialisation du scan async…")

            try:
                with st.spinner("Pipeline async I/O en cours…"):
                    raw_results = _run_async_isolated(
                        lambda: run_institutional_scan(
                            symbols_to_scan, access_token, account_id, min_touches,
                        ),
                        timeout=600.0,
                    )
            except OandaAuthError as e:
                st.error(str(e))
                st.session_state["scanning"] = False
                st.session_state.pop("scan_token", None)
                st.stop()
            except concurrent.futures.TimeoutError:
                st.error("Scan timeout (> 10 min). Réessayez avec moins d'actifs.")
                st.session_state["scanning"] = False
                st.session_state.pop("scan_token", None)
                st.stop()
            except Exception as e:  # pylint: disable=broad-exception-caught
                sanitized_tb = _sanitize_traceback(
                    traceback.format_exc(), [access_token, account_id],
                )
                logging.exception("Scan failure")
                st.error(f"Erreur inattendue : {type(e).__name__} — {sanitized_tb[-400:]}")
                st.session_state["scanning"] = False
                st.session_state.pop("scan_token", None)
                st.stop()

            collected = _collect_scan_results(raw_results, progress_bar)
            progress_bar.empty()
            st.session_state["scanning"] = False
            st.session_state.pop("scan_token", None)

            n_ok = len(symbols_to_scan) - len(collected["scan_errors"])
            n_failures = len(collected["scan_errors"])
            if n_failures == 0:
                st.success(f"✅ Scan terminé — {n_ok} actifs analysés avec succès.")
            else:
                st.warning(f"⚠️ Scan terminé — {n_ok} actifs OK, {n_failures} en erreur.")
            if collected["anomalies_map"]:
                st.warning(
                    f"⚠️ {len(collected['anomalies_map'])} anomalie(s) de prix détectée(s)."
                )
            if collected["missing_tfs_map"]:
                st.info(
                    f"📡 {len(collected['missing_tfs_map'])} actif(s) avec des TFs manquants."
                )

            st.info("🔍 Analyse des confluences multi-timeframes…")
            conf_full = _compute_confluences(
                symbols_to_scan,
                collected["all_zones_map"], collected["prices_map"],
                collected["bars_map_global"], confluence_threshold,
                collected["scan_errors"],
            )
            summaries = _build_summaries(
                symbols_to_scan,
                collected["prices_map"], collected["trends_map"],
                collected["all_zones_map"], conf_full,
                collected["missing_tfs_map"], collected["price_fallback_map"],
            )

            df_h4 = pd.DataFrame(collected["results_h4"])
            df_daily = pd.DataFrame(collected["results_daily"])
            df_wk = pd.DataFrame(collected["results_weekly"])
            rep_dict = {
                "H4": _apply_pdf_filter(df_h4),
                "Daily": _apply_pdf_filter(df_daily),
                "Weekly": _apply_pdf_filter(df_wk),
            }
            st.session_state["scan_results"] = {
                "df_h4": df_h4, "df_daily": df_daily, "df_weekly": df_wk,
                "conf_full": conf_full, "report_dict": rep_dict,
                "summaries": summaries, "anomalies": collected["anomalies_map"],
                "scan_errors": collected["scan_errors"], "max_dist": max_dist_filter,
                "missing_tfs_map": collected["missing_tfs_map"],
            }

elif not symbols_to_scan and not st.session_state.get("scanning", False):
    st.info("Sélectionnez des actifs à scanner dans la barre latérale.")
elif (not st.session_state.get("pending_scan", False)
      and not st.session_state.get("scanning", False)):
    st.info(
        "Configurez les paramètres dans la barre latérale, "
        "puis cliquez sur **LANCER LE SCAN COMPLET**."
    )

if "scan_results" in st.session_state and not st.session_state.get("pending_scan", False):
    _display_results(st.session_state["scan_results"], max_dist_filter)
