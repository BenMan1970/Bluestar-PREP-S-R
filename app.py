"""Moteur quant + pipeline OANDA (sans Streamlit)."""
import asyncio
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except ImportError:
    _NY_TZ = None

_ICT_TIMEZONE_AVAILABLE = _NY_TZ is not None

import aiohttp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==============================================================================
# [ LAYER 0: GLOBAL CONFIG & LOGGING ]
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# [ LAYER 0b: THREAD-SAFE DATA CACHE ]
# ==============================================================================
# FIX BUG-011 : threading.Lock au lieu d'asyncio.Lock (immune aux event-loops
#   éphémères créés par asyncio.run successifs dans Streamlit).
# FIX BUG-005 : TTL relatif par TF (H4=60s, Daily=300s, Weekly=600s) — plus de
#   bucket discret qui sert des bougies fraîchement closes mais en cache vieux.
# FIX CONC-001 : Cache borné (LRU implicite via purge systématique à chaque accès)
#   et purge à la LECTURE (pas seulement à l'écriture).
_CACHE_TTL_BY_TF: Dict[str, int] = {"h4": 60, "daily": 300, "weekly": 600}
_CACHE_TTL_DEFAULT = 300
_CACHE_MAX_ENTRIES = 256
_CACHE_LOCK = threading.Lock()
# (symbol, tf) -> (fetched_at: float, df: Optional[pd.DataFrame])
_OANDA_CACHE: Dict[tuple, tuple] = {}


def _cache_ttl(tf: str) -> int:
    return _CACHE_TTL_BY_TF.get(tf, _CACHE_TTL_DEFAULT)


def _cache_is_fresh(fetched_at: float, tf: str) -> bool:
    return (time.time() - fetched_at) <= _cache_ttl(tf)


def _cache_purge_stale_locked() -> None:
    """Purge des entrées expirées. Doit être appelée sous _CACHE_LOCK."""
    now = time.time()
    stale = [
        k for k, (ts, _) in _OANDA_CACHE.items()
        if (now - ts) > _cache_ttl(k[1])
    ]
    for k in stale:
        del _OANDA_CACHE[k]
    # Borne supérieure dure : garder les N plus récentes
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
        # AUD-001 : copie défensive — toute mutation aval ne corrompt pas le cache global
        return df.copy(deep=True) if df is not None else None


def _cache_set(symbol: str, tf: str, df: Optional[pd.DataFrame]) -> None:
    # AUD-003 : ne pas cacher None — évite le cache négatif qui bloque les retries
    if df is None:
        return
    with _CACHE_LOCK:
        _OANDA_CACHE[(symbol, tf)] = (time.time(), df)
        _cache_purge_stale_locked()


def clear_oanda_cache() -> None:
    """Vide le cache global (tests / reset manuel)."""
    with _CACHE_LOCK:
        _OANDA_CACHE.clear()


SCANNER_VERSION = "8.5-SNAPSHOT"
# AUD-015 : rejeter les cotations live plus vieilles que ce seuil (secondes)
_MAX_LIVE_PRICE_AGE_SEC = 30
# AUD-021 : parallélisme CPU post-I/O (analyse par symbole)
_SCAN_CPU_MAX_WORKERS = 8

OANDA_ENV_PRACTICE = "practice"
OANDA_ENV_LIVE = "live"
OANDA_API_URLS: Dict[str, str] = {
    OANDA_ENV_PRACTICE: "https://api-fxpractice.oanda.com",
    OANDA_ENV_LIVE: "https://api-fxtrade.oanda.com",
}


def resolve_oanda_env(env_raw: Optional[str]) -> str:
    """AUD-013 : environnement explicite — pas d'auto-détection practice/live."""
    if env_raw is None or not str(env_raw).strip():
        return OANDA_ENV_PRACTICE
    key = str(env_raw).strip().lower()
    if key in OANDA_API_URLS:
        return key
    aliases = {
        "demo": OANDA_ENV_PRACTICE,
        "paper": OANDA_ENV_PRACTICE,
        "fxpractice": OANDA_ENV_PRACTICE,
        "production": OANDA_ENV_LIVE,
        "trade": OANDA_ENV_LIVE,
        "fxtrade": OANDA_ENV_LIVE,
        "real": OANDA_ENV_LIVE,
    }
    if key in aliases:
        return aliases[key]
    if key.startswith("https://"):
        return key.rstrip("/")
    raise ValueError(
        f"OANDA_ENV invalide : {env_raw!r}. Valeurs : practice, live "
        f"(ou secret URL complète https://...)"
    )


def oanda_base_url(env_key: str) -> str:
    if env_key in OANDA_API_URLS:
        return OANDA_API_URLS[env_key]
    if env_key.startswith("https://"):
        return env_key.rstrip("/")
    raise ValueError(f"URL OANDA inconnue pour env={env_key!r}")

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
    # FIX BUG-006 : seuil d'écart prix live vs dernier close, par classe d'actif
    max_live_vs_close_pct: float = 5.0
    # FIX BUG-013 : seuil PDF intégré au profil (anciennement PDF_DIST_THRESHOLDS dead)
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


def flag_data_anomaly(symbol, current_price, support_levels, last_candle_close=None):
    """FIX BUG-006 : seuils par profil + cas explicites."""
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


# ==============================================================================
# [ LAYER 2: ASYNC DATA PIPELINE (OANDA) ]
# ==============================================================================
class OandaAuthError(Exception):
    """Levée quand l'authentification OANDA échoue."""


class OandaClientNotInitializedError(Exception):
    """AUD-014 : client utilisé avant initialize() (env_url absent)."""


class OandaHttpError(Exception):
    """AUD-003 : erreur HTTP OANDA typée (status + contexte)."""

    def __init__(self, status: int, context: str, symbol: str = "", tf: str = ""):
        self.status = status
        self.context = context
        self.symbol = symbol
        self.tf = tf
        super().__init__(f"OANDA HTTP {status} [{context}] {symbol}/{tf}".strip("/"))


class IctTimezoneError(Exception):
    """AUD-018 : tzdata / zoneinfo requis pour les sessions ICT."""


@dataclass(frozen=True)
class LivePriceQuote:
    """AUD-015 : mid marché (bids/asks) + âge de la cotation."""
    mid: float
    bid: float
    ask: float
    quote_time_utc: datetime
    age_seconds: float


def _parse_oanda_live_quote(price_obj: dict, *, now_utc: Optional[datetime] = None) -> Optional[LivePriceQuote]:
    """Extrait bid/ask/mid et l'âge depuis la réponse pricing OANDA v3."""
    if not price_obj:
        return None
    bids = price_obj.get("bids") or []
    asks = price_obj.get("asks") or []
    if bids and asks:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
    elif price_obj.get("closeoutBid") and price_obj.get("closeoutAsk"):
        bid = float(price_obj["closeoutBid"])
        ask = float(price_obj["closeoutAsk"])
    else:
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    time_raw = price_obj.get("time")
    if not time_raw:
        return None
    quote_dt = pd.to_datetime(time_raw, utc=True).to_pydatetime()
    if quote_dt.tzinfo is None:
        quote_dt = quote_dt.replace(tzinfo=timezone.utc)
    now = now_utc or datetime.now(timezone.utc)
    age = max(0.0, (now - quote_dt).total_seconds())
    return LivePriceQuote(
        mid=(bid + ask) / 2.0,
        bid=bid,
        ask=ask,
        quote_time_utc=quote_dt,
        age_seconds=age,
    )


def require_ict_timezone() -> None:
    """AUD-018 : pas de fallback UTC-5 (DST incorrect)."""
    if not _ICT_TIMEZONE_AVAILABLE:
        raise IctTimezoneError(
            "Sessions ICT indisponibles : installez tzdata "
            "(pip install tzdata) pour America/New_York."
        )


def _get_ict_session(dt_utc: datetime) -> str:
    """AUD-018 / BUG-017 : America/New_York obligatoire (tzdata)."""
    require_ict_timezone()
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    ny = dt_utc.astimezone(_NY_TZ)
    h = ny.hour
    if 18 <= h or h < 3:
        return "ASIAN"
    if 3 <= h < 8:
        return "LONDON"
    if 8 <= h < 12:
        return "OVERLAP_LDN_NY"
    return "NEW_YORK"


class AsyncOandaClient:
    def __init__(self, token: str, account_id: str, oanda_env: str = OANDA_ENV_PRACTICE):
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = account_id
        self.oanda_env = resolve_oanda_env(oanda_env)
        self.env_url: Optional[str] = None
        self.env_label: str = self.oanda_env
        # AUD-002 : singleflight — une requête HTTP en vol par (symbol, tf)
        self._inflight_candles: Dict[Tuple[str, str], asyncio.Task] = {}
        self._inflight_lock: Optional[asyncio.Lock] = None

    def _inflight_async_lock(self) -> asyncio.Lock:
        if self._inflight_lock is None:
            self._inflight_lock = asyncio.Lock()
        return self._inflight_lock

    async def initialize(self, session: aiohttp.ClientSession) -> bool:
        """AUD-013 : une seule URL configurée (practice ou live), pas de fallback silencieux."""
        url = oanda_base_url(self.oanda_env)
        try:
            async with session.get(
                f"{url}/v3/accounts/{self.account_id}/summary",
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                if r.status == 200:
                    self.env_url = url
                    self.env_label = (
                        OANDA_ENV_PRACTICE if OANDA_ENV_PRACTICE in url else OANDA_ENV_LIVE
                    )
                    return True
                logging.error(
                    "OANDA auth HTTP %s — env=%s url=%s", r.status, self.oanda_env, url
                )
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as exc:
            logging.error("OANDA auth erreur réseau (%s) env=%s", exc, self.oanda_env)
        return False

    async def _fetch_candles_uncached(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
        tf: str,
        limit: int,
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
        if not self.env_url:
            raise OandaClientNotInitializedError(
                "fetch_candles appelé sans env_url — exécutez initialize() d'abord."
            )

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
                        logging.warning(
                            "Candles HTTP %s pour %s/%s (env=%s)",
                            r.status, symbol, tf, self.env_label,
                        )
                        if r.status in (401, 403, 429):
                            raise OandaHttpError(r.status, "candles", symbol, tf)
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
                        logging.warning("Aucune bougie complete pour %s/%s", symbol, tf)
                        return symbol, tf, None
                    df = pd.DataFrame(candles).tail(limit).set_index("date")
                    result_df = df if not df.empty else None
                    _cache_set(symbol, tf, result_df)
                    return symbol, tf, result_df
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, KeyError, ValueError):
                return symbol, tf, None

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
            logging.debug("Cache HIT: %s/%s", symbol, tf)
            return symbol, tf, cached

        key = (symbol, tf)
        async with self._inflight_async_lock():
            task = self._inflight_candles.get(key)
            if task is None or task.done():
                task = asyncio.create_task(
                    self._fetch_candles_uncached(session, sem, symbol, tf, limit)
                )
                self._inflight_candles[key] = task

        try:
            return await task
        finally:
            async with self._inflight_async_lock():
                if self._inflight_candles.get(key) is task and task.done():
                    del self._inflight_candles[key]

    async def fetch_price(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        symbol: str,
    ) -> Tuple[str, Optional[LivePriceQuote]]:
        """AUD-015 : mid bids/asks + horodatage ; rejet si trop vieux en aval."""
        if not self.env_url:
            raise OandaClientNotInitializedError(
                "fetch_price appelé sans env_url — exécutez initialize() d'abord."
            )
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
                        if r.status in (401, 403, 429):
                            raise OandaHttpError(r.status, "pricing", symbol)
                        return symbol, None
                    data = await r.json()
                    if "prices" in data and data["prices"]:
                        quote = _parse_oanda_live_quote(data["prices"][0])
                        if quote is not None:
                            return symbol, quote
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError, KeyError, ValueError):
                pass
        return symbol, None

# ==============================================================================
# [ LAYER 3: QUANTITATIVE ENGINE (VECTORIZED) ]
# ==============================================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """FIX BUG-003 / BUG-015 : fallback scale-aware au lieu de 0.001 magique.

    Retourne None si l'ATR ne peut être calculé de façon fiable. Les callers
    doivent gérer ce None (typiquement : retourner DataFrames vides + log).
    """
    if df is None or len(df) < period + 1:
        # Fallback : range moyen brut sur ce qui est disponible
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
    """FIX BUG-010 : t-stat sur résidus de régression au lieu de slope/std brut.

    Le ratio précédent (slope/std des prix normalisés) sous-estimait la pente
    en trend fort (std gonflé par la pente elle-même). On utilise désormais
    t_stat = slope / (std_residuals / sqrt(N)), interprétation statistique
    standard. Seuils recalibrés : |t| > 2.0 (~95% confiance) → trend.
    """
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
    """FIX BUG-001 / BUG-002 / BUG-016 — Swing pivot detection rigoureuse.

    Corrections appliquées :
    - BUG-001 : fenêtre droite = [i+1..i+n] strictement (pas de saut). Méthode
      vectorisée : on compare highs[i] aux maxima glissants des fenêtres
      futures construites en sens inverse, puis on shift de -1 pour exclure i.
    - BUG-002 : pas de fillna sur next_close. Les n dernières bougies sont
      exclues du fait de min_periods=n côté droit → pivots non confirmés
      structurellement impossibles.
    - BUG-016 : suppression de la condition redondante `next_close < highs`,
      déjà couverte par `highs > roll_high_right`.
    """
    if df is None or len(df) < 2 * 3 + 2 or atr_val is None or atr_val <= 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    n = 3
    prominence = atr_val * profile.pivot_prominence_atr

    # Index 0..N-1 unifié pour toutes les Series
    highs = pd.Series(df["high"].values)
    lows = pd.Series(df["low"].values)
    closes = pd.Series(df["close"].values)
    opens = pd.Series(df["open"].values)

    # Côté gauche : fenêtre [i-n..i-1] (shift d'abord, puis rolling)
    roll_high_left = highs.shift(1).rolling(n, min_periods=n).max()
    roll_low_left = lows.shift(1).rolling(n, min_periods=n).min()

    # FIX BUG-001 : Côté droit = fenêtre [i+1..i+n] strictement.
    # Construction : on inverse la série, on calcule rolling max (qui couvre
    # [i..i+n-1] en sens normal), puis shift(1) en sens inversé pour exclure i,
    # puis on remet en ordre. Équivalent à : pour chaque i, max(highs[i+1..i+n]).
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

    # FIX BUG-002 / BUG-016 : pas de `next_close`, condition right-side
    # min_periods=n élimine les n dernières bougies automatiquement.
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
    """AUD-007 : complete-link 1D — le nouveau point doit être à ≤ bandwidth de TOUS les membres.

    Évite l'effet chaîne single-link (A proche de B, B proche de C, A loin de C).
    """
    if not price_weight_pairs or bandwidth <= 0:
        return [[pw] for pw in price_weight_pairs] if price_weight_pairs else []
    sorted_pw = sorted(price_weight_pairs, key=lambda x: x[0])
    clusters: List[List[tuple]] = [[sorted_pw[0]]]
    for item in sorted_pw[1:]:
        price = item[0]
        cluster = clusters[-1]
        if all(abs(price - member[0]) <= bandwidth for member in cluster):
            cluster.append(item)
        else:
            clusters.append([item])
    return clusters


def _level_proximity_pct(level_a: float, level_b: float) -> float:
    if level_a <= 0 or level_b <= 0:
        return float("inf")
    return abs(level_a - level_b) / level_a * 100.0


def _union_find_confluence_components(
    levels: np.ndarray,
    threshold_pct: float,
) -> List[List[int]]:
    """AUD-008 : composantes connexes par proximité % — déterministe, ordre indépendant."""
    n = len(levels)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _level_proximity_pct(float(levels[i]), float(levels[j])) <= threshold_pct:
                union(i, j)

    buckets: Dict[int, List[int]] = {}
    for idx in range(n):
        buckets.setdefault(find(idx), []).append(idx)
    return [sorted(v) for v in buckets.values()]


def classify_zone_status(
    level: float,
    zone_type: str,
    df: pd.DataFrame,
    formation_idx: int,
    atr_val: float,
) -> str:
    """FIX BUG-007 : classification directionnelle.

    Un support se teste par le bas : le low s'approche du niveau ET la close
    reste au-dessus (pas de cassure). Une résistance par le haut : le high
    s'approche ET la close reste en dessous.
    """
    if formation_idx >= len(df) - 1 or atr_val is None or atr_val <= 0:
        return "Vierge"
    tolerance = atr_val * 0.25

    c_arr = df["close"].values[formation_idx + 1:]
    h_arr = df["high"].values[formation_idx + 1:]
    l_arr = df["low"].values[formation_idx + 1:]
    if len(c_arr) == 0:
        return "Vierge"

    # FIX BUG-007 : test directionnel
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
    """FIX BUG-003 : si atr_val est None, on retourne vide proprement."""
    if atr_val is None or atr_val <= 0 or df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    profile = get_profile(symbol)
    n_total = len(df)

    pivot_highs, pivot_lows = detect_swing_pivots(df, profile, atr_val, timeframe)
    used_peaks_fallback = len(pivot_highs) + len(pivot_lows) < 3

    if used_peaks_fallback:
        dist = {"h4": 5, "daily": 8, "weekly": 10}.get(timeframe, 5)
        pk = {"distance": dist, "prominence": atr_val * profile.pivot_prominence_atr}
        r_idx, _ = find_peaks(df["high"].values, **pk)
        s_idx, _ = find_peaks(-df["low"].values, **pk)
        # FIX BUG-001 (cohérent) : exclure les n dernières barres aussi du fallback
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
        [(float(p), int(i), (int(i) + 1e-6) / n_total, "high") for i, p in pivot_highs.items()]
        + [(float(p), int(i), (int(i) + 1e-6) / n_total, "low") for i, p in pivot_lows.items()]
    )
    if not all_pivots:
        return pd.DataFrame(), pd.DataFrame()

    bandwidth = atr_val * profile.cluster_radius_atr
    # AUD-006 : on transmet (price, weight, idx, ptype) — agglomerative_1d_clustering n'accède
    # qu'à x[0] pour la comparaison, les champs supplémentaires sont passés sans modification.
    price_weight_pairs = [(p, w, idx, pt) for p, idx, w, pt in all_pivots]
    clusters_raw = agglomerative_1d_clustering(price_weight_pairs, bandwidth)

    strong = []
    # AUD-009 : Role Reverse > Testee (priorité métier), Consommee en dernier
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}

    for grp_pw in clusters_raw:
        if len(grp_pw) < min_touches:
            continue
        # AUD-006 : indices extraits directement depuis le cluster — plus de re-lookup flottant
        grp_prices_arr = np.array([item[0] for item in grp_pw])
        grp_weights_arr = np.array([item[1] for item in grp_pw])
        grp_indices = [item[2] for item in grp_pw]
        grp_ptypes = [item[3] for item in grp_pw]
        if grp_weights_arr.sum() <= 0:
            continue
        lvl = float(np.average(grp_prices_arr, weights=grp_weights_arr))
        if lvl <= 0:
            continue
        last_idx = max(grp_indices)
        age = max(0, n_total - 1 - last_idx)
        # AUD-005 : type structurel par majorité pivot highs/lows, indépendant du prix live
        n_highs = grp_ptypes.count("high")
        n_lows = grp_ptypes.count("low")
        ztype = "Resistance" if n_highs >= n_lows else "Support"
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
    # AUD-010 : is_pivot = pivot structurel swing ; near_price = proximité au prix live
    df_zones["is_pivot"] = not used_peaks_fallback
    if current_price and current_price > 0:
        df_zones["near_price"] = (
            np.abs(df_zones["level"] - current_price) / current_price * 100
        ) <= 0.50
    else:
        df_zones["near_price"] = False
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
    """AUD-008 + BUG-009 : union-find sur proximité %, dédup par TF dans chaque composante."""
    if not zones_dict or not current_price:
        return []
    STATUS_PRIORITY = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}

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
            cols = ["tf", "level", "strength", "age_bars", "status", "type", "is_pivot"]
            if "near_price" in tmp.columns:
                cols.append("near_price")
            frames.append(tmp[cols])

    if not frames:
        return []
    z_df = pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
    z_df = z_df[z_df["level"] > 0]
    if z_df.empty:
        return []

    profile = get_profile(symbol.replace("/", "_"))
    threshold = confluence_threshold if confluence_threshold is not None else profile.confluence_threshold_pct

    components = _union_find_confluence_components(z_df["level"].values, threshold)
    confluences = []

    for member_idxs in components:
        group_full = z_df.loc[member_idxs]
        tfs = group_full["tf"].unique()
        if len(tfs) < 2:
            continue

        sub_avg = float(group_full["level"].mean())
        group_full = group_full.assign(_dist_to_center=(group_full["level"] - sub_avg).abs())
        keep_idx = group_full.groupby("tf")["_dist_to_center"].idxmin().values
        group = group_full.loc[keep_idx].drop(columns=["_dist_to_center"])

        sub_nb_tf = int(group["tf"].nunique())
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

        has_structural = bool(group["is_pivot"].any()) if "is_pivot" in group.columns else False
        if has_structural:
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
            "Nb TF": sub_nb_tf,
            "Force Totale": int(group["strength"].sum()),
            "Score": score,
            "Statut": status,
            "Distance %": round(sub_dist, 3),
            "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else ""),
        })

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
    missing_tfs: List[str] = field(default_factory=list)  # FIX BUG-004
    price_is_fallback: bool = False  # FIX cohérence prix live vs close
    snapshot_at: Optional[float] = None  # AUD-004 : epoch UTC post-bougies + re-fetch prix
    price_quote_age_sec: Optional[float] = None  # AUD-015
    price_is_stale: bool = False  # AUD-015 : cotation live rejetée (âge > seuil)


def _resolve_symbol_price(
    sym_data: Dict[str, Optional[pd.DataFrame]],
    quote: Optional[LivePriceQuote],
) -> Tuple[Optional[float], bool, bool, Optional[float], Optional[str]]:
    """Retourne (cp, fallback, stale, age_sec, message_anomalie_partiel)."""
    age = quote.age_seconds if quote else None
    if quote is not None and quote.age_seconds <= _MAX_LIVE_PRICE_AGE_SEC:
        return quote.mid, False, False, age, None

    stale = quote is not None and quote.age_seconds > _MAX_LIVE_PRICE_AGE_SEC
    partial = None
    if stale:
        partial = (
            f"Prix live expire ({quote.age_seconds:.0f}s > {_MAX_LIVE_PRICE_AGE_SEC}s)"
        )

    for tf_k in ("daily", "h4", "weekly"):
        df = sym_data.get(tf_k)
        if df is not None and not df.empty:
            close_px = float(df["close"].iloc[-1])
            fb_msg = partial or "Prix live indisponible"
            return close_px, True, stale, age, fb_msg

    return None, True, stale, age, partial or "Prix indisponible ou non valide"


def _analyze_symbol(
    sym: str,
    sym_data: Dict[str, Optional[pd.DataFrame]],
    quote: Optional[LivePriceQuote],
    snapshot_at: float,
    min_touches_ui: int,
) -> ScanResult:
    """AUD-021 : moteur quant synchrone par symbole (offload ThreadPool)."""
    try:
        profile = get_profile(sym)
        cp, price_is_fallback, price_is_stale, quote_age, price_msg = _resolve_symbol_price(
            sym_data, quote
        )
        sym_d = sym.replace("_", "/")

        rows = {"H4": None, "Daily": None, "Weekly": None}
        zones_d: Dict[str, tuple] = {}
        trends: Dict[str, str] = {}
        bars_map: Dict[str, int] = {}
        price_ctx = ""
        missing_tfs: List[str] = []

        for tf_k, tf_name in [("h4", "H4"), ("daily", "Daily"), ("weekly", "Weekly")]:
            df = sym_data.get(tf_k)
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
                logging.warning("ATR non calculable pour %s/%s — zones ignorées.", sym, tf_name)
                continue

            min_t = max(3, min_touches_ui) if tf_k == "h4" else max(2, min_touches_ui)
            sup, res = find_strong_sr_zones(df, cp, sym, atr_val, tf_k, min_t)
            zones_d[tf_name] = (sup, res)

            if tf_k == "daily":
                price_ctx = get_price_context(cp, sup, res)

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
        _daily_df = sym_data.get("daily")
        _last_close = (
            float(_daily_df["close"].iloc[-1])
            if (_daily_df is not None and not _daily_df.empty)
            else None
        )
        _anomaly = flag_data_anomaly(sym, cp, _sup_levels, last_candle_close=_last_close)

        if price_is_fallback and cp is not None:
            pf_msg = f"Prix live indisponible — utilisation du dernier close ({cp:.5f})"
            _anomaly = f"{_anomaly} | {pf_msg}" if _anomaly else pf_msg
        elif price_msg:
            extra = (
                f"{price_msg} — fallback close ({cp:.5f})"
                if price_is_fallback and cp
                else price_msg
            )
            _anomaly = f"{_anomaly} | {extra}" if _anomaly else extra

        return ScanResult(
            sym, rows, zones_d, cp, trends, bars_map,
            price_context=price_ctx, anomaly=_anomaly,
            missing_tfs=missing_tfs, price_is_fallback=price_is_fallback,
            snapshot_at=snapshot_at,
            price_quote_age_sec=quote_age,
            price_is_stale=price_is_stale,
        )
    except Exception as e:
        logging.exception("Scan error for %s", sym)
        return ScanResult(sym, {}, {}, None, {}, {}, scan_error=f"{type(e).__name__}: {e}")


def _make_row(z: dict, ztype: str, cp: float, atr_val: float, sym_d: str,
              tf_name: str, df_len: int, profile: InstrumentProfile) -> dict:
    """FIX BUG-014 : closure extraite en fonction pure."""
    dist = abs(cp - z["level"]) / cp * 100 if cp else 0.0
    dist_atr = f"{round(abs(cp - z['level']) / atr_val, 1)}x" if (atr_val and atr_val > 0) else "N/A"
    # FIX BUG-013 : seuil PDF par profil d'instrument
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
    oanda_env: str = OANDA_ENV_PRACTICE,
    scan_run_id: Optional[str] = None,
) -> Tuple[List[ScanResult], Dict[str, str]]:
    client = AsyncOandaClient(token, account_id, oanda_env=oanda_env)
    async with aiohttp.ClientSession() as session:
        if not await client.initialize(session):
            raise OandaAuthError(
                f"Authentification OANDA échouée (env={client.oanda_env}). "
                "Vérifiez OANDA_ACCESS_TOKEN, OANDA_ACCOUNT_ID et OANDA_ENV (practice|live)."
            )

        sem = asyncio.Semaphore(15)

        # AUD-004 : bougies d'abord, puis prix live (évite cp vieilli vs analyse)
        candle_tasks = []
        for sym in symbols:
            for tf in _GRANULARITY_MAP:
                candle_tasks.append(client.fetch_candles(session, sem, sym, tf))
        candles_res = await asyncio.gather(*candle_tasks)
        data_cube: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
        for sym, tf, df in candles_res:
            data_cube.setdefault(sym, {})[tf] = df

        snapshot_at = time.time()
        price_tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
        prices_res = await asyncio.gather(*price_tasks)
        live_quotes: Dict[str, Optional[LivePriceQuote]] = dict(prices_res)

    # AUD-021 : analyse CPU parallèle par symbole (post-I/O)
    workers = min(_SCAN_CPU_MAX_WORKERS, max(1, len(symbols)))
    result_by_sym: Dict[str, ScanResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _analyze_symbol,
                sym,
                data_cube.get(sym, {}),
                live_quotes.get(sym),
                snapshot_at,
                min_touches_ui,
            ): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            res = fut.result()
            result_by_sym[res.symbol] = res

    results = [result_by_sym[sym] for sym in symbols if sym in result_by_sym]

    scan_meta = {
        "scan_run_id": scan_run_id or uuid.uuid4().hex,
        "oanda_env": client.oanda_env,
        "oanda_env_url": client.env_url or "",
        "oanda_env_label": client.env_label,
        "cpu_workers": str(workers),
    }
    return results, scan_meta
