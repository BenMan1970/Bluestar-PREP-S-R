"""Bluestar S/R — coeur pur (Layers 0-5 + orchestration headless).

Extrait mécaniquement de app.py v8.7.3-AUDIT100 (Phase 1).
AUCUNE dépendance Streamlit : importable depuis un script headless.
Aucune logique métier modifiée lors de l'extraction — vérifié par
tests/test_phase1_extraction.py (parité fonction par fonction avec
baseline/app.py.orig).

Les trois fonctions historiquement décorées @st.cache_data
(compute_atr, compute_institutional_trend, find_strong_sr_zones) sont
ici NON décorées. app.py ré-applique le cache Streamlit en patchant
les globals de ce module (voir app.py, section [CACHE PATCH]) afin de
préserver exactement le comportement de la v8.7.3 côté UI.
"""

# Faux positifs au niveau module, documentés ci-dessous :
#  - W0621 (redefined-outer-name) : Streamlit exécute le script au niveau
#    module ; des noms comme `df`, `res`, `sym`, `ctx` existent à la fois
#    dans le scope global UI et comme paramètres de fonctions pures. C'est
#    sans danger (pas de mutation croisée) et conforme au modèle Streamlit.
#  - C0302 (too-many-lines) : application mono-fichier volontaire (déploiement
#    Streamlit Cloud simplifié).
#  - R0902/R0913/R0914/R0917/R0911/R0916 : seuils par défaut de pylint plus
#    stricts que nos dataclasses de contexte et nos signatures quant.
# pylint: disable=redefined-outer-name,too-many-lines,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-return-statements,too-many-boolean-expressions
# pylint: disable=too-few-public-methods

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import random
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
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from scipy.signal import find_peaks

# ==============================================================================
# [ LAYER 0: CONFIG & LOGGING ]
# ==============================================================================
SCANNER_VERSION: Final[str] = "8.8.0-CALIBRATED-20260912"

# PATCH JSON-2 (section 4) : version du profil de calibration, tracée dans le JSON
# pour que le merger sache sur quelle calibration les seuils reposent.
CALIBRATION_PROFILE_VERSION: Final[str] = "v1-2026-09-12 (Phase 3 — recalibrage empirique sur historique profond)"

# Instrumentation de diagnostic (classe A). Strictement inactive quand False :
# aucun calcul, aucune clé ajoutée à scan_results. Mettre à True uniquement
# pour le diagnostic du pipeline de détection. Aucun effet sur la logique métier.
DEBUG_INSTRUMENTATION: Final[bool] = False

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
        """Applique la redaction sur le message, les args et la stacktrace."""
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

# ------------------------------------------------------------------------------
# PATCH AGE-1 (Phase 3, point 3) — normalisation de l'âge DÉCOUPLÉE de la
# profondeur de fetch.
#
# AVANT : age_r = age_bars / len(df). Approfondir l'historique (ex. H4 passant
# de 500 à 7800 bougies) faisait chuter age_r pour TOUTES les zones et gonflait
# silencieusement chaque score (facteur exp(-λ·age_r)), sans aucun changement de
# logique métier.
#
# APRÈS : le dénominateur est une CONSTANTE de référence par TF. `age_bars` est
# déjà invariant à la profondeur (mesuré depuis la formation de la zone jusqu'à
# la dernière bougie : approfondir l'historique décale l'index du pivot du même
# nombre de bougies que n_total). Seul `total_bars` variait.
#
# VALEUR : 500 = profondeur effective mesurée de la v8.7.3 (fetch limit=500 ->
# len(df)=500, vérifié empiriquement sur EUR_USD/XAU_USD/GBP_NZD × 3 TF). Ce
# choix rend age_r STRICTEMENT identique au comportement v8.7.3 à profondeur
# actuelle, tout en le rendant invariant si l'historique est approfondi.
_AGE_REFERENCE_BARS: Final[Dict[str, int]] = {"H4": 500, "Daily": 500, "Weekly": 500}
_AGE_REFERENCE_DEFAULT: Final[int] = 500

# ------------------------------------------------------------------------------
# PATCH DEPTH-1 (Phase 2/3, point 5) — profondeur d'historique par TF.
#
# AVANT : `_fetch_candles_cube` appelait fetch_candles sans `limit` -> défaut
# limit=500 -> count=501 -> len(df)=500 pour les 3 TF, soit H4 ≈ 83 jours,
# Daily ≈ 1.9 an, Weekly ≈ 9.6 ans. AUCUNE TF n'atteignait 3 ans de façon
# homogène ; H4 en était très loin.
#
# APRÈS : profondeur explicite par TF. Valeurs choisies pour dépasser 3 ans
# partout (contrainte de couverture) en restant mesurées sur l'API réelle :
#   H4     : 5000 bougies ≈ 3.2 ans (mesuré : 5000 H4 = 2023-06-26 → 2026-09-11)
#   Daily  : 1500 bougies ≈ 5.8 ans
#   Weekly : 1000 bougies ≈ 19 ans (l'historique OANDA W remonte à 2002)
# L'API plafonne `count` à 5000 (mesuré : 5001 -> HTTP 400). Aucune valeur ne
# dépasse ce plafond.
_FETCH_LIMIT_BY_TF: Final[Dict[str, int]] = {"h4": 5000, "daily": 1500, "weekly": 1000}
_FETCH_LIMIT_DEFAULT: Final[int] = 500

# ------------------------------------------------------------------------------
# PATCH PERF-2 — fenêtre de DÉTECTION découplée de la profondeur de FETCH.
#
# Le walk-forward de calibration (rapport §5.2) a été mené sur des fenêtres de
# 500 bougies, et _AGE_REFERENCE_BARS vaut 500 : le régime de détection validé
# empiriquement est celui d'une fenêtre courte. Détecter sur 5000 barres H4 est
# donc un régime NON calibré : les pivots anciens entrent dans les clusters avec
# un strength plein mais un poids de décroissance au plancher (exp(-2) ~ 0.135),
# ce qui gonfle `Force Totale` et élargit les clusters de confluence.
#
# On conserve la profondeur de FETCH (couverture >= 3 ans, data_window honnête)
# et on borne la profondeur de DÉTECTION. Daily/Weekly restent à leur profondeur
# de fetch (coût négligeable, zéro changement de sortie) ; seul H4 est borné.
# Mettre la valeur à None pour restaurer exactement le comportement v8.8.0.
_DETECTION_WINDOW_BY_TF: Final[Dict[str, Optional[int]]] = {
    "h4": 1500,      # ~1 an ; 3x la fenêtre calibrée, 3.3x moins de barres que 5000
    "daily": None,   # = profondeur de fetch (1500)
    "weekly": None,  # = profondeur de fetch (1000)
}


def _detection_frame(frame: pd.DataFrame, tf_key: str) -> pd.DataFrame:
    """Restreint le DataFrame à la fenêtre de détection calibrée (PATCH PERF-2)."""
    win = _DETECTION_WINDOW_BY_TF.get(str(tf_key).lower())
    if win is None or frame is None or len(frame) <= win:
        return frame
    return frame.tail(win)

# PATCH DEPTH-1 : plafond de `count` de l'API OANDA candles, MESURÉ (et non
# supposé) — count=5000 -> 200, count=5001 -> HTTP 400 "Maximum value for
# 'count' exceeded". Cf. tools/probe_oanda.py et reports/api_probe.txt.
_OANDA_MAX_COUNT: Final[int] = 5000

# PATCH ENV-1 (point 14) : mapping explicite environnement -> URL, pour forcer
# l'environnement OANDA et tracer sans ambiguïté l'environnement d'un run.
_OANDA_ENV_URLS: Final[Dict[str, str]] = {
    "practice": "https://api-fxpractice.oanda.com",
    "trade": "https://api-fxtrade.oanda.com",
}


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
    # PATCH THRESH-1 (point 4) : Optional -> None signifie "non calibré,
    # utiliser le seuil UI". Une valeur explicite rend le champ AUTHORITAIRE.
    confluence_threshold_pct: Optional[float] = None
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
        price_max=8000.0,  # mis à jour 2026 : XAU/USD > 4200, plafond relevé 6000→8000
        confluence_threshold_pct=1.5,  # PATCH THRESH-1 : migré de _SYMBOL_CONFLUENCE_THRESHOLD
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
        confluence_threshold_pct=1.5,  # PATCH THRESH-1 : migré de _SYMBOL_CONFLUENCE_THRESHOLD
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
        price_max=50000.0,  # CORRECTION: plafond relevé à 50000 (NAS100 dépasse régulièrement 30000 en 2026+)
        confluence_threshold_pct=1.5,  # PATCH THRESH-1 : migré de _SYMBOL_CONFLUENCE_THRESHOLD
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
        confluence_threshold_pct=1.2,  # PATCH THRESH-1 : migré de _SYMBOL_CONFLUENCE_THRESHOLD
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
        confluence_threshold_pct=1.3,  # PATCH THRESH-1 : migré de _SYMBOL_CONFLUENCE_THRESHOLD
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
def _candle_ohlc_floats(candle: dict) -> Tuple[float, float, float, float]:
    """Extrait (open, high, low, close) d'une chandelle OANDA."""
    mid = candle["mid"]
    return float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])


def _ohlc_geometry_ok(o: float, h: float, lo: float, cl: float) -> bool:
    """Vérifie la cohérence géométrique d'une bougie OHLC."""
    if not all(np.isfinite(x) for x in (o, h, lo, cl)):
        return False
    if lo <= 0 or h <= 0 or h < lo:
        return False
    return lo <= o <= h and lo <= cl <= h


def _is_valid_candle_dict(c: dict, profile: Optional[InstrumentProfile] = None) -> bool:
    """Vérifie qu'un dictionnaire de chandelle OANDA est valide."""
    try:
        prof = profile or _DEFAULT_PROFILE
        o, h, lo, cl = _candle_ohlc_floats(c)
        if not _ohlc_geometry_ok(o, h, lo, cl):
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


def validate_ohlc_frame(
    frame: Optional[pd.DataFrame], profile: Optional[InstrumentProfile] = None, context: str = ""
) -> pd.DataFrame:
    """PATCH VALID-1 (point 10) — rend ``DataValidationError`` réellement vivante.

    AVANT : ``DataValidationError`` était définie mais jamais levée ni catchée
    (code mort à 100 %). APRÈS : elle est levée dès qu'un DataFrame destiné au
    moteur de détection est structurellement inexploitable. Utilisée par le
    harnais de collecte (Phase 2) pour ÉCHOUER BRUYAMMENT au lieu de propager
    silencieusement des données corrompues vers la recalibration.

    Contrat : lève ``DataValidationError`` ; retourne le DataFrame (inchangé) si
    valide. Ne duplique PAS la logique de ``_sanitize_ohlc_dataframe`` (qui, elle,
    nettoie) : ici on ne fait que constater et refuser.
    """
    label = f"[{context}] " if context else ""
    if frame is None:
        raise DataValidationError(f"{label}DataFrame is None")
    if len(frame) == 0:
        raise DataValidationError(f"{label}DataFrame is empty")
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"{label}missing OHLC columns: {sorted(missing)}")
    sub = frame[["open", "high", "low", "close"]]
    if sub.isna().to_numpy().any():
        raise DataValidationError(f"{label}NaN values in OHLC")
    if not np.isfinite(sub.to_numpy(dtype=float)).all():
        raise DataValidationError(f"{label}non-finite OHLC values")
    if (sub["high"] < sub["low"]).any():
        raise DataValidationError(f"{label}high < low detected")
    if (sub["low"] <= 0).any():
        raise DataValidationError(f"{label}non-positive low detected")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{label}index is not a DatetimeIndex")
    if frame.index.has_duplicates:
        raise DataValidationError(f"{label}duplicate timestamps in index")
    return frame


class AsyncOandaClient:
    """Client asynchrone pour l'API OANDA."""

    def __init__(self, token: str, oanda_account_id: str) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.account_id = oanda_account_id
        self.env_url: Optional[str] = None
        # PATCH ENV-1 (point 14) : nom de l'environnement résolu, journalisé.
        self.env_name: Optional[str] = None

    @staticmethod
    def _backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
        """PATCH BACKOFF-1 (point 13) — backoff exponentiel + JITTER.

        AVANT : ``0.5 * 2**attempt`` pur. Avec semaphore=12, les 12 workers
        retentaient en phase (thundering herd) et pouvaient re-saturer l'API.
        Le jitter décorrèle les tentatives. Si OANDA fournit un ``Retry-After``,
        il est respecté tel quel (aucune valeur de rate-limit n'est supposée).
        """
        if retry_after is not None:
            return float(retry_after)
        return 0.5 * (2 ** attempt) + random.uniform(0.0, 0.5)

    @staticmethod
    async def _get_json_with_retry(session, url, headers, params, timeout_total, retries=3):
        """Effectue une requête GET avec retry (backoff + jitter, cf. PATCH BACKOFF-1)."""
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
                        ra = None
                        if r.status == 429:
                            try:
                                ra = float(r.headers.get("Retry-After", "")) if r.headers.get("Retry-After") else None
                            except (TypeError, ValueError):
                                ra = None
                        await asyncio.sleep(AsyncOandaClient._backoff_delay(attempt, ra))
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError):
                if attempt < retries - 1:
                    await asyncio.sleep(AsyncOandaClient._backoff_delay(attempt))
                continue
        return None

    async def initialize(self, session: aiohttp.ClientSession, env_override: Optional[str] = None) -> bool:
        """Détermine l'environnement OANDA (practice ou trade).

        PATCH ENV-1 (point 14) : ``env_override`` ("practice" | "trade") force
        l'environnement et supprime toute ambiguïté. Si fourni, SEUL cet
        environnement est sondé. L'environnement résolu est TOUJOURS journalisé
        explicitement, ce qui garantit qu'un run de calibrage est traçable.
        """
        order = [env_override] if env_override else ["practice", "trade"]
        for name in order:
            url = _OANDA_ENV_URLS.get(name)
            if not url:
                _LOG.error("OANDA env inconnu demandé: %r", name)
                continue
            try:
                async with session.get(
                    f"{url}/v3/accounts/{self.account_id}/summary",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        self.env_url = url
                        self.env_name = name
                        _LOG.info(
                            "OANDA environment resolved = %s (%s)%s",
                            name, url, " [forced]" if env_override else " [auto-detect]",
                        )
                        return True
                    _LOG.info("OANDA env probe %s -> HTTP %s", name, r.status)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue
        _LOG.error("OANDA environment resolution FAILED (no 200 from %s)", order)
        return False

    @staticmethod
    def _parse_candles_payload(data: dict, profile: InstrumentProfile, limit: int):
        """Transforme une réponse OANDA en DataFrame OHLC nettoyé (ou None).

        PATCH PERF-1 — parsing VECTORISÉ.
        AVANT : une list-comprehension appelait `pd.to_datetime()` scalaire et
        `_is_valid_candle_dict()` (try/except) par bougie. Avec _FETCH_LIMIT_BY_TF
        (5000 H4 / 1500 D / 1000 W) x 33 actifs = 247 500 appels scalaires, exécutés
        DANS la coroutine -> l'event loop était bloqué et les 12 fetchs concurrents
        se sérialisaient derrière le parsing.
        APRÈS : une seule conversion datetime vectorisée + un masque booléen numpy.
        La logique de validation est IDENTIQUE à _is_valid_candle_dict (mêmes
        prédicats, même ordre, même max_high_low_ratio) -> parité de sortie.
        """
        raw = data.get("candles") or []
        if not raw:
            return None

        times: List[str] = []
        o_raw: List[Any] = []
        h_raw: List[Any] = []
        l_raw: List[Any] = []
        c_raw: List[Any] = []
        v_raw: List[Any] = []

        for c in raw:
            if not c.get("complete"):
                continue
            mid = c.get("mid")
            if not mid:
                continue
            t = c.get("time")
            if t is None:
                continue
            try:
                o_raw.append(mid["o"])
                h_raw.append(mid["h"])
                l_raw.append(mid["l"])
                c_raw.append(mid["c"])
            except KeyError:
                continue
            times.append(t)
            v_raw.append(c.get("volume", 0))

        if not times:
            return None

        try:
            idx = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601")
        except (ValueError, TypeError):
            # Repli si le format OANDA dévie de l'ISO8601 strict.
            idx = pd.to_datetime(pd.Series(times), utc=True, errors="coerce")

        frame = pd.DataFrame(
            {
                "date": idx,
                "open": pd.to_numeric(pd.Series(o_raw), errors="coerce"),
                "high": pd.to_numeric(pd.Series(h_raw), errors="coerce"),
                "low": pd.to_numeric(pd.Series(l_raw), errors="coerce"),
                "close": pd.to_numeric(pd.Series(c_raw), errors="coerce"),
                "volume": pd.to_numeric(pd.Series(v_raw), errors="coerce").fillna(0),
            }
        )
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        if frame.empty:
            return None

        max_ratio = float(getattr(profile, "max_high_low_ratio", 1.8))
        o = frame["open"].to_numpy(dtype=float)
        h = frame["high"].to_numpy(dtype=float)
        lo = frame["low"].to_numpy(dtype=float)
        cl = frame["close"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(lo > 0, h / lo, np.inf)

        ok = (
            np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(cl)
            & (lo > 0) & (h > 0) & (h >= lo)
            & (o >= lo) & (o <= h)
            & (cl >= lo) & (cl <= h)
            & (ratio <= max_ratio)
        )
        frame = frame[ok]
        if frame.empty:
            return None

        frame["volume"] = frame["volume"].astype("int64", errors="ignore")
        return _sanitize_ohlc_dataframe(
            frame.set_index("date").tail(limit),
            profile,
        )

    async def fetch_candles(self, session, sem, symbol, timeframe, limit=500, force=False):
        """Récupère les chandelles pour un symbole/timeframe.

        PATCH RETRY-1 : ``force=True`` court-circuite le cache (y compris les
        entrées NÉGATIVES, TTL 20 s) afin de permettre une reprise explicite
        après un échec transitoire — sans ce drapeau, une reprise immédiate
        relirait l'entrée négative et ne retenterait rien.
        """
        if not force:
            cache_hit, cached = _cache_get(self.env_url, self.account_id, symbol, timeframe)
            if cache_hit:
                return symbol, timeframe, cached
        gran = _GRANULARITY_MAP.get(timeframe)
        if not gran or not self.env_url:
            return symbol, timeframe, None
        url = f"{self.env_url}/v3/instruments/{symbol}/candles"
        # PATCH DEPTH-2 — plafond API. Sans ce plafond, limit=5000 -> count=5001
        # -> HTTP 400 "Maximum value for 'count' exceeded" -> la TF H4 devenait
        # SILENCIEUSEMENT vide (bug détecté empiriquement : data_window.H4
        # candles_used=0 après PATCH DEPTH-1). Le plafond est MESURÉ (5000), pas supposé.
        params = {"count": min(limit + 1, _OANDA_MAX_COUNT), "granularity": gran, "price": "M"}
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, params, _PER_REQUEST_TIMEOUT_S
            )
            if data is None:
                _cache_set(self.env_url, self.account_id, symbol, timeframe, None)
                return symbol, timeframe, None
            try:
                profile = get_profile(symbol)
                df_clean = self._parse_candles_payload(data, profile, limit)
                _cache_set(self.env_url, self.account_id, symbol, timeframe, df_clean)
                return symbol, timeframe, df_clean
            except (KeyError, ValueError, TypeError):
                _cache_set(self.env_url, self.account_id, symbol, timeframe, None)
                return symbol, timeframe, None

    async def fetch_price(self, session, sem, symbol):
        """Récupère le prix courant (mid) pour un symbole.

        Retourne un tuple (symbol, price, source) où source vaut :
          - "live"          : prix bid/ask mid en temps réel (tradeable=True)
          - "stale"         : marché fermé (tradeable=False dans OANDA pricing) ;
                             le price retourné est quand même le dernier mid connu.
          - None            : fetch échoué (price=None, source=None).
        """
        if not self.env_url:
            return symbol, None, None
        url = f"{self.env_url}/v3/accounts/{self.account_id}/pricing"
        async with sem:
            data = await self._get_json_with_retry(
                session, url, self.headers, {"instruments": symbol}, 5
            )
            try:
                if data and "prices" in data and data["prices"]:
                    price_obj = data["prices"][0]
                    bid = float(price_obj["closeoutBid"])
                    ask = float(price_obj["closeoutAsk"])
                    if np.isfinite(bid) and np.isfinite(ask) and bid > 0:
                        mid = (bid + ask) / 2
                        # SR-1 FIX: OANDA retourne tradeable=False quand le marché
                        # est fermé (ex: indices US hors session, weekend).
                        # Dans ce cas le prix mid est stale (dernier prix connu).
                        is_tradeable = bool(price_obj.get("tradeable", True))
                        source = "live" if is_tradeable else "stale"
                        return symbol, mid, source
            except (KeyError, ValueError, IndexError, TypeError):
                pass
        return symbol, None, None


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


def _trend_residual_band(resid: np.ndarray, atr_val: float, current_price: float) -> float:
    """Calcule la largeur de bande à partir des résidus (avec fallback robuste)."""
    band = float(np.std(resid))
    if np.isfinite(band) and band > 0:
        return band
    anchor_mag = abs(float(np.quantile(resid, 0.5)))
    if anchor_mag > 0:
        return anchor_mag
    return atr_val / max(current_price, 1e-9)


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


@dataclass(frozen=True)
class _PivotArrays:
    """Vecteurs intermédiaires pour la détection de pivots vectorisée."""

    highs: pd.Series
    lows: pd.Series
    closes: pd.Series
    opens: pd.Series


def _build_pivot_arrays(dataframe: pd.DataFrame, n_total: int) -> _PivotArrays:
    """Construit les Series indexées pour la détection vectorisée des pivots."""
    idx = pd.RangeIndex(n_total)
    return _PivotArrays(
        highs=pd.Series(dataframe["high"].to_numpy(dtype=float), index=idx),
        lows=pd.Series(dataframe["low"].to_numpy(dtype=float), index=idx),
        closes=pd.Series(dataframe["close"].to_numpy(dtype=float), index=idx),
        opens=pd.Series(dataframe["open"].to_numpy(dtype=float), index=idx),
    )


def _swing_break_masks(
    arr: _PivotArrays,
    n: int,
    profile: InstrumentProfile,
    timeframe: str,
) -> Tuple[pd.Series, pd.Series]:
    """Calcule les masques de swing-high / swing-low (avec filtre de mèche)."""
    highs, lows, closes, opens = arr.highs, arr.lows, arr.closes, arr.opens

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
            (highs > roll_high_left) & (highs > roll_high_right) & (upper_wick_pct >= wick_threshold)
        )
        sl_mask = (
            (lows < roll_low_left) & (lows < roll_low_right) & (lower_wick_pct >= wick_threshold)
        )
    return sh_mask, sl_mask


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

        arr = _build_pivot_arrays(dataframe, n_total)
        sh_mask, sl_mask = _swing_break_masks(arr, n, profile, timeframe)

        roll_low_around = arr.lows.rolling(2 * n + 1, center=True, min_periods=1).min()
        roll_high_around = arr.highs.rolling(2 * n + 1, center=True, min_periods=1).max()
        high_prom = arr.highs - roll_low_around
        low_prom = roll_high_around - arr.lows

        sh_mask = sh_mask & (high_prom >= prominence_min)
        sl_mask = sl_mask & (low_prom >= prominence_min)

        safe_cutoff = n_total - n
        sh_idx = [i for i in sh_mask[sh_mask].index.tolist() if i < safe_cutoff]
        sl_idx = [i for i in sl_mask[sl_mask].index.tolist() if i < safe_cutoff]

        pivots: List[PivotPoint] = []
        for sh_index in sh_idx:
            pivots.append(
                PivotPoint(
                    price=float(arr.highs.iloc[sh_index]),
                    weight=_time_decay_weight(int(sh_index), n_total),
                    index=int(sh_index),
                    kind="high",
                    prominence=float(high_prom.iloc[sh_index]),
                )
            )
        for sl_index in sl_idx:
            pivots.append(
                PivotPoint(
                    price=float(arr.lows.iloc[sl_index]),
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


def _confirm_local_extremum(arr: np.ndarray, peak_idx: int, n: int, is_high: bool) -> bool:
    """Confirme qu'un pic détecté est bien un extremum local strict."""
    if peak_idx < n or peak_idx + n >= len(arr):
        return False
    left = arr[peak_idx - n : peak_idx]
    right = arr[peak_idx + 1 : peak_idx + n + 1]
    if is_high:
        return arr[peak_idx] > np.nanmax(left) and arr[peak_idx] > np.nanmax(right)
    return arr[peak_idx] < np.nanmin(left) and arr[peak_idx] < np.nanmin(right)


def _peaks_to_pivots(
    arr: np.ndarray,
    peak_idx: np.ndarray,
    props: dict,
    n: int,
    n_total: int,
    safe_cutoff: int,
    kind: str,
) -> List[PivotPoint]:
    """Convertit les pics scipy en PivotPoint validés."""
    is_high = kind == "high"
    out: List[PivotPoint] = []
    for k, p_idx in enumerate(peak_idx):
        if p_idx >= safe_cutoff:
            continue
        if not _confirm_local_extremum(arr, p_idx, n, is_high):
            continue
        out.append(
            PivotPoint(
                price=float(arr[p_idx]),
                weight=_time_decay_weight(int(p_idx), n_total),
                index=int(p_idx),
                kind=kind,
                prominence=float(props["prominences"][k]),
            )
        )
    return out


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
    extra = _peaks_to_pivots(
        high_arr, high_idx, high_props, n, n_total, safe_cutoff, "high"
    ) + _peaks_to_pivots(low_arr, low_idx, low_props, n, n_total, safe_cutoff, "low")
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


def _zone_test_break_masks(
    level: float,
    zone_type: str,
    c_arr: np.ndarray,
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule les masques de test et de cassure d'une zone."""
    if zone_type == "Support":
        test_mask = (l_arr <= level + tolerance) & (c_arr > level - tolerance)
        break_mask = c_arr < level - tolerance
    else:
        test_mask = (h_arr >= level - tolerance) & (c_arr < level + tolerance)
        break_mask = c_arr > level + tolerance
    return test_mask, break_mask


def _classify_post_break(
    level: float,
    zone_type: str,
    c_arr: np.ndarray,
    h_arr: np.ndarray,
    l_arr: np.ndarray,
    break_idx: int,
    tolerance: float,
) -> str:
    """Classifie le statut après une cassure (Consommee / Role Reverse)."""
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
    test_mask, break_mask = _zone_test_break_masks(
        level, zone_type, c_arr, h_arr, l_arr, tolerance
    )
    has_approach = bool(test_mask.any())
    break_positions = np.where(break_mask)[0]
    if len(break_positions) == 0:
        return "Testee" if has_approach else "Vierge"
    break_idx = int(break_positions[0])
    return _classify_post_break(level, zone_type, c_arr, h_arr, l_arr, break_idx, tolerance)


def compute_structural_score(
    strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int = 0
) -> float:
    """Score structurel d'une zone.

    PATCH AGE-1 (point 3) : la décroissance temporelle utilise désormais une
    fenêtre de référence FIXE par TF (``_AGE_REFERENCE_BARS``) et non plus la
    profondeur d'historique réellement chargée. ``total_bars`` est conservé dans
    la signature pour la stabilité d'API mais n'influence PLUS le calcul.
    """
    _ = total_bars  # conservé pour compatibilité d'API — n'entre plus dans age_r
    tf_w = TF_WEIGHT.get(tf_name, 1.0)
    ref = _AGE_REFERENCE_BARS.get(tf_name, _AGE_REFERENCE_DEFAULT)
    age_r = max(age_bars, 0) / max(ref, 1)
    lam = _TF_LAMBDA.get(tf_name, 1.5)
    return round((strength * tf_w * nb_tf) * float(np.exp(-lam * age_r)), 1)


_STATUS_PRIORITY: Final[Dict[str, int]] = {
    "Vierge": 0,
    "Testee": 1,
    "Role Reverse": 2,
    "Consommee": 3,
}


def _cluster_metrics(grp: List[PivotPoint]) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Calcule prix/poids/proéminences d'un cluster (ou None si invalide)."""
    prices = np.array([p.price for p in grp], dtype=float)
    weights = np.array([p.weight for p in grp], dtype=float)
    prominences = np.array([p.prominence for p in grp], dtype=float)
    if not np.all(np.isfinite(prices)):
        return None
    if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
        weights = np.ones_like(prices)
    max_prominence = float(np.nanmax(prominences)) if len(prominences) else 0.0
    avg_prominence = (
        float(np.average(prominences, weights=weights)) if weights.sum() > 0 else max_prominence
    )
    return prices, weights, max_prominence, avg_prominence


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
    metrics = _cluster_metrics(grp)
    if metrics is None:
        return None
    prices, weights, max_prominence, avg_prominence = metrics

    touches = len({p.index for p in grp})
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


def _merge_two_zones(prev: dict, z: dict, same_type: bool) -> dict:
    """Fusionne deux zones adjacentes en une seule (pondérée par la force)."""
    prev_strength = max(int(prev.get("strength", 1)), 1)
    z_strength = max(int(z.get("strength", 1)), 1)
    new_strength = prev_strength + z_strength
    new_level = (prev["level"] * prev_strength + z["level"] * z_strength) / new_strength
    return {
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
        merged[-1] = _merge_two_zones(prev, z, same_type)
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


def _swing_clustered_zones(ctx: _SwingZoneContext) -> Tuple[List[dict], List[dict]]:
    """Phase 1 : clustering des pivots de swing en supports/résistances."""
    n_total = len(ctx.dataframe)
    pivots = _get_pivots_with_fallback_meta(ctx.dataframe, ctx.profile, ctx.atr_val, ctx.timeframe)

    raw_bandwidth = max(ctx.atr_val * ctx.profile.cluster_radius_atr, ctx.current_price * 0.0015)
    max_bandwidth = ctx.current_price * (ctx.profile.max_cluster_width_pct / 100.0)
    bandwidth = float(min(raw_bandwidth, max_bandwidth))

    if not (pivots and bandwidth > 0 and np.isfinite(bandwidth)):
        return [], []

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
    return support_zones, resistance_zones


def _add_trend_structure_union(
    ctx: _SwingZoneContext, support_zones: List[dict], resistance_zones: List[dict]
) -> None:
    """Phase 2 : ajoute (union) les zones trend-structure pour INDEX/METAL."""
    if ctx.profile.asset_class not in ("INDEX", "METAL"):
        return
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


def _compute_swing_zones(ctx: _SwingZoneContext) -> Tuple[List[dict], List[dict]]:
    """Phase 1+2 : clustering swing + union trend-structure."""
    support_zones, resistance_zones = _swing_clustered_zones(ctx)
    _add_trend_structure_union(ctx, support_zones, resistance_zones)
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


def _sr_inputs_invalid(
    dataframe: pd.DataFrame, current_price: float, atr_val: Optional[float]
) -> bool:
    """Vérifie si les entrées de détection S/R sont invalides."""
    if atr_val is None or atr_val <= 0:
        return True
    if dataframe is None or dataframe.empty:
        return True
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        return True
    return False


def _build_zones_dataframe(
    support_zones: List[dict],
    resistance_zones: List[dict],
    atr_val: float,
    current_price: float,
    profile: InstrumentProfile,
) -> Optional[pd.DataFrame]:
    """Fusionne, ordonne et annote l'ensemble des zones en un DataFrame."""
    merge_thresh_raw = atr_val * profile.merge_threshold_atr
    merge_thresh_cap = current_price * 0.0075
    merge_thresh = float(min(merge_thresh_raw, merge_thresh_cap))

    # PATCH 1 — Filtrage pré-fusion Consommee (méta-audit validé, risque LOW)
    # Une zone Consommee (priorité 3 dans _STATUS_PRIORITY) absorbe silencieusement
    # une zone valide adjacente dans _merge_two_zones, la faisant disparaître lors
    # du filtre aval _flatten_one_tf. Les Consommee sont exclues avant fusion car
    # _flatten_one_tf les élimine de toute façon ; elles ne doivent pas participer
    # à la fusion.
    support_zones = _merge_adjacent_zones(
        [z for z in support_zones if z.get("status") != "Consommee"], merge_thresh
    )
    resistance_zones = _merge_adjacent_zones(
        [z for z in resistance_zones if z.get("status") != "Consommee"], merge_thresh
    )

    all_zones = support_zones + resistance_zones
    if not all_zones:
        return None

    df_zones = pd.DataFrame(all_zones)
    if df_zones.empty or "level" not in df_zones.columns:
        return None

    df_zones = df_zones.sort_values(
        ["level", "zone_type", "age_bars"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    df_zones = _reclassify_orphan_zones(df_zones, current_price)
    df_zones["near_price"] = (
        np.abs(df_zones["level"] - current_price) / current_price * 100
    ) <= 0.50
    return df_zones


def _trend_mode_safety_net(
    dataframe: pd.DataFrame,
    current_price: float,
    profile: InstrumentProfile,
    atr_val: float,
    symbol: str,
    timeframe: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filet de sécurité trend-structure quand les zones orphelines bloquent l'output."""
    _LOG.info("Trend-mode safety net for %s %s (orphan zones blocked output)", symbol, timeframe)
    tr_zones = _detect_trend_structure_zones(
        dataframe, current_price, profile, atr_val, "Support"
    ) + _detect_trend_structure_zones(dataframe, current_price, profile, atr_val, "Resistance")
    if not tr_zones:
        return pd.DataFrame(), pd.DataFrame()
    tr_df = pd.DataFrame(tr_zones)
    tr_df["near_price"] = (np.abs(tr_df["level"] - current_price) / current_price * 100) <= 0.50
    return _split_supports_resistances(tr_df, current_price)


def find_strong_sr_zones(
    dataframe: pd.DataFrame,
    current_price: float,
    symbol: str,
    atr_val: Optional[float],
    timeframe: str,
    min_touches_required: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fonction principale de détection des zones de support/résistance."""
    if _sr_inputs_invalid(dataframe, current_price, atr_val):
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

    df_zones = _build_zones_dataframe(
        support_zones, resistance_zones, atr_val, current_price, profile
    )
    if df_zones is None:
        return pd.DataFrame(), pd.DataFrame()

    supports, resistances = _split_supports_resistances(df_zones, current_price)

    if supports.empty and resistances.empty and profile.asset_class in ("INDEX", "METAL"):
        supports, resistances = _trend_mode_safety_net(
            dataframe, current_price, profile, atr_val, symbol, timeframe
        )

    return supports, resistances


def _flatten_one_tf(df_z: pd.DataFrame, timeframe_key: str, ztype: str) -> Optional[pd.DataFrame]:
    """Aplatit une (sous-)table de zones d'une timeframe (ou None si vide)."""
    if df_z is None or df_z.empty:
        return None
    tmp = df_z[df_z["status"] != "Consommee"].copy()
    if tmp.empty:
        return None
    tmp = tmp.assign(tf=timeframe_key, type=tmp["near_price"].map({True: "Pivot", False: ztype}))
    cols = ["tf", "level", "strength", "age_bars", "status", "type", "near_price"]
    for c in ["prominence_atr", "is_major"]:
        if c in tmp.columns:
            cols.append(c)
    return tmp[[c for c in cols if c in tmp.columns]]


def _flatten_zones_to_dataframe(zones_dict: dict, atr_map: Optional[dict] = None) -> pd.DataFrame:
    """Aplatit le dictionnaire de zones en un DataFrame unique.

    PATCH JSON-2 (point 8) : ``atr_map`` (TF -> ATR) permet d'attacher l'ATR de
    chaque TF à ses zones, afin de calculer ``distance_atr`` dans l'export JSON v2.
    """
    frames = []
    for timeframe_key, pair in zones_dict.items():
        try:
            sup, res = pair
        except (ValueError, TypeError):
            continue
        for df_z, ztype in [(sup, "Support"), (res, "Resistance")]:
            flat = _flatten_one_tf(df_z, timeframe_key, ztype)
            if flat is not None:
                frames.append(flat)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
    if atr_map:
        out["atr"] = out["tf"].map(lambda t: atr_map.get(t))
    return out


def _group_score(group: pd.DataFrame, sub_nb_tf: int) -> float:
    """Calcule le score pondéré d'un groupe de confluences.

    PATCH SCORE-1 (sqrt nb_tf) : remplace sub_nb_tf par sqrt(sub_nb_tf) pour
    supprimer l'effet quadratique O(nb_tf²). La somme contient déjà sub_nb_tf
    termes (un par TF après groupby.idxmin) — multiplier chaque terme par
    sub_nb_tf créait une croissance quadratique qui écrasait les setups forex
    valides derrière XAU/indices. sqrt est monotone : "plus de TF = mieux"
    est préservé, seul l'effet quadratique disparaît.

    PATCH SCORE-2 (major_bonus borné) : min(is_major.sum(), 2) sature le bonus
    à +60% max (au lieu de +90%/+120% non borné). Atténue sans réétalonner.

    Recalibration seuils : XAU 920 → ~531, équivalent seuil ~57 post-patch
    (vs 100 pré-patch). Mettre à jour llm_min_score en conséquence.

    PATCH AGE-1 (point 3) : la décroissance temporelle utilise la fenêtre de
    référence FIXE par TF (``_AGE_REFERENCE_BARS``) au lieu de ``bars_map``
    (profondeur réellement chargée). ``bars_map`` n'est donc plus un paramètre.
    """
    tf_w = group["tf"].map(TF_WEIGHT).fillna(1.0).values
    refs = (
        group["tf"]
        .map(lambda t: _AGE_REFERENCE_BARS.get(t, _AGE_REFERENCE_DEFAULT))
        .values.astype(float)
    )
    age_r = np.clip(group["age_bars"].values / np.maximum(refs, 1), 0, 1)
    lams = group["tf"].map(_TF_LAMBDA).fillna(1.5).values
    major_bonus = 1.0
    if "is_major" in group.columns:
        # PATCH SCORE-2 : borne le bonus (saturation +60% max)
        major_bonus = 1.0 + 0.3 * min(int(group["is_major"].sum()), 2)
    # PATCH SCORE-1 : sqrt(nb_tf) au lieu de nb_tf — supprime O(nb_tf²)
    mtf_factor = float(np.sqrt(max(sub_nb_tf, 1)))
    return round(
        float(
            (group["strength"].values * tf_w * mtf_factor * np.exp(-lams * age_r)).sum()
            * major_bonus
        ),
        1,
    )


def _classify_confluence_type(
    group: pd.DataFrame, safe_cp: float, sub_dist: float
) -> Tuple[str, str]:
    """Détermine le type (Pivot/Support/Resistance) et le signal d'une confluence."""
    if sub_dist <= 0.50:
        return "Pivot", "↔ PIVOT ZONE"
    n_sup = (group["level"] < safe_cp).sum()
    ctype = "Support" if n_sup >= len(group) - n_sup else "Resistance"
    sig = "🟢 BUY ZONE" if ctype == "Support" else "🔴 SELL ZONE"
    return ctype, sig


def compute_zone_id(symbol: str, level: float, zone_type: str) -> str:
    """PATCH JSON-2 (point 8) — identité STABLE de zone, déterministe.

    Hash de ``symbol + niveau arrondi + type``. Deux runs successifs produisent
    le MÊME ``zone_id`` pour la même zone, ce qui permet au merger de suivre son
    cycle de vie (Vierge -> Testee -> Role Reverse -> Consommee) d'un run à
    l'autre, au lieu de recevoir des instantanés déconnectés.
    """
    key = f"{symbol}|{round(float(level), 5)}|{zone_type}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _zone_per_timeframe(group: pd.DataFrame) -> List[dict]:
    """Détail par TF d'une confluence (une entrée par TF après idxmin)."""
    out: List[dict] = []
    for tf in sorted(group["tf"].unique()):
        r = group[group["tf"] == tf].iloc[0]
        prom = r.get("prominence_atr")
        try:
            prom = None if pd.isna(prom) else round(float(prom), 3)
        except (TypeError, ValueError):
            prom = None
        out.append(
            {
                "tf": tf,
                "level": round(float(r["level"]), 5),
                "touches": int(r["strength"]),
                "age_bars": int(r["age_bars"]),
                "status": r["status"],
                "prominence_atr": prom,
                "is_major": bool(r.get("is_major", False)) if "is_major" in group.columns else False,
            }
        )
    return out


# ==============================================================================
# [ CALIBRATION EMPIRIQUE — tables mesurées, rapport v1-2026-09-12 ]
# ==============================================================================
# Source : §5.6 "Contrôle du confondant distance" — taux de RETOUCHE observé sur
# 16 693 confluences / 12 actifs, walk-forward aligné H4, ventilé mono vs multi-TF.
# Ces valeurs sont des FRÉQUENCES OBSERVÉES, pas un modèle. Elles décrivent la
# PORTÉE (la zone sera-t-elle atteinte dans l'horizon ?), jamais la TENUE.
_REACH_BY_DIST_ATR_MULTI_TF: Final[List[Tuple[float, float, int]]] = [
    (0.5, 1.000, 125),
    (1.0, 0.954, 129),
    (2.0, 0.892, 305),
    (4.0, 0.690, 697),
    (8.0, 0.438, 1393),
    (float("inf"), 0.051, 6790),
]
_REACH_BY_DIST_ATR_MONO_TF: Final[List[Tuple[float, float, int]]] = [
    (0.5, 1.000, 19),
    (1.0, 1.000, 20),
    (2.0, 0.849, 53),
    (4.0, 0.711, 83),
    (8.0, 0.500, 200),
    (float("inf"), 0.013, 6879),
]

# Source : §5.6 / §5.8 / §10.10 — P(respect | touch) est INVARIANTE : ~0.14 quelle
# que soit la profondeur de confluence (mono 0.146 / bi 0.140 / tri 0.141) et quel
# que soit le seuil de confluence balayé (0.1388 / 0.1384 / 0.1415). C'est donc une
# CONSTANTE globale, et surtout PAS un discriminant par zone. Toute tentative de la
# décliner par zone serait une invention : le rapport démontre l'inverse (§5.7).
_RESPECT_GIVEN_TOUCH_GLOBAL: Final[float] = 0.14

_CALIBRATION_EVIDENCE: Final[Dict[str, Any]] = {
    "source_report": "v1-2026-09-12 (Phase 3 — recalibrage empirique sur historique profond)",
    "sample": "16693 confluences / 12 actifs / walk-forward out-of-sample, aligné H4",
    "reach_table_ref": "§5.6 (taux de retouche par tranche de distance ATR, mono vs multi-TF)",
    "respect_given_touch": _RESPECT_GIVEN_TOUCH_GLOBAL,
    "respect_given_touch_ref": "§5.6 / §5.8 / §10.10 — invariante (~0.14)",
    "horizon_bars": {"H4": 120, "Daily": 40, "Weekly": 12},
    "regime": "2023-2026, régime de marché unique (cf. §5.9d, §11)",
    "known_limits": [
        "reach_probability = PORTÉE (zone atteinte), jamais TENUE (zone défendue).",
        "P(respect|touch) ~0.14 est globale et non discriminante par zone (§5.7).",
        "Le score n'est prédictif que dans la bande 4-8 ATR (+10.6 pp, p=0.0026) "
        "et n'est PAS un prédicteur de tenue -> confidence_tier reste null.",
        "Table mesurée sur un seul régime de marché ; à répliquer avant usage en dur.",
        "Distance mesurée au centroïde dans l'étude ; distance_atr_edge est fourni "
        "en complément mais n'a PAS été validé par le walk-forward.",
    ],
}


def _reach_probability(distance_atr: Optional[float], nb_tf: int) -> Optional[float]:
    """Taux de retouche OBSERVÉ pour cette tranche de distance (§5.6).

    Retourne une fréquence empirique par tranche, sans interpolation : interpoler
    entre deux tranches mesurées produirait un chiffre qui n'a jamais été observé.
    """
    if distance_atr is None or not np.isfinite(distance_atr) or distance_atr < 0:
        return None
    table = _REACH_BY_DIST_ATR_MULTI_TF if nb_tf >= 2 else _REACH_BY_DIST_ATR_MONO_TF
    for upper, rate, _n in table:
        if distance_atr < upper:
            return rate
    return table[-1][1]


def _reach_bucket_label(distance_atr: Optional[float]) -> Optional[str]:
    """Libellé de la tranche de distance utilisée (traçabilité pour le merger)."""
    if distance_atr is None or not np.isfinite(distance_atr):
        return None
    for upper, label in (
        (0.5, "<0.5 ATR"), (1.0, "0.5-1 ATR"), (2.0, "1-2 ATR"),
        (4.0, "2-4 ATR"), (8.0, "4-8 ATR"),
    ):
        if distance_atr < upper:
            return label
    return ">8 ATR"


def _score_and_classify_group(
    group: pd.DataFrame,
    current_price: float,
    symbol: str,
    full_cluster: Optional[pd.DataFrame] = None,
) -> dict:
    """Score et classification d'un groupe de confluences.

    PATCH JSON-2 : enrichit la sortie pour le schéma v2 (zone_id, per_timeframe,
    zone_bounds, distance_atr, confidence_tier). AUCUN champ historique n'est
    retiré : les nouveaux champs s'AJOUTENT (rétro-compatibilité stricte).

    PATCH BOUNDS-1 — sémantique de `zone_bounds` CORRIGÉE.
      AVANT : bounds = min/max de `full_cluster` (la composante union-find
      ENTIÈRE), alors que `Niveau`, `Score`, `Force Totale`, `Distance %` et
      `per_timeframe` sont calculés sur `group` (un niveau par TF après
      groupby.idxmin). Les deux ne décrivaient pas le même objet : sur l'export
      v8.8.0, AUD/CHF 0.54019 annonçait des bornes larges de 9.4 % alors que ses
      trois niveaux scorés ne s'étalent que sur 1.8 %. Un merger lisant
      `zone_bounds` comme bande tradable recevait un couloir 5x trop large.
      APRÈS : `zone_bounds` = étendue des niveaux SCORÉS (cohérent avec Niveau) ;
      l'étendue de la composante complète est exposée à part dans `cluster_bounds`,
      avec son compte de membres, pour l'audit.

    PATCH DIST-1 — `distance_atr_edge` : distance au BORD le plus proche de la
      zone, en plus de la distance au centroïde. Le centroïde est ce que le
      walk-forward a mesuré (donc `distance_atr` est conservé tel quel et reste la
      seule variable validée) ; le bord est ce qu'un filtre d'actionnabilité doit
      regarder. Les deux sont fournis, explicitement étiquetés.

    PATCH REACH-1 — `reach_probability` : fréquence de retouche OBSERVÉE pour la
      tranche de distance de la zone (rapport §5.6). C'est une PORTÉE, pas une
      qualité. `confidence_tier` reste null : le rapport démontre (§5.7, §10.7)
      qu'aucune statistique de TENUE par zone n'existe dans les données.
    """
    sub_avg = group["level"].mean()
    sub_nb_tf = group["tf"].nunique()
    safe_cp = current_price if current_price and current_price > 0 else 1.0
    sub_dist = abs(safe_cp - sub_avg) / safe_cp * 100
    score = _group_score(group, sub_nb_tf)
    status = max(group["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 1))
    ctype, sig = _classify_confluence_type(group, safe_cp, sub_dist)
    force_totale = int(group["strength"].sum())
    nb_tf = int(sub_nb_tf)
    dist_pct = round(sub_dist, 3)

    # --- Enrichissements v2 ---
    level_round = round(float(sub_avg), 5)

    # PATCH BOUNDS-1 : bornes = niveaux SCORÉS (group), pas la composante entière.
    bounds_low = round(float(group["level"].min()), 5)
    bounds_high = round(float(group["level"].max()), 5)
    zone_width_pct = round((bounds_high - bounds_low) / safe_cp * 100.0, 4)

    cluster = full_cluster if full_cluster is not None else group
    cluster_low = round(float(cluster["level"].min()), 5)
    cluster_high = round(float(cluster["level"].max()), 5)
    cluster_bounds = {
        "low": cluster_low,
        "high": cluster_high,
        "members": int(len(cluster)),
        "width_pct": round((cluster_high - cluster_low) / safe_cp * 100.0, 4),
        "note": "Étendue de la composante de clustering complète. NON scorée : "
                "Niveau/Score/per_timeframe portent sur zone_bounds.",
    }

    # ATR médian des TF représentées, utilisé pour les deux distances ATR.
    atr_med: Optional[float] = None
    if "atr" in group.columns:
        atrs = pd.to_numeric(group["atr"], errors="coerce").dropna()
        if len(atrs) > 0 and float(atrs.median()) > 0:
            atr_med = float(atrs.median())

    distance_atr = None
    distance_atr_edge = None
    if atr_med:
        distance_atr = round(abs(safe_cp - level_round) / atr_med, 3)
        if safe_cp < bounds_low:
            edge_gap = bounds_low - safe_cp
        elif safe_cp > bounds_high:
            edge_gap = safe_cp - bounds_high
        else:
            edge_gap = 0.0  # prix DANS la zone
        distance_atr_edge = round(edge_gap / atr_med, 3)

    reach_p = _reach_probability(distance_atr, nb_tf)
    expected_hold = (
        round(reach_p * _RESPECT_GIVEN_TOUCH_GLOBAL, 4) if reach_p is not None else None
    )

    return {
        # --- Champs originaux (ne pas modifier : compatibilité dashboards) ---
        "Actif": symbol,
        "Signal": sig,
        "Niveau": level_round,
        "Type": ctype,
        "Timeframes": " + ".join(sorted(group["tf"].unique())),
        "Nb TF": nb_tf,
        "Force Totale": force_totale,
        "Score": round(score, 1),
        "Statut": status,
        "Distance %": dist_pct,
        "Alerte": "🔥 ZONE CHAUDE" if sub_dist < 0.5 else ("⚠️ Proche" if sub_dist < 1.5 else ""),
        # --- Champs normalisés snake_case (SR-2) ---
        "zone_type": ctype,
        "zone_signal": sig,
        "zone_strength": force_totale,
        "zone_tf_count": nb_tf,
        "zone_distance_pct": dist_pct,
        # --- Biais directionnel Pivot (SR-3) : à calculer par le merger ---
        "pivot_bias": None,
        # --- Enrichissements schéma v2.0 ---
        "zone_id": compute_zone_id(symbol, level_round, ctype),
        "distance_atr": distance_atr,
        "zone_bounds": {"low": bounds_low, "high": bounds_high},
        "per_timeframe": _zone_per_timeframe(group),
        "confidence_tier": None,
        # --- Enrichissements v2.1 (PATCH BOUNDS-1 / DIST-1 / REACH-1) ---
        "zone_width_pct": zone_width_pct,
        "cluster_bounds": cluster_bounds,
        "atr_reference": round(atr_med, 8) if atr_med else None,
        "distance_atr_edge": distance_atr_edge,
        "price_inside_zone": bool(bounds_low <= safe_cp <= bounds_high),
        "reach_probability": reach_p,
        "reach_distance_bucket": _reach_bucket_label(distance_atr),
        "reach_basis": "empirical_frequency_§5.6" if reach_p is not None else None,
        "respect_given_touch_global": _RESPECT_GIVEN_TOUCH_GLOBAL,
        "expected_hold_probability": expected_hold,
        "score_is_predictive_here": (
            bool(distance_atr is not None and 4.0 <= distance_atr < 8.0)
        ),
    }


class _UnionFind:
    """Union-Find (disjoint set) pour le clustering 1D des confluences."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Trouve la racine avec compression de chemin."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        """Fusionne deux ensembles par rang."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


# PATCH WIDTH-1 — plafond de largeur des composantes de confluence.
# Le clustering mono-TF est doublement borné (span > 2.5*bandwidth dans
# agglomerative_1d_clustering, ET max_cluster_width_pct dans _swing_clustered_zones).
# La confluence multi-TF, elle, ne testait que la distance de proche en proche puis
# fusionnait la composante entière : une chaîne de N niveaux espacés de moins que
# `threshold` produisait un cluster de largeur illimitée. Effet mesuré sur l'export
# v8.8.0 du 2026-09-12 : AUD/CHF niveau 0.54019 -> zone_bounds 0.51546-0.56630,
# soit 9.4 % de largeur pour un seuil de 0.8 % ; AUD/CAD 0.90923 -> ~4.9 % ;
# AUD/JPY 97.50472 -> ~5.4 %.
# Le multiplicateur reprend la borne déjà utilisée en mono-TF (2.5x).
_CONFLUENCE_MAX_SPAN_MULT: Final[float] = 2.5


def _cluster_levels_union_find(
    levels_arr: np.ndarray,
    threshold: float,
    max_span_mult: float = _CONFLUENCE_MAX_SPAN_MULT,
) -> Dict[int, List[int]]:
    """Regroupe les indices de niveaux proches (balayage trié, largeur bornée).

    PRÉCONDITION (inchangée) : ``levels_arr`` trié par ordre croissant —
    ``detect_confluences`` trie avant l'appel.

    Critère de rattachement : gap au voisin <= ``threshold`` (%) ET étendue
    cumulée depuis le premier membre du cluster <= ``threshold * max_span_mult``.
    Le premier critère est celui de la v8.8.0 ; le second est le plafond ajouté.
    Le nom et le contrat de retour sont conservés (composante -> indices).
    """
    n = len(levels_arr)
    comp_map: Dict[int, List[int]] = {}
    if n == 0:
        return comp_map
    if not np.isfinite(threshold) or threshold <= 0:
        return {i: [i] for i in range(n)}

    span_cap = float(threshold) * float(max_span_mult)
    root = 0
    current: List[int] = [0]

    for i in range(1, n):
        prev = float(levels_arr[i - 1])
        first = float(levels_arr[current[0]])
        cur = float(levels_arr[i])

        gap_ok = prev > 0 and ((cur - prev) / prev * 100.0) <= threshold
        span_ok = first > 0 and ((cur - first) / first * 100.0) <= span_cap

        if gap_ok and span_ok:
            current.append(i)
        else:
            comp_map[root] = current
            root = i
            current = [i]

    comp_map[root] = current
    return comp_map


def resolve_confluence_threshold(
    profile: InstrumentProfile, ui_threshold: Optional[float]
) -> float:
    """PATCH THRESH-1 (point 4) — rend ``confluence_threshold_pct`` réellement effectif.

    AVANT : ``_compute_all_confluences`` passait TOUJOURS un seuil explicite
    (``_SYMBOL_CONFLUENCE_THRESHOLD`` ou le slider UI), donc le fallback interne
    sur ``profile.confluence_threshold_pct`` n'était JAMAIS atteint : le
    calibrage "par instrument" annoncé par le code était illusoire.

    APRÈS : le champ devient AUTHORITATIF quand il est renseigné (non None).
    ``InstrumentProfile.confluence_threshold_pct`` passe à ``Optional[float]``
    (défaut None). Ordre de priorité :
      1. valeur explicite du profil (calibrée par instrument) ;
      2. sinon, seuil UI (comportement historique pour les instruments non
         calibrés explicitement — zéro régression).
    """
    if profile.confluence_threshold_pct is not None:
        return float(profile.confluence_threshold_pct)
    if ui_threshold is not None:
        return float(ui_threshold)
    return 1.0


def detect_confluences(
    symbol: str,
    zones_dict: dict,
    current_price: float,
    confluence_threshold_pct: Optional[float] = None,
    atr_map: Optional[dict] = None,
) -> list:
    """Détecte les confluences multi‑timeframes.

    PATCH AGE-1 : le paramètre ``bars_map`` a été retiré (la normalisation de
    l'âge n'en dépend plus). PATCH THRESH-1 : le seuil est résolu par
    ``resolve_confluence_threshold`` (profil autoritaire, sinon UI).
    PATCH JSON-2 : ``atr_map`` (TF -> ATR) alimente ``distance_atr``.
    """
    if not zones_dict or not current_price or current_price <= 0 or not np.isfinite(current_price):
        return []
    z_df = _flatten_zones_to_dataframe(zones_dict, atr_map=atr_map)
    if z_df.empty:
        return []
    profile = get_profile(symbol.replace("/", "_"))
    threshold = resolve_confluence_threshold(profile, confluence_threshold_pct)
    z_df = z_df.sort_values("level").reset_index(drop=True)
    levels_arr = z_df["level"].values
    comp_map = _cluster_levels_union_find(levels_arr, threshold, _CONFLUENCE_MAX_SPAN_MULT)

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
        cluster_no_dist = group_full.drop(columns=["_dist"])
        confluences.append(
            _score_and_classify_group(
                group_full.loc[keep_idx].drop(columns=["_dist"]),
                current_price,
                symbol,
                full_cluster=cluster_no_dist,
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
    # SR-1: source du prix courant ("live" | "candle_close" | "stale" | None)
    current_price_source: Optional[str] = None


@dataclass(frozen=True)
class _RowContext:
    """Contexte pour la construction des lignes d'affichage.

    PATCH AGE-1 : le champ ``df_len`` a été retiré — il ne servait qu'à
    alimenter la normalisation de l'âge, désormais basée sur une constante de
    référence par TF (plus aucune dépendance à la profondeur de fetch).
    """

    current_price: float
    atr_val: float
    sym_d: str
    tf_name: str
    profile: InstrumentProfile


def _make_row(z: dict, ztype: str, ctx: _RowContext) -> Dict[str, Any]:
    """Crée une ligne formatée pour l'interface."""
    cp = ctx.current_price
    dist = abs(cp - z["level"]) / cp * 100 if cp else 0.0
    dist_atr = (
        f"{round(abs(cp - z['level']) / ctx.atr_val, 1)}x"
        if (ctx.atr_val and ctx.atr_val > 0)
        else "N/A"
    )
    return {
        "Actif": ctx.sym_d,
        "Prix Actuel": f"{cp:.5f}" if cp else "N/A",
        "Type": ztype,
        "Niveau": f"{z['level']:.5f}",
        "Force": f"{z['strength']} touches",
        "Score (1TF)": compute_structural_score(
            z["strength"], 1, ctx.tf_name, z["age_bars"]
        ),
        "Statut": z["status"],
        "Dist. %": f"{dist:.2f}%",
        "Dist. ATR": dist_atr,
        "_dist_num": dist,
        "_in_pdf": dist <= ctx.profile.pdf_max_dist_pct,
    }


async def _fetch_live_prices(client, session, sem, symbols):
    """Récupère les prix live pour tous les symboles.

    Retourne deux dicts : prices {sym: float|None} et price_sources {sym: str|None}.
    price_sources[sym] vaut "live", "stale", ou None (fetch échoué).
    """
    tasks = [client.fetch_price(session, sem, sym) for sym in symbols]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    prices: Dict[str, Optional[float]] = {}
    price_sources: Dict[str, Optional[str]] = {}
    for sym, item in zip(symbols, res):
        if isinstance(item, BaseException):
            prices[sym] = None
            price_sources[sym] = None
        else:
            fetched_sym, price, source = item
            prices[fetched_sym] = price
            price_sources[fetched_sym] = source
    return prices, price_sources


async def _fetch_candles_cube(client, session, sem, symbols):
    """Récupère les chandelles pour toutes les timeframes.

    PATCH DEPTH-1 (point 5) : un `limit` explicite par TF est désormais passé à
    fetch_candles (avant : défaut limit=500 -> aucune TF n'atteignait 3 ans de
    façon homogène, H4 ≈ 83 jours seulement).
    """
    data_cube = {sym: {tf: None for tf in _GRANULARITY_MAP} for sym in symbols}
    tasks = [
        client.fetch_candles(session, sem, sym, tf, _FETCH_LIMIT_BY_TF.get(tf, _FETCH_LIMIT_DEFAULT))
        for sym in symbols
        for tf in _GRANULARITY_MAP
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

    # PATCH RETRY-1 — reprise SÉQUENTIELLE des (sym, tf) en échec.
    # Mesuré en Phase 3 : sous 99 requêtes concurrentes, 7/99 couples échouaient
    # de façon transitoire (AUD/USD H4+Weekly, NZD/USD H4+Daily, EUR/NZD H4,
    # AUD/JPY H4, USD/CAD Weekly, GBP/NZD Weekly, NZD/JPY Daily). Ces échecs
    # rendaient le scan non déterministe. La reprise est un NO-OP quand tout
    # réussit, et `force=True` contourne l'entrée de cache négative.
    failed = [
        (sym, tf)
        for sym in symbols
        for tf in _GRANULARITY_MAP
        if data_cube.get(sym, {}).get(tf) is None
    ]
    if failed:
        _LOG.info("Reprise séquentielle de %d fetch(s) en échec", len(failed))
        for sym, tf in failed:
            try:
                _, _, frame = await client.fetch_candles(
                    session, sem, sym, tf,
                    _FETCH_LIMIT_BY_TF.get(tf, _FETCH_LIMIT_DEFAULT), force=True,
                )
                if frame is not None:
                    data_cube[sym][tf] = frame
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
                continue
    return data_cube


def _nearest_support_label(current_price: float, sup: pd.DataFrame) -> Optional[str]:
    """Construit le libellé du support le plus proche sous le prix."""
    if sup is None or sup.empty:
        return None
    s_near = sup[
        (sup["level"] < current_price)
        & (abs(sup["level"] - current_price) / current_price * 100 <= 5.0)
    ]
    if s_near.empty:
        return None
    n_s = s_near.nlargest(1, "level").iloc[0]
    dist = abs(current_price - n_s["level"]) / current_price * 100
    label = "SUR support" if dist < 0.5 else "S proche"
    return f"{label}: {n_s['level']:.5f} (-{dist:.2f}%)"


def _nearest_resistance_label(current_price: float, res: pd.DataFrame) -> Optional[str]:
    """Construit le libellé de la résistance la plus proche au-dessus du prix."""
    if res is None or res.empty:
        return None
    r_near = res[
        (res["level"] > current_price)
        & (abs(res["level"] - current_price) / current_price * 100 <= 5.0)
    ]
    if r_near.empty:
        return None
    n_r = r_near.nsmallest(1, "level").iloc[0]
    dist = abs(current_price - n_r["level"]) / current_price * 100
    label = "SUR resistance" if dist < 0.5 else "R proche"
    return f"{label}: {n_r['level']:.5f} (+{dist:.2f}%)"


def _build_daily_price_context(
    current_price: float, sup: pd.DataFrame, res: pd.DataFrame
) -> str:
    """Construit le contexte de prix quotidien."""
    parts = []
    s_label = _nearest_support_label(current_price, sup)
    if s_label:
        parts.append(s_label)
    r_label = _nearest_resistance_label(current_price, res)
    if r_label:
        parts.append(r_label)
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


def _debug_pivot_count(ctx: _TFProcessingContext, atr_val: float) -> Optional[int]:
    """Compte les pivots pour le debug (None si erreur)."""
    try:
        return len(
            _get_pivots_with_fallback_meta(ctx.dataframe, ctx.profile, atr_val, ctx.tf_key)
        )
    except (ValueError, TypeError):
        return None


def _debug_trend_zone_count(ctx: _TFProcessingContext, atr_val: float) -> Optional[int]:
    """Compte les zones trend-structure pour le debug (None si erreur, 0 si non applicable)."""
    if ctx.profile.asset_class not in ("INDEX", "METAL"):
        return 0
    try:
        return len(
            _detect_trend_structure_zones(
                ctx.dataframe, ctx.current_price, ctx.profile, atr_val, "Support"
            )
        ) + len(
            _detect_trend_structure_zones(
                ctx.dataframe, ctx.current_price, ctx.profile, atr_val, "Resistance"
            )
        )
    except (ValueError, TypeError):
        return None


def _rows_from_zones(sup: pd.DataFrame, res: pd.DataFrame, row_ctx: _RowContext) -> List[dict]:
    """Construit et dédoublonne les lignes d'affichage à partir des zones."""
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
    return uniq


def _process_tf_frame(ctx: _TFProcessingContext):
    """Traite une timeframe unique et retourne les lignes, zones, contexte prix et debug.

    PATCH PERF-3 — les compteurs de debug ne rejouent plus la détection.
      AVANT : `_debug_pivot_count` et `_debug_trend_zone_count` étaient appelés
      INCONDITIONNELLEMENT, puis `find_strong_sr_zones` recalculait pivots et
      zones trend-structure en interne -> la détection tournait 2 à 3 fois par
      (actif, TF), soit 99 fois par scan, pour alimenter des compteurs que l'UI
      n'affiche que si `show_debug` est coché. Gaté sur DEBUG_INSTRUMENTATION.

    PATCH PERF-2 — la détection travaille sur la fenêtre calibrée (cf.
    _DETECTION_WINDOW_BY_TF), pas sur la profondeur de fetch complète.
    """
    debug = {
        "atr": None,
        "n_pivots": 0,
        "n_zones": 0,
        "n_trend_zones": 0,
        "min_touches": None,
        "tf": ctx.tf_name,
        "instrumented": bool(DEBUG_INSTRUMENTATION),
    }
    try:
        det_df = _detection_frame(ctx.dataframe, ctx.tf_key)
        debug["bars_fetched"] = int(len(ctx.dataframe)) if ctx.dataframe is not None else 0
        debug["bars_detection"] = int(len(det_df)) if det_df is not None else 0

        atr_val = compute_atr(det_df)
        debug["atr"] = atr_val
        if atr_val is None:
            return None, None, "", debug

        min_t = _min_touches_for_tf(ctx.profile, ctx.tf_key, ctx.min_touches_ui)
        debug["min_touches"] = min_t

        det_ctx = replace(ctx, dataframe=det_df) if det_df is not ctx.dataframe else ctx

        # PATCH PERF-3 : coûteux (rejoue la détection) -> uniquement si instrumenté.
        if DEBUG_INSTRUMENTATION:
            debug["n_pivots"] = _debug_pivot_count(det_ctx, atr_val)
            debug["n_trend_zones"] = _debug_trend_zone_count(det_ctx, atr_val)

        sup, res = find_strong_sr_zones(
            det_df, ctx.current_price, ctx.symbol, atr_val, ctx.tf_key, min_t
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
            profile=ctx.profile,
        )
        uniq = _rows_from_zones(sup, res, row_ctx)
        return (uniq if uniq else None), (sup, res), price_ctx, debug
    except (ValueError, KeyError, TypeError) as e:
        _LOG.warning("TF processing error %s/%s: %s", ctx.symbol, ctx.tf_name, type(e).__name__)
        debug["error"] = type(e).__name__
        return None, None, "", debug


def _resolve_working_price(cp_live, data_cube, symbol, cp_live_source: Optional[str] = None):
    """Détermine le prix de travail (live ou fallback).

    Retourne (price, is_fallback, current_price_source) où source vaut :
      - "live"         : prix bid/ask mid OANDA en temps réel (tradeable=True)
      - "stale"        : prix OANDA mais marché fermé (tradeable=False)
      - "candle_close" : fallback sur le dernier close de bougie (live indispo)
    """
    if cp_live and cp_live > 0 and np.isfinite(cp_live):
        # SR-1 FIX: propager la source fournie par fetch_price ("live" ou "stale").
        # Si cp_live_source est None (fetch_price ancien comportement ou test),
        # on considère "live" par défaut.
        source = cp_live_source if cp_live_source in ("live", "stale") else "live"
        return cp_live, False, source
    for tf_k in ("daily", "h4", "weekly"):
        df = data_cube.get(symbol, {}).get(tf_k)
        if df is not None and not df.empty:
            try:
                last_close = float(df["close"].iloc[-1])
                if np.isfinite(last_close) and last_close > 0:
                    return last_close, True, "candle_close"
            except (ValueError, KeyError, IndexError):
                continue
    return None, False, None


# PATCH 2 — Facteur de détection de prix corrompu (méta-audit validé, risque LOW)
# Heuristique conservatrice, non calibrée empiriquement.
# Un prix inférieur à price_min / FACTOR ou supérieur à price_max * FACTOR
# est considéré comme un artefact de données (ex : feed erroné, valeur aberrante).
# Facteur 10 choisi pour ne jamais déclencher sur un mouvement de marché réel,
# même extrême (crash -50%, rally +200% sur indices). Toute révision nécessite
# une validation empirique sur données historiques.
_BOUNDS_CORRUPTION_FACTOR: Final[float] = 10.0


def _validate_price_bounds_post(
    current_price: float, profile: InstrumentProfile
) -> Tuple[Optional[float], Optional[str]]:
    """Vérifie que le prix est dans les bornes autorisées.

    PATCH 2 — méta-audit validé, risque LOW :
    Remplace l'interruption systématique par un warning + clipping sécurisé.
    Si l'écart dépasse _BOUNDS_CORRUPTION_FACTOR × la borne, le prix est
    considéré corrompu et le scan est interrompu (retour None, erreur).
    Retourne (prix_corrigé_ou_None, erreur_ou_None).
    Appelant unique : _process_symbol (vérifié par grep, un seul call-site).
    """
    low  = profile.price_min
    high = profile.price_max

    if low is not None and current_price < low:
        if current_price < low / _BOUNDS_CORRUPTION_FACTOR:
            return None, f"PRIX CORROMPU ({current_price:.2f} << {low:.0f})"
        _LOG.warning(
            "Prix hors borne basse pour %s : %.2f < %.0f — clipping appliqué",
            profile.symbol, current_price, low,
        )
        return low, None

    if high is not None and current_price > high:
        if current_price > high * _BOUNDS_CORRUPTION_FACTOR:
            return None, f"PRIX CORROMPU ({current_price:.2f} >> {high:.0f})"
        _LOG.warning(
            "Prix hors borne haute pour %s : %.2f > %.0f — clipping appliqué",
            profile.symbol, current_price, high,
        )
        return high, None

    return current_price, None


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


def _ratio_anomaly(current_price: float, support_levels: list, skip_ratio_check: bool) -> bool:
    """Vérifie l'anomalie d'écart aberrant entre prix et supports."""
    if skip_ratio_check or len(support_levels) < 3:
        return False
    median_sup = float(np.median(support_levels))
    return median_sup > 0 and current_price / median_sup > 3.0


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
    if _ratio_anomaly(current_price, support_levels, profile.skip_ratio_check):
        msgs.append("Ecart aberrant")
    if last_candle_close and last_candle_close > 0:
        dev = abs(current_price - last_candle_close) / last_candle_close * 100
        # PATCH DEV-1 (point 9) : `dev_threshold_pct` (déclaré et renseigné pour
        # chaque profil mais jusqu'ici JAMAIS lu) est désormais CÂBLÉ comme seuil
        # primaire de déviation live/close. `max_live_vs_close_pct` reste un
        # plafond absolu de sécurité -> seuil effectif = min(dev, max).
        # Zéro régression quand dev_threshold_pct >= max_live_vs_close_pct.
        dev_threshold = min(
            float(profile.dev_threshold_pct or profile.max_live_vs_close_pct),
            float(profile.max_live_vs_close_pct),
        )
        if dev > dev_threshold:
            msgs.append(f"Ecart live/close ({dev:.1f}%)")
    return " | ".join(msgs) if msgs else None


def _collect_support_levels(zones_d: dict) -> list:
    """Agrège tous les niveaux de support détectés."""
    sup_levels: list = []
    for zp in zones_d.values():
        if zp[0] is not None and not zp[0].empty:
            sup_levels.extend(zp[0]["level"].tolist())
    return sup_levels


def _last_daily_close(data_cube: dict, symbol: str) -> Optional[float]:
    """Retourne le dernier close daily disponible (ou None)."""
    daily_df = data_cube.get(symbol, {}).get("daily")
    if daily_df is not None and not daily_df.empty:
        return float(daily_df["close"].iloc[-1])
    return None


def _process_symbol(
    symbol: str,
    cp_live: Optional[float],
    data_cube: dict,
    min_touches_ui: int,
    cp_live_source: Optional[str] = None,
):
    """Traite un symbole complet."""
    try:
        profile = get_profile(symbol)
        sym_d = symbol.replace("_", "/")
        current_price, price_is_fallback, cp_source = _resolve_working_price(
            cp_live, data_cube, symbol, cp_live_source=cp_live_source
        )
        if current_price is None:
            return ScanResult(symbol, {}, {}, None, {}, {}, scan_error="Aucune donnee disponible")
        current_price, bounds_err = _validate_price_bounds_post(current_price, profile)
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
        sup_levels = _collect_support_levels(zones_d)
        last_close = _last_daily_close(data_cube, symbol)
        anomaly = flag_data_anomaly(symbol, current_price, sup_levels, last_candle_close=last_close)
        if price_is_fallback:
            anomaly = f"{anomaly} | Prix fallback" if anomaly else "Prix fallback"
        # PATCH STALE: marché fermé (tradeable=False) — prix potentiellement décalé
        # après gap d'ouverture. Signalé dans l'expander anomalies Streamlit UI.
        if cp_source == "stale":
            anomaly = f"{anomaly} | Prix STALE (marché fermé)" if anomaly else "Prix STALE (marché fermé)"
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
            current_price_source=cp_source,
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


def _patch_coverage_gaps(results: List[ScanResult], coverage: Dict[str, Any]) -> None:
    """Ajoute des résultats d'erreur pour les symboles manquants (violation de couverture)."""
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


async def run_institutional_scan(symbols, token, oanda_account_id, min_touches_ui, oanda_env=None):
    """Exécute le scan institutionnel complet.

    PATCH ENV-1 (point 14) : ``oanda_env`` ("practice" | "trade") force
    l'environnement OANDA et le journalise. None = auto-détection.

    PATCH ENV-2 : l'environnement RÉSOLU est désormais exposé via
    ``run_institutional_scan.last_env`` pour que l'appelant puisse le threader
    jusqu'à create_json_export. Sur l'export v8.8.0, `oanda_environment` valait
    null parce que l'UI lisait os.environ["OANDA_ENV"] (non positionné) au lieu
    de l'environnement effectivement résolu par le client.
    """
    client = AsyncOandaClient(token, oanda_account_id)
    run_institutional_scan.last_env = None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=10)
    ) as session:
        if not await client.initialize(session, env_override=oanda_env):
            raise OandaAuthError("Auth OANDA echouee")
        run_institutional_scan.last_env = client.env_name
        sem = asyncio.Semaphore(_OANDA_SEMAPHORE_LIMIT)
        live_prices, price_sources = await _fetch_live_prices(client, session, sem, symbols)
        data_cube = await _fetch_candles_cube(client, session, sem, symbols)

    results: List[ScanResult] = []
    for sym in symbols:
        res = _process_symbol(
            sym, live_prices.get(sym), data_cube, min_touches_ui,
            cp_live_source=price_sources.get(sym),
        )
        results.append(res)

    coverage = _validate_symbol_coverage(symbols, results)
    if not coverage["ok"]:
        _patch_coverage_gaps(results, coverage)
    return results


run_institutional_scan.last_env = None


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
        """En-tête de page (titre + horodatage)."""
        self.set_font("Helvetica", "B", 15)
        self.cell(
            0,
            10,
            _safe_pdf_str("Rapport Scanner Bluestar - S/R"),
            border=0,
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
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
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(4)

    def footer(self):
        """Pied de page (numéro de page)."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")

    def chapter_title(self, title):
        """Titre de chapitre."""
        self.set_font("Helvetica", "B", 12)
        self.cell(
            0,
            10,
            _safe_pdf_str(title),
            border=0,
            align="L",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(4)

    def _column_widths(self, df):
        """Retourne la table des largeurs de colonnes selon le type de DataFrame."""
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

    def _render_header_row(self, cols, col_widths, x_start):
        """Dessine la ligne d'en-tête du tableau."""
        self.set_font("Helvetica", "B", 7)
        self.set_x(x_start)
        for col in cols:
            self.cell(
                col_widths[col],
                6,
                _safe_pdf_str(col),
                border=1,
                align="C",
                new_x=XPos.RIGHT,
                new_y=YPos.TOP,
            )
        self.ln()

    def _render_data_rows(self, df, cols, col_widths, x_start):
        """Dessine les lignes de données du tableau."""
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
                    new_x=XPos.RIGHT,
                    new_y=YPos.TOP,
                )
            self.ln()

    def chapter_body(self, df):
        """Corps de chapitre : rend un DataFrame sous forme de tableau."""
        if df is None or df.empty:
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 10, "Aucune donnee a afficher.")
            return
        col_widths = self._column_widths(df)
        cols = [c for c in col_widths if c in df.columns]
        total_w = sum(col_widths[c] for c in cols)
        x_start = self.l_margin + max(0, (self.w - self.l_margin - self.r_margin - total_w) / 2)
        self._render_header_row(cols, col_widths, x_start)
        self._render_data_rows(df, cols, col_widths, x_start)


def _pdf_render_anomalies(pdf: "PDF", anomalies: dict) -> None:
    """Rend la section des anomalies dans le PDF."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(
        0,
        7,
        _safe_pdf_str("ALERTES ANOMALIES"),
        border=0,
        align="L",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 8)
    for sym, msg in anomalies.items():
        pdf.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}"))
    pdf.ln(4)


def create_pdf_report(results_dict, confluences_df=None, summary_list=None, anomalies=None):
    """Génère un rapport PDF complet.

    Note: ``summary_list`` fait partie de la signature publique d'export (parité
    avec ``create_json_export`` / ``create_llm_brief``) et est conservé même s'il
    n'est pas consommé dans le rendu PDF actuel.
    """
    _ = summary_list  # paramètre d'API conservé volontairement (parité signatures)
    pdf = PDF("L", "mm", "A4")
    pdf.set_margins(5, 10, 5)
    pdf.add_page()
    if anomalies:
        _pdf_render_anomalies(pdf, anomalies)
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


def _filter_confluences(
    confluences_df: Optional[pd.DataFrame],
    max_dist: Optional[float],
    min_score: float,
    allowed_statuts: tuple,
) -> pd.DataFrame:
    """Filtre les confluences par distance / score / statut.

    PATCH JSON-1 (point 2) : ``max_dist=None`` désactive complètement le filtre
    de distance (utile pour l'export JSON destiné à une machine). Un statut
    vide/None désactive le filtre de statut.
    """
    if confluences_df is None or confluences_df.empty:
        return pd.DataFrame()
    df = confluences_df.copy()
    df["dist_num"] = pd.to_numeric(
        df["Distance %"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(999999.0)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)
    mask = df["Score"] >= float(min_score)
    if max_dist is not None:
        mask &= df["dist_num"] <= float(max_dist)
    if allowed_statuts:
        mask &= df["Statut"].isin(tuple(allowed_statuts))
    return df[mask].drop(columns=["dist_num"], errors="ignore")


_BARS_PER_YEAR: Final[Dict[str, int]] = {"H4": 1560, "Daily": 260, "Weekly": 52}


def _data_window_from_bars(bars_map: Optional[dict]) -> dict:
    """Fenêtre de données réellement utilisée par TF (couverture minimale garantie)."""
    out: Dict[str, Any] = {}
    for tf_name in ("H4", "Daily", "Weekly"):
        counts = []
        for b in (bars_map or {}).values():
            if isinstance(b, dict) and b.get(tf_name):
                counts.append(int(b[tf_name]))
        if counts:
            n = min(counts)  # couverture minimale garantie sur tous les actifs
            years = n / _BARS_PER_YEAR.get(tf_name, 260)
            out[tf_name] = {
                "candles_used": n,
                "coverage_years": round(years, 2),
                "coverage_ok_3y": years >= 3.0,
            }
        else:
            out[tf_name] = {"candles_used": 0, "coverage_years": 0.0, "coverage_ok_3y": False}
    return out


def create_json_export(
    summary_list,
    confluences_df,
    max_dist=None,
    min_score=0.0,
    allowed_statuts=("Vierge", "Testee", "Role Reverse"),
    *,
    scan_errors=None,
    missing_tfs_map=None,
    anomalies=None,
    bars_map=None,
    data_window=None,
    oanda_environment=None,
    calibration_profile_version=None,
):
    """Exporte les résultats au format JSON — SCHÉMA v2.1.

    v2.1 = sur-ensemble STRICT de v2.0 (aucune clé retirée). Changements :

    PATCH TS-1 — `current_price_timestamp` ne mentait plus.
      AVANT : `now_utc` dès que la source était "live" OU "stale". Sur l'export
      v8.8.0 du 2026-09-12 (samedi), les 33 actifs portaient un horodatage de
      samedi 19:46 UTC pour un prix de clôture de vendredi. Le commentaire du code
      refusait d'"inventer" l'heure pour le repli candle_close (-> null) mais
      affirmait une fraîcheur fausse pour stale.
      APRÈS : horodatage rempli SEULEMENT si "live". Pour "stale", null + le
      booléen explicite `price_is_stale` et `price_timestamp_note`. La CLÉ est
      conservée (rétro-compat) ; seule la valeur devient honnête.

    PATCH ENV-2 — `oanda_environment` n'est plus silencieusement null.
      Sur l'export v8.8.0 le champ valait null alors que le rapport annonçait
      "practice (forcé et journalisé)" : la variable d'env n'était pas positionnée
      et l'appelant ne threadait pas l'environnement RÉSOLU par le client. On
      journalise désormais un avertissement au lieu d'exporter un null muet.

    PATCH RUNID-1 — `run_id` réellement stable.
      AVANT : le hash portait sur `assets`, qui contenait `current_price_timestamp`
      = now -> le run_id changeait à chaque run même à données identiques, ce qui
      contredisait le commentaire "stable entre runs identiques".
      APRÈS : les champs volatils sont exclus du périmètre de hash.

    PATCH CTX-1 — `price_context` réconcilié avec les zones exportées.
      `_nearest_support_label` travaille sur les niveaux Daily BRUTS, pas sur les
      confluences fusionnées : le contexte pouvait annoncer "SUR resistance
      0.58640" alors que la plus proche confluence exportée était 0.58863. Le
      merger lisait un niveau sans zone_id, sans statut, sans score. On ajoute
      `nearest_zones` dérivé des zones RÉELLEMENT exportées, et on marque
      `price_context` comme indicatif.
    """
    now_utc = datetime.now(timezone.utc)
    scan_errors = scan_errors or {}
    missing_tfs_map = missing_tfs_map or {}
    anomalies = anomalies or {}

    if not oanda_environment:
        _LOG.warning(
            "PATCH ENV-2 : oanda_environment non fourni -> exporté à null. "
            "Passer l'environnement RÉSOLU par AsyncOandaClient (client.env_name) "
            "ou définir OANDA_ENV pour rendre le run traçable."
        )

    resolved_window = data_window if data_window is not None else _data_window_from_bars(bars_map)

    output: Dict[str, Any] = {
        "schema_version": "2.1",
        "generated_at": now_utc.isoformat(),
        "scanner_version": SCANNER_VERSION,
        "oanda_environment": oanda_environment,
        "data_window": resolved_window,
        "detection_window_by_tf": {
            "H4": _DETECTION_WINDOW_BY_TF.get("h4"),
            "Daily": _DETECTION_WINDOW_BY_TF.get("daily"),
            "Weekly": _DETECTION_WINDOW_BY_TF.get("weekly"),
            "note": "Profondeur de DÉTECTION (null = toute la profondeur de fetch). "
                    "data_window décrit la profondeur FETCHÉE. Le walk-forward de "
                    "calibration a été mené sur 500 bougies (rapport §5.2).",
        },
        "calibration_profile_version": calibration_profile_version,
        "calibration_evidence": _CALIBRATION_EVIDENCE,
        "confluence_width_cap": {
            "max_span_mult": _CONFLUENCE_MAX_SPAN_MULT,
            "note": "PATCH WIDTH-1 : une composante de confluence est bornée à "
                    "threshold * max_span_mult en largeur. Sans ce plafond, le "
                    "chaînage transitif produisait des clusters de 5 à 10 % de "
                    "large pour un seuil de 0.8 %.",
        },
        "export_filters_applied": {
            "max_dist_pct": max_dist,
            "min_score": float(min_score),
            "allowed_statuts": list(allowed_statuts) if allowed_statuts else None,
            "note": "JSON non pré-filtré par distance par défaut (max_dist=None) ; "
                    "le merger applique ses propres critères.",
            "consommee_unreachable": True,
            "consommee_note": "Le statut 'Consommee' est INSATISFIABLE par "
                              "construction : les zones Consommee sont exclues avant "
                              "fusion dans _build_zones_dataframe puis re-filtrées "
                              "dans _flatten_one_tf. Aucune zone Consommee ne peut "
                              "apparaître dans cet export, quel que soit ce filtre.",
        },
        "merger_contract": {
            "recommended_distance_field": "distance_atr",
            "recommended_distance_note": "distance_atr (centroïde) est la SEULE "
                "variable de distance validée par le walk-forward. "
                "distance_atr_edge est fourni en complément mais n'a pas été validé.",
            "recommended_filter": "distance_atr <= 4.0 (au-delà de 4 ATR le taux de "
                "retouche observé tombe à 0.438 puis 0.051 ; cf. §5.4/§5.6). "
                "Le rapport recommande 2-4 ATR selon l'appétit de précision (§12).",
            "do_not_use_as_quality": [
                "Score (non prédictif de la tenue ; discriminant seulement en 4-8 ATR)",
                "Nb TF / zone_tf_count (aucun avantage à distance comparable, §5.6)",
                "Statut (artefact de composition en distance, §5.9b)",
                "confidence_tier (toujours null, par choix : §5.7)",
            ],
            "zone_bounds_semantics": "Étendue des niveaux SCORÉS (un par TF). "
                "cluster_bounds décrit la composante de clustering complète, non scorée.",
            "pivot_bias": "Toujours null : à calculer par le merger en croisant "
                "trend_daily/trend_h4/trend_weekly avec le biais GPS externe.",
            "price_context": "Champ TEXTE indicatif, dérivé des niveaux Daily BRUTS. "
                "Ne correspond PAS forcément à une zone de `zones`. Utiliser "
                "`nearest_zones` pour un lien fiable par zone_id.",
        },
        "assets": [],
    }

    summary_map = {
        s["symbol"]: s for s in summary_list if isinstance(s, dict) and "symbol" in s
    }
    filtered_conf = _filter_confluences(confluences_df, max_dist, min_score, allowed_statuts)

    assets_with_no_zones: List[str] = []
    assets_with_scan_error: List[str] = []
    assets_with_stale_price: List[str] = []
    assets_with_missing_tf: List[str] = []
    total_zones = 0

    for sym, summary in summary_map.items():
        sym_clean = sym.replace("/", "_")
        profile = get_profile(sym_clean)

        if not filtered_conf.empty and "Actif" in filtered_conf.columns:
            zones = filtered_conf[filtered_conf["Actif"] == sym].to_dict("records")
        else:
            zones = []

        cp = summary.get("current_price")

        # Tri par actionnabilité : distance ATR croissante (variable validée),
        # repli sur la distance en % quand l'ATR n'est pas disponible.
        def _sort_key(z):
            d_atr = z.get("distance_atr")
            if d_atr is None or not isinstance(d_atr, (int, float)):
                return (1, float(z.get("Distance %") or 1e9))
            return (0, float(d_atr))

        zones.sort(key=_sort_key)
        total_zones += len(zones)

        scan_error = scan_errors.get(sym)
        missing = list(missing_tfs_map.get(sym, []) or [])
        anomaly = anomalies.get(sym)
        price_source = summary.get("current_price_source")
        is_stale = price_source == "stale"

        if scan_error:
            zones_scan_status = "scan_error"
            assets_with_scan_error.append(sym)
        elif missing and not zones:
            zones_scan_status = "missing_data"
        elif zones:
            zones_scan_status = "found"
        else:
            zones_scan_status = "none_detected"
            assets_with_no_zones.append(sym)
        if missing:
            assets_with_missing_tf.append(sym)
        if is_stale:
            assets_with_stale_price.append(sym)

        # PATCH CTX-1 : plus proche support / résistance parmi les zones EXPORTÉES.
        nearest_zones: Dict[str, Any] = {"support": None, "resistance": None}
        if cp:
            below = [z for z in zones if (z.get("Niveau") or 0) < cp]
            above = [z for z in zones if (z.get("Niveau") or 0) >= cp]
            if below:
                z = max(below, key=lambda x: x["Niveau"])
                nearest_zones["support"] = {
                    "zone_id": z.get("zone_id"), "Niveau": z.get("Niveau"),
                    "distance_atr": z.get("distance_atr"),
                    "distance_pct": z.get("Distance %"), "Statut": z.get("Statut"),
                    "reach_probability": z.get("reach_probability"),
                }
            if above:
                z = min(above, key=lambda x: x["Niveau"])
                nearest_zones["resistance"] = {
                    "zone_id": z.get("zone_id"), "Niveau": z.get("Niveau"),
                    "distance_atr": z.get("distance_atr"),
                    "distance_pct": z.get("Distance %"), "Statut": z.get("Statut"),
                    "reach_probability": z.get("reach_probability"),
                }

        output["assets"].append(
            {
                # --- champs historiques (inchangés) ---
                "symbol": sym,
                "current_price": round(cp, 5) if cp is not None else None,
                "current_price_source": price_source,
                "zones_scan_status": zones_scan_status,
                "zones": zones,
                # --- enrichissements v2 ---
                "asset_class": profile.asset_class,
                "pip_value": profile.pip_value,
                # PATCH TS-1 : rempli seulement si le prix est réellement live.
                "current_price_timestamp": now_utc.isoformat() if price_source == "live" else None,
                "trend_h4": summary.get("trend_h4"),
                "trend_daily": summary.get("trend_daily"),
                "trend_weekly": summary.get("trend_weekly"),
                "price_context": summary.get("price_context", ""),
                "scan_error": scan_error,
                "missing_timeframes": missing,
                "anomalies": [anomaly] if anomaly else [],
                # --- enrichissements v2.1 ---
                "price_is_stale": is_stale,
                "price_timestamp_note": (
                    "Marché fermé (OANDA tradeable=False) : dernier mid connu, "
                    "horodatage réel non threadé -> null plutôt qu'inventé. "
                    "Toutes les distances de cet actif sont calculées sur ce prix."
                    if is_stale else None
                ),
                "zone_count": len(zones),
                "nearest_zones": nearest_zones,
                "actionable_zone_count": sum(
                    1 for z in zones
                    if isinstance(z.get("distance_atr"), (int, float)) and z["distance_atr"] <= 4.0
                ),
            }
        )

    output["assets"].sort(key=lambda x: x["symbol"])

    # PATCH RUNID-1 : hash déterministe sur le CONTENU non volatil.
    _volatile = {"current_price_timestamp", "price_timestamp_note"}
    hash_payload = [
        {k: v for k, v in a.items() if k not in _volatile} for a in output["assets"]
    ]
    output["run_id"] = hashlib.sha1(
        json.dumps(hash_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]

    output["diagnostics"] = {
        "assets_with_no_zones": sorted(assets_with_no_zones),
        "assets_with_scan_error": sorted(assets_with_scan_error),
        "assets_with_stale_price": sorted(assets_with_stale_price),
        "assets_with_missing_tf": sorted(assets_with_missing_tf),
        "coverage_summary": {tf: v for tf, v in resolved_window.items()},
        "total_assets": len(output["assets"]),
        "total_zones_exported": total_zones,
        "all_prices_stale": (
            len(assets_with_stale_price) == len(output["assets"]) and bool(output["assets"])
        ),
    }

    ordered = {
        "schema_version": output["schema_version"],
        "run_id": output["run_id"],
        "generated_at": output["generated_at"],
        "scanner_version": output["scanner_version"],
        "oanda_environment": output["oanda_environment"],
        "data_window": output["data_window"],
        "detection_window_by_tf": output["detection_window_by_tf"],
        "calibration_profile_version": output["calibration_profile_version"],
        "calibration_evidence": output["calibration_evidence"],
        "confluence_width_cap": output["confluence_width_cap"],
        "export_filters_applied": output["export_filters_applied"],
        "merger_contract": output["merger_contract"],
        "assets": output["assets"],
        "diagnostics": output["diagnostics"],
    }
    return json.dumps(ordered, indent=2, ensure_ascii=False).encode("utf-8")


def create_llm_brief(
    summary_list,
    confluences_df,
    max_dist=2.0,
    min_score=57.0,  # PATCH SCORE-1 : recalibré depuis 100.0 (échelle sqrt(nb_tf))
    allowed_statuts=("Vierge", "Testee", "Role Reverse"),
):
    """Crée un résumé textuel pour LLM.

    Note: ``summary_list`` est conservé pour la parité de signature des exports
    (``create_json_export`` / ``create_pdf_report``).
    """
    _ = summary_list  # paramètre d'API conservé volontairement (parité signatures)
    lines = [
        "# BRIEF S/R — Scanner Bluestar",
        f"_Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}_",
        "",
    ]
    if confluences_df is None or confluences_df.empty:
        return "\n".join(lines).encode("utf-8")
    filtered = _filter_confluences(confluences_df, max_dist, min_score, allowed_statuts)
    if filtered.empty:
        return "\n".join(lines).encode("utf-8")
    for sym in filtered["Actif"].unique():
        lines.append(f"### {sym}")
        for _, row in filtered[filtered["Actif"] == sym].iterrows():
            lines.append(
                f"- {row['Signal']} `{row['Niveau']}` | Sc:{row['Score']} | "
                f"{row['Statut']} | {row['Distance %']} | {row['Timeframes']}"
            )
        lines.append("")
    return "\n".join(lines).encode("utf-8")



def _accumulate_scan_results(raw_results, progress_bar):
    """Agrège les ScanResult bruts en structures d'affichage et d'export."""
    results_h4, results_daily, results_weekly = [], [], []
    all_zones_map, prices_map, trends_map = {}, {}, {}
    anomalies_map, scan_errors, bars_map = {}, {}, {}
    missing_tfs_map, price_fallback_map, debug_map = {}, {}, {}
    price_sources_map: Dict[str, Optional[str]] = {}

    total = len(raw_results)
    for idx, res in enumerate(raw_results):
        progress_bar.progress((idx + 1) / total, text=f"Processing {res.symbol}...")
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
        # SR-1: collecter la source du prix courant pour chaque symbole
        price_sources_map[res.symbol] = res.current_price_source
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
    return {
        "results_h4": results_h4,
        "results_daily": results_daily,
        "results_weekly": results_weekly,
        "all_zones_map": all_zones_map,
        "prices_map": prices_map,
        "trends_map": trends_map,
        "anomalies_map": anomalies_map,
        "scan_errors": scan_errors,
        "bars_map": bars_map,
        "missing_tfs_map": missing_tfs_map,
        "debug_map": debug_map,
        "price_sources_map": price_sources_map,
    }


# PATCH THRESH-1 (point 4) : l'ancienne table `_SYMBOL_CONFLUENCE_THRESHOLD` a été
# SUPPRIMÉE. Ses 5 valeurs (US30 1.5, NAS100 1.5, SPX500 1.2, XAU 1.5, DE30 1.3)
# ont été migrées dans `InstrumentProfile.confluence_threshold_pct`, qui est
# désormais AUTHORITAIRE via `resolve_confluence_threshold()`. Pour les
# instruments sans valeur explicite, le seuil UI reste utilisé (zéro régression).


def _atr_map_for_symbol(agg: dict, sym_d: str) -> dict:
    """PATCH JSON-2 : ATR par TF (TF -> ATR) depuis debug_map, pour distance_atr."""
    dbg = (agg.get("debug_map") or {}).get(sym_d, {}) or {}
    out = {}
    for tf_name, d in dbg.items():
        if isinstance(d, dict) and d.get("atr"):
            out[tf_name] = float(d["atr"])
    return out


def _compute_all_confluences(symbols_to_scan, agg, confluence_threshold):
    """Calcule les confluences pour tous les symboles scannés."""
    all_confs = []
    for sym in symbols_to_scan:
        sym_d = sym.replace("_", "/")
        if sym_d in agg["scan_errors"]:
            continue
        all_confs.extend(
            detect_confluences(
                sym_d,
                agg["all_zones_map"].get(sym, {}),
                agg["prices_map"].get(sym),
                confluence_threshold,
                atr_map=_atr_map_for_symbol(agg, sym_d),
            )
        )
    return pd.DataFrame(all_confs) if all_confs else pd.DataFrame()


def _build_summaries(symbols_to_scan, agg):
    """Construit la liste de résumés par symbole."""
    summaries = []
    for sym in symbols_to_scan:
        cp = agg["prices_map"].get(sym)
        ctx = ""
        zones_for_sym = agg["all_zones_map"].get(sym, {})
        if "Daily" in zones_for_sym and cp:
            ctx = _build_daily_price_context(
                cp, zones_for_sym["Daily"][0], zones_for_sym["Daily"][1]
            )
        summaries.append(
            {
                "symbol": sym.replace("_", "/"),
                "trend_h4": agg["trends_map"].get(sym, {}).get("H4", "NEUTRE"),
                "trend_daily": agg["trends_map"].get(sym, {}).get("Daily", "NEUTRE"),
                "trend_weekly": agg["trends_map"].get(sym, {}).get("Weekly", "NEUTRE"),
                "price_context": ctx,
                "current_price": cp,
                # SR-1: source de traçabilité du prix courant pour le JSON downstream
                "current_price_source": agg.get("price_sources_map", {}).get(sym),
            }
        )
    return summaries


def build_class_a_metrics(
    symbols_to_scan: List[str],
    agg: Dict[str, Any],
    conf_full: pd.DataFrame,
    json_args: Tuple[float, float, tuple],
    llm_args: Tuple[float, float, tuple],
) -> Optional[Dict[str, Any]]:
    """Agrège les sept métriques de classe A depuis des sorties déjà calculées.

    Aucune détection : lit debug_map / conf_full et rejoue _filter_confluences
    (fonction d'export). Retourne None si DEBUG_INSTRUMENTATION est False.
    """
    if not DEBUG_INSTRUMENTATION:
        return None

    debug_map = agg.get("debug_map", {}) or {}

    # Rejeu des deux filtres d'export (paramètres effectifs de l'UI, pas les
    # défauts des fonctions). Aujourd'hui identiques ; comptés séparément au cas
    # où les paramètres divergeraient.
    exported_json_df = _filter_confluences(conf_full, *json_args)
    exported_llm_df = _filter_confluences(conf_full, *llm_args)

    def _count_actif(frame: Optional[pd.DataFrame], sym_d: str) -> int:
        if frame is None or frame.empty or "Actif" not in frame.columns:
            return 0
        return int((frame["Actif"] == sym_d).sum())

    per_symbol: Dict[str, Any] = {}
    total_pivots = 0
    total_trend_zones = 0
    total_zones = 0
    total_after_confluence = 0
    total_exported_json = 0
    total_exported_llm = 0

    for sym in symbols_to_scan:
        sym_d = sym.replace("_", "/")
        tf_debug = debug_map.get(sym_d, {}) or {}

        sym_pivots = 0
        sym_trend_zones = 0
        sym_zones = 0
        per_tf: Dict[str, Any] = {}
        for tf_name, dbg in tf_debug.items():
            if not isinstance(dbg, dict):
                continue
            n_piv = dbg.get("n_pivots") or 0
            n_tz = dbg.get("n_trend_zones") or 0
            n_z = dbg.get("n_zones") or 0
            sym_pivots += int(n_piv)
            sym_trend_zones += int(n_tz)
            sym_zones += int(n_z)
            per_tf[tf_name] = {
                "n_pivots": int(n_piv),
                "n_trend_zones": int(n_tz),
                "n_zones": int(n_z),
                "min_touches": dbg.get("min_touches"),
                "atr": dbg.get("atr"),
            }

        sym_after_conf = _count_actif(conf_full, sym_d)
        sym_json = _count_actif(exported_json_df, sym_d)
        sym_llm = _count_actif(exported_llm_df, sym_d)

        per_symbol[sym_d] = {
            "n_pivots": sym_pivots,
            "n_trend_zones": sym_trend_zones,
            "n_zones": sym_zones,
            "after_confluence": sym_after_conf,
            "exported_json": sym_json,
            "exported_llm": sym_llm,
            "by_tf": per_tf,
        }

        total_pivots += sym_pivots
        total_trend_zones += sym_trend_zones
        total_zones += sym_zones
        total_after_confluence += sym_after_conf
        total_exported_json += sym_json
        total_exported_llm += sym_llm

    return {
        "per_symbol": per_symbol,
        "totals": {
            "n_pivots": total_pivots,
            "n_trend_zones": total_trend_zones,
            "n_zones": total_zones,
            "after_confluence": total_after_confluence,
            "exported_json": total_exported_json,
            "exported_llm": total_exported_llm,
        },
    }


