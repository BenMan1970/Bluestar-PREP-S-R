"""RSI & Divergence Screener Pro — Streamlit application."""
# pylint: disable=too-many-lines

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import concurrent.futures
import html as html_lib
import json
import logging
import secrets
import threading
import time
from datetime import datetime, timezone

# Third-party
import numpy as np
import pandas as pd
import streamlit as st
from oandapyV20 import API
from oandapyV20.endpoints import instruments as oanda_instruments
from oandapyV20.exceptions import V20Error

# pylint: disable=import-error
from fpdf import FPDF
from scipy.signal import find_peaks
# pylint: enable=import-error

# =============================================================================
# 2. CONFIGURATION & CONSTANTES
# =============================================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.captureWarnings(True)
logger = logging.getLogger("rsi_screener")

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MAX_RETRIES = 3
API_TIMEOUT = 10
SCAN_COOLDOWN = 30

st.set_page_config(
    page_title="RSI & Divergence Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_SYS_RAND = secrets.SystemRandom()

# =============================================================================
# 3. STYLES CSS — minifié pour éviter lignes trop longues (C0301)
# =============================================================================

_CSS_STYLES = """
<style>
.main>div{padding-top:2rem}
.screener-header{font-size:28px;font-weight:bold;color:#FAFAFA;
margin-bottom:15px;text-align:center}
.update-info{background-color:#262730;padding:8px 15px;border-radius:5px;
margin-bottom:20px;font-size:14px;color:#A9A9A9;border:1px solid #333A49;
text-align:center}
.legend-container{display:flex;justify-content:center;flex-wrap:wrap;
gap:25px;margin:25px 0;padding:15px;border-radius:5px;background-color:#1A1C22}
.legend-item{display:flex;align-items:center;gap:8px;font-size:14px;
color:#D3D3D3}
.legend-dot{width:12px;height:12px;border-radius:50%}
.oversold-dot{background-color:#FF4B4B}
.overbought-dot{background-color:#3D9970}
h3{color:#EAEAEA;text-align:center;margin-top:30px;margin-bottom:15px}
.rsi-table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px;
box-shadow:0 4px 8px 0 rgba(0,0,0,0.1)}
.rsi-table th{background-color:#333A49;color:#EAEAEA!important;
padding:14px 10px;text-align:center;font-weight:bold;font-size:15px;
border:1px solid #262730}
.rsi-table td{padding:12px 10px;text-align:center;border:1px solid #262730;
font-size:14px}
.devises-cell{font-weight:bold!important;color:#E0E0E0!important;
font-size:15px!important;text-align:left!important;padding-left:15px!important}
.oversold-cell{background-color:rgba(255,75,75,0.7)!important;
color:white!important;font-weight:bold}
.overbought-cell{background-color:rgba(61,153,112,0.7)!important;
color:white!important;font-weight:bold}
.neutral-cell{color:#C0C0C0!important;background-color:#161A1D}
.divergence-arrow{font-size:20px;font-weight:bold;vertical-align:middle;
margin-left:6px}
.bullish-arrow{color:#3D9970}
.bearish-arrow{color:#FF4B4B}
div[data-testid="stButton"]>button[kind="primary"]{background-color:#D32F2F;
color:white;border:1px solid #B71C1C;transition:all 0.2s}
div[data-testid="stButton"]>button[kind="primary"]:hover{
background-color:#B71C1C;border-color:#D32F2F;
box-shadow:0 4px 12px rgba(211,47,47,0.4)}
div[data-testid="stButton"]>button[kind="primary"]:active{
background-color:#D32F2F;transform:scale(0.98)}
div[data-testid="stButton"]>button{font-weight:600}
</style>
"""

st.markdown(_CSS_STYLES, unsafe_allow_html=True)

# =============================================================================
# 4. SECRETS OANDA
# =============================================================================

try:
    OANDA_ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
    OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    OANDA_ENVIRONMENT = st.secrets.get("OANDA_ENVIRONMENT", "practice")
except KeyError:
    st.error(
        "Secrets non trouvés ! Vérifiez votre fichier .streamlit/secrets.toml"
    )
    st.stop()

if OANDA_ENVIRONMENT not in ("practice", "live"):
    st.error(
        f"OANDA_ENVIRONMENT invalide : '{OANDA_ENVIRONMENT}'. "
        "Valeurs acceptées : 'practice' ou 'live'."
    )
    st.stop()

# =============================================================================
# 5. DONNÉES MARCHÉ
# =============================================================================

ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD",
    "GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD",
    "AUD/JPY", "AUD/CAD", "AUD/CHF", "AUD/NZD",
    "CAD/JPY", "CAD/CHF", "CHF/JPY", "NZD/JPY", "NZD/CAD", "NZD/CHF",
    "DE30/EUR", "XAU/USD", "SPX500/USD", "NAS100/USD", "US30/USD",
]
RESTRICTED_ASSETS = {"DE30/EUR", "SPX500/USD", "NAS100/USD", "US30/USD", "XAU/USD"}
TIMEFRAMES = [("H1", "H1"), ("H4", "H4"), ("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")]
TIMEFRAMES_DISPLAY = [tf[0] for tf in TIMEFRAMES]
TIMEFRAMES_FETCH_KEYS = [tf[1] for tf in TIMEFRAMES]
CANDLE_COUNT = {"H1": 200, "H4": 200, "D": 150, "W": 100, "M": 60}
CANDLE_COUNT_RESTRICTED = {"H1": 200, "H4": 200, "D": 100, "W": 52, "M": 24}
DIVERGENCE_LOOKBACK = {"H1": 40, "H4": 35, "D": 30, "W": 20, "M": 15}
ASSET_ORDER = {a: i for i, a in enumerate(ASSETS)}

# Q-006 : Pondérations temporelles prêtes pour les futurs scores structurels / confluences
_TF_LAMBDA = {"H4": 2.0, "Daily": 1.0, "Weekly": 0.5}

# =============================================================================
# 6. DATA PROVIDER — Thread-safe OANDA client (AUD-001)
# =============================================================================

_thread_local = threading.local()


def get_thread_oanda_client():
    """AUD-001 : Thread-local client — un client HTTP isolé par worker."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = API(
            access_token=OANDA_ACCESS_TOKEN,
            environment=OANDA_ENVIRONMENT,
            request_params={"timeout": API_TIMEOUT},
        )
    return _thread_local.client


@st.cache_resource
def get_oanda_semaphore():
    """Sémaphore partagé au niveau process (limitation OANDA par compte)."""
    return threading.Semaphore(3)


# =============================================================================
# 7. INDICATEURS — RSI & Divergence
# =============================================================================

def calculate_rsi(prices, period=RSI_PERIOD):
    """RSI Wilder vectorisé via ewm — équivalent mathématique strict de la
    boucle Wilder après warmup, avec gestion epsilon des cas limites (AUD-006, Q-001).

    Returns:
        tuple: (dernier_rsi: float, série_rsi: pd.Series | None)
    """
    if prices is None or len(prices) < period + 1:
        return np.nan, None

    close_prices = prices["Close"]
    delta = close_prices.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder smoothing = EMA avec alpha = 1/period, sans correction de biais.
    # min_periods=period garantit NaN pendant le warmup.
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    epsilon = 1e-12
    gain_arr = avg_gain.to_numpy()
    loss_arr = avg_loss.to_numpy()

    # Gestion explicite des 3 cas limites :
    #   - gain≈0 ET loss≈0 → marché plat → RSI = 50
    #   - loss≈0 seul       → uniquement des hausses → RSI = 100
    #   - gain≈0 seul       → uniquement des baisses → RSI = 0
    #   - sinon             → RSI = 100 - 100 / (1 + RS)
    both_zero = (np.abs(gain_arr) < epsilon) & (np.abs(loss_arr) < epsilon)
    loss_zero = (np.abs(loss_arr) < epsilon) & ~both_zero
    gain_zero = (np.abs(gain_arr) < epsilon) & ~both_zero

    # Calcul sécurisé : on remplace loss=0 par epsilon pour la division,
    # puis on écrase les cas limites avec np.where (ordre important).
    safe_loss = np.where(np.abs(loss_arr) < epsilon, epsilon, loss_arr)
    rs = gain_arr / safe_loss
    rsi_arr = 100.0 - 100.0 / (1.0 + rs)

    rsi_arr = np.where(both_zero, 50.0, rsi_arr)
    rsi_arr = np.where(loss_zero, 100.0, rsi_arr)
    rsi_arr = np.where(gain_zero, 0.0, rsi_arr)

    # On préserve les NaN du warmup.
    warmup_mask = np.isnan(gain_arr) | np.isnan(loss_arr)
    rsi_arr = np.where(warmup_mask, np.nan, rsi_arr)

    rsi_series = pd.Series(rsi_arr, index=close_prices.index)

    if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
        return np.nan, None

    return float(rsi_series.iloc[-1]), rsi_series


def _get_price_delta(pair_name):
    """Seuil min_price_delta adapté au type d'instrument."""
    if "JPY" in pair_name:
        return 0.0003
    if "XAU" in pair_name:
        return 0.002
    if any(idx in pair_name for idx in ("DE30", "SPX500", "NAS100", "US30")):
        return 0.003
    return 0.001


def _get_peak_distance(timeframe_key):
    """Distance entre pics selon le timeframe."""
    distance_map = {"H1": 3, "H4": 5, "D": 4, "W": 3, "M": 2}
    return distance_map.get(timeframe_key, 5)


def _get_lookback(timeframe_key, data_len):
    """Lookback adaptatif borné par la longueur des données."""
    lookback = DIVERGENCE_LOOKBACK.get(timeframe_key, 30)
    return min(lookback, data_len)


def _rsi_window_value(rsi_vals, window_idx, mode):
    """Valeur min ou max du RSI dans une fenêtre autour de window_idx."""
    lo = max(0, window_idx - 2)
    hi = min(len(rsi_vals), window_idx + 3)
    window = rsi_vals[lo:hi]
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return np.nan
    if mode == "max":
        return float(np.max(valid))
    return float(np.min(valid))


def _check_bearish_divergence(
    price_close, rsi_vals, peak_indices, min_price_delta, min_rsi_delta
):
    """Détection divergence baissière (higher high prix + lower high RSI)."""
    if len(peak_indices) < 2:
        return False
    prev_peak = peak_indices[-2]
    last_peak = peak_indices[-1]
    price_diff_ok = price_close[last_peak] > price_close[prev_peak] * (
        1 + min_price_delta
    )
    if not price_diff_ok:
        return False
    rsi_max_lp = _rsi_window_value(rsi_vals, last_peak, "max")
    rsi_max_pp = _rsi_window_value(rsi_vals, prev_peak, "max")
    if np.isnan(rsi_max_lp) or np.isnan(rsi_max_pp):
        return False
    return rsi_max_lp < rsi_max_pp - min_rsi_delta


def _check_bullish_divergence(
    price_close, rsi_vals, trough_indices, min_price_delta, min_rsi_delta
):
    """Détection divergence haussière (lower low prix + higher low RSI)."""
    if len(trough_indices) < 2:
        return False
    prev_trough = trough_indices[-2]
    last_trough = trough_indices[-1]
    price_diff_ok = price_close[last_trough] < price_close[prev_trough] * (
        1 - min_price_delta
    )
    if not price_diff_ok:
        return False
    rsi_min_lt = _rsi_window_value(rsi_vals, last_trough, "min")
    rsi_min_pt = _rsi_window_value(rsi_vals, prev_trough, "min")
    if np.isnan(rsi_min_lt) or np.isnan(rsi_min_pt):
        return False
    return rsi_min_lt > rsi_min_pt + min_rsi_delta


def detect_divergence(price_data, rsi_series, timeframe_key, pair_name=""):
    # pylint: disable=too-many-locals
    """Détection divergence avec prominence absolue minimale (AUD-012)."""
    if rsi_series is None or len(price_data) < 10:
        return "Aucune"

    lookback = _get_lookback(timeframe_key, len(price_data))
    peak_distance = _get_peak_distance(timeframe_key)
    min_price_delta = _get_price_delta(pair_name)
    min_rsi_delta = 2.0

    recent_price = price_data.iloc[-lookback:]
    rsi_vals = rsi_series.reindex(recent_price.index).values
    price_close = recent_price["Close"].values

    price_std = np.std(price_close)
    prominence_val = (
        max(price_std * 0.5, min_price_delta)
        if price_std > 0
        else min_price_delta
    )

    price_peaks_idx, _ = find_peaks(
        price_close, distance=peak_distance, prominence=prominence_val
    )
    if _check_bearish_divergence(
        price_close, rsi_vals, price_peaks_idx, min_price_delta, min_rsi_delta
    ):
        return "Baissière"

    price_troughs_idx, _ = find_peaks(
        -price_close, distance=peak_distance, prominence=prominence_val
    )
    if _check_bullish_divergence(
        price_close, rsi_vals, price_troughs_idx, min_price_delta, min_rsi_delta
    ):
        return "Haussière"

    return "Aucune"


# =============================================================================
# 8. FETCH — OANDA avec validation & retry ciblé (AUD-003, AUD-011)
# =============================================================================

def validate_ohlc(df, pair, timeframe_key, min_rows=RSI_PERIOD + 1):
    """A-003 : Validation OHLC centralisée — un seul point d'entrée pipeline.

    Vérifie l'intégrité structurelle des bougies après fetch et avant tout
    calcul d'indicateur. Drop les lignes invalides, retourne None si la série
    nettoyée est trop courte pour produire un signal fiable.

    Args:
        df: DataFrame OHLCV indexé par timestamp.
        pair: nom de la paire (pour logging).
        timeframe_key: clé du timeframe (pour logging).
        min_rows: nombre minimum de lignes valides requis en sortie.

    Returns:
        tuple: (df_clean: pd.DataFrame | None, warnings: list[str])
    """
    warnings_list = []

    if df is None or df.empty:
        return None, ["empty_or_none_input"]

    initial_len = len(df)
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        msg = f"missing_columns:{sorted(missing_cols)}"
        logger.error("OHLC validation %s %s — %s", pair, timeframe_key, msg)
        return None, [msg]

    # 1) NaN dans OHLC → drop ligne (Volume NaN toléré, sera mis à 0)
    ohlc_cols = ["Open", "High", "Low", "Close"]
    nan_mask = df[ohlc_cols].isna().any(axis=1)
    if nan_mask.any():
        warnings_list.append(f"dropped_nan_ohlc:{int(nan_mask.sum())}")
        df = df.loc[~nan_mask].copy()

    # 2) Cohérence High/Low : High >= max(O,C,L) ET Low <= min(O,C,H)
    max_ocl = df[["Open", "Close", "Low"]].max(axis=1)
    min_och = df[["Open", "Close", "High"]].min(axis=1)
    invalid_hi = df["High"] < max_ocl
    invalid_lo = df["Low"] > min_och
    invalid_hl = df["High"] < df["Low"]
    bad_mask = invalid_hi | invalid_lo | invalid_hl
    if bad_mask.any():
        warnings_list.append(f"dropped_inconsistent_ohlc:{int(bad_mask.sum())}")
        df = df.loc[~bad_mask].copy()

    # 3) Volume négatif → impossible, drop
    if "Volume" in df.columns:
        neg_vol = df["Volume"] < 0
        if neg_vol.any():
            warnings_list.append(f"dropped_negative_volume:{int(neg_vol.sum())}")
            df = df.loc[~neg_vol].copy()

    # 4) Prix non strictement positifs (devises/indices : impossible)
    non_positive = (df[ohlc_cols] <= 0).any(axis=1)
    if non_positive.any():
        warnings_list.append(f"dropped_non_positive_price:{int(non_positive.sum())}")
        df = df.loc[~non_positive].copy()

    # 5) Vérification longueur finale
    final_len = len(df)
    if final_len < min_rows:
        msg = f"insufficient_rows_after_clean:{final_len}<{min_rows}"
        logger.warning(
            "OHLC validation %s %s — %s (initial=%d)",
            pair, timeframe_key, msg, initial_len,
        )
        warnings_list.append(msg)
        return None, warnings_list

    if warnings_list:
        logger.warning(
            "OHLC validation %s %s — cleaned %d→%d rows, warnings=%s",
            pair, timeframe_key, initial_len, final_len, warnings_list,
        )

    return df, warnings_list


def _build_candle_dataframe(candles, pair, timeframe_key):
    """Construit et valide un DataFrame à partir des bougies OANDA."""
    data_list = []
    for candle in candles:
        if not candle.get("complete"):
            continue
        if "mid" not in candle:
            logger.error("Missing 'mid' key in candle for %s %s", pair, timeframe_key)
            continue
        try:
            data_list.append(
                {
                    "Time": candle["time"],
                    "Open": float(candle["mid"]["o"]),
                    "High": float(candle["mid"]["h"]),
                    "Low": float(candle["mid"]["l"]),
                    "Close": float(candle["mid"]["c"]),
                    "Volume": int(candle["volume"]),
                }
            )
        except (KeyError, ValueError) as parse_err:
            logger.error(
                "Candle parse error for %s %s: %s", pair, timeframe_key, parse_err
            )
            continue

    if not data_list:
        logger.warning("No complete candles for %s %s", pair, timeframe_key)
        return None

    df = pd.DataFrame(data_list)
    df["Time"] = pd.to_datetime(df["Time"])
    df.set_index("Time", inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    return df


def _handle_v20_error(error, pair, timeframe_key, attempt):
    """Gestion ciblée des erreurs V20 pour décider retry ou abandon."""
    err_code = getattr(error, "code", None)
    if err_code in (400, 401, 403):
        logger.error(
            "Fatal OANDA error %s for %s %s — aborting retries: %s",
            err_code,
            pair,
            timeframe_key,
            error,
        )
        return "abort"
    logger.warning(
        "fetch_forex_data_oanda attempt %d/%d V20Error %s for %s %s: %s",
        attempt + 1,
        MAX_RETRIES,
        err_code,
        pair,
        timeframe_key,
        error,
    )
    return "retry"


def _compute_backoff_delay(attempt):
    """Délai exponentiel avec jitter sécurisé (bandit B311 fix)."""
    return min(60, 2**attempt) + _SYS_RAND.random()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forex_data_oanda(pair, timeframe_key, _cache_version=0):
    """
    Fetch OANDA avec retry sélectif, validation stricte des séries (AUD-011),
    et client thread-local (AUD-001).
    """
    count_map = (
        CANDLE_COUNT_RESTRICTED if pair in RESTRICTED_ASSETS else CANDLE_COUNT
    )
    count = count_map.get(timeframe_key, 150)

    instrument = pair.replace("/", "_")
    params = {"granularity": timeframe_key, "count": count}

    oanda_semaphore = get_oanda_semaphore()

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(_SYS_RAND.uniform(0.05, 0.15))

            with oanda_semaphore:
                api_client = get_thread_oanda_client()
                req = oanda_instruments.InstrumentsCandles(
                    instrument=instrument, params=params
                )
                api_client.request(req)

            candles = req.response.get("candles", [])
            raw_df = _build_candle_dataframe(candles, pair, timeframe_key)
            if raw_df is None:
                return None
            clean_df, _warns = validate_ohlc(raw_df, pair, timeframe_key)
            return clean_df

        except V20Error as exc:
            action = _handle_v20_error(exc, pair, timeframe_key, attempt)
            if action == "abort":
                return None
            if attempt < MAX_RETRIES - 1:
                time.sleep(_compute_backoff_delay(attempt))

        except (TimeoutError, ConnectionError) as exc:
            logger.warning(
                "fetch_forex_data_oanda attempt %d/%d network error for %s %s: %s",
                attempt + 1,
                MAX_RETRIES,
                pair,
                timeframe_key,
                exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(_compute_backoff_delay(attempt))

    logger.error("Fetch definitively failed for %s %s", pair, timeframe_key)
    return None


# =============================================================================
# 9. UI HELPERS
# =============================================================================

def format_rsi(value):
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def get_rsi_class(value):
    """Classe CSS en fonction de la valeur RSI."""
    if pd.isna(value):
        return "neutral-cell"
    if value <= RSI_OVERSOLD:
        return "oversold-cell"
    if value >= RSI_OVERBOUGHT:
        return "overbought-cell"
    return "neutral-cell"


def _pdf_str(text):
    return text.encode("latin-1", errors="replace").decode("latin-1")


# =============================================================================
# 10. STATISTIQUES
# =============================================================================

def _compute_tf_stats(tf_data_list):
    """Calcule les statistiques pour un seul timeframe."""
    counts = {
        "extreme_oversold": 0,
        "oversold": 0,
        "extreme_overbought": 0,
        "overbought": 0,
        "bull_div": 0,
        "bear_div": 0,
    }
    valid_rsi = []
    for data in tf_data_list:
        rsi = data.get("rsi")
        if pd.notna(rsi):
            valid_rsi.append(rsi)
            if rsi <= 20:
                counts["extreme_oversold"] += 1
            elif rsi <= RSI_OVERSOLD:
                counts["oversold"] += 1
            elif rsi >= 80:
                counts["extreme_overbought"] += 1
            elif rsi >= RSI_OVERBOUGHT:
                counts["overbought"] += 1
        div = data.get("divergence")
        if div == "Haussière":
            counts["bull_div"] += 1
        elif div == "Baissière":
            counts["bear_div"] += 1
    counts["valid_count"] = len(valid_rsi)
    return counts


def _compute_market_bias(avg_rsi):
    """Détermine le biais de marché global."""
    if avg_rsi < 45:
        return "BEARISH (Pression Vendeuse)", (220, 20, 60)
    if avg_rsi > 55:
        return "BULLISH (Pression Acheteuse)", (0, 180, 80)
    return "NEUTRE / INCERTAIN", (100, 100, 100)


def compute_statistics(results_data):
    """Calcul centralisé avec séparation états RSI / signaux divergence."""
    global_rsi_values = []
    stats_by_tf = {}
    for tf_name in TIMEFRAMES_DISPLAY:
        tf_data = [asset_row.get(tf_name, {}) for asset_row in results_data]
        stats_by_tf[tf_name] = _compute_tf_stats(tf_data)
        global_rsi_values.extend(
            d.get("rsi") for d in tf_data if pd.notna(d.get("rsi"))
        )

    avg_global_rsi = (
        float(np.mean(global_rsi_values)) if global_rsi_values else 50.0
    )
    total_bull_div = sum(s["bull_div"] for s in stats_by_tf.values())
    total_bear_div = sum(s["bear_div"] for s in stats_by_tf.values())
    extreme_count = sum(
        s["extreme_oversold"] + s["extreme_overbought"]
        for s in stats_by_tf.values()
    )

    market_bias, bias_color = _compute_market_bias(avg_global_rsi)

    return {
        "by_tf": stats_by_tf,
        "avg_rsi": avg_global_rsi,
        "total_bull_div": total_bull_div,
        "total_bear_div": total_bear_div,
        "extreme_count": extreme_count,
        "market_bias": market_bias,
        "bias_color": bias_color,
    }


# =============================================================================
# 11. SIGNAL ENGINE
# =============================================================================

def process_single_asset(pair_name, cache_version=0):
    """
    Traitement d'un asset avec gestion d'erreur ciblée (AUD-003).
    En cas d'erreur inattendue, tous les TF sont invalidés.
    """
    row_data = {"Devises": pair_name, "Status": "OK"}
    try:
        for tf_display_name, tf_key in TIMEFRAMES:
            data_ohlc = fetch_forex_data_oanda(pair_name, tf_key, cache_version)

            if data_ohlc is None:
                row_data[tf_display_name] = {
                    "rsi": np.nan,
                    "divergence": "Aucune",
                }
                row_data["Status"] = "PARTIAL"
                continue

            rsi_value, rsi_series = calculate_rsi(data_ohlc)
            divergence_signal = (
                detect_divergence(data_ohlc, rsi_series, tf_key, pair_name)
                if rsi_series is not None
                else "Aucune"
            )
            row_data[tf_display_name] = {
                "rsi": rsi_value,
                "divergence": divergence_signal,
            }

    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        logger.exception(
            "Crash in process_single_asset for %s: %s", pair_name, exc
        )
        row_data["Status"] = "ERROR"
        for tf_display, _ in TIMEFRAMES:
            row_data[tf_display] = {
                "rsi": np.nan,
                "divergence": "Aucune",
            }

    return row_data


# =============================================================================
# 12. ORCHESTRATION — Scan parallèle avec timeout réel
# =============================================================================

def _submit_scan_tasks(executor, cache_ver):
    """Soumet les tâches de scan au ThreadPoolExecutor."""
    return {
        executor.submit(process_single_asset, asset, cache_ver): asset
        for asset in ASSETS
    }


def _handle_future_result(future, asset_name):
    """Extrait le résultat d'un Future et gère les exceptions."""
    try:
        data = future.result()
        if data and data.get("Status") in ("ERROR", "PARTIAL"):
            logger.warning("Asset %s: status=%s", asset_name, data.get("Status"))
        return data
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        logger.error("Future failed for %s: %s", asset_name, exc)
        return {
            "Devises": asset_name,
            "Status": "ERROR",
            **{
                tf: {"rsi": np.nan, "divergence": "Aucune"}
                for tf in TIMEFRAMES_DISPLAY
            },
        }


def _inject_timeout_assets(results_list, pending_assets):
    """Injecte les assets manquants avec statut TIMEOUT (AUD-007)."""
    for missing_asset in pending_assets:
        results_list.append(
            {
                "Devises": missing_asset,
                "Status": "TIMEOUT",
                **{
                    tf: {"rsi": np.nan, "divergence": "Aucune"}
                    for tf in TIMEFRAMES_DISPLAY
                },
            }
        )


def run_analysis_process():
    # pylint: disable=too-many-locals
    """
    Scan parallèle avec timeout d'ensemble et gestion explicite des assets
    manquants en TIMEOUT (AUD-002, AUD-007).
    """
    results_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initialisation du scan parallèle...")

    cache_ver = st.session_state.get("cache_version", 0) % 1000
    st.session_state.cache_version = cache_ver

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    future_to_asset = _submit_scan_tasks(executor, cache_ver)

    completed = 0
    total = len(ASSETS)
    timed_out_assets = set(ASSETS)

    try:
        for future in concurrent.futures.as_completed(
            future_to_asset, timeout=300
        ):
            asset_name = future_to_asset[future]
            timed_out_assets.discard(asset_name)

            data = _handle_future_result(future, asset_name)
            if data:
                results_list.append(data)

            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(
                f"Scan terminé : {asset_name} ({completed}/{total})"
            )

    except concurrent.futures.TimeoutError:
        logger.error(
            "Scan timeout d'ensemble après 300s — %d/%d assets completed",
            completed,
            total,
        )
        st.warning(
            f"⏱ Timeout du scan après 300s — {completed}/{total} actifs traités."
        )
        _inject_timeout_assets(results_list, timed_out_assets)

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    results_list.sort(key=lambda x: ASSET_ORDER.get(x["Devises"], 999))

    scan_ts = datetime.now(timezone.utc)
    computed_stats = compute_statistics(results_list)
    scan_ts_s = scan_ts.strftime("%d/%m/%Y %H:%M:%S UTC")

    new_state = {
        "results": results_list,
        "last_scan_time": scan_ts,
        "scan_done": True,
        "stats": computed_stats,
        "pdf_data": create_pdf_report(results_list, computed_stats, scan_ts_s),
        "json_data": create_json_export(results_list, computed_stats, scan_ts),
        "csv_data": create_csv_export(results_list),
    }
    st.session_state.update(new_state)

    status_text.empty()
    progress_bar.empty()


# =============================================================================
# 13. REPORTING — Exports JSON / CSV / PDF
# =============================================================================

def _flatten_results(results_data):
    """Structure plate pour l'export CSV uniquement."""
    records = []
    for result_row in results_data:
        record = {
            "Devises": result_row["Devises"],
            "Status": result_row.get("Status", "OK"),
        }
        for tf_display in TIMEFRAMES_DISPLAY:
            cell = result_row.get(tf_display, {})
            rsi = cell.get("rsi", np.nan)
            record[f"RSI_{tf_display}"] = (
                round(float(rsi), 2) if pd.notna(rsi) else None
            )
            record[f"DIV_{tf_display}"] = cell.get("divergence", "Aucune")
        records.append(record)
    return records


_DIV_ENUM = {"Haussière": "BULL", "Baissière": "BEAR", "Aucune": "NONE"}
_TF_KEY_MAP = dict(TIMEFRAMES)


def _market_status(scan_ts):
    """AUD-004 : statut marché basé sur UTC avec fenêtres réelles Forex."""
    weekday = scan_ts.weekday()
    hour = scan_ts.hour
    if weekday == 5:
        return "closed_saturday"
    if weekday == 6 and hour < 22:
        return "closed_sunday"
    if weekday == 4 and hour >= 22:
        return "closed_friday_night"
    return "open"


def _build_json_meta(scan_ts):
    """Construit le bloc meta de l'export JSON."""
    return {
        "scan_ts": scan_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rsi_period": RSI_PERIOD,
        "thresholds": {
            "oversold": RSI_OVERSOLD,
            "overbought": RSI_OVERBOUGHT,
            "extreme_low": 20,
            "extreme_high": 80,
        },
        "market_status": _market_status(scan_ts),
        "instruments_count": len(ASSETS),
        "timeframes": TIMEFRAMES_FETCH_KEYS,
        "price_mode": "mid",
        "execution_note": "ANALYTIC_ONLY_NO_SPREAD",
    }


def _build_json_summary(statistics):
    """Construit le bloc summary de l'export JSON."""
    by_tf_summary = {}
    for tf_display in TIMEFRAMES_DISPLAY:
        stat_block = statistics["by_tf"][tf_display]
        fetch_key = _TF_KEY_MAP[tf_display]
        by_tf_summary[fetch_key] = {
            "extreme_oversold": stat_block["extreme_oversold"],
            "oversold": stat_block["oversold"],
            "overbought": stat_block["overbought"],
            "extreme_overbought": stat_block["extreme_overbought"],
            "div_bull": stat_block["bull_div"],
            "div_bear": stat_block["bear_div"],
            "valid_count": stat_block["valid_count"],
        }

    raw_bias = statistics["market_bias"]
    if "BEARISH" in raw_bias:
        bias_key = "BEARISH"
    elif "BULLISH" in raw_bias:
        bias_key = "BULLISH"
    else:
        bias_key = "NEUTRAL"

    return {
        "market_bias": bias_key,
        "avg_rsi": round(statistics["avg_rsi"], 2),
        "total_div_bull": statistics["total_bull_div"],
        "total_div_bear": statistics["total_bear_div"],
        "total_extremes": statistics["extreme_count"],
        "by_timeframe": by_tf_summary,
    }


def _build_json_instruments(results_data):
    """Construit le bloc instruments de l'export JSON."""
    instruments_out = []
    for result_row in results_data:
        tf_data = {}
        for tf_display, tf_fetch in TIMEFRAMES:
            cell = result_row.get(tf_display, {})
            rsi = cell.get("rsi", np.nan)
            div = cell.get("divergence", "Aucune")
            tf_data[tf_fetch] = {
                "rsi": round(float(rsi), 2) if pd.notna(rsi) else None,
                "div": _DIV_ENUM.get(div, "NONE"),
            }
        instruments_out.append(
            {
                "pair": result_row["Devises"],
                "status": result_row.get("Status", "OK"),
                "timeframes": tf_data,
            }
        )
    return instruments_out


def create_json_export(results_data, statistics, scan_ts):
    # pylint: disable=too-many-locals
    """Export JSON enrichi avec annotation mid-price (AUD-008)."""
    payload = {
        "meta": _build_json_meta(scan_ts),
        "summary": _build_json_summary(statistics),
        "instruments": _build_json_instruments(results_data),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def create_csv_export(results_data):
    """Génère l'export CSV."""
    df = pd.DataFrame(_flatten_results(results_data))
    return df.to_csv(index=False).encode("utf-8-sig")


class _ReportPDF(FPDF):
    """Classe PDF interne avec header/footer personnalisés."""

    def __init__(self, scan_ts="", **kwargs):
        super().__init__(**kwargs)
        self._scan_ts = scan_ts

    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(20, 20, 20)
        self.cell(
            0, 10, _pdf_str("MARKET SCANNER - RAPPORT STRATEGIQUE"), 0, 1, "C"
        )
        self.set_font("Arial", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, _pdf_str("Genere le: " + self._scan_ts), 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0,
            10,
            _pdf_str(
                "Page "
                + str(self.page_no())
                + " | Analyse technique automatisee"
            ),
            0,
            0,
            "C",
        )


def _pdf_set_cell_colors(pdf, val):
    """Applique les couleurs de fond/texte selon la valeur RSI."""
    if pd.notna(val):
        if val <= 20:
            pdf.set_fill_color(255, 100, 100)
            pdf.set_text_color(255, 255, 255)
        elif val <= 30:
            pdf.set_fill_color(220, 20, 60)
            pdf.set_text_color(255, 255, 255)
        elif val >= 80:
            pdf.set_fill_color(100, 255, 100)
            pdf.set_text_color(0, 0, 0)
        elif val >= 70:
            pdf.set_fill_color(0, 180, 80)
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_fill_color(240, 240, 240)
            pdf.set_text_color(10, 10, 10)
    else:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(10, 10, 10)


def _pdf_format_rsi_cell(val, div):
    """Formate le texte d'une cellule RSI avec divergence."""
    txt = f"{val:.2f}" if pd.notna(val) else "N/A"
    if div == "Haussière":
        txt += " (BULL)"
    elif div == "Baissière":
        txt += " (BEAR)"
    return txt


def _render_pdf_summary(pdf, statistics):
    """Rendu du bloc récapitulatif dans le PDF."""
    color_text_dark = (10, 10, 10)
    avg_rsi = statistics["avg_rsi"]
    bias = statistics["market_bias"]
    bias_color = statistics["bias_color"]
    bull = statistics["total_bull_div"]
    bear = statistics["total_bear_div"]
    extreme = statistics["extreme_count"]

    pdf.set_fill_color(245, 247, 250)
    pdf.rect(10, 25, 277, 35, "F")
    pdf.set_xy(15, 30)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*color_text_dark)
    pdf.cell(50, 8, _pdf_str("BIAIS DE MARCHE:"), 0, 0, "L")
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(*bias_color)
    pdf.cell(100, 8, _pdf_str(bias), 0, 1, "L")
    pdf.set_xy(15, 40)
    pdf.set_text_color(*color_text_dark)
    pdf.set_font("Arial", "", 10)
    pdf.cell(
        0,
        6,
        _pdf_str(
            f"RSI Moyen Global: {avg_rsi:.2f} | "
            f"Signaux Extremes (<20/>80): {extreme}"
        ),
        0,
        1,
        "L",
    )
    pdf.cell(
        0,
        6,
        _pdf_str(
            f"Divergences: {bull} Haussieres (BULL) vs {bear} Baissieres (BEAR)"
        ),
        0,
        1,
        "L",
    )
    pdf.ln(15)

    pdf.set_text_color(*color_text_dark)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _pdf_str("STATISTIQUES PAR TIMEFRAME"), 0, 1, "L")
    pdf.set_font("Arial", "", 9)
    for tf_display in TIMEFRAMES_DISPLAY:
        stat_block = statistics["by_tf"][tf_display]
        pdf.cell(
            0,
            6,
            _pdf_str(
                f"[{tf_display}] :: <=20: {stat_block['extreme_oversold']} | "
                f"20-30: {stat_block['oversold']} || "
                f">=80: {stat_block['extreme_overbought']} | "
                f"70-80: {stat_block['overbought']} || "
                f"DIV.BULL: {stat_block['bull_div']} | "
                f"DIV.BEAR: {stat_block['bear_div']}"
            ),
            0,
            1,
            "L",
        )
    pdf.ln(5)


def _render_pdf_data_table(pdf, results_data):
    """Rendu du tableau de données dans le PDF."""
    color_bg_header = (44, 62, 80)
    color_text_header = (255, 255, 255)
    color_neutral_bg = (240, 240, 240)
    color_text_dark = (10, 10, 10)

    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(*color_bg_header)
    pdf.set_text_color(*color_text_header)
    w_pair = 40
    w_tf = (277 - w_pair) / len(TIMEFRAMES_DISPLAY)
    pdf.cell(w_pair, 9, _pdf_str("Paire"), 1, 0, "C", True)
    for tf_display in TIMEFRAMES_DISPLAY:
        pdf.cell(w_tf, 9, _pdf_str(tf_display), 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for result_row in results_data:
        pdf.set_fill_color(*color_neutral_bg)
        pdf.set_text_color(*color_text_dark)
        pdf.cell(w_pair, 8, _pdf_str(result_row["Devises"]), 1, 0, "C", True)
        for tf_label in TIMEFRAMES_DISPLAY:
            cell = result_row.get(tf_label, {})
            val = cell.get("rsi", np.nan)
            div = cell.get("divergence", "Aucune")
            _pdf_set_cell_colors(pdf, val)
            txt = _pdf_format_rsi_cell(val, div)
            pdf.cell(w_tf, 8, _pdf_str(txt), 1, 0, "C", True)
        pdf.ln()


def create_pdf_report(results_data, statistics, last_scan_time):
    # pylint: disable=too-many-locals,too-many-statements
    """Génération PDF avec stats pré-calculées."""
    pdf = _ReportPDF(
        scan_ts=str(last_scan_time), orientation="L", unit="mm", format="A4"
    )
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    _render_pdf_summary(pdf, statistics)
    _render_pdf_data_table(pdf, results_data)
    return bytes(pdf.output())


# =============================================================================
# 14. UI — Interface Streamlit
# =============================================================================

st.markdown(
    '<h1 class="screener-header">Screener RSI & Divergence Pro</h1>',
    unsafe_allow_html=True,
)

last_scan_ts = st.session_state.get("last_scan_time")
cooldown_active = False
if last_scan_ts and isinstance(last_scan_ts, datetime):
    elapsed = (
        datetime.now(timezone.utc) - last_scan_ts.replace(tzinfo=timezone.utc)
    ).total_seconds()
    cooldown_active = elapsed < SCAN_COOLDOWN
    if cooldown_active:
        st.info(
            f"⏳ Cooldown actif : veuillez attendre "
            f"{int(SCAN_COOLDOWN - elapsed)}s avant un nouveau scan."
        )

if "scan_done" in st.session_state and st.session_state.scan_done:
    last_scan_time_str = st.session_state.last_scan_time.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    st.markdown(
        f'<div class="update-info">Dernière mise à jour : {last_scan_time_str}</div>',
        unsafe_allow_html=True,
    )

col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

with col2:
    if st.button(
        "Rescan", use_container_width=True, disabled=cooldown_active
    ):
        st.session_state.scan_done = False
        st.session_state.cache_version = (
            st.session_state.get("cache_version", 0) + 1
        ) % 1000
        st.rerun()

with col3:
    if st.session_state.get("pdf_data"):
        st.download_button(
            label="⬇ PDF",
            data=st.session_state.pdf_data,
            file_name=(
                f"RSI_Report_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

with col4:
    if st.session_state.get("json_data"):
        st.download_button(
            label="⬇ JSON",
            data=st.session_state.json_data,
            file_name=(
                f"RSI_Report_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

with col5:
    if st.session_state.get("csv_data"):
        st.download_button(
            label="⬇ CSV",
            data=st.session_state.csv_data,
            file_name=(
                f"RSI_Report_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

if "scan_done" not in st.session_state or not st.session_state.scan_done:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "LANCER LE SCAN COMPLET",
        type="primary",
        use_container_width=True,
        disabled=cooldown_active,
    ):
        run_analysis_process()
        st.rerun()

if st.session_state.get("results"):
    st.markdown(
        f"""
    <div class="legend-container">
        <div class="legend-item">
            <div class="legend-dot oversold-dot"></div>
            <span>Oversold (RSI &le; {RSI_OVERSOLD})</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot overbought-dot"></div>
            <span>Overbought (RSI &ge; {RSI_OVERBOUGHT})</span>
        </div>
        <div class="legend-item">
            <span class="divergence-arrow bullish-arrow">&#8593;</span>
            <span>Bullish Divergence</span>
        </div>
        <div class="legend-item">
            <span class="divergence-arrow bearish-arrow">&#8595;</span>
            <span>Bearish Divergence</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### RSI & Divergence Analysis Results")

    header_cells = [f"<th>{tf}</th>" for tf in TIMEFRAMES_DISPLAY]
    html_table = (
        '<table class="rsi-table"><thead><tr><th>Devises</th>'
        + "".join(header_cells)
        + "</tr></thead><tbody>"
    )

    error_count = 0
    timeout_count = 0
    for asset_row in st.session_state.results:
        pair_safe = html_lib.escape(str(asset_row["Devises"]))
        status = asset_row.get("Status", "OK")
        status_badge = ""
        if status == "ERROR":
            status_badge = ' <span style="color:#FF4B4B;font-size:10px;">⚠ERR</span>'
            error_count += 1
        elif status == "PARTIAL":
            status_badge = ' <span style="color:#FFA500;font-size:10px;">⚠PART</span>'
        elif status == "TIMEOUT":
            status_badge = ' <span style="color:#FF4B4B;font-size:10px;">⏱TIMEOUT</span>'
            timeout_count += 1

        row_html = f'<tr><td class="devises-cell">{pair_safe}{status_badge}</td>'
        row_cells = []
        for tf_label in TIMEFRAMES_DISPLAY:
            cell_data = asset_row.get(tf_label, {"rsi": np.nan, "divergence": "Aucune"})
            rsi_val = cell_data.get("rsi", np.nan)
            divergence = cell_data.get("divergence", "Aucune")
            css_class = get_rsi_class(rsi_val)
            formatted_val = format_rsi(rsi_val)
            icon_html = ""
            if divergence == "Haussière":
                icon_html = '<span class="divergence-arrow bullish-arrow">&#8593;</span>'
            elif divergence == "Baissière":
                icon_html = '<span class="divergence-arrow bearish-arrow">&#8595;</span>'
            row_cells.append(
                f'<td class="{css_class}">{formatted_val} {icon_html}</td>'
            )
        row_html += "".join(row_cells) + "</tr>"
        html_table += row_html

    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)

    if error_count > 0:
        st.warning(
            f"⚠️ {error_count} actif(s) en erreur lors du scan. "
            "Vérifiez les logs ou relancez."
        )
    if timeout_count > 0:
        st.warning(
            f"⏱ {timeout_count} actif(s) en timeout — "
            "le scan a dépassé la limite de 300s."
        )

    st.markdown("### Signal Statistics")

    display_stats = st.session_state.get("stats")
    if display_stats is None:
        display_stats = compute_statistics(st.session_state.get("results", []))

    st.markdown("**États RSI par Timeframe**")
    rsi_cols = st.columns(len(TIMEFRAMES_DISPLAY))
    for col_idx, tf_label in enumerate(TIMEFRAMES_DISPLAY):
        tf_stat_data = display_stats["by_tf"][tf_label]
        total_rsi = (
            tf_stat_data["extreme_oversold"]
            + tf_stat_data["oversold"]
            + tf_stat_data["extreme_overbought"]
            + tf_stat_data["overbought"]
        )
        with rsi_cols[col_idx]:
            st.metric(label=f"RSI States {tf_label}", value=str(total_rsi))
            st.markdown(
                f"≤20:{tf_stat_data['extreme_oversold']} | "
                f"20-30:{tf_stat_data['oversold']} | "
                f"70-80:{tf_stat_data['overbought']} | "
                f"≥80:{tf_stat_data['extreme_overbought']}"
            )

    st.markdown("**Signaux de Divergence par Timeframe**")
    div_cols = st.columns(len(TIMEFRAMES_DISPLAY))
    for col_idx, tf_label in enumerate(TIMEFRAMES_DISPLAY):
        tf_stat_data = display_stats["by_tf"][tf_label]
        with div_cols[col_idx]:
            st.metric(
                label=f"Divergences {tf_label}",
                value=str(tf_stat_data["bull_div"] + tf_stat_data["bear_div"]),
            )
            st.markdown(
                f"↑ Haussières: {tf_stat_data['bull_div']} | "
                f"↓ Baissières: {tf_stat_data['bear_div']}"
            )

with st.expander("Configuration", expanded=False):
    st.markdown(
        f"""
    **RSI Period:** {RSI_PERIOD} | **Oversold ≤** {RSI_OVERSOLD}
    | **Overbought ≥** {RSI_OVERBOUGHT}  
    **Bougies Forex:** H1=200 | H4=200 | Daily=150 | Weekly=100 | Monthly=60  
    **Bougies Restreints:** H1=200 | H4=200 | Daily=100 | Weekly=52 | Monthly=24  
    **Actifs restreints (historique limité) :**
    {', '.join(sorted(RESTRICTED_ASSETS))}  
    **Workers:** 6 Threads | **Semaphore:** 3 req. simultanées
    | **Timeout API:** {API_TIMEOUT}s | **Cache:** 300s  
    **Cooldown scan:** {SCAN_COOLDOWN}s
    | **Environment OANDA :** `{OANDA_ENVIRONMENT}`  
    **Assets:** {len(ASSETS)} instruments
    ({len(ASSETS) - len(RESTRICTED_ASSETS)} Forex + {len(RESTRICTED_ASSETS)} Restreints)
    """
    )
# =============================================================================
# END OF FILE app.py
# =============================================================================
     
