# ============================================================
# SCANNER SMC OANDA — v6.0  "INSTITUTIONAL GRADE"
# Auteur    : refonte complète sur base v5.7
# Style     : Swing Trading H4/D1
# Méthode   : Smart Money Concepts (SMC) authentique
#
# ARCHITECTURE :
#   1. Swing Points réels (HH/HL/LH/LL) — pas de fractals locaux
#   2. Market Structure Shift (MSS) authentique
#   3. Break of Structure (BOS) confirmé par clôture
#   4. Order Block (OB) institutionnel — dernière bougie opposée avant le BOS/MSS
#   5. Fair Value Gap (FVG) — imbalance 3 bougies autour du breakout
#   6. Score de confluence (0-100) — seuil configurable
#   7. Filtre multi-timeframe : H4 validé par D1, H1 validé par H4
#   8. Filtre session : London open + NY open privilégiés
#   9. Pipeline JSON pipeline-grade (types natifs, ISO 8601)
#
# PHILOSOPHIE :
#   Moins de signaux, mais des signaux qui ont une raison d'exister.
#   Un signal sans OB ET sans FVG n'a pas de zone d'entrée définie — il est ignoré.
#   Un signal H4 contre la structure D1 est ignoré.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import logging
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, asdict
from typing import Optional

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SCANNER_VERSION = "6.0"

# ===================== CONFIG =====================
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

# Timeframes principaux pour le swing trading
# HTF = contexte de tendance, TF = timeframe de signal
TIMEFRAMES = {
    "H4": {"gran": "H4", "htf": "D",  "count": 350},
    "D1": {"gran": "D",  "htf": "W",  "count": 250},
}

# Timeframe de confirmation (pour pipeline uniquement, pas de scan direct)
HTF_ONLY = {
    "W":  {"gran": "W", "count": 100},
    "D":  {"gran": "D", "count": 250},
}

VOLATILITY_STATIC = {
    "EUR_USD": "Basse",   "GBP_USD": "Basse",   "USD_JPY": "Basse",
    "USD_CHF": "Basse",   "USD_CAD": "Basse",
    "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",  "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne",  "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",   "GBP_CHF": "Haute",    "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",   "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",   "AUD_CAD": "Moyenne",  "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute",    "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",   "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "XAU_USD": "Très Haute", "SPX500_USD": "Très Haute",
    "NAS100_USD": "Très Haute", "US30_USD": "Très Haute",
}

# ── Paramètres SMC ─────────────────────────────────────────────────────────────
SWING_LOOKBACK   = 10   # bougies de chaque côté pour valider un swing point
SWING_HISTORY    = 50   # nombre de bougies pour chercher les swing points
OB_MAX_LOOKBACK  = 8    # bougies avant le breakout pour chercher l'OB
FVG_MAX_LOOKBACK = 5    # bougies autour du breakout pour chercher le FVG
MIN_FVG_PCT      = 0.03 # FVG minimum en % du prix (filtre micro-gaps)

# ── Score de confluence minimum pour générer un signal ─────────────────────────
# Score 0-100 : OB=30, FVG=25, HTF aligné=25, Session=10, Structure forte=10
MIN_CONFLUENCE_SCORE = 55   # en dessous → signal ignoré

# ── Filtre de distance ─────────────────────────────────────────────────────────
MAX_DIST_PCT = {"H4": 1.5, "D1": 4.0}

# ── Timeouts ───────────────────────────────────────────────────────────────────
SCAN_GLOBAL_TIMEOUT   = 240
FUTURE_RESULT_TIMEOUT = 25
MAX_WORKERS           = 8

# ── Colonnes UI / export ───────────────────────────────────────────────────────
DISPLAY_COLS = [
    "Paire", "TF", "Type", "Dir", "Score",
    "OB_Zone", "FVG_Zone", "Distance%",
    "HTF_Trend", "Session", "Statut", "Heure (UTC)"
]
EXPORT_COLS = DISPLAY_COLS


# ===================== DATACLASSES =====================

@dataclass
class SwingPoint:
    idx:   int
    price: float
    kind:  str   # "HH" | "HL" | "LH" | "LL"


@dataclass
class OrderBlock:
    top:      float
    bottom:   float
    idx:      int
    polarity: str   # "Bullish" | "Bearish"


@dataclass
class FairValueGap:
    top:      float
    bottom:   float
    idx:      int
    polarity: str   # "Bullish" | "Bearish"


@dataclass
class SMCSignal:
    # ── Identité ──────────────────────────────────────
    instrument:  str
    timeframe:   str
    direction:   str          # "Bullish" | "Bearish"
    sig_type:    str          # "MSS" | "BOS"
    signal_time: datetime

    # ── Niveaux ───────────────────────────────────────
    broken_level: float       # niveau de structure cassé
    close_price:  float
    distance_pct: Optional[float]

    # ── Confluence ────────────────────────────────────
    ob:              Optional[OrderBlock]
    fvg:             Optional[FairValueGap]
    htf_trend:       str      # "Bullish" | "Bearish" | "Range" | "Unknown"
    htf_aligned:     bool
    session:         str
    confluence_score: int

    # ── Contexte ──────────────────────────────────────
    volatility:   str
    statut:       str         # "Fresh" | "Aged" | "Stale"
    candles_since: int
    scan_time:    datetime


# ===================== API THREAD-SAFE =====================
_thread_local = threading.local()

def _get_api() -> API:
    if not hasattr(_thread_local, "api"):
        _thread_local.api = API(
            access_token=st.secrets["OANDA_ACCESS_TOKEN"],
            request_params={"timeout": 15}
        )
    return _thread_local.api

try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"Token OANDA manquant : {e}")
    st.stop()


# ===================== UTILITAIRES =====================

def instrument_precision(inst: str) -> int:
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return 2
    if "JPY" in inst:
        return 3
    return 5


def fmt_price(price: float, inst: str) -> str:
    return f"{price:.{instrument_precision(inst)}f}"


def fmt_zone(top: float, bot: float, inst: str) -> str:
    return f"{fmt_price(bot, inst)} – {fmt_price(top, inst)}"


def get_session(dt: datetime) -> str:
    h = dt.hour
    london = 7 <= h < 16
    ny     = 13 <= h < 22
    tokyo  = 0 <= h < 9
    if london and ny: return "London_NY_Overlap"
    if london:        return "London"
    if ny:            return "NewYork"
    if tokyo:         return "Tokyo"
    return "Off"


def is_session_premium(session: str) -> bool:
    """London open (07-09) et NY open (13-15) sont les sessions de liquidité maximale."""
    return session in ("London", "NewYork", "London_NY_Overlap")


def calc_distance_pct(level: float, close: float) -> Optional[float]:
    if level is None or close is None or np.isclose(level, 0, atol=1e-8):
        return None
    dist = abs(close - level) / abs(level) * 100
    return dist if dist <= 100 else None


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])
    ))
    if len(tr) < period * 2:
        return float("nan")
    return float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def calc_volatility(atr: float, df: pd.DataFrame, inst: str) -> str:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])
    ))
    if np.isnan(atr) or len(tr) < 10:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    window    = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isclose(median_tr, 0, atol=1e-10):
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr / median_tr
    if ratio >= 1.8: return "Très Haute"
    if ratio >= 1.2: return "Haute"
    if ratio >= 0.7: return "Moyenne"
    return "Basse"


# ===================== CANDLES =====================

def get_candles(inst: str, gran: str, count: int) -> Optional[pd.DataFrame]:
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran}
        )
        _get_api().request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 60:
            return None
        df = pd.DataFrame([{
            "time":  pd.to_datetime(c["time"], utc=True),
            "open":  float(c["mid"]["o"]),
            "high":  float(c["mid"]["h"]),
            "low":   float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except V20Error as e:
        logger.warning(f"V20Error [{inst}/{gran}] code={e.code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur get_candles [{inst}/{gran}]: {type(e).__name__}: {e}")
        return None


# ===================== SMC CORE — SWING POINTS =====================

def detect_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK,
                         history: int = SWING_HISTORY) -> list[SwingPoint]:
    """
    Détecte les vrais swing points (pivots significatifs) en regardant
    `lookback` bougies de chaque côté. Plus robuste que les fractals de 5 bougies.
    Classifie ensuite chaque pivot en HH/HL/LH/LL selon la structure.
    """
    h = df["high"].values
    l = df["low"].values
    n = len(h)

    start = max(lookback, n - history - lookback)
    end   = n - lookback - 1

    raw_highs: list[tuple[int, float]] = []
    raw_lows:  list[tuple[int, float]] = []

    for i in range(start, end):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == max(window_h):
            raw_highs.append((i, h[i]))
        if l[i] == min(window_l):
            raw_lows.append((i, l[i]))

    # Construire la séquence alternée highs/lows pour classifier HH/HL/LH/LL
    swings: list[SwingPoint] = []
    prev_high: Optional[float] = None
    prev_low:  Optional[float] = None

    # Fusion : interleave highs et lows par ordre d'index
    all_pivots = [(idx, price, "H") for idx, price in raw_highs] + \
                 [(idx, price, "L") for idx, price in raw_lows]
    all_pivots.sort(key=lambda x: x[0])

    for idx, price, kind in all_pivots:
        if kind == "H":
            label = "HH" if (prev_high is None or price > prev_high) else "LH"
            swings.append(SwingPoint(idx=idx, price=price, kind=label))
            prev_high = price
        else:
            label = "HL" if (prev_low is None or price > prev_low) else "LL"
            swings.append(SwingPoint(idx=idx, price=price, kind=label))
            prev_low = price

    return swings


def get_market_structure(swings: list[SwingPoint]) -> str:
    """
    Détermine la structure de marché à partir des swing points.
    Uptrend   = derniers pivots sont HH + HL
    Downtrend = derniers pivots sont LH + LL
    Range     = mixte
    """
    if len(swings) < 4:
        return "Unknown"
    recent = swings[-6:]
    highs  = [s for s in recent if s.kind in ("HH", "LH")]
    lows   = [s for s in recent if s.kind in ("HL", "LL")]
    if not highs or not lows:
        return "Unknown"
    last_high_is_hh = highs[-1].kind == "HH"
    last_low_is_hl  = lows[-1].kind  == "HL"
    if last_high_is_hh and last_low_is_hl:
        return "Bullish"
    last_high_is_lh = highs[-1].kind == "LH"
    last_low_is_ll  = lows[-1].kind  == "LL"
    if last_high_is_lh and last_low_is_ll:
        return "Bearish"
    return "Range"


# ===================== SMC CORE — MSS / BOS =====================

def detect_mss_bos(df: pd.DataFrame, swings: list[SwingPoint],
                   structure: str) -> tuple[Optional[str], Optional[str],
                                            Optional[float], Optional[int]]:
    """
    Détecte un MSS ou BOS authentique sur les dernières bougies.

    MSS (Market Structure Shift) = cassure CONTRE la structure courante
        → En Uptrend  : cassure d'un HL (swing low) → signal Bearish
        → En Downtrend: cassure d'un LH (swing high) → signal Bullish

    BOS (Break of Structure) = cassure DANS la structure courante
        → En Uptrend  : cassure d'un HH → signal Bullish (continuation)
        → En Downtrend: cassure d'un LL → signal Bearish (continuation)

    Règle de confirmation : la bougie DOIT clôturer au-delà du niveau.
    Le niveau doit avoir tenu au moins 3 bougies (évite les faux breakouts immédiats).

    Retourne : (sig_type, direction, broken_level, idx_breakout)
    """
    if structure == "Unknown" or not swings:
        return None, None, None, None

    c  = df["close"].values
    h  = df["high"].values
    l  = df["low"].values
    n  = len(c)

    # On cherche sur les 5 dernières bougies max (signal récent uniquement)
    for offset in range(5):
        idx = n - 1 - offset
        if idx < 3:
            break

        # ── Uptrend : cherche MSS (cassure d'un HL) ou BOS (cassure d'un HH) ──
        if structure == "Bullish":
            # MSS : clôture sous le dernier HL → retournement baissier
            hl_swings = [s for s in swings if s.kind == "HL" and s.idx < idx - 2]
            if hl_swings:
                last_hl = hl_swings[-1]
                if c[idx] < last_hl.price and c[idx - 1] >= last_hl.price:
                    return "MSS", "Bearish", last_hl.price, idx
            # BOS : clôture au-dessus du dernier HH → continuation haussière
            hh_swings = [s for s in swings if s.kind == "HH" and s.idx < idx - 2]
            if hh_swings:
                last_hh = hh_swings[-1]
                if c[idx] > last_hh.price and c[idx - 1] <= last_hh.price:
                    return "BOS", "Bullish", last_hh.price, idx

        # ── Downtrend : cherche MSS (cassure d'un LH) ou BOS (cassure d'un LL) ──
        elif structure == "Bearish":
            # MSS : clôture au-dessus du dernier LH → retournement haussier
            lh_swings = [s for s in swings if s.kind == "LH" and s.idx < idx - 2]
            if lh_swings:
                last_lh = lh_swings[-1]
                if c[idx] > last_lh.price and c[idx - 1] <= last_lh.price:
                    return "MSS", "Bullish", last_lh.price, idx
            # BOS : clôture sous le dernier LL → continuation baissière
            ll_swings = [s for s in swings if s.kind == "LL" and s.idx < idx - 2]
            if ll_swings:
                last_ll = ll_swings[-1]
                if c[idx] < last_ll.price and c[idx - 1] >= last_ll.price:
                    return "BOS", "Bearish", last_ll.price, idx

        # ── Range : uniquement les MSS (cassures des extrêmes de range) ──
        else:
            all_highs = [s for s in swings if s.kind in ("HH", "LH") and s.idx < idx - 2]
            all_lows  = [s for s in swings if s.kind in ("HL", "LL") and s.idx < idx - 2]
            if all_highs:
                range_high = max(all_highs, key=lambda s: s.price)
                if c[idx] > range_high.price and c[idx - 1] <= range_high.price:
                    return "MSS", "Bullish", range_high.price, idx
            if all_lows:
                range_low = min(all_lows, key=lambda s: s.price)
                if c[idx] < range_low.price and c[idx - 1] >= range_low.price:
                    return "MSS", "Bearish", range_low.price, idx

    return None, None, None, None


# ===================== SMC CORE — ORDER BLOCK =====================

def detect_order_block(df: pd.DataFrame, direction: str,
                       breakout_idx: int) -> Optional[OrderBlock]:
    """
    L'Order Block institutionnel = la DERNIÈRE bougie de couleur opposée
    au breakout, juste avant le mouvement impulsif.

    Pour un BOS/MSS Bullish : on cherche la dernière bougie baissière
    (close < open) dans les OB_MAX_LOOKBACK bougies précédant le breakout.

    Pour un BOS/MSS Bearish : on cherche la dernière bougie haussière
    (close > open) dans les OB_MAX_LOOKBACK bougies précédant le breakout.

    Règle de qualité : le corps de la bougie doit représenter > 40% de sa range.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    search_start = max(0, breakout_idx - OB_MAX_LOOKBACK)
    search_end   = breakout_idx  # on n'inclut pas la bougie de breakout

    last_ob_idx: Optional[int] = None

    for i in range(search_start, search_end):
        body = abs(c[i] - o[i])
        rng  = h[i] - l[i]
        if np.isclose(rng, 0, atol=1e-10) or (body / rng) < 0.40:
            continue

        if direction == "Bullish" and c[i] < o[i]:   # bougie baissière → OB haussier
            last_ob_idx = i
        elif direction == "Bearish" and c[i] > o[i]:  # bougie haussière → OB baissier
            last_ob_idx = i

    if last_ob_idx is None:
        return None

    # Zone OB = high/low de la bougie OB
    return OrderBlock(
        top      = float(h[last_ob_idx]),
        bottom   = float(l[last_ob_idx]),
        idx      = last_ob_idx,
        polarity = direction,
    )


# ===================== SMC CORE — FAIR VALUE GAP =====================

def detect_fvg(df: pd.DataFrame, direction: str,
               breakout_idx: int) -> Optional[FairValueGap]:
    """
    Fair Value Gap (imbalance) = gap entre la mèche de la bougie i-1
    et la mèche de la bougie i+1, autour du mouvement impulsif (bougie i).

    Structure : [bougie_avant] [bougie_impulsive] [bougie_après]
    FVG Bullish : high[i-1] < low[i+1]  → gap non comblé au-dessus
    FVG Bearish : low[i-1]  > high[i+1] → gap non comblé en-dessous

    On cherche dans les FVG_MAX_LOOKBACK bougies autour du breakout.
    Filtre : le gap doit être > MIN_FVG_PCT% du prix (évite les micro-gaps).
    """
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)

    search_start = max(1, breakout_idx - FVG_MAX_LOOKBACK)
    search_end   = min(n - 1, breakout_idx + 1)

    best_fvg: Optional[FairValueGap] = None
    best_size = 0.0

    for i in range(search_start, search_end):
        if i + 1 >= n:
            break

        if direction == "Bullish":
            gap_bottom = h[i - 1]
            gap_top    = l[i + 1]
            if gap_top > gap_bottom:
                size = gap_top - gap_bottom
                mid  = (gap_top + gap_bottom) / 2
                if mid > 0 and (size / mid * 100) >= MIN_FVG_PCT and size > best_size:
                    best_fvg  = FairValueGap(top=gap_top, bottom=gap_bottom,
                                              idx=i, polarity="Bullish")
                    best_size = size

        elif direction == "Bearish":
            gap_top    = l[i - 1]
            gap_bottom = h[i + 1]
            if gap_top > gap_bottom:
                size = gap_top - gap_bottom
                mid  = (gap_top + gap_bottom) / 2
                if mid > 0 and (size / mid * 100) >= MIN_FVG_PCT and size > best_size:
                    best_fvg  = FairValueGap(top=gap_top, bottom=gap_bottom,
                                              idx=i, polarity="Bearish")
                    best_size = size

    return best_fvg


# ===================== SMC CORE — CONFLUENCE SCORE =====================

def compute_confluence_score(ob: Optional[OrderBlock], fvg: Optional[FairValueGap],
                              htf_aligned: bool, session: str,
                              sig_type: str, direction: str,
                              structure: str) -> int:
    """
    Score de confluence 0-100.
    Un signal ne passe le filtre que si score >= MIN_CONFLUENCE_SCORE (55).

    Décomposition :
      OB présent          +30  (zone d'entrée institutionnelle définie)
      FVG présent         +25  (imbalance = zone de retour probable)
      HTF aligné          +25  (filtre de tendance supérieure)
      Session premium     +10  (London/NY open = liquidité maximale)
      MSS (vs BOS)        +10  (signal de retournement = plus fort)
    """
    score = 0
    if ob  is not None: score += 30
    if fvg is not None: score += 25
    if htf_aligned:     score += 25
    if is_session_premium(session): score += 10
    if sig_type == "MSS":           score += 10
    return min(score, 100)


# ===================== SMC CORE — STATUT =====================

def compute_statut(candles_since: int, tf: str) -> str:
    thresholds = {
        "H4": {"Fresh": 3,  "Aged": 8},
        "D1": {"Fresh": 2,  "Aged": 5},
    }
    t = thresholds.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_since <= t["Fresh"]: return "Fresh"
    if candles_since <= t["Aged"]:  return "Aged"
    return "Stale"


# ===================== HTF CONTEXT =====================

def get_htf_trend(inst: str, htf_gran: str, htf_count: int) -> str:
    """
    Récupère la structure HTF (Daily pour H4, Weekly pour D1).
    Utilise les mêmes swing points SMC pour la cohérence.
    Retourne "Bullish" | "Bearish" | "Range" | "Unknown".
    """
    df_htf = get_candles(inst, htf_gran, htf_count)
    if df_htf is None or len(df_htf) < 40:
        return "Unknown"
    swings_htf  = detect_swing_points(df_htf, lookback=5, history=30)
    structure   = get_market_structure(swings_htf)
    return structure


# ===================== PIPELINE PRINCIPAL =====================

def scan_instrument(inst: str, tf_name: str, tf_config: dict,
                    scan_time: datetime) -> Optional[SMCSignal]:
    """
    Pipeline complet pour un instrument + timeframe.
    Retourne un SMCSignal si toutes les conditions sont remplies, None sinon.
    """
    gran      = tf_config["gran"]
    htf_gran  = tf_config["htf"]
    count     = tf_config["count"]
    htf_count = HTF_ONLY[htf_gran]["count"]

    # ── 1. Récupération des données ───────────────────────────────────────────
    df = get_candles(inst, gran, count)
    if df is None or len(df) < 80:
        return None

    # ── 2. Détection des swing points et structure ────────────────────────────
    swings    = detect_swing_points(df)
    structure = get_market_structure(swings)

    # ── 3. Détection MSS / BOS ────────────────────────────────────────────────
    sig_type, direction, broken_level, breakout_idx = detect_mss_bos(df, swings, structure)
    if not sig_type:
        return None

    # ── 4. Contexte HTF (filtre de tendance supérieure) ───────────────────────
    htf_trend   = get_htf_trend(inst, htf_gran, htf_count)
    htf_aligned = (
        (direction == "Bullish" and htf_trend == "Bullish") or
        (direction == "Bearish" and htf_trend == "Bearish")
    )

    # ── 5. Order Block ────────────────────────────────────────────────────────
    df_context = df.iloc[:breakout_idx + 1]
    ob = detect_order_block(df_context, direction, breakout_idx)

    # ── 6. Fair Value Gap ─────────────────────────────────────────────────────
    fvg = detect_fvg(df_context, direction, breakout_idx)

    # ── 7. Session ────────────────────────────────────────────────────────────
    signal_time = df.index[breakout_idx]
    session     = get_session(signal_time)

    # ── 8. Score de confluence ────────────────────────────────────────────────
    score = compute_confluence_score(ob, fvg, htf_aligned, session, sig_type,
                                     direction, structure)
    if score < MIN_CONFLUENCE_SCORE:
        logger.info(f"Signal ignoré {inst}/{tf_name}: score {score} < {MIN_CONFLUENCE_SCORE}")
        return None

    # ── 9. Filtre de distance ─────────────────────────────────────────────────
    close_price  = float(df["close"].iloc[-1])
    distance_pct = calc_distance_pct(broken_level, close_price)
    max_dist     = MAX_DIST_PCT.get(tf_name, 3.0)
    if distance_pct is not None and distance_pct > max_dist:
        logger.info(f"Signal ignoré {inst}/{tf_name}: distance {distance_pct:.3f}% > {max_dist}%")
        return None

    # ── 10. Statut et volatilité ──────────────────────────────────────────────
    candles_since = (len(df) - 1) - breakout_idx
    statut        = compute_statut(candles_since, tf_name)
    atr           = calc_atr(df_context)
    volatility    = calc_volatility(atr, df_context, inst)

    return SMCSignal(
        instrument    = inst,
        timeframe     = tf_name,
        direction     = direction,
        sig_type      = sig_type,
        signal_time   = signal_time,
        broken_level  = float(broken_level),
        close_price   = close_price,
        distance_pct  = distance_pct,
        ob            = ob,
        fvg           = fvg,
        htf_trend     = htf_trend,
        htf_aligned   = htf_aligned,
        session       = session,
        confluence_score = score,
        volatility    = volatility,
        statut        = statut,
        candles_since = candles_since,
        scan_time     = scan_time,
    )


# ===================== PAYLOAD JSON PIPELINE =====================

def build_pipeline_payload(sig: SMCSignal) -> dict:
    prec = instrument_precision(sig.instrument)
    return {
        "signal_id":       f"{sig.instrument}__{sig.timeframe}__{sig.signal_time.strftime('%Y%m%dT%H%M')}",
        "scanner_version": SCANNER_VERSION,
        "generated_at":    sig.scan_time.isoformat(),

        "pair":            sig.instrument.replace("_", "/"),
        "pair_oanda":      sig.instrument,
        "timeframe":       sig.timeframe,

        "type":            sig.sig_type,
        "direction":       sig.direction,
        "is_bullish":      sig.direction == "Bullish",
        "order":           "buy" if sig.direction == "Bullish" else "sell",
        "htf_trend":       sig.htf_trend,
        "htf_aligned":     sig.htf_aligned,
        "status":          sig.statut,

        "confluence_score": sig.confluence_score,

        "level":           round(sig.broken_level, prec),
        "close_price":     round(sig.close_price, prec),
        "distance_pct":    round(sig.distance_pct, 4) if sig.distance_pct else None,

        "order_block": {
            "top":    round(sig.ob.top,    prec),
            "bottom": round(sig.ob.bottom, prec),
        } if sig.ob else None,

        "fair_value_gap": {
            "top":    round(sig.fvg.top,    prec),
            "bottom": round(sig.fvg.bottom, prec),
        } if sig.fvg else None,

        "volatility":       sig.volatility,
        "signal_time":      sig.signal_time.isoformat(),
        "session":          sig.session,
        "candles_elapsed":  sig.candles_since,
    }


# ===================== EXPORT PDF =====================

def create_pdf(rows: list[dict]) -> io.BytesIO:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                               leftMargin=15, rightMargin=15,
                               topMargin=30, bottomMargin=30)
    elements = []
    styles   = getSampleStyleSheet()
    elements.append(Paragraph(f"SMC Scanner v{SCANNER_VERSION} — Signaux qualifiés", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC "
        f"| Confluence minimum : {MIN_CONFLUENCE_SCORE}/100",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 16))

    if not rows:
        elements.append(Paragraph("Aucun signal qualifié.", styles["Normal"]))
        doc.build(elements)
        buffer.seek(0)
        return buffer

    headers = ["Paire", "TF", "Type", "Dir", "Score", "OB Zone", "FVG Zone",
               "Dist%", "HTF", "Session", "Statut", "Heure UTC"]
    data    = [headers]
    for r in rows:
        data.append([
            r.get("Paire", ""),    r.get("TF", ""),      r.get("Type", ""),
            r.get("Dir", ""),      str(r.get("Score", "")),
            r.get("OB_Zone", ""),  r.get("FVG_Zone", ""), r.get("Distance%", ""),
            r.get("HTF_Trend", ""), r.get("Session", ""), r.get("Statut", ""),
            r.get("Heure (UTC)", ""),
        ])

    col_w = [52, 32, 38, 42, 36, 85, 85, 45, 55, 85, 40, 90]
    table = Table(data, colWidths=col_w, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor("#0f172a")),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0),  8),
        ('FONTSIZE',      (0, 1), (-1, -1), 7),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.HexColor("#334155")),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== SIGNAL → ROW UI =====================

def signal_to_row(sig: SMCSignal) -> dict:
    inst = sig.instrument
    return {
        "Paire":       inst.replace("_", "/"),
        "TF":          sig.timeframe,
        "Type":        sig.sig_type,
        "Dir":         "↑ " + sig.direction if sig.direction == "Bullish" else "↓ " + sig.direction,
        "Score":       sig.confluence_score,
        "OB_Zone":     fmt_zone(sig.ob.top, sig.ob.bottom, inst) if sig.ob else "—",
        "FVG_Zone":    fmt_zone(sig.fvg.top, sig.fvg.bottom, inst) if sig.fvg else "—",
        "Distance%":   f"{sig.distance_pct:.3f}%" if sig.distance_pct else "N/A",
        "HTF_Trend":   sig.htf_trend,
        "Session":     sig.session,
        "Statut":      sig.statut,
        "Heure (UTC)": sig.signal_time.strftime("%Y-%m-%d %H:%M"),
        # colonnes cachées pour tri
        "_time_sort":  sig.signal_time,
        "_score_sort": sig.confluence_score,
        "_htf_align":  sig.htf_aligned,
    }


# ===================== UI =====================
st.set_page_config(page_title="SMC Scanner v6", layout="wide")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-family: 'Courier New', monospace;
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 18px;
        text-align: center;
    }
</style>
<div class="main-title">◈ SMC INSTITUTIONAL SCANNER v6.0</div>
<div class="sub-title">Swing H4/D1 · MSS · BOS · Order Block · Fair Value Gap · Confluence Filter</div>
""", unsafe_allow_html=True)

# ── Paramètres sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    min_score    = st.slider("Score confluence minimum", 40, 90, MIN_CONFLUENCE_SCORE, 5,
                              help="En dessous de ce score, le signal est ignoré")
    htf_only     = st.checkbox("HTF aligné obligatoire", value=True,
                                help="N'affiche que les signaux alignés avec le timeframe supérieur")
    show_stale   = st.checkbox("Afficher les signaux Stale", value=False)
    selected_tfs = st.multiselect("Timeframes", list(TIMEFRAMES.keys()),
                                   default=list(TIMEFRAMES.keys()))

    st.markdown("---")
    st.markdown(f"**Paires scannées :** {len(INSTRUMENTS)}")
    st.markdown(f"**Combinaisons :** {len(INSTRUMENTS) * len(selected_tfs)}")
    st.markdown(f"**Version :** {SCANNER_VERSION}")
    st.markdown("---")
    st.markdown("**Scoring :**")
    st.markdown("🟦 OB présent : +30")
    st.markdown("🟩 FVG présent : +25")
    st.markdown("🟨 HTF aligné : +25")
    st.markdown("🟧 Session premium : +10")
    st.markdown("🟥 MSS (vs BOS) : +10")

if "scanning" not in st.session_state:
    st.session_state.scanning = False

# ── Bouton scan ───────────────────────────────────────────────────────────────
if st.button("🔍  Lancer le Scan SMC", type="primary",
             use_container_width=True, disabled=st.session_state.scanning):

    st.session_state.scanning  = True
    st.session_state.scan_time = datetime.now(timezone.utc)
    scan_time  = st.session_state.scan_time
    tfs_to_run = {k: v for k, v in TIMEFRAMES.items() if k in selected_tfs}
    n_combos   = len(INSTRUMENTS) * len(tfs_to_run)

    try:
        with st.spinner(f"Scan SMC en cours — {n_combos} combinaisons…"):
            signals:  list[SMCSignal] = []
            payloads: list[dict]      = []
            errors:   list[str]       = []

            tasks = [
                (inst, tf_name, tf_cfg)
                for inst    in INSTRUMENTS
                for tf_name, tf_cfg in tfs_to_run.items()
            ]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(scan_instrument, inst, tf_name, tf_cfg, scan_time):
                        (inst, tf_name)
                    for inst, tf_name, tf_cfg in tasks
                }

                try:
                    for future in as_completed(future_map, timeout=SCAN_GLOBAL_TIMEOUT):
                        inst, tf_name = future_map[future]
                        try:
                            result = future.result(timeout=FUTURE_RESULT_TIMEOUT)
                            if result is not None:
                                signals.append(result)
                                if result.statut in ("Fresh", "Aged"):
                                    payloads.append(build_pipeline_payload(result))
                        except FuturesTimeoutError:
                            errors.append(f"{inst}/{tf_name}: timeout")
                            future.cancel()
                        except Exception as e:
                            errors.append(f"{inst}/{tf_name}: {e}")
                except FuturesTimeoutError:
                    st.warning("Timeout global atteint — résultats partiels.")

        if errors:
            with st.expander(f"⚠️ {len(errors)} erreur(s) de scan"):
                for e in errors[:20]:
                    st.text(e)

        st.session_state.signals  = signals
        st.session_state.payloads = payloads

        n_htf = sum(1 for s in signals if s.htf_aligned)
        st.success(
            f"✅ Scan terminé — **{len(signals)} signaux qualifiés** "
            f"({n_htf} alignés HTF) sur {n_combos} combinaisons "
            f"| {len(payloads)} dans le pipeline JSON"
        )

    except Exception as e:
        st.error(f"Erreur critique : {e}")
        logger.exception("Erreur critique scan SMC")
    finally:
        st.session_state.scanning = False


# ===================== AFFICHAGE =====================
if "signals" in st.session_state and st.session_state.signals:
    signals: list[SMCSignal] = st.session_state.signals
    payloads = st.session_state.get("payloads", [])
    ts       = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Filtres dynamiques ────────────────────────────────────────────────────
    filtered = [s for s in signals if s.confluence_score >= min_score]
    if htf_only:
        filtered = [s for s in filtered if s.htf_aligned]
    if not show_stale:
        filtered = [s for s in filtered if s.statut != "Stale"]

    # ── Métriques ─────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Signaux qualifiés",     len(filtered))
    m2.metric("HTF alignés",           sum(1 for s in filtered if s.htf_aligned))
    m3.metric("MSS (retournements)",   sum(1 for s in filtered if s.sig_type == "MSS"))
    m4.metric("Avec OB + FVG",         sum(1 for s in filtered if s.ob and s.fvg))
    m5.metric("Score moyen",
              f"{np.mean([s.confluence_score for s in filtered]):.0f}/100"
              if filtered else "—")

    st.divider()

    # ── Tableau ───────────────────────────────────────────────────────────────
    if filtered:
        rows = [signal_to_row(s) for s in filtered]
        df_display = (
            pd.DataFrame(rows)
            .sort_values(["_score_sort", "_time_sort"], ascending=[False, False])
            .drop(columns=["_time_sort", "_score_sort", "_htf_align"])
            .reset_index(drop=True)
        )

        def style_type(val):
            if val == "MSS": return "color:#f97316;font-weight:800"
            return "color:#60a5fa;font-weight:600"

        def style_dir(val):
            if "Bullish" in str(val): return "color:#4ade80;font-weight:700"
            if "Bearish" in str(val): return "color:#f87171;font-weight:700"
            return ""

        def style_score(val):
            try:
                v = int(val)
                if v >= 80: return "color:#4ade80;font-weight:800"
                if v >= 65: return "color:#facc15;font-weight:700"
                return "color:#94a3b8"
            except: return ""

        def style_statut(val):
            if val == "Fresh": return "color:#4ade80;font-weight:700"
            if val == "Aged":  return "color:#fb923c;font-weight:700"
            return "color:#f87171"

        def style_htf(val):
            if val == "Bullish":  return "color:#4ade80"
            if val == "Bearish":  return "color:#f87171"
            if val == "Range":    return "color:#facc15"
            return "color:#94a3b8"

        def style_session(val):
            if "Overlap" in str(val): return "color:#c084fc;font-weight:700"
            if val in ("London", "NewYork"): return "color:#38bdf8"
            return "color:#94a3b8"

        cols_show = [c for c in DISPLAY_COLS if c in df_display.columns]
        st.dataframe(
            df_display[cols_show].style
            .map(style_type,   subset=["Type"])
            .map(style_dir,    subset=["Dir"])
            .map(style_score,  subset=["Score"])
            .map(style_statut, subset=["Statut"])
            .map(style_htf,    subset=["HTF_Trend"])
            .map(style_session,subset=["Session"]),
            hide_index=True,
            use_container_width=True,
            height=min(600, 60 + len(df_display) * 38),
        )
    else:
        st.info("Aucun signal ne satisfait les filtres actuels. Réduisez le score minimum ou désactivez 'HTF aligné obligatoire'.")

    st.divider()

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("#### Exports")
    c1, c2, c3 = st.columns(3)

    rows_export = [signal_to_row(s) for s in filtered]

    with c1:
        df_csv = pd.DataFrame(rows_export).drop(
            columns=["_time_sort", "_score_sort", "_htf_align"], errors="ignore"
        )
        st.download_button("⬇️ CSV", df_csv.to_csv(index=False).encode(),
                           f"smc_{ts}.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ PDF", create_pdf(rows_export),
                           f"smc_{ts}.pdf", "application/pdf", use_container_width=True)
    with c3:
        scan_time_meta = st.session_state.get("scan_time", datetime.now(timezone.utc))
        pipeline_json  = json.dumps({
            "meta": {
                "scanner_version":   SCANNER_VERSION,
                "generated_at":      scan_time_meta.isoformat(),
                "signal_count":      len(payloads),
                "min_confluence":    min_score,
                "htf_filter_active": htf_only,
            },
            "signals": payloads,
        }, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        st.download_button("⬇️ JSON Pipeline", pipeline_json,
                           f"smc_pipeline_{ts}.json", "application/json",
                           use_container_width=True)

    # ── Aperçu JSON ───────────────────────────────────────────────────────────
    if payloads:
        with st.expander(f"Aperçu JSON Pipeline — premier signal"):
            st.json(payloads[0])

elif "signals" in st.session_state and not st.session_state.signals:
    st.info("Aucun signal SMC qualifié détecté sur ce scan.")
