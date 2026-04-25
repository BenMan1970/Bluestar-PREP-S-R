# app.py → v5.0 — Optimisation CASCADE ENGINE
# [1] Colonnes Niveau + Distance% (niveau fractal cassé + écart prix actuel)
# [2] Monthly exclu du scan — Weekly conservé pour analyse personnelle
# [3] Noms de paires normalisés sans slash dans PDF/CSV (EURUSD) — Streamlit conserve EUR/USD
# [4] Filtre Stale : Fresh+Aged uniquement dans PDF/CSV — Stale visible Streamlit uniquement

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== CONFIG =====================
INSTRUMENTS = [
    # 28 paires Forex
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    # 5 indices et métaux
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

# Volatilité statique — fallback si ATR non disponible
VOLATILITY_STATIC = {
    "EUR_USD": "Basse",  "GBP_USD": "Basse",  "USD_JPY": "Basse",
    "USD_CHF": "Basse",  "USD_CAD": "Basse",
    "AUD_USD": "Moyenne","NZD_USD": "Moyenne", "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne","EUR_CHF": "Moyenne", "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne","EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",  "GBP_CHF": "Haute",  "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",  "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",  "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne","CAD_JPY": "Haute",   "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",  "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne","NZD_CHF": "Haute",
    "DE30_EUR":  "Très Haute", "XAU_USD":    "Très Haute",
    "SPX500_USD":"Très Haute", "NAS100_USD": "Très Haute",
    "US30_USD":  "Très Haute",
}

# [2] Monthly retiré — Weekly conservé pour analyse personnelle
TIMEFRAMES      = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LEN     = {"H1": 5, "H4": 6, "D1": 7, "Weekly": 8}
TF_HOURS        = {"H1": 1, "H4": 4, "D1": 24, "Weekly": 168}

TF_STATUT = {
    "H1":     {"Fresh": 3,  "Aged": 8},
    "H4":     {"Fresh": 2,  "Aged": 5},
    "D1":     {"Fresh": 2,  "Aged": 5},
    "Weekly": {"Fresh": 2,  "Aged": 4},
}

TF_LOOKBACK     = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
TF_EMA_LOOKBACK = {"H1": 5, "H4": 5, "D1": 10, "Weekly": 15}

# Colonnes Streamlit (affichage complet avec slash)
DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

# Colonnes exports PDF/CSV (paire normalisée sans slash, Stale exclus)
EXPORT_COLS = [
    "Paire", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

# ===================== API =====================
try:
    api = API(
        access_token=st.secrets["OANDA_ACCESS_TOKEN"],
        request_params={"timeout": 10}
    )
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()

# ===================== FONCTIONS =====================
def get_candles(inst, gran):
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": 500, "granularity": gran}
        )
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
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
    except Exception:
        return None


def calc_atr(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return np.nan
    return float(np.mean(tr[-period:]))


def atr_to_volatility(atr_val, inst, df):
    if np.isnan(atr_val) or len(df) < 28:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if median_tr == 0:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8:
        return "Très Haute"
    if ratio >= 1.2:
        return "Haute"
    if ratio >= 0.7:
        return "Moyenne"
    return "Basse"


def compute_bb_width(df, length=20, std=2):
    close = df["close"]
    if len(close) < length * 2:
        return "N/A"
    sma     = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper   = sma + std * std_dev
    lower   = sma - std * std_dev
    bb_w    = (upper - lower) / sma
    bb_avg  = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or avg_last == 0:
        return "N/A"
    pct = ((bb_w - bb_avg) / bb_avg * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct < -25:
        return f"{sign}{pct:.0f}%_Squeeze"
    if pct > 25:
        return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(time_sig, tf):
    try:
        now     = datetime.now(timezone.utc)
        sig_utc = time_sig if time_sig.tzinfo is not None else time_sig.replace(tzinfo=timezone.utc)
        elapsed_h       = (now - sig_utc).total_seconds() / 3600
        candles_elapsed = elapsed_h / TF_HOURS.get(tf, 1)
        thresholds      = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
        if candles_elapsed <= thresholds["Fresh"]:
            return "Fresh"
        if candles_elapsed <= thresholds["Aged"]:
            return "Aged"
        return "Stale"
    except Exception:
        return "N/A"


def get_trend_context(df, tf):
    close = df["close"]
    if len(close) < 55:
        return "Unknown"
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    e20_last = ema20.iloc[-1]
    e50_last = ema50.iloc[-1]
    lookback = TF_EMA_LOOKBACK.get(tf, 5)
    e20_prev = ema20.iloc[-(lookback + 1)] if len(ema20) > lookback else ema20.iloc[0]
    if e20_last > e50_last and e20_last > e20_prev:
        return "Uptrend"
    if e20_last < e50_last and e20_last < e20_prev:
        return "Downtrend"
    return "Range"


def detect_choch(df, tf):
    """
    Retourne un 7-tuple :
      (sig_raw, time_sig, force, idx_cur, trend, niveau_casse, close_actuel)

    [1] niveau_casse : prix exact du fractal rompu (last_high ou last_low)
    [1] close_actuel : close de la dernière bougie disponible pour Distance%
    """
    length   = FRACTAL_LEN.get(tf, 5)
    p        = length // 2
    lookback = TF_LOOKBACK.get(tf, 3)

    h_all = df["high"].values
    l_all = df["low"].values
    c_all = df["close"].values
    o_all = df["open"].values

    min_len = p * 2 + 1 + lookback + 1
    if len(c_all) < min_len:
        return None, None, None, None, None, None, None

    candle_ranges = h_all[1:] - l_all[1:]
    p25 = float(np.percentile(candle_ranges, 25)) if len(candle_ranges) >= 10 else None
    p75 = float(np.percentile(candle_ranges, 75)) if len(candle_ranges) >= 10 else None

    def get_force(rng):
        if p25 is None or p75 is None:
            return "Moyen"
        if rng >= p75:
            return "Fort"
        if rng <= p25:
            return "Faible"
        return "Moyen"

    def is_valid_breakout_candle(idx):
        rng = h_all[idx] - l_all[idx]
        if rng == 0:
            return False
        body = abs(c_all[idx] - o_all[idx])
        return (body / rng) >= 0.3

    for offset in range(lookback):
        idx_cur  = len(df) - 1 - offset
        idx_prev = idx_cur - 1

        if idx_prev < p:
            break

        h = h_all[:idx_cur + 1]
        l = l_all[:idx_cur + 1]
        c = c_all[:idx_cur + 1]

        last_high = None
        last_low  = None
        for i in range(p, len(h) - p):
            if h[i] == max(h[i - p:i + p + 1]):
                last_high = h[i]
            if l[i] == min(l[i - p:i + p + 1]):
                last_low = l[i]

        if last_high is None and last_low is None:
            continue

        breakout_range = h_all[idx_cur] - l_all[idx_cur]

        if not is_valid_breakout_candle(idx_cur):
            continue

        sig_raw      = None
        time_sig     = None
        force        = None
        niveau_casse = None

        if last_high is not None and c[idx_cur] > last_high and c[idx_prev] < last_high:
            sig_raw      = "Bullish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = float(last_high)

        elif last_low is not None and c[idx_cur] < last_low and c[idx_prev] > last_low:
            sig_raw      = "Bearish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = float(last_low)

        if sig_raw is None:
            continue

        trend = get_trend_context(df.iloc[:idx_cur + 1], tf)
        if trend == "Range" and force == "Faible":
            continue

        close_actuel = float(c_all[-1])
        return sig_raw, time_sig, force, idx_cur, trend, niveau_casse, close_actuel

    return None, None, None, None, None, None, None


def format_niveau(niveau, inst):
    """Décimales adaptées par type d'instrument."""
    if niveau is None:
        return "N/A"
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return f"{niveau:.2f}"
    if "JPY" in inst:
        return f"{niveau:.3f}"
    return f"{niveau:.5f}"


def format_distance(niveau, close_actuel, inst):
    """
    [1] Distance% = |close_actuel - niveau| / niveau × 100
    Seuils CASCADE ENGINE : ≤0.15% excellent / 0.15–0.40% acceptable / >0.40% raté
    """
    if niveau is None or close_actuel is None or niveau == 0:
        return "N/A"
    dist = abs(close_actuel - niveau) / niveau * 100
    return f"{dist:.3f}%"


def normalize_pair(inst_slash):
    """[3] Normalisation v1.5 CASCADE ENGINE — supprime '/' pour exports."""
    return inst_slash.replace("/", "")


def get_inst_type(inst):
    """
    Classifie l'instrument pour calibrer les seuils Distance%.
    Retourne : 'indice' | 'jpy' | 'forex'
    """
    inst_up = inst.upper().replace("/", "_")
    if any(k in inst_up for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return "indice"
    if "JPY" in inst_up:
        return "jpy"
    return "forex"


# Seuils Distance% par type d'instrument
# Forex standard : pip ~0.0001 → seuils serrés
# JPY cross      : pip ~0.01   → légèrement élargis
# Indices/métaux : moves en points → seuils ×3–4 vs forex
DISTANCE_THRESHOLDS = {
    "forex":  (0.15, 0.40),   # vert ≤0.15% | orange ≤0.40% | rouge >0.40%
    "jpy":    (0.20, 0.50),   # vert ≤0.20% | orange ≤0.50% | rouge >0.50%
    "indice": (0.50, 1.20),   # vert ≤0.50% | orange ≤1.20% | rouge >1.20%
}


# ===================== PDF =====================
def create_pdf(df_export):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40
    )
    elements = []
    styles   = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC"
        " — Fresh & Aged uniquement — Monthly exclu",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    data         = [cols_present] + df_export[cols_present].values.tolist()
    col_widths   = [65, 48, 42, 42, 82, 68, 52, 58, 45, 90, 45, 105]
    col_widths   = col_widths[:len(cols_present)]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.beige]),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== PNG =====================
def generate_png(df, display_cols):
    fig, ax = plt.subplots(figsize=(22, max(5, len(df) * 0.35)))
    ax.axis('off')
    disp = df[[c for c in display_cols if c in df.columns]]
    tbl  = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>"
    "Scanner Change of Character (CHoCH) — v5.0</h1>",
    unsafe_allow_html=True
)

if "scanning" not in st.session_state:
    st.session_state.scanning = False

if st.button(
    "Lancer le Scan",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.scanning
):
    st.session_state.scanning = True
    n_combos = len(INSTRUMENTS) * len(TIMEFRAMES)

    with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
        results = []
        errors  = []

        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {
                executor.submit(get_candles, inst, tf_code): (inst, tf_name)
                for inst in INSTRUMENTS
                for tf_name, tf_code in TIMEFRAMES.items()
            }
            for future in as_completed(futures):
                inst, tf_name = futures[future]
                try:
                    df = future.result()
                except Exception as e:
                    errors.append(f"{inst}/{tf_name}: {e}")
                    continue

                if df is not None:
                    sig_raw, time_sig, strength, idx_sig, trend, niveau_casse, close_actuel = \
                        detect_choch(df, tf_name)

                    if sig_raw:
                        atr_val    = calc_atr(df)
                        volatilite = atr_to_volatility(atr_val, inst, df)
                        df_sig     = df.iloc[:idx_sig + 1] if idx_sig is not None else df

                        is_bos = (
                            (trend == "Uptrend"   and sig_raw == "Bullish") or
                            (trend == "Downtrend" and sig_raw == "Bearish")
                        )
                        type_label   = "BOS" if is_bos else "CHoCH"
                        signal_label = f"{sig_raw} {type_label}"
                        inst_display = inst.replace("_", "/")
                        inst_export  = normalize_pair(inst_display)
                        statut       = compute_statut(time_sig, tf_name)

                        results.append({
                            "Instrument":  inst_display,       # Streamlit (avec slash)
                            "Paire":       inst_export,        # Export normalisé (sans slash)
                            "_inst_type":  get_inst_type(inst),# Interne — seuils Distance%
                            "Timeframe":   tf_name,
                            "Type":        type_label,
                            "Ordre":       "Achat" if sig_raw == "Bullish" else "Vente",
                            "Signal":      signal_label,
                            "Niveau":      format_niveau(niveau_casse, inst),
                            "Distance%":   format_distance(niveau_casse, close_actuel, inst),
                            "Volatilité":  volatilite,
                            "Force":       strength or "Moyen",
                            "BB_Width":    compute_bb_width(df_sig),
                            "Statut":      statut,
                            "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M"),
                        })

        if errors:
            st.warning(f"{len(errors)} erreur(s) silencieuse(s) : {'; '.join(errors[:5])}")

        if results:
            df_result = (
                pd.DataFrame(results)
                .sort_values("Heure (UTC)", ascending=False)
                .drop_duplicates(subset=["Instrument", "Timeframe"], keep="first")
                .reset_index(drop=True)
            )
            st.session_state.df      = df_result
            st.session_state.png_buf = None
            st.success(f"Scan terminé – {len(df_result)} signaux sur {n_combos} combinaisons !")
        else:
            st.info("Aucun signal CHoCH/BOS récent détecté")

    st.session_state.scanning = False

# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df_all = st.session_state.df.copy()
    ts     = datetime.now().strftime("%Y%m%d_%H%M")

    # [4] Exports = Fresh + Aged uniquement — Stale visible Streamlit, exclu PDF/CSV
    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()

    if st.session_state.get("png_buf") is None:
        st.session_state.png_buf = generate_png(df_all, DISPLAY_COLS)

    n_stale = len(df_all[df_all["Statut"] == "Stale"])
    if n_stale > 0:
        st.info(
            f"{n_stale} signal(s) Stale visible(s) dans le tableau — "
            f"exclus du PDF/CSV (Fresh & Aged uniquement dans les exports)."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "CSV",
            df_export[[c for c in EXPORT_COLS if c in df_export.columns]]
                .to_csv(index=False).encode(),
            f"choch_{ts}.csv", "text/csv"
        )
    with col2:
        st.session_state.png_buf.seek(0)
        st.download_button(
            "PNG",
            st.session_state.png_buf,
            f"choch_{ts}.png", "image/png"
        )
    with col3:
        st.download_button(
            "PDF",
            create_pdf(df_export),
            f"choch_signaux_{ts}.pdf", "application/pdf"
        )

    def style_bb(val):
        if "Squeeze"   in str(val): return "color:#ff9800;font-weight:bold"
        if "Expansion" in str(val): return "color:#ab47bc;font-weight:bold"
        return "color:#90a4ae"

    def style_distance_row(row):
        """
        Seuils Distance% calibrés par type d'instrument (colonne _inst_type) :
          forex  : vert ≤0.15% | orange ≤0.40% | rouge >0.40%
          jpy    : vert ≤0.20% | orange ≤0.50% | rouge >0.50%
          indice : vert ≤0.50% | orange ≤1.20% | rouge >1.20%
        Retourne un dict de styles pour toutes les colonnes de la ligne —
        seule Distance% reçoit un style non-vide.
        """
        styles_row = {col: "" for col in row.index}
        try:
            v         = float(str(row["Distance%"]).replace("%", ""))
            itype     = row.get("_inst_type", "forex")
            low, high = DISTANCE_THRESHOLDS.get(itype, (0.15, 0.40))
            if v <= low:
                styles_row["Distance%"] = "color:#00c853;font-weight:bold"
            elif v <= high:
                styles_row["Distance%"] = "color:#ff9800;font-weight:bold"
            else:
                styles_row["Distance%"] = "color:#ff5252;font-weight:bold"
        except Exception:
            styles_row["Distance%"] = "color:#90a4ae"
        return pd.Series(styles_row)

    # _inst_type présent dans df_all mais absent de DISPLAY_COLS — non affiché
    cols_display = [c for c in DISPLAY_COLS if c in df_all.columns]
    styled = (
        df_all[cols_display + ["_inst_type"]]
        .style
        .apply(style_distance_row, axis=1)
        .map(
            lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH"
            else "color:#94a3b8" if x == "BOS" else "",
            subset=["Type"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
            else "color:#ff5252;font-weight:bold" if x == "Vente" else "",
            subset=["Ordre"]
        )
        .map(
            lambda x: "color:#00c853" if "Bull" in str(x)
            else "color:#ff5252" if "Bear" in str(x) else "",
            subset=["Signal"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fort"
            else "color:#ff5252" if x == "Faible"
            else "color:#ff9800" if x == "Moyen" else "",
            subset=["Force"]
        )
        .map(style_bb, subset=["BB_Width"])
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fresh"
            else "color:#ff9800;font-weight:bold" if x == "Aged"
            else "color:#ff5252;font-weight:bold" if x == "Stale" else "",
            subset=["Statut"]
        )
        .hide(axis="columns", subset=["_inst_type"])
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)
