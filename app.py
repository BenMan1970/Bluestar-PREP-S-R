# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import concurrent.futures

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Scanner S/R Exhaustif",
    page_icon="📡",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    [data-testid="stDataFrame"] > div { overflow-x: visible !important; overflow-y: visible !important; }
    [data-testid="stDataFrame"] iframe { width: 100% !important; height: auto !important; }
    ::-webkit-scrollbar { width: 0px !important; height: 0px !important; }

    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #D32F2F !important;
        color: white !important;
        border: 1px solid #B71C1C !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 14px 40px !important;
        border-radius: 8px !important;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #B71C1C !important;
        box-shadow: 0 4px 16px rgba(211,47,47,0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Scanner S/R Exhaustif (H4, D1, W)")
st.markdown("Génère les zones de Support/Résistance clés avec **distance en ATR**, force et confluences multi-timeframes.")

col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    scan_button = st.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# ===================== ASSETS — ALIGNÉS AVEC CHOCH ET RSI SCANNER =====================
ALL_SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "XAG_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
]

# Zone width adaptative : indices ont des prix ~10000-50000x plus grands
# → une zone fixe de 0.5% serait trop large sur indices
ZONE_WIDTH_MAP = {
    "US30_USD": 0.15, "NAS100_USD": 0.15, "SPX500_USD": 0.15, "DE30_EUR": 0.15,
    "XAU_USD":  0.20, "XAG_USD":   0.25, "XPT_USD":   0.25,
}
DEFAULT_ZONE_WIDTH = 0.5  # Forex classique

# --- Fonctions API OANDA ---
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    environments = {
        "Practice (Démo)": "https://api-fxpractice.oanda.com",
        "Live (Réel)":     "https://api-fxtrade.oanda.com"
    }
    for name, url in environments.items():
        try:
            response = requests.get(f"{url}/v3/accounts/{account_id}/summary",
                                    headers=headers, timeout=5)
            if response.status_code == 200:
                return url, name
        except requests.RequestException:
            continue
    return None, None


@st.cache_data(ttl=600)
def get_oanda_data(base_url, access_token, symbol, timeframe='daily', limit=500):
    url     = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {
        "count":       limit,
        "granularity": {'h4': 'H4', 'daily': 'D', 'weekly': 'W'}[timeframe],
        "price":       "M"
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get('candles'):
            return None
        candles = [
            {
                'date':   pd.to_datetime(c['time']),
                'open':   float(c['mid']['o']),
                'high':   float(c['mid']['h']),
                'low':    float(c['mid']['l']),
                'close':  float(c['mid']['c']),
                'volume': int(c['volume'])
            }
            for c in data.get('candles', []) if c.get('complete')
        ]
        return pd.DataFrame(candles).set_index('date')
    except requests.RequestException:
        return None


# FIX #3 : cache 60s au lieu de 15s — évite les rafales API pendant le scan parallèle
@st.cache_data(ttl=60)
def get_oanda_current_price(base_url, access_token, account_id, symbol):
    url     = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    params  = {"instruments": symbol}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'prices' in data and data.get('prices'):
            bid = float(data['prices'][0]['closeoutBid'])
            ask = float(data['prices'][0]['closeoutAsk'])
            return (bid + ask) / 2
        return None
    except requests.RequestException:
        return None


# --- MOTEUR D'ANALYSE ---

def get_adaptive_distance(timeframe):
    return {'h4': 5, 'daily': 8, 'weekly': 10}.get(timeframe, 5)


def compute_atr(df, period=14):
    if df is None or len(df) < period + 1:
        return None
    high  = df['high']
    low   = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def find_strong_sr_zones(df, current_price, zone_percentage_width=0.5,
                          min_touches=2, timeframe='daily'):
    """
    FIX #2 : retourne TOUTES les zones valides (supports et résistances),
    pas seulement la plus proche.
    """
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    if current_price is None:
        current_price = df['close'].iloc[-1]

    distance  = get_adaptive_distance(timeframe)
    r_indices, _ = find_peaks(df['high'],  distance=distance)
    s_indices, _ = find_peaks(-df['low'],  distance=distance)
    pivots_high  = df.iloc[r_indices]['high']
    pivots_low   = df.iloc[s_indices]['low']
    all_pivots   = pd.concat([pivots_high, pivots_low]).sort_values()
    if all_pivots.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Regroupement en zones
    zones = []
    current_zone = [all_pivots.iloc[0]]
    for price in all_pivots.iloc[1:]:
        zone_avg = np.mean(current_zone)
        if abs(price - zone_avg) < (zone_avg * zone_percentage_width / 100):
            current_zone.append(price)
        else:
            zones.append(list(current_zone))
            current_zone = [price]
    zones.append(list(current_zone))

    strong_zones = []
    for zone in zones:
        if len(zone) >= min_touches:
            strong_zones.append({
                'level':    np.mean(zone),
                'strength': len(zone),
            })

    if not strong_zones:
        return pd.DataFrame(), pd.DataFrame()

    zones_df   = pd.DataFrame(strong_zones).sort_values(by='level').reset_index(drop=True)
    supports   = zones_df[zones_df['level'] <  current_price].copy()
    resistances = zones_df[zones_df['level'] >= current_price].copy()
    return supports, resistances


def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0):
    if not zones_dict or current_price is None:
        return []

    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, zone in supports.iterrows():
            all_zones.append({'tf': tf, 'level': zone['level'],
                               'strength': zone['strength'], 'type': 'Support'})
        for _, zone in resistances.iterrows():
            all_zones.append({'tf': tf, 'level': zone['level'],
                               'strength': zone['strength'], 'type': 'Résistance'})

    if not all_zones:
        return []

    zones_df   = pd.DataFrame(all_zones).sort_values('level')
    used_indices = set()
    confluences  = []

    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue
        similar_zones = zones_df[
            (abs(zones_df['level'] - zone['level']) / zone['level'] * 100 <= confluence_threshold) &
            (zones_df.index != i)
        ]
        if len(similar_zones) >= 1:
            confluence_group = pd.concat([zones_df.loc[[i]], similar_zones])
            timeframes = confluence_group['tf'].unique()
            if len(timeframes) >= 2:
                avg_level     = confluence_group['level'].mean()
                total_strength = confluence_group['strength'].sum()
                dist_pct      = abs(current_price - avg_level) / current_price * 100
                zone_type     = confluence_group.iloc[0]['type']
                signal        = '🟢 BUY ZONE' if zone_type == 'Support' else '🔴 SELL ZONE'
                tf_label      = ' + '.join(sorted(timeframes))
                alerte        = '🔥 ZONE CHAUDE' if dist_pct < 0.5 else ('⚠️ Proche' if dist_pct < 1.5 else '')
                confluences.append({
                    'Actif':        symbol,
                    'Signal':       signal,
                    'Niveau':       f"{avg_level:.5f}",
                    'Type':         zone_type,
                    'Timeframes':   tf_label,
                    'Nb TF':        len(timeframes),
                    'Force Totale': int(total_strength),
                    'Distance %':   f"{dist_pct:.2f}%",
                    'Alerte':       alerte,
                })
                used_indices.update(confluence_group.index)

    return confluences


# FIX #1 : Scan parallèle par actif
def scan_single_symbol(args):
    """Scan complet d'un symbole sur H4 + Daily + Weekly en parallèle."""
    symbol, base_url, access_token, account_id, zone_width, min_touches = args
    current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)

    # Zone width adaptative par instrument
    adaptive_width = ZONE_WIDTH_MAP.get(symbol, zone_width)

    rows    = {'H4': None, 'Daily': None, 'Weekly': None}
    zones_d = {}
    atr_cache = {}

    for tf_key, tf_cap in [('h4','H4'), ('daily','Daily'), ('weekly','Weekly')]:
        df = get_oanda_data(base_url, access_token, symbol, tf_key, limit=500)
        if df is None or df.empty:
            continue

        cp = current_price if current_price is not None else df['close'].iloc[-1]
        supports, resistances = find_strong_sr_zones(
            df, cp,
            zone_percentage_width=adaptive_width,
            min_touches=min_touches,
            timeframe=tf_key
        )
        zones_d[tf_cap] = (supports, resistances)

        atr_val = compute_atr(df, period=14)
        atr_cache[tf_cap] = atr_val

        # FIX #2 : toutes les zones S et R, une ligne par zone
        sym_display = symbol.replace('_', '/')

        def make_row(zone, ztype):
            lvl      = zone['level']
            strength = int(zone['strength'])
            dist_pct = abs(cp - lvl) / cp * 100
            dist_atr = round(abs(cp - lvl) / atr_val, 1) if (atr_val and atr_val > 0) else np.nan
            return {
                'Actif':        sym_display,
                'Prix Actuel':  f"{cp:.5f}",
                'Type':         ztype,
                'Niveau':       f"{lvl:.5f}",
                'Force':        f"{strength} touches",
                'Dist. %':      f"{dist_pct:.2f}%",
                'Dist. ATR':    f"{dist_atr:.1f}x" if not np.isnan(dist_atr) else 'N/A',
            }

        tf_rows = []
        for _, s in supports.iterrows():
            tf_rows.append(make_row(s, 'Support'))
        for _, r in resistances.iterrows():
            tf_rows.append(make_row(r, 'Résistance'))

        if tf_rows:
            rows[tf_cap] = tf_rows

    return symbol, rows, zones_d, current_price


# --- PDF ---

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Rapport de Scan Support/Résistance', border=0, align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 10, f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                  border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, border=0, align='L', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 10, "Aucune donnee a afficher.")
            self.ln()
            return

        if 'Timeframes' in df.columns:
            # Confluences
            col_widths = {
                'Actif': 22, 'Signal': 28, 'Niveau': 24, 'Type': 24,
                'Timeframes': 68, 'Nb TF': 13, 'Force Totale': 22,
                'Distance %': 20, 'Alerte': 64,
            }
            font_size = 7
        else:
            # Tableau S/R par TF — FIX #2 nouvelles colonnes
            col_widths = {
                'Actif': 28, 'Prix Actuel': 28, 'Type': 22,
                'Niveau': 28, 'Force': 24,
                'Dist. %': 18, 'Dist. ATR': 18,
            }
            font_size = 7

        total_w = sum(col_widths.get(c, 18) for c in df.columns)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start  = self.l_margin + max(0, (usable_w - total_w) / 2)

        self.set_font('Helvetica', 'B', font_size)
        self.set_x(x_start)
        for col_name in df.columns:
            w = col_widths.get(col_name, 18)
            self.cell(w, 6, col_name, border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()

        self.set_font('Helvetica', '', font_size)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col_name in df.columns:
                w   = col_widths.get(col_name, 18)
                val = str(row[col_name])
                max_chars = int(w / 1.25)
                if len(val) > max_chars:
                    val = val[:max_chars - 1] + '.'
                self.cell(w, 5, val, border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()


def strip_emojis_df(df):
    emoji_map = {
        '🟢': '[BUY]', '🔴': '[SELL]', '🔥': '[CHAUD]', '⚠️': '[PROCHE]',
        '📈': '', '📉': '', '↔️': '', '✅': '[OK]', '❌': '[X]',
    }
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        for emoji, replacement in emoji_map.items():
            clean[col] = clean[col].astype(str).str.replace(emoji, replacement, regex=False)
        clean[col] = clean[col].apply(
            lambda x: x.encode('latin-1', errors='ignore').decode('latin-1')
        )
    return clean


def create_pdf_report(results_dict, confluences_df=None):
    """
    FIX #5 : le PDF contient TOUTES les données (sans filtre distance).
    Le filtre est uniquement appliqué à l'affichage Streamlit.
    """
    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)
    pdf.add_page()

    if confluences_df is not None and not confluences_df.empty:
        pdf.chapter_title('*** ZONES DE CONFLUENCE MULTI-TIMEFRAMES ***')
        pdf.chapter_body(strip_emojis_df(confluences_df))
        pdf.ln(10)

    title_map = {
        'H4':     'Analyse 4 Heures (H4)',
        'Daily':  'Analyse Journaliere (Daily)',
        'Weekly': 'Analyse Hebdomadaire (Weekly)'
    }
    for tf_key, df in results_dict.items():
        pdf.chapter_title(title_map[tf_key])
        pdf.chapter_body(strip_emojis_df(df))
        pdf.ln(10)

    return bytes(pdf.output())


def create_csv_report(results_dict, confluences_df=None):
    all_dfs = []
    if confluences_df is not None and not confluences_df.empty:
        conf_copy = confluences_df.copy()
        conf_copy['Section'] = 'CONFLUENCES'
        all_dfs.append(conf_copy)
    for timeframe, df in results_dict.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['Timeframe'] = timeframe
            all_dfs.append(df_copy)
    if not all_dfs:
        return b""
    csv_buffer = BytesIO()
    pd.concat(all_dfs, ignore_index=True).to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue()


def _display_results(sr, max_dist_filter):
    df_h4      = sr['df_h4']
    df_daily   = sr['df_daily']
    df_weekly  = sr['df_weekly']
    # FIX #5 : confluences COMPLÈTES pour le PDF, filtrées pour l'affichage
    conf_full     = sr['conf_full']
    conf_filtered = sr['conf_filtered']
    report_dict   = sr['report_dict']

    tf_cfg = {
        "Actif":       st.column_config.TextColumn("Actif",       width="small"),
        "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"),
        "Type":        st.column_config.TextColumn("Type",        width="small"),
        "Niveau":      st.column_config.TextColumn("Niveau",      width="small"),
        "Force":       st.column_config.TextColumn("Force",       width="medium"),
        "Dist. %":     st.column_config.TextColumn("Dist. %",     width="small"),
        "Dist. ATR":   st.column_config.TextColumn("Dist. ATR",   width="small"),
    }

    # Confluences
    if not conf_filtered.empty:
        st.divider()
        st.subheader("🔥 ZONES DE CONFLUENCE MULTI-TIMEFRAMES")
        st.markdown("**Ces zones sont validées par plusieurs timeframes - HAUTE PROBABILITÉ**")
        conf_display = conf_filtered.sort_values(
            by=['Alerte', 'Force Totale'], ascending=[False, False]
        ).reset_index(drop=True)
        hot    = len(conf_display[conf_display['Alerte'] == '🔥 ZONE CHAUDE'])
        proche = len(conf_display[conf_display['Alerte'] == '⚠️ Proche'])
        buy_z  = len(conf_display[conf_display['Signal'] == '🟢 BUY ZONE'])
        sell_z = len(conf_display[conf_display['Signal'] == '🔴 SELL ZONE'])
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("🔥 Zones Chaudes",  hot)
        mc2.metric("⚠️ Zones Proches",  proche)
        mc3.metric("🟢 BUY Zones",      buy_z)
        mc4.metric("🔴 SELL Zones",     sell_z)
        st.dataframe(conf_display, column_config={
            "Actif":        st.column_config.TextColumn("Actif",        width="small"),
            "Signal":       st.column_config.TextColumn("Signal",       width="small"),
            "Niveau":       st.column_config.TextColumn("Niveau",       width="small"),
            "Type":         st.column_config.TextColumn("Type",         width="small"),
            "Timeframes":   st.column_config.TextColumn("Timeframes",   width="medium"),
            "Nb TF":        st.column_config.NumberColumn("Nb TF",      width="small"),
            "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
            "Distance %":   st.column_config.TextColumn("Distance %",   width="small"),
            "Alerte":       st.column_config.TextColumn("Alerte",       width="small"),
        }, hide_index=True, use_container_width=True,
           height=min(len(conf_display) * 35 + 38, 700))
    else:
        st.info("Aucune confluence détectée. Essayez d'augmenter le seuil de confluence.")

    # Export — FIX #5 : PDF avec conf_full (non filtré)
    st.subheader("📋 Options d'Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_pdf_report(report_dict, conf_full)
            st.download_button(
                "📄 Télécharger le Rapport (PDF)", data=pdf_bytes,
                file_name=f"rapport_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True
            )
        with col2:
            csv_bytes = create_csv_report(report_dict, conf_full)
            st.download_button(
                "📊 Télécharger les Données (CSV)", data=csv_bytes,
                file_name=f"donnees_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True
            )

    # Tableaux TF — affichage filtré
    def filter_tf(df, max_pct):
        if df.empty:
            return df
        def dist_val(s):
            try: return float(str(s).replace('%', ''))
            except: return 999.0
        return df[df['Dist. %'].apply(dist_val) <= max_pct].reset_index(drop=True)

    st.divider()
    st.subheader("📅 Analyse 4 Heures (H4)")
    filtered = filter_tf(df_h4, max_dist_filter)
    st.dataframe(filtered, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(filtered) * 35 + 38, 600))

    st.subheader("📅 Analyse Journalière (Daily)")
    filtered = filter_tf(df_daily, max_dist_filter)
    st.dataframe(filtered, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(filtered) * 35 + 38, 600))

    st.subheader("📅 Analyse Hebdomadaire (Weekly)")
    filtered = filter_tf(df_weekly, max_dist_filter)
    st.dataframe(filtered, column_config=tf_cfg, hide_index=True,
                 use_container_width=True, height=min(len(filtered) * 35 + 38, 600))


# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Connexion")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id   = st.secrets["OANDA_ACCOUNT_ID"]
    except Exception:
        access_token, account_id = None, None
        st.error("Secrets OANDA non trouvés.")

    st.header("2. Sélection des Actifs")
    st.info("Cochez la case pour scanner tous les actifs.")
    select_all = st.checkbox(f"Scanner tous les actifs ({len(ALL_SYMBOLS)})", value=True)

    if select_all:
        symbols_to_scan = ALL_SYMBOLS
    else:
        default_sel = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY"]
        symbols_to_scan = st.multiselect(
            "Ou choisissez des actifs spécifiques :",
            options=ALL_SYMBOLS,
            default=default_sel
        )

    st.header("3. Paramètres de Détection")
    zone_width = st.slider(
        "Largeur de zone Forex (%)", 0.1, 2.0, 0.5, 0.1,
        help="Les indices utilisent automatiquement 0.15% (adaptatif)."
    )
    min_touches = st.slider(
        "Force minimale (touches)", 2, 10, 3, 1,
        help="Nombre de contacts minimum pour valider une zone."
    )
    confluence_threshold = st.slider(
        "Seuil de confluence (%)", 0.3, 2.0, 1.0, 0.1,
        help="Distance max pour considérer une confluence entre TFs."
    )
    max_dist_filter = st.slider(
        "Afficher zones < (%) — affichage uniquement", 1.0, 10.0, 3.0, 0.5,
        help="Filtre visuel uniquement. Le PDF contient toutes les zones."
    )


# --- LOGIQUE PRINCIPALE ---
if scan_button and symbols_to_scan:
    st.session_state.pop('scan_results', None)
    if not access_token or not account_id:
        st.warning("Veuillez configurer vos secrets OANDA pour lancer l'analyse.")
    else:
        base_url, _ = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.error("Impossible de valider vos identifiants OANDA. Vérifiez vos secrets.")
        else:
            progress_bar  = st.progress(0, text="Initialisation du scan parallèle...")
            results_h4    = []
            results_daily = []
            results_weekly = []
            all_zones_map  = {}
            prices_map     = {}

            # FIX #1 : ThreadPoolExecutor — scan parallèle par actif
            args_list = [
                (sym, base_url, access_token, account_id, zone_width, min_touches)
                for sym in symbols_to_scan
            ]
            total     = len(args_list)
            completed = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_map = {executor.submit(scan_single_symbol, args): args[0]
                              for args in args_list}
                for future in concurrent.futures.as_completed(future_map):
                    symbol = future_map[future]
                    completed += 1
                    progress_bar.progress(
                        completed / total,
                        text=f"Scan... ({completed}/{total}) {symbol.replace('_', '/')}"
                    )
                    try:
                        sym, rows, zones_d, cp = future.result()
                        all_zones_map[sym] = zones_d
                        prices_map[sym]    = cp
                        for tf_cap, tf_rows in rows.items():
                            if tf_rows:
                                if tf_cap == 'H4':     results_h4.extend(tf_rows)
                                elif tf_cap == 'Daily': results_daily.extend(tf_rows)
                                elif tf_cap == 'Weekly': results_weekly.extend(tf_rows)
                    except Exception:
                        pass

            progress_bar.empty()
            st.success("✅ Scan terminé !")

            # Analyse des confluences
            st.info("🔍 Analyse des confluences multi-timeframes en cours...")
            all_confluences = []
            for sym in symbols_to_scan:
                cp = prices_map.get(sym)
                confs = detect_confluences(
                    sym.replace('_', '/'),
                    all_zones_map.get(sym, {}),
                    cp,
                    confluence_threshold
                )
                all_confluences.extend(confs)

            conf_full = pd.DataFrame(all_confluences)

            # FIX #5 : filtre distance → uniquement pour l'affichage
            if not conf_full.empty:
                conf_full['_dist_num'] = conf_full['Distance %'].str.replace('%', '').astype(float)
                conf_filtered = conf_full[conf_full['_dist_num'] <= max_dist_filter].drop(columns=['_dist_num'])
                conf_full     = conf_full.drop(columns=['_dist_num'])
            else:
                conf_filtered = pd.DataFrame()

            df_h4     = pd.DataFrame(results_h4)
            df_daily  = pd.DataFrame(results_daily)
            df_weekly = pd.DataFrame(results_weekly)

            report_dict = {'H4': df_h4, 'Daily': df_daily, 'Weekly': df_weekly}

            st.session_state['scan_results'] = {
                'df_h4':          df_h4,
                'df_daily':       df_daily,
                'df_weekly':      df_weekly,
                'conf_full':      conf_full,      # FIX #5 : non filtré pour PDF
                'conf_filtered':  conf_filtered,  # filtré pour affichage
                'report_dict':    report_dict,
                'max_dist':       max_dist_filter,
            }

            _display_results(st.session_state['scan_results'], max_dist_filter)

elif not symbols_to_scan:
    st.info("Veuillez sélectionner des actifs à scanner ou cocher la case 'Scanner tous les actifs'.")
else:
    st.info("Cliquez sur 'Lancer le Scan Complet' pour commencer.")

# Ré-affichage persistant après rerun
if 'scan_results' in st.session_state and not scan_button:
    _display_results(
        st.session_state['scan_results'],
        st.session_state['scan_results'].get('max_dist', 3.0)
    )
      
