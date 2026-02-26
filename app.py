# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

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

    /* Bouton SCAN rouge centré */
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

# --- BOUTON SCAN CENTRÉ EN HAUT EN ROUGE ---
col_l, col_c, col_r = st.columns([2, 1, 2])
with col_c:
    scan_button = st.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# --- Fonctions de l'API OANDA ---
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    environments = {"Practice (Démo)": "https://api-fxpractice.oanda.com", "Live (Réel)": "https://api-fxtrade.oanda.com"}
    for name, url in environments.items():
        try:
            response = requests.get(f"{url}/v3/accounts/{account_id}/summary", headers=headers, timeout=5)
            if response.status_code == 200: return url, name
        except requests.RequestException: continue
    return None, None

@st.cache_data(ttl=600)
def get_oanda_data(base_url, access_token, symbol, timeframe='daily', limit=500):
    url = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"count": limit, "granularity": {'h4': 'H4', 'daily': 'D', 'weekly': 'W'}[timeframe], "price": "M"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get('candles'): return None
        candles = [{'date': pd.to_datetime(c['time']), 'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']), 'close': float(c['mid']['c']), 'volume': int(c['volume'])}
                   for c in data.get('candles', []) if c.get('complete')]
        return pd.DataFrame(candles).set_index('date')
    except requests.RequestException: return None

@st.cache_data(ttl=15)
def get_oanda_current_price(base_url, access_token, account_id, symbol):
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"instruments": symbol}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'prices' in data and data.get('prices'):
            bid = float(data['prices'][0]['closeoutBid'])
            ask = float(data['prices'][0]['closeoutAsk'])
            return (bid + ask) / 2
        return None
    except requests.RequestException: return None

# --- MOTEUR D'ANALYSE AMÉLIORÉ ---

def get_adaptive_distance(timeframe):
    """Distance adaptative selon le timeframe"""
    distances = {'h4': 5, 'daily': 8, 'weekly': 10}
    return distances.get(timeframe, 5)

def find_strong_sr_zones(df, current_price, zone_percentage_width=0.5, min_touches=2, timeframe='daily'):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    
    # Fallback si current_price est None
    if current_price is None:
        current_price = df['close'].iloc[-1]
    
    # Distance adaptative
    distance = get_adaptive_distance(timeframe)
    
    r_indices, _ = find_peaks(df['high'], distance=distance)
    s_indices, _ = find_peaks(-df['low'], distance=distance)
    pivots_high = df.iloc[r_indices]['high']
    pivots_low = df.iloc[s_indices]['low']
    all_pivots = pd.concat([pivots_high, pivots_low]).sort_values()
    if all_pivots.empty: return pd.DataFrame(), pd.DataFrame()
    
    zones = []
    if not all_pivots.empty:
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
                'level': np.mean(zone), 
                'strength': len(zone),
                'last_touch_index': max([all_pivots[all_pivots == p].index[0] for p in zone])
            })
    
    if not strong_zones: return pd.DataFrame(), pd.DataFrame()
    zones_df = pd.DataFrame(strong_zones).sort_values(by='level').reset_index(drop=True)
    
    supports = zones_df[zones_df['level'] < current_price].copy()
    resistances = zones_df[zones_df['level'] >= current_price].copy()
    
    return supports, resistances


def compute_atr(df, period=14):
    """Calcule l'ATR sur les dernières bougies pour contextualiser la distance."""
    if df is None or len(df) < period + 1:
        return None
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def detect_confluences(symbol, zones_dict, current_price, confluence_threshold=1.0):
    """Détecte les confluences entre timeframes"""
    if not zones_dict or current_price is None:
        return []
    
    confluences = []
    
    # Récupérer toutes les zones de tous les timeframes
    all_zones = []
    for tf, (supports, resistances) in zones_dict.items():
        for _, zone in supports.iterrows():
            all_zones.append({'tf': tf, 'level': zone['level'], 'strength': zone['strength'], 'type': 'Support'})
        for _, zone in resistances.iterrows():
            all_zones.append({'tf': tf, 'level': zone['level'], 'strength': zone['strength'], 'type': 'Résistance'})
    
    if not all_zones:
        return []
    
    # Détecter les confluences (zones proches entre différents TF)
    zones_df = pd.DataFrame(all_zones)
    zones_df = zones_df.sort_values('level')
    
    used_indices = set()
    for i, zone in zones_df.iterrows():
        if i in used_indices:
            continue
        
        # Trouver les zones proches (dans le seuil de confluence)
        similar_zones = zones_df[
            (abs(zones_df['level'] - zone['level']) / zone['level'] * 100 <= confluence_threshold) &
            (zones_df.index != i)
        ]
        
        if len(similar_zones) >= 1:  # Au moins 2 timeframes (zone actuelle + 1 autre)
            confluence_group = pd.concat([zones_df.loc[[i]], similar_zones])
            timeframes = confluence_group['tf'].unique()
            
            if len(timeframes) >= 2:  # Confluence confirmée
                avg_level = confluence_group['level'].mean()
                total_strength = confluence_group['strength'].sum()
                dist_pct = abs(current_price - avg_level) / current_price * 100
                
                zone_type = confluence_group.iloc[0]['type']
                signal = '🟢 BUY ZONE' if zone_type == 'Support' else '🔴 SELL ZONE'
                nb_tf = len(timeframes)
                tf_label = ' + '.join(sorted(timeframes))
                alerte = '🔥 ZONE CHAUDE' if dist_pct < 0.5 else ('⚠️ Proche' if dist_pct < 1.5 else '')
                confluences.append({
                    'Actif': symbol,
                    'Signal': signal,
                    'Niveau': f"{avg_level:.5f}",
                    'Type': zone_type,
                    'Timeframes': tf_label,
                    'Nb TF': nb_tf,
                    'Force Totale': int(total_strength),
                    'Distance %': f"{dist_pct:.2f}%",
                    'Alerte': alerte
                })
                
                # Marquer les indices utilisés
                used_indices.update(confluence_group.index)
    
    return confluences

# --- Fonctions de Création de Rapport ---

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Rapport de Scan Support/Résistance', border=0, align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 10, f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", border=0, align='C', new_x='LMARGIN', new_y='NEXT')
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
            self.multi_cell(0, 10, "Aucune donnee a afficher pour ce timeframe.")
            self.ln()
            return

        # A4 landscape: 297mm. Marges 5mm chaque cote = 287mm utile.
        if 'Timeframes' in df.columns:
            # Confluences 9 colonnes → total 285mm
            col_widths = {
                'Actif': 22, 'Signal': 28, 'Niveau': 24, 'Type': 24,
                'Timeframes': 68, 'Nb TF': 13, 'Force Totale': 22,
                'Distance %': 20, 'Alerte': 64,
            }
            font_size = 7
        else:
            # TF tables 10 colonnes → total 285mm
            col_widths = {
                'Actif': 26, 'Prix Actuel': 28, 'Support': 28,
                'Force (S)': 26, 'Dist. (S) %': 18, 'Dist. S ATR': 18,
                'Resistance': 28, 'Résistance': 28,
                'Force (R)': 26, 'Dist. (R) %': 18, 'Dist. R ATR': 18,
            }
            font_size = 7

        # Centrer le tableau horizontalement
        total_w = sum(col_widths.get(c, 18) for c in df.columns)
        usable_w = self.w - self.l_margin - self.r_margin
        x_start = self.l_margin + max(0, (usable_w - total_w) / 2)

        # En-têtes
        self.set_font('Helvetica', 'B', font_size)
        self.set_x(x_start)
        for col_name in df.columns:
            w = col_widths.get(col_name, 18)
            self.cell(w, 6, col_name, border=1, align='C', new_x='RIGHT', new_y='TOP')
        self.ln()

        # Données
        self.set_font('Helvetica', '', font_size)
        for _, row in df.iterrows():
            self.set_x(x_start)
            for col_name in df.columns:
                w = col_widths.get(col_name, 18)
                val = str(row[col_name])
                # Tronquer seulement si vraiment nécessaire
                max_chars = int(w / 1.25)
                if len(val) > max_chars:
                    val = val[:max_chars - 1] + '.'
                self.cell(w, 5, val, border=1, align='C', new_x='RIGHT', new_y='TOP')
            self.ln()

def strip_emojis_df(df):
    """Remplace tous les emojis et caractères non-latin1 d'un DataFrame pour FPDF."""
    emoji_map = {
        '🟢': '[BUY]', '🔴': '[SELL]', '🔥': '[CHAUD]', '⚠️': '[PROCHE]',
        '📈': '', '📉': '', '↔️': '', '✅': '[OK]', '❌': '[X]',
    }
    clean = df.copy()
    for col in clean.select_dtypes(include='object').columns:
        for emoji, replacement in emoji_map.items():
            clean[col] = clean[col].astype(str).str.replace(emoji, replacement, regex=False)
        # Fallback: encode to latin-1, drop unencodable chars
        clean[col] = clean[col].apply(
            lambda x: x.encode('latin-1', errors='ignore').decode('latin-1')
        )
    return clean

def create_pdf_report(results_dict, confluences_df=None):
    """Crée un rapport PDF à partir du dictionnaire de résultats."""
    pdf = PDF('L', 'mm', 'A4')
    pdf.set_margins(5, 10, 5)  # left, top, right → 287mm usable
    pdf.add_page()
    
    # Ajouter les confluences en premier si disponibles
    if confluences_df is not None and not confluences_df.empty:
        clean_df = strip_emojis_df(confluences_df)
        
        pdf.chapter_title('*** ZONES DE CONFLUENCE MULTI-TIMEFRAMES ***')
        pdf.chapter_body(clean_df)
        pdf.ln(10)
    
    title_map = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journaliere (Daily)', 'Weekly': 'Analyse Hebdomadaire (Weekly)'}

    for timeframe_key, df in results_dict.items():
        pdf.chapter_title(title_map[timeframe_key])
        pdf.chapter_body(strip_emojis_df(df))
        pdf.ln(10)

    return bytes(pdf.output())

def create_csv_report(results_dict, confluences_df=None):
    """Combine tous les résultats dans un seul DataFrame et le retourne en CSV."""
    all_dfs = []
    
    # Ajouter les confluences en premier
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

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    csv_buffer = BytesIO()
    full_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue()


def _display_results(sr):
    df_h4, df_daily, df_weekly = sr['df_h4'], sr['df_daily'], sr['df_weekly']
    conf_filtered, report_dict  = sr['conf_filtered'], sr['report_dict']

    tf_cfg = {
        "Actif": st.column_config.TextColumn("Actif", width="small"),
        "Prix Actuel": st.column_config.TextColumn("Prix Actuel", width="small"),
        "Support": st.column_config.TextColumn("Support", width="small"),
        "Force (S)": st.column_config.TextColumn("Force (S)", width="medium"),
        "Dist. (S) %": st.column_config.TextColumn("Dist. S %", width="small"),
        "Dist. S ATR": st.column_config.TextColumn("Dist. S ATR", width="small"),
        "Résistance": st.column_config.TextColumn("Résistance", width="small"),
        "Force (R)": st.column_config.TextColumn("Force (R)", width="medium"),
        "Dist. (R) %": st.column_config.TextColumn("Dist. R %", width="small"),
        "Dist. R ATR": st.column_config.TextColumn("Dist. R ATR", width="small"),
    }

    # Confluences
    if not conf_filtered.empty:
        st.divider()
        st.subheader("🔥 ZONES DE CONFLUENCE MULTI-TIMEFRAMES")
        st.markdown("**Ces zones sont validées par plusieurs timeframes - HAUTE PROBABILITÉ**")
        conf_display = conf_filtered.sort_values(by=['Alerte', 'Force Totale'], ascending=[False, False]).reset_index(drop=True)
        hot    = len(conf_display[conf_display['Alerte'] == '🔥 ZONE CHAUDE'])
        proche = len(conf_display[conf_display['Alerte'] == '⚠️ Proche'])
        buy_z  = len(conf_display[conf_display['Signal'] == '🟢 BUY ZONE'])
        sell_z = len(conf_display[conf_display['Signal'] == '🔴 SELL ZONE'])
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("🔥 Zones Chaudes", hot)
        mc2.metric("⚠️ Zones Proches", proche)
        mc3.metric("🟢 BUY Zones", buy_z)
        mc4.metric("🔴 SELL Zones", sell_z)
        st.dataframe(conf_display, column_config={
            "Actif": st.column_config.TextColumn("Actif", width="small"),
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "Niveau": st.column_config.TextColumn("Niveau", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Timeframes": st.column_config.TextColumn("Timeframes", width="medium"),
            "Nb TF": st.column_config.NumberColumn("Nb TF", width="small"),
            "Force Totale": st.column_config.NumberColumn("Force Totale", width="small"),
            "Distance %": st.column_config.TextColumn("Distance %", width="small"),
            "Alerte": st.column_config.TextColumn("Alerte", width="small"),
        }, hide_index=True, use_container_width=True,
           height=min(len(conf_display) * 35 + 38, 700))
    else:
        st.info("Aucune confluence détectée. Essayez d'augmenter le seuil de confluence.")

    # Export
    st.subheader("📋 Options d'Exportation du Rapport")
    with st.expander("Cliquez ici pour télécharger les résultats"):
        col1, col2 = st.columns(2)
        with col1:
            pdf_bytes = create_pdf_report(report_dict, conf_filtered)
            st.download_button("📄 Télécharger le Rapport (PDF)", data=pdf_bytes,
                file_name=f"rapport_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", use_container_width=True)
        with col2:
            csv_bytes = create_csv_report(report_dict, conf_filtered)
            st.download_button("📊 Télécharger les Données (CSV)", data=csv_bytes,
                file_name=f"donnees_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)

    # Tableaux
    st.divider()
    st.subheader("📅 Analyse 4 Heures (H4)")
    st.dataframe(df_h4.sort_values(by='Actif').reset_index(drop=True),
        column_config=tf_cfg, hide_index=True, use_container_width=True,
        height=min(len(df_h4) * 35 + 38, 600))
    st.subheader("📅 Analyse Journalière (Daily)")
    st.dataframe(df_daily.sort_values(by='Actif').reset_index(drop=True),
        column_config=tf_cfg, hide_index=True, use_container_width=True,
        height=min(len(df_daily) * 35 + 38, 600))
    st.subheader("📅 Analyse Hebdomadaire (Weekly)")
    st.dataframe(df_weekly.sort_values(by='Actif').reset_index(drop=True),
        column_config=tf_cfg, hide_index=True, use_container_width=True,
        height=min(len(df_weekly) * 35 + 38, 600))


# --- INTERFACE UTILISATEUR (SIDEBAR) ---
with st.sidebar:
    st.header("1. Connexion")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
    except:
        access_token, account_id = None, None
        st.error("Secrets OANDA non trouvés.")
    
    st.header("2. Sélection des Actifs")
    all_symbols = sorted([
        "XAU_USD", "XAG_USD", "XPT_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", 
        "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD", "EUR_CHF", 
        "GBP_JPY", "GBP_AUD", "GBP_NZD", "GBP_CAD", "GBP_CHF", 
        "AUD_NZD", "AUD_CAD", "AUD_CHF", "AUD_JPY", 
        "NZD_CAD", "NZD_CHF", "NZD_JPY", 
        "CAD_CHF", "CAD_JPY", "CHF_JPY"
    ])
    
    st.info("Cochez la case pour scanner tous les actifs.")
    select_all = st.checkbox("Scanner tous les actifs (35)", value=True)
    
    if select_all:
        symbols_to_scan = all_symbols
    else:
        default_selection = sorted(["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY"])
        symbols_to_scan = st.multiselect("Ou choisissez des actifs spécifiques :", options=all_symbols, default=default_selection)
    
    st.header("3. Paramètres de Détection")
    zone_width = st.slider("Largeur de zone (%)", 0.1, 2.0, 0.5, 0.1, help="Largeur de la zone pour regrouper les pivots.")
    min_touches = st.slider("Force minimale (touches)", 2, 10, 3, 1, help="Nombre de contacts minimum pour valider une zone.")
    confluence_threshold = st.slider("Seuil de confluence (%)", 0.3, 2.0, 1.0, 0.1, help="Distance max pour considérer une confluence entre TFs.")
    max_dist_filter = st.slider("Afficher zones < (%)", 1.0, 10.0, 3.0, 0.5, help="Masquer les confluences trop éloignées du prix.")
    
    # Bouton déplacé en haut de la page (voir ci-dessus)

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
            results = {'H4': [], 'Daily': [], 'Weekly': []}
            all_zones = {}  # Pour stocker les zones de chaque actif
            timeframes = ['h4', 'daily', 'weekly']
            progress_bar = st.progress(0, text="Initialisation...")
            total_steps = len(symbols_to_scan) * len(timeframes)
            
            for i, symbol in enumerate(symbols_to_scan):
                current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
                all_zones[symbol] = {}
                
                for j, timeframe in enumerate(timeframes):
                    progress_step = (i * len(timeframes) + j + 1)
                    progress_text = f"Scan... ({progress_step}/{total_steps}) {symbol.replace('_', '/')} - {timeframe.upper()}"
                    progress_bar.progress(progress_step / total_steps, text=progress_text)
                    
                    df = get_oanda_data(base_url, access_token, symbol, timeframe, limit=500)
                    if df is not None and not df.empty:
                        supports, resistances = find_strong_sr_zones(df, current_price, zone_percentage_width=zone_width, min_touches=min_touches, timeframe=timeframe)
                        
                        # Stocker pour analyse de confluence
                        all_zones[symbol][timeframe.capitalize()] = (supports, resistances)
                        
                        # Sécurité : vérifier si les DataFrames ne sont pas vides avant d'accéder
                        sup = supports.iloc[-1] if not supports.empty else None
                        res = resistances.iloc[0] if not resistances.empty else None
                        
                        if current_price is None:
                            current_price = df['close'].iloc[-1]
                        
                        dist_s = (abs(current_price - sup['level']) / current_price) * 100 if sup is not None else np.nan
                        dist_r = (abs(current_price - res['level']) / current_price) * 100 if res is not None else np.nan
                        
                        atr_val = compute_atr(df, period=14)
                        dist_s_atr = round(abs(current_price - sup['level']) / atr_val, 1) if (sup is not None and atr_val and atr_val > 0) else np.nan
                        dist_r_atr = round(abs(current_price - res['level']) / atr_val, 1) if (res is not None and atr_val and atr_val > 0) else np.nan
                        results[timeframe.capitalize()].append({
                            'Actif': symbol.replace('_', '/'),
                            'Prix Actuel': f"{current_price:.5f}" if current_price is not None else 'N/A',
                            'Support': f"{sup['level']:.5f}" if sup is not None else 'N/A',
                            'Force (S)': f"{int(sup['strength'])} touches" if sup is not None else 'N/A',
                            'Dist. (S) %': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                            'Dist. S ATR': f"{dist_s_atr:.1f}x" if not np.isnan(dist_s_atr) else 'N/A',
                            'Résistance': f"{res['level']:.5f}" if res is not None else 'N/A',
                            'Force (R)': f"{int(res['strength'])} touches" if res is not None else 'N/A',
                            'Dist. (R) %': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                            'Dist. R ATR': f"{dist_r_atr:.1f}x" if not np.isnan(dist_r_atr) else 'N/A',
                        })
            
            progress_bar.empty()
            st.success("✅ Scan terminé !")
            
            # Analyse des confluences
            st.info("🔍 Analyse des confluences multi-timeframes en cours...")
            all_confluences = []
            for symbol in symbols_to_scan:
                current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
                confluences = detect_confluences(symbol.replace('_', '/'), all_zones.get(symbol, {}), current_price, confluence_threshold)
                all_confluences.extend(confluences)
            
            confluences_df = pd.DataFrame(all_confluences)

            # Filtrer les zones trop lointaines — appliqué au PDF ET à l'affichage
            if not confluences_df.empty:
                confluences_df['_dist_num'] = confluences_df['Distance %'].str.replace('%', '').astype(float)
                conf_filtered = confluences_df[confluences_df['_dist_num'] <= max_dist_filter].drop(columns=['_dist_num'])
            else:
                conf_filtered = pd.DataFrame()
            
            # Préparer les DataFrames pour export
            def filter_tf_table(df, max_pct):
                if df.empty:
                    return df
                def dist_val(s):
                    try: return float(str(s).replace('%',''))
                    except: return 999.0
                dist_s = df['Dist. (S) %'].apply(dist_val)
                dist_r = df['Dist. (R) %'].apply(dist_val)
                mask = (dist_s <= max_pct) | (dist_r <= max_pct)
                return df[mask].reset_index(drop=True)

            df_h4 = filter_tf_table(pd.DataFrame(results['H4']), max_dist_filter)
            df_daily = filter_tf_table(pd.DataFrame(results['Daily']), max_dist_filter)
            df_weekly = filter_tf_table(pd.DataFrame(results['Weekly']), max_dist_filter)
            report_dict = {'H4': df_h4, 'Daily': df_daily, 'Weekly': df_weekly}
            # ── Sauvegarde pour survivre aux reruns (clic PDF/CSV) ──
            st.session_state['scan_results'] = {
                'df_h4': df_h4, 'df_daily': df_daily, 'df_weekly': df_weekly,
                'conf_filtered': conf_filtered, 'report_dict': report_dict,
            }

            _display_results(st.session_state['scan_results'])

elif not symbols_to_scan:
    st.info("Veuillez sélectionner des actifs à scanner ou cocher la case 'Scanner tous les actifs'.")
else:
    st.info("Cliquez sur 'Lancer le Scan Complet' pour commencer.")

# ── Ré-affichage persistant après rerun (ex: clic téléchargement) ──
if 'scan_results' in st.session_state and not scan_button:
    _display_results(st.session_state['scan_results'])
