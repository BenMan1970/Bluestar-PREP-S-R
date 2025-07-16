# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from fpdf import FPDF # Import pour la génération de PDF

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Scanner S/R Exhaustif",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Scanner S/R Exhaustif (H4, D1, W)")
st.markdown("Génère une liste complète des zones de Support/Résistance pour une analyse de confluences approfondie.")

# --- Fonctions de l'API OANDA (inchangées) ---
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

# --- MOTEUR D'ANALYSE PROFESSIONNEL (corrigé) ---

# MODIFICATION 1: La signature de la fonction accepte 'current_price' et la logique de séparation a été corrigée.
def find_strong_sr_zones(df, current_price, zone_percentage_width=0.5, min_touches=2):
    # Ajout d'une vérification pour s'assurer que current_price n'est pas None
    if df is None or df.empty or len(df) < 20 or current_price is None:
        return pd.DataFrame(), pd.DataFrame()
        
    r_indices, _ = find_peaks(df['high'], distance=5)
    s_indices, _ = find_peaks(-df['low'], distance=5)
    pivots_high = df.iloc[r_indices]['high']
    pivots_low = df.iloc[s_indices]['low']
    all_pivots = pd.concat([pivots_high, pivots_low]).sort_values()
    if all_pivots.empty: return pd.DataFrame(), pd.DataFrame()
    zones = []
    if not all_pivots.empty:
        current_zone = [all_pivots.iloc[0]]
        for price in all_pivots.iloc[1:]:
            zone_avg = np.mean(current_zone)
            if abs(price - zone_avg) < (zone_avg * zone_percentage_width / 100): current_zone.append(price)
            else: zones.append(list(current_zone)); current_zone = [price]
        zones.append(list(current_zone))
    strong_zones = []
    for zone in zones:
        if len(zone) >= min_touches:
            strong_zones.append({'level': np.mean(zone), 'strength': len(zone)})
    if not strong_zones: return pd.DataFrame(), pd.DataFrame()
    zones_df = pd.DataFrame(strong_zones).sort_values(by='level').reset_index(drop=True)
    
    # Correction logique : On utilise le 'current_price' pour séparer les supports des résistances.
    supports = zones_df[zones_df['level'] < current_price].copy()
    resistances = zones_df[zones_df['level'] >= current_price].copy()
    
    return supports, resistances

# --- Fonctions de Création de Rapport ---

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'Rapport de Scan Support/Résistance', 0, 1, 'C')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 10, f"Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, df):
        if df.empty:
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 10, "Aucune donnée à afficher pour ce timeframe.")
            self.ln()
            return

        self.set_font('Helvetica', 'B', 8)
        col_widths = {'Actif': 25, 'Prix Actuel': 25, 'Support': 25, 'Force (S)': 25,
                      'Dist. (S) %': 20, 'Résistance': 25, 'Force (R)': 25, 'Dist. (R) %': 20}
        
        for col_name in df.columns:
            self.cell(col_widths.get(col_name, 20), 7, col_name, 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 7)
        for index, row in df.iterrows():
            for col_name in df.columns:
                self.cell(col_widths.get(col_name, 20), 6, str(row[col_name]), 1, 0, 'C')
            self.ln()

def create_pdf_report(results_dict):
    """Crée un rapport PDF à partir du dictionnaire de résultats."""
    pdf = PDF('L', 'mm', 'A4')
    pdf.add_page()
    
    title_map = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journalière (Daily)', 'Weekly': 'Analyse Hebdomadaire (Weekly)'}

    for timeframe_key, df in results_dict.items():
        pdf.chapter_title(title_map[timeframe_key])
        pdf.chapter_body(df)
        pdf.ln(10)

    # On convertit le 'bytearray' retourné par fpdf2 en 'bytes' pour Streamlit.
    return bytes(pdf.output())

def create_csv_report(results_dict):
    """Combine tous les résultats dans un seul DataFrame et le retourne en CSV."""
    all_dfs = []
    for timeframe, df in results_dict.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['Timeframe'] = timeframe
            all_dfs.append(df_copy)
    
    if not all_dfs:
        return b"" # Retourne des bytes vides

    full_df = pd.concat(all_dfs, ignore_index=True)
    cols = ['Timeframe'] + [col for col in full_df.columns if col != 'Timeframe']
    full_df = full_df[cols]
    
    csv_buffer = BytesIO()
    full_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue()

# --- INTERFACE UTILISATEUR (SIDEBAR) (inchangée) ---
with st.sidebar:
    st.header("1. Connexion")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets OANDA chargés.")
    except:
        access_token, account_id = None, None
        st.error("Secrets OANDA non trouvés.")
    st.header("2. Sélection des Actifs")
    all_symbols = sorted(["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD", "EUR_CHF", "GBP_JPY", "GBP_AUD", "GBP_NZD", "GBP_CAD", "GBP_CHF", "AUD_NZD", "AUD_CAD", "AUD_CHF", "AUD_JPY", "NZD_CAD", "NZD_CHF", "NZD_JPY", "CAD_CHF", "CAD_JPY", "CHF_JPY"])
    st.info("Cochez la case pour scanner tous les actifs.")
    select_all = st.checkbox("Scanner les 29 actifs")
    if select_all:
        symbols_to_scan = all_symbols
    else:
        default_selection = sorted(["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY"])
        symbols_to_scan = st.multiselect("Ou choisissez des actifs spécifiques :", options=all_symbols, default=default_selection)
    st.header("3. Paramètres de Détection")
    zone_width = st.slider("Largeur de zone (%)", 0.1, 2.0, 0.4, 0.1, help="Largeur de la zone pour regrouper les pivots.")
    min_touches = st.slider("Force minimale (touches)", 2, 10, 3, 1, help="Nombre de contacts minimum pour valider une zone.")
    scan_button = st.button("🚀 Lancer le Scan Complet", type="primary", use_container_width=True)

# --- LOGIQUE PRINCIPALE ---
if scan_button and symbols_to_scan:
    if not access_token or not account_id:
        st.warning("Veuillez configurer vos secrets OANDA pour lancer l'analyse.")
    else:
        base_url, _ = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.error("Impossible de valider vos identifiants OANDA. Vérifiez vos secrets.")
        else:
            results = {'H4': [], 'Daily': [], 'Weekly': []}
            timeframes = ['h4', 'daily', 'weekly']
            progress_bar = st.progress(0, text="Initialisation...")
            total_steps = len(symbols_to_scan) * len(timeframes)
            for i, symbol in enumerate(symbols_to_scan):
                current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
                for j, timeframe in enumerate(timeframes):
                    progress_step = (i * len(timeframes) + j + 1)
                    progress_text = f"Scan... ({progress_step}/{total_steps}) {symbol.replace('_', '/')} - {timeframe.upper()}"
                    progress_bar.progress(progress_step / total_steps, text=progress_text)
                    df = get_oanda_data(base_url, access_token, symbol, timeframe, limit=500)
                    if df is not None and not df.empty:
                        # MODIFICATION 2: On passe le 'current_price' pour une classification cohérente.
                        supports, resistances = find_strong_sr_zones(df, current_price, zone_percentage_width=zone_width, min_touches=min_touches)
                        
                        sup = supports.iloc[-1] if not supports.empty else None
                        res = resistances.iloc[0] if not resistances.empty else None
                        dist_s = (abs(current_price - sup['level']) / current_price) * 100 if sup is not None and current_price is not None else np.nan
                        dist_r = (abs(current_price - res['level']) / current_price) * 100 if res is not None and current_price is not None else np.nan
                        results[timeframe.capitalize()].append({'Actif': symbol.replace('_', '/'), 'Prix Actuel': f"{current_price:.5f}" if current_price is not None else 'N/A', 'Support': f"{sup['level']:.5f}" if sup is not None else 'N/A', 'Force (S)': f"{int(sup['strength'])} touches" if sup is not None else 'N/A', 'Dist. (S) %': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A', 'Résistance': f"{res['level']:.5f}" if res is not None else 'N/A', 'Force (R)': f"{int(res['strength'])} touches" if res is not None else 'N/A', 'Dist. (R) %': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A'})
            progress_bar.empty()
            st.success("Scan terminé !")
            
            df_h4 = pd.DataFrame(results['H4'])
            df_daily = pd.DataFrame(results['Daily'])
            df_weekly = pd.DataFrame(results['Weekly'])
            report_dict = {'H4': df_h4, 'Daily': df_daily, 'Weekly': df_weekly}

            st.subheader("📋 Options d'Exportation du Rapport")
            with st.expander("Cliquez ici pour télécharger les résultats"):
                
                col1, col2 = st.columns(2)

                with col1:
                    pdf_bytes = create_pdf_report(report_dict)
                    st.download_button(
                        label="📄 Télécharger le Rapport (PDF)",
                        data=pdf_bytes,
                        file_name=f"rapport_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                with col2:
                    csv_bytes = create_csv_report(report_dict)
                    st.download_button(
                        label="📊 Télécharger les Données (CSV)",
                        data=csv_bytes,
                        file_name=f"donnees_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


            st.divider()
            st.subheader("--- Analyse 4 Heures (H4) ---")
            st.dataframe(df_h4.sort_values(by='Actif').reset_index(drop=True), use_container_width=True, hide_index=True)
            st.subheader("--- Analyse Journalière (Daily) ---")
            st.dataframe(df_daily.sort_values(by='Actif').reset_index(drop=True), use_container_width=True, hide_index=True)
            st.subheader("--- Analyse Hebdomadaire (Weekly) ---")
            st.dataframe(df_weekly.sort_values(by='Actif').reset_index(drop=True), use_container_width=True, hide_index=True)

elif not symbols_to_scan:
    st.info("Veuillez sélectionner des actifs à scanner ou cocher la case 'Scanner les 29 actifs'.")
else:
    st.info("Cliquez sur 'Lancer le Scan Complet' pour commencer.")
