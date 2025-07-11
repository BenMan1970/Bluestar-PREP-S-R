# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont # Imports nécessaires pour l'image

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

# --- MOTEUR D'ANALYSE PROFESSIONNEL (inchangé) ---
def find_strong_sr_zones(df, zone_percentage_width=0.5, min_touches=2):
    if df is None or df.empty or len(df) < 20: return pd.DataFrame(), pd.DataFrame()
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
    last_price = df['close'].iloc[-1]
    supports = zones_df[zones_df['level'] < last_price].copy()
    resistances = zones_df[zones_df['level'] >= last_price].copy()
    return supports, resistances

# --- Fonctions de Création de Rapport ---
def generate_text_report(results_dict):
    """Génère un rapport textuel complet pour copier-coller."""
    report_lines = ["Rapport Complet des Niveaux de Support/Résistance :\n"]
    title_map = {'H4': '--- Analyse 4 Heures (H4) ---', 'Daily': '--- Analyse Journalière (Daily) ---', 'Weekly': '--- Analyse Hebdomadaire (Weekly) ---'}
    for timeframe_key, df in results_dict.items():
        if not df.empty:
            report_lines.append(title_map[timeframe_key])
            report_lines.append(df.to_string(index=False))
            report_lines.append("\n")
    return "\n".join(report_lines)

# NOUVELLE FONCTION : CRÉATION DU RAPPORT EN IMAGE
def create_image_report(results_dict):
    """Crée une image PNG à partir du dictionnaire de résultats."""
    full_text = generate_text_report(results_dict)

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.multiline_textbbox((0, 0), full_text, font=font)
    
    padding = 25
    width = text_bbox[2] + 2 * padding
    height = text_bbox[3] + 2 * padding
    
    img = Image.new('RGB', (width, height), color=(20, 25, 35)) # Fond sombre
    draw = ImageDraw.Draw(img)
    draw.multiline_text((padding, padding), full_text, font=font, fill=(230, 230, 230)) # Texte clair
    
    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

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
                        supports, resistances = find_strong_sr_zones(df, zone_percentage_width=zone_width, min_touches=min_touches)
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

            # Section de rapport avec les DEUX options
            st.subheader("📋 Options d'Exportation du Rapport")
            with st.expander("Cliquez ici pour voir les options d'exportation"):
                # Option 1: Texte
                st.markdown("**1. Copier le Rapport Texte**")
                report_text = generate_text_report(report_dict)
                st.code(report_text, language="text")

                # Option 2: Image
                st.markdown("**2. Télécharger le Rapport Image**")
                image_bytes = create_image_report(report_dict)
                st.download_button(
                    label="🖼️ Télécharger le Rapport (Image)",
                    data=image_bytes,
                    file_name=f"rapport_sr_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png"
                )

            # Affichage des résultats (inchangé)
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
