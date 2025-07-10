# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Détecteur S/R Pro (H4, D, W)",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Détecteur de Zones de Support/Résistance Pro")
st.markdown("Analyse des timeframes H4, Daily et Weekly avec un score de force pour chaque niveau.")

# --- Fonctions de l'API OANDA (inchangées) ---
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    environments = {"Practice (Démo)": "https://api-fxpractice.oanda.com", "Live (Réel)": "https://api-fxtrade.oanda.com"}
    for name, url in environments.items():
        try:
            response = requests.get(f"{url}/v3/accounts/{account_id}/summary", headers=headers, timeout=5)
            if response.status_code == 200:
                return url, name
        except requests.RequestException:
            continue
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
    except requests.RequestException:
        return None

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
    except requests.RequestException:
        return None

# --- NOUVEL INDICATEUR DE S/R PROFESSIONNEL (inchangé) ---
def find_strong_sr_zones(df, zone_percentage_width=0.5, min_touches=2):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), pd.DataFrame()
    r_indices, _ = find_peaks(df['high'], distance=5)
    s_indices, _ = find_peaks(-df['low'], distance=5)
    pivots_high = df.iloc[r_indices]['high']
    pivots_low = df.iloc[s_indices]['low']
    all_pivots = pd.concat([pivots_high, pivots_low]).sort_values()
    if all_pivots.empty:
        return pd.DataFrame(), pd.DataFrame()
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
            level = np.mean(zone)
            last_touch_date = all_pivots[all_pivots.isin(zone)].index.max()
            strong_zones.append({'level': level, 'strength': len(zone), 'last_touch_date': last_touch_date})
    if not strong_zones:
        return pd.DataFrame(), pd.DataFrame()
    zones_df = pd.DataFrame(strong_zones).sort_values(by='level').reset_index(drop=True)
    last_price = df['close'].iloc[-1]
    supports = zones_df[zones_df['level'] < last_price].copy()
    resistances = zones_df[zones_df['level'] >= last_price].copy()
    return supports, resistances

# --- Fonction de création d'image (inchangée) ---
def create_image_report(results_dict):
    full_text = ""
    title_map = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journalière (Daily)', 'Weekly': 'Analyse Hebdomadaire (Weekly)'}
    for timeframe, df in results_dict.items():
        full_text += title_map[timeframe] + "\n" + ("-"*80) + "\n"
        df_display = df.rename(columns={'Force (S)': 'F(S)', 'Dist. (S) %': 'D(S)%', 'Force (R)': 'F(R)', 'Dist. (R) %': 'D(R)%'})
        full_text += df_display.to_string(index=False) if not df.empty else "Aucune zone détectée."
        full_text += "\n\n"
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.multiline_textbbox((0, 0), full_text, font=font)
    padding = 20
    width = text_bbox[2] + 2 * padding
    height = text_bbox[3] + 2 * padding
    img = Image.new('RGB', (width, height), color=(20, 25, 35))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((padding, padding), full_text, font=font, fill=(230, 230, 230))
    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

# --- Interface Utilisateur (Sidebar) (inchangée) ---
with st.sidebar:
    st.header("Connexion OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets OANDA chargés.")
    except:
        access_token, account_id = None, None
        st.error("Secrets OANDA non trouvés.")
    st.header("Sélection des Actifs")
    all_symbols = sorted(["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY"])
    symbols_to_scan = st.multiselect("Choisissez les actifs", options=all_symbols, default=["XAU_USD", "EUR_USD", "GBP_USD"])
    st.header("Paramètres de Détection Pro")
    zone_width = st.slider(
        "Largeur de zone (%)", 0.1, 2.0, 0.4, 0.1,
        help="Pourcentage du prix pour regrouper les pivots. Plus la valeur est grande, plus les zones sont larges."
    )
    min_touches = st.slider(
        "Force minimale (touches)", 2, 10, 3, 1,
        help="Nombre minimum de contacts avec une zone pour la considérer comme valide."
    )
    scan_button = st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale ---
if not access_token or not account_id:
    st.warning("Veuillez configurer `OANDA_ACCESS_TOKEN` et `OANDA_ACCOUNT_ID` dans les secrets de Streamlit.")
elif scan_button and symbols_to_scan:
    base_url, env_name = determine_oanda_environment(access_token, account_id)
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
                progress_step = i * len(timeframes) + j + 1
                progress_text = f"Analyse... ({progress_step}/{total_steps}) {symbol.replace('_', '/')} - {timeframe.upper()}"
                progress_bar.progress(progress_step / total_steps, text=progress_text)
                df = get_oanda_data(base_url, access_token, symbol, timeframe, limit=500)
                
                if df is not None and not df.empty:
                    supports, resistances = find_strong_sr_zones(df, zone_percentage_width=zone_width, min_touches=min_touches)
                    
                    sup = supports.iloc[-1] if not supports.empty else None
                    res = resistances.iloc[0] if not resistances.empty else None
                    
                    # --- CORRECTION APPLIQUÉE ICI ---
                    # On utilise "is not None" pour éviter l'erreur d'ambiguïté de Pandas
                    dist_s = (abs(current_price - sup['level']) / current_price) * 100 if sup is not None and current_price is not None else np.nan
                    dist_r = (abs(current_price - res['level']) / current_price) * 100 if res is not None and current_price is not None else np.nan
                    
                    results[timeframe.capitalize()].append({
                        'Actif': symbol.replace('_', '/'), 
                        'Prix Actuel': f"{current_price:.5f}" if current_price is not None else 'N/A',
                        'Support': f"{sup['level']:.5f}" if sup is not None else 'N/A',
                        'Force (S)': f"{sup['strength']} touches" if sup is not None else 'N/A',
                        'Dist. (S) %': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                        'Résistance': f"{res['level']:.5f}" if res is not None else 'N/A',
                        'Force (R)': f"{res['strength']} touches" if res is not None else 'N/A',
                        'Dist. (R) %': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                    })
        
        progress_bar.empty()
        st.success(f"Analyse terminée sur l'environnement : {env_name}")
        df_h4_results = pd.DataFrame(results['H4'])
        df_daily_results = pd.DataFrame(results['Daily'])
        df_weekly_results = pd.DataFrame(results['Weekly'])
        results_for_image = {'H4': df_h4_results, 'Daily': df_daily_results, 'Weekly': df_weekly_results}
        
        if any(not df.empty for df in results_for_image.values()):
            st.divider()
            image_bytes = create_image_report(results_for_image)
            st.download_button(
                label="🖼️ Télécharger le Rapport Complet (Image)",
                data=image_bytes,
                file_name=f"rapport_sr_pro_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                mime='image/png',
                use_container_width=True
            )
        
        title_map = {'H4': 'Analyse 4 Heures (H4)', 'Daily': 'Analyse Journalière (Daily)', 'Weekly': 'Analyse Hebdomadaire (Weekly)'}
        for label, df_data in [('H4', df_h4_results), ('Daily', df_daily_results), ('Weekly', df_weekly_results)]:
            st.subheader(title_map[label])
            if not df_data.empty:
                df_res = df_data.sort_values(by='Actif').reset_index(drop=True)
                st.dataframe(df_res, use_container_width=True, hide_index=True)
            else:
                st.info(f"Aucune zone de S/R suffisamment forte n'a été trouvée pour cette unité de temps.")
else:
    st.info("Cliquez sur 'Lancer l'Analyse' pour commencer.")
