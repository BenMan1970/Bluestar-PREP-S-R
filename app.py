# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Système de Trading S/R Pro",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Système de Trading S/R Pro")
st.markdown("Identifie et classe les meilleures opportunités de trading par un score de confluence (Force + Proximité + Timeframe).")

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

# --- MOTEUR D'ANALYSE PROFESSIONNEL ---
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

def calculate_confluence_score(strength, distance, timeframe):
    """Calcule un score de pertinence pour une opportunité de trading."""
    tf_weights = {'H4': 1.0, 'Daily': 1.5, 'Weekly': 2.5}
    # Plus la force est grande, plus le score est élevé
    # Plus la distance est petite, plus le score est élevé
    # Le poids du timeframe multiplie l'importance
    if distance < 0.01: distance = 0.01 # Pour éviter la division par zéro
    score = (strength * tf_weights.get(timeframe, 1.0)) / distance
    return score

def generate_text_report(df):
    """Génère un rapport textuel pour copier-coller."""
    report_lines = ["Rapport des Top Opportunités S/R :\n"]
    for _, row in df.head(15).iterrows(): # Top 15
        report_lines.append(
            f"- **{row['Actif']} ({row['Timeframe']})** : {row['Type']} à **{row['Niveau']:.5f}**, "
            f"Force: {row['Force']}, Distance: {row['Distance (%)']}"
        )
    return "\n".join(report_lines)

# --- INTERFACE UTILISATEUR (SIDEBAR) ---
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
    default_selection = sorted(["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY", "GBP_JPY"])
    symbols_to_scan = st.multiselect("Choisissez les actifs", options=all_symbols, default=default_selection)

    st.header("3. Paramètres de Détection")
    zone_width = st.slider("Largeur de zone (%)", 0.1, 2.0, 0.4, 0.1, help="Largeur de la zone pour regrouper les pivots.")
    min_touches = st.slider("Force minimale (touches)", 2, 10, 3, 1, help="Nombre de contacts minimum pour valider une zone.")

    scan_button = st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True)

# --- LOGIQUE PRINCIPALE ---
if scan_button and symbols_to_scan:
    if not access_token or not account_id:
        st.warning("Veuillez configurer vos secrets OANDA pour lancer l'analyse.")
    else:
        base_url, env_name = determine_oanda_environment(access_token, account_id)
        if not base_url:
            st.error("Impossible de valider vos identifiants OANDA. Vérifiez vos secrets.")
        else:
            all_opportunities = []
            timeframes = ['h4', 'daily', 'weekly']
            progress_bar = st.progress(0, text="Initialisation...")
            st.success(f"Connecté à l'environnement OANDA : {env_name}")

            for i, symbol in enumerate(symbols_to_scan):
                # OPTIMISATION : Obtenir le prix une seule fois par symbole
                current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
                if current_price is None: continue

                for j, timeframe in enumerate(timeframes):
                    progress_step = (i * len(timeframes) + j) / (len(symbols_to_scan) * len(timeframes))
                    progress_bar.progress(progress_step, text=f"Analyse : {symbol.replace('_','/')} ({timeframe.upper()})")

                    df = get_oanda_data(base_url, access_token, symbol, timeframe, limit=500)
                    if df is not None and not df.empty:
                        supports, resistances = find_strong_sr_zones(df, zone_percentage_width=zone_width, min_touches=min_touches)

                        for _, s_row in supports.iterrows():
                            distance = (current_price - s_row['level']) / current_price * 100
                            all_opportunities.append({
                                'Actif': symbol.replace('_','/'), 'Timeframe': timeframe.upper(), 'Type': 'Support 📈',
                                'Niveau': s_row['level'], 'Force': int(s_row['strength']), 'Distance (%)': f"{distance:.2f}%",
                                'Score': calculate_confluence_score(s_row['strength'], distance, timeframe.upper())
                            })
                        for _, r_row in resistances.iterrows():
                            distance = (r_row['level'] - current_price) / current_price * 100
                            all_opportunities.append({
                                'Actif': symbol.replace('_','/'), 'Timeframe': timeframe.upper(), 'Type': 'Résistance 📉',
                                'Niveau': r_row['level'], 'Force': int(r_row['strength']), 'Distance (%)': f"{distance:.2f}%",
                                'Score': calculate_confluence_score(r_row['strength'], distance, timeframe.upper())
                            })
            
            progress_bar.progress(1.0, text="Analyse terminée !")

            if not all_opportunities:
                st.info("Aucune opportunité correspondant à vos critères n'a été trouvée.")
            else:
                results_df = pd.DataFrame(all_opportunities)
                results_df = results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
                
                st.subheader("🏆 Top Opportunités de Trading (Classées par Score)")
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                st.subheader("📋 Rapport pour Analyse Approfondie avec Gemini")
                with st.expander("Cliquez ici pour voir et copier le rapport"):
                    report_text = generate_text_report(results_df)
                    st.code(report_text, language="markdown")
                
                st.success("Analyse terminée. Les meilleures opportunités sont en haut du tableau.")

elif not symbols_to_scan:
    st.info("Veuillez sélectionner au moins un actif dans la barre latérale.")
else:
    st.info("Cliquez sur 'Lancer l'Analyse' pour commencer.")
