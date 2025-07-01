# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Détecteur S/R Forex & Or (OANDA)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Détecteur de Supports & Résistances")

# --- Fonctions Logiques pour OANDA ---
@st.cache_data(ttl=3600)
def determine_oanda_environment(access_token, account_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    environments = {"Practice (Démo)": "https://api-fxpractice.oanda.com", "Live (Réel)": "https://api-fxtrade.oanda.com"}
    for name, url in environments.items():
        try:
            response = requests.get(f"{url}/v3/accounts/{account_id}/summary", headers=headers, timeout=5)
            if response.status_code == 200:
                return url, name
        except:
            continue
    return None, None

@st.cache_data(ttl=600)
def get_oanda_data(base_url, access_token, symbol, timeframe='daily', limit=300):
    url = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"count": limit, "granularity": {'daily': 'D', 'weekly': 'W'}[timeframe], "price": "M"}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('candles'): return None
        candles = [{'date': pd.to_datetime(c['time']), 'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']), 'close': float(c['mid']['c']), 'volume': int(c['volume'])}
                   for c in data.get('candles', []) if c.get('complete')]
        return pd.DataFrame(candles)
    except:
        return None

def find_pivots(df, left_bars, right_bars):
    if df is None or df.empty: return None, None
    distance = left_bars + right_bars
    r_indices, _ = find_peaks(df['high'], distance=distance)
    s_indices, _ = find_peaks(-df['low'], distance=distance)
    resistances = df.iloc[r_indices][['date', 'high']].rename(columns={'high': 'level'})
    supports = df.iloc[s_indices][['date', 'low']].rename(columns={'low': 'level'})
    return supports, resistances

@st.cache_data(ttl=15)
def get_oanda_current_price(base_url, access_token, account_id, symbol):
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"instruments": symbol}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if 'prices' in data and data.get('prices'):
            bid = float(data['prices'][0]['closeoutBid'])
            ask = float(data['prices'][0]['closeoutAsk'])
            return (bid + ask) / 2
        return None
    except:
        return None

# --- Interface Utilisateur (Sidebar) ---
with st.sidebar:
    st.header("Paramètres")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
    except:
        access_token, account_id = None, None

    st.header("Sélection des Actifs")
    all_available_symbols = sorted(list(set([
        "XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD", "EUR_CHF", "GBP_JPY", "GBP_AUD",
        "GBP_NZD", "GBP_CAD", "GBP_CHF", "AUD_NZD", "AUD_CAD", "AUD_CHF", "AUD_JPY", "NZD_CAD",
        "NZD_CHF", "NZD_JPY", "CAD_CHF", "CAD_JPY", "CHF_JPY"
    ])))
    
    symbols_to_scan = st.multiselect("Choisissez les actifs", 
                                     options=all_available_symbols, 
                                     default=all_available_symbols)

    st.header("Paramètres de Détection")
    left_bars = st.slider("Left Bars (gauche)", 1, 50, 15)
    right_bars = st.slider("Right Bars (droite)", 1, 50, 15)
    
    scan_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale ---
if not access_token or not account_id:
    st.warning("Veuillez configurer `OANDA_ACCESS_TOKEN` et `OANDA_ACCOUNT_ID` dans `secrets.toml`.")
else:
    base_url, env_name = determine_oanda_environment(access_token, account_id)
    if not base_url:
        st.error("Impossible de valider vos identifiants OANDA. Vérifiez `secrets.toml`.")
    elif not scan_button:
        st.success(f"Connecté à l'environnement OANDA : **{env_name}**. Prêt à lancer l'analyse.")
        st.info("Configurez vos actifs et cliquez sur 'Lancer l'Analyse'.")

    if scan_button and base_url and symbols_to_scan:
        st.success(f"Analyse en cours sur l'environnement : **{env_name}**")
        results = {'Daily': [], 'Weekly': []}
        failed_symbols = []
        progress_bar = st.progress(0, text="Initialisation...")
        total_steps = len(symbols_to_scan) * 2
        
        for i, symbol in enumerate(symbols_to_scan):
            data_fetched = False
            current_price = get_oanda_current_price(base_url, access_token, account_id, symbol)
            
            for j, timeframe in enumerate(['daily', 'weekly']):
                progress_step = i * 2 + j + 1
                label = timeframe.capitalize()
                progress_text = f"Analyse... ({progress_step}/{total_steps}) {symbol.replace('_', '/')} - {label}"
                progress_bar.progress(progress_step / total_steps, text=progress_text)
                df = get_oanda_data(base_url, access_token, symbol, timeframe, limit=300)
                
                if df is not None and not df.empty:
                    data_fetched = True
                    supports, resistances = find_pivots(df, left_bars, right_bars)
                    last_s = supports.iloc[-1] if supports is not None and not supports.empty else None
                    last_r = resistances.iloc[-1] if resistances is not None and not resistances.empty else None
                    dist_s = (abs(current_price - last_s['level']) / current_price) * 100 if last_s is not None and current_price else np.nan
                    dist_r = (abs(current_price - last_r['level']) / current_price) * 100 if last_r is not None and current_price else np.nan
                    
                    results[label].append({
                        'Actif': symbol.replace('_', '/'), 'Prix Actuel': f"{current_price:.5f}" if current_price else 'N/A',
                        'Dernier Support': f"{last_s['level']:.5f}" if last_s is not None else 'N/A',
                        'Date Support': last_s['date'].strftime('%Y-%m-%d') if last_s is not None else 'N/A',
                        'Dist. Support (%)': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                        'Dernière Résistance': f"{last_r['level']:.5f}" if last_r is not None else 'N/A',
                        'Date Résistance': last_r['date'].strftime('%Y-%m-%d') if last_r is not None else 'N/A',
                        'Dist. Résistance (%)': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                    })
            if not data_fetched: failed_symbols.append(symbol.replace('_', '/'))

        progress_bar.empty()
        st.info("Analyse terminée !")
        if failed_symbols:
            st.warning(f"**Données non trouvées pour :** {', '.join(sorted(failed_symbols))}.")
            
        for label in ['Daily', 'Weekly']:
            st.subheader(f"Analyse {label.lower().replace('y', 'ière')} ({label})")
            if results[label]:
                df_res = pd.DataFrame(results[label]).sort_values(by='Actif').reset_index(drop=True)
                # --- CHANGEMENT ICI : On utilise st.table au lieu de st.dataframe ---
                st.table(df_res)
            else:
                st.info(f"Aucun résultat pour l'analyse {label.lower().replace('y', 'ière')}.")
