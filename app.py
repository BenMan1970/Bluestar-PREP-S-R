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
st.info("Configuré pour l'environnement de démo (Practice) OANDA.")

# --- Fonctions Logiques pour OANDA ---

# --- CHANGEMENT ICI : L'URL de l'environnement de démo est maintenant fixée ---
BASE_URL = "https://api-fxpractice.oanda.com"

@st.cache_data(ttl=600)
def get_oanda_data(api_key, symbol, timeframe='daily', limit=300):
    """Récupère les données de bougies historiques depuis l'API OANDA."""
    instrument = symbol.replace('/', '_')
    granularity_map = {'daily': 'D', 'weekly': 'W'}
    granularity = granularity_map.get(timeframe, 'D')

    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"count": limit, "granularity": granularity, "price": "M"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data['candles']: return None

        candles = [{'date': pd.to_datetime(c['time']), 'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']), 'close': float(c['mid']['c']), 'volume': int(c['volume'])}
                   for c in data['candles'] if c['complete']]
        
        return pd.DataFrame(candles)

    except:
        return None

def find_pivots(df, left_bars, right_bars):
    """Identique à avant, trouve les pivots sur un DataFrame."""
    if df is None or df.empty: return None, None
    distance = left_bars + right_bars
    r_indices, _ = find_peaks(df['high'], distance=distance)
    s_indices, _ = find_peaks(-df['low'], distance=distance)
    resistances = df.iloc[r_indices][['date', 'high']].rename(columns={'high': 'level'})
    supports = df.iloc[s_indices][['date', 'low']].rename(columns={'low': 'level'})
    return supports, resistances

@st.cache_data(ttl=15)
def get_oanda_current_price(api_key, account_id, symbol):
    """Récupère le prix actuel d'un instrument depuis l'API OANDA."""
    instrument = symbol.replace('/', '_')
    url = f"{BASE_URL}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"instruments": instrument}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if 'prices' in data and data['prices']:
            bid = float(data['prices'][0]['closeoutBid'])
            ask = float(data['prices'][0]['closeoutAsk'])
            return (bid + ask) / 2
        return None
    except:
        return None

# --- Interface Utilisateur (Sidebar) ---
with st.sidebar:
    st.header("Paramètres OANDA")
    
    # --- CHANGEMENT ICI : Logique de secrets simplifiée ---
    try:
        api_key = st.secrets["OANDA_API_KEY"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Identifiants OANDA chargés.", icon="✅")
    except:
        st.error("Identifiants non trouvés dans secrets.toml.")
        api_key = st.text_input("Entrez votre Clé API OANDA", type="password", value="")
        account_id = st.text_input("Entrez votre Account ID OANDA", value="")

    st.header("Sélection des Actifs")
    default_symbols = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
    all_major_pairs = [
        "EUR_USD", "EUR_GBP", "EUR_JPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
        "GBP_USD", "GBP_JPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
        "AUD_USD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
        "NZD_USD", "NZDCAD", "NZDCHF", "NZDJPY",
        "USD_CAD", "USD_CHF", "USD_JPY", "CADCHF", "CADJPY", "CHFJPY"
    ]
    # Remplacer les noms pour qu'ils correspondent au format OANDA
    all_major_pairs = [p.replace("AUD", "_AUD_").replace("NZD", "_NZD_").replace("CAD", "_CAD_").replace("CHF", "_CHF_").replace("JPY", "_JPY_").replace("__", "_") for p in all_major_pairs]
    all_available_symbols = ["XAU_USD"] + all_major_pairs
    
    symbols_to_scan_oanda = st.multiselect(
        "Choisissez les actifs à analyser",
        options=sorted(list(set(all_available_symbols))),
        default=default_symbols
    )

    st.header("Paramètres de Détection")
    left_bars = st.slider("Left Bars (gauche)", 1, 50, 15)
    right_bars = st.slider("Right Bars (droite)", 1, 50, 15)
    
    scan_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale de l'Application ---
if scan_button and api_key and account_id and symbols_to_scan_oanda:
    results = {'Daily': [], 'Weekly': []}
    failed_symbols = []
    
    total_steps = len(symbols_to_scan_oanda) * 2
    progress_bar = st.progress(0, text="Initialisation...")
    
    for i, symbol_oanda in enumerate(symbols_to_scan_oanda):
        display_symbol = symbol_oanda.replace('_', '')
        data_fetched = False
        current_price = get_oanda_current_price(api_key, account_id, symbol_oanda)
        
        for timeframe, label in [('daily', 'Daily'), ('weekly', 'Weekly')]:
            progress_text = f"Analyse... ({i*2 + (1 if timeframe=='daily' else 2)}/{total_steps}) {display_symbol} - {label}"
            progress_bar.progress((i*2 + (1 if timeframe=='daily' else 2)) / total_steps, text=progress_text)
            
            df = get_oanda_data(api_key, symbol_oanda, timeframe, limit=300)
            
            if df is not None and not df.empty:
                data_fetched = True
                supports, resistances = find_pivots(df, left_bars, right_bars)
                last_s = supports.iloc[-1] if supports is not None and not supports.empty else None
                last_r = resistances.iloc[-1] if resistances is not None and not resistances.empty else None
                dist_s = (abs(current_price - last_s['level']) / current_price) * 100 if last_s is not None and current_price else np.nan
                dist_r = (abs(current_price - last_r['level']) / current_price) * 100 if last_r is not None and current_price else np.nan
                
                results[label].append({
                    'Actif': display_symbol, 'Prix Actuel': f"{current_price:.5f}" if current_price else 'N/A',
                    'Dernier Support': f"{last_s['level']:.5f}" if last_s is not None else 'N/A',
                    'Date Support': last_s['date'].strftime('%Y-%m-%d') if last_s is not None else 'N/A',
                    'Dist. Support (%)': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                    'Dernière Résistance': f"{last_r['level']:.5f}" if last_r is not None else 'N/A',
                    'Date Résistance': last_r['date'].strftime('%Y-%m-%d') if last_r is not None else 'N/A',
                    'Dist. Résistance (%)': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                })
        
        if not data_fetched: failed_symbols.append(display_symbol)

    progress_bar.progress(1.0, text="Analyse terminée !")

    if failed_symbols:
        st.warning(f"**Données non trouvées pour :** {', '.join(sorted(failed_symbols))}.")

    for label in ['Daily', 'Weekly']:
        st.subheader(f"Analyse {label.lower().replace('y', 'ière')} ({label})")
        if results[label]:
            df = pd.DataFrame(results[label]).sort_values(by='Actif').reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"Aucun résultat pour l'analyse {label.lower().replace('y', 'ière')}.")
else:
    st.info("Veuillez configurer vos paramètres dans la barre latérale et cliquer sur 'Lancer l'Analyse'.")
