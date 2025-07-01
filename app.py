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

st.title("📈 Détecteur de Supports & Résistances (OANDA)")
st.caption("Analyse des S/R basée sur la Price Action via l'API OANDA.")

# --- Fonctions Logiques pour OANDA ---

def get_oanda_environment(environment_name):
    """Retourne l'URL de base de l'API OANDA pour l'environnement choisi."""
    if environment_name == 'Practice (Démo)':
        return "https://api-fxpractice.oanda.com"
    else: # Live (Réel)
        return "https://api-fxtrade.oanda.com"

@st.cache_data(ttl=600) # Mise en cache des données pour 10 minutes
def get_oanda_data(base_url, api_key, symbol, timeframe='daily', limit=300):
    """Récupère les données de bougies historiques depuis l'API OANDA."""
    instrument = symbol.replace('/', '_') # Convertir EURUSD en EUR_USD
    
    # Mapper nos noms de timeframe aux codes de granularité d'OANDA
    granularity_map = {'daily': 'D', 'weekly': 'W'}
    granularity = granularity_map.get(timeframe, 'D')

    url = f"{base_url}/v3/instruments/{instrument}/candles"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "count": limit,
        "granularity": granularity,
        "price": "M" # M = Midpoint (moyenne de bid/ask)
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data['candles']:
            return None

        candles = []
        for candle in data['candles']:
            if candle['complete']: # On ne prend que les bougies complètes
                candles.append({
                    'date': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        
        df = pd.DataFrame(candles)
        return df

    except requests.exceptions.HTTPError as err:
        # st.error(f"Erreur HTTP pour {symbol}: {err.response.text}")
        return None # Retourne None pour indiquer un échec
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données OANDA pour {symbol}: {e}")
        return None

def find_pivots(df, left_bars, right_bars):
    """Identique à avant, trouve les pivots sur un DataFrame."""
    if df is None or df.empty:
        return None, None
    distance_between_pivots = left_bars + right_bars
    peak_indices, _ = find_peaks(df['high'], distance=distance_between_pivots)
    resistances = df.iloc[peak_indices][['date', 'high']].rename(columns={'high': 'level'})
    resistances['type'] = 'Résistance'
    trough_indices, _ = find_peaks(-df['low'], distance=distance_between_pivots)
    supports = df.iloc[trough_indices][['date', 'low']].rename(columns={'low': 'level'})
    supports['type'] = 'Support'
    return supports, resistances

@st.cache_data(ttl=15) # Cache court pour le prix actuel
def get_oanda_current_price(base_url, api_key, account_id, symbol):
    """Récupère le prix actuel d'un instrument depuis l'API OANDA."""
    instrument = symbol.replace('/', '_')
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"instruments": instrument}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'prices' in data and len(data['prices']) > 0:
            # On calcule le prix moyen entre le bid et le ask
            bid = float(data['prices'][0]['closeoutBid'])
            ask = float(data['prices'][0]['closeoutAsk'])
            return (bid + ask) / 2
        return None
    except:
        return None

# --- Interface Utilisateur (Sidebar) ---
with st.sidebar:
    st.header("Paramètres OANDA")

    # Choix de l'environnement
    environment = st.radio("Environnement OANDA", ('Practice (Démo)', 'Live (Réel)'))
    
    # Clé API et Account ID (recommandé via secrets.toml)
    try:
        api_key = st.secrets["OANDA_API_KEY"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Identifiants OANDA chargés.", icon="✅")
    except:
        st.warning("Ajoutez vos identifiants dans secrets.toml pour plus de sécurité.")
        api_key = st.text_input("Entrez votre Clé API OANDA", type="password")
        account_id = st.text_input("Entrez votre Account ID OANDA")

    # --- LISTE DES 28 PAIRES MAJEURES + XAUUSD ---
    st.header("Sélection des Actifs")
    default_symbols = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
    
    all_major_pairs = [
        "EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD", "EUR_CHF",
        "GBP_USD", "GBP_JPY", "GBP_AUD", "GBP_NZD", "GBP_CAD", "GBP_CHF",
        "AUD_USD", "AUD_NZD", "AUD_CAD", "AUD_CHF", "AUD_JPY",
        "NZD_USD", "NZD_CAD", "NZD_CHF", "NZD_JPY",
        "USD_CAD", "USD_CHF", "USD_JPY",
        "CAD_CHF", "CAD_JPY",
        "CHF_JPY"
    ]
    all_available_symbols = ["XAU_USD"] + all_major_pairs
    
    # Utiliser le format OANDA (EUR_USD) pour l'affichage
    symbols_to_scan_oanda = st.multiselect(
        "Choisissez les actifs à analyser",
        options=sorted(list(set(all_available_symbols))),
        default=default_symbols
    )

    st.header("Paramètres de Détection")
    left_bars = st.slider("Left Bars (gauche)", min_value=1, max_value=50, value=15, help="Nombre de barres à gauche d'un pivot.")
    right_bars = st.slider("Right Bars (droite)", min_value=1, max_value=50, value=15, help="Nombre de barres à droite d'un pivot.")
    
    scan_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale de l'Application ---
if scan_button and api_key and account_id and symbols_to_scan_oanda:
    results = {'Daily': [], 'Weekly': []}
    failed_symbols = []
    base_url = get_oanda_environment(environment)
    
    total_steps = len(symbols_to_scan_oanda) * 2
    current_step = 0
    progress_bar = st.progress(0, text="Initialisation de l'analyse...")
    
    for symbol_oanda in symbols_to_scan_oanda:
        # On reconvertit au format standard pour l'affichage
        display_symbol = symbol_oanda.replace('_', '')
        
        data_fetched_for_symbol = False
        current_price = get_oanda_current_price(base_url, api_key, account_id, symbol_oanda)
        
        for timeframe, label in [('daily', 'Daily'), ('weekly', 'Weekly')]:
            current_step += 1
            progress_value = current_step / total_steps
            progress_text = f"Analyse... ({current_step}/{total_steps}) {display_symbol} - {label}"
            progress_bar.progress(progress_value, text=progress_text)
            
            df = get_oanda_data(base_url, api_key, symbol_oanda, timeframe, limit=300)
            
            if df is not None and not df.empty:
                data_fetched_for_symbol = True
                supports, resistances = find_pivots(df, left_bars, right_bars)
                last_support = supports.iloc[-1] if supports is not None and not supports.empty else None
                last_resistance = resistances.iloc[-1] if resistances is not None and not resistances.empty else None
                
                dist_s = (abs(current_price - last_support['level']) / current_price) * 100 if last_support is not None and current_price else np.nan
                dist_r = (abs(current_price - last_resistance['level']) / current_price) * 100 if last_resistance is not None and current_price else np.nan
                
                results[label].append({
                    'Actif': display_symbol,
                    'Prix Actuel': f"{current_price:.5f}" if current_price else 'N/A',
                    'Dernier Support': f"{last_support['level']:.5f}" if last_support is not None else 'N/A',
                    'Date Support': last_support['date'].strftime('%Y-%m-%d') if last_support is not None else 'N/A',
                    'Dist. Support (%)': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                    'Dernière Résistance': f"{last_resistance['level']:.5f}" if last_resistance is not None else 'N/A',
                    'Date Résistance': last_resistance['date'].strftime('%Y-%m-%d') if last_resistance is not None else 'N/A',
                    'Dist. Résistance (%)': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                })
        
        if not data_fetched_for_symbol:
            failed_symbols.append(display_symbol)

    progress_bar.progress(1.0, text="Analyse terminée !")

    if failed_symbols:
        st.warning(f"**Données non trouvées pour les actifs suivants :** {', '.join(sorted(failed_symbols))}. Vérifiez si ces instruments sont disponibles sur votre compte OANDA.")

    st.subheader("Analyse Journalière (Daily)")
    if results['Daily']:
        daily_df = pd.DataFrame(results['Daily']).sort_values(by='Actif').reset_index(drop=True)
        st.dataframe(daily_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun résultat pour l'analyse journalière.")
        
    st.subheader("Analyse Hebdomadaire (Weekly)")
    if results['Weekly']:
        weekly_df = pd.DataFrame(results['Weekly']).sort_values(by='Actif').reset_index(drop=True)
        st.dataframe(weekly_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun résultat pour l'analyse hebdomadaire.")
else:
    st.info("Veuillez configurer vos paramètres OANDA dans la barre latérale et cliquer sur 'Lancer l'Analyse'.")
