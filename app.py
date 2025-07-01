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

@st.cache_data(ttl=3600) # On met en cache la détection pour 1h
def determine_oanda_environment(api_key, account_id):
    """Teste les identifiants sur les environnements Practice et Live pour trouver le bon."""
    headers = {"Authorization": f"Bearer {api_key}"}
    environments = {
        "Practice (Démo)": "https://api-fxpractice.oanda.com",
        "Live (Réel)": "https://api-fxtrade.oanda.com"
    }

    for name, url in environments.items():
        try:
            # On fait un appel simple pour tester la connexion
            test_url = f"{url}/v3/accounts/{account_id}/summary"
            response = requests.get(test_url, headers=headers, timeout=5)
            if response.status_code == 200:
                return url, name # On retourne l'URL et le nom de l'environnement valide
        except requests.exceptions.RequestException:
            continue # Si erreur de connexion, on essaie le suivant
    
    return None, None # Si aucun ne fonctionne

@st.cache_data(ttl=600)
def get_oanda_data(base_url, api_key, symbol, timeframe='daily', limit=300):
    """Récupère les données de bougies historiques."""
    url = f"{base_url}/v3/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}
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
    except requests.exceptions.HTTPError:
        return None # L'échec sera géré dans la boucle principale


def find_pivots(df, left_bars, right_bars):
    if df is None or df.empty: return None, None
    distance = left_bars + right_bars
    r_indices, _ = find_peaks(df['high'], distance=distance)
    s_indices, _ = find_peaks(-df['low'], distance=distance)
    resistances = df.iloc[r_indices][['date', 'high']].rename(columns={'high': 'level'})
    supports = df.iloc[s_indices][['date', 'low']].rename(columns={'low': 'level'})
    return supports, resistances

@st.cache_data(ttl=15)
def get_oanda_current_price(base_url, api_key, account_id, symbol):
    url = f"{base_url}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {api_key}"}
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
    
    # Récupération des secrets
    try:
        api_key = st.secrets["OANDA_API_KEY"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
    except:
        api_key = None
        account_id = None

    # Sélection des actifs
    st.header("Sélection des Actifs")
    default_symbols = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
    all_available_symbols = sorted(list(set(default_symbols + [
        "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_NZD", "EUR_CAD", "EUR_CHF", "GBP_JPY", "GBP_AUD", 
        "GBP_NZD", "GBP_CAD", "GBP_CHF", "AUD_NZD", "AUD_CAD", "AUD_CHF", "AUD_JPY", "NZD_CAD", 
        "NZD_CHF", "NZD_JPY", "CAD_CHF", "CAD_JPY", "CHF_JPY"
    ])))
    symbols_to_scan = st.multiselect("Choisissez les actifs", options=all_available_symbols, default=default_symbols)

    # Paramètres de détection
    st.header("Paramètres de Détection")
    left_bars = st.slider("Left Bars (gauche)", 1, 50, 15)
    right_bars = st.slider("Right Bars (droite)", 1, 50, 15)
    
    scan_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale de l'Application ---
if not api_key or not account_id:
    st.warning("Veuillez configurer `OANDA_API_KEY` et `OANDA_ACCOUNT_ID` dans votre fichier `secrets.toml`.")
else:
    base_url, env_name = determine_oanda_environment(api_key, account_id)
    
    if not base_url:
        st.error("Impossible de valider vos identifiants OANDA sur les environnements Practice ou Live. Veuillez vérifier votre clé API et votre Account ID.")
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
            current_price = get_oanda_current_price(base_url, api_key, account_id, symbol)
            
            for j, timeframe in enumerate(['daily', 'weekly']):
                progress_step = i * 2 + j + 1
                progress_text = f"Analyse... ({progress_step}/{total_steps}) {symbol.replace('_', '/')} - {timeframe.capitalize()}"
                progress_bar.progress(progress_step / total_steps, text=progress_text)

                df = get_oanda_data(base_url, api_key, symbol, timeframe, limit=300)
                
                if df is not None and not df.empty:
                    data_fetched = True
                    label = timeframe.capitalize()
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
            
            if not data_fetched:
                failed_symbols.append(symbol.replace('_', '/'))

        progress_bar.empty()
        st.info("Analyse terminée !")

        if failed_symbols:
            st.warning(f"**Données non trouvées pour :** {', '.join(sorted(failed_symbols))}. Ces instruments ne sont peut-être pas disponibles sur votre compte.")

        for label in ['Daily', 'Weekly']:
            st.subheader(f"Analyse {label.lower().replace('y', 'ière')} ({label})")
            if results[label]:
                df_res = pd.DataFrame(results[label]).sort_values(by='Actif').reset_index(drop=True)
                st.dataframe(df_res, use_container_width=True, hide_index=True)
            else:
                st.info(f"Aucun résultat pour l'analyse {label.lower().replace('y', 'ière')}.")
