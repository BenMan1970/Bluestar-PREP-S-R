# app.py
import streamlit as st
import pandas as pd
import requests
from scipy.signal import find_peaks
import numpy as np

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Détecteur S/R Forex & Or",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Détecteur de Supports & Résistances (Style LuxAlgo)")
st.caption("Traduction en Python du script PineScript pour trouver les S/R basés sur la Price Action.")

# --- Fonctions Logiques ---

@st.cache_data(ttl=900)
def get_fmp_data(api_key, symbol, timeframe='daily', limit=200):
    url_symbol = symbol.replace('/', '')
    if timeframe == 'weekly':
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{url_symbol}?apikey={api_key}"
    else:
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1day/{url_symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
        elif data:
            df = pd.DataFrame(data)
        else:
            return None 
            
        df = df.iloc[::-1].reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        
        if timeframe == 'weekly' and 'open' in df.columns:
             df.set_index('date', inplace=True)
             df_weekly = df.resample('W-FRI').agg({
                 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
             }).dropna()
             df = df_weekly.reset_index()

        return df.tail(limit)
    except requests.exceptions.HTTPError:
        return None
    except Exception as e:
        st.error(f"Erreur inattendue pour {symbol}: {e}")
        return None

def find_pivots(df, left_bars, right_bars):
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

@st.cache_data(ttl=60)
def get_current_price(api_key, symbol):
    url_symbol = symbol.replace('/', '')
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{url_symbol}?apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]['price']
        return None
    except:
        return None

# --- Interface Utilisateur (Sidebar) ---
with st.sidebar:
    st.header("Paramètres")
    try:
        api_key = st.secrets["FMP_API_KEY"]
        st.success("Clé API FMP chargée.", icon="✅")
    except:
        api_key = st.text_input("Entrez votre clé API FinancialModelingPrep", type="password")
        if not api_key:
            st.warning("Ajoutez votre clé API dans les secrets ou entrez-la ci-dessus.")

    # --- LISTE DES 28 PAIRES MAJEURES + XAUUSD ---
    default_symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
    
    # Génération de toutes les combinaisons des 8 devises majeures
    all_major_pairs = [
        # Contre l'EUR
        "EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
        # Contre le GBP (sans l'EUR déjà listé)
        "GBPUSD", "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
        # Contre le JPY
        "JPYUSD", "JPYGBP", "JPYAUD", "JPYNZD", "JPYCAD", "JPYCHF", # Note: Ce sont les paires inversées, on utilise les standards
        # ...La méthode systématique est plus simple.
        
        # Les 28 paires majeures et croisées standard
        "EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
        "GBPUSD", "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
        "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
        "NZDUSD", "NZDCAD", "NZDCHF", "NZDJPY",
        "USDCAD", "USDCHF", "USDJPY",
        "CADCHF", "CADJPY",
        "CHFJPY"
    ]
    
    # Ajout de l'Or à la liste finale
    all_available_symbols = ["XAUUSD"] + all_major_pairs
    
    st.subheader("Sélection des Actifs")
    symbols_to_scan = st.multiselect(
        "Choisissez les actifs à analyser (29)",
        options=sorted(list(set(all_available_symbols))), # Tri et suppression des doublons pour la propreté
        default=default_symbols
    )

    st.subheader("Paramètres de Détection")
    left_bars = st.slider("Left Bars (gauche)", min_value=1, max_value=50, value=15, help="Nombre de barres à gauche d'un pivot.")
    right_bars = st.slider("Right Bars (droite)", min_value=1, max_value=50, value=15, help="Nombre de barres à droite d'un pivot.")
    
    scan_button = st.button("Lancer l'Analyse", type="primary", use_container_width=True)

# --- Logique Principale de l'Application ---
if scan_button and api_key and symbols_to_scan:
    results = {'Daily': [], 'Weekly': []}
    failed_symbols = [] 

    total_steps = len(symbols_to_scan) * 2
    current_step = 0
    progress_bar = st.progress(0, text="Initialisation de l'analyse...")
    
    for symbol in symbols_to_scan:
        data_fetched_for_symbol = False
        current_price = get_current_price(api_key, symbol)
        
        for timeframe, label in [('daily', 'Daily'), ('weekly', 'Weekly')]:
            current_step += 1
            progress_value = current_step / total_steps
            progress_text = f"Analyse en cours... ({current_step}/{total_steps}) {symbol} - {label}"
            progress_bar.progress(progress_value, text=progress_text)
            
            df = get_fmp_data(api_key, symbol, timeframe, limit=300)
            if df is not None and not df.empty:
                data_fetched_for_symbol = True
                supports, resistances = find_pivots(df, left_bars, right_bars)
                last_support = supports.iloc[-1] if supports is not None and not supports.empty else None
                last_resistance = resistances.iloc[-1] if resistances is not None and not resistances.empty else None
                
                dist_s = (abs(current_price - last_support['level']) / current_price) * 100 if last_support is not None and current_price else np.nan
                dist_r = (abs(current_price - last_resistance['level']) / current_price) * 100 if last_resistance is not None and current_price else np.nan
                
                results[label].append({
                    'Actif': symbol,
                    'Prix Actuel': f"{current_price:.5f}" if current_price else 'N/A',
                    'Dernier Support': f"{last_support['level']:.5f}" if last_support is not None else 'N/A',
                    'Date Support': last_support['date'].strftime('%Y-%m-%d') if last_support is not None else 'N/A',
                    'Dist. Support (%)': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                    'Dernière Résistance': f"{last_resistance['level']:.5f}" if last_resistance is not None else 'N/A',
                    'Date Résistance': last_resistance['date'].strftime('%Y-%m-%d') if last_resistance is not None else 'N/A',
                    'Dist. Résistance (%)': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                })
        
        if not data_fetched_for_symbol and symbol not in failed_symbols:
            failed_symbols.append(symbol)

    progress_bar.progress(1.0, text="Analyse terminée !")

    if failed_symbols:
        st.warning(
            f"**Données non trouvées pour les actifs suivants :** {', '.join(sorted(failed_symbols))}. "
            "Cela est probablement dû à des limitations de votre plan API sur FinancialModelingPrep."
        )

    st.subheader("Analyse Journalière (Daily)")
    if results['Daily']:
        daily_df = pd.DataFrame(results['Daily'])
        st.dataframe(daily_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun résultat pour l'analyse journalière.")
        
    st.subheader("Analyse Hebdomadaire (Weekly)")
    if results['Weekly']:
        weekly_df = pd.DataFrame(results['Weekly'])
        st.dataframe(weekly_df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun résultat pour l'analyse hebdomadaire.")
else:
    st.info("Veuillez configurer les paramètres dans la barre latérale et cliquer sur 'Lancer l'Analyse'.")
