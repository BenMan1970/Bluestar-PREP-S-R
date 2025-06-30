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

# Fonction pour récupérer les données depuis l'API FinancialModelingPrep
@st.cache_data(ttl=900) # Mise en cache des données pour 15 minutes
def get_fmp_data(api_key, symbol, timeframe='daily', limit=200):
    """Récupère les données historiques depuis l'API FMP."""
    if timeframe == 'weekly':
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}"
    else: # daily
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # FMP a des formats de sortie différents
        if 'historical' in data:
            df = pd.DataFrame(data['historical'])
        else:
            df = pd.DataFrame(data)
            
        df = df.iloc[::-1].reset_index(drop=True) # Inverser pour avoir les dates les plus anciennes en premier
        df['date'] = pd.to_datetime(df['date'])
        
        # FMP ne fournit pas de données hebdomadaires directement pour les Forex/Crypto
        # On va les re-échantillonner si nécessaire
        if timeframe == 'weekly' and 'open' in df.columns:
             df.set_index('date', inplace=True)
             df_weekly = df.resample('W').agg({
                 'open': 'first',
                 'high': 'max',
                 'low': 'min',
                 'close': 'last',
                 'volume': 'sum'
             }).dropna()
             df = df_weekly.reset_index()

        return df.tail(limit) # Retourner le nombre de barres demandé
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        return None

# Fonction pour trouver les pivots (la logique principale du script)
def find_pivots(df, left_bars, right_bars):
    """
    Trouve les pivots hauts (résistances) et bas (supports).
    Utilise scipy.signal.find_peaks, une méthode efficace pour trouver les maxima/minima locaux.
    """
    if df is None or df.empty:
        return None, None

    # La distance est le nombre minimal de barres entre deux pivots. 
    # left_bars + right_bars est une bonne approximation de la logique PineScript.
    distance_between_pivots = left_bars + right_bars

    # Trouver les résistances (pics sur les plus hauts)
    peak_indices, _ = find_peaks(df['high'], distance=distance_between_pivots)
    resistances = df.iloc[peak_indices][['date', 'high']].rename(columns={'high': 'level'})
    resistances['type'] = 'Résistance'
    
    # Trouver les supports (pics sur les plus bas, en inversant les données)
    trough_indices, _ = find_peaks(-df['low'], distance=distance_between_pivots)
    supports = df.iloc[trough_indices][['date', 'low']].rename(columns={'low': 'level'})
    supports['type'] = 'Support'
    
    return supports, resistances

# Fonction pour récupérer le prix actuel
@st.cache_data(ttl=60)
def get_current_price(api_key, symbol):
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()[0]['price']
    except:
        return None

# --- Interface Utilisateur (Sidebar) ---
with st.sidebar:
    st.header("Paramètres")
    
    # Clé API FMP (utilisation des secrets de Streamlit recommandée)
    try:
        api_key = st.secrets["FMP_API_KEY"]
        st.success("Clé API FMP chargée.", icon="✅")
    except:
        st.warning("Ajoutez votre clé API dans un fichier secrets.toml pour ne pas la copier/coller.")
        api_key = st.text_input("Entrez votre clé API FinancialModelingPrep", type="password")

    # Sélection des actifs
    default_symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    symbols_to_scan = st.multiselect(
        "Choisissez les actifs à analyser",
        options=default_symbols + ["BTCUSD", "ETHUSD"], # Ajoutez d'autres si besoin
        default=default_symbols
    )

    st.subheader("Paramètres de Détection (PineScript)")
    left_bars = st.slider("Left Bars", min_value=1, max_value=50, value=15, help="Nombre de barres à gauche d'un pivot.")
    right_bars = st.slider("Right Bars", min_value=1, max_value=50, value=15, help="Nombre de barres à droite d'un pivot.")
    
    scan_button = st.button("Lancer l'Analyse", type="primary")

# --- Logique Principale de l'Application ---
if scan_button and api_key and symbols_to_scan:
    results = {'Daily': [], 'Weekly': []}
    
    progress_bar = st.progress(0, text="Analyse en cours...")
    
    for i, symbol in enumerate(symbols_to_scan):
        current_price = get_current_price(api_key, symbol)
        
        for timeframe, label in [('daily', 'Daily'), ('weekly', 'Weekly')]:
            # Mettre à jour la barre de progression
            progress_text = f"Analyse en cours... ({i+1}/{len(symbols_to_scan)}) {symbol} - {label}"
            progress_bar.progress((i / len(symbols_to_scan)), text=progress_text)
            
            # Récupération et analyse
            df = get_fmp_data(api_key, symbol, timeframe, limit=250) # On prend une période assez large
            if df is not None:
                supports, resistances = find_pivots(df, left_bars, right_bars)
                
                # Extraire le dernier support et la dernière résistance trouvés
                last_support = supports.iloc[-1] if not supports.empty else None
                last_resistance = resistances.iloc[-1] if not resistances.empty else None
                
                # Calculer la distance par rapport au prix actuel
                dist_s = (abs(current_price - last_support['level']) / current_price) * 100 if last_support is not None and current_price else np.nan
                dist_r = (abs(current_price - last_resistance['level']) / current_price) * 100 if last_resistance is not None and current_price else np.nan
                
                # Ajouter aux résultats
                results[label].append({
                    'Actif': symbol,
                    'Prix Actuel': current_price,
                    'Dernier Support': last_support['level'] if last_support is not None else 'N/A',
                    'Date Support': last_support['date'].strftime('%Y-%m-%d') if last_support is not None else 'N/A',
                    'Dist. Support (%)': f"{dist_s:.2f}%" if not np.isnan(dist_s) else 'N/A',
                    'Dernière Résistance': last_resistance['level'] if last_resistance is not None else 'N/A',
                    'Date Résistance': last_resistance['date'].strftime('%Y-%m-%d') if last_resistance is not None else 'N/A',
                    'Dist. Résistance (%)': f"{dist_r:.2f}%" if not np.isnan(dist_r) else 'N/A',
                })
    
    progress_bar.progress(1.0, text="Analyse terminée !")

    # --- Affichage des Résultats ---
    st.subheader("Analyse Journalière (Daily)")
    if results['Daily']:
        daily_df = pd.DataFrame(results['Daily'])
        st.dataframe(daily_df, use_container_width=True)
    else:
        st.warning("Aucun résultat pour l'analyse journalière.")
        
    st.subheader("Analyse Hebdomadaire (Weekly)")
    if results['Weekly']:
        weekly_df = pd.DataFrame(results['Weekly'])
        st.dataframe(weekly_df, use_container_width=True)
    else:
        st.warning("Aucun résultat pour l'analyse hebdomadaire.")

else:
    st.info("Veuillez configurer les paramètres dans la barre latérale et cliquer sur 'Lancer l'Analyse'.")
