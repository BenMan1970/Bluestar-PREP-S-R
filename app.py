# presentation/streamlit_app.py
import streamlit as st
from pipeline.scanner import ScannerOrchestrator
from core.config import get_settings

def main():
    st.title("📡 Institutional S/R Quant Engine")
    # ... UI widgets ...
    
    if st.button("RUN SCAN"):
        with st.spinner("Executing institutional quant pipeline..."):
            # Lancement asynchrone du moteur
            orchestrator = ScannerOrchestrator(api_keys=..., profiles=...)
            results = orchestrator.run_scan(symbols_selected)
            
            # Affichage (0 calcul ici, juste de la data viz)
            st.dataframe(results.confluences)
