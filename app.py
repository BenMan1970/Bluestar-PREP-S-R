"""Scanner Bluestar S/R — UI Streamlit (wrapper mince).

v8.8.0 — la logique métier (Layers 0-5) vit désormais dans bluestar_core.py,
importable en headless. Ce fichier ne contient plus que la couche UI.

[CACHE PATCH] Les trois fonctions pures qui étaient décorées @st.cache_data
dans la v8.7.3 sont ré-enveloppées ici, et les globals du module coeur sont
patchés, afin que les appels INTERNES du coeur (ex. _process_tf_frame ->
compute_atr) continuent de bénéficier du cache Streamlit exactement comme
avant. Parité stricte avec la v8.7.3.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import bluestar_core as core
from bluestar_core import (  # noqa: F401  (noms ré-exportés pour l'UI)
    ALL_SYMBOLS,
    OandaAuthError,
    SCANNER_VERSION,
    ScanTimeoutError,
    _SCAN_LOCK_TTL_S,
    _accumulate_scan_results,
    _build_summaries,
    _cache_clear,
    _compute_all_confluences,
    _hash_df,
    _run_async_isolated,
    build_class_a_metrics,
    create_json_export,
    create_llm_brief,
    create_pdf_report,
    run_institutional_scan,
)

# [CACHE PATCH] parité v8.7.3 : mêmes ttl / max_entries / hash_funcs.
core.compute_atr = st.cache_data(
    ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df}
)(core.compute_atr)
core.compute_institutional_trend = st.cache_data(
    ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: core._hash_series}
)(core.compute_institutional_trend)
core.find_strong_sr_zones = st.cache_data(
    ttl=120, max_entries=256, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df}
)(core.find_strong_sr_zones)

# ==============================================================================
# [ LAYER 6: STREAMLIT UI ]
# ==============================================================================
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")
st.title("📡 Scanner Bluestar Supports et Resistances")
st.markdown(
    "Zones S/R avec **Swing Adaptatif**, **Hybrid Touch Logic** "
    "et **Trend-Structure (fix signe v8.7)**."
)


def _is_scanning_locked(session_state):
    """Vérifie si le verrou de scan est actif."""
    lock_ts = session_state.get("scanning_lock_ts")
    if lock_ts and (time.time() - lock_ts) < _SCAN_LOCK_TTL_S:
        return True
    return False


def _coerce_dist_num(series: pd.Series) -> pd.Series:
    """Convertit une colonne de distance en numérique."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(999999.0)


with st.sidebar:
    st.header("1. Connexion OANDA")
    try:
        access_token = st.secrets["OANDA_ACCESS_TOKEN"]
        account_id = st.secrets["OANDA_ACCOUNT_ID"]
        st.success("Secrets charges ✓")
    except KeyError:
        access_token, account_id = None, None
        st.error("Secrets OANDA manquants")

    st.header("2. Selection")
    select_all = st.checkbox(f"Tous les actifs ({len(ALL_SYMBOLS)})", value=True)
    symbols_to_scan = (
        ALL_SYMBOLS
        if select_all
        else st.multiselect(
            "Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"]
        )
    )

    st.header("3. Parametres d'export")
    st.caption("Brief LLM — résumé humain (filtres serrés)")
    llm_max_dist = st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    llm_min_score = st.slider("Score min LLM Brief", 10, 175, 57, 5, key="llm_min_score")
    llm_statuts = st.multiselect(
        "Statuts autorises (LLM)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    # PATCH JSON-1 (point 2) : contrôles JSON DÉDIÉS, découplés du brief LLM.
    # Avant, le JSON héritait de `llm_max_dist` (2.0 %) et de `llm_statuts` :
    # 8/33 actifs sortaient en none_detected alors que la détection produisait
    # 4 à 15 confluences (zones 3-TF de score 58.5 écartées pour 3.6 % de distance).
    st.caption("JSON — destiné au merger (non pré-filtré par défaut)")
    json_filter_dist = st.checkbox(
        "Filtrer le JSON par distance", value=False, key="json_filter_dist"
    )
    json_max_dist = st.slider(
        "Dist. max (%) JSON", 0.5, 15.0, 5.0, 0.5, key="json_max_dist",
        disabled=not json_filter_dist,
    )
    json_min_score = st.slider("Score min JSON (merger)", 0, 175, 0, 5, key="json_min_score")
    json_statuts = st.multiselect(
        "Statuts autorises (JSON)",
        options=["Vierge", "Testee", "Role Reverse", "Consommee"],
        default=["Vierge", "Testee", "Role Reverse", "Consommee"],
        key="json_statuts",
    )

    st.header("4. Detection")
    min_touches = st.slider("Min touches Forex H4", 2, 10, 2, 1)
    confluence_threshold = st.slider("Seuil confluence Forex (%)", 0.3, 2.0, 0.8, 0.1)
    max_dist_filter = st.slider("Filtre visuel Dist (%)", 1.0, 15.0, 3.0, 0.5)
    show_debug = st.checkbox("Afficher debug pipeline", value=False)

    if st.button("🧹 Vider le cache"):
        st.success(f"Cache vide : {_cache_clear()} entrees")
    if st.button("🔓 Forcer liberation lock"):
        st.session_state.pop("scanning_lock_ts", None)
        st.success("Lock libere")


scan_button = st.button(
    "🚀 LANCER LE SCAN COMPLET",
    type="primary",
    use_container_width=True,
    disabled=_is_scanning_locked(st.session_state),
)

if scan_button and symbols_to_scan and not _is_scanning_locked(st.session_state):
    st.session_state["scanning_lock_ts"] = time.time()
    st.session_state["pending_scan"] = True
    st.rerun()



def _execute_scan(
    symbols_to_scan,
    access_token,
    account_id,
    min_touches,
    confluence_threshold,
    llm_max_dist,
    llm_min_score,
    llm_statuts,
    json_max_dist,
    json_min_score,
    json_statuts,
):
    """Exécute le scan complet et persiste les résultats dans la session."""
    progress_bar = st.progress(0, text="Initialisation...")
    raw_results = _run_async_isolated(
        lambda: run_institutional_scan(symbols_to_scan, access_token, account_id, min_touches)
    )
    agg = _accumulate_scan_results(raw_results, progress_bar)
    conf_df = _compute_all_confluences(symbols_to_scan, agg, confluence_threshold)
    summaries = _build_summaries(symbols_to_scan, agg)

    # Instrumentation de classe A : non invasive, neutre quand le flag est False.
    # PATCH JSON-1 : json_args reflète désormais les paramètres JSON DÉDIÉS
    # (découplés du brief LLM), plus les réglages LLM.
    json_args = (json_max_dist, json_min_score, tuple(json_statuts))
    llm_args = (llm_max_dist, llm_min_score, tuple(llm_statuts))
    class_a_metrics = build_class_a_metrics(
        symbols_to_scan, agg, conf_df, json_args, llm_args
    )

    df_h4 = pd.DataFrame(agg["results_h4"])
    df_d = pd.DataFrame(agg["results_daily"])
    df_w = pd.DataFrame(agg["results_weekly"])
    st.session_state["scan_results"] = {
        "df_h4": df_h4,
        "df_daily": df_d,
        "df_weekly": df_w,
        "conf_full": conf_df,
        "report_dict": {"H4": df_h4, "Daily": df_d, "Weekly": df_w},
        "summaries": summaries,
        "anomalies": agg["anomalies_map"],
        "scan_errors": agg["scan_errors"],
        "missing_tfs_map": agg["missing_tfs_map"],
        "debug_map": agg["debug_map"],
        # PATCH JSON-2 : bars_map expose la profondeur reellement chargee par TF,
        # necessaire au bloc data_window du JSON v2.
        "bars_map": agg["bars_map"],
        "class_a_metrics": class_a_metrics,
    }


if st.session_state.get("pending_scan", False):
    st.session_state.pop("pending_scan", None)
    if not access_token or not account_id:
        st.error("Secrets manquants")
        st.session_state.pop("scanning_lock_ts", None)
    else:
        try:
            _execute_scan(
                symbols_to_scan,
                access_token,
                account_id,
                min_touches,
                confluence_threshold,
                llm_max_dist,
                llm_min_score,
                llm_statuts,
                json_max_dist if json_filter_dist else None,
                json_min_score,
                json_statuts,
            )
            st.session_state.pop("scanning_lock_ts", None)
            st.success("Scan termine !")
            st.rerun()
        except (ScanTimeoutError, OandaAuthError, KeyError, ValueError) as e:
            st.error(f"Crash critique: {e}")
            st.session_state.pop("scanning_lock_ts", None)


def _render_messages(res: dict, show_debug: bool) -> None:
    """Affiche les blocs erreurs / anomalies / debug."""
    if res["scan_errors"]:
        with st.expander("❌ Erreurs"):
            for s, e in res["scan_errors"].items():
                st.error(f"{s}: {e}")
    if res["anomalies"]:
        # Séparer les anomalies pures "marché fermé" (info) des vraies anomalies (warning)
        stale_only = {
            s: m for s, m in res["anomalies"].items()
            if m.strip() == "Prix STALE (marché fermé)"
        }
        real_anomalies = {
            s: m for s, m in res["anomalies"].items()
            if s not in stale_only
        }
        if stale_only:
            with st.expander(f"🌙 Marchés fermés ({len(stale_only)})"):
                for s, m in stale_only.items():
                    st.info(f"{s}: {m}")
        if real_anomalies:
            with st.expander(f"⚠️ Anomalies ({len(real_anomalies)})"):
                for s, m in real_anomalies.items():
                    st.warning(f"{s}: {m}")
    if show_debug and res.get("debug_map"):
        with st.expander("🔍 Debug pipeline (n_pivots / n_zones / n_trend_zones par TF)"):
            for s, dbg in res["debug_map"].items():
                st.write(f"**{s}**", dbg)


def _render_confluences(res: dict, max_dist_filter: float) -> None:
    """Affiche le tableau des confluences multi-TF."""
    if res["conf_full"].empty:
        return
    st.subheader("🔥 CONFLUENCES MULTI-TF")
    c_df = res["conf_full"].copy()
    c_df["dist_num"] = _coerce_dist_num(c_df["Distance %"])
    filtered_c = c_df[c_df["dist_num"] <= max_dist_filter].drop(columns=["dist_num"])
    st.dataframe(filtered_c.sort_values("Score", ascending=False), use_container_width=True)


def _render_tf_tables(res: dict, max_dist_filter: float) -> None:
    """Affiche les tableaux S/R par timeframe."""
    for label, df in [
        ("H4", res["df_h4"]),
        ("Daily", res["df_daily"]),
        ("Weekly", res["df_weekly"]),
    ]:
        st.subheader(f"Analyse {label}")
        if not df.empty:
            df_f = df.copy()
            df_f["dist_num"] = _coerce_dist_num(df_f["Dist. %"])
            st.dataframe(
                df_f[df_f["dist_num"] <= max_dist_filter].drop(columns=["dist_num"]),
                use_container_width=True,
            )


def _render_downloads(
    res: dict,
    llm_max_dist,
    llm_min_score,
    llm_statuts,
    json_max_dist,
    json_min_score,
    json_statuts,
) -> None:
    """Affiche les boutons de téléchargement (PDF / JSON / LLM).

    PATCH JSON-1 (point 2) : le JSON utilise désormais ses PROPRES paramètres
    (json_max_dist / json_min_score / json_statuts) et non plus ceux du brief LLM.
    """
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        # Exclure les anomalies "marché fermé" seules du PDF : ce sont des infos
        # d'état opérationnel (OANDA tradeable=False), pas des anomalies de données.
        # Les vraies anomalies composites (ex: "Ecart aberrant | Prix STALE") sont conservées.
        pdf_anomalies = {
            s: m for s, m in (res["anomalies"] or {}).items()
            if m.strip() != "Prix STALE (marché fermé)"
        }
        pdf_b = create_pdf_report(
            res["report_dict"], res["conf_full"], res["summaries"], pdf_anomalies or None
        )
        st.download_button("📄 PDF", data=pdf_b, file_name="rapport_bluestar.pdf")
    with col2:
        json_b = create_json_export(
            res["summaries"],
            res["conf_full"],
            json_max_dist,
            json_min_score,
            tuple(json_statuts),
            scan_errors=res.get("scan_errors"),
            missing_tfs_map=res.get("missing_tfs_map"),
            anomalies=res.get("anomalies"),
            bars_map=res.get("bars_map"),
            oanda_environment=os.environ.get("OANDA_ENV") or os.environ.get("OANDA_ENVIRONMENT"),
            calibration_profile_version=core.CALIBRATION_PROFILE_VERSION,
        )
        st.download_button("🔧 JSON", data=json_b, file_name="supports et resistances.json")
    with col3:
        llm_bytes = create_llm_brief(
            res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, tuple(llm_statuts)
        )
        st.download_button("🤖 LLM Brief", data=llm_bytes, file_name="brief_llm.md")


if "scan_results" in st.session_state:
    res = st.session_state["scan_results"]
    _render_messages(res, show_debug)
    _render_confluences(res, max_dist_filter)
    _render_tf_tables(res, max_dist_filter)
    _render_downloads(
        res,
        llm_max_dist,
        llm_min_score,
        llm_statuts,
        json_max_dist if json_filter_dist else None,
        json_min_score,
        json_statuts,
    )
