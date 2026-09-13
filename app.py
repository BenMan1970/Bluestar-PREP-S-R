"""Scanner Bluestar S/R — UI Streamlit (wrapper mince).

v8.8.1-PROD — la logique métier (Layers 0-5) vit dans bluestar_core.py,
importable en headless. Ce fichier ne contient que la couche UI.

[CACHE PATCH] Les trois fonctions pures décorées @st.cache_data en v8.7.3 sont
ré-enveloppées ici, et les globals du module coeur sont patchés, afin que les
appels INTERNES du coeur bénéficient du cache exactement comme avant.

[DEPLOY FIX] Accès aux secrets durci : sur Streamlit Cloud sans secrets.toml,
`st.secrets[...]` lève StreamlitSecretNotFoundError (pas KeyError) AU CHARGEMENT
du module -> page blanche / traceback avant tout rendu. Le bloc est désormais
tolérant et propose un repli variables d'environnement.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

# PATCH ENV-3 (audit-2) — lire .env en local AVANT toute lecture de secret.
# override=False : sur Streamlit Cloud, st.secrets doit rester prioritaire sur
# un .env résiduel. ImportError toléré : dotenv est optionnel (Cloud n'en a
# pas besoin ; les variables d'environnement du process suffisent).
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

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
    _cache_stats,
    _compute_all_confluences,
    _hash_df,
    _hash_series,
    _run_async_isolated,
    build_class_a_metrics,
    create_json_export,
    create_llm_brief,
    create_pdf_report,
    run_institutional_scan,
)

# ==============================================================================
# [ PAGE CONFIG — doit précéder tout autre appel Streamlit ]
# ==============================================================================
st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")

# ==============================================================================
# [CACHE PATCH] — IDEMPOTENT (PATCH CACHE-1, audit-2 point 1).
# Streamlit réexécute ce script à chaque interaction mais le module `core`
# persiste dans sys.modules : sans garde, chaque rerun empile une couche de
# wrapper st.cache_data supplémentaire (hachage de DataFrame en cascade).
# Paramètres (ttl / max_entries / hash_funcs) STRICTEMENT identiques à ceux
# d'avant le garde -> zéro changement de comportement. Les originaux sont
# conservés dans core._CACHE_ORIGINALS pour dé-patcher sans redémarrer.
# ==============================================================================
if not getattr(core, "_CACHE_PATCHED", False):
    core._CACHE_ORIGINALS = {
        "compute_atr": core.compute_atr,
        "compute_institutional_trend": core.compute_institutional_trend,
        "find_strong_sr_zones": core.find_strong_sr_zones,
    }
    core.compute_atr = st.cache_data(
        ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df}
    )(core.compute_atr)
    core.compute_institutional_trend = st.cache_data(
        ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: _hash_series}
    )(core.compute_institutional_trend)
    core.find_strong_sr_zones = st.cache_data(
        ttl=120, max_entries=256, show_spinner=False, hash_funcs={pd.DataFrame: _hash_df}
    )(core.find_strong_sr_zones)
    core._CACHE_PATCHED = True

st.title("📡 Scanner Bluestar Supports et Resistances")
st.caption(
    f"Version `{SCANNER_VERSION}` — zones S/R multi-TF (Swing Adaptatif, "
    f"Hybrid Touch Logic, Trend-Structure). Export JSON schéma v2.1."
)


# ==============================================================================
# [ SECRETS — accès durci pour le déploiement ]
# ==============================================================================
def _read_credentials():
    """Lit les identifiants OANDA : st.secrets d'abord, variables d'env en repli.

    DEPLOY FIX : `st.secrets[...]` lève StreamlitSecretNotFoundError quand aucun
    secrets.toml n'est présent (cas d'un premier déploiement Streamlit Cloud). Ce
    n'est pas une KeyError : l'ancien `except KeyError` laissait donc l'exception
    remonter et la page plantait au chargement. On attrape large et on retombe
    proprement sur l'environnement.
    """
    token = account = None
    source = None
    try:
        token = st.secrets.get("OANDA_ACCESS_TOKEN")
        account = st.secrets.get("OANDA_ACCOUNT_ID")
        if token and account:
            source = "st.secrets"
    except Exception:  # noqa: BLE001 — secrets.toml absent/illisible : on continue.
        token = account = None
    if not (token and account):
        token = os.environ.get("OANDA_ACCESS_TOKEN")
        account = os.environ.get("OANDA_ACCOUNT_ID")
        if token and account:
            source = "env"
    env_name = (
        (st.secrets.get("OANDA_ENV") if source == "st.secrets" else None)
        or os.environ.get("OANDA_ENV")
        or os.environ.get("OANDA_ENVIRONMENT")
    )
    if env_name:
        env_name = str(env_name).strip().lower()
        if env_name not in ("practice", "trade"):
            env_name = None
    return token, account, source, env_name


def _is_scanning_locked(session_state):
    """Vérifie si le verrou de scan est actif."""
    lock_ts = session_state.get("scanning_lock_ts")
    return bool(lock_ts and (time.time() - lock_ts) < _SCAN_LOCK_TTL_S)


def _coerce_dist_num(series: pd.Series) -> pd.Series:
    """Convertit une colonne de distance en numérique."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ).fillna(999999.0)


access_token, account_id, creds_source, oanda_env_cfg = _read_credentials()

with st.sidebar:
    st.header("1. Connexion OANDA")
    if access_token and account_id:
        st.success(f"Identifiants chargés ✓ ({creds_source})")
    else:
        st.error("Identifiants OANDA manquants")
        with st.expander("Comment les configurer"):
            st.markdown(
                "**Streamlit Cloud** — *Settings → Secrets* :\n"
                "```toml\n"
                'OANDA_ACCESS_TOKEN = "..."\n'
                'OANDA_ACCOUNT_ID = "101-004-..."\n'
                'OANDA_ENV = "..."  (facultatif)\n'
                "```\n"
                "**Local** — `.streamlit/secrets.toml` (même contenu), ou variables "
                "d'environnement de mêmes noms."
            )
    st.header("2. Sélection")
    select_all = st.checkbox(f"Tous les actifs ({len(ALL_SYMBOLS)})", value=True)
    symbols_to_scan = (
        ALL_SYMBOLS
        if select_all
        else st.multiselect(
            "Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"]
        )
    )

    st.header("3. Paramètres d'export")
    st.caption("Brief LLM — résumé humain (filtres serrés)")
    llm_max_dist = st.slider("Dist. max (%) brief LLM", 0.5, 5.0, 2.0, 0.5, key="llm_max_dist")
    llm_min_score = st.slider("Score min LLM Brief", 10, 175, 57, 5, key="llm_min_score")
    llm_statuts = st.multiselect(
        "Statuts autorisés (LLM)",
        options=["Vierge", "Testee", "Role Reverse"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="llm_statuts",
    )

    # PATCH JSON-1 : contrôles JSON DÉDIÉS, découplés du brief LLM.
    st.caption("JSON — destiné au merger (non pré-filtré par défaut)")
    json_filter_dist = st.checkbox(
        "Filtrer le JSON par distance", value=False, key="json_filter_dist"
    )
    json_max_dist = st.slider(
        "Dist. max (%) JSON", 0.5, 15.0, 5.0, 0.5, key="json_max_dist",
        disabled=not json_filter_dist,
    )
    json_min_score = st.slider("Score min JSON (merger)", 0, 175, 0, 5, key="json_min_score")
    # "Consommee" retiré des options : insatisfiable par construction (les zones
    # Consommee sont exclues avant fusion dans _build_zones_dataframe puis
    # re-filtrées dans _flatten_one_tf). Mesuré : 0 zone exclue sur 16 693.
    json_statuts = st.multiselect(
        "Statuts autorisés (JSON)",
        options=["Vierge", "Testee", "Role Reverse"],
        default=["Vierge", "Testee", "Role Reverse"],
        key="json_statuts",
        help="'Consommee' a été retiré : ce statut ne peut jamais apparaître dans "
             "l'export (filtré en amont de la fusion).",
    )

    st.header("4. Détection")
    min_touches = st.slider("Min touches Forex H4", 2, 10, 2, 1)
    confluence_threshold = st.slider(
        "Seuil confluence Forex (%)", 0.3, 2.0, 0.8, 0.1,
        help="Réglage de VOLUME, pas de qualité : P(respect|touch) est constante "
             "(0.138-0.142) sur toute la plage (rapport §5.8). Les profils "
             "XAU/US30/NAS100/SPX500/DE30 ont un seuil autoritaire qui prime.",
    )
    max_dist_filter = st.slider("Filtre visuel Dist (%)", 1.0, 15.0, 3.0, 0.5)
    show_debug = st.checkbox("Afficher debug pipeline", value=False)
    if show_debug and not core.DEBUG_INSTRUMENTATION:
        st.caption(
            "⚠️ Compteurs pivots/trend-zones désactivés : ils rejouaient la "
            "détection (2-3x le coût CPU). Mettre `DEBUG_INSTRUMENTATION = True` "
            "dans bluestar_core.py pour les réactiver."
        )

    st.divider()
    if st.button("🧹 Vider le cache"):
        st.success(f"Cache vidé : {_cache_clear()} entrées")
    if st.button("🔓 Forcer libération lock"):
        st.session_state.pop("scanning_lock_ts", None)
        st.success("Lock libéré")
    try:
        _stats = _cache_stats()
        st.caption(f"Cache : {_stats['entries']} entrées / {_stats['bytes'] / 1e6:.1f} Mo")
    except Exception:  # noqa: BLE001 — affichage best-effort.
        pass


scan_button = st.button(
    "🚀 LANCER LE SCAN COMPLET",
    type="primary",
    # AUDIT-3 (E1) : migration officielle de use_container_width (dépréciée,
    # warning mesuré sous 1.63). Le DÉFAUT du bouton est "content" (vérifié
    # par inspect) -> width="stretch" explicite pour conserver la pleine largeur.
    width="stretch",
    disabled=_is_scanning_locked(st.session_state) or not (access_token and account_id),
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
    oanda_env,
):
    """Exécute le scan complet et persiste les résultats dans la session."""
    t0 = time.perf_counter()
    progress_bar = st.progress(0, text="Initialisation...")
    raw_results = _run_async_isolated(
        lambda: run_institutional_scan(
            symbols_to_scan, access_token, account_id, min_touches, oanda_env=oanda_env
        )
    )
    agg = _accumulate_scan_results(raw_results, progress_bar)
    conf_df = _compute_all_confluences(symbols_to_scan, agg, confluence_threshold)
    summaries = _build_summaries(symbols_to_scan, agg)

    json_args = (json_max_dist, json_min_score, tuple(json_statuts))
    llm_args = (llm_max_dist, llm_min_score, tuple(llm_statuts))
    class_a_metrics = build_class_a_metrics(
        symbols_to_scan, agg, conf_df, json_args, llm_args
    )

    df_h4 = pd.DataFrame(agg["results_h4"])
    df_d = pd.DataFrame(agg["results_daily"])
    df_w = pd.DataFrame(agg["results_weekly"])
    progress_bar.empty()
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
        "bars_map": agg["bars_map"],
        "spans_map": agg.get("spans_map"),
        "class_a_metrics": class_a_metrics,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


if st.session_state.get("pending_scan", False):
    st.session_state.pop("pending_scan", None)
    if not access_token or not account_id:
        st.error("Identifiants OANDA manquants")
        st.session_state.pop("scanning_lock_ts", None)
    else:
        scan_ok = False
        try:
            with st.spinner("Scan en cours..."):
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
                    oanda_env_cfg,
                )
            scan_ok = True
        except (ScanTimeoutError, OandaAuthError, KeyError, ValueError) as e:
            st.error(f"Scan interrompu : {type(e).__name__} — {e}")
        except Exception as e:  # noqa: BLE001 — ne jamais laisser le lock coincé.
            st.error(f"Erreur inattendue : {type(e).__name__} — {e}")
        finally:
            # AUDIT-3 (OPUS C3 / KIMI #5) : le verrou se libère sur TOUTES les
            # sorties, y compris l'exception de contrôle (Rerun/Stop). MESURÉ
            # sur streamlit 1.63 : RerunException hérite de BaseException ->
            # elle filtrait entre les deux except ci-dessus et laissait le
            # verrou posé jusqu'au TTL 900 s si une interaction arrivait pendant
            # le scan ou si l'onglet se déconnectait en cours de route.
            st.session_state.pop("scanning_lock_ts", None)
        # AUDIT-3 : st.rerun() SORTI du try — sur streamlit <1.38 RerunException
        # hérite d'Exception et aurait été capturée par le catch-all (erreur
        # fantôme après un scan réussi). Le rerun ne se fait que sur succès :
        # le faire aussi sur erreur effacerait le message st.error qui vient
        # d'être rendu (le bouton reste grisé jusqu'à la prochaine interaction
        # — résiduel cosmétique documenté, verrou libéré lui est garanti).
        if scan_ok:
            st.rerun()


def _render_messages(res: dict, show_debug: bool) -> None:
    """Affiche les blocs erreurs / anomalies / debug."""
    if res["scan_errors"]:
        with st.expander(f"❌ Erreurs ({len(res['scan_errors'])})"):
            for s, e in res["scan_errors"].items():
                st.error(f"{s}: {e}")
    if res["anomalies"]:
        stale_only = {
            s: m for s, m in res["anomalies"].items()
            if m.strip() == "Prix STALE (marché fermé)"
        }
        real_anomalies = {s: m for s, m in res["anomalies"].items() if s not in stale_only}
        if stale_only:
            with st.expander(f"🌙 Marchés fermés ({len(stale_only)})"):
                st.info(
                    "Prix de dernière clôture : toutes les distances de ces actifs "
                    "sont calculées sur ce prix, pas sur un prix live."
                )
                for s, m in stale_only.items():
                    st.write(f"{s}: {m}")
        if real_anomalies:
            with st.expander(f"⚠️ Anomalies ({len(real_anomalies)})"):
                for s, m in real_anomalies.items():
                    st.warning(f"{s}: {m}")
    if show_debug and res.get("debug_map"):
        with st.expander("🔍 Debug pipeline"):
            for s, dbg in res["debug_map"].items():
                st.write(f"**{s}**", dbg)


def _render_confluences(res: dict, max_dist_filter: float) -> None:
    """Affiche le tableau des confluences multi-TF."""
    if res["conf_full"].empty:
        st.info("Aucune confluence détectée.")
        return
    st.subheader("🔥 CONFLUENCES MULTI-TF")
    c_df = res["conf_full"].copy()
    c_df["dist_num"] = _coerce_dist_num(c_df["Distance %"])
    filtered_c = c_df[c_df["dist_num"] <= max_dist_filter].drop(columns=["dist_num"])
    cols = [
        c for c in [
            "Actif", "Signal", "Niveau", "Type", "Timeframes", "Nb TF",
            "Force Totale", "Score", "Statut", "Distance %", "distance_atr",
            "distance_atr_edge", "reach_probability", "zone_width_pct", "Alerte",
        ] if c in filtered_c.columns
    ]
    sort_col = "distance_atr" if "distance_atr" in filtered_c.columns else "Score"
    # AUDIT-3 (E1, correctif asymétrique OPUS) : pas de kwarg ici — le DÉFAUT
    # de st.dataframe est width="stretch" (vérifié par inspect sur 1.63),
    # exactement ce que faisait use_container_width=True. Retirer le kwarg
    # supprime le warning sans changer le rendu.
    st.dataframe(
        filtered_c[cols].sort_values(sort_col, ascending=(sort_col == "distance_atr")),
    )
    st.caption(
        "`reach_probability` = fréquence de RETOUCHE observée pour cette tranche de "
        "distance (rapport §5.6). Ce n'est pas une probabilité de rebond : "
        "P(respect|touch) ≈ 0.14 quelle que soit la zone."
    )


def _render_tf_tables(res: dict, max_dist_filter: float) -> None:
    """Affiche les tableaux S/R par timeframe."""
    for label, df in [
        ("H4", res["df_h4"]), ("Daily", res["df_daily"]), ("Weekly", res["df_weekly"]),
    ]:
        with st.expander(f"Analyse {label}"):
            if df.empty:
                st.caption("Aucune zone.")
                continue
            df_f = df.copy()
            df_f["dist_num"] = _coerce_dist_num(df_f["Dist. %"])
            # AUDIT-3 (E1) : défaut "stretch" — kwarg retiré (cf. _render_confluences).
            st.dataframe(
                df_f[df_f["dist_num"] <= max_dist_filter].drop(columns=["dist_num"]),
            )


@st.cache_data(
    ttl=600, show_spinner=False,
    hash_funcs={pd.DataFrame: _hash_df, pd.Series: _hash_series},
)
def _build_export_bytes(
    res, json_max_dist, json_min_score, json_statuts,
    llm_max_dist, llm_min_score, llm_statuts,
):
    """Construit (PDF, JSON, brief LLM) — UNE fois par (résultat, filtres).

    AUDIT-3 (OPUS C6) : AVANT, les trois fabrications lourdes — dont l'export
    JSON ~1 Mo — tournaient à CHAQUE rerun, donc à chaque coup de curseur.
    La clé de cache est fonction du contenu (hash complet mesuré, batch B) et
    des filtres : toute dérive invalide naturellement. Les tuples sont passés
    plutôt que les listes pour un hashable stable.
    """
    pdf_anomalies = {
        s: m for s, m in (res["anomalies"] or {}).items()
        if m.strip() != "Prix STALE (marché fermé)"
    }
    pdf_b = create_pdf_report(
        res["report_dict"], res["conf_full"], res["summaries"], pdf_anomalies or None
    )
    json_b = create_json_export(
        res["summaries"],
        res["conf_full"],
        json_max_dist,
        json_min_score,
        json_statuts,
        scan_errors=res.get("scan_errors"),
        missing_tfs_map=res.get("missing_tfs_map"),
        anomalies=res.get("anomalies"),
        bars_map=res.get("bars_map"),
        spans_map=res.get("spans_map"),
        calibration_profile_version=core.CALIBRATION_PROFILE_VERSION,
    )
    llm_b = create_llm_brief(
        res["summaries"], res["conf_full"], llm_max_dist, llm_min_score, llm_statuts
    )
    return pdf_b, json_b, llm_b


def _render_downloads(
    res, llm_max_dist, llm_min_score, llm_statuts,
    json_max_dist, json_min_score, json_statuts,
) -> None:
    """Affiche les boutons de téléchargement (PDF / JSON / LLM)."""
    st.divider()
    pdf_b, json_b, llm_b = _build_export_bytes(
        res,
        json_max_dist,
        json_min_score,
        tuple(json_statuts),
        llm_max_dist,
        llm_min_score,
        tuple(llm_statuts),
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📄 PDF", data=pdf_b, file_name="rapport_bluestar.pdf")
    with col2:
        st.download_button(
            "🔧 JSON (merger)", data=json_b, file_name="supports et resistances.json",
            mime="application/json",
        )
    with col3:
        st.download_button("🤖 LLM Brief", data=llm_b, file_name="brief_llm.md")


if "scan_results" in st.session_state:
    res = st.session_state["scan_results"]
    c1, c2 = st.columns(2)
    c1.metric("Durée du scan", f"{res.get('elapsed_s', '?')} s")
    c2.metric("Confluences", len(res["conf_full"]) if not res["conf_full"].empty else 0)
    _render_messages(res, show_debug)
    _render_confluences(res, max_dist_filter)
    _render_tf_tables(res, max_dist_filter)
    _render_downloads(
        res, llm_max_dist, llm_min_score, llm_statuts,
        json_max_dist if json_filter_dist else None, json_min_score, json_statuts,
    )
