--- appS&R13.py (original)
+++ appS&R13_fixed.py (corrigé)
@@ -31,4 +31,5 @@
   F2-F10. Voir historique.
 """
+
 from __future__ import annotations
 
@@ -49,4 +50,5 @@
 try:
     from zoneinfo import ZoneInfo
+
     _NY_TZ: Optional[ZoneInfo] = ZoneInfo("America/New_York")
 except ImportError:
@@ -81,4 +83,5 @@
     for pat in _TOKEN_REDACT_PATTERNS:
         try:
+
             def _repl(match: re.Match) -> str:
                 if match.lastindex and match.lastindex >= 1:
@@ -87,4 +90,5 @@
                         return prefix + "***REDACTED***"
                 return "***REDACTED***"
+
             out = pat.sub(_repl, out)
         except Exception:  # noqa: BLE001 - redaction best-effort, ne doit jamais crasher le logging
@@ -121,5 +125,5 @@
 
 
-logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
+logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
 _sensitive_filter = _SensitiveDataFilter()
 root_logger = logging.getLogger()
@@ -194,5 +198,6 @@
     now = time.monotonic()
     stale = [
-        k for k, (ts, payload, _sz) in _OANDA_CACHE.items()
+        k
+        for k, (ts, payload, _sz) in _OANDA_CACHE.items()
         if (now - ts) > _cache_ttl(k[3], payload is _CACHE_EMPTY)
     ]
@@ -226,5 +231,7 @@
 def _cache_set(env_url, acct_id, symbol, timeframe, frame):
     k = _cache_key(env_url, acct_id, symbol, timeframe)
-    payload, sz = (_CACHE_EMPTY, 64) if frame is None else (_make_readonly(frame), _df_approx_bytes(frame))
+    payload, sz = (
+        (_CACHE_EMPTY, 64) if frame is None else (_make_readonly(frame), _df_approx_bytes(frame))
+    )
     with _CACHE_LOCK:
         old = _OANDA_CACHE.pop(k, None)
@@ -254,10 +261,37 @@
 # ==============================================================================
 ALL_SYMBOLS: Final[List[str]] = [
-    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
-    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
-    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
-    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
-    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
-    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR",
+    "EUR_USD",
+    "GBP_USD",
+    "USD_JPY",
+    "USD_CHF",
+    "USD_CAD",
+    "AUD_USD",
+    "NZD_USD",
+    "EUR_GBP",
+    "EUR_JPY",
+    "EUR_CHF",
+    "EUR_AUD",
+    "EUR_CAD",
+    "EUR_NZD",
+    "GBP_JPY",
+    "GBP_CHF",
+    "GBP_AUD",
+    "GBP_CAD",
+    "GBP_NZD",
+    "AUD_JPY",
+    "AUD_CAD",
+    "AUD_CHF",
+    "AUD_NZD",
+    "CAD_JPY",
+    "CAD_CHF",
+    "CHF_JPY",
+    "NZD_JPY",
+    "NZD_CAD",
+    "NZD_CHF",
+    "XAU_USD",
+    "US30_USD",
+    "NAS100_USD",
+    "SPX500_USD",
+    "DE30_EUR",
 ]
 _GRANULARITY_MAP: Final[Dict[str, str]] = {"h4": "H4", "daily": "D", "weekly": "W"}
@@ -281,6 +315,10 @@
             h.update(f"idx:{frame.index[0]}:{frame.index[-1]}|".encode())
         n = len(frame)
-        sample = frame if n <= 32 else pd.concat(
-            [frame.iloc[:8], frame.iloc[n // 2 - 4:n // 2 + 4], frame.iloc[-8:]], copy=False
+        sample = (
+            frame
+            if n <= 32
+            else pd.concat(
+                [frame.iloc[:8], frame.iloc[n // 2 - 4 : n // 2 + 4], frame.iloc[-8:]], copy=False
+            )
         )
         h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
@@ -327,7 +365,10 @@
         normalized = [
             {k: str(v)[:80] for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
-            for d in lst if isinstance(d, dict)
+            for d in lst
+            if isinstance(d, dict)
         ]
-        return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()[:32]
+        return hashlib.sha256(
+            json.dumps(normalized, sort_keys=True, default=str).encode()
+        ).hexdigest()[:32]
     except Exception:  # noqa: BLE001 - hash fallback deterministe
         return f"unhashable_list_{len(lst)}"
@@ -367,11 +408,27 @@
 
 _PROFILES: Final[Dict[str, InstrumentProfile]] = {
-    "EUR_USD":    InstrumentProfile("EUR_USD",    "FOREX", 0.0001, 1.2,  0.8,  0.6,  1.5, False),
-    "GBP_USD":    InstrumentProfile(
-        "GBP_USD", "FOREX", 0.0001, 1.3, 0.85, 0.65, 1.5, False,
-        min_touches_h4=2, min_touches_daily=2, min_touches_weekly=1,
+    "EUR_USD": InstrumentProfile("EUR_USD", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False),
+    "GBP_USD": InstrumentProfile(
+        "GBP_USD",
+        "FOREX",
+        0.0001,
+        1.3,
+        0.85,
+        0.65,
+        1.5,
+        False,
+        min_touches_h4=2,
+        min_touches_daily=2,
+        min_touches_weekly=1,
     ),
-    "USD_JPY":    InstrumentProfile(
-        "USD_JPY", "FOREX", 0.01, 0.9, 0.5, 0.2, 1.5, False,
+    "USD_JPY": InstrumentProfile(
+        "USD_JPY",
+        "FOREX",
+        0.01,
+        0.9,
+        0.5,
+        0.2,
+        1.5,
+        False,
         ignore_wick_filter=True,
         min_touches_h4=1,
@@ -381,38 +438,110 @@
         max_high_low_ratio=1.8,
     ),
-    "XAU_USD":    InstrumentProfile(
-        "XAU_USD", "METAL", 0.01, 2.5, 1.5, 0.3, 3.0, True,
-        ignore_wick_filter=True, min_touches_h4=1, min_touches_daily=2,
-        min_touches_weekly=2, price_min=1500.0, price_max=6000.0,
-        major_pivot_mult=0.5, max_high_low_ratio=2.5, max_cluster_width_pct=1.5,
+    "XAU_USD": InstrumentProfile(
+        "XAU_USD",
+        "METAL",
+        0.01,
+        2.5,
+        1.5,
+        0.3,
+        3.0,
+        True,
+        ignore_wick_filter=True,
+        min_touches_h4=1,
+        min_touches_daily=2,
+        min_touches_weekly=2,
+        price_min=1500.0,
+        price_max=6000.0,
+        major_pivot_mult=0.5,
+        max_high_low_ratio=2.5,
+        max_cluster_width_pct=1.5,
     ),
-    "US30_USD":   InstrumentProfile(
-        "US30_USD", "INDEX", 1.0, 2.5, 1.5, 0.4, 2.5, True,
-        ignore_wick_filter=True, min_touches_h4=1, min_touches_daily=2,
-        min_touches_weekly=2, price_min=25000.0, price_max=60000.0,
-        major_pivot_mult=0.5, max_high_low_ratio=2.5, max_cluster_width_pct=1.5,
+    "US30_USD": InstrumentProfile(
+        "US30_USD",
+        "INDEX",
+        1.0,
+        2.5,
+        1.5,
+        0.4,
+        2.5,
+        True,
+        ignore_wick_filter=True,
+        min_touches_h4=1,
+        min_touches_daily=2,
+        min_touches_weekly=2,
+        price_min=25000.0,
+        price_max=60000.0,
+        major_pivot_mult=0.5,
+        max_high_low_ratio=2.5,
+        max_cluster_width_pct=1.5,
     ),
     "NAS100_USD": InstrumentProfile(
-        "NAS100_USD", "INDEX", 1.0, 2.5, 1.5, 0.4, 2.5, True,
-        ignore_wick_filter=True, min_touches_h4=1, min_touches_daily=2,
-        min_touches_weekly=2, price_min=10000.0, price_max=50000.0,
-        major_pivot_mult=0.5, max_high_low_ratio=2.5, max_cluster_width_pct=1.5,
+        "NAS100_USD",
+        "INDEX",
+        1.0,
+        2.5,
+        1.5,
+        0.4,
+        2.5,
+        True,
+        ignore_wick_filter=True,
+        min_touches_h4=1,
+        min_touches_daily=2,
+        min_touches_weekly=2,
+        price_min=10000.0,
+        price_max=50000.0,
+        major_pivot_mult=0.5,
+        max_high_low_ratio=2.5,
+        max_cluster_width_pct=1.5,
     ),
     "SPX500_USD": InstrumentProfile(
-        "SPX500_USD", "INDEX", 0.1, 2.2, 1.2, 0.35, 2.0, True,
-        ignore_wick_filter=True, min_touches_h4=1, min_touches_daily=2,
-        min_touches_weekly=2, price_min=3000.0, price_max=12000.0,
-        major_pivot_mult=0.5, max_high_low_ratio=2.5, max_cluster_width_pct=1.5,
+        "SPX500_USD",
+        "INDEX",
+        0.1,
+        2.2,
+        1.2,
+        0.35,
+        2.0,
+        True,
+        ignore_wick_filter=True,
+        min_touches_h4=1,
+        min_touches_daily=2,
+        min_touches_weekly=2,
+        price_min=3000.0,
+        price_max=12000.0,
+        major_pivot_mult=0.5,
+        max_high_low_ratio=2.5,
+        max_cluster_width_pct=1.5,
     ),
-    "DE30_EUR":   InstrumentProfile(
-        "DE30_EUR", "INDEX", 0.1, 2.2, 1.2, 0.35, 2.0, True,
-        ignore_wick_filter=True, min_touches_h4=1, min_touches_daily=2,
-        min_touches_weekly=2, price_min=10000.0, price_max=30000.0,
-        major_pivot_mult=0.5, max_high_low_ratio=2.5, max_cluster_width_pct=1.5,
+    "DE30_EUR": InstrumentProfile(
+        "DE30_EUR",
+        "INDEX",
+        0.1,
+        2.2,
+        1.2,
+        0.35,
+        2.0,
+        True,
+        ignore_wick_filter=True,
+        min_touches_h4=1,
+        min_touches_daily=2,
+        min_touches_weekly=2,
+        price_min=10000.0,
+        price_max=30000.0,
+        major_pivot_mult=0.5,
+        max_high_low_ratio=2.5,
+        max_cluster_width_pct=1.5,
     ),
 }
 
 _DEFAULT_PROFILE: Final[InstrumentProfile] = InstrumentProfile(
-    "DEFAULT", "FOREX", 0.0001, 1.2, 0.8, 0.6, 1.5, False,
+    "DEFAULT",
+    "FOREX",
+    0.0001,
+    1.2,
+    0.8,
+    0.6,
+    1.5,
+    False,
     max_high_low_ratio=1.8,
 )
@@ -420,13 +549,20 @@
 
 def get_profile(symbol: str) -> InstrumentProfile:
-    sym = str(symbol).upper().replace("/", "_").strip()
-    if sym in _PROFILES:
-        return _PROFILES[sym]
-    parts = sym.split("_")
-    base = parts[0] if len(parts) >= 1 else sym
+    sym_clean = str(symbol).upper().replace("/", "_").strip()
+    if sym_clean in _PROFILES:
+        return _PROFILES[sym_clean]
+    parts = sym_clean.split("_")
+    base = parts[0] if len(parts) >= 1 else sym_clean
     quote = parts[1] if len(parts) >= 2 else ""
     if quote == "JPY":
         return InstrumentProfile(
-            sym, "FOREX", 0.01, 0.9, 0.5, 0.2, 1.5, False,
+            sym_clean,
+            "FOREX",
+            0.01,
+            0.9,
+            0.5,
+            0.2,
+            1.5,
+            False,
             max_high_low_ratio=1.8,
             ignore_wick_filter=True,
@@ -440,5 +576,12 @@
         # et min_touches herites du defaut dataclass (2).
         return InstrumentProfile(
-            sym, "FOREX", 0.0001, 1.2, 0.8, 0.5, 1.5, False,
+            sym_clean,
+            "FOREX",
+            0.0001,
+            1.2,
+            0.8,
+            0.5,
+            1.5,
+            False,
             max_high_low_ratio=1.8,
         )
@@ -528,5 +671,7 @@
             try:
                 async with session.get(
-                    url, headers=headers, params=params,
+                    url,
+                    headers=headers,
+                    params=params,
                     timeout=aiohttp.ClientTimeout(total=timeout_total),
                 ) as r:
@@ -536,10 +681,10 @@
                         return None
                     if r.status in (429, 500, 502, 503, 504) and attempt < retries - 1:
-                        await asyncio.sleep(backoff * (2 ** attempt))
+                        await asyncio.sleep(backoff * (2**attempt))
                         continue
                     return None
             except Exception:  # noqa: BLE001 - retry reseau, backoff puis continue
                 if attempt < retries - 1:
-                    await asyncio.sleep(backoff * (2 ** attempt))
+                    await asyncio.sleep(backoff * (2**attempt))
                 continue
         return None
@@ -677,6 +822,6 @@
             axis=1,
         ).max(axis=1)
-        res = tr.rolling(period).mean().iloc[-1]
-        return float(res) if pd.notna(res) and res > 0 else None
+        atr_result = tr.rolling(period).mean().iloc[-1]
+        return float(atr_result) if pd.notna(atr_result) and atr_result > 0 else None
     except Exception:  # noqa: BLE001 - ATR indisponible -> None
         return None
@@ -684,5 +829,7 @@
 
 @st.cache_data(ttl=120, max_entries=512, show_spinner=False, hash_funcs={pd.Series: _hash_series})
-def compute_institutional_trend(closes: pd.Series, lookback: int = 20, threshold: float = 2.0) -> str:
+def compute_institutional_trend(
+    closes: pd.Series, lookback: int = 20, threshold: float = 2.0
+) -> str:
     if closes is None or len(closes) < lookback:
         return "NEUTRE"
@@ -711,5 +858,7 @@
 
 
-def _trend_regression_fit(recent: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
+def _trend_regression_fit(
+    recent: pd.DataFrame,
+) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
     """Helper pur extrait de _detect_trend_structure_zones (aucun changement de calcul).
 
@@ -734,5 +883,7 @@
 
 
-def _make_trend_zone(lvl: float, mult: float, current_price: float, atr_val: float, zone_type: str) -> dict:
+def _make_trend_zone(
+    lvl: float, mult: float, current_price: float, atr_val: float, zone_type: str
+) -> dict:
     """Helper pur extrait : construction d'une zone trend (logique identique)."""
     return {
@@ -749,6 +900,10 @@
 
 def _trend_support_levels(
-    lows: np.ndarray, fitted: np.ndarray, last_fit: float,
-    current_price: float, atr_val: float, max_dist_pct: float,
+    lows: np.ndarray,
+    fitted: np.ndarray,
+    last_fit: float,
+    current_price: float,
+    atr_val: float,
+    max_dist_pct: float,
 ) -> List[dict]:
     """Branche Support extraite (logique bit-a-bit identique a l'original)."""
@@ -782,6 +937,10 @@
 
 def _trend_resistance_levels(
-    highs: np.ndarray, fitted: np.ndarray, last_fit: float,
-    current_price: float, atr_val: float, max_dist_pct: float,
+    highs: np.ndarray,
+    fitted: np.ndarray,
+    last_fit: float,
+    current_price: float,
+    atr_val: float,
+    max_dist_pct: float,
 ) -> List[dict]:
     """Branche Resistance extraite (logique bit-a-bit identique a l'original)."""
@@ -895,13 +1054,15 @@
 
 def _pivot_lookback_for_tf(timeframe: str) -> int:
-    tf = timeframe.lower()
-    if tf == "weekly":
+    tf_lower = timeframe.lower()
+    if tf_lower == "weekly":
         return 5
-    if tf == "daily":
+    if tf_lower == "daily":
         return 3
     return 3
 
 
-def _pivot_prominence_threshold(df: pd.DataFrame, profile: InstrumentProfile, atr_val: float) -> float:
+def _pivot_prominence_threshold(
+    df: pd.DataFrame, profile: InstrumentProfile, atr_val: float
+) -> float:
     current_p = float(df["close"].iloc[-1])
     if current_p <= 0 or not np.isfinite(current_p):
@@ -936,6 +1097,10 @@
         rev_low = lows.iloc[::-1].reset_index(drop=True)
 
-        roll_high_right = rev_high.shift(1).rolling(n, min_periods=n).max().iloc[::-1].reset_index(drop=True)
-        roll_low_right = rev_low.shift(1).rolling(n, min_periods=n).min().iloc[::-1].reset_index(drop=True)
+        roll_high_right = (
+            rev_high.shift(1).rolling(n, min_periods=n).max().iloc[::-1].reset_index(drop=True)
+        )
+        roll_low_right = (
+            rev_low.shift(1).rolling(n, min_periods=n).min().iloc[::-1].reset_index(drop=True)
+        )
 
         candle_range = (highs - lows).clip(lower=1e-10)
@@ -983,22 +1148,22 @@
 
         pivots: List[PivotPoint] = []
-        for idx in sh_idx:
+        for sh_index in sh_idx:
             pivots.append(
                 PivotPoint(
-                    price=float(highs.iloc[idx]),
-                    weight=_time_decay_weight(int(idx), n_total),
-                    index=int(idx),
+                    price=float(highs.iloc[sh_index]),
+                    weight=_time_decay_weight(int(sh_index), n_total),
+                    index=int(sh_index),
                     kind="high",
-                    prominence=float(high_prom.iloc[idx]),
+                    prominence=float(high_prom.iloc[sh_index]),
                 )
             )
-        for idx in sl_idx:
+        for sl_index in sl_idx:
             pivots.append(
                 PivotPoint(
-                    price=float(lows.iloc[idx]),
-                    weight=_time_decay_weight(int(idx), n_total),
-                    index=int(idx),
+                    price=float(lows.iloc[sl_index]),
+                    weight=_time_decay_weight(int(sl_index), n_total),
+                    index=int(sl_index),
                     kind="low",
-                    prominence=float(low_prom.iloc[idx]),
+                    prominence=float(low_prom.iloc[sl_index]),
                 )
             )
@@ -1029,6 +1194,10 @@
         if len(fp) > len(pivots):
             pivots = fp
-            _LOG.info("Soft-fallback pivot recovery: %s %s -> %d pivots",
-                      profile.symbol, timeframe, len(pivots))
+            _LOG.info(
+                "Soft-fallback pivot recovery: %s %s -> %d pivots",
+                profile.symbol,
+                timeframe,
+                len(pivots),
+            )
 
     if len(pivots) >= 3:
@@ -1053,35 +1222,39 @@
         safe_cutoff = n_total - n
         extra: List[PivotPoint] = []
-        for k, idx in enumerate(high_idx):
-            if idx >= safe_cutoff:
+        for k, peak_idx in enumerate(high_idx):
+            if peak_idx >= safe_cutoff:
                 continue
             # Validation: extremum strict sur n barres gauche/droite
-            if idx < n or idx + n >= len(high_arr):
+            if peak_idx < n or peak_idx + n >= len(high_arr):
                 continue
-            if not (high_arr[idx] > np.nanmax(high_arr[idx - n:idx]) and
-                    high_arr[idx] > np.nanmax(high_arr[idx + 1:idx + n + 1])):
+            if not (
+                high_arr[peak_idx] > np.nanmax(high_arr[peak_idx - n : peak_idx])
+                and high_arr[peak_idx] > np.nanmax(high_arr[peak_idx + 1 : peak_idx + n + 1])
+            ):
                 continue
             extra.append(
                 PivotPoint(
-                    price=float(high_arr[idx]),
-                    weight=_time_decay_weight(int(idx), n_total),
-                    index=int(idx),
+                    price=float(high_arr[peak_idx]),
+                    weight=_time_decay_weight(int(peak_idx), n_total),
+                    index=int(peak_idx),
                     kind="high",
                     prominence=float(high_props["prominences"][k]),
                 )
             )
-        for k, idx in enumerate(low_idx):
-            if idx >= safe_cutoff:
+        for k, peak_idx in enumerate(low_idx):
+            if peak_idx >= safe_cutoff:
                 continue
-            if idx < n or idx + n >= len(low_arr):
+            if peak_idx < n or peak_idx + n >= len(low_arr):
                 continue
-            if not (low_arr[idx] < np.nanmin(low_arr[idx - n:idx]) and
-                    low_arr[idx] < np.nanmin(low_arr[idx + 1:idx + n + 1])):
+            if not (
+                low_arr[peak_idx] < np.nanmin(low_arr[peak_idx - n : peak_idx])
+                and low_arr[peak_idx] < np.nanmin(low_arr[peak_idx + 1 : peak_idx + n + 1])
+            ):
                 continue
             extra.append(
                 PivotPoint(
-                    price=float(low_arr[idx]),
-                    weight=_time_decay_weight(int(idx), n_total),
-                    index=int(idx),
+                    price=float(low_arr[peak_idx]),
+                    weight=_time_decay_weight(int(peak_idx), n_total),
+                    index=int(peak_idx),
                     kind="low",
                     prominence=float(low_props["prominences"][k]),
@@ -1103,5 +1276,7 @@
 
 
-def agglomerative_1d_clustering(pivots: List[PivotPoint], bandwidth: float) -> List[List[PivotPoint]]:
+def agglomerative_1d_clustering(
+    pivots: List[PivotPoint], bandwidth: float
+) -> List[List[PivotPoint]]:
     if not pivots:
         return []
@@ -1125,12 +1300,14 @@
 
 
-def classify_zone_status(level: float, zone_type: str, df: pd.DataFrame, formation_idx: int, atr_val: float) -> str:
+def classify_zone_status(
+    level: float, zone_type: str, df: pd.DataFrame, formation_idx: int, atr_val: float
+) -> str:
     if formation_idx >= len(df) - 1 or atr_val is None or atr_val <= 0:
         return "Vierge"
     tolerance = atr_val * 0.25
     try:
-        c_arr = df["close"].values[formation_idx + 1:]
-        h_arr = df["high"].values[formation_idx + 1:]
-        l_arr = df["low"].values[formation_idx + 1:]
+        c_arr = df["close"].values[formation_idx + 1 :]
+        h_arr = df["high"].values[formation_idx + 1 :]
+        l_arr = df["low"].values[formation_idx + 1 :]
     except Exception:  # noqa: BLE001 - slicing defensif
         return "Vierge"
@@ -1149,7 +1326,7 @@
     break_idx = int(break_positions[0])
     retest_tol = tolerance * 2
-    rc = c_arr[break_idx + 1:]
-    rh = h_arr[break_idx + 1:]
-    rl = l_arr[break_idx + 1:]
+    rc = c_arr[break_idx + 1 :]
+    rh = h_arr[break_idx + 1 :]
+    rl = l_arr[break_idx + 1 :]
     if len(rc) == 0:
         return "Consommee"
@@ -1158,17 +1335,17 @@
         return "Consommee"
     retest_idx = int(np.where(retest_mask)[0][0])
-    rc_after = rc[retest_idx + 1:]
+    rc_after = rc[retest_idx + 1 :]
     if len(rc_after) == 0:
         return "Role Reverse"
     # FIX CRITIQUE: pour Resistance, second break = retomber sous le niveau (moins tolerance)
     second_break = (
-        (rc_after > level + tolerance)
-        if zone_type == "Support"
-        else (rc_after < level - tolerance)
+        (rc_after > level + tolerance) if zone_type == "Support" else (rc_after < level - tolerance)
     )
     return "Consommee" if second_break.any() else "Role Reverse"
 
 
-def compute_structural_score(strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int) -> float:
+def compute_structural_score(
+    strength: int, nb_tf: int, tf_name: str, age_bars: int, total_bars: int
+) -> float:
     tf_w = TF_WEIGHT.get(tf_name, 1.0)
     age_r = max(age_bars, 0) / max(total_bars, 1)
@@ -1177,5 +1354,10 @@
 
 
-_STATUS_PRIORITY: Final[Dict[str, int]] = {"Vierge": 0, "Testee": 1, "Role Reverse": 2, "Consommee": 3}
+_STATUS_PRIORITY: Final[Dict[str, int]] = {
+    "Vierge": 0,
+    "Testee": 1,
+    "Role Reverse": 2,
+    "Consommee": 3,
+}
 
 
@@ -1206,10 +1388,9 @@
         touches = len(unique_touch_indices)
         max_prominence = float(np.nanmax(prominences)) if len(prominences) else 0.0
-        avg_prominence = float(np.average(prominences, weights=weights)) if weights.sum() > 0 else max_prominence
+        avg_prominence = (
+            float(np.average(prominences, weights=weights)) if weights.sum() > 0 else max_prominence
+        )
         # is_major: information pour scoring, pas garde-fou absolu
-        is_major = (
-            touches == 1
-            and max_prominence >= major_threshold
-        )
+        is_major = touches == 1 and max_prominence >= major_threshold
         # Garde-fou standard: moins de touches que requis, sauf pivot majeur
         if touches < min_touches_required and not is_major:
@@ -1219,5 +1400,7 @@
             continue
         last_idx = max(p.index for p in grp)
-        status = classify_zone_status(level=lvl, zone_type=zone_type, df=df, formation_idx=last_idx, atr_val=atr_val)
+        status = classify_zone_status(
+            level=lvl, zone_type=zone_type, df=df, formation_idx=last_idx, atr_val=atr_val
+        )
         strong.append(
             {
@@ -1332,8 +1515,18 @@
         if tr_res:
             resistance_zones.extend(tr_res)
-            _LOG.info("Trend-structure union: %s %s added %d resistance zones", symbol, timeframe, len(tr_res))
+            _LOG.info(
+                "Trend-structure union: %s %s added %d resistance zones",
+                symbol,
+                timeframe,
+                len(tr_res),
+            )
         if tr_sup:
             support_zones.extend(tr_sup)
-            _LOG.info("Trend-structure union: %s %s added %d support zones", symbol, timeframe, len(tr_sup))
+            _LOG.info(
+                "Trend-structure union: %s %s added %d support zones",
+                symbol,
+                timeframe,
+                len(tr_sup),
+            )
 
     return support_zones, resistance_zones
@@ -1345,23 +1538,23 @@
     Logique identique a l'original, extraite en helper.
     """
-    for idx, row in df_zones.iterrows():
+    for zone_idx, row in df_zones.iterrows():
         lvl = row["level"]
         ztype = row["zone_type"]
         if ztype == "Resistance" and lvl < current_price:
-            df_zones.at[idx, "zone_type"] = "Pivot"
+            df_zones.at[zone_idx, "zone_type"] = "Pivot"
         elif ztype == "Support" and lvl >= current_price:
-            df_zones.at[idx, "zone_type"] = "Pivot"
+            df_zones.at[zone_idx, "zone_type"] = "Pivot"
     return df_zones
 
 
-def _split_supports_resistances(df_zones: pd.DataFrame, current_price: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
+def _split_supports_resistances(
+    df_zones: pd.DataFrame, current_price: float
+) -> Tuple[pd.DataFrame, pd.DataFrame]:
     """Separation supports/resistances (helper pur, logique identique)."""
     supports = df_zones[
-        (df_zones["level"] < current_price)
-        & (df_zones["zone_type"].isin(["Support", "Pivot"]))
+        (df_zones["level"] < current_price) & (df_zones["zone_type"].isin(["Support", "Pivot"]))
     ].copy()
     resistances = df_zones[
-        (df_zones["level"] >= current_price)
-        & (df_zones["zone_type"].isin(["Resistance", "Pivot"]))
+        (df_zones["level"] >= current_price) & (df_zones["zone_type"].isin(["Resistance", "Pivot"]))
     ].copy()
     return supports, resistances
@@ -1428,12 +1621,15 @@
     #  ont bloque l'activation, on force un second essai)
     if supports.empty and resistances.empty and profile.asset_class in ("INDEX", "METAL"):
-        _LOG.info("Trend-mode safety net for %s %s (orphan zones blocked output)", symbol, timeframe)
-        tr_zones = (
-            _detect_trend_structure_zones(df, current_price, profile, atr_val, "Support")
-            + _detect_trend_structure_zones(df, current_price, profile, atr_val, "Resistance")
-        )
+        _LOG.info(
+            "Trend-mode safety net for %s %s (orphan zones blocked output)", symbol, timeframe
+        )
+        tr_zones = _detect_trend_structure_zones(
+            df, current_price, profile, atr_val, "Support"
+        ) + _detect_trend_structure_zones(df, current_price, profile, atr_val, "Resistance")
         if tr_zones:
             tr_df = pd.DataFrame(tr_zones)
-            tr_df["near_price"] = (np.abs(tr_df["level"] - current_price) / current_price * 100) <= 0.50
+            tr_df["near_price"] = (
+                np.abs(tr_df["level"] - current_price) / current_price * 100
+            ) <= 0.50
             supports, resistances = _split_supports_resistances(tr_df, current_price)
 
@@ -1443,5 +1639,5 @@
 def _flatten_zones_to_dataframe(zones_dict: dict) -> pd.DataFrame:
     frames = []
-    for tf, pair in zones_dict.items():
+    for timeframe_key, pair in zones_dict.items():
         try:
             sup, res = pair
@@ -1454,5 +1650,7 @@
             if tmp.empty:
                 continue
-            tmp = tmp.assign(tf=tf, type=tmp["near_price"].map({True: "Pivot", False: ztype}))
+            tmp = tmp.assign(
+                tf=timeframe_key, type=tmp["near_price"].map({True: "Pivot", False: ztype})
+            )
             cols = ["tf", "level", "strength", "age_bars", "status", "type", "near_price"]
             for c in ["prominence_atr", "is_major"]:
@@ -1462,9 +1660,12 @@
     return (
         pd.concat(frames, ignore_index=True).sort_values("level").reset_index(drop=True)
-        if frames else pd.DataFrame()
+        if frames
+        else pd.DataFrame()
     )
 
 
-def _score_and_classify_group(group: pd.DataFrame, current_price: float, bars_map: dict, symbol: str) -> dict:
+def _score_and_classify_group(
+    group: pd.DataFrame, current_price: float, bars_map: dict, symbol: str
+) -> dict:
     sub_avg = group["level"].mean()
     sub_nb_tf = group["tf"].nunique()
@@ -1479,5 +1680,11 @@
     if "is_major" in group.columns:
         major_bonus = 1.0 + 0.3 * group["is_major"].sum()
-    score = round(float((group["strength"].values * tf_w * sub_nb_tf * np.exp(-lams * age_r)).sum() * major_bonus), 1)
+    score = round(
+        float(
+            (group["strength"].values * tf_w * sub_nb_tf * np.exp(-lams * age_r)).sum()
+            * major_bonus
+        ),
+        1,
+    )
     status = max(group["status"].tolist(), key=lambda s: _STATUS_PRIORITY.get(s, 1))
     is_near_price = sub_dist <= 0.50
@@ -1504,5 +1711,8 @@
 
 def detect_confluences(
-    symbol: str, zones_dict: dict, current_price: float, bars_map: dict,
+    symbol: str,
+    zones_dict: dict,
+    current_price: float,
+    bars_map: dict,
     confluence_threshold_pct: Optional[float] = None,
 ) -> list:
@@ -1514,5 +1724,6 @@
     profile = get_profile(symbol.replace("/", "_"))
     threshold = (
-        confluence_threshold_pct if confluence_threshold_pct is not None
+        confluence_threshold_pct
+        if confluence_threshold_pct is not None
         else profile.confluence_threshold_pct
     )
@@ -1550,7 +1761,7 @@
             union(i, j)
     comp_map = {}
-    for idx in range(n):
-        root = find(idx)
-        comp_map.setdefault(root, []).append(idx)
+    for union_idx in range(n):
+        root = find(union_idx)
+        comp_map.setdefault(root, []).append(union_idx)
     confluences = []
     for indices in comp_map.values():
@@ -1607,5 +1818,6 @@
     dist_atr = (
         f"{round(abs(ctx.cp - z['level']) / ctx.atr_val, 1)}x"
-        if (ctx.atr_val and ctx.atr_val > 0) else "N/A"
+        if (ctx.atr_val and ctx.atr_val > 0)
+        else "N/A"
     )
     return {
@@ -1615,5 +1827,7 @@
         "Niveau": f"{z['level']:.5f}",
         "Force": f"{z['strength']} touches",
-        "Score (1TF)": compute_structural_score(z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len),
+        "Score (1TF)": compute_structural_score(
+            z["strength"], 1, ctx.tf_name, z["age_bars"], ctx.df_len
+        ),
         "Statut": z["status"],
         "Dist. %": f"{dist:.2f}%",
@@ -1639,7 +1853,5 @@
     data_cube = {sym: {tf: None for tf in _GRANULARITY_MAP} for sym in symbols}
     tasks = [
-        client.fetch_candles(session, sem, sym, tf)
-        for sym in symbols
-        for tf in _GRANULARITY_MAP
+        client.fetch_candles(session, sem, sym, tf) for sym in symbols for tf in _GRANULARITY_MAP
     ]
     res = await asyncio.gather(*tasks, return_exceptions=True)
@@ -1663,20 +1875,37 @@
         if not s_near.empty:
             n_s = s_near.nlargest(1, "level").iloc[0]
-            label = 'SUR support' if abs(cp - n_s['level']) / cp * 100 < 0.5 else 'S proche'
-            parts.append(f"{label}: {n_s['level']:.5f} (-{abs(cp - n_s['level']) / cp * 100:.2f}%)")
+            price_label = "SUR support" if abs(cp - n_s["level"]) / cp * 100 < 0.5 else "S proche"
+            parts.append(
+                f"{price_label}: {n_s['level']:.5f} (-{abs(cp - n_s['level']) / cp * 100:.2f}%)"
+            )
     if res is not None and not res.empty:
         r_near = res[(res["level"] > cp) & (abs(res["level"] - cp) / cp * 100 <= 5.0)]
         if not r_near.empty:
             n_r = r_near.nsmallest(1, "level").iloc[0]
-            label = 'SUR resistance' if abs(cp - n_r['level']) / cp * 100 < 0.5 else 'R proche'
-            parts.append(f"{label}: {n_r['level']:.5f} (+{abs(cp - n_r['level']) / cp * 100:.2f}%)")
+            price_label = (
+                "SUR resistance" if abs(cp - n_r["level"]) / cp * 100 < 0.5 else "R proche"
+            )
+            parts.append(
+                f"{price_label}: {n_r['level']:.5f} (+{abs(cp - n_r['level']) / cp * 100:.2f}%)"
+            )
     return "  |  ".join(parts) if parts else "Zone intermediaire"
 
 
-_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {"H4": (50, 2.0), "Daily": (50, 1.8), "Weekly": (20, 1.5)}
+_TF_TREND_PARAMS: Final[Dict[str, Tuple[int, float]]] = {
+    "H4": (50, 2.0),
+    "Daily": (50, 1.8),
+    "Weekly": (20, 1.5),
+}
 
 
 def _process_tf_frame(sym, tf_k, tf_name, df, cp, min_touches_ui, profile, sym_d):
-    debug = {"atr": None, "n_pivots": 0, "n_zones": 0, "n_trend_zones": 0, "min_touches": None, "tf": tf_name}
+    debug = {
+        "atr": None,
+        "n_pivots": 0,
+        "n_zones": 0,
+        "n_trend_zones": 0,
+        "min_touches": None,
+        "tf": tf_name,
+    }
     try:
         atr_val = compute_atr(df)
@@ -1697,8 +1926,7 @@
         if profile.asset_class in ("INDEX", "METAL"):
             try:
-                tr_count = (
-                    len(_detect_trend_structure_zones(df, cp, profile, atr_val, "Support"))
-                    + len(_detect_trend_structure_zones(df, cp, profile, atr_val, "Resistance"))
-                )
+                tr_count = len(
+                    _detect_trend_structure_zones(df, cp, profile, atr_val, "Support")
+                ) + len(_detect_trend_structure_zones(df, cp, profile, atr_val, "Resistance"))
                 debug["n_trend_zones"] = tr_count
             except Exception:  # noqa: BLE001 - debug observabilite, non bloquant
@@ -1709,9 +1937,14 @@
 
         price_ctx = _build_daily_price_context(cp, sup, res) if tf_k == "daily" else ""
-        row_ctx = _RowContext(cp=cp, atr_val=atr_val, sym_d=sym_d, tf_name=tf_name, df_len=len(df), profile=profile)
-        tf_r = (
-            [_make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx) for _, z in sup.iterrows()]
-            + [_make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx) for _, z in res.iterrows()]
-        )
+        row_ctx = _RowContext(
+            cp=cp, atr_val=atr_val, sym_d=sym_d, tf_name=tf_name, df_len=len(df), profile=profile
+        )
+        tf_r = [
+            _make_row(z, "PIVOT" if z.get("near_price") else "Support", row_ctx)
+            for _, z in sup.iterrows()
+        ] + [
+            _make_row(z, "PIVOT" if z.get("near_price") else "Resistance", row_ctx)
+            for _, z in res.iterrows()
+        ]
         seen, uniq = set(), []
         for r in tf_r:
@@ -1813,20 +2046,35 @@
                 sup_levels.extend(zp[0]["level"].tolist())
         daily_df = data_cube.get(sym, {}).get("daily")
-        last_close = float(daily_df["close"].iloc[-1]) if daily_df is not None and not daily_df.empty else None
+        last_close = (
+            float(daily_df["close"].iloc[-1])
+            if daily_df is not None and not daily_df.empty
+            else None
+        )
         anomaly = flag_data_anomaly(sym, cp, sup_levels, last_candle_close=last_close)
         if price_is_fallback:
             anomaly = f"{anomaly} | Prix fallback" if anomaly else "Prix fallback"
         return ScanResult(
-            sym, rows, zones_d, cp, trends, bars_map,
-            price_context=price_ctx, anomaly=anomaly,
-            missing_tfs=missing_tfs, price_is_fallback=price_is_fallback,
+            sym,
+            rows,
+            zones_d,
+            cp,
+            trends,
+            bars_map,
+            price_context=price_ctx,
+            anomaly=anomaly,
+            missing_tfs=missing_tfs,
+            price_is_fallback=price_is_fallback,
             debug_info=debug,
         )
     except Exception as e:  # noqa: BLE001 - isolation par symbole, log + ScanResult erreur
         _LOG.exception("Symbol processing error: %s", sym)
-        return ScanResult(sym, {}, {}, None, {}, {}, scan_error=f"Erreur interne : {type(e).__name__}")
-
-
-def _validate_symbol_coverage(requested_symbols: List[str], results: List[ScanResult]) -> Dict[str, Any]:
+        return ScanResult(
+            sym, {}, {}, None, {}, {}, scan_error=f"Erreur interne : {type(e).__name__}"
+        )
+
+
+def _validate_symbol_coverage(
+    requested_symbols: List[str], results: List[ScanResult]
+) -> Dict[str, Any]:
     requested = set(requested_symbols)
     returned = {r.symbol for r in results if isinstance(r, ScanResult)}
@@ -1882,10 +2130,14 @@
 # ==============================================================================
 _ACCENT_MAP = str.maketrans(
-    'àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ',
-    'aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN'
+    "àâäáãèéêëîïíìôöóòõùûüúçñÀÂÄÁÈÉÊËÎÏÍÔÖÓÙÛÜÚÇÑ", "aaaaaeeeeiiiiooooouuuucnAAAAEEEEIIIOOOUUUUCN"
 )
 _EMOJI_MAP = [
-    ('🟢', '[BUY]'), ('🔴', '[SELL]'), ('🔥', '[CHAUD]'),
-    ('↔️', '[PIVOT]'), ('↔', '[PIVOT]'), ('⚠️', '[PROCHE]'), ('⚠', '[PROCHE]')
+    ("🟢", "[BUY]"),
+    ("🔴", "[SELL]"),
+    ("🔥", "[CHAUD]"),
+    ("↔️", "[PIVOT]"),
+    ("↔", "[PIVOT]"),
+    ("⚠️", "[PROCHE]"),
+    ("⚠", "[PROCHE]"),
 ]
 
@@ -1899,5 +2151,5 @@
             s = s.replace(e, r)
         s = s.encode("latin-1", errors="replace").decode("latin-1")
-        return s[:max_chars - 3] + "..." if len(s) > max_chars else s
+        return s[: max_chars - 3] + "..." if len(s) > max_chars else s
     except Exception:  # noqa: BLE001 - sanitation PDF best-effort
         return ""
@@ -1906,14 +2158,25 @@
 class PDF(FPDF):
     def header(self):
-        self.set_font('Helvetica', 'B', 15)
+        self.set_font("Helvetica", "B", 15)
         self.cell(
-            0, 10, _safe_pdf_str('Rapport Scanner Bluestar - S/R'),
-            border=0, align='C', new_x='LMARGIN', new_y='NEXT'
-        )
-        self.set_font('Helvetica', '', 8)
+            0,
+            10,
+            _safe_pdf_str("Rapport Scanner Bluestar - S/R"),
+            border=0,
+            align="C",
+            new_x="LMARGIN",
+            new_y="NEXT",
+        )
+        self.set_font("Helvetica", "", 8)
         self.cell(
-            0, 6,
-            _safe_pdf_str(f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | v{SCANNER_VERSION}"),
-            border=0, align='C', new_x='LMARGIN', new_y='NEXT'
+            0,
+            6,
+            _safe_pdf_str(
+                f"Genere le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | v{SCANNER_VERSION}"
+            ),
+            border=0,
+            align="C",
+            new_x="LMARGIN",
+            new_y="NEXT",
         )
         self.ln(4)
@@ -1921,27 +2184,42 @@
     def footer(self):
         self.set_y(-15)
-        self.set_font('Helvetica', 'I', 8)
-        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')
+        self.set_font("Helvetica", "I", 8)
+        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")
 
     def chapter_title(self, title):
-        self.set_font('Helvetica', 'B', 12)
-        self.cell(0, 10, _safe_pdf_str(title), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
+        self.set_font("Helvetica", "B", 12)
+        self.cell(0, 10, _safe_pdf_str(title), border=0, align="L", new_x="LMARGIN", new_y="NEXT")
         self.ln(4)
 
     def _column_widths(self, df):
-        if 'Timeframes' in df.columns:
+        if "Timeframes" in df.columns:
             return {
-                'Actif': 20, 'Signal': 26, 'Niveau': 22, 'Type': 22, 'Timeframes': 50,
-                'Nb TF': 12, 'Force Totale': 20, 'Score': 18, 'Statut': 22,
-                'Distance %': 18, 'Alerte': 55,
+                "Actif": 20,
+                "Signal": 26,
+                "Niveau": 22,
+                "Type": 22,
+                "Timeframes": 50,
+                "Nb TF": 12,
+                "Force Totale": 20,
+                "Score": 18,
+                "Statut": 22,
+                "Distance %": 18,
+                "Alerte": 55,
             }
         return {
-            'Actif': 24, 'Prix Actuel': 24, 'Type': 20, 'Niveau': 24, 'Force': 20,
-            'Score (1TF)': 18, 'Statut': 22, 'Dist. %': 16, 'Dist. ATR': 16,
+            "Actif": 24,
+            "Prix Actuel": 24,
+            "Type": 20,
+            "Niveau": 24,
+            "Force": 20,
+            "Score (1TF)": 18,
+            "Statut": 22,
+            "Dist. %": 16,
+            "Dist. ATR": 16,
         }
 
     def chapter_body(self, df):
         if df is None or df.empty:
-            self.set_font('Helvetica', '', 10)
+            self.set_font("Helvetica", "", 10)
             self.multi_cell(0, 10, "Aucune donnee a afficher.")
             return
@@ -1950,10 +2228,18 @@
         total_w = sum(col_widths[c] for c in cols)
         x_start = self.l_margin + max(0, (self.w - self.l_margin - self.r_margin - total_w) / 2)
-        self.set_font('Helvetica', 'B', 7)
+        self.set_font("Helvetica", "B", 7)
         self.set_x(x_start)
         for col in cols:
-            self.cell(col_widths[col], 6, _safe_pdf_str(col), border=1, align='C', new_x='RIGHT', new_y='TOP')
+            self.cell(
+                col_widths[col],
+                6,
+                _safe_pdf_str(col),
+                border=1,
+                align="C",
+                new_x="RIGHT",
+                new_y="TOP",
+            )
         self.ln()
-        self.set_font('Helvetica', '', 7)
+        self.set_font("Helvetica", "", 7)
         for _, row in df.iterrows():
             self.set_x(x_start)
@@ -1962,7 +2248,11 @@
                 max_c = int(col_widths[col] / 1.25)
                 self.cell(
-                    col_widths[col], 5,
-                    (val[:max_c - 1] + '.' if len(val) > max_c else val),
-                    border=1, align='C', new_x='RIGHT', new_y='TOP',
+                    col_widths[col],
+                    5,
+                    (val[: max_c - 1] + "." if len(val) > max_c else val),
+                    border=1,
+                    align="C",
+                    new_x="RIGHT",
+                    new_y="TOP",
                 )
             self.ln()
@@ -1970,16 +2260,24 @@
 
 def create_pdf_report(results_dict, confluences_df=None, summary_list=None, anomalies=None):
-    pdf = PDF('L', 'mm', 'A4')
+    pdf = PDF("L", "mm", "A4")
     pdf.set_margins(5, 10, 5)
     pdf.add_page()
     if anomalies:
-        pdf.set_font('Helvetica', 'B', 10)
-        pdf.cell(0, 7, _safe_pdf_str('ALERTES ANOMALIES'), border=0, align='L', new_x='LMARGIN', new_y='NEXT')
-        pdf.set_font('Helvetica', '', 8)
+        pdf.set_font("Helvetica", "B", 10)
+        pdf.cell(
+            0,
+            7,
+            _safe_pdf_str("ALERTES ANOMALIES"),
+            border=0,
+            align="L",
+            new_x="LMARGIN",
+            new_y="NEXT",
+        )
+        pdf.set_font("Helvetica", "", 8)
         for sym, msg in anomalies.items():
             pdf.multi_cell(0, 5, _safe_pdf_str(f"[!] {sym} : {msg}"))
         pdf.ln(4)
     if confluences_df is not None and not confluences_df.empty:
-        pdf.chapter_title('ZONES DE CONFLUENCE')
+        pdf.chapter_title("ZONES DE CONFLUENCE")
         pdf.chapter_body(confluences_df)
         pdf.ln(10)
@@ -2005,9 +2303,5 @@
         "assets": [],
     }
-    summary_map = {
-        s["symbol"]: s
-        for s in summary_list
-        if isinstance(s, dict) and "symbol" in s
-    }
+    summary_map = {s["symbol"]: s for s in summary_list if isinstance(s, dict) and "symbol" in s}
 
     filtered_conf = pd.DataFrame()
@@ -2031,9 +2325,11 @@
         else:
             zones = []
-        output["assets"].append({
-            "symbol": sym,
-            "current_price": summary.get("current_price"),
-            "zones": zones,
-        })
+        output["assets"].append(
+            {
+                "symbol": sym,
+                "current_price": summary.get("current_price"),
+                "zones": zones,
+            }
+        )
 
     output["assets"].sort(key=lambda x: x["symbol"])
@@ -2042,5 +2338,8 @@
 
 def create_llm_brief(
-    summary_list, confluences_df, max_dist=2.0, min_score=100.0,
+    summary_list,
+    confluences_df,
+    max_dist=2.0,
+    min_score=100.0,
     allowed_statuts=("Vierge", "Testee", "Role Reverse"),
 ):
@@ -2082,5 +2381,7 @@
 st.set_page_config(page_title="Scanner Bluestar S/R", page_icon="📡", layout="wide")
 st.title("📡 Scanner Bluestar Supports et Resistances")
-st.markdown("Zones S/R avec **Swing Adaptatif**, **Hybrid Touch Logic** et **Trend-Structure (fix signe v8.7)**.")
+st.markdown(
+    "Zones S/R avec **Swing Adaptatif**, **Hybrid Touch Logic** et **Trend-Structure (fix signe v8.7)**."
+)
 
 
@@ -2112,6 +2413,10 @@
     st.header("2. Selection")
     select_all = st.checkbox(f"Tous les actifs ({len(ALL_SYMBOLS)})", value=True)
-    symbols_to_scan = ALL_SYMBOLS if select_all else st.multiselect(
-        "Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"]
+    symbols_to_scan = (
+        ALL_SYMBOLS
+        if select_all
+        else st.multiselect(
+            "Actifs :", options=ALL_SYMBOLS, default=["XAU_USD", "NAS100_USD", "US30_USD"]
+        )
     )
 
@@ -2160,19 +2465,23 @@
         try:
             raw_results = _run_async_isolated(
-                lambda: run_institutional_scan(symbols_to_scan, access_token, account_id, min_touches)
+                lambda: run_institutional_scan(
+                    symbols_to_scan, access_token, account_id, min_touches
+                )
             )
 
             results_h4, results_daily, results_weekly = [], [], []
             all_zones_map, prices_map, trends_map = {}, {}, {}
-            anomalies_map, scan_errors, bars_map_global = {}, {}, {}
+            anomalies_map, scan_errors, bars_map = {}, {}, {}
             missing_tfs_map, price_fallback_map, debug_map = {}, {}, {}
 
             for idx, res in enumerate(raw_results):
-                progress_bar.progress((idx + 1) / len(raw_results), text=f"Processing {res.symbol}...")
+                progress_bar.progress(
+                    (idx + 1) / len(raw_results), text=f"Processing {res.symbol}..."
+                )
                 if res.scan_error:
                     scan_errors[res.symbol.replace("_", "/")] = res.scan_error
                     continue
                 all_zones_map[res.symbol], prices_map[res.symbol] = res.zones, res.price
-                trends_map[res.symbol], bars_map_global[res.symbol] = res.trends, res.bars_map
+                trends_map[res.symbol], bars_map[res.symbol] = res.trends, res.bars_map
                 if res.anomaly:
                     anomalies_map[res.symbol.replace("_", "/")] = res.anomaly
@@ -2196,5 +2505,8 @@
                     continue
                 sym_thresh = {
-                    "US30_USD": 1.5, "NAS100_USD": 1.5, "SPX500_USD": 1.2, "XAU_USD": 1.5,
+                    "US30_USD": 1.5,
+                    "NAS100_USD": 1.5,
+                    "SPX500_USD": 1.2,
+                    "XAU_USD": 1.5,
                 }.get(sym, confluence_threshold)
                 all_confs.extend(
@@ -2203,5 +2515,5 @@
                         all_zones_map.get(sym, {}),
                         prices_map.get(sym),
-                        bars_map_global.get(sym, {}),
+                        bars_map.get(sym, {}),
                         sym_thresh,
                     )
@@ -2217,12 +2529,14 @@
                         cp, all_zones_map[sym]["Daily"][0], all_zones_map[sym]["Daily"][1]
                     )
-                summaries.append({
-                    "symbol": sym.replace("_", "/"),
-                    "trend_h4": trends_map.get(sym, {}).get("H4", "NEUTRE"),
-                    "trend_daily": trends_map.get(sym, {}).get("Daily", "NEUTRE"),
-                    "trend_weekly": trends_map.get(sym, {}).get("Weekly", "NEUTRE"),
-                    "price_context": ctx,
-                    "current_price": cp,
-                })
+                summaries.append(
+                    {
+                        "symbol": sym.replace("_", "/"),
+                        "trend_h4": trends_map.get(sym, {}).get("H4", "NEUTRE"),
+                        "trend_daily": trends_map.get(sym, {}).get("Daily", "NEUTRE"),
+                        "trend_weekly": trends_map.get(sym, {}).get("Weekly", "NEUTRE"),
+                        "price_context": ctx,
+                        "current_price": cp,
+                    }
+                )
 
             df_h4 = pd.DataFrame(results_h4)
@@ -2269,5 +2583,9 @@
         filtered_c = c_df[c_df["dist_num"] <= max_dist_filter].drop(columns=["dist_num"])
         st.dataframe(filtered_c.sort_values("Score", ascending=False), use_container_width=True)
-    for label, df in [("H4", res["df_h4"]), ("Daily", res["df_daily"]), ("Weekly", res["df_weekly"])]:
+    for label, df in [
+        ("H4", res["df_h4"]),
+        ("Daily", res["df_daily"]),
+        ("Weekly", res["df_weekly"]),
+    ]:
         st.subheader(f"Analyse {label}")
         if not df.empty:
@@ -2282,5 +2600,7 @@
     col1, col2, col3 = st.columns(3)
     with col1:
-        pdf_b = create_pdf_report(res["report_dict"], res["conf_full"], res["summaries"], res["anomalies"])
+        pdf_b = create_pdf_report(
+            res["report_dict"], res["conf_full"], res["summaries"], res["anomalies"]
+        )
         st.download_button("📄 PDF", data=pdf_b, file_name="rapport_bluestar.pdf")
     with col2:
