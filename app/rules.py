# app/rules.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ----------------------------
# Parámetros por defecto
# ----------------------------
DEFAULTS = {
    # Geo-velocidad imposible
    "geo_min_distance_km": 100.0,   # distancia mínima entre dos compras cercanas en tiempo
    "geo_max_minutes": 15.0,        # ventana de minutos para considerar "cercanas"

    # Retiros ATM en ráfaga y crecientes
    "atm_spree_window_min": 30.0,   # ventana de minutos para buscar ráfaga
    "atm_spree_min_count": 3,       # mínimo de retiros en ráfaga
    "atm_increasing_tolerance": 0.01,  # tolerancia para considerar "creciente"

    # Hora inusual (2–5 am)
    "odd_hours_start": 2,
    "odd_hours_end": 5,             # exclusivo (2 <= h < 5 ó <=5 según implementación)
    "odd_hours_min_burst": 3,       # 3 compras en una hora en esa franja
    "odd_hours_history_days": 60,   # historial para estimar si "nunca ocurre"

    # Nuevo merchant + monto alto + fuera de zona
    "new_merchant_days": 90,        # si no se ha visto en 90 días
    "high_z_threshold": 2.0,        # z-score alto (usa z de categoría/merchant si existe)
    "far_from_home_km": 50.0,       # fuera de zona habitual

    # Spike de canal
    "channel_spike_window_days": 1,   # compara últimas 24h vs últimos 30d
    "channel_baseline_days": 30,
    "channel_spike_ratio": 3.0,       # 24h_share >= ratio * 30d_share para disparar
    "channel_target": "online",       # evalúa spike para este canal (online por default)
}

# ----------------------------
# Utilidades
# ----------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.deg2rad(lat1).astype(float)
    lon1 = np.deg2rad(lon1).astype(float)
    lat2 = np.deg2rad(lat2).astype(float)
    lon2 = np.deg2rad(lon2).astype(float)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def _as_dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True)

def _get_params(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = dict(DEFAULTS)
    if payload and isinstance(payload.get("params"), dict):
        p.update(payload["params"])
    return p

def _safe_recent_df(payload: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    df = payload.get("recent_df")
    if isinstance(df, pd.DataFrame):
        # asegurar tipos
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = _as_dt(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
        return df
    return pd.DataFrame()

# ----------------------------
# Reglas individuales → (score, reason or None)
# ----------------------------
def rule_geo_velocity(feats_row: pd.Series, recent_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Dos compras muy juntas en tiempo con distancia grande → imposible físicamente.
    Busca entre los últimos 2–3 eventos para simplificar.
    """
    if recent_df.empty:
        return 0.0, None
    df = recent_df.dropna(subset=["lat", "lon", "timestamp"]).tail(3)
    if len(df) < 2:
        return 0.0, None
    # último y penúltimo
    p2 = df.iloc[-1]
    p1 = df.iloc[-2]
    dt_min = (p2["timestamp"] - p1["timestamp"]).total_seconds() / 60.0
    if dt_min < 0:
        return 0.0, None
    dist = _haversine_km(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
    if (dt_min <= float(params["geo_max_minutes"])) and (dist >= float(params["geo_min_distance_km"])):
        # score alto
        return 0.9, f"Geo-velocidad imposible: {dist:.0f} km en {dt_min:.0f} min"
    return 0.0, None

def rule_atm_spree(feats_row: pd.Series, recent_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Ráfaga de retiros ATM (>= N en W minutos) y crecientes.
    """
    if recent_df.empty:
        return 0.0, None
    W = float(params["atm_spree_window_min"])
    N = int(params["atm_spree_min_count"])
    tol = float(params["atm_increasing_tolerance"])

    end = recent_df["timestamp"].max()
    start = end - pd.Timedelta(minutes=W)
    g = recent_df[(recent_df["timestamp"] >= start) & (recent_df["type"] == "withdrawal")].copy()
    if len(g) < N:
        return 0.0, None

    g.sort_values("timestamp", inplace=True)
    amts = pd.to_numeric(g["amount"], errors="coerce").fillna(0.0).values
    increasing = np.all(np.diff(amts) >= -tol) and (amts[-1] > amts[0] + tol)
    if increasing:
        return 0.8, f"Retiros ATM en ráfaga: {len(g)} en {W:.0f} min, montos crecientes"
    return 0.0, None

def rule_odd_hours_burst(feats_row: pd.Series, recent_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Cliente casi nunca compra 2–5am; hoy hubo >=3 en 1h en esa franja.
    """
    if recent_df.empty:
        return 0.0, None
    h_start = int(params["odd_hours_start"])
    h_end = int(params["odd_hours_end"])
    burst_min = int(params["odd_hours_min_burst"])
    hist_days = int(params["odd_hours_history_days"])

    recent_df = recent_df.copy()
    recent_df["hour"] = _as_dt(recent_df["timestamp"]).dt.hour
    recent_df.set_index("timestamp", inplace=True)

    end = recent_df.index.max()
    start_hist = end - pd.Timedelta(days=hist_days)
    hist = recent_df[(recent_df.index >= start_hist)]
    hist_odd = hist[(hist["hour"] >= h_start) & (hist["hour"] < h_end)]
    # si "nunca" (≈ <=1 por mes) y ahora hubo ráfaga
    rarely = len(hist_odd) <= 1

    last_hour = end - pd.Timedelta(hours=1)
    now_odd = recent_df.loc[(recent_df.index >= last_hour) & (recent_df["hour"] >= h_start) & (recent_df["hour"] < h_end)]
    if rarely and (len(now_odd) >= burst_min):
        return 0.7, f"Hora inusual: {len(now_odd)} transacciones entre {h_start}:00–{h_end}:00"
    return 0.0, None

def rule_new_merchant_high_amount_outzone(feats_row: pd.Series, recent_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Nuevo merchant en 90d + monto alto (z-score) + lejos de home
    """
    mid = str(feats_row.get("merchant_id") or "")
    if not mid or recent_df.empty:
        return 0.0, None

    days = int(params["new_merchant_days"])
    cutoff = recent_df["timestamp"].max() - pd.Timedelta(days=days)
    seen = recent_df[(recent_df["timestamp"] >= cutoff) & (recent_df["merchant_id"].astype(str) == mid)]
    is_new = seen.empty

    # monto alto: usa el mayor z disponible (cat_z_30d/mer_z_30d/7d/90d)
    z_cols = [c for c in feats_row.index if c.startswith("cat_z_") or c.startswith("mer_z_")]
    z_val = 0.0
    if z_cols:
        z_val = float(np.nanmax([abs(feats_row[c]) if pd.notna(feats_row[c]) else 0.0 for c in z_cols]))
    high_z = z_val >= float(params["high_z_threshold"])

    far = float(feats_row.get("distance_from_home") or 0.0) >= float(params["far_from_home_km"])

    if is_new and high_z and far:
        return 0.8, "Nuevo merchant + monto alto + fuera de zona"
    return 0.0, None

def rule_channel_spike(feats_row: pd.Series, recent_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Spike en canal (por defecto: online). Compara share 24h vs 30d y exige multiplicador.
    """
    if recent_df.empty:
        return 0.0, None
    target = str(params["channel_target"])
    w_days = float(params["channel_spike_window_days"])
    b_days = float(params["channel_baseline_days"])
    ratio = float(params["channel_spike_ratio"])

    recent_df = recent_df.copy()
    recent_df["timestamp"] = _as_dt(recent_df["timestamp"])
    end = recent_df["timestamp"].max()
    w_start = end - pd.Timedelta(days=w_days)
    b_start = end - pd.Timedelta(days=b_days)

    w = recent_df[recent_df["timestamp"] >= w_start]
    b = recent_df[recent_df["timestamp"] >= b_start]
    if b.empty or w.empty:
        return 0.0, None

    def share(df: pd.DataFrame, channel: str) -> float:
        if "channel" not in df.columns:
            return 0.0
        total = len(df)
        if total == 0:
            return 0.0
        return float((df["channel"].astype(str) == channel).sum()) / float(total)

    w_share = share(w, target)
    b_share = share(b, target)
    if b_share == 0.0 and w_share >= 0.5:
        # antes casi nunca, ahora domina
        return 0.6, f"Spike de canal {target}: {w_share:.0%} en 24h (historial ≈ 0%)"
    if b_share > 0.0 and (w_share >= ratio * b_share) and w_share >= 0.5:
        return 0.6, f"Spike de canal {target}: {w_share:.0%} vs {b_share:.0%} histórico"
    return 0.0, None

# ----------------------------
# Evaluador principal
# ----------------------------
def evaluate(payload: Optional[Dict[str, Any]], feats_row: pd.Series) -> Tuple[float, List[str]]:
    """
    Retorna (rule_score in [0,1], reasons[])
    """
    params = _get_params(payload)
    recent_df = _safe_recent_df(payload)

    scores: List[float] = []
    reasons: List[str] = []

    # Ejecuta reglas (agrega aquí si sumas más)
    for fn in (
        rule_geo_velocity,
        rule_atm_spree,
        rule_odd_hours_burst,
        rule_new_merchant_high_amount_outzone,
        rule_channel_spike,
    ):
        try:
            s, r = fn(feats_row, recent_df, params)
        except Exception:
            s, r = 0.0, None
        scores.append(float(np.clip(s, 0.0, 1.0)))
        if r:
            reasons.append(r)

    # Combina reglas: tomamos el máximo y la media para un score balanceado
    if scores:
        rule_score = 0.5 * max(scores) + 0.5 * float(np.mean(scores))
    else:
        rule_score = 0.0

    return float(np.clip(rule_score, 0.0, 1.0)), reasons
