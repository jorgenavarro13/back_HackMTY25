# app/features.py
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import storage

# ----------------------------
# Config de features
# ----------------------------
TOP_K_CATEGORIES = 12          # one-hot para las K categorías más comunes
TRAVEL_DISTANCE_KM = 200.0     # umbral para considerar viaje (distancia a home)
TRAVEL_MIN_CONSEC_DAYS = 3     # días consecutivos lejos de home para marcar flag

# ----------------------------
# Utilidades
# ----------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Distancia Haversine en km. Acepta escalares o arrays (vectorizable).
    """
    R = 6371.0088  # radio medio Tierra en km
    lat1 = np.deg2rad(lat1).astype(float)
    lon1 = np.deg2rad(lon1).astype(float)
    lat2 = np.deg2rad(lat2).astype(float)
    lon2 = np.deg2rad(lon2).astype(float)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c


def _safe_log1p(x: pd.Series) -> pd.Series:
    # Evita log de negativos si llegaran reembolsos; recorta a 0 para log1p
    return np.log1p(np.clip(x.fillna(0.0), a_min=0.0, a_max=None))


def _extract_balance_from_raw(raw_json):
    """
    Intenta extraer el balance (si existiera) desde raw_json de la transacción.
    Retorna None si no se encuentra.
    """
    if not isinstance(raw_json, dict):
        return None
    for key in ("balance", "newBalance", "post_balance", "running_balance"):
        if key in raw_json:
            try:
                return float(raw_json[key])
            except Exception:
                pass
    # A veces viene anidado
    acct = raw_json.get("account") if isinstance(raw_json.get("account"), dict) else None
    if acct and "balance" in acct:
        try:
            return float(acct["balance"])
        except Exception:
            return None
    return None


def _compute_home_centroid(df_tx: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Centroide "home" del cliente: mediana de lat/lon de todas las transacciones con lat/lon válido.
    """
    loc = df_tx[["lat", "lon"]].dropna()
    if loc.empty:
        return None, None
    return float(loc["lat"].median()), float(loc["lon"].median())


def _travel_flag_by_day(df_tx: pd.DataFrame, home_lat: float, home_lon: float) -> pd.Series:
    """
    Marca días en los que el centroide diario se aleja de home > TRAVEL_DISTANCE_KM
    y agrupa en rachas; si hay >= TRAVEL_MIN_CONSEC_DAYS consecutivos, flag=1 para esos días.
    """
    if home_lat is None or home_lon is None:
        return pd.Series(0, index=df_tx.index)

    # Centroide diario
    day_centroid = (df_tx
                    .dropna(subset=["lat", "lon"])
                    .groupby(df_tx["timestamp"].dt.date)[["lat", "lon"]]
                    .mean())
    if day_centroid.empty:
        return pd.Series(0, index=df_tx.index)

    day_centroid["dist_home_km"] = _haversine_km(day_centroid["lat"], day_centroid["lon"], home_lat, home_lon)
    day_centroid["far"] = (day_centroid["dist_home_km"] > TRAVEL_DISTANCE_KM).astype(int)

    # Detecta rachas de días "far" consecutivos
    # Etiquetado de rachas: cada vez que far==0, aumenta el grupo
    grp = (day_centroid["far"] == 0).cumsum()
    day_centroid["run_len"] = day_centroid.groupby(grp)["far"].transform(lambda s: s.cumsum())

    # Un día pertenece a "viaje" si la racha far de su segmento es >= N
    day_centroid["travel_day"] = ((day_centroid["far"] == 1) & (day_centroid["run_len"] >= TRAVEL_MIN_CONSEC_DAYS)).astype(int)

    # Mapea a cada transacción por su fecha
    travel_map = day_centroid["travel_day"].to_dict()
    flags = df_tx["timestamp"].dt.date.map(travel_map).fillna(0).astype(int)
    return flags


def _zscore(x: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    return (x - mean) / std.replace({0.0: np.nan})


# ----------------------------
# Construcción de features
# ----------------------------
def build_features(customer_id: str) -> pd.DataFrame:
    """
    Lee transacciones y merchants del storage, calcula:
    - Básicos por transacción
    - Ventanas temporales (1h/24h/7d) por cuenta
    - Agregados rolling 7d/30d/90d por categoría y merchant
    - Distancia a centroide home
    - One-hot de canal y top-K categorías + target-encoding por merchant
    - Flags de viaje

    Retorna DataFrame indexado por (account_id, timestamp, idx) con columnas de features.
    """
    # ---- Carga de datos ----
    tx_list = storage.read_transactions_by_customer(customer_id=customer_id, since=None, limit=1_000_000)
    merchants = storage.read_merchants(limit=1_000_000)

    if not tx_list:
        # DataFrame vacío con columnas esperadas
        return pd.DataFrame()

    df = pd.DataFrame(tx_list)

    # Asegura tipos
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.sort_values(["account_id", "timestamp"], inplace=True, kind="mergesort")

    # Merchant info (categoría)
    df_mer = pd.DataFrame(merchants) if merchants else pd.DataFrame(columns=["merchant_id", "name", "category", "lat", "lon", "raw_json"])
    df = df.merge(
        df_mer[["merchant_id", "category"]].rename(columns={"category": "merchant_category"}),
        how="left",
        on="merchant_id"
    )

    # ---- Features básicos por transacción ----
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["log_amount"] = _safe_log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour.astype("Int64")
    df["day_of_week"] = df["timestamp"].dt.weekday.astype("Int64")
    df["channel"] = df["channel"].fillna("unknown").astype(str)

    # pct_of_balance si existe
    df["pct_of_balance"] = df["raw_json"].apply(_extract_balance_from_raw)
    df["pct_of_balance"] = np.where(df["pct_of_balance"].notna() & (df["pct_of_balance"] != 0),
                                    df["amount"] / df["pct_of_balance"],
                                    np.nan)

    # delta_t (segundos) desde la transacción previa por cuenta
    df["delta_t"] = (df.groupby("account_id")["timestamp"].diff().dt.total_seconds()).fillna(np.nan)

    # ---- Conteos por ventana temporal (1h / 24h / 7d) por cuenta ----
    # Usamos índice de tiempo para rolling time-based
    df.set_index("timestamp", inplace=True)
    # Para cada cuenta
    def rolling_count(g: pd.DataFrame, window: str) -> pd.Series:
        return g["amount"].rolling(window, closed="both").count()

    df["count_1h"] = df.groupby("account_id", group_keys=False).apply(lambda g: rolling_count(g, "1h")).fillna(0).astype(int)
    df["count_24h"] = df.groupby("account_id", group_keys=False).apply(lambda g: rolling_count(g, "24h")).fillna(0).astype(int)
    df["count_7d"] = df.groupby("account_id", group_keys=False).apply(lambda g: rolling_count(g, "7d")).fillna(0).astype(int)

    # ---- Distancia a centroide "home" ----
    # Recupera lat/lon (ya están en df)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Centroide home global del cliente
    # (requerimos lat/lon válidos para estimarlo)
    df_reset = df.reset_index()
    home_lat, home_lon = _compute_home_centroid(df_reset)

    if home_lat is not None and home_lon is not None:
        df["distance_from_home"] = _haversine_km(df["lat"], df["lon"], home_lat, home_lon)
    else:
        df["distance_from_home"] = np.nan

    # ---- One-hot channel ----
    channel_dummies = pd.get_dummies(df["channel"], prefix="ch", dtype=int)
    # Nos interesa al menos estas columnas
    for ch in ["ch_atm", "ch_branch", "ch_online", "ch_present", "ch_unknown"]:
        if ch not in channel_dummies.columns:
            channel_dummies[ch] = 0
    df = pd.concat([df, channel_dummies], axis=1)

    # ---- One-hot top-K categorías + "other" ----
    cat_counts = df["merchant_category"].fillna("unknown").astype(str).value_counts()
    top_cats = set(cat_counts.head(TOP_K_CATEGORIES).index.tolist())
    df["cat_top"] = df["merchant_category"].fillna("unknown").astype(str).apply(lambda c: c if c in top_cats else "other")
    cat_dummies = pd.get_dummies(df["cat_top"], prefix="cat", dtype=int)
    df = pd.concat([df, cat_dummies], axis=1)

    # ---- Target-encoding (mean amount) por merchant a 90d ----
    # Usamos rolling por merchant-id (time-based 90d)
    def target_encode_mean_amount_90d(g: pd.DataFrame) -> pd.Series:
        return g["amount"].rolling("90d", closed="both").mean()

    df["te_merchant_amount_90d"] = (df.groupby("merchant_id", group_keys=False)
                                      .apply(target_encode_mean_amount_90d)
                                      .fillna(df["amount"].mean()))

    # ---- Agregados por categoría y merchant (7d/30d/90d) y z-scores ----
    def rolling_stats_amount(g: pd.DataFrame, window: str) -> pd.DataFrame:
        m = g["amount"].rolling(window, closed="both")
        out = pd.DataFrame({
            f"mean_{window}": m.mean(),
            f"std_{window}": m.std(ddof=0)  # población
        }, index=g.index)
        return out

    # Por categoría
    cat_grp = df.groupby("cat_top", group_keys=False)
    cat_7 = cat_grp.apply(lambda g: rolling_stats_amount(g, "7d"))
    cat_30 = cat_grp.apply(lambda g: rolling_stats_amount(g, "30d"))
    cat_90 = cat_grp.apply(lambda g: rolling_stats_amount(g, "90d"))

    for w, frame in [("7d", cat_7), ("30d", cat_30), ("90d", cat_90)]:
        df[f"cat_mean_{w}"] = frame[f"mean_{w}"]
        df[f"cat_std_{w}"] = frame[f"std_{w}"]
        df[f"cat_z_{w}"] = _zscore(df["amount"], df[f"cat_mean_{w}"], df[f"cat_std_{w}"])

    # Por merchant
    mer_grp = df.groupby("merchant_id", group_keys=False)
    mer_7 = mer_grp.apply(lambda g: rolling_stats_amount(g, "7d"))
    mer_30 = mer_grp.apply(lambda g: rolling_stats_amount(g, "30d"))
    mer_90 = mer_grp.apply(lambda g: rolling_stats_amount(g, "90d"))

    for w, frame in [("7d", mer_7), ("30d", mer_30), ("90d", mer_90)]:
        df[f"mer_mean_{w}"] = frame[f"mean_{w}"]
        df[f"mer_std_{w}"] = frame[f"std_{w}"]
        df[f"mer_z_{w}"] = _zscore(df["amount"], df[f"mer_mean_{w}"], df[f"mer_std_{w}"])

    # ---- Flag de viaje (reduce sensibilidad de fraude cuando está en viaje) ----
    travel_flag = _travel_flag_by_day(df_reset, home_lat, home_lon) if (home_lat is not None) else pd.Series(0, index=df_reset.index)
    # Vuelve a index de df (timestamp) conservando orden original
    df["travel_flag"] = travel_flag.values

    # ---- Limpieza final y orden ----
    # Reponemos timestamp como columna (ya que lo usamos como índice)
    df.reset_index(inplace=True)  # 'timestamp' vuelve a ser columna
    # Índice compuesto útil (opcional)
    df["row_idx"] = np.arange(len(df))
    df.set_index(["account_id", "timestamp", "row_idx"], inplace=True)

    # Selección de columnas de salida (puedes agregar/retirar según tu modelo)
    base_cols = [
        "customer_id", "type", "merchant_id", "merchant_category",
        "amount", "log_amount", "hour_of_day", "day_of_week",
        "pct_of_balance", "delta_t",
        "count_1h", "count_24h", "count_7d",
        "distance_from_home",
        "te_merchant_amount_90d",
        "cat_mean_7d", "cat_std_7d", "cat_z_7d",
        "cat_mean_30d", "cat_std_30d", "cat_z_30d",
        "cat_mean_90d", "cat_std_90d", "cat_z_90d",
        "mer_mean_7d", "mer_std_7d", "mer_z_7d",
        "mer_mean_30d", "mer_std_30d", "mer_z_30d",
        "mer_mean_90d", "mer_std_90d", "mer_z_90d",
        "travel_flag"
    ]

    # Agrega one-hots de canal y categorías
    ohe_cols = [c for c in df.columns if c.startswith("ch_") or c.startswith("cat_")]
    # Evita duplicar columnas base
    ohe_cols = [c for c in ohe_cols if c not in base_cols]

    out_cols = base_cols + ohe_cols

    # Asegura que existan todas (si alguna ventana quedó vacía)
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ordena columnas
    df = df[out_cols]

    # Tipos numéricos donde aplica
    numeric_cols = [c for c in df.columns if c not in ("customer_id", "type", "merchant_id", "merchant_category")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df


# --- Pega esto dentro de app/features.py (por ejemplo, debajo de build_features) ---
from datetime import timedelta
import re

REC_WINDOWS = {
    7:  (6, 8),          # 7 ±1
    30: (27, 33),        # 30 ±3
    365:(358, 372),      # 365 ±7
}
AMOUNT_TOL = 0.15       # ±15%
REC_MIN_HITS = 3        # al menos 3 ocurrencias dentro de la ventana para considerarlo suscripción
REC_LOOKBACK_DAYS = 540 # cuánto histórico mirar para detectar
NO_USE_WINDOW_DAYS = 90 # ventana para decidir "sin uso"
DUPLICATE_SIM_PERIOD_DAYS = 5  # diferencia máxima de periodicidad para considerar duplicadas

_SUBS_CATEGORY_HINTS = [
    "stream", "video", "tv", "music", "cloud", "storage", "drive",
    "news", "newspaper", "press", "magazine", "fitness", "gym",
    "phone", "mobile", "cell", "carrier", "internet", "isp",
    "hosting", "domain", "dns", "vpn", "game", "gaming", "plus", "premium"
]

def _norm_service_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _service_equivalence_key(name: str, category: str | None) -> str:
    """
    Crea una clave 'equivalente' para detectar servicios duplicados:
    - usa categoría si existe
    - si no, usa tokens del nombre con hints de suscripciones
    """
    cat = (category or "").lower().strip()
    base = _norm_service_name(name)
    tokens = base.split()
    hits = [t for t in tokens if any(h in t for h in _SUBS_CATEGORY_HINTS)]
    if cat:
        return f"cat::{cat}"
    if hits:
        return "hint::" + "|".join(sorted(set(hits)))
    # fallback: primer token del nombre (muy laxo, pero útil en hackathon)
    return "name::" + (tokens[0] if tokens else "")

def _dominant_period(days_diffs: list[int]) -> tuple[int | None, int]:
    """
    Dadas diferencias de días entre cargos, devuelve (periodicity_days, hits_en_ventana)
    escogiendo la ventana que más hits tenga.
    """
    best_period = None
    best_hits = 0
    for p, (lo, hi) in REC_WINDOWS.items():
        hits = sum(1 for d in days_diffs if lo <= d <= hi)
        if hits > best_hits:
            best_hits = hits
            best_period = p
    return best_period, best_hits

def _is_amount_similar(amt: float, ref: float, tol: float = AMOUNT_TOL) -> bool:
    if ref == 0:
        return False
    return abs(amt - ref) <= tol * ref

def detect_subscriptions(customer_id: str) -> list[dict]:
    """
    Detecta suscripciones recurrentes por merchant_id.
    Reglas:
      - Al menos 3 cargos con diferencias entre fechas en una de las ventanas definidas
      - Montos dentro de ±15% de la media de la serie recurrente
      - Clasificación:
          * activa_ok: recurrente y hay actividad variable con el merchant (p.ej., uso) en 90 días
          * activa_sin_uso: recurrente pero sin otra actividad con ese merchant en 90 días
          * duplicada: dos suscripciones en la misma 'equivalence key' y periodicidad similar
    Retorna lista de dicts con: merchant_id, merchant_name, periodicity_days, avg_amount, status, recommendation
    """
    # Carga datos
    tx = storage.read_transactions_by_customer(customer_id=customer_id, since=None, limit=1_000_000)
    if not tx:
        return []

    df = pd.DataFrame(tx)
    # Asegura timestamp y orden
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.sort_values(["merchant_id", "timestamp"], inplace=True)
    # Considera solo compras/cargos (puedes incluir transfers/bills si tu fuente lo requiere)
    df = df[df["type"].isin(["purchase", "withdrawal", "transfer", "deposit"]) | df["type"].isna()]

    # Merchants (para nombres/categorías)
    mers = storage.read_merchants(limit=1_000_000)
    df_mer = pd.DataFrame(mers) if mers else pd.DataFrame(columns=["merchant_id", "name", "category"])
    df = df.merge(
        df_mer[["merchant_id", "name", "category"]].rename(columns={"name": "merchant_name", "category": "merchant_category"}),
        on="merchant_id", how="left"
    )

    # Recorta horizonte
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=REC_LOOKBACK_DAYS)
    df = df[df["timestamp"] >= cutoff]

    subs_candidates: list[dict] = []

    # Evalúa cada merchant_id
    for mid, g in df.groupby("merchant_id", dropna=True):
        if g.empty or pd.isna(mid):
            continue

        g = g.sort_values("timestamp")
        # Solo considera montos positivos (cargos)
        g["amount"] = pd.to_numeric(g["amount"], errors="coerce")
        g = g[g["amount"] > 0]

        if len(g) < REC_MIN_HITS:
            continue

        # Diffs entre fechas consecutivas en días
        diffs = g["timestamp"].diff().dt.total_seconds().dropna() / (24 * 3600)
        diffs = diffs.round().astype(int).tolist()
        period, hits = _dominant_period(diffs)
        if period is None or hits < (REC_MIN_HITS - 1):  # p.ej., con 3 cargos hay 2 diffs
            continue

        # Filtra la serie recurrente por monto similar a la mediana
        med = float(g["amount"].median())
        g["is_core_amt"] = g["amount"].apply(lambda a: _is_amount_similar(a, med))
        core = g[g["is_core_amt"]]
        if len(core) < REC_MIN_HITS:
            continue

        avg_amount = float(core["amount"].mean())
        mname = (core["merchant_name"].dropna().iloc[-1] if "merchant_name" in core and not core["merchant_name"].dropna().empty else None)
        mcat = (core["merchant_category"].dropna().iloc[-1] if "merchant_category" in core and not core["merchant_category"].dropna().empty else None)

        # "Uso" en 90 días: actividad con el merchant que NO sea estrictamente igual a la serie (monto fuera de ±15%) o transacción fuera de la cadencia
        lookback = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=NO_USE_WINDOW_DAYS)
        recent = g[g["timestamp"] >= lookback]
        recent_var_amt = recent[~recent["amount"].apply(lambda a: _is_amount_similar(a, avg_amount))]
        has_use = len(recent_var_amt) > 0

        subs_candidates.append({
            "merchant_id": str(mid),
            "merchant_name": mname,
            "merchant_category": mcat,
            "periodicity_days": int(period),
            "avg_amount": round(avg_amount, 2),
            "has_use": bool(has_use)
        })

    if not subs_candidates:
        return []

    # Duplicadas: misma clave de equivalencia + periodicidad similar
    # Construye claves
    for c in subs_candidates:
        c["equiv_key"] = _service_equivalence_key(c.get("merchant_name") or "", c.get("merchant_category"))

    # Marca duplicadas por grupo
    results: list[dict] = []
    by_key: dict[str, list[dict]] = {}
    for c in subs_candidates:
        by_key.setdefault(c["equiv_key"], []).append(c)

    # Cruce con bills (si hay)
    bills = storage.read_bills_by_customer(customer_id)
    df_bills = pd.DataFrame(bills) if bills else pd.DataFrame(columns=["bill_id","merchant_id","status","amount","due_date","raw_json"])
    # para recomendación "pausar/cancelar"
    has_bills_for = set(df_bills["merchant_id"].astype(str).dropna().tolist()) if not df_bills.empty else set()

    for key, group in by_key.items():
        # Ordena por monto promedio descendente (útil para sugerir cuál mantener)
        group = sorted(group, key=lambda x: x["avg_amount"], reverse=True)

        # Detecta periodicidades muy cercanas
        # Si hay >=2 con |p_i - p_j| <= DUPLICATE_SIM_PERIOD_DAYS -> marcamos duplicadas (excepto la más cara por default)
        if len(group) >= 2:
            base = group[0]
            dups = [g for g in group[1:] if abs(g["periodicity_days"] - base["periodicity_days"]) <= DUPLICATE_SIM_PERIOD_DAYS]
        else:
            dups = []

        for i, sub in enumerate(group):
            is_dup = any(abs(sub["periodicity_days"] - s["periodicity_days"]) <= DUPLICATE_SIM_PERIOD_DAYS for s in group if s is not sub)
            if is_dup and i > 0:
                status = "duplicada"
                rec = "Parece duplicada de otro servicio similar; considera cancelar la de menor valor."
            else:
                status = "activa_ok" if sub["has_use"] else "activa_sin_uso"
                if status == "activa_sin_uso":
                    rec = "No se observa uso en 90 días; considera pausar/cancelar temporalmente."
                else:
                    rec = "Sin acción sugerida."

            # Si hay bill asociado, enfatiza acción
            if sub["merchant_id"] in has_bills_for:
                if status == "activa_sin_uso":
                    rec = rec.replace("considera pausar/cancelar", "puedes pausar/cancelar desde Bills")
                elif status == "duplicada":
                    rec = rec + " Puedes cancelar una desde Bills."

            results.append({
                "merchant_id": sub["merchant_id"],
                "merchant_name": sub.get("merchant_name"),
                "periodicity_days": int(sub["periodicity_days"]),
                "avg_amount": float(sub["avg_amount"]),
                "status": status,  # activa_ok | activa_sin_uso | duplicada
                "recommendation": rec
            })

    return results

# --- Pega esto en app/features.py (por ejemplo, debajo de detect_subscriptions) ---
import re
from typing import Iterable

# Parámetros (puedes ajustarlos desde tu endpoint si lo deseas)
SMALL_TICKET_MAX = 12.0          # ticket < $12
WEEKLY_FREQ_MIN = 3              # >3 compras por semana en promedio (últimas 8 semanas)
MONTH_SHARE_MIN = 0.05           # >5% del gasto total del último mes
LOOKBACK_DAYS = 120              # horizonte para medir patrones
WEEK_WINDOW = 8                  # semanas para promediar frecuencia
MONTH_WINDOW = 30                # días para gasto mensual

# Palabras clave para clasificar por bucket
_BUCKET_KEYWORDS = {
    "cafés": [
        r"\bcoffee\b", r"\bcafé\b", r"\bcafe\b", r"\bstarbucks\b", r"\bcafeter(ia|ía)\b",
        r"\bespresso\b", r"\blatte\b", r"\bcapuccino\b"
    ],
    "snacks": [
        r"\bsnack\b", r"\bdonut\b", r"\bkrispy\b", r"\bcookie\b", r"\bpanader(ia|ía)\b",
        r"\bchips\b", r"\bchocolate\b", r"\bhelado\b", r"\bice cream\b", r"\bpostre\b"
    ],
    "fast_food": [
        r"\bfast\s*food\b", r"\bburger\b", r"\bmc ?donald", r"\bkfc\b", r"\btaco\b",
        r"\bpizza\b", r"\bsubway\b", r"\bbk\b", r"\bburger king\b", r"\bwendy'?s\b"
    ],
    "delivery": [
        r"\bubereats\b", r"\buber eats\b", r"\bdidi food\b", r"\bdidifood\b",
        r"\brappi\b", r"\bpostmates\b", r"\bdoor ?dash\b", r"\bcornershop\b", r"\benv(í|i)o\b",
        r"\bdelivery\b", r"\bentrega\b"
    ],
}

def _norm_text(x: str | None) -> str:
    if not x:
        return ""
    s = x.lower()
    s = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _bucket_from_text(name: str, category: str | None) -> str | None:
    base = _norm_text(name)
    cat = _norm_text(category or "")
    text = f"{base} {cat}".strip()
    if not text:
        return None
    for bucket, patterns in _BUCKET_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text):
                return bucket
    # fallback: si la categoría es muy genérica de comida, envíalo a fast_food/snacks por heurística suave
    if "food" in text or "comida" in text or "restaurant" in text or "restaurante" in text:
        return "fast_food"
    return None

def _safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=True)

def detect_leaks(
    customer_id: str,
    small_ticket_max: float = SMALL_TICKET_MAX,
    weekly_freq_min: int = WEEKLY_FREQ_MIN,
    month_share_min: float = MONTH_SHARE_MIN,
    lookback_days: int = LOOKBACK_DAYS,
    week_window: int = WEEK_WINDOW,
    month_window: int = MONTH_WINDOW
) -> list[dict]:
    """
    Detecta 'gastos hormiga' por buckets discrecionales:
      - ticket < small_ticket_max
      - clasifica por texto merchant/category en {cafés, snacks, fast_food, delivery}
      - frecuencia semanal promedio (últimas `week_window` semanas) > weekly_freq_min
      - gasto mensual del bucket / gasto mensual total > month_share_min

    Devuelve top-3 leaks:
      [{bucket, monthly, annual, save30}]
    """
    # 1) Carga transacciones y merchants
    tx = storage.read_transactions_by_customer(customer_id=customer_id, since=None, limit=1_000_000)
    if not tx:
        return []

    df = pd.DataFrame(tx)
    df["timestamp"] = _safe_to_datetime(df["timestamp"])
    df = df[df["timestamp"].notna()]
    df.sort_values("timestamp", inplace=True)

    # Solo cargos positivos y de compras (puedes ajustar tipos si deseas)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df[(df["amount"] > 0) & (df["type"].isin(["purchase", "withdrawal", "transfer", "deposit"]) | df["type"].isna())]

    # Lookback
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return []

    # Merchants para nombre/categoría
    mers = storage.read_merchants(limit=1_000_000)
    df_mer = pd.DataFrame(mers) if mers else pd.DataFrame(columns=["merchant_id", "name", "category"])
    df = df.merge(
        df_mer[["merchant_id", "name", "category"]].rename(columns={"name": "merchant_name", "category": "merchant_category"}),
        on="merchant_id", how="left"
    )

    # 2) Filtra small-tickets y asigna bucket
    df_small = df[df["amount"] < small_ticket_max].copy()
    if df_small.empty:
        return []

    df_small["bucket"] = [
        _bucket_from_text(row.get("merchant_name", ""), row.get("merchant_category"))
        for _, row in df_small.iterrows()
    ]
    df_small = df_small[df_small["bucket"].notna()]
    if df_small.empty:
        return []

    # 3) Frecuencia semanal promedio por bucket (últimas `week_window` semanas)
    #    y gasto mensual del bucket vs total mensual
    end = df["timestamp"].max()
    start_week = end - pd.Timedelta(weeks=week_window)
    start_month = end - pd.Timedelta(days=month_window)

    # Dataframes recortados a cada ventana
    df_week = df_small[df_small["timestamp"] >= start_week].copy()
    df_month = df[df["timestamp"] >= start_month].copy()
    if df_week.empty or df_month.empty:
        return []

    # Frecuencia semanal por bucket
    # Re-sample semanal y promedia el conteo
    freq_week = (
        df_week.set_index("timestamp")
               .groupby("bucket")["amount"]
               .resample("W-MON")  # semana que inicia lunes; consistente
               .count()
               .groupby("bucket")
               .mean()
               .rename("weekly_avg_count")
               .to_frame()
    )

    # Gasto mensual por bucket (small-tickets) y total mensual
    month_bucket = (
        df_small[df_small["timestamp"] >= start_month]
        .groupby("bucket")["amount"].sum().rename("monthly")
        .to_frame()
    )
    month_total = float(df_month["amount"].sum()) or 1.0  # evita div/0
    month_bucket["share"] = (month_bucket["monthly"] / month_total).clip(lower=0.0)

    # Une métricas
    metrics = freq_week.join(month_bucket, how="inner").fillna(0.0)

    # 4) Filtra por umbrales
    keep = metrics[
        (metrics["weekly_avg_count"] > float(weekly_freq_min)) &
        (metrics["share"] >= float(month_share_min))
    ].copy()

    if keep.empty:
        return []

    # 5) Construye salida y ordena por impacto mensual
    keep["annual"] = keep["monthly"] * 12.0
    keep["save30"] = keep["monthly"] * 0.30

    keep = keep.sort_values("monthly", ascending=False)

    # Normaliza a tipos primitivos y redondea
    def _row_to_item(idx, r):
        return {
            "bucket": str(idx),
            "monthly": round(float(r["monthly"]), 2),
            "annual": round(float(r["annual"]), 2),
            "save30": round(float(r["save30"]), 2)
        }

    items = [_row_to_item(idx, r) for idx, r in keep.head(3).iterrows()]
    return items
