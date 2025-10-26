# app/main.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from . import storage
from . import ingest as ingest_mod
from .features import build_features
from .features import detect_subscriptions, detect_leaks  # insights
from .model import train_customer_model, train_global_model, predict_with_fallback
from . import rules, scoring
from .schemas import (
    ScoreRequest, ScoreResponse, ScoreResult,
    IngestResponse, TrainRequest, TrainResponse,
    InsightsResponse, SubscriptionInsight, LeakInsight,
    FeedbackRequest, FeedbackResponse
)

app = FastAPI(title="AURA Risk API", version="0.2.0")

# Inicializa BD al arrancar
storage.bootstrap()


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

# ---------------------------
# Helpers internos
# ---------------------------
def _to_dt(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce", utc=True)

def _build_recent_df(customer_id: str, days: int = 90) -> pd.DataFrame:
    """
    DataFrame con columnas mínimas para reglas:
    timestamp, type, amount, lat, lon, merchant_id, channel, account_id
    """
    rows = storage.read_transactions_by_customer(customer_id, since=None, limit=1_000_000)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "type", "amount", "lat", "lon", "merchant_id", "channel", "account_id"])

    df = pd.DataFrame(rows)
    df["timestamp"] = _to_dt(df["timestamp"])
    cut = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cut].copy()
    # Tipos y saneo
    for c in ("amount", "lat", "lon"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in ("type", "merchant_id", "channel", "account_id"):
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].astype(str)
    df.sort_values("timestamp", inplace=True)
    return df

def _pick_nearest_feat_row(df_feats: pd.DataFrame, account_id: str, ts) -> Optional[pd.Series]:
    """
    Dado el DF de features (índice: account_id, timestamp, row_idx),
    elige la fila del mismo account_id cuya timestamp sea más cercana a ts.
    """
    if df_feats is None or df_feats.empty:
        return None
    try:
        # Extrae sólo ese account
        sub = df_feats.xs(account_id, level=0, drop_level=False).reset_index()
    except Exception:
        # Si falla por estructura del índice, intenta resetear todo
        sub = df_feats.reset_index()
        sub = sub[sub["account_id"] == account_id]

    if sub.empty:
        return None

    sub["timestamp"] = _to_dt(sub["timestamp"])
    ts = _to_dt(ts)
    sub["__dt"] = (sub["timestamp"] - ts).abs()
    sub.sort_values("__dt", inplace=True)
    return sub.iloc[0].drop(labels="__dt", errors="ignore")

def _z_amount_from_row(row: pd.Series) -> float:
    """
    Toma el mayor |z| disponible entre columnas de z-scores.
    """
    if row is None or not isinstance(row, pd.Series):
        return 0.0
    z_cols = [c for c in row.index if ("_z_" in c) or c.startswith("cat_z_") or c.startswith("mer_z_")]
    if not z_cols:
        return 0.0
    vals = []
    for c in z_cols:
        try:
            v = float(row.get(c))
            if np.isfinite(v):
                vals.append(abs(v))
        except Exception:
            continue
    return float(max(vals) if vals else 0.0)

def _distance_km_from_row(row: pd.Series) -> float:
    try:
        v = float(row.get("distance_from_home"))
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


# ---------------------------
# Endpoints
# ---------------------------

@app.post(
    "/ingest/{customer_id}",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Descarga y normaliza datos desde Nessie para un cliente"
)
async def ingest_customer(customer_id: str) -> IngestResponse:
    """
    Jala cuentas, transacciones, bills y merchants desde Nessie y guarda en BD local.
    """
    metrics = await ingest_mod.ingest_customer(customer_id)
    return IngestResponse(
        customer_id=customer_id,
        accounts_found=metrics.get("accounts_found", 0),
        transactions_inserted=metrics.get("transactions_inserted", 0),
        bills_inserted=metrics.get("bills_inserted", 0),
        merchants_upserted=metrics.get("merchants_upserted", 0),
        message="Ingest complete"
    )


@app.post(
    "/train/{customer_id}",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Entrena el modelo de anomalías del cliente (y opcional el global)"
)
def train(customer_id: str, body: TrainRequest) -> TrainResponse:
    rep_c = train_customer_model(customer_id)
    # Global opcional: usa al menos este customer para el batch
    global_trained = False
    if body.retrain_global:
        try:
            rep_g = train_global_model([customer_id])
            global_trained = rep_g.n_samples > 0
        except Exception:
            global_trained = False

    return TrainResponse(
        customer_id=customer_id,
        customer_model_trained=(rep_c.n_samples > 0),
        global_model_trained=global_trained,
        n_samples_customer=rep_c.n_samples,
        params=rep_c.params,
        message=rep_c.message
    )


@app.get(
    "/insights/{customer_id}",
    response_model=InsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtiene insights: suscripciones y gastos hormiga"
)
def insights(customer_id: str) -> InsightsResponse:
    subs = detect_subscriptions(customer_id) or []
    leaks = detect_leaks(customer_id) or []

    # Normaliza a schemas
    subs_out = [
        SubscriptionInsight(
            merchant_id=s.get("merchant_id"),
            merchant_name=s.get("merchant_name"),
            periodicity_days=int(s.get("periodicity_days", 0)),
            avg_amount=float(s.get("avg_amount", 0.0)),
            status=s.get("status", "activa_ok"),
            recommendation=s.get("recommendation")
        )
        for s in subs
    ]
    leaks_out = [
        LeakInsight(
            bucket=str(l.get("bucket")),
            monthly=float(l.get("monthly", 0.0)),
            annual=float(l.get("annual", 0.0)),
            save30=float(l.get("save30", 0.0))
        )
        for l in leaks
    ]
    return InsightsResponse(customer_id=customer_id, subscriptions=subs_out, leaks=leaks_out)


@app.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Scoring de riesgo para 1..N transacciones (fusión ML + reglas)"
)
def score(body: ScoreRequest) -> ScoreResponse:
    if not body.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    # Asumimos todas las TX del mismo customer (si vienen varios, procesamos por cada una)
    # Para el contexto de reglas, construimos recent_df del primer customer.
    first_customer = body.transactions[0].customer_id
    recent_df = _build_recent_df(first_customer, days=90)
    payload_ctx = {"recent_df": recent_df}

    # Precomputar features del customer (para empatar filas cercanas)
    df_feats = build_features(first_customer)

    results: List[ScoreResult] = []
    for idx, tx in enumerate(body.transactions):
        # Empata fila de features más cercana para ese account_id y timestamp
        row = _pick_nearest_feat_row(df_feats, tx.account_id, tx.timestamp)

        if row is None:
            # Sin features: degrada con scores bajos
            ml_score = 0.0
            rule_score, reasons = 0.0, []
            z_amt = 0.0
            dist_km = 0.0
        else:
            # ML con fallback (incluye top z-features si necesitas)
            pred = predict_with_fallback(tx.customer_id, row, rules_payload=None)
            ml_score = float(pred["ml_score"])

            # Reglas con contexto reciente
            rule_score, reasons = rules.evaluate(payload_ctx, row)

            # Debugs
            z_amt = _z_amount_from_row(row)
            dist_km = _distance_km_from_row(row)

        # Fusión y color
        fused = scoring.finalize_response(ml_score, rule_score, reasons)

        # Ensambla resultado por transacción con bloque debug
        results.append(ScoreResult(
            transaction_idx=idx,
            risk_score=fused["risk_score"],
            color=fused["color"],
            reasons=fused["reasons"],
            ml_score=fused["ml_score"],
            rule_score=fused["rule_score"],
            debug={"z_amount": round(z_amt, 2), "distance_km": round(dist_km, 1)}
        ))

    return ScoreResponse(results=results)


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Guarda feedback del usuario/jurado para recalibrar"
)
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    try:
        storage.insert_feedback(body.customer_id, label=body.label, txn_id=body.txn_id)
        return FeedbackResponse(customer_id=body.customer_id, stored=True, message="Feedback stored")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not store feedback: {e}")
