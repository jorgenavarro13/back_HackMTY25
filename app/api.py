# app/api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, status
from typing import List

from .schemas import (
    IngestResponse,
    TrainRequest, TrainResponse,
    InsightsResponse, SubscriptionInsight, LeakInsight,
    ScoreRequest, ScoreResponse, ScoreResult,
    FeedbackRequest, FeedbackResponse
)

router = APIRouter(prefix="", tags=["risk"])

# Nota:
# - Aquí solo conectamos firmas y respuestas.
# - Dentro de cada endpoint, verás comentarios "TODO" donde debes llamar a tus módulos reales:
#   ingest.py, features.py, model.py, rules.py, scoring.py, storage.py

@router.post(
    "/ingest/{customer_id}",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Descarga y normaliza datos desde Nessie para un cliente"
)
async def ingest_customer(customer_id: str) -> IngestResponse:
    """
    Jala cuentas, transacciones, bills y merchants desde Nessie
    y los guarda en tu base local (DuckDB/SQLite).
    """
    # TODO: Llamar a app.ingest.run_ingest(customer_id) y devolver sus métricas.
    # Ejemplo simulado para que el router funcione de inmediato:
    accounts_found = 2
    transactions_inserted = 420
    bills_inserted = 6
    merchants_upserted = 95

    return IngestResponse(
        customer_id=customer_id,
        accounts_found=accounts_found,
        transactions_inserted=transactions_inserted,
        bills_inserted=bills_inserted,
        merchants_upserted=merchants_upserted,
        message="Ingest complete"
    )


@router.post(
    "/train/{customer_id}",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Entrena el modelo de anomalías del cliente (y opcionalmente el global)"
)
async def train_models(customer_id: str, body: TrainRequest) -> TrainResponse:
    """
    Genera features, entrena Isolation Forest por cliente y (opcional) modelo global.
    """
    # TODO:
    # - features.build_features(customer_id)
    # - model.train_customer_model(customer_id)
    # - if body.retrain_global: model.train_global_model()
    customer_model_trained = True
    global_model_trained = bool(body.retrain_global)
    n_samples_customer = 350
    params = {"algo": "IsolationForest", "contamination": 0.03, "n_estimators": 200}

    return TrainResponse(
        customer_id=customer_id,
        customer_model_trained=customer_model_trained,
        global_model_trained=global_model_trained,
        n_samples_customer=n_samples_customer,
        params=params,
        message="Training complete"
    )


@router.get(
    "/insights/{customer_id}",
    response_model=InsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtiene insights: suscripciones y gastos hormiga"
)
async def get_insights(customer_id: str) -> InsightsResponse:
    """
    Detecta suscripciones (recurrencias) y fugas de gasto (gastos hormiga).
    """
    # TODO:
    # - subs = features.detect_subscriptions(customer_id)
    # - leaks = features.detect_leaks(customer_id)
    # Simulación:
    subs = [
        SubscriptionInsight(
            merchant_id="m_stream_1",
            merchant_name="StreamFlix",
            periodicity_days=30,
            avg_amount=9.99,
            status="activa_sin_uso",
            recommendation="Considera pausar 2 meses"
        )
    ]
    leaks = [
        LeakInsight(bucket="cafés", monthly=38.2, annual=458.4, save30=11.46),
        LeakInsight(bucket="delivery", monthly=52.0, annual=624.0, save30=15.6),
    ]
    return InsightsResponse(customer_id=customer_id, subscriptions=subs, leaks=leaks)


@router.post(
    "/score",
    response_model=ScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Scoring de riesgo para 1..N transacciones"
)
async def score_transactions(body: ScoreRequest) -> ScoreResponse:
    """
    Combina ML (anomalías) + reglas para etiquetar cada transacción con color y explicación.
    """
    if not body.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    # TODO:
    # for each tx:
    #   feats = features.from_transaction(tx)
    #   ml_score = model.predict_with_fallback(customer_id, feats)
    #   rule_score, reasons = rules.evaluate(tx, feats)
    #   risk = scoring.blend_scores(ml_score, rule_score)
    #   color = thresholds(risk)
    # Simulación determinista simple:
    results: List[ScoreResult] = []
    for idx, tx in enumerate(body.transactions):
        # Dummy heurística de ejemplo:
        ml_score = min(1.0, max(0.0, (tx.amount / 500.0)))       # más monto => más riesgo
        rule_score = 0.75 if (tx.channel == "online" and tx.amount > 200) else 0.25
        risk = 0.6 * ml_score + 0.4 * rule_score
        color = "rojo" if risk >= 0.75 else ("amarillo" if risk >= 0.45 else "verde")
        reasons = []
        if rule_score >= 0.7:
            reasons.append("Monto alto en canal online")
        if ml_score > 0.8:
            reasons.append("Monto fuera de patrón histórico")
        debug = {"approx_ml_score": ml_score, "approx_rule_score": rule_score}

        results.append(ScoreResult(
            transaction_idx=idx,
            risk_score=round(risk, 3),
            ml_score=round(ml_score, 3),
            rule_score=round(rule_score, 3),
            color=color,
            reasons=reasons or ["Perfil de riesgo bajo"],
            debug=debug
        ))

    return ScoreResponse(results=results)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Guarda feedback del usuario/jurado para recalibrar"
)
async def store_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """
    Guarda etiquetas: not_me/fraud/legit/unknown. Útil para recalibrar umbrales,
    whitelists/blacklists y evaluación del modelo.
    """
    # TODO:
    # - storage.insert_feedback(body)
    # - opcional: actualizar umbrales en caliente o colas de retraining
    stored_ok = True
    return FeedbackResponse(customer_id=body.customer_id, stored=stored_ok, message="Feedback stored")
