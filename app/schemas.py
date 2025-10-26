# app/schemas.py
from __future__ import annotations
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

# ==== Core domain ====

Channel = Literal["atm", "branch", "online", "present", "unknown"]

class TransactionIn(BaseModel):
    customer_id: str = Field(..., example="cust_123")
    account_id: str = Field(..., example="acc_001")
    amount: float = Field(..., gt=0, example=210.50)
    currency: Optional[str] = Field(default="USD", example="USD")
    merchant_id: Optional[str] = Field(default=None, example="m_42")
    mcc: Optional[str] = Field(default=None, example="5814")  # Merchant Category Code
    lat: Optional[float] = Field(default=None, ge=-90, le=90, example=29.42)
    lon: Optional[float] = Field(default=None, ge=-180, le=180, example=-98.49)
    timestamp: datetime = Field(..., example="2025-10-25T13:10:00Z")
    channel: Optional[Channel] = Field(default="present")

class ScoreRequest(BaseModel):
    # Permite batch scoring
    transactions: List[TransactionIn]

class ScoreResult(BaseModel):
    transaction_idx: int
    risk_score: float = Field(..., ge=0, le=1)
    ml_score: float = Field(..., ge=0, le=1)
    rule_score: float = Field(..., ge=0, le=1)
    color: Literal["rojo", "amarillo", "verde"]
    reasons: List[str]
    debug: Optional[dict] = None

class ScoreResponse(BaseModel):
    results: List[ScoreResult]

# ==== Ingest ====

class IngestResponse(BaseModel):
    customer_id: str
    accounts_found: int
    transactions_inserted: int
    bills_inserted: int
    merchants_upserted: int
    message: str = "Ingest complete"

# ==== Train ====

class TrainRequest(BaseModel):
    retrain_global: bool = Field(default=True)

class TrainResponse(BaseModel):
    customer_id: str
    customer_model_trained: bool
    global_model_trained: bool
    n_samples_customer: int
    params: dict = Field(default_factory=dict)
    message: str = "Training complete"

# ==== Insights (suscripciones y gastos hormiga) ====

class SubscriptionInsight(BaseModel):
    merchant_id: str
    merchant_name: Optional[str] = None
    periodicity_days: int
    avg_amount: float
    status: Literal["activa_ok", "activa_sin_uso", "duplicada"]
    recommendation: Optional[str] = None  # e.g., "Considera pausar/cancelar"

class LeakInsight(BaseModel):
    bucket: str = Field(..., example="cafés")
    monthly: float = Field(..., example=38.2)
    annual: float = Field(..., example=458.4)
    save30: float = Field(..., example=11.46)

class InsightsResponse(BaseModel):
    customer_id: str
    subscriptions: List[SubscriptionInsight]
    leaks: List[LeakInsight]

# ==== Feedback ====

FeedbackLabel = Literal["not_me", "fraud", "legit", "unknown"]

class FeedbackRequest(BaseModel):
    customer_id: str
    txn_id: Optional[str] = Field(default=None, description="ID interno de la transacción si existe")
    label: FeedbackLabel
    note: Optional[str] = None
    context_url: Optional[HttpUrl] = None  # por si enlazas a un dashboard

class FeedbackResponse(BaseModel):
    customer_id: str
    stored: bool
    message: str = "Feedback stored"
