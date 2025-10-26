# app/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import storage
from .features import build_features

RANDOM_STATE = 42
IF_N_ESTIMATORS = 200
IF_CONTAMINATION = 0.03

# columnas de "z-score" candidatas para explicar outliers
Z_FEATURE_HINTS = ("_z_", "cat_z_", "mer_z_")

@dataclass
class TrainReport:
    customer_id: Optional[str]
    n_samples: int
    n_features: int
    params: Dict[str, Any]
    message: str = "Training complete"


# ----------------------------
# Utilidades internas
# ----------------------------
def _make_pipeline() -> Pipeline:
    """
    Pipeline = Imputer(median) -> StandardScaler -> IsolationForest
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Selecciona columnas numéricas para el modelo.
    Excluye identificadores y campos claramente no numéricos.
    """
    exclude = {"customer_id", "type", "merchant_id", "merchant_category"}
    numeric_cols = df.columns.difference(list(exclude))
    # Filtra numéricos reales
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric_cols)


def _align_features_for_inference(
    feats_row: pd.Series,
    feature_names: List[str]
) -> np.ndarray:
    """
    Garantiza que la fila de features tenga exactamente el orden/las columnas usadas en entrenamiento.
    Si falta alguna columna, la rellena con 0; si sobran, las ignora.
    """
    vec = []
    for col in feature_names:
        val = feats_row.get(col, 0.0)
        try:
            val = float(val)
        except Exception:
            val = 0.0
        vec.append(val)
    return np.asarray(vec, dtype=float).reshape(1, -1)


def _risk_from_isoforest(pipeline: Pipeline, X: np.ndarray) -> float:
    """
    IsolationForest: decision_function(X) mayor => menos anómalo.
    Convertimos a riesgo en [0,1] donde mayor => más anómalo.
    Heurística estable: risk = sigmoid(-decision_function).
    """
    # Asegura que existe el paso 'clf'
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        # fallback: intenta pipeline.decision_function
        raw = -float(pipeline.decision_function(X)[0])
    else:
        raw = -float(clf.decision_function(pipeline[:-1].transform(X))[0])
    # sigmoide
    risk = 1.0 / (1.0 + np.exp(-raw))
    # recorta por si acaso
    return float(np.clip(risk, 0.0, 1.0))


def _top_z_features(feats_row: pd.Series, topk: int = 5) -> List[Tuple[str, float]]:
    """
    Extrae las features cuyo nombre sugiere un z-score y ordena por |valor| descendente.
    Devuelve una lista [(col, valor), ...] de tamaño <= topk.
    """
    candidates = []
    for col, val in feats_row.items():
        if any(h in col for h in Z_FEATURE_HINTS):
            try:
                v = float(val)
            except Exception:
                continue
            if np.isfinite(v):
                candidates.append((col, abs(v)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [(c, float(feats_row[c])) for c, _ in candidates[:topk]]


# ----------------------------
# Entrenamiento
# ----------------------------
def train_customer_model(customer_id: str) -> TrainReport:
    """
    Entrena un IsolationForest para un cliente y lo guarda en storage.models.
    """
    df = build_features(customer_id)
    if df.empty:
        return TrainReport(customer_id=customer_id, n_samples=0, n_features=0,
                           params={}, message="No data to train")

    feat_cols = _select_feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float, copy=False)

    pipe = _make_pipeline()
    pipe.fit(X)

    params = {
        "algorithm": "IsolationForest",
        "n_estimators": IF_N_ESTIMATORS,
        "contamination": IF_CONTAMINATION,
        "random_state": RANDOM_STATE,
        "feature_names": feat_cols
    }
    # Persistimos pipeline completo
    storage.insert_customer_model(customer_id, pipe, params=params)

    return TrainReport(customer_id=customer_id, n_samples=X.shape[0], n_features=X.shape[1], params=params)


def train_global_model(customer_ids: Optional[List[str]] = None) -> TrainReport:
    """
    Entrena un modelo global juntando muestras de varios clientes.
    Si no se proveen customer_ids, intenta entrenar con *todos* los registros de raw_transactions
    leyendo por batches de clientes a partir de feedback/transactions; para simplicidad,
    aquí requerimos una lista explícita o entrenamos con un único cliente si la lista es None
    y hay datos en su tabla (puedes adaptar esto a tu flujo).
    """
    # Estrategia: si no hay lista, usa un único cliente "sintético" leyendo feedback para
    # inferir IDs usados recientemente. En este MVP, aceptamos una lista explícita; si None,
    # lanzamos excepción amigable.
    if not customer_ids:
        # Como fallback práctico en hackathon: entrena con TODOS los tx de un cliente demo
        # Llama a quien use esta función a pasar la lista real de clientes si la tiene.
        raise ValueError("train_global_model requiere una lista de customer_ids (p.ej., ['cust_123','cust_456']).")

    frames = []
    for cid in customer_ids:
        df = build_features(cid)
        if not df.empty:
            frames.append(df)

    if not frames:
        return TrainReport(customer_id=None, n_samples=0, n_features=0, params={}, message="No data to train")

    df_all = pd.concat(frames, axis=0, ignore_index=False)
    feat_cols = _select_feature_columns(df_all)
    X = df_all[feat_cols].to_numpy(dtype=float, copy=False)

    pipe = _make_pipeline()
    pipe.fit(X)

    params = {
        "algorithm": "IsolationForest",
        "n_estimators": IF_N_ESTIMATORS,
        "contamination": IF_CONTAMINATION,
        "random_state": RANDOM_STATE,
        "feature_names": feat_cols,
        "customers": customer_ids
    }
    storage.insert_global_model(pipe, params=params)

    return TrainReport(customer_id=None, n_samples=X.shape[0], n_features=X.shape[1], params=params)


# ----------------------------
# Predicción + Fallback
# ----------------------------
def predict_with_fallback(
    customer_id: str,
    feats_row: pd.Series,
    rules_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predice riesgo con el modelo del cliente; si no existe, usa el global.
    Retorna:
    {
      "ml_score": float in [0,1],
      "used_model": "customer"|"global",
      "top_features": [(name, value), ...],   # basado en |z|
      "rules": { "rule_score": float, "reasons": [..] }  # si rules.evaluate está disponible y se pasó payload
    }
    """
    loaded = storage.get_latest_customer_model(customer_id)
    used_model = "customer"
    if not loaded:
        loaded = storage.get_latest_global_model()
        used_model = "global"

    if not loaded:
        # No hay modelo entrenado
        return {
            "ml_score": 0.0,
            "used_model": "none",
            "top_features": [],
            "rules": None
        }

    model_obj, params, _trained_at = loaded
    feature_names = params.get("feature_names", [])

    # Alinear vector según columnas del entrenamiento
    X = _align_features_for_inference(feats_row, feature_names)
    ml_score = _risk_from_isoforest(model_obj, X)

    # Explicación simple basada en |z|
    top_feats = _top_z_features(feats_row, topk=5)

    # (Opcional) Ejecutar reglas si están disponibles y se entregó contexto
    rules_result = None
    if rules_payload is not None:
        try:
            # Intento perezoso para no romper si aún no tienes rules.py
            from . import rules  # type: ignore
            rule_score, reasons = rules.evaluate(rules_payload, feats_row)  # define esta firma en tu rules.py
            rules_result = {"rule_score": float(rule_score), "reasons": list(reasons)}
        except Exception:
            rules_result = None

    return {
        "ml_score": float(ml_score),
        "used_model": used_model,
        "top_features": top_feats,
        "rules": rules_result
    }
