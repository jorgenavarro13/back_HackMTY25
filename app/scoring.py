# app/scoring.py
from __future__ import annotations
from typing import Dict, List, Tuple

# Pesos y umbrales (puedes exponerlos vía ENV si quieres)
ML_WEIGHT = 0.6
RULE_WEIGHT = 0.4

THRESHOLDS = {
    "red": 0.75,
    "yellow": 0.45
    # green < 0.45
}

def blend(ml_score: float, rule_score: float) -> float:
    """
    Fusión convexa en [0,1]: risk = 0.6*ml + 0.4*rules
    """
    ml = float(max(0.0, min(1.0, ml_score)))
    rl = float(max(0.0, min(1.0, rule_score)))
    risk = ML_WEIGHT * ml + RULE_WEIGHT * rl
    return float(max(0.0, min(1.0, risk)))

def color_from_risk(risk: float) -> str:
    if risk >= THRESHOLDS["red"]:
        return "rojo"
    if risk >= THRESHOLDS["yellow"]:
        return "amarillo"
    return "verde"

def finalize_response(
    ml_score: float,
    rule_score: float,
    reasons: List[str]
) -> Dict[str, object]:
    """
    Empaqueta los resultados estándar de la API:
    {
      "risk_score": float,
      "color": "rojo|amarillo|verde",
      "ml_score": float,
      "rule_score": float,
      "reasons": [...]
    }
    """
    risk = blend(ml_score, rule_score)
    return {
        "risk_score": round(risk, 3),
        "color": color_from_risk(risk),
        "ml_score": round(float(ml_score), 3),
        "rule_score": round(float(rule_score), 3),
        "reasons": reasons if reasons else ["Perfil de riesgo bajo"]
    }

# Opcional: helper de “end-to-end” si ya traes las piezas separadas
def blend_scores_from_parts(ml_score: float, rule_score: float, reasons: List[str]) -> Dict[str, object]:
    return finalize_response(ml_score, rule_score, reasons)
