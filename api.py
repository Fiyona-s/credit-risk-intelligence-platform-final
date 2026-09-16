"""FastAPI REST endpoint, alongside the Streamlit app — same preprocessor, same saved
LightGBM model, same SHAP explainer as app.py's Risk Prediction / Explainability tabs.
No separate inference logic: this is a thin JSON wrapper around
src.ml.predict.predict_applicant() and src.explainability.shap_explainer's SHAP
computation, so the two surfaces (UI and API) can never silently drift apart.

Run standalone:  uvicorn api:app --host 0.0.0.0 --port 8000
In Docker:        started by supervisord alongside Streamlit — see docker-compose.yml
                  and supervisord.conf.
"""
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ml.predict import MODEL_PATH, load_model, predict_applicant
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    description="Score a loan applicant: risk score, risk band, and top SHAP drivers.",
    version="1.0.0",
)


class ShapDriver(BaseModel):
    feature: str
    value: Any
    direction: str = Field(description='"increased" or "decreased" risk')
    impact: float = Field(description="abs(SHAP value) — magnitude of this feature's contribution")


class PredictResponse(BaseModel):
    # NOTE (accuracy fix, no behavior change): this field is named "risk_score" but
    # actually returns the RAW LightGBM probability (0-1), not the 0-1000 integer
    # risk_score src.ml.risk_scoring computes from calibrated PD, nor the calibrated
    # probability itself. Kept as-is for backward compatibility with existing API
    # consumers; documented accurately here rather than silently relabeled or expanded.
    # This endpoint does not currently expose calibrated_probability, the Milestone 5
    # risk_score, or the Milestone 6 cost-sensitive decision — only the raw
    # probability and risk_band (also raw-probability-independent as of Milestone 5;
    # see below) plus SHAP drivers.
    risk_score: float = Field(
        description="Raw LightGBM predicted probability of default, 0-1 (NOT the "
        "calibrated probability, and NOT the separate 0-1000 risk_score field "
        "src.ml.risk_scoring computes internally — same field name, different value, "
        "kept for backward compatibility)."
    )
    risk_band: str = Field(
        description='One of "Low", "Moderate", "High", "Very High" — a project-defined '
        "risk category derived from calibrated PD (src.ml.risk_scoring), not an "
        "official banking/credit-bureau grade. Falls back to raw-probability-based "
        "bands if no calibrator is available."
    )
    top_shap_drivers: List[ShapDriver]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe. Always returns 200 if the process itself is up —
    model_loaded=False just means /predict isn't ready yet (e.g. no trained model
    committed/mounted), not that the API server is unhealthy.
    """
    import os
    return HealthResponse(status="ok", model_loaded=os.path.exists(MODEL_PATH))


@app.post("/predict", response_model=PredictResponse)
def predict(applicant: Dict[str, Any]) -> PredictResponse:
    """Score one applicant. Body is a flat JSON object of raw feature values — same
    shape src.ml.predict.predict_applicant() and the Streamlit Risk Prediction tab's
    manual-entry form already accept (missing fields are imputed by the trained
    pipeline, same as elsewhere in this project — no separate validation schema here).
    """
    try:
        load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    try:
        result = predict_applicant(applicant)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not score this applicant: {e}") from e

    try:
        from src.explainability.shap_explainer import explain_applicant_structured
        top_drivers = explain_applicant_structured(applicant, top_n=3)
    except Exception as e:
        # Explainability is a bonus on top of the score, not the score itself — a SHAP
        # failure shouldn't take down a working prediction. Logged, not silently dropped.
        log.warning(f"SHAP explanation failed, returning prediction without it: {e}")
        top_drivers = []

    return PredictResponse(
        risk_score=result["probability_of_default"],
        risk_band=result["risk_band"],
        top_shap_drivers=[ShapDriver(**d) for d in top_drivers],
    )
