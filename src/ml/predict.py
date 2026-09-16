"""Inference: given applicant features, output a default-probability risk score and a risk band.

Milestone 5: predict_applicant() is now a thin, backward-compatible wrapper
around src.ml.risk_scoring.score_applicant() — the single centralized
risk-scoring engine (feature engineering -> LightGBM -> raw probability ->
calibration -> calibrated PD -> risk score -> risk band). See that module's
docstring for the full architecture and the risk-band methodology.

Terminology (see src/ml/calibration.py and src/ml/risk_scoring.py for the
full methodology):
    raw_probability          — the deployed LightGBM pipeline's predict_proba()
                                output, unqualified. `probability_of_default`
                                is kept as an exact alias of this value — same
                                meaning as before Milestone 4/5 — for backward
                                compatibility with every existing caller
                                (app.py, api.py, tests).
    calibrated_probability   — raw_probability passed through the saved
                                probability calibrator (models/probability_
                                calibrator.joblib), when that artifact exists.
                                None if no calibrator has been fit yet.
    risk_band                — Milestone 5 CHANGE: now derived from
                                calibrated_probability (falling back to
                                raw_probability, explicitly, if no calibrator
                                exists — see calibration_method=="none").
                                Previously (Milestones 1-4) this was derived
                                from raw_probability against
                                RISK_THRESHOLD_LOW/MEDIUM. This is a documented,
                                intentional redefinition — see
                                reports/risk_scoring_report.md for the old-vs-
                                new comparison. The field NAME is unchanged for
                                backward compatibility; its DEFINITION changed.
    cost_optimal_decision,
    optimal_threshold        — UNCHANGED: still computed from raw_probability
                                against models/optimal_threshold.json (itself
                                swept on raw probabilities in src.ml.train).
                                Milestone 6 will explicitly move this onto
                                calibrated PD; this milestone deliberately does
                                not touch it (see src.ml.risk_scoring's
                                docstring, "Business decision threshold stays
                                separate").
"""
import json
import os
from typing import Dict, Optional, Union

import joblib
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")
OPTIMAL_THRESHOLD_PATH = os.path.join(settings.models_dir, "optimal_threshold.json")
CALIBRATOR_PATH = os.path.join(settings.models_dir, "probability_calibrator.joblib")
DECISION_ARTIFACT_PATH = os.path.join(settings.models_dir, "decision_threshold.json")

_model_cache = None
_optimal_threshold_cache = None
_calibrator_cache = None
_calibrator_checked = False
_decision_artifact_cache = None
_decision_artifact_checked = False


def _load_decision_artifact():
    """The Milestone 6 calibrated-PD decision artifact (models/decision_threshold.json),
    if src.ml.decision_optimization.run_decision_optimization_pipeline() has been
    run yet. Returns None (never raises) when unavailable — predict_applicant()
    must then report decision=None / decision_probability_source="unavailable"
    rather than fabricating a decision (see src.ml.decision_optimization.decide()).
    """
    global _decision_artifact_cache, _decision_artifact_checked
    if not _decision_artifact_checked:
        from src.ml.decision_optimization import DecisionArtifact

        _decision_artifact_cache = DecisionArtifact.load(DECISION_ARTIFACT_PATH)
        _decision_artifact_checked = True
        if _decision_artifact_cache is not None:
            log.info(f"Loaded decision artifact ({_decision_artifact_cache.scenario} scenario, "
                      f"threshold={_decision_artifact_cache.selected_threshold:.4f}) from "
                      f"{DECISION_ARTIFACT_PATH}")
    return _decision_artifact_cache


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python -m src.ml.train` first."
            )
        _model_cache = joblib.load(MODEL_PATH)
        log.info(f"Loaded model from {MODEL_PATH}")
    return _model_cache


def load_optimal_threshold(path: str = OPTIMAL_THRESHOLD_PATH) -> Optional[dict]:
    """The cost-sensitive threshold sweep saved by src/ml/train.py (see
    sweep_cost_sensitive_threshold there) — the optimal threshold under
    FP_COST/FN_COST at training time, plus the full per-threshold false_positives/
    false_negatives arrays so a caller (app.py's Risk Prediction tab) can recompute
    the optimal threshold for a *different* cost ratio via pure arithmetic, without
    needing the model or validation data again. Returns None if training hasn't
    been run with cost-sensitive sweeping yet (i.e. no optimal_threshold.json exists)
    — callers should fall back to the fixed RISK_THRESHOLD_LOW/MEDIUM bands.
    """
    global _optimal_threshold_cache
    if _optimal_threshold_cache is None:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            _optimal_threshold_cache = json.load(f)
    return _optimal_threshold_cache


def load_calibrator(path: str = CALIBRATOR_PATH):
    """The probability calibrator saved by src.ml.calibration.run_calibration_pipeline(),
    if one has been fit yet. Returns None (never raises) when unavailable — a
    fresh checkout or a model retrained without rerunning `python -m
    src.ml.calibration` has no calibrator, and predict_applicant() must degrade
    to reporting calibrated_probability=None rather than fabricating one.

    Lazy-imports src.ml.calibration (which itself imports src.ml.train) so that
    processes never touching calibration — most Streamlit tabs, most API
    requests — don't pay that import cost, consistent with this codebase's
    existing lazy-import pattern for heavier optional dependencies.
    """
    global _calibrator_cache, _calibrator_checked
    if not _calibrator_checked:
        from src.ml.calibration import ProbabilityCalibrator

        _calibrator_cache = ProbabilityCalibrator.load(path)
        _calibrator_checked = True
        if _calibrator_cache is not None:
            log.info(f"Loaded probability calibrator ({_calibrator_cache.method}) from {path}")
    return _calibrator_cache


def cost_optimal_decision(probability: float, optimal_threshold: Optional[dict] = None) -> Optional[str]:
    """"Approve" or "Decline" based on the cost-sensitive optimal threshold — an
    additive alternative to the fixed Low/Medium/High risk_band, not a replacement
    for it. Returns None if no optimal_threshold.json is available.
    """
    optimal_threshold = optimal_threshold if optimal_threshold is not None else load_optimal_threshold()
    if optimal_threshold is None:
        return None
    return "Decline" if probability >= optimal_threshold["optimal_threshold"] else "Approve"


def predict_applicant(applicant: Union[Dict, pd.DataFrame]) -> dict:
    """Score a single applicant (dict of raw feature values or a one-row DataFrame).

    Thin, backward-compatible wrapper around src.ml.risk_scoring.score_applicant()
    — the centralized risk-scoring engine (Milestone 5). All historical keys
    (probability_of_default, risk_band, cost_optimal_decision, optimal_threshold,
    features_used) are preserved; risk_band's definition changed (now derived
    from calibrated_probability, falling back to raw_probability if no
    calibrator exists) — see this module's docstring and
    reports/risk_scoring_report.md for the old-vs-new comparison.

    Milestone 6 ADDS a second, parallel business-decision surface
    (decision/decision_threshold/decision_probability_source/decision_scenario/
    fp_cost/fn_cost) computed from CALIBRATED PD against
    models/decision_threshold.json (src.ml.decision_optimization) — see that
    module's docstring for the full cost-sensitive optimization methodology.
    cost_optimal_decision/optimal_threshold are UNCHANGED: still computed from
    raw_probability against the legacy models/optimal_threshold.json. Both
    surfaces coexist deliberately; neither silently replaces the other.
    """
    from src.ml.risk_scoring import score_applicant

    result = score_applicant(applicant)

    optimal_threshold = load_optimal_threshold()
    legacy_decision = cost_optimal_decision(result.raw_probability, optimal_threshold)

    from src.ml.decision_optimization import decide

    decision_artifact = _load_decision_artifact()
    new_decision = decide(result.calibrated_probability, decision_artifact)

    return {
        # Unchanged for backward compatibility — every existing caller (app.py,
        # api.py) keeps using this exact value exactly as before: the RAW
        # model score. See calibrated_probability for the empirically-
        # calibrated one, and risk_scoring.RiskScoringResult's docstring for
        # the full field contract.
        # Milestone 7 fix: RiskScoringResult has computed this correctly since
        # Milestone 5, but this wrapper never surfaced it — a real gap, caught
        # by src.explainability.shap_explainer needing a reliable applicant_id.
        "applicant_id": result.applicant_id,
        "probability_of_default": result.raw_probability,
        "raw_probability": result.raw_probability,
        "calibrated_probability": result.calibrated_probability,
        "calibration_method": result.calibration_method,
        "risk_score": result.risk_score,
        "risk_band": result.risk_band,
        "model_version": result.model_version,
        "feature_version": result.feature_version,
        # Legacy business decision (raw probability, Milestone 1) — unchanged.
        "cost_optimal_decision": legacy_decision,
        "optimal_threshold": optimal_threshold["optimal_threshold"] if optimal_threshold else None,
        # New business decision (calibrated PD, Milestone 6) — see
        # src.ml.decision_optimization.decide()'s docstring for field meanings.
        "decision": new_decision["decision"],
        "decision_probability_source": new_decision["decision_probability_source"],
        "decision_threshold": new_decision["decision_threshold"],
        "decision_scenario": new_decision["decision_scenario"],
        "decision_scenario_expected_cost": new_decision["decision_scenario_expected_cost"],
        "fp_cost": new_decision["fp_cost"],
        "fn_cost": new_decision["fn_cost"],
        "features_used": result.features_used,
    }


if __name__ == "__main__":
    from src.data.loader import build_joined_dataset

    df = build_joined_dataset(is_train=True)
    sample = df.iloc[0].to_dict()
    result = predict_applicant(sample)
    print(f"Probability of default: {result['probability_of_default']:.4f}")
    print(f"Risk band: {result['risk_band']}")
    print(f"Actual TARGET: {sample['TARGET']}")
