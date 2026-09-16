"""Unit tests for the Milestone 5 centralized risk-scoring engine (src/ml/risk_scoring.py)
and its integration into src/ml/predict.py.

Synthetic fixtures throughout — no real Home Credit CSVs required.
"""
import numpy as np
import pytest

from src.ml import predict as predict_module
from src.ml import risk_scoring as rs


class _FakeCalibrator:
    """A trivial, deterministic stand-in for ProbabilityCalibrator — halves the
    raw probability, so raw != calibrated is easy to assert on and the mapping
    is simple enough to hand-verify in test assertions.
    """
    method = "sigmoid"

    def transform(self, raw_probability):
        return np.asarray(raw_probability) * 0.5


class _FakePipeline:
    def __init__(self, proba_default: float):
        self._proba_default = proba_default

    def predict_proba(self, df):
        return np.array([[1 - self._proba_default, self._proba_default]])


def _fake_bundle(proba_default: float) -> dict:
    return {
        "pipeline": _FakePipeline(proba_default),
        "numeric_cols": ["AMT_INCOME_TOTAL"],
        "categorical_cols": [],
    }


# ---- Test 1: risk score increases monotonically with PD --------------------

def test_risk_score_increases_with_pd():
    scores = [rs.compute_risk_score(pd) for pd in [0.01, 0.05, 0.10, 0.30, 0.60, 0.99]]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)  # strictly increasing, no ties in this spread


def test_risk_score_bounded_and_clips_out_of_range_input():
    assert rs.compute_risk_score(0.0) == 0
    assert rs.compute_risk_score(1.0) == rs.RISK_SCORE_SCALE
    assert rs.compute_risk_score(1.5) == rs.RISK_SCORE_SCALE  # clipped, not out of bounds
    assert rs.compute_risk_score(-0.2) == 0  # clipped


# ---- Test 2: risk score is deterministic ------------------------------------

def test_risk_score_deterministic():
    results = [rs.compute_risk_score(0.1234) for _ in range(10)]
    assert len(set(results)) == 1


# ---- Test 3 & 4: risk bands correspond to calibrated PD, thresholds applied correctly --

@pytest.mark.parametrize("pd_value,expected_band", [
    (0.0, "Low"),
    (0.049, "Low"),
    (0.05, "Moderate"),
    (0.099, "Moderate"),
    (0.10, "High"),
    (0.199, "High"),
    (0.20, "Very High"),
    (0.99, "Very High"),
])
def test_risk_band_thresholds_applied_correctly(pd_value, expected_band):
    assert rs.compute_risk_band(pd_value) == expected_band


# ---- Test 5: band ordering is monotonic -------------------------------------

def test_band_order_thresholds_are_strictly_increasing():
    lows = [rs.RISK_BAND_THRESHOLDS[band][0] for band in rs.RISK_BAND_ORDER]
    assert lows == sorted(lows)
    assert len(set(lows)) == len(lows)


def test_higher_band_never_has_lower_pd_range():
    for i in range(len(rs.RISK_BAND_ORDER) - 1):
        current_band = rs.RISK_BAND_ORDER[i]
        next_band = rs.RISK_BAND_ORDER[i + 1]
        assert rs.RISK_BAND_THRESHOLDS[current_band][1] == rs.RISK_BAND_THRESHOLDS[next_band][0]


# ---- Test 6: applicants with missing history still receive a valid risk score --

def test_missing_history_applicant_gets_valid_score(monkeypatch):
    """Simulates the Milestone 1 "no bureau/prev-app history" scenario at the
    risk-scoring layer: build_aggregate_features/engineer_features tolerate
    missing aggregate columns (already tested in test_feature_engineering.py);
    this test confirms score_applicant() still produces a complete, valid
    result end to end for such an applicant.
    """
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.15))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())

    # No SK_ID_CURR at all -> build_aggregate_features() is a no-op passthrough
    # (per src/data/preprocessor.py), exactly like a brand-new applicant with no
    # bureau/previous-application history anywhere.
    applicant = {"AMT_INCOME_TOTAL": 120000.0}
    result = rs.score_applicant(applicant)

    assert result.applicant_id is None
    assert 0.0 <= result.raw_probability <= 1.0
    assert result.risk_score is not None
    assert 0 <= result.risk_score <= rs.RISK_SCORE_SCALE
    assert result.risk_band in rs.RISK_BAND_ORDER


# ---- Test 7: raw and calibrated probabilities remain distinct --------------

def test_raw_and_calibrated_probabilities_are_distinct(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.40))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())

    result = rs.score_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result.raw_probability == pytest.approx(0.40)
    assert result.calibrated_probability == pytest.approx(0.20)  # _FakeCalibrator halves it
    assert result.raw_probability != result.calibrated_probability


# ---- Test 8: calibration method is correctly reported -----------------------

def test_calibration_method_reported_when_calibrator_present(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.2))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())

    result = rs.score_applicant({"AMT_INCOME_TOTAL": 100000.0})
    assert result.calibration_method == "sigmoid"


# ---- Test 9: the underlying model probability is unchanged by risk-score calc --

def test_raw_probability_unaffected_by_score_and_band_computation(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.6789))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())

    result = rs.score_applicant({"AMT_INCOME_TOTAL": 100000.0})
    # raw_probability must be exactly what the (fake) LightGBM pipeline produced —
    # computing risk_score/risk_band from the CALIBRATED value must never feed
    # back into or alter the raw one.
    assert result.raw_probability == pytest.approx(0.6789)


# ---- Test 10: existing prediction callers remain compatible ----------------

def test_predict_applicant_preserves_all_historical_keys(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.25))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: {"optimal_threshold": 0.5})

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    for key in ("probability_of_default", "risk_band", "cost_optimal_decision",
                "optimal_threshold", "features_used", "raw_probability",
                "calibrated_probability", "calibration_method", "risk_score",
                "model_version", "feature_version"):
        assert key in result, f"missing expected key: {key}"

    # probability_of_default must still mean exactly "raw model probability" —
    # the historical meaning, unchanged.
    assert result["probability_of_default"] == pytest.approx(result["raw_probability"])
    assert result["probability_of_default"] == pytest.approx(0.25)


# ---- Test 11: business decision threshold stays separate from risk-band thresholds --

def test_cost_optimal_decision_independent_of_risk_band(monkeypatch):
    """Construct a case where the raw probability (0.55) crosses the cost-optimal
    threshold (0.5, so cost_optimal_decision="Decline") while the risk band is
    computed entirely separately, from calibrated PD against its own thresholds —
    demonstrating the two systems use different inputs/boundaries and are not
    silently coupled.
    """
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.55))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: {"optimal_threshold": 0.5})

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result["cost_optimal_decision"] == "Decline"  # raw 0.55 >= optimal_threshold 0.5
    # calibrated = 0.55 * 0.5 = 0.275 -> falls in [0.20, 1.0) -> "Very High"
    assert result["risk_band"] == "Very High"
    assert result["calibrated_probability"] == pytest.approx(0.275)


def test_cost_optimal_decision_uses_raw_not_calibrated_probability(monkeypatch):
    """More directly: cost_optimal_decision must be computed from raw_probability,
    not calibrated_probability — verified by picking values where the two
    disagree about which side of the cost threshold they fall on.
    """
    # raw=0.6 -> above optimal_threshold 0.5 -> Decline.
    # calibrated (halved) = 0.3 -> also above 0.5? No: 0.3 < 0.5 -> if the code
    # mistakenly used calibrated probability, it would say Approve instead.
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.6))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: {"optimal_threshold": 0.5})

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})
    assert result["cost_optimal_decision"] == "Decline"
    assert result["raw_probability"] == pytest.approx(0.6)
    assert result["calibrated_probability"] == pytest.approx(0.3)


# ---- Test 12: missing calibrator -> explicit raw-probability fallback ------

def test_missing_calibrator_falls_back_to_raw_probability_explicitly(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.07))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: None)

    result = rs.score_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result.calibration_method == "none"
    assert result.calibrated_probability is None
    # Band/score must fall back to raw_probability (0.07 -> "Moderate" band,
    # since 0.05 <= 0.07 < 0.10), not silently break or default to "Low".
    assert result.risk_band == rs.compute_risk_band(0.07)
    assert result.risk_score == rs.compute_risk_score(0.07)


def test_missing_calibrator_via_predict_applicant_wrapper(monkeypatch):
    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.30))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: None)
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})
    assert result["calibration_method"] == "none"
    assert result["calibrated_probability"] is None
    assert result["risk_band"] == "Very High"  # raw 0.30 >= 0.20 fallback threshold


# ---- RiskScoringResult.to_dict() ---------------------------------------------

def test_to_dict_includes_all_fields_and_preserves_dataframe():
    import pandas as pd

    df = pd.DataFrame({"a": [1]})
    result = rs.RiskScoringResult(
        applicant_id=42, raw_probability=0.3, calibrated_probability=0.2,
        calibration_method="sigmoid", risk_score=200, risk_band="Moderate",
        model_version="lightgbm-abc123", feature_version="v1", features_used=df,
    )
    d = result.to_dict()
    assert d["applicant_id"] == 42
    assert d["risk_score"] == 200
    assert d["features_used"] is df  # same object, not deep-copied
