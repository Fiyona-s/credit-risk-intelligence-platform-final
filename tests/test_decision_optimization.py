"""Unit tests for the Milestone 6 business decision optimization module
(src/ml/decision_optimization.py) and its integration into src/ml/predict.py.

Synthetic fixtures throughout — no real Home Credit CSVs required.
"""
import numpy as np
import pytest

from src.ml import decision_optimization as do
from src.ml import predict as predict_module


def _synthetic_calibrated_pd_and_labels(n=1000, seed=0):
    """A synthetic, informative, roughly-calibrated probability array: PD
    correlates with the true label, and (approximately) PD ~= P(default), so
    a real cost-minimizing threshold exists to find.
    """
    rng = np.random.default_rng(seed)
    true_pd = rng.uniform(0.0, 0.5, size=n)
    y = rng.binomial(1, true_pd)
    return true_pd, y


class _FakeCalibrator:
    method = "sigmoid"

    def transform(self, raw_probability):
        return np.asarray(raw_probability) * 0.5


class _FakePipeline:
    def __init__(self, proba_default: float):
        self._proba_default = proba_default

    def predict_proba(self, df):
        return np.array([[1 - self._proba_default, self._proba_default]])


def _fake_bundle(proba_default: float) -> dict:
    return {"pipeline": _FakePipeline(proba_default), "numeric_cols": ["AMT_INCOME_TOTAL"], "categorical_cols": []}


# ---- Test 1: expected cost calculation is correct ---------------------------

def test_expected_cost_formula():
    assert do.expected_cost(fp=10, fn=5, fp_cost=1.0, fn_cost=5.0) == pytest.approx(10 * 1.0 + 5 * 5.0)
    assert do.expected_cost(fp=0, fn=0, fp_cost=2.0, fn_cost=3.0) == 0.0
    assert do.expected_cost(fp=4, fn=0, fp_cost=1.5, fn_cost=100.0) == pytest.approx(6.0)


# ---- Test 2: threshold sweep returns valid thresholds -----------------------

def test_sweep_thresholds_returns_valid_range_and_counts():
    pd_values, y = _synthetic_calibrated_pd_and_labels()
    sweep = do.sweep_thresholds(y, pd_values, threshold_min=0.05, threshold_max=0.95, threshold_step=0.05,
                                 fp_cost=1.0, fn_cost=1.0)

    assert (sweep["threshold"] >= 0.05 - 1e-9).all()
    assert (sweep["threshold"] <= 0.95 + 1e-9).all()
    # Every row's TP+FP+FN+TN must equal the total sample size.
    totals = sweep["tp"] + sweep["fp"] + sweep["fn"] + sweep["tn"]
    assert (totals == len(y)).all()
    assert (sweep["precision"] >= 0).all() and (sweep["precision"] <= 1).all()
    assert (sweep["recall"] >= 0).all() and (sweep["recall"] <= 1).all()
    assert (sweep["acceptance_rate"] + sweep["rejection_rate"]).apply(lambda x: abs(x - 1.0) < 1e-9).all()


# ---- Test 3: optimal threshold minimizes expected cost ----------------------

def test_optimal_threshold_actually_minimizes_expected_cost():
    pd_values, y = _synthetic_calibrated_pd_and_labels()
    sweep = do.sweep_thresholds(y, pd_values, fp_cost=1.0, fn_cost=5.0)
    optimal = do.find_optimal_threshold(sweep)

    assert optimal["expected_cost"] == sweep["expected_cost"].min()
    # Brute-force cross-check: no row in the full sweep table beats it.
    assert (sweep["expected_cost"] >= optimal["expected_cost"] - 1e-9).all()


def test_find_optimal_threshold_raises_on_empty_sweep():
    import pandas as pd
    with pytest.raises(ValueError):
        do.find_optimal_threshold(pd.DataFrame())


# ---- Test 4: changing FP/FN costs can change the optimal threshold ---------

def test_changing_cost_ratio_changes_optimal_threshold():
    pd_values, y = _synthetic_calibrated_pd_and_labels()

    balanced = do.find_optimal_threshold(do.sweep_thresholds(y, pd_values, fp_cost=1.0, fn_cost=1.0))
    conservative = do.find_optimal_threshold(do.sweep_thresholds(y, pd_values, fp_cost=1.0, fn_cost=10.0))

    # Raising FN_COST should push the optimal threshold DOWN (catch more
    # defaulters) — not asserting an exact number (data-dependent), but the
    # direction should hold and the two should differ.
    assert conservative["threshold"] <= balanced["threshold"]
    assert conservative["threshold"] != balanced["threshold"]


# ---- Test 5: decision optimization uses calibrated probability -------------

def test_predict_applicant_decision_uses_calibrated_not_raw(monkeypatch, tmp_path):
    """raw=0.6 would cross a threshold of 0.5, but calibrated (halved) = 0.3
    would not — the NEW decision must follow the calibrated value.
    """
    artifact = do.DecisionArtifact(
        selected_threshold=0.5, probability_source="calibrated", scenario="Conservative",
        fp_cost=1.0, fn_cost=5.0, expected_cost=100.0, precision=0.2, recall=0.7, f1=0.3,
        acceptance_rate=0.8, rejection_rate=0.2, tp=10, fp=20, fn=5, tn=100,
        model_version="lightgbm-test", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=135,
    )
    artifact_path = str(tmp_path / "decision_threshold.json")
    artifact.save(artifact_path)

    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.6))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "DECISION_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(predict_module, "_decision_artifact_checked", False)
    monkeypatch.setattr(predict_module, "_decision_artifact_cache", None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result["raw_probability"] == pytest.approx(0.6)
    assert result["calibrated_probability"] == pytest.approx(0.3)
    assert result["decision"] == "Approve"  # calibrated 0.3 < threshold 0.5


# ---- Test 6: raw probability remains available ------------------------------

def test_raw_probability_still_present_alongside_new_decision(monkeypatch, tmp_path):
    artifact = do.DecisionArtifact(
        selected_threshold=0.2, probability_source="calibrated", scenario="Balanced",
        fp_cost=1.0, fn_cost=1.0, expected_cost=50.0, precision=0.3, recall=0.6, f1=0.4,
        acceptance_rate=0.7, rejection_rate=0.3, tp=5, fp=10, fn=3, tn=50,
        model_version="lightgbm-test", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=68,
    )
    artifact_path = str(tmp_path / "decision_threshold.json")
    artifact.save(artifact_path)

    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.44))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "DECISION_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(predict_module, "_decision_artifact_checked", False)
    monkeypatch.setattr(predict_module, "_decision_artifact_cache", None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})
    assert "raw_probability" in result
    assert result["raw_probability"] == pytest.approx(0.44)


# ---- Test 7: decision probability source is explicitly reported ------------

def test_decision_probability_source_reported_explicitly():
    artifact = do.DecisionArtifact(
        selected_threshold=0.1, probability_source="calibrated", scenario="Balanced",
        fp_cost=1.0, fn_cost=1.0, expected_cost=10.0, precision=0.5, recall=0.5, f1=0.5,
        acceptance_rate=0.9, rejection_rate=0.1, tp=1, fp=1, fn=1, tn=9,
        model_version="v", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=12,
    )
    result = do.decide(0.5, artifact)
    assert result["decision_probability_source"] == "calibrated"

    unavailable = do.decide(None, None)
    assert unavailable["decision_probability_source"] == "unavailable"


# ---- Test 8: scenario analysis returns all requested metrics ---------------

def test_scenario_analysis_returns_all_metrics():
    pd_values, y = _synthetic_calibrated_pd_and_labels()
    results = do.run_scenario_analysis(y, pd_values)

    assert len(results) == len(do.SCENARIOS)
    required_keys = {
        "scenario", "fp_cost", "fn_cost", "optimal_threshold", "expected_cost",
        "precision", "recall", "f1", "acceptance_rate", "rejection_rate",
        "tp", "fp", "fn", "tn",
    }
    for r in results:
        assert required_keys.issubset(r.keys())


# ---- Test 9: threshold artifact can be saved and loaded --------------------

def test_decision_artifact_save_and_load_roundtrip(tmp_path):
    artifact = do.DecisionArtifact(
        selected_threshold=0.37, probability_source="calibrated", scenario="Conservative",
        fp_cost=1.0, fn_cost=5.0, expected_cost=42.0, precision=0.25, recall=0.65, f1=0.36,
        acceptance_rate=0.75, rejection_rate=0.25, tp=13, fp=39, fn=7, tn=117,
        model_version="lightgbm-abc", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=176,
    )
    path = str(tmp_path / "decision_threshold.json")
    artifact.save(path)

    loaded = do.DecisionArtifact.load(path)
    assert loaded is not None
    assert loaded.selected_threshold == pytest.approx(0.37)
    assert loaded.scenario == "Conservative"
    assert loaded.fn_cost == pytest.approx(5.0)


def test_decision_artifact_load_returns_none_when_missing(tmp_path):
    assert do.DecisionArtifact.load(str(tmp_path / "nope.json")) is None


# ---- Test 10: changing calibration (probability values) can change the optimal threshold --

def test_different_probability_arrays_can_yield_different_optimal_thresholds():
    """Two different "calibrations" of the SAME underlying signal (one
    compressed toward low values, one stretched toward high values) must
    select different optimal thresholds — demonstrating the optimizer
    genuinely depends on the probability scale it's given, not just on y.
    Both scaled arrays are derived FROM true_pd (so they carry real signal
    about y), unlike two independent random draws which would degenerately
    both minimize at the sweep floor regardless of scale.
    """
    true_pd, y = _synthetic_calibrated_pd_and_labels()

    compressed = true_pd * 0.2      # a "calibration" that systematically underestimates risk
    stretched = np.clip(true_pd * 3.0, 0, 1)  # one that systematically overestimates it

    opt_compressed = do.find_optimal_threshold(do.sweep_thresholds(y, compressed, fp_cost=1.0, fn_cost=5.0))
    opt_stretched = do.find_optimal_threshold(do.sweep_thresholds(y, stretched, fp_cost=1.0, fn_cost=5.0))

    assert opt_compressed["threshold"] != opt_stretched["threshold"]


# ---- Test 11: risk-band thresholds and decision thresholds remain independent --

def test_risk_band_and_decision_thresholds_are_independent_systems(monkeypatch, tmp_path):
    from src.ml import risk_scoring as rs

    # A decision threshold that does NOT line up with any risk-band boundary.
    artifact = do.DecisionArtifact(
        selected_threshold=0.15, probability_source="calibrated", scenario="Balanced",
        fp_cost=1.0, fn_cost=1.0, expected_cost=10.0, precision=0.4, recall=0.4, f1=0.4,
        acceptance_rate=0.85, rejection_rate=0.15, tp=2, fp=3, fn=3, tn=17,
        model_version="v", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=25,
    )
    artifact_path = str(tmp_path / "decision_threshold.json")
    artifact.save(artifact_path)

    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.34))  # calibrated -> 0.17
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: _FakeCalibrator())
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "DECISION_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(predict_module, "_decision_artifact_checked", False)
    monkeypatch.setattr(predict_module, "_decision_artifact_cache", None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    # calibrated PD = 0.17 -> risk_band via risk_scoring's OWN thresholds -> "High"
    # (0.10 <= 0.17 < 0.20), computed entirely independently of the decision
    # threshold (0.15) even though they're numerically close here by coincidence.
    assert result["risk_band"] == rs.compute_risk_band(0.17)
    assert result["decision"] == "Decline"  # 0.17 >= decision threshold 0.15
    # Changing one must not require or imply changing the other.
    assert rs.RISK_BAND_THRESHOLDS != {"decision": artifact.selected_threshold}


# ---- Test 12: missing calibration artifact -> no false calibrated decision --

def test_missing_calibrator_produces_no_false_calibrated_decision(monkeypatch, tmp_path):
    artifact = do.DecisionArtifact(
        selected_threshold=0.2, probability_source="calibrated", scenario="Balanced",
        fp_cost=1.0, fn_cost=1.0, expected_cost=10.0, precision=0.4, recall=0.4, f1=0.4,
        acceptance_rate=0.85, rejection_rate=0.15, tp=2, fp=3, fn=3, tn=17,
        model_version="v", calibration_method="sigmoid", feature_version="v1",
        evaluation_sample_size=25,
    )
    artifact_path = str(tmp_path / "decision_threshold.json")
    artifact.save(artifact_path)

    monkeypatch.setattr(predict_module, "load_model", lambda: _fake_bundle(0.5))
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: None)  # no calibrator
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "DECISION_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(predict_module, "_decision_artifact_checked", False)
    monkeypatch.setattr(predict_module, "_decision_artifact_cache", None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result["calibrated_probability"] is None
    # Must NOT silently apply the calibrated-PD decision artifact's threshold
    # to the raw probability and call it a calibrated decision.
    assert result["decision"] is None
    assert result["decision_probability_source"] == "unavailable"
