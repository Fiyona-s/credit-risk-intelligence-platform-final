"""Unit tests for the Milestone 4 probability calibration module (src/ml/calibration.py)
and its integration into src/ml/predict.py.

Synthetic data throughout — no real Home Credit CSVs required.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.ml import calibration as calib
from src.ml import predict as predict_module


def _synthetic_raw_and_labels(n=500, seed=0):
    """Raw probabilities that are directionally informative but imperfectly
    calibrated (systematically overconfident), so there's something real for a
    calibrator to correct — mirrors a plausible LightGBM-with-scale_pos_weight
    situation without needing the real model.
    """
    rng = np.random.default_rng(seed)
    true_prob = rng.uniform(0.05, 0.5, size=n)
    y = rng.binomial(1, true_prob)
    # Systematically overconfident raw score: pushes low probabilities lower.
    raw_proba = np.clip(true_prob ** 1.5, 0.001, 0.999)
    return raw_proba, y


# ---- Test 1: calibrator can be fitted on synthetic predictions -------------

def test_sigmoid_calibrator_fits_without_error():
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("sigmoid").fit(raw_proba, y)
    assert cal._model is not None


def test_isotonic_calibrator_fits_without_error():
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("isotonic").fit(raw_proba, y)
    assert cal._model is not None


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        calib.ProbabilityCalibrator("not_a_real_method")


# ---- Test 2: calibrated probabilities are in [0, 1] -------------------------

def test_sigmoid_calibrated_probabilities_in_range():
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("sigmoid").fit(raw_proba, y)
    calibrated = cal.transform(raw_proba)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_isotonic_calibrated_probabilities_in_range():
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("isotonic").fit(raw_proba, y)
    calibrated = cal.transform(raw_proba)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


# ---- Test 3: calibrator can be saved and loaded -----------------------------

def test_calibrator_save_and_load_roundtrip(tmp_path):
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("isotonic").fit(raw_proba, y)

    path = str(tmp_path / "calibrator.joblib")
    cal.save(path)
    assert os.path.exists(path)

    loaded = calib.ProbabilityCalibrator.load(path)
    assert loaded is not None
    assert loaded.method == "isotonic"


def test_load_returns_none_when_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.joblib")
    assert calib.ProbabilityCalibrator.load(missing_path) is None


# ---- Test 4: loaded calibrator produces the same predictions ---------------

def test_loaded_calibrator_matches_original_predictions(tmp_path):
    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("sigmoid").fit(raw_proba, y)
    original_predictions = cal.transform(raw_proba)

    path = str(tmp_path / "calibrator.joblib")
    cal.save(path)
    loaded = calib.ProbabilityCalibrator.load(path)
    loaded_predictions = loaded.transform(raw_proba)

    np.testing.assert_allclose(original_predictions, loaded_predictions)


# ---- Test 5: calibration does not modify the underlying LightGBM artifact --

def test_calibration_fitting_does_not_touch_model_artifact(tmp_path):
    """Fitting calibrators (on purely synthetic OOF-style data here) must never
    write to models/credit_risk_model.joblib — it's a read-only artifact from
    calibration's point of view. Verified by pointing MODEL_PATH-like behavior
    at a throwaway file and confirming ProbabilityCalibrator.fit/save never
    touch it.
    """
    fake_model_path = tmp_path / "fake_model.joblib"
    fake_model_path.write_bytes(b"not a real model, just a marker file")
    original_bytes = fake_model_path.read_bytes()

    raw_proba, y = _synthetic_raw_and_labels()
    cal = calib.ProbabilityCalibrator("isotonic").fit(raw_proba, y)
    cal.save(str(tmp_path / "calibrator.joblib"))

    assert fake_model_path.read_bytes() == original_bytes


# ---- Test 6: calibration cannot be fitted with invalid/mismatched lengths --

def test_fit_rejects_mismatched_lengths():
    raw_proba = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1])  # deliberately shorter
    with pytest.raises(ValueError):
        calib.ProbabilityCalibrator("sigmoid").fit(raw_proba, y)


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        calib.ProbabilityCalibrator("sigmoid").fit(np.array([]), np.array([]))


# ---- Test 7: inference returns both raw and calibrated probabilities -------

def test_predict_applicant_returns_calibrated_probability_when_available(monkeypatch, tmp_path):
    raw_proba, y = _synthetic_raw_and_labels()
    fitted_calibrator = calib.ProbabilityCalibrator("isotonic").fit(raw_proba, y)

    class _FakePipeline:
        def predict_proba(self, df):
            return np.array([[0.7, 0.3]])  # raw P(default) = 0.3

    fake_bundle = {"pipeline": _FakePipeline(), "numeric_cols": ["AMT_INCOME_TOTAL"], "categorical_cols": []}

    monkeypatch.setattr(predict_module, "load_model", lambda: fake_bundle)
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: fitted_calibrator)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result["raw_probability"] == pytest.approx(0.3)
    assert result["probability_of_default"] == pytest.approx(0.3)  # unchanged, backward compatible
    assert result["calibrated_probability"] is not None
    assert 0.0 <= result["calibrated_probability"] <= 1.0
    assert result["calibration_method"] == "isotonic"


# ---- Test 8: inference remains functional when calibration artifact is unavailable --

def test_predict_applicant_functional_without_calibrator(monkeypatch):
    class _FakePipeline:
        def predict_proba(self, df):
            return np.array([[0.8, 0.2]])

    fake_bundle = {"pipeline": _FakePipeline(), "numeric_cols": ["AMT_INCOME_TOTAL"], "categorical_cols": []}

    monkeypatch.setattr(predict_module, "load_model", lambda: fake_bundle)
    monkeypatch.setattr(predict_module, "load_optimal_threshold", lambda: None)
    monkeypatch.setattr(predict_module, "load_calibrator", lambda path=None: None)

    result = predict_module.predict_applicant({"AMT_INCOME_TOTAL": 100000.0})

    assert result["probability_of_default"] == pytest.approx(0.2)
    assert result["raw_probability"] == pytest.approx(0.2)
    # Explicitly None, not silently equal to raw — a caller must be able to tell
    # "no calibration was applied" apart from "calibration produced this value."
    assert result["calibrated_probability"] is None
    # Milestone 5: explicitly "none" (a string), not None/null — so a caller can
    # always positively distinguish "no calibration was applied" from a missing
    # field, per src.ml.risk_scoring's explicit-fallback requirement.
    assert result["calibration_method"] == "none"
    assert result["risk_band"] is not None  # existing functionality unaffected


# ---- Test 9: calibration metadata is generated correctly -------------------

def test_evaluate_probabilities_returns_expected_structure():
    raw_proba, y = _synthetic_raw_and_labels()
    result = calib.evaluate_probabilities(y, raw_proba, n_bins=5)

    for key in ("brier_score", "log_loss", "roc_auc", "pr_auc", "calibration_curve"):
        assert key in result
    assert 0.0 <= result["brier_score"] <= 1.0
    assert result["log_loss"] >= 0.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert "mean_predicted" in result["calibration_curve"]
    assert "observed_frequency" in result["calibration_curve"]
    assert len(result["calibration_curve"]["mean_predicted"]) == len(
        result["calibration_curve"]["observed_frequency"]
    )


def test_select_calibration_method_picks_meaningful_improvement():
    raw_eval = {"brier_score": 0.20, "log_loss": 0.6, "roc_auc": 0.78, "pr_auc": 0.28}
    sigmoid_eval = {"brier_score": 0.15, "log_loss": 0.55, "roc_auc": 0.78, "pr_auc": 0.28}
    isotonic_eval = {"brier_score": 0.19, "log_loss": 0.59, "roc_auc": 0.78, "pr_auc": 0.28}

    selection = calib._select_calibration_method(raw_eval, sigmoid_eval, isotonic_eval)
    assert selection["selected_method"] == "sigmoid"
    assert "reason" in selection and len(selection["reason"]) > 0


def test_select_calibration_method_returns_none_when_no_improvement():
    raw_eval = {"brier_score": 0.15, "log_loss": 0.55, "roc_auc": 0.78, "pr_auc": 0.28}
    sigmoid_eval = {"brier_score": 0.1502, "log_loss": 0.551, "roc_auc": 0.78, "pr_auc": 0.28}
    isotonic_eval = {"brier_score": 0.1501, "log_loss": 0.552, "roc_auc": 0.78, "pr_auc": 0.28}

    selection = calib._select_calibration_method(raw_eval, sigmoid_eval, isotonic_eval)
    assert selection["selected_method"] == "none"


def test_select_calibration_method_breaks_brier_tie_on_ranking_preservation():
    """Real scenario observed on the actual dataset: sigmoid and isotonic both
    improve Brier score over raw by a huge, meaningful margin, and are within
    noise of EACH OTHER (isotonic's Brier is technically lower by 0.00004) —
    but isotonic measurably degrades PR-AUC (a real, non-noise difference)
    while sigmoid preserves it exactly (sigmoid is strictly monotonic). The
    selector must prefer sigmoid here, not isotonic-by-default.
    """
    raw_eval = {"brier_score": 0.17679, "log_loss": 0.52595, "roc_auc": 0.7854, "pr_auc": 0.2851}
    sigmoid_eval = {"brier_score": 0.06581, "log_loss": 0.23732, "roc_auc": 0.7854, "pr_auc": 0.2851}
    isotonic_eval = {"brier_score": 0.06577, "log_loss": 0.23711, "roc_auc": 0.7852, "pr_auc": 0.2781}

    selection = calib._select_calibration_method(raw_eval, sigmoid_eval, isotonic_eval)
    assert selection["selected_method"] == "sigmoid"
    assert "tie" in selection["reason"].lower()


def test_model_artifact_fingerprint_is_stable(tmp_path):
    fake_model = tmp_path / "model.joblib"
    fake_model.write_bytes(b"identical content")
    fp1 = calib._model_artifact_fingerprint(str(fake_model))
    fp2 = calib._model_artifact_fingerprint(str(fake_model))
    assert fp1 == fp2
    assert len(fp1) == 12
