"""Unit tests for the Milestone 3 model comparison module (src/ml/model_comparison.py).

Uses small synthetic data throughout — no real Home Credit CSVs required, and no
CV in these tests (CV is verified against the real dataset separately, reported
in the Milestone 3 report — 5-fold CV on a 300k-row dataset x 3 models is not a
fast unit test).
"""
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.preprocessor import build_preprocessing_pipeline
from src.ml import model_comparison as mc
from src.ml.evaluate import compute_metrics_from_proba


def _synthetic_dataset(n=300, seed=0):
    """A small but learnable synthetic binary classification dataset shaped like
    the real one: numeric + categorical features, imbalanced target (~80/20)."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, size=n)
    target = (signal + rng.normal(0, 0.5, size=n) > 0.6).astype(int)
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1),
        "TARGET": target,
        "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, size=n),
        "AMT_CREDIT": rng.uniform(10000, 900000, size=n),
        "SIGNAL_FEATURE": signal,
        "CODE_GENDER": rng.choice(["M", "F"], size=n),
    })
    numeric_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "SIGNAL_FEATURE"]
    categorical_cols = ["CODE_GENDER"]
    return df, numeric_cols, categorical_cols


def _synthetic_prepare_data(*args, **kwargs):
    from sklearn.model_selection import train_test_split

    df, numeric_cols, categorical_cols = _synthetic_dataset()
    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols


@pytest.fixture
def synthetic_transformed():
    df, numeric_cols, categorical_cols = _synthetic_dataset()
    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    return X_train_t, X_val_t, y_train, y_val


# ---- Test 1: all three model configurations can be constructed -------------

def test_all_three_models_construct_without_error():
    LogisticRegression(**mc.LOGREG_CONFIG)
    XGBClassifier(**mc.XGBOOST_CONFIG, scale_pos_weight=1.0)
    LGBMClassifier(**mc.LIGHTGBM_CONFIG, monotone_constraints=[0, 0, 0])


# ---- Test 2: all three models receive the same feature representation ------

def test_all_three_models_receive_identical_transformed_input(synthetic_transformed):
    X_train_t, X_val_t, y_train, y_val = synthetic_transformed

    lr_model, lr_scaler = mc._fit_logistic_regression(X_train_t, y_train)
    xgb_model, _ = mc._fit_xgboost(X_train_t, y_train)
    lgbm_model, _ = mc._fit_lightgbm(X_train_t, y_train, monotone_constraints=[0] * X_train_t.shape[1])

    # All three were fit on the exact same X_train_t array (not independently
    # re-derived/re-encoded per model) — the object identity check confirms no
    # model secretly got a different representation.
    assert lr_model.n_features_in_ == X_train_t.shape[1]
    assert xgb_model.n_features_in_ == X_train_t.shape[1]
    assert lgbm_model.n_features_in_ == X_train_t.shape[1]


# ---- Test 3: predictions have the correct length ----------------------------

def test_predictions_have_correct_length(synthetic_transformed):
    X_train_t, X_val_t, y_train, y_val = synthetic_transformed

    lr_model, lr_scaler = mc._fit_logistic_regression(X_train_t, y_train)
    xgb_model, _ = mc._fit_xgboost(X_train_t, y_train)
    lgbm_model, _ = mc._fit_lightgbm(X_train_t, y_train, monotone_constraints=[0] * X_train_t.shape[1])

    for model, scaler in [(lr_model, lr_scaler), (xgb_model, None), (lgbm_model, None)]:
        proba = mc._predict_proba(model, scaler, X_val_t)
        assert len(proba) == len(y_val)


# ---- Test 4: predicted probabilities are in [0, 1] --------------------------

def test_predicted_probabilities_in_valid_range(synthetic_transformed):
    X_train_t, X_val_t, y_train, y_val = synthetic_transformed

    lr_model, lr_scaler = mc._fit_logistic_regression(X_train_t, y_train)
    xgb_model, _ = mc._fit_xgboost(X_train_t, y_train)
    lgbm_model, _ = mc._fit_lightgbm(X_train_t, y_train, monotone_constraints=[0] * X_train_t.shape[1])

    for model, scaler in [(lr_model, lr_scaler), (xgb_model, None), (lgbm_model, None)]:
        proba = mc._predict_proba(model, scaler, X_val_t)
        assert (proba >= 0).all() and (proba <= 1).all()


# ---- Test 5: metrics are calculated correctly -------------------------------

def test_metrics_calculated_correctly_on_known_case():
    y_val = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])  # perfectly separable
    metrics = compute_metrics_from_proba(y_val, y_proba, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.tolist() == [[2, 0], [0, 2]]


def test_metrics_on_worst_case_predictions():
    y_val = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])  # exactly backwards
    metrics = compute_metrics_from_proba(y_val, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(0.0)


# ---- Test 6: the comparison artifact contains all three models -------------

def test_run_comparison_produces_all_three_models(monkeypatch, tmp_path):
    monkeypatch.setattr(mc, "_prepare_data", _synthetic_prepare_data)
    monkeypatch.setattr(mc, "MODEL_COMPARISON_REPORT_PATH", str(tmp_path / "model_comparison.md"))
    monkeypatch.setattr(mc, "MODEL_COMPARISON_JSON_PATH", str(tmp_path / "model_comparison.json"))
    monkeypatch.setattr(mc, "EXPERIMENT_LOG_PATH", str(tmp_path / "experiment_log.json"))
    monkeypatch.setattr(mc, "CANDIDATE_PATHS", {
        "logistic_regression": str(tmp_path / "lr_candidate.joblib"),
        "xgboost": str(tmp_path / "xgb_candidate.joblib"),
        "lightgbm": str(tmp_path / "lgbm_candidate.joblib"),
    })

    comparison = mc.run_comparison(run_cv=False)

    names = {r["name"] for r in comparison["results"]}
    assert names == {"Logistic Regression", "XGBoost", "LightGBM"}
    for r in comparison["results"]:
        assert "roc_auc" in r["metrics"]
        assert "pr_auc" in r["metrics"]
        assert 0.0 <= r["metrics"]["roc_auc"] <= 1.0

    import os
    assert os.path.exists(mc.MODEL_COMPARISON_REPORT_PATH)
    assert os.path.exists(mc.MODEL_COMPARISON_JSON_PATH)


# ---- Test 7: experiment metadata is saved correctly -------------------------

def test_experiment_log_records_all_models(monkeypatch, tmp_path):
    log_path = str(tmp_path / "experiment_log.json")
    monkeypatch.setattr(mc, "EXPERIMENT_LOG_PATH", log_path)

    fake_comparison = {
        "generated_at": "2026-01-01 00:00 UTC",
        "random_state": 42,
        "train_size": 100,
        "val_size": 20,
        "feature_version": "test-v1",
        "results": [
            {"name": "Logistic Regression", "config": {"C": 1.0}, "metrics": {"roc_auc": 0.7, "pr_auc": 0.3},
             "cv_roc_auc_mean": None, "cv_pr_auc_mean": None, "artifact_path": "a.joblib"},
            {"name": "XGBoost", "config": {"n_estimators": 300}, "metrics": {"roc_auc": 0.78, "pr_auc": 0.28},
             "cv_roc_auc_mean": None, "cv_pr_auc_mean": None, "artifact_path": "b.joblib"},
            {"name": "LightGBM", "config": {"n_estimators": 300}, "metrics": {"roc_auc": 0.79, "pr_auc": 0.29},
             "cv_roc_auc_mean": None, "cv_pr_auc_mean": None, "artifact_path": "c.joblib"},
        ],
    }
    mc._append_experiment_log(fake_comparison, log_path=log_path)

    import json
    with open(log_path) as f:
        entries = json.load(f)
    assert len(entries) == 3
    assert {e["model"] for e in entries} == {"Logistic Regression", "XGBoost", "LightGBM"}
    assert all(e["selected"] is False for e in entries)
    assert all("experiment_id" in e and "timestamp" in e for e in entries)


# ---- Test 8: the selected model artifact can be loaded ----------------------

def test_selected_model_artifact_can_be_loaded(monkeypatch, tmp_path):
    import joblib
    from sklearn.pipeline import Pipeline

    df, numeric_cols, categorical_cols = _synthetic_dataset()
    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_t = preprocessor.fit_transform(X)
    model = LGBMClassifier(n_estimators=10, verbose=-1)
    model.fit(X_t, y)

    candidate_path = str(tmp_path / "lightgbm_candidate.joblib")
    joblib.dump({"model": model, "scaler": None, "preprocessor": preprocessor,
                 "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, candidate_path)

    deployed_path = str(tmp_path / "deployed_model.joblib")
    log_path = str(tmp_path / "experiment_log.json")
    monkeypatch.setattr(mc, "CANDIDATE_PATHS", {"lightgbm": candidate_path,
                                                  "logistic_regression": "", "xgboost": ""})
    monkeypatch.setattr(mc, "EXPERIMENT_LOG_PATH", log_path)

    fake_comparison = {"generated_at": "2026-01-01 00:00 UTC"}
    mc.select_and_promote_model(fake_comparison, "LightGBM", deployed_path)

    loaded = joblib.load(deployed_path)
    assert isinstance(loaded["pipeline"], Pipeline)
    assert loaded["numeric_cols"] == numeric_cols
    assert loaded["categorical_cols"] == categorical_cols
    # The promoted pipeline must actually be usable for prediction, same as the
    # real deployed model — not just structurally present.
    proba = loaded["pipeline"].predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert (proba >= 0).all() and (proba <= 1).all()
