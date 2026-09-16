"""Milestone 3: a clean, reproducible comparison of Logistic Regression, XGBoost,
and LightGBM on identical data/split/preprocessing/target/evaluation — NOT a
hyperparameter search. Every model here is fit ONCE on a fixed configuration
(recorded in *_CONFIG below) and evaluated on the same held-out validation split.

Design choices, so the comparison is actually fair (see the Milestone 3 report
for the full writeup):
  - ONE call to src.ml.train._prepare_data() produces the split; ALL THREE models
    see the exact same X_train/X_val/y_train/y_val — not three independently
    reconstructed (if deterministic) splits.
  - ONE ColumnTransformer (src.data.preprocessor.build_preprocessing_pipeline) is
    fit once on X_train and reused (via .transform(), never re-fit) for every
    model's encoded input — Logistic Regression additionally needs its own
    StandardScaler on top (tree models don't), which is the ONE necessary
    preprocessing difference and is called out explicitly wherever it applies.
  - Every model gets an imbalance treatment that is the natural equivalent for
    its algorithm family (class_weight="balanced" for Logistic Regression,
    scale_pos_weight for XGBoost/LightGBM, computed identically as
    neg_count/pos_count on the SAME y_train for both) — not "one model tuned for
    imbalance, one left at defaults."
  - LightGBM keeps the existing deployed monotonic constraints on
    AMT_INCOME_TOTAL/EXT_SOURCE_1/2/3 — not removed to chase a metric.
  - No hyperparameter search anywhere in this module. Every *_CONFIG dict below
    is a fixed, recorded configuration.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.preprocessor import build_preprocessing_pipeline, get_output_feature_names
from src.ml.evaluate import compute_metrics_from_proba
from src.ml.train import (
    LGBM_DEFAULTS,
    MONOTONIC_CONSTRAINTS_MAP,
    _build_monotone_constraints,
    _prepare_data,
)
from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FEATURE_VERSION = "milestone1-credit-behavior-v1"  # bumped when src/data/feature_registry.py's
                                                     # feature set changes; recorded per experiment.

MODEL_COMPARISON_REPORT_PATH = os.path.join("reports", "model_comparison.md")
MODEL_COMPARISON_JSON_PATH = os.path.join("reports", "model_comparison.json")
EXPERIMENT_LOG_PATH = os.path.join("reports", "experiment_log.json")

CANDIDATE_PATHS = {
    "logistic_regression": os.path.join(settings.models_dir, "logistic_regression_candidate.joblib"),
    "xgboost": os.path.join(settings.models_dir, "xgboost_candidate.joblib"),
    "lightgbm": os.path.join(settings.models_dir, "lightgbm_candidate.joblib"),
}

# Fixed, recorded configurations — no search over these.
LOGREG_CONFIG = dict(
    penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
    class_weight="balanced", random_state=RANDOM_STATE,
)
XGBOOST_CONFIG = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, reg_alpha=0.0,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
)
LIGHTGBM_CONFIG = dict(**LGBM_DEFAULTS, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


@dataclass
class ModelResult:
    name: str
    config: dict
    imbalance_strategy: str
    metrics: dict  # roc_auc, pr_auc, precision, recall, f1, confusion_matrix (as list of lists), threshold
    cv_roc_auc_mean: Optional[float] = None
    cv_roc_auc_std: Optional[float] = None
    cv_pr_auc_mean: Optional[float] = None
    cv_pr_auc_std: Optional[float] = None
    cv_scores: Optional[dict] = None
    fit_seconds: float = 0.0
    artifact_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _scale_pos_weight(y_train) -> float:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    return neg / pos


def _fit_logistic_regression(X_train_t, y_train):
    """Logistic Regression needs scaled inputs (the ColumnTransformer's raw numeric
    output has wildly different scales — AMT_CREDIT in the hundreds of thousands vs.
    ratio features near 0-1 — which without scaling causes lbfgs to fail to converge).
    Trees are scale-invariant and don't need this, so this StandardScaler is specific
    to this one model, applied on top of the SAME ColumnTransformer output every
    model receives, not a replacement for it. class_weight="balanced" is Logistic
    Regression's native equivalent of scale_pos_weight for imbalance handling.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_t)
    model = LogisticRegression(**LOGREG_CONFIG)
    model.fit(X_train_s, y_train)
    return model, scaler


def _fit_xgboost(X_train_t, y_train):
    model = XGBClassifier(**XGBOOST_CONFIG, scale_pos_weight=_scale_pos_weight(y_train))
    model.fit(X_train_t, y_train)
    return model, None


def _fit_lightgbm(X_train_t, y_train, monotone_constraints):
    model = LGBMClassifier(
        **LIGHTGBM_CONFIG, monotone_constraints=monotone_constraints,
        scale_pos_weight=_scale_pos_weight(y_train),
    )
    model.fit(X_train_t, y_train)
    return model, None


def _predict_proba(model, scaler, X_t) -> np.ndarray:
    X_input = scaler.transform(X_t) if scaler is not None else X_t
    return model.predict_proba(X_input)[:, 1]


def _cross_validate_model(
    fit_fn: Callable, X: pd.DataFrame, y: pd.Series,
    numeric_cols: List[str], categorical_cols: List[str],
    n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE,
) -> dict:
    """Stratified K-fold CV, re-fitting BOTH the ColumnTransformer and the model
    inside each fold — preprocessing (imputation medians, one-hot categories) must
    be learned from that fold's training rows only, never from its held-out rows,
    or the fold's score would be optimistic. fit_fn(X_train_t, y_train) -> (model, scaler)
    so the same helper works for Logistic Regression (needs a scaler) and the two
    tree models (don't).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_scores, pr_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_va_t = preprocessor.transform(X_va)

        model, scaler = fit_fn(X_tr_t, y_tr, preprocessor)
        y_proba = _predict_proba(model, scaler, X_va_t)

        roc_scores.append(float(compute_metrics_from_proba(y_va, y_proba)["roc_auc"]))
        pr_scores.append(float(compute_metrics_from_proba(y_va, y_proba)["pr_auc"]))
        log.info(f"  [CV fold {fold}/{n_splits}] ROC-AUC={roc_scores[-1]:.4f}  PR-AUC={pr_scores[-1]:.4f}")

    return {
        "roc_auc_mean": float(np.mean(roc_scores)), "roc_auc_std": float(np.std(roc_scores)),
        "pr_auc_mean": float(np.mean(pr_scores)), "pr_auc_std": float(np.std(pr_scores)),
        "roc_scores": roc_scores, "pr_scores": pr_scores,
    }


def run_comparison(
    use_bureau_features: bool = True, run_cv: bool = True,
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE,
) -> dict:
    """Runs the full Milestone 3 protocol: one shared split, one shared
    preprocessor fit, three models fit under matched imbalance handling,
    evaluated identically, optionally 5-fold CV per model, candidate artifacts
    saved separately (never overwriting the deployed model), and a
    reports/model_comparison.md + .json + reports/experiment_log.json written.
    Does NOT select or deploy a model — see select_and_promote_model() for that,
    a deliberately separate, explicit step.
    """
    log.info(f"Model comparison: random_state={random_state}, test_size={test_size}, "
              f"use_bureau_features={use_bureau_features}")

    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )
    feature_count = len(numeric_cols) + len(categorical_cols)
    log.info(f"Train: {X_train.shape}, Val: {X_val.shape}, raw feature count (pre-encoding): {feature_count}")

    # ONE preprocessor, fit ONCE on X_train, shared by every model below —
    # this is what makes "same preprocessing" true rather than merely likely.
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    encoded_feature_count = X_train_t.shape[1]
    monotone_constraints = _build_monotone_constraints(preprocessor, numeric_cols)

    results: List[ModelResult] = []

    # ---- Logistic Regression ----
    t0 = time.time()
    lr_model, lr_scaler = _fit_logistic_regression(X_train_t, y_train)
    lr_fit_seconds = time.time() - t0
    lr_proba = _predict_proba(lr_model, lr_scaler, X_val_t)
    lr_metrics = compute_metrics_from_proba(y_val, lr_proba)
    log.info(f"[Logistic Regression] ROC-AUC={lr_metrics['roc_auc']:.4f}  PR-AUC={lr_metrics['pr_auc']:.4f}")
    results.append(ModelResult(
        name="Logistic Regression", config=LOGREG_CONFIG,
        imbalance_strategy="class_weight=balanced",
        metrics=_serializable_metrics(lr_metrics), fit_seconds=lr_fit_seconds,
    ))

    # ---- XGBoost ----
    t0 = time.time()
    xgb_model, _ = _fit_xgboost(X_train_t, y_train)
    xgb_fit_seconds = time.time() - t0
    xgb_proba = _predict_proba(xgb_model, None, X_val_t)
    xgb_metrics = compute_metrics_from_proba(y_val, xgb_proba)
    log.info(f"[XGBoost] ROC-AUC={xgb_metrics['roc_auc']:.4f}  PR-AUC={xgb_metrics['pr_auc']:.4f}")
    xgb_scale_pos_weight = _scale_pos_weight(y_train)
    results.append(ModelResult(
        name="XGBoost", config={**XGBOOST_CONFIG, "scale_pos_weight": xgb_scale_pos_weight},
        imbalance_strategy=f"scale_pos_weight={xgb_scale_pos_weight:.2f}",
        metrics=_serializable_metrics(xgb_metrics), fit_seconds=xgb_fit_seconds,
    ))

    # ---- LightGBM (existing deployed configuration, monotonic constraints kept) ----
    t0 = time.time()
    lgbm_model, _ = _fit_lightgbm(X_train_t, y_train, monotone_constraints)
    lgbm_fit_seconds = time.time() - t0
    lgbm_proba = _predict_proba(lgbm_model, None, X_val_t)
    lgbm_metrics = compute_metrics_from_proba(y_val, lgbm_proba)
    log.info(f"[LightGBM] ROC-AUC={lgbm_metrics['roc_auc']:.4f}  PR-AUC={lgbm_metrics['pr_auc']:.4f}")
    lgbm_scale_pos_weight = _scale_pos_weight(y_train)
    n_constrained = sum(1 for c in monotone_constraints if c != 0)
    results.append(ModelResult(
        name="LightGBM", config={**LIGHTGBM_CONFIG, "scale_pos_weight": lgbm_scale_pos_weight,
                                   "monotonic_constraints": MONOTONIC_CONSTRAINTS_MAP},
        imbalance_strategy=f"scale_pos_weight={lgbm_scale_pos_weight:.2f}",
        metrics=_serializable_metrics(lgbm_metrics), fit_seconds=lgbm_fit_seconds,
    ))
    log.info(f"LightGBM: monotonic constraints applied to {n_constrained}/{len(monotone_constraints)} "
             "encoded features (unchanged from the deployed configuration).")

    # ---- Optional 5-fold stratified CV, same methodology for all three ----
    # CV runs over the full applicant population (train split + validation split
    # concatenated back together) — the same set of rows src.ml.train.cross_validate()
    # already uses for its own 5-fold CV, just parameterized here per model. This is
    # separate from, and does not touch, the primary train/validation split's own
    # held-out evaluation above.
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    if run_cv:
        log.info("Running 5-fold stratified CV for Logistic Regression...")
        lr_cv = _cross_validate_model(
            lambda Xt, yt, prep: _fit_logistic_regression(Xt, yt),
            X_full, y_full,
            numeric_cols, categorical_cols, random_state=random_state,
        )
        results[0].cv_roc_auc_mean, results[0].cv_roc_auc_std = lr_cv["roc_auc_mean"], lr_cv["roc_auc_std"]
        results[0].cv_pr_auc_mean, results[0].cv_pr_auc_std = lr_cv["pr_auc_mean"], lr_cv["pr_auc_std"]
        results[0].cv_scores = {"roc_scores": lr_cv["roc_scores"], "pr_scores": lr_cv["pr_scores"]}

        log.info("Running 5-fold stratified CV for XGBoost...")
        xgb_cv = _cross_validate_model(
            lambda Xt, yt, prep: _fit_xgboost(Xt, yt),
            X_full, y_full,
            numeric_cols, categorical_cols, random_state=random_state,
        )
        results[1].cv_roc_auc_mean, results[1].cv_roc_auc_std = xgb_cv["roc_auc_mean"], xgb_cv["roc_auc_std"]
        results[1].cv_pr_auc_mean, results[1].cv_pr_auc_std = xgb_cv["pr_auc_mean"], xgb_cv["pr_auc_std"]
        results[1].cv_scores = {"roc_scores": xgb_cv["roc_scores"], "pr_scores": xgb_cv["pr_scores"]}

        log.info("Running 5-fold stratified CV for LightGBM...")
        lgbm_cv = _cross_validate_model(
            lambda Xt, yt, prep: _fit_lightgbm(Xt, yt, _build_monotone_constraints(prep, numeric_cols)),
            X_full, y_full,
            numeric_cols, categorical_cols, random_state=random_state,
        )
        results[2].cv_roc_auc_mean, results[2].cv_roc_auc_std = lgbm_cv["roc_auc_mean"], lgbm_cv["roc_auc_std"]
        results[2].cv_pr_auc_mean, results[2].cv_pr_auc_std = lgbm_cv["pr_auc_mean"], lgbm_cv["pr_auc_std"]
        results[2].cv_scores = {"roc_scores": lgbm_cv["roc_scores"], "pr_scores": lgbm_cv["pr_scores"]}

    # ---- Save each candidate artifact separately (never touches the deployed model) ----
    ensure_dir(settings.models_dir)
    bundles = {
        "logistic_regression": {"model": lr_model, "scaler": lr_scaler, "preprocessor": preprocessor,
                                  "numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
        "xgboost": {"model": xgb_model, "scaler": None, "preprocessor": preprocessor,
                     "numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
        "lightgbm": {"model": lgbm_model, "scaler": None, "preprocessor": preprocessor,
                      "numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
    }
    for key, result in zip(["logistic_regression", "xgboost", "lightgbm"], results):
        path = CANDIDATE_PATHS[key]
        joblib.dump(bundles[key], path)
        result.artifact_path = path
        log.info(f"Saved candidate artifact: {path}")

    # ---- Feature importance sanity check (tree models only) ----
    feature_names = get_output_feature_names(preprocessor, numeric_cols)
    xgb_importance = _top_importances(xgb_model, feature_names, top_n=15)
    lgbm_importance = _top_importances(lgbm_model, feature_names, top_n=15)

    comparison = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "random_state": random_state,
        "test_size": test_size,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "raw_feature_count": feature_count,
        "encoded_feature_count": int(encoded_feature_count),
        "feature_version": FEATURE_VERSION,
        "results": [r.to_dict() for r in results],
        "xgboost_top_importance": xgb_importance,
        "lightgbm_top_importance": lgbm_importance,
    }

    ensure_dir("reports")
    with open(MODEL_COMPARISON_JSON_PATH, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    _write_markdown_report(comparison)
    _append_experiment_log(comparison)

    log.info(f"Wrote {MODEL_COMPARISON_REPORT_PATH}, {MODEL_COMPARISON_JSON_PATH}, {EXPERIMENT_LOG_PATH}")
    return comparison


def _serializable_metrics(metrics: dict) -> dict:
    out = dict(metrics)
    out["confusion_matrix"] = np.asarray(metrics["confusion_matrix"]).tolist()
    for k in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        out[k] = float(out[k])
    return out


def _top_importances(model, feature_names: List[str], top_n: int = 15) -> List[dict]:
    if not hasattr(model, "feature_importances_"):
        return []
    pairs = sorted(zip(feature_names, model.feature_importances_), key=lambda p: p[1], reverse=True)[:top_n]
    return [{"feature": name, "importance": float(imp)} for name, imp in pairs]


def _write_markdown_report(comparison: dict) -> None:
    lines = [
        "# Model Comparison Report (Milestone 3)",
        "",
        f"Generated: {comparison['generated_at']}",
        "",
        "**Protocol**: one fixed stratified train/validation split "
        f"(random_state={comparison['random_state']}, test_size={comparison['test_size']}), "
        "one shared preprocessing fit (ColumnTransformer fit once on the training split, "
        "reused via .transform() for every model), identical target, identical evaluation "
        "at threshold=0.5. Logistic Regression additionally gets a StandardScaler on top of "
        "the shared preprocessor's output (required for solver convergence; trees don't need "
        "it). No hyperparameter search — every configuration below is fixed and recorded.",
        "",
        f"- Train size: {comparison['train_size']:,}",
        f"- Validation size: {comparison['val_size']:,}",
        f"- Raw feature count (pre-encoding): {comparison['raw_feature_count']}",
        f"- Encoded feature count: {comparison['encoded_feature_count']}",
        f"- Feature version: {comparison['feature_version']}",
        "",
        "## Primary comparison (single split, threshold=0.5)",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison["results"]:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )
    lines.append("")

    if comparison["results"][0].get("cv_roc_auc_mean") is not None:
        lines += [
            "## 5-fold stratified cross-validation",
            "",
            "| Model | CV ROC-AUC | CV PR-AUC |",
            "|---|---:|---:|",
        ]
        for r in comparison["results"]:
            lines.append(
                f"| {r['name']} | {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | "
                f"{r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.4f} |"
            )
        lines.append("")

    lines += ["## Confusion matrices (threshold=0.5)", ""]
    for r in comparison["results"]:
        cm = r["metrics"]["confusion_matrix"]
        lines += [
            f"**{r['name']}** (rows=actual, cols=predicted):",
            "```",
            f"              Pred: No Default   Pred: Default",
            f"Actual: No Default   {cm[0][0]:>7,}          {cm[0][1]:>7,}",
            f"Actual: Default      {cm[1][0]:>7,}          {cm[1][1]:>7,}",
            "```",
            "",
        ]

    lines += ["## Fit time", "", "| Model | Fit time (s) |", "|---|---:|"]
    for r in comparison["results"]:
        lines.append(f"| {r['name']} | {r['fit_seconds']:.1f} |")
    lines.append("")

    if comparison.get("xgboost_top_importance") or comparison.get("lightgbm_top_importance"):
        lines += [
            "## Feature importance (tree models only — sanity check, not SHAP)",
            "",
            "**Not directly comparable between models**: LightGBM's sklearn "
            "`feature_importances_` defaults to split-count (how many times a feature "
            "was used to split, an unbounded integer), while XGBoost's defaults to "
            "gain-based fractions that sum to 1 across all features. The two lists "
            "below are each internally ranked correctly, but the raw numbers are on "
            "different scales — do not read '482 vs 0.03' as XGBoost caring 16,000x "
            "less about its top feature.",
            "",
        ]
        for model_key, label in [("xgboost_top_importance", "XGBoost"), ("lightgbm_top_importance", "LightGBM")]:
            if not comparison.get(model_key):
                continue
            lines.append(f"**{label}** top {len(comparison[model_key])}:")
            lines.append("")
            for i, entry in enumerate(comparison[model_key], start=1):
                lines.append(f"{i}. `{entry['feature']}` — {entry['importance']:.4f}")
            lines.append("")

    output_path = MODEL_COMPARISON_REPORT_PATH
    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _append_experiment_log(comparison: dict, log_path: Optional[str] = None) -> None:
    """Append one entry per model in this comparison run to a running JSON log —
    lightweight experiment tracking (no MLflow), per the Milestone 3 spec.

    log_path defaults to None (resolved to the module-level EXPERIMENT_LOG_PATH
    *inside* the function body, not as a parameter default) deliberately —
    `log_path: str = EXPERIMENT_LOG_PATH` would bind to whatever EXPERIMENT_LOG_PATH
    was at function-DEFINITION time, so a caller (e.g. a test) monkeypatching
    module.EXPERIMENT_LOG_PATH afterward would have no effect on the default
    actually used. Caught by a test writing to the real reports/experiment_log.json
    despite patching module.EXPERIMENT_LOG_PATH, because run_comparison() calls this
    without an explicit log_path.
    """
    log_path = log_path if log_path is not None else EXPERIMENT_LOG_PATH
    entries = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            try:
                entries = json.load(f)
            except json.JSONDecodeError:
                entries = []

    timestamp = comparison["generated_at"]
    for r in comparison["results"]:
        entries.append({
            "experiment_id": f"{timestamp}-{r['name'].lower().replace(' ', '_')}",
            "timestamp": timestamp,
            "model": r["name"],
            "config": r["config"],
            "feature_version": comparison["feature_version"],
            "random_state": comparison["random_state"],
            "train_size": comparison["train_size"],
            "val_size": comparison["val_size"],
            "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
            "cv_roc_auc_mean": r.get("cv_roc_auc_mean"),
            "cv_pr_auc_mean": r.get("cv_pr_auc_mean"),
            "artifact_path": r["artifact_path"],
            "selected": False,  # flipped to True for the chosen model by select_and_promote_model()
        })

    ensure_dir(os.path.dirname(log_path) or ".")
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def select_and_promote_model(comparison: dict, selected_name: str, deployed_model_path: str) -> None:
    """Explicit, separate promotion step: copies the selected candidate's artifact
    to the deployed model path AND marks it "selected": true in the experiment log.
    Never called automatically by run_comparison() — model selection is a judgment
    call made in the Milestone 3 report, not an automatic "pick the highest number."
    """
    key_by_name = {"Logistic Regression": "logistic_regression", "XGBoost": "xgboost", "LightGBM": "lightgbm"}
    key = key_by_name[selected_name]
    candidate_path = CANDIDATE_PATHS[key]

    bundle = joblib.load(candidate_path)
    # The deployed model's own consumers (predict.py, shap_explainer.py, rule_derivation.py)
    # expect {"pipeline", "numeric_cols", "categorical_cols"} — wrap the candidate's
    # (already-fit) preprocessor + model into that exact shape, same pattern train() uses.
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[("preprocessor", bundle["preprocessor"]), ("model", bundle["model"])])
    ensure_dir(os.path.dirname(deployed_model_path) or ".")
    joblib.dump({
        "pipeline": pipeline,
        "numeric_cols": bundle["numeric_cols"],
        "categorical_cols": bundle["categorical_cols"],
    }, deployed_model_path)
    log.info(f"Promoted {selected_name} ({candidate_path}) to deployed model at {deployed_model_path}")

    if os.path.exists(EXPERIMENT_LOG_PATH):
        with open(EXPERIMENT_LOG_PATH) as f:
            entries = json.load(f)
        for entry in entries:
            entry["selected"] = (entry["model"] == selected_name and entry["timestamp"] == comparison["generated_at"])
        with open(EXPERIMENT_LOG_PATH, "w") as f:
            json.dump(entries, f, indent=2, default=str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cv", action="store_true", help="Skip the 5-fold CV step (faster, single-split only)")
    parser.add_argument("--no-bureau-features", action="store_true")
    args = parser.parse_args()

    run_comparison(use_bureau_features=not args.no_bureau_features, run_cv=not args.no_cv)
