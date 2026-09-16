"""Trains a LightGBM classifier to predict TARGET (probability of default).

Imbalance handling: TARGET is ~92/8 imbalanced (see notebooks/eda.py). We use
LightGBM's built-in `scale_pos_weight` rather than SMOTE oversampling as the
primary strategy — with ~300k rows and 100+ features, SMOTE-generated synthetic
minority samples in high-dimensional one-hot space tend to sit in unrealistic
regions and mainly slow training without improving ROC-AUC/PR-AUC over a
correctly weighted loss. scale_pos_weight = (negative count / positive count)
tells the model to penalize missed defaults proportionally to their rarity,
which is the standard, cheaper first choice for tree-boosting models. SMOTE
remains available (`--smote` flag) for comparison.

The default `python -m src.ml.train` (no flags) stays fast and is the only
path that writes models/credit_risk_model.joblib — the artifact the rest of
the app (predict.py, shap_explainer.py, rule_derivation.py) loads. The four
extra reporting paths below (--baseline, --cv, --tune, --xgboost) are opt-in
and never change what gets saved/deployed:
  --baseline  trains a simple LogisticRegression on the same split, to back
              up the "LightGBM chosen over logistic regression" README claim
              with an actual number instead of an unsupported assertion.
  --cv        5-fold StratifiedKFold cross-validation, reporting mean±std
              ROC-AUC/PR-AUC — honest stability reporting, since a single
              80/20 split's number has no confidence interval on its own.
  --tune      bounded RandomizedSearchCV (n_iter=15, cv=3) over a small
              LightGBM hyperparameter space, scored on PR-AUC given the class
              imbalance. Reports best params found; does not auto-apply them.
  --xgboost   trains an XGBClassifier on the same split with the same
              scale_pos_weight, a second gradient-boosting comparison point
              alongside the LogisticRegression baseline.

--monitor is different: it's a modifier on the default training run (not a
standalone reporting path), saving reports/reference_data.parquet — the raw
training-set snapshot src/monitoring/drift_report.py (and the app's Monitoring
tab) compare a current batch of applicants against to detect data/target drift.

Monotonic constraints: AMT_INCOME_TOTAL and EXT_SOURCE_1/2/3 get a -1
(monotonically decreasing risk) constraint in the default training path —
see the "Model selection & imbalance handling" section of the README for the
fair-lending rationale on why these specific features and not others.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import build_joined_dataset
from src.data.preprocessor import (
    build_preprocessing_pipeline,
    engineer_features,
    get_aggregate_table_for_validation,
    get_feature_columns,
    get_output_feature_names,
)
from src.data.validation import (
    check_aggregate_fill_contamination,
    generate_markdown_report,
    raise_if_failed,
    run_all_checks,
)
from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")
DATA_VALIDATION_REPORT_PATH = os.path.join("reports", "data_validation_report.md")
OPTIMAL_THRESHOLD_PATH = os.path.join(settings.models_dir, "optimal_threshold.json")

# Sweep range/step for cost-sensitive threshold selection (see sweep_cost_sensitive_threshold).
THRESHOLD_SWEEP_MIN = 0.1
THRESHOLD_SWEEP_MAX = 0.9
THRESHOLD_SWEEP_STEP = 0.01

# Applied only to features where the risk direction is genuinely well-justified
# by domain knowledge, not blanket across all features (see README). -1 means
# "higher feature value -> monotonically lower predicted risk of default".
MONOTONIC_CONSTRAINTS_MAP = {
    "AMT_INCOME_TOTAL": -1,
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
}

LGBM_DEFAULTS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)


def _prepare_data(use_bureau_features: bool, test_size: float, random_state: int):
    """Shared load -> engineer -> split step, reused by train(), cross_validate(), tune(), train_baseline().

    Returns df (the full engineered dataset, pre-split) alongside the split
    itself — src.data.validation's checks need the full dataset (applicant-ID
    uniqueness, duplicate rows, TARGET-correlation, registry consistency) as well
    as the split (distribution-shift checks), and this avoids recomputing
    build_joined_dataset()/engineer_features() a second time just for validation.
    """
    df = build_joined_dataset(is_train=True, use_bureau_features=use_bureau_features)
    df = engineer_features(df)

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    numeric_cols, categorical_cols = get_feature_columns(df)
    X = X[numeric_cols + categorical_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols


def run_data_validation(
    df: pd.DataFrame, numeric_cols: list, categorical_cols: list,
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    report_path: str = DATA_VALIDATION_REPORT_PATH,
) -> list:
    """Milestone 2: runs the full src.data.validation check suite before the
    preprocessing pipeline is fit, writes reports/data_validation_report.md, and
    raises DataValidationError (aborting training) if any check FAILs. Warnings
    are logged but do not stop training — see src.data.validation.raise_if_failed.

    Only wired into train() (the path that actually saves/deploys a model) — the
    reporting-only paths (--baseline/--cv/--tune/--xgboost) intentionally stay as
    lightweight as before, consistent with their existing "reporting only, never
    saved" character.
    """
    results = run_all_checks(df, numeric_cols, categorical_cols, X_train, X_val)

    # Aggregate-fill contamination needs the cached aggregate table + the actual
    # train-split applicant IDs — not something run_all_checks() can derive on its
    # own from df/numeric_cols/categorical_cols alone, so it's run here separately
    # and appended to the same results list before reporting/raising.
    aggregate_table, count_col_prefixes = get_aggregate_table_for_validation()
    if not aggregate_table.empty and "SK_ID_CURR" in df.columns:
        train_ids = df.loc[X_train.index, "SK_ID_CURR"] if "SK_ID_CURR" in df.columns else pd.Series(dtype=int)
        results.append(check_aggregate_fill_contamination(aggregate_table, train_ids, count_col_prefixes))

    generate_markdown_report(results, report_path)
    raise_if_failed(results)
    return results


def _build_monotone_constraints(preprocessor, numeric_cols) -> list:
    """Map MONOTONIC_CONSTRAINTS_MAP onto the preprocessor's actual output column order.

    One-hot-encoded categorical columns (which never appear in the map) get 0
    automatically, since they simply won't match any key.
    """
    output_feature_names = get_output_feature_names(preprocessor, numeric_cols)
    return [MONOTONIC_CONSTRAINTS_MAP.get(name, 0) for name in output_feature_names]


def sweep_cost_sensitive_threshold(
    y_val, y_proba, fp_cost: float, fn_cost: float,
    threshold_min: float = THRESHOLD_SWEEP_MIN, threshold_max: float = THRESHOLD_SWEEP_MAX,
    threshold_step: float = THRESHOLD_SWEEP_STEP,
) -> dict:
    """Sweep classification thresholds and find the one minimizing
    total_cost = fp_cost * false_positives + fn_cost * false_negatives on (y_val, y_proba).

    Returns the full per-threshold false_positives/false_negatives arrays (not just
    total_cost at this fp_cost/fn_cost) so a caller — the Risk Prediction tab's cost-ratio
    slider — can recompute total_cost for a *different* cost ratio via pure arithmetic
    on these cached counts, without needing the model or validation data again.
    """
    y_val = np.asarray(y_val)
    y_proba = np.asarray(y_proba)
    thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 4)

    false_positives, false_negatives = [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        false_positives.append(int(((y_pred == 1) & (y_val == 0)).sum()))
        false_negatives.append(int(((y_pred == 0) & (y_val == 1)).sum()))

    false_positives = np.array(false_positives)
    false_negatives = np.array(false_negatives)
    total_cost = fp_cost * false_positives + fn_cost * false_negatives

    best_idx = int(np.argmin(total_cost))
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "min_total_cost": float(total_cost[best_idx]),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "thresholds": thresholds.tolist(),
        "false_positives": false_positives.tolist(),
        "false_negatives": false_negatives.tolist(),
        "total_cost": total_cost.tolist(),
    }


def save_optimal_threshold(result: dict, path: str = OPTIMAL_THRESHOLD_PATH) -> None:
    import json

    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        f"Saved optimal threshold to {path}: {result['optimal_threshold']:.2f} "
        f"(fp_cost={result['fp_cost']}, fn_cost={result['fn_cost']}, "
        f"min_total_cost={result['min_total_cost']:.1f})"
    )


def train(
    use_smote: bool = False,
    use_bureau_features: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    lgbm_params: dict = None,
    save_monitoring_reference: bool = False,
) -> dict:
    lgbm_params = {**LGBM_DEFAULTS, **(lgbm_params or {})}

    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )
    log.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    validation_results = run_data_validation(df, numeric_cols, categorical_cols, X_train, X_val)

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    monotone_constraints = _build_monotone_constraints(preprocessor, numeric_cols)
    n_constrained = sum(1 for c in monotone_constraints if c != 0)
    log.info(f"Applying monotonic constraints to {n_constrained} of {len(monotone_constraints)} features")

    if use_smote:
        log.info("Using SMOTE oversampling on the training split")
        X_train_res, y_train_res = SMOTE(random_state=random_state).fit_resample(X_train_t, y_train)
        model = LGBMClassifier(
            **lgbm_params, monotone_constraints=monotone_constraints,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train_res, y_train_res)
    else:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
        log.info(f"Using scale_pos_weight={scale_pos_weight:.2f} (neg={neg}, pos={pos})")
        model = LGBMClassifier(
            **lgbm_params, monotone_constraints=monotone_constraints,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train_t, y_train)

    # preprocessor and model are both already fit above; wrapping them in a
    # Pipeline here is just for a single object with .predict_proba() /
    # .named_steps["preprocessor"] — Pipeline.fit() is intentionally not
    # called again (sklearn doesn't require it when every step is pre-fit).
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    ensure_dir(settings.models_dir)
    joblib.dump({
        "pipeline": pipeline,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }, MODEL_PATH)
    log.info(f"Saved trained pipeline to {MODEL_PATH}")

    y_proba_val = pipeline.predict_proba(X_val)[:, 1]
    threshold_sweep = sweep_cost_sensitive_threshold(y_val, y_proba_val, settings.fp_cost, settings.fn_cost)
    save_optimal_threshold(threshold_sweep)

    if save_monitoring_reference:
        # The raw (pre-preprocessing) training rows — this is what a "current" batch
        # of new applicants gets compared against in src/monitoring/drift_report.py,
        # so it needs to be in the same raw, unencoded form, not the transformed
        # array the model actually trains on.
        from src.monitoring.drift_report import save_reference_snapshot

        reference_df = X_train.copy()
        reference_df["TARGET"] = y_train.values
        save_reference_snapshot(reference_df)

    return {
        "pipeline": pipeline,
        "X_val": X_val,
        "y_val": y_val,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def train_baseline(use_bureau_features: bool = True, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Simple LogisticRegression baseline on the same split, to back up the

    "LightGBM chosen over logistic regression" claim with an actual number.
    Not tuned, not saved to models/ — reporting only.

    Unlike LightGBM (tree-based, scale-invariant), logistic regression needs
    scaled inputs — the raw preprocessor's numeric output has wildly different
    scales (e.g. AMT_CREDIT in the hundreds of thousands vs. ratio features
    near 0-1), which without scaling causes the lbfgs solver to fail to
    converge and can overflow. A StandardScaler here (applied only for this
    baseline, not the main pipeline other modules load) makes this a fair
    comparison rather than an artificially crippled one.
    """
    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train_t)
    X_val_t = scaler.transform(X_val_t)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(X_train_t, y_train)

    y_proba = model.predict_proba(X_val_t)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    log.info(f"[Baseline: LogisticRegression] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def train_xgboost(use_bureau_features: bool = True, test_size: float = 0.2, random_state: int = 42) -> dict:
    """XGBoostClassifier on the same 80/20 stratified split, with scale_pos_weight matching
    the LightGBM setup (neg/pos ratio on the training split) — a second gradient-boosting
    comparison point alongside the LogisticRegression baseline. Not tuned, not saved to
    models/ — reporting only, exactly like train_baseline().
    """
    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"[XGBoost] Using scale_pos_weight={scale_pos_weight:.2f} (neg={neg}, pos={pos})")

    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train_t, y_train)

    y_proba = model.predict_proba(X_val_t)[:, 1]
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    log.info(f"[XGBoost] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def cross_validate(
    n_splits: int = 5, use_bureau_features: bool = True, random_state: int = 42
) -> dict:
    """5-fold StratifiedKFold CV, reporting mean+-std ROC-AUC/PR-AUC.

    For honest stability reporting only — does NOT save a model. The deployed
    model still comes from a single train() call on the standard 80/20 split.
    """
    df = build_joined_dataset(is_train=True, use_bureau_features=use_bureau_features)
    df = engineer_features(df)
    y = df["TARGET"]
    numeric_cols, categorical_cols = get_feature_columns(df)
    X = df[numeric_cols + categorical_cols]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_scores, pr_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_va_t = preprocessor.transform(X_va)

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        model = LGBMClassifier(
            **LGBM_DEFAULTS, scale_pos_weight=neg / pos,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr_t, y_tr)

        y_proba = model.predict_proba(X_va_t)[:, 1]
        roc = roc_auc_score(y_va, y_proba)
        pr = average_precision_score(y_va, y_proba)
        roc_scores.append(roc)
        pr_scores.append(pr)
        log.info(f"[CV fold {fold}/{n_splits}] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    roc_mean, roc_std = np.mean(roc_scores), np.std(roc_scores)
    pr_mean, pr_std = np.mean(pr_scores), np.std(pr_scores)
    log.info(f"[5-fold CV] ROC-AUC = {roc_mean:.4f} +/- {roc_std:.4f}")
    log.info(f"[5-fold CV] PR-AUC  = {pr_mean:.4f} +/- {pr_std:.4f}")

    return {
        "roc_auc_mean": roc_mean, "roc_auc_std": roc_std,
        "pr_auc_mean": pr_mean, "pr_auc_std": pr_std,
        "roc_scores": roc_scores, "pr_scores": pr_scores,
    }


def tune(
    n_iter: int = 15, cv: int = 3, use_bureau_features: bool = True,
    test_size: float = 0.2, random_state: int = 42,
) -> dict:
    """Bounded RandomizedSearchCV over a small LightGBM hyperparameter space.

    Scored on PR-AUC (average_precision), not accuracy, given the class
    imbalance. Reports the best params found and compares them against
    LGBM_DEFAULTS on the held-out validation set — does not auto-apply
    anything to train()'s defaults.
    """
    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    param_distributions = {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 300, 500, 800],
        "min_child_samples": [10, 20, 50, 100],
    }

    base_model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1, verbose=-1,
    )
    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=n_iter, cv=cv,
        scoring="average_precision", random_state=random_state, n_jobs=-1,
    )
    search.fit(X_train_t, y_train)

    best_model = search.best_estimator_
    y_proba_best = best_model.predict_proba(X_val_t)[:, 1]
    tuned_roc = roc_auc_score(y_val, y_proba_best)
    tuned_pr = average_precision_score(y_val, y_proba_best)

    default_model = LGBMClassifier(
        **LGBM_DEFAULTS, scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    default_model.fit(X_train_t, y_train)
    y_proba_default = default_model.predict_proba(X_val_t)[:, 1]
    default_roc = roc_auc_score(y_val, y_proba_default)
    default_pr = average_precision_score(y_val, y_proba_default)

    roc_delta = tuned_roc - default_roc
    log.info(f"[Tune] Best params: {search.best_params_}")
    log.info(f"[Tune] Best CV PR-AUC (train-side): {search.best_score_:.4f}")
    log.info(f"[Tune] Validation — defaults: ROC-AUC={default_roc:.4f} PR-AUC={default_pr:.4f}")
    log.info(f"[Tune] Validation — tuned:    ROC-AUC={tuned_roc:.4f} PR-AUC={tuned_pr:.4f}")
    log.info(f"[Tune] ROC-AUC delta (tuned - default): {roc_delta:+.4f}")
    if roc_delta < 0.002:
        log.info("[Tune] Improvement is marginal (<0.002 ROC-AUC) — keeping current defaults, not chasing noise.")
    else:
        log.info("[Tune] Improvement exceeds the 0.002 ROC-AUC threshold — consider updating LGBM_DEFAULTS.")

    return {
        "best_params": search.best_params_,
        "default_roc_auc": default_roc, "default_pr_auc": default_pr,
        "tuned_roc_auc": tuned_roc, "tuned_pr_auc": tuned_pr,
        "roc_auc_delta": roc_delta,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smote", action="store_true", help="Use SMOTE instead of scale_pos_weight")
    parser.add_argument(
        "--no-bureau-features", action="store_true",
        help="Disable bureau/previous-application/POS/credit-card/installments aggregate features "
             "(train on application-level features only, for an ablation comparison)",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Train a LogisticRegression baseline (reporting only, not saved) instead of the default LightGBM run",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold stratified cross-validation (reporting only, not saved) instead of the default run",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run a bounded RandomizedSearchCV hyperparameter search (reporting only, not saved) "
             "instead of the default run",
    )
    parser.add_argument(
        "--xgboost", action="store_true",
        help="Train an XGBoostClassifier comparison (reporting only, not saved) instead of the default run",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="After the default training run, also save reports/reference_data.parquet — the raw "
             "training-set snapshot compared against a current batch by src/monitoring/drift_report.py "
             "(and the app's Monitoring tab) to detect data/target drift. Only applies to the default "
             "training run, not --baseline/--cv/--tune/--xgboost.",
    )
    args = parser.parse_args()

    if args.baseline:
        train_baseline(use_bureau_features=not args.no_bureau_features)
    elif args.cv:
        cross_validate(use_bureau_features=not args.no_bureau_features)
    elif args.tune:
        tune(use_bureau_features=not args.no_bureau_features)
    elif args.xgboost:
        train_xgboost(use_bureau_features=not args.no_bureau_features)
    else:
        result = train(
            use_smote=args.smote, use_bureau_features=not args.no_bureau_features,
            save_monitoring_reference=args.monitor,
        )

        from src.ml.evaluate import evaluate_model
        evaluate_model(result["pipeline"], result["X_val"], result["y_val"])
