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
the app (predict.py, shap_explainer.py, rule_derivation.py) loads. The three
extra reporting paths below (--baseline, --cv, --tune) are opt-in and never
change what gets saved/deployed:
  --baseline  trains a simple LogisticRegression on the same split, to back
              up the "LightGBM chosen over logistic regression" README claim
              with an actual number instead of an unsupported assertion.
  --cv        5-fold StratifiedKFold cross-validation, reporting mean±std
              ROC-AUC/PR-AUC — honest stability reporting, since a single
              80/20 split's number has no confidence interval on its own.
  --tune      bounded RandomizedSearchCV (n_iter=15, cv=3) over a small
              LightGBM hyperparameter space, scored on PR-AUC given the class
              imbalance. Reports best params found; does not auto-apply them.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import build_joined_dataset
from src.data.preprocessor import (
    build_preprocessing_pipeline,
    engineer_features,
    get_feature_columns,
    get_output_feature_names,
)
from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")

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
    """Shared load -> engineer -> split step, reused by train(), cross_validate(), tune(), train_baseline()."""
    df = build_joined_dataset(is_train=True, use_bureau_features=use_bureau_features)
    df = engineer_features(df)

    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    numeric_cols, categorical_cols = get_feature_columns(df)
    X = X[numeric_cols + categorical_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val, numeric_cols, categorical_cols


def _build_monotone_constraints(preprocessor, numeric_cols) -> list:
    """Map MONOTONIC_CONSTRAINTS_MAP onto the preprocessor's actual output column order.

    One-hot-encoded categorical columns (which never appear in the map) get 0
    automatically, since they simply won't match any key.
    """
    output_feature_names = get_output_feature_names(preprocessor, numeric_cols)
    return [MONOTONIC_CONSTRAINTS_MAP.get(name, 0) for name in output_feature_names]


def train(
    use_smote: bool = False,
    use_bureau_features: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    lgbm_params: dict = None,
) -> dict:
    lgbm_params = {**LGBM_DEFAULTS, **(lgbm_params or {})}

    X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, test_size, random_state
    )
    log.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

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
    X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
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
    X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
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
    args = parser.parse_args()

    if args.baseline:
        train_baseline(use_bureau_features=not args.no_bureau_features)
    elif args.cv:
        cross_validate(use_bureau_features=not args.no_bureau_features)
    elif args.tune:
        tune(use_bureau_features=not args.no_bureau_features)
    else:
        result = train(use_smote=args.smote, use_bureau_features=not args.no_bureau_features)

        from src.ml.evaluate import evaluate_model
        evaluate_model(result["pipeline"], result["X_val"], result["y_val"])
