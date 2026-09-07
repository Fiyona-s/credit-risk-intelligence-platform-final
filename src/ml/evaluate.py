"""Evaluation metrics appropriate for an imbalanced binary classification problem.

Accuracy alone is misleading here (predicting "no default" for everyone scores
~92%). We report ROC-AUC, PR-AUC (average precision), and precision/recall/F1
at a chosen threshold, plus a confusion matrix.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate_model(pipeline, X_val, y_val, threshold: float = 0.5) -> dict:
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    log.info(f"ROC-AUC: {roc_auc:.4f}")
    log.info(f"PR-AUC (average precision): {pr_auc:.4f}")
    log.info(f"Precision @ {threshold}: {precision:.4f}")
    log.info(f"Recall @ {threshold}: {recall:.4f}")
    log.info(f"F1 @ {threshold}: {f1:.4f}")
    log.info(f"Confusion matrix (rows=actual, cols=predicted):\n{cm}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def compute_roc_pr_curves(pipeline, X_val, y_val) -> dict:
    """Return the raw curve data behind ROC-AUC/PR-AUC, for plotting (e.g. the Model Evaluation UI tab).

    Kept separate from evaluate_model() so the CLI path (python -m src.ml.evaluate)
    isn't forced to compute curve arrays it doesn't use, while the UI can call this
    without duplicating the predict_proba() call or the metric logic itself.
    """
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_proba)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, y_proba)
    return {
        "y_proba": y_proba,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_thresholds": pr_thresholds,
    }


def print_model_comparison(lightgbm_metrics: dict, baseline_metrics: dict) -> None:
    """Print a small LightGBM vs. logistic-regression-baseline comparison table.

    Backs up the README's "LightGBM chosen over logistic regression" claim with
    an actual measured number, rather than an unsupported assertion.
    """
    log.info("Model comparison (same validation split):")
    log.info(f"{'Model':<28s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")
    log.info(f"{'LightGBM':<28s} {lightgbm_metrics['roc_auc']:>10.4f} {lightgbm_metrics['pr_auc']:>10.4f}")
    log.info(f"{'Logistic Regression (base)':<28s} {baseline_metrics['roc_auc']:>10.4f} {baseline_metrics['pr_auc']:>10.4f}")


def top_feature_importances(pipeline, numeric_cols, top_n: int = 20) -> list:
    """Return the top_n (feature_name, importance) pairs from the trained LightGBM model.

    Uses LightGBM's built-in gain-based importance rather than recomputing SHAP here
    (src.explainability.shap_explainer already covers per-prediction SHAP) — this is
    for a single global "what does the model lean on" view after adding new features.
    """
    from src.data.preprocessor import get_output_feature_names

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_output_feature_names(preprocessor, numeric_cols)
    importances = model.feature_importances_

    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)[:top_n]

    log.info(f"Top {top_n} features by LightGBM gain importance:")
    for rank, (name, imp) in enumerate(pairs, start=1):
        log.info(f"  {rank:2d}. {name:<40s} {imp}")

    return pairs
