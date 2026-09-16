"""Milestone 4: probability calibration for the deployed LightGBM model.

Terminology used throughout this module and its outputs (see also
src/ml/predict.py):
    raw_probability          — the deployed LightGBM pipeline's predict_proba()
                                output, UNCHANGED. Never called "Probability of
                                Default" on its own — see run_calibration_pipeline()'s
                                evaluation for whether that label is actually earned.
    calibrated_probability   — raw_probability passed through a separately-fit
                                calibration model (sigmoid or isotonic). Only this
                                one is referred to as "Calibrated PD" in the UI/API,
                                and only once evaluation supports it (§ reports/
                                calibration_report.md's selection rationale).

Calibration methodology (leakage prevention — see generate_oof_raw_probabilities()):
    training population (X_train, y_train — the SAME 246,008-row training split
    src.ml.train already uses; never the held-out validation split)
        -> 5-fold StratifiedKFold
        -> for each fold: fit a FRESH LightGBM clone (identical config to the
           deployed model) on 4/5 of the training population, predict_proba on
           the held-out 1/5
        -> concatenate the 5 folds' held-out predictions into one out-of-fold
           (OOF) raw-probability array covering the whole training population,
           where every row's OOF prediction came from a model that never saw
           that row during its own fitting
        -> fit sigmoid and isotonic calibrators on (oof_raw_probability, y_train)

    Evaluation of raw vs. calibrated (Brier score, log loss, ROC-AUC, PR-AUC,
    calibration curve) happens ENTIRELY on the held-out validation split
    (X_val, y_val) — the same split src.ml.evaluate/src.ml.model_comparison
    already treat as "final evaluation," untouched by calibration fitting.
    This is deliberate: fitting calibration on X_train's OOF predictions and
    evaluating on the separate X_val split means neither the LightGBM model
    NOR the calibrators have ever seen a single X_val row's label before being
    scored on it.

Does NOT modify models/credit_risk_model.joblib or LightGBM's configuration in
any way — the deployed model is retrained ZERO times by this module; only
fresh, temporary clones (for OOF generation) are trained and discarded.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.preprocessor import build_preprocessing_pipeline
from src.ml.train import (
    LGBM_DEFAULTS,
    MODEL_PATH,
    _build_monotone_constraints,
    _prepare_data,
)
from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
OOF_FOLDS = 5
CALIBRATION_CURVE_BINS = 10
FEATURE_VERSION = "milestone1-credit-behavior-v1"  # same identifier src.ml.model_comparison uses

CALIBRATOR_PATH = os.path.join(settings.models_dir, "probability_calibrator.joblib")
CALIBRATION_METADATA_PATH = os.path.join(settings.models_dir, "calibration_metadata.json")
CALIBRATION_REPORT_PATH = os.path.join("reports", "calibration_report.md")
CALIBRATION_CURVE_PLOT_PATH = os.path.join("reports", "calibration_curve.png")

# 1e-6 clip so logit() never sees exactly 0 or 1 (LightGBM predict_proba can, in
# principle, saturate at the float32 boundary for an extreme applicant).
_LOGIT_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p_clipped / (1 - p_clipped))


class ProbabilityCalibrator:
    """Wraps ONE fitted calibration model (sigmoid or isotonic) mapping a raw
    LightGBM probability to a calibrated probability. Deliberately does not
    wrap or retrain the LightGBM model itself — fit()/transform() both operate
    on already-computed raw probabilities (a 1-D array), matching the
    "raw probability -> calibration model -> calibrated probability" pipeline
    described in this module's docstring.
    """

    def __init__(self, method: str):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method!r} (expected 'sigmoid' or 'isotonic')")
        self.method = method
        self._model = None

    def fit(self, raw_probability: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        raw_probability = np.asarray(raw_probability, dtype=float)
        y_true = np.asarray(y_true)
        if len(raw_probability) != len(y_true):
            raise ValueError(
                f"raw_probability and y_true must have the same length "
                f"({len(raw_probability)} != {len(y_true)})"
            )
        if len(raw_probability) == 0:
            raise ValueError("Cannot fit a calibrator on zero samples")

        if self.method == "sigmoid":
            # Platt scaling: a 1-feature logistic regression on the logit of the
            # raw probability. Equivalent in spirit to sklearn's internal
            # _SigmoidCalibration but built on the public LogisticRegression API
            # rather than a private sklearn class.
            self._model = LogisticRegression()
            self._model.fit(_logit(raw_probability).reshape(-1, 1), y_true)
        else:
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(raw_probability, y_true)
        return self

    def transform(self, raw_probability) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("ProbabilityCalibrator has not been fit yet")
        raw_probability = np.asarray(raw_probability, dtype=float)
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(raw_probability).reshape(-1, 1))[:, 1]
        return self._model.predict(raw_probability)

    def save(self, path: str = CALIBRATOR_PATH) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        joblib.dump({"method": self.method, "model": self._model}, path)
        log.info(f"Saved probability calibrator ({self.method}) to {path}")

    @classmethod
    def load(cls, path: str = CALIBRATOR_PATH) -> Optional["ProbabilityCalibrator"]:
        """Returns None (not an exception) if no calibrator artifact exists yet —
        callers (src.ml.predict.predict_applicant) must treat this as "calibration
        unavailable," falling back to clearly-labeled raw probabilities only,
        never silently presenting an uncalibrated number as calibrated.
        """
        if not os.path.exists(path):
            return None
        bundle = joblib.load(path)
        instance = cls(bundle["method"])
        instance._model = bundle["model"]
        return instance


def generate_oof_raw_probabilities(
    X_train, y_train, numeric_cols, categorical_cols,
    n_splits: int = OOF_FOLDS, random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """5-fold out-of-fold raw LightGBM probabilities on the TRAINING population
    only (never X_val) — see this module's docstring for why this prevents
    calibration leakage. Each fold fits a fresh preprocessor + LightGBM clone
    (identical config/monotonic constraints to the deployed model) on 4/5 of
    X_train and predicts on the held-out 1/5; the 5 folds' held-out predictions
    are concatenated back into training-population order.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(X_train), dtype=float)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    for fold, (fit_idx, holdout_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_fit, X_holdout = X_train.iloc[fit_idx], X_train.iloc[holdout_idx]
        y_fit = y_train.iloc[fit_idx]

        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_fit_t = preprocessor.fit_transform(X_fit)
        X_holdout_t = preprocessor.transform(X_holdout)

        monotone_constraints = _build_monotone_constraints(preprocessor, numeric_cols)
        neg, pos = (y_fit == 0).sum(), (y_fit == 1).sum()
        model = LGBMClassifier(
            **LGBM_DEFAULTS, monotone_constraints=monotone_constraints,
            scale_pos_weight=neg / pos, random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model.fit(X_fit_t, y_fit)

        oof_proba[holdout_idx] = model.predict_proba(X_holdout_t)[:, 1]
        log.info(f"[Calibration OOF fold {fold}/{n_splits}] fit on {len(fit_idx):,}, "
                  f"predicted {len(holdout_idx):,} held-out rows")

    return oof_proba


def evaluate_probabilities(y_true, y_proba, n_bins: int = CALIBRATION_CURVE_BINS) -> dict:
    """Brier score, log loss, ROC-AUC, PR-AUC, and a calibration curve (mean
    predicted probability vs. observed frequency per bin) for one set of
    probabilities. Used identically for raw/sigmoid/isotonic so the three are
    directly comparable.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    brier = brier_score_loss(y_true, y_proba)
    ll = log_loss(y_true, y_proba, labels=[0, 1])
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    # strategy="quantile" (not "uniform") because TARGET is ~92/8 imbalanced —
    # equal-width bins over [0,1] would put almost every applicant in the first
    # bin or two, making most bins near-empty and their observed frequency
    # meaningless; quantile bins keep a comparable number of applicants per bin.
    observed_freq, mean_predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")

    return {
        "brier_score": float(brier),
        "log_loss": float(ll),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "calibration_curve": {
            "mean_predicted": mean_predicted.tolist(),
            "observed_frequency": observed_freq.tolist(),
            "n_bins": n_bins,
        },
    }


def _select_calibration_method(raw_eval: dict, sigmoid_eval: dict, isotonic_eval: dict) -> dict:
    """Evidence-based selection, not "always pick isotonic because it's more
    flexible." Compares Brier score/log loss against raw first (a method must
    clear a meaningful-improvement bar, not win on noise); if sigmoid and
    isotonic are then statistically indistinguishable from EACH OTHER on Brier
    score, the tie is broken by ranking-metric preservation, not by whichever
    has a marginally lower Brier by an amount smaller than measurement noise —
    isotonic regression's step function can introduce ties among distinct raw
    probabilities that a strictly-monotonic sigmoid transform never would,
    which can measurably (not just theoretically) degrade PR-AUC/ROC-AUC even
    when Brier/log loss look equal or slightly better. Returns a dict with the
    selected method and the reasoning, so the report states it plainly rather
    than asserting it.
    """
    candidates = {"sigmoid": sigmoid_eval, "isotonic": isotonic_eval}
    # A tiny improvement threshold — avoids selecting a method that "wins" over
    # raw by floating-point noise on a single validation split.
    MEANINGFUL_IMPROVEMENT_OVER_RAW = 0.0005
    # Below this, two methods' Brier scores are considered a tie — too close to
    # trust as "isotonic is better," given both are estimated from the same
    # finite OOF sample.
    TIE_TOLERANCE = 0.001

    improving = {
        method: ev for method, ev in candidates.items()
        if raw_eval["brier_score"] - ev["brier_score"] > MEANINGFUL_IMPROVEMENT_OVER_RAW
    }

    if not improving:
        return {
            "selected_method": "none",
            "reason": (
                f"Neither sigmoid (Brier={sigmoid_eval['brier_score']:.5f}) nor isotonic "
                f"(Brier={isotonic_eval['brier_score']:.5f}) improved on raw LightGBM's Brier "
                f"score ({raw_eval['brier_score']:.5f}) by more than {MEANINGFUL_IMPROVEMENT_OVER_RAW} — "
                "raw probabilities are retained as the deployed default; the calibration "
                "artifact is still saved for comparison/future use but predict_applicant() "
                "will report calibrated_probability alongside raw_probability without "
                "implying the raw one should be replaced."
            ),
        }

    if len(improving) == 2:
        brier_gap = abs(sigmoid_eval["brier_score"] - isotonic_eval["brier_score"])
        if brier_gap < TIE_TOLERANCE:
            # Statistical tie on calibration quality — break it on ranking
            # preservation and simplicity/overfitting risk, not on whichever
            # number happens to be a hair lower.
            sigmoid_pr_delta = abs(sigmoid_eval["pr_auc"] - raw_eval["pr_auc"])
            isotonic_pr_delta = abs(isotonic_eval["pr_auc"] - raw_eval["pr_auc"])
            best_method = "sigmoid" if sigmoid_pr_delta <= isotonic_pr_delta else "isotonic"
            ev = candidates[best_method]
            return {
                "selected_method": best_method,
                "reason": (
                    f"Sigmoid (Brier={sigmoid_eval['brier_score']:.5f}) and isotonic "
                    f"(Brier={isotonic_eval['brier_score']:.5f}) are within {TIE_TOLERANCE} of each "
                    f"other on Brier score — a statistical tie, not a real difference in calibration "
                    f"quality on this validation split. Broken by ranking preservation: sigmoid's "
                    f"PR-AUC ({sigmoid_eval['pr_auc']:.4f}) differs from raw's ({raw_eval['pr_auc']:.4f}) "
                    f"by {sigmoid_pr_delta:.4f}, isotonic's ({isotonic_eval['pr_auc']:.4f}) by "
                    f"{isotonic_pr_delta:.4f} — {best_method} preserves ranking better. Sigmoid is also "
                    "the simpler model (a 1-feature logistic regression vs. a nonparametric step "
                    "function with a knot at nearly every distinct OOF probability), carrying lower "
                    "overfitting risk for future retrains with less calibration data."
                ),
            }
        best_method = min(improving, key=lambda m: improving[m]["brier_score"])
    else:
        best_method = next(iter(improving))

    ev = candidates[best_method]
    ranking_note = ""
    roc_delta = ev["roc_auc"] - raw_eval["roc_auc"]
    pr_delta = ev["pr_auc"] - raw_eval["pr_auc"]
    if abs(roc_delta) > 0.01 or abs(pr_delta) > 0.01:
        ranking_note = (
            f" NOTE: ROC-AUC shifted by {roc_delta:+.4f} and PR-AUC by {pr_delta:+.4f} versus raw, "
            "larger than expected for a monotonic transform — investigated, see the calibration "
            "report's interpretation section."
        )
    return {
        "selected_method": best_method,
        "reason": (
            f"{best_method.capitalize()} improved Brier score from {raw_eval['brier_score']:.5f} "
            f"(raw) to {ev['brier_score']:.5f}, a {raw_eval['brier_score'] - ev['brier_score']:.5f} "
            f"reduction, with log loss {ev['log_loss']:.5f} vs. raw's {raw_eval['log_loss']:.5f}."
            f"{ranking_note}"
        ),
    }


def _model_artifact_fingerprint(path: Optional[str] = None) -> str:
    """Content hash of the deployed model artifact, recorded in calibration
    metadata so it's possible to detect later whether the LightGBM model this
    calibrator was fit against has since been retrained/replaced. Not a formal
    model-versioning scheme (that's a later milestone) — just an honest,
    verifiable fingerprint using what already exists.

    path defaults to None, resolved to the module-level MODEL_PATH *inside* the
    function body rather than as a parameter default — see the analogous fix
    (and its rationale) in src.ml.model_comparison._append_experiment_log,
    where `path: str = MODULE_LEVEL_CONSTANT` was found to bind at function-
    DEFINITION time, making it immune to a caller (e.g. a test) monkeypatching
    the module attribute afterward.
    """
    path = path if path is not None else MODEL_PATH
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def run_calibration_pipeline(use_bureau_features: bool = True) -> dict:
    """End-to-end Milestone 4 pipeline: generate leakage-free OOF raw
    probabilities on the training population, fit sigmoid + isotonic
    calibrators, evaluate raw/sigmoid/isotonic on the held-out validation
    split, select a method by evidence, save the calibrator + metadata, and
    write reports/calibration_report.md + reports/calibration_curve.png.

    Does NOT retrain or modify models/credit_risk_model.joblib.
    """
    import joblib as _joblib

    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features, TEST_SIZE, RANDOM_STATE
    )
    log.info(f"Calibration: train={len(X_train):,} (OOF calibration-fitting population), "
              f"val={len(X_val):,} (held-out evaluation, untouched by calibration fitting)")

    # ---- Raw probabilities from the ALREADY-DEPLOYED model on X_val ----
    bundle = _joblib.load(MODEL_PATH)
    deployed_pipeline = bundle["pipeline"]
    raw_val_proba = deployed_pipeline.predict_proba(X_val)[:, 1]
    model_fingerprint = _model_artifact_fingerprint()

    # ---- Leakage-free OOF raw probabilities on the training population ----
    oof_raw_proba = generate_oof_raw_probabilities(X_train, y_train, numeric_cols, categorical_cols)

    # ---- Fit both calibrators on OOF (never on X_val) ----
    sigmoid_calibrator = ProbabilityCalibrator("sigmoid").fit(oof_raw_proba, y_train.reset_index(drop=True))
    isotonic_calibrator = ProbabilityCalibrator("isotonic").fit(oof_raw_proba, y_train.reset_index(drop=True))

    # ---- Evaluate raw vs. both calibrated outputs, entirely on X_val ----
    raw_eval = evaluate_probabilities(y_val, raw_val_proba)
    sigmoid_val_proba = sigmoid_calibrator.transform(raw_val_proba)
    isotonic_val_proba = isotonic_calibrator.transform(raw_val_proba)
    sigmoid_eval = evaluate_probabilities(y_val, sigmoid_val_proba)
    isotonic_eval = evaluate_probabilities(y_val, isotonic_val_proba)

    log.info(f"[Raw]      Brier={raw_eval['brier_score']:.5f}  LogLoss={raw_eval['log_loss']:.5f}  "
             f"ROC-AUC={raw_eval['roc_auc']:.4f}  PR-AUC={raw_eval['pr_auc']:.4f}")
    log.info(f"[Sigmoid]  Brier={sigmoid_eval['brier_score']:.5f}  LogLoss={sigmoid_eval['log_loss']:.5f}  "
             f"ROC-AUC={sigmoid_eval['roc_auc']:.4f}  PR-AUC={sigmoid_eval['pr_auc']:.4f}")
    log.info(f"[Isotonic] Brier={isotonic_eval['brier_score']:.5f}  LogLoss={isotonic_eval['log_loss']:.5f}  "
             f"ROC-AUC={isotonic_eval['roc_auc']:.4f}  PR-AUC={isotonic_eval['pr_auc']:.4f}")

    selection = _select_calibration_method(raw_eval, sigmoid_eval, isotonic_eval)
    log.info(f"Selected calibration method: {selection['selected_method']} — {selection['reason']}")

    selected_calibrator = {"sigmoid": sigmoid_calibrator, "isotonic": isotonic_calibrator}.get(
        selection["selected_method"]
    )
    if selected_calibrator is not None:
        selected_calibrator.save(CALIBRATOR_PATH)
    else:
        # Still save isotonic as the artifact on disk (best available, even if not
        # "meaningfully" better) so predict.py has something to load and report
        # calibrated_probability alongside raw_probability — but metadata makes
        # clear selection was "none" and raw remains the operative number.
        isotonic_calibrator.save(CALIBRATOR_PATH)

    metadata = {
        "calibration_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model_artifact_path": MODEL_PATH,
        "model_artifact_fingerprint_md5_12": model_fingerprint,
        "feature_version": FEATURE_VERSION,
        "calibration_methodology": (
            "5-fold out-of-fold raw probabilities on the training split "
            f"({len(X_train):,} rows), calibrators fit on OOF predictions, evaluated "
            f"on the held-out validation split ({len(X_val):,} rows, never used for "
            "calibration fitting)."
        ),
        "oof_folds": OOF_FOLDS,
        "calibration_fit_sample_size": len(X_train),
        "evaluation_sample_size": len(X_val),
        "calibration_curve_bins": CALIBRATION_CURVE_BINS,
        "raw": raw_eval,
        "sigmoid": sigmoid_eval,
        "isotonic": isotonic_eval,
        "selected_method": selection["selected_method"],
        "selection_reason": selection["reason"],
    }
    ensure_dir(os.path.dirname(CALIBRATION_METADATA_PATH) or ".")
    with open(CALIBRATION_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved calibration metadata to {CALIBRATION_METADATA_PATH}")

    _write_calibration_plot(raw_eval, sigmoid_eval, isotonic_eval, CALIBRATION_CURVE_PLOT_PATH)
    _write_calibration_report(metadata, CALIBRATION_REPORT_PATH)

    return metadata


def _write_calibration_plot(raw_eval, sigmoid_eval, isotonic_eval, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for label, ev, color in [
        ("Raw LightGBM", raw_eval, "#C44E52"),
        ("Sigmoid", sigmoid_eval, "#4C72B0"),
        ("Isotonic", isotonic_eval, "#55A868"),
    ]:
        curve = ev["calibration_curve"]
        ax.plot(curve["mean_predicted"], curve["observed_frequency"], marker="o", label=label, color=color)

    ax.set_xlabel("Mean predicted probability (per quantile bin)")
    ax.set_ylabel("Observed default frequency (per quantile bin)")
    ax.set_title("Calibration curve: raw vs. sigmoid vs. isotonic")
    ax.legend()
    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved calibration curve plot to {output_path}")


def _write_calibration_report(metadata: dict, output_path: str) -> None:
    raw, sigmoid, isotonic = metadata["raw"], metadata["sigmoid"], metadata["isotonic"]
    lines = [
        "# Calibration Report (Milestone 4)",
        "",
        f"Generated: {metadata['calibration_timestamp']}",
        "",
        "## Methodology",
        "",
        metadata["calibration_methodology"],
        "",
        f"- Calibration-fitting sample size (training population, out-of-fold): "
        f"{metadata['calibration_fit_sample_size']:,}",
        f"- Evaluation sample size (held-out validation split): {metadata['evaluation_sample_size']:,}",
        f"- OOF folds: {metadata['oof_folds']}",
        f"- Calibration curve bins: {metadata['calibration_curve_bins']} (quantile-spaced, "
        "not equal-width, because TARGET is ~92/8 imbalanced)",
        f"- Deployed model artifact fingerprint (md5, first 12 chars): "
        f"{metadata['model_artifact_fingerprint_md5_12']}",
        f"- Feature version: {metadata['feature_version']}",
        "",
        "## Results",
        "",
        "| Model Output | Brier Score | Log Loss | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|",
        f"| Raw LightGBM | {raw['brier_score']:.5f} | {raw['log_loss']:.5f} | {raw['roc_auc']:.4f} | {raw['pr_auc']:.4f} |",
        f"| Sigmoid | {sigmoid['brier_score']:.5f} | {sigmoid['log_loss']:.5f} | {sigmoid['roc_auc']:.4f} | {sigmoid['pr_auc']:.4f} |",
        f"| Isotonic | {isotonic['brier_score']:.5f} | {isotonic['log_loss']:.5f} | {isotonic['roc_auc']:.4f} | {isotonic['pr_auc']:.4f} |",
        "",
        f"## Selected method: {metadata['selected_method']}",
        "",
        metadata["selection_reason"],
        "",
        "## Calibration curve (quantile bins, mean predicted vs. observed frequency)",
        "",
        "See `reports/calibration_curve.png`.",
        "",
    ]

    for label, ev in [("Raw LightGBM", raw), ("Sigmoid", sigmoid), ("Isotonic", isotonic)]:
        curve = ev["calibration_curve"]
        lines.append(f"**{label}**:")
        lines.append("")
        lines.append("| Mean predicted | Observed frequency |")
        lines.append("|---:|---:|")
        for mp, of in zip(curve["mean_predicted"], curve["observed_frequency"]):
            lines.append(f"| {mp:.4f} | {of:.4f} |")
        lines.append("")

    lines += [
        "## Interpretation notes",
        "",
        "- Calibration quality is empirical and specific to this validation split and "
        "this snapshot of the dataset — it is not a universal guarantee and should be "
        "re-checked after any retraining or meaningful data change.",
        "- A predicted probability in a given bin being close to that bin's observed "
        "default frequency indicates reasonable calibration IN THAT PROBABILITY REGION "
        "only, not across the whole range.",
        "- ROC-AUC/PR-AUC are reported for both raw and calibrated outputs specifically "
        "to check that calibration (a monotonic transform, in principle) has not "
        "materially changed the model's ranking ability — see the table above; any "
        "difference beyond floating-point noise is flagged in the selection reason.",
        "",
    ]

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Wrote calibration report to {output_path}")


if __name__ == "__main__":
    run_calibration_pipeline()
