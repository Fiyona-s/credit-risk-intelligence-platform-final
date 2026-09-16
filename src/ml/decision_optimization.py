"""Milestone 6: cost-sensitive business decision optimization on CALIBRATED PD.

Problem being fixed: src.ml.train.sweep_cost_sensitive_threshold() (Milestone 1)
sweeps thresholds against the RAW LightGBM probability — the same probability
Milestone 4 showed is severely overconfident (Brier 0.177 raw vs. 0.066
calibrated). A cost-optimal threshold computed on that overconfident scale
(0.67, saved in models/optimal_threshold.json) is not wrong on its own terms,
but it is answering "where's the cost-minimizing cutoff on the RAW scale,"
which is a different question from "where's the cost-minimizing cutoff on a
scale that actually reflects observed default frequency." This module answers
the second question, on calibrated PD, and keeps the two systems clearly
separate (see src.ml.predict for how both surface in predict_applicant()'s
output — the legacy raw-based fields are preserved unchanged, new
calibrated-PD fields are added alongside).

Decision architecture:

    Raw Probability
        |
    Calibration
        |
    Calibrated PD
        |
    Cost-sensitive threshold optimization (THIS MODULE)
        |
    Business Decision (Approve / Decline)

This is explicitly SEPARATE from risk banding (src.ml.risk_scoring):
    Calibrated PD -> risk_band                     (describes applicant risk)
    Calibrated PD -> decision_threshold -> decision (an operational policy choice)
A "Very High" risk band does NOT automatically mean "Decline" — that would be
hardcoding business policy into the risk-scoring layer, which this project
deliberately avoids (see § "Do not hardcode business policy" in the Milestone
6 spec). The mapping from PD to a decision is entirely parameterized by
FP_COST/FN_COST, swept and optimized here, never a fixed PD cutoff written
directly into code.

Cost function (explicit, not left ambiguous):
    expected_cost = FP_COST * FP + FN_COST * FN
    where, in this credit-risk context:
        "positive" = predicted high-risk / predicted default (probability >= threshold)
        "negative" = predicted acceptable / predicted non-default (probability < threshold)
        FP = predicted high-risk, actually repaid   (a good applicant wrongly declined/flagged)
        FN = predicted acceptable, actually defaulted (a bad applicant wrongly approved)
"""
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

DECISION_ARTIFACT_PATH = os.path.join(settings.models_dir, "decision_threshold.json")
DECISION_OPTIMIZATION_REPORT_PATH = os.path.join("reports", "decision_optimization_report.md")
THRESHOLD_COST_CURVE_PLOT_PATH = os.path.join("reports", "threshold_cost_curve.png")

# Default sweep range — configurable per call, not hardcoded to 0.10-0.90 like
# the legacy raw-probability sweep in src.ml.train.
DEFAULT_THRESHOLD_MIN = 0.01
DEFAULT_THRESHOLD_MAX = 0.99
DEFAULT_THRESHOLD_STEP = 0.01

# Illustrative decision-analysis scenarios (Milestone 6 § 8) — NOT real banking
# policy, NOT claimed to represent any actual lender's cost structure. FP_COST
# is held at 1 in every scenario; only the relative FN/FP ratio matters for
# where the cost-minimizing threshold falls.
SCENARIOS = [
    {"name": "Balanced", "fp_cost": 1.0, "fn_cost": 1.0},
    {"name": "Conservative", "fp_cost": 1.0, "fn_cost": 5.0},
    {"name": "Very Conservative", "fp_cost": 1.0, "fn_cost": 10.0},
]


def expected_cost(fp: int, fn: int, fp_cost: float, fn_cost: float) -> float:
    """expected_cost = FP_COST * FP + FN_COST * FN. See module docstring for the
    exact positive/negative definitions this FP/FN pair assumes.
    """
    return fp_cost * fp + fn_cost * fn


def sweep_thresholds(
    y_true, probabilities,
    threshold_min: float = DEFAULT_THRESHOLD_MIN,
    threshold_max: float = DEFAULT_THRESHOLD_MAX,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
    fp_cost: float = 1.0, fn_cost: float = 1.0,
) -> pd.DataFrame:
    """Sweep thresholds over [threshold_min, threshold_max] and compute, at each:
    FP, FN, TP, TN, expected_cost, precision, recall, F1, acceptance_rate
    (share predicted non-default/"approved"), rejection_rate (share predicted
    default/"declined"). `probabilities` is whatever probability array the
    caller supplies — this function has no opinion on raw vs. calibrated; the
    caller (run_decision_optimization / evaluate_scenario) is responsible for
    passing calibrated PD when that's the intent (Milestone 6's core
    requirement), which is why every real-data call site in this module makes
    that explicit in its own docstring/logging rather than leaving it implicit
    here.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    n = len(y_true)

    thresholds = np.round(np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 4)
    rows = []
    for t in thresholds:
        y_pred = (probabilities >= t).astype(int)  # 1 = predicted high-risk/default
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acceptance_rate = (tn + fn) / n if n > 0 else 0.0  # predicted NOT high-risk -> "approved"
        rejection_rate = (tp + fp) / n if n > 0 else 0.0  # predicted high-risk -> "declined"

        rows.append({
            "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "expected_cost": expected_cost(fp, fn, fp_cost, fn_cost),
            "precision": precision, "recall": recall, "f1": f1,
            "acceptance_rate": acceptance_rate, "rejection_rate": rejection_rate,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(sweep_table: pd.DataFrame) -> dict:
    """The row of sweep_table with minimum expected_cost. Ties broken by the
    lowest threshold (arbitrary but deterministic — reported explicitly rather
    than silently picking whichever row happens to sort first).
    """
    if sweep_table.empty:
        raise ValueError("sweep_table is empty — nothing to optimize over")
    min_cost = sweep_table["expected_cost"].min()
    candidates = sweep_table[sweep_table["expected_cost"] == min_cost]
    best_row = candidates.iloc[0]
    return best_row.to_dict()


def evaluate_scenario(
    name: str, y_true, probabilities, fp_cost: float, fn_cost: float,
    threshold_min: float = DEFAULT_THRESHOLD_MIN, threshold_max: float = DEFAULT_THRESHOLD_MAX,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
) -> dict:
    """Sweep + optimize for one named scenario, returning a flat dict with the
    optimal threshold and its full metrics — the reusable per-scenario contract
    (Milestone 6 § 9) a future UI could call directly.
    """
    sweep_table = sweep_thresholds(
        y_true, probabilities, threshold_min, threshold_max, threshold_step, fp_cost, fn_cost
    )
    optimal = find_optimal_threshold(sweep_table)
    return {
        "scenario": name,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "optimal_threshold": optimal["threshold"],
        "expected_cost": optimal["expected_cost"],
        "precision": optimal["precision"],
        "recall": optimal["recall"],
        "f1": optimal["f1"],
        "acceptance_rate": optimal["acceptance_rate"],
        "rejection_rate": optimal["rejection_rate"],
        "tp": int(optimal["tp"]), "fp": int(optimal["fp"]),
        "fn": int(optimal["fn"]), "tn": int(optimal["tn"]),
        "sweep_table": sweep_table,
    }


def run_scenario_analysis(
    y_true, probabilities, scenarios: List[dict] = None,
    threshold_min: float = DEFAULT_THRESHOLD_MIN, threshold_max: float = DEFAULT_THRESHOLD_MAX,
    threshold_step: float = DEFAULT_THRESHOLD_STEP,
) -> List[dict]:
    """Evaluate every scenario in `scenarios` (default: SCENARIOS — Balanced/
    Conservative/Very Conservative, illustrative only, not real bank policy).
    """
    scenarios = scenarios if scenarios is not None else SCENARIOS
    return [
        evaluate_scenario(s["name"], y_true, probabilities, s["fp_cost"], s["fn_cost"],
                           threshold_min, threshold_max, threshold_step)
        for s in scenarios
    ]


@dataclass
class DecisionArtifact:
    """The Milestone 6 decision artifact — deliberately a NEW file
    (models/decision_threshold.json), not overwriting the legacy
    models/optimal_threshold.json, per the spec's "do not overwrite the old
    artifact until the new implementation is verified." Both coexist: the old
    one still backs src.ml.predict.cost_optimal_decision() (raw probability,
    unchanged); this one backs the new calibrated-PD decision path.
    """
    selected_threshold: float
    probability_source: str  # "calibrated" or "raw" (explicit, never implicit)
    scenario: str
    fp_cost: float
    fn_cost: float
    expected_cost: float
    precision: float
    recall: float
    f1: float
    acceptance_rate: float
    rejection_rate: float
    tp: int
    fp: int
    fn: int
    tn: int
    model_version: str
    calibration_method: str
    feature_version: str
    evaluation_sample_size: int
    optimization_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str = DECISION_ARTIFACT_PATH) -> None:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.info(f"Saved decision artifact to {path} (threshold={self.selected_threshold:.4f}, "
                  f"source={self.probability_source}, scenario={self.scenario})")

    @classmethod
    def load(cls, path: str = DECISION_ARTIFACT_PATH) -> Optional["DecisionArtifact"]:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def decide(calibrated_pd: Optional[float], artifact: Optional[DecisionArtifact]) -> dict:
    """Apply a saved DecisionArtifact's threshold to one applicant's calibrated
    PD. Returns a dict with the decision plus full context (never just a bare
    "Approve"/"Decline" string) — decision_probability_source is ALWAYS
    reported explicitly, per Milestone 6 § 3/§ 12.

    If artifact is None (no decision optimization has been run yet) or
    calibrated_pd is None (no calibrator available), returns decision=None
    rather than fabricating one — an explicit "not available" state, not a
    silent fallback to a different probability without saying so.
    """
    if artifact is None or calibrated_pd is None:
        return {
            "decision": None,
            "decision_probability_source": "unavailable",
            "decision_threshold": None,
            "fp_cost": None,
            "fn_cost": None,
            "decision_scenario": None,
            "decision_scenario_expected_cost": None,
        }
    decision = "Decline" if calibrated_pd >= artifact.selected_threshold else "Approve"
    return {
        "decision": decision,
        "decision_probability_source": artifact.probability_source,
        "decision_threshold": artifact.selected_threshold,
        "fp_cost": artifact.fp_cost,
        "fn_cost": artifact.fn_cost,
        "decision_scenario": artifact.scenario,
        "decision_scenario_expected_cost": artifact.expected_cost,
    }


def run_decision_optimization_pipeline(baseline_scenario_name: str = "Conservative") -> dict:
    """Real end-to-end Milestone 6 pipeline against the held-out validation
    split: sweeps thresholds on CALIBRATED PD for all SCENARIOS, saves the
    baseline scenario (Conservative, FP=1/FN=5 — matching the legacy
    optimal_threshold.json's cost assumptions, for direct old-vs-new
    comparability) as models/decision_threshold.json, computes the old
    (raw-probability) vs. new (calibrated-PD) comparison at that same cost
    ratio, and writes reports/decision_optimization_report.md +
    reports/threshold_cost_curve.png.

    Does NOT retrain or modify models/credit_risk_model.joblib or
    models/probability_calibrator.joblib, and does NOT overwrite
    models/optimal_threshold.json (the legacy raw-probability artifact stays
    exactly as Milestone 1 left it).
    """
    import joblib

    from src.ml.calibration import FEATURE_VERSION, ProbabilityCalibrator, _model_artifact_fingerprint
    from src.ml.predict import CALIBRATOR_PATH, MODEL_PATH, OPTIMAL_THRESHOLD_PATH
    from src.ml.train import _prepare_data

    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features=True, test_size=0.2, random_state=42
    )
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    raw_proba = pipeline.predict_proba(X_val)[:, 1]

    calibrator = ProbabilityCalibrator.load(CALIBRATOR_PATH)
    if calibrator is None:
        raise RuntimeError(
            f"No calibrator found at {CALIBRATOR_PATH} — run `python -m src.ml.calibration` "
            "before running decision optimization, since it operates on calibrated PD."
        )
    calibrated_proba = calibrator.transform(raw_proba)
    y_val_arr = y_val.reset_index(drop=True).to_numpy()

    log.info(f"Decision optimization: {len(y_val_arr):,} validation applicants, "
              f"sweeping thresholds {DEFAULT_THRESHOLD_MIN}-{DEFAULT_THRESHOLD_MAX} "
              f"(step {DEFAULT_THRESHOLD_STEP}) on CALIBRATED PD")

    scenario_results = run_scenario_analysis(y_val_arr, calibrated_proba)
    for r in scenario_results:
        log.info(f"[{r['scenario']}] FP_COST={r['fp_cost']} FN_COST={r['fn_cost']} -> "
                  f"threshold={r['optimal_threshold']:.3f} expected_cost={r['expected_cost']:.1f} "
                  f"precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} "
                  f"acceptance_rate={r['acceptance_rate']:.4f}")

    baseline = next(r for r in scenario_results if r["scenario"] == baseline_scenario_name)
    artifact = DecisionArtifact(
        selected_threshold=baseline["optimal_threshold"],
        probability_source="calibrated",
        scenario=baseline["scenario"],
        fp_cost=baseline["fp_cost"], fn_cost=baseline["fn_cost"],
        expected_cost=baseline["expected_cost"],
        precision=baseline["precision"], recall=baseline["recall"], f1=baseline["f1"],
        acceptance_rate=baseline["acceptance_rate"], rejection_rate=baseline["rejection_rate"],
        tp=baseline["tp"], fp=baseline["fp"], fn=baseline["fn"], tn=baseline["tn"],
        model_version=f"lightgbm-{_model_artifact_fingerprint(MODEL_PATH)}",
        calibration_method=calibrator.method,
        feature_version=FEATURE_VERSION,
        evaluation_sample_size=len(y_val_arr),
    )
    artifact.save(DECISION_ARTIFACT_PATH)

    # ---- Old (raw probability, legacy optimal_threshold.json) vs. new (calibrated PD) ----
    old_comparison = None
    if os.path.exists(OPTIMAL_THRESHOLD_PATH):
        with open(OPTIMAL_THRESHOLD_PATH) as f:
            old_artifact = json.load(f)
        old_threshold = old_artifact["optimal_threshold"]
        old_fp_cost, old_fn_cost = old_artifact["fp_cost"], old_artifact["fn_cost"]
        old_y_pred = (raw_proba >= old_threshold).astype(int)
        old_tp = int(((old_y_pred == 1) & (y_val_arr == 1)).sum())
        old_fp = int(((old_y_pred == 1) & (y_val_arr == 0)).sum())
        old_fn = int(((old_y_pred == 0) & (y_val_arr == 1)).sum())
        old_tn = int(((old_y_pred == 0) & (y_val_arr == 0)).sum())
        old_comparison = {
            "probability_source": "raw",
            "threshold": old_threshold,
            "fp_cost": old_fp_cost, "fn_cost": old_fn_cost,
            "expected_cost": expected_cost(old_fp, old_fn, old_fp_cost, old_fn_cost),
            "tp": old_tp, "fp": old_fp, "fn": old_fn, "tn": old_tn,
            "acceptance_rate": (old_tn + old_fn) / len(y_val_arr),
            "rejection_rate": (old_tp + old_fp) / len(y_val_arr),
        }
        # Re-evaluate the SAME old raw threshold's expected cost under the NEW
        # (FP=1/FN=5, matching this pipeline's baseline scenario) cost ratio too,
        # so "old vs new" is an apples-to-apples cost comparison, not just a
        # different-threshold-different-cost-ratio comparison.
        old_comparison["expected_cost_at_baseline_cost_ratio"] = expected_cost(
            old_fp, old_fn, baseline["fp_cost"], baseline["fn_cost"]
        )

    _write_threshold_cost_plot(scenario_results, THRESHOLD_COST_CURVE_PLOT_PATH)
    report = {
        "validation_sample_size": len(y_val_arr),
        "scenarios": scenario_results,
        "baseline_scenario": baseline_scenario_name,
        "old_comparison": old_comparison,
        "new_baseline": baseline,
    }
    _write_decision_optimization_report(report, DECISION_OPTIMIZATION_REPORT_PATH)

    return report


def _write_threshold_cost_plot(scenario_results: List[dict], output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#C44E52", "#55A868"]
    for r, color in zip(scenario_results, colors):
        sweep = r["sweep_table"]
        ax.plot(sweep["threshold"], sweep["expected_cost"], label=f"{r['scenario']} "
                f"(FP={r['fp_cost']:g}, FN={r['fn_cost']:g})", color=color)
        ax.axvline(r["optimal_threshold"], color=color, linestyle="--", alpha=0.6)

    ax.set_xlabel("Threshold (on calibrated PD)")
    ax.set_ylabel("Expected cost")
    ax.set_title("Threshold vs. expected cost, by scenario (calibrated PD)")
    ax.legend()
    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path) or ".")
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved threshold-cost curve plot to {output_path}")


def _write_decision_optimization_report(report: dict, output_path: str) -> None:
    lines = [
        "# Decision Optimization Report (Milestone 6)",
        "",
        f"Validation sample size: {report['validation_sample_size']:,}",
        "",
        "**Cost function**: `expected_cost = FP_COST * FP + FN_COST * FN`, where "
        "\"positive\" = predicted high-risk/default (calibrated PD >= threshold), "
        "\"negative\" = predicted acceptable/non-default. FP = predicted high-risk, "
        "actually repaid (a good applicant wrongly declined). FN = predicted "
        "acceptable, actually defaulted (a bad applicant wrongly approved).",
        "",
        "**Scenarios are illustrative decision-analysis examples only** — they do "
        "NOT represent any real lender's actual cost structure or policy.",
        "",
        "## Scenario results (threshold optimized on CALIBRATED PD)",
        "",
        "| Scenario | FP Cost | FN Cost | Optimal Threshold | Expected Cost | Precision | "
        "Recall | F1 | Acceptance Rate | Rejection Rate | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["scenarios"]:
        lines.append(
            f"| {r['scenario']} | {r['fp_cost']:g} | {r['fn_cost']:g} | {r['optimal_threshold']:.3f} | "
            f"{r['expected_cost']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['acceptance_rate']:.4f} | {r['rejection_rate']:.4f} | {r['tp']:,} | {r['fp']:,} | "
            f"{r['fn']:,} | {r['tn']:,} |"
        )

    lines += [
        "",
        f"## Baseline scenario selected for the decision artifact: {report['baseline_scenario']}",
        "",
        f"Saved to `models/decision_threshold.json` — used going forward by the new "
        "calibrated-PD decision path (src.ml.decision_optimization.decide()), alongside "
        "(not replacing) the legacy `models/optimal_threshold.json` raw-probability artifact.",
        "",
    ]

    if report["old_comparison"]:
        old = report["old_comparison"]
        new = report["new_baseline"]
        lines += [
            "## Old (raw probability) vs. new (calibrated PD) — same cost assumptions "
            f"(FP_COST={new['fp_cost']:g}, FN_COST={new['fn_cost']:g})",
            "",
            "**Important interpretation note**: calibration changed the probability scale. "
            "The old threshold and new threshold are NOT directly numerically comparable "
            "(\"the threshold improved from X to Y\" would be a meaningless claim) — each "
            "threshold is only meaningful relative to the probability scale it was optimized "
            "on. What IS comparable is the resulting business outcome (FP/FN counts, "
            "acceptance/rejection rates, expected cost under a shared cost assumption).",
            "",
            "| | Old (raw probability) | New (calibrated PD) |",
            "|---|---:|---:|",
            f"| Probability source | {old['probability_source']} | calibrated |",
            f"| Decision threshold | {old['threshold']:.4f} (on raw scale) | {new['optimal_threshold']:.4f} (on calibrated scale) |",
            f"| TP | {old['tp']:,} | {new['tp']:,} |",
            f"| FP | {old['fp']:,} | {new['fp']:,} |",
            f"| FN | {old['fn']:,} | {new['fn']:,} |",
            f"| TN | {old['tn']:,} | {new['tn']:,} |",
            f"| Acceptance rate | {old['acceptance_rate']:.4f} | {new['acceptance_rate']:.4f} |",
            f"| Rejection rate | {old['rejection_rate']:.4f} | {new['rejection_rate']:.4f} |",
            f"| Expected cost (at its OWN original cost ratio, FP=1/FN={old['fn_cost']:g}) | "
            f"{old['expected_cost']:.1f} | — |",
            f"| Expected cost (both at FP={new['fp_cost']:g}/FN={new['fn_cost']:g}) | "
            f"{old['expected_cost_at_baseline_cost_ratio']:.1f} | {new['expected_cost']:.1f} |",
            "",
            "The last row is the only apples-to-apples comparison: both FP/FN outcomes "
            "priced under the identical cost assumption. A lower number there reflects the "
            "new threshold being genuinely better-calibrated to that specific cost tradeoff on "
            "this validation split — not merely a different, incomparable number.",
            "",
        ]

    lines += [
        "## Limitations",
        "",
        "- FP_COST/FN_COST scenarios (Balanced/Conservative/Very Conservative) are "
        "illustrative decision-analysis examples, not real banking policy or an actual "
        "lender's cost structure.",
        "- The optimal threshold depends entirely on the supplied cost assumptions — a "
        "different, equally defensible cost ratio would select a different threshold.",
        "- Evaluated on a random 80/20 validation split; no temporal validation has been "
        "performed, and the optimal threshold could shift under a time-based split or "
        "after retraining.",
        "- Expected cost is only meaningful to the extent the supplied FP_COST/FN_COST "
        "ratio genuinely reflects the decision-maker's priorities — this module optimizes "
        "correctly against whatever ratio it's given, but does not itself validate that "
        "the ratio is realistic.",
        "- Calibration stability over time is unverified (see reports/calibration_report.md's "
        "own limitations) — a decision threshold computed on calibrated PD inherits any "
        "future calibration drift.",
        "",
    ]

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Wrote decision optimization report to {output_path}")


if __name__ == "__main__":
    run_decision_optimization_pipeline()
