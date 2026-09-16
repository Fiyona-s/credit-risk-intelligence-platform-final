"""Milestone 5: centralized risk-scoring engine.

Single authoritative path from a raw applicant to a structured risk-scoring
result — every other module (FastAPI, Streamlit, the chatbot, SHAP, rules)
should go through this, not recompute probability/score/band on its own.

    Applicant
        |
    Feature Engineering (src.data.preprocessor.build_aggregate_features,
                          engineer_features — unchanged, Milestone 1)
        |
    LightGBM (models/credit_risk_model.joblib — unchanged, Milestone 3)
        |
    Raw Probability  (pipeline.predict_proba()[:, 1] — unchanged, unqualified)
        |
    Sigmoid Calibration (models/probability_calibrator.joblib — unchanged, Milestone 4)
        |
    Calibrated PD
        |
    Risk Score   (deterministic, monotonic transform of calibrated PD — see
                  RISK_SCORE_SCALE below)
        |
    Risk Band    (Low / Moderate / High / Very High — thresholds on calibrated
                  PD, chosen from real decile analysis on the validation
                  population; see RISK_BAND_THRESHOLDS below)

Separately (NOT part of the chain above — see § "Business decision threshold
stays separate"):

    Calibrated PD  -- for now, still uses raw_probability + optimal_threshold.json,
        |             UNCHANGED from Milestone 1/4 (see cost_optimal_decision()
        |             in src.ml.predict) — Milestone 6 will explicitly move this
        v             onto calibrated PD. Recorded here only so the distinction
    Business decision  is visible in one place, not silently blurred.
    threshold
        |
    Decision (Approve/Decline)

## Risk-band methodology (evidence-based, not guessed)

Real decile analysis of calibrated PD on the held-out validation split
(61,503 applicants, overall observed default rate 8.07%) — see
reports/risk_scoring_report.md for the full table — showed:

  decile 0-4  (PD  < ~0.051): observed default rate 0.9%-4.3%  (well under half the 8.07% base rate)
  decile 5-6  (PD ~0.051-0.091): observed default rate 5.6%-6.8% (below/near base rate)
  decile 7-8  (PD ~0.091-0.196): observed default rate 10.4%-15.6% (1.3x-1.9x base rate)
  decile 9    (PD ~0.196-0.771): observed default rate 30.2%  (3.7x base rate)

This is a HYBRID of Option A (fixed, reproducible PD thresholds — chosen so
band meaning doesn't silently drift on every retrain the way raw quantile
cutoffs would) informed by Option C (empirical default-rate separation — the
thresholds below were not picked arbitrarily, they sit at points in the
decile table above where the observed default rate visibly steps to a new
regime relative to the population base rate). Chosen thresholds, in round
numbers close to those empirical breakpoints:

  Low       : calibrated PD <  0.05  (below half the population base rate)
  Moderate  : 0.05 <= PD <  0.10     (near/below base rate)
  High      : 0.10 <= PD <  0.20     (1.3x-1.9x base rate)
  Very High : PD >= 0.20             (~3.7x base rate; the top ~10% of applicants)

These are APPLICATION-DEFINED risk categories for this project, evaluated on
one validation snapshot of one dataset. They are NOT official banking, FICO,
or credit-bureau risk grades, and should not be presented as such.
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---- Risk band thresholds (on CALIBRATED PD) — see module docstring for the
# empirical analysis behind these exact numbers. Ordered ascending; band i's
# lower bound is thresholds[i-1], its upper bound is thresholds[i] (or +inf
# for the last band).
RISK_BAND_THRESHOLDS = {
    "Low": (0.0, 0.05),
    "Moderate": (0.05, 0.10),
    "High": (0.10, 0.20),
    "Very High": (0.20, 1.0),
}
RISK_BAND_ORDER = ["Low", "Moderate", "High", "Very High"]

# ---- Risk score: a simple, deterministic, monotonic, bounded transform of
# calibrated PD. Higher score = higher predicted default risk (NOT a FICO-style
# "higher is better" scale — chosen deliberately to avoid that exact confusion,
# documented explicitly here rather than left to be guessed from the number).
#   risk_score = round(calibrated_pd * RISK_SCORE_SCALE), clipped to [0, RISK_SCORE_SCALE]
# RISK_SCORE_SCALE=1000 gives a familiar-feeling 0-1000 range without
# resembling any specific real credit-scoring product's numeric range
# (FICO 300-850, VantageScore 300-850, etc.) — deliberately different so it is
# never mistaken for one of those.
RISK_SCORE_SCALE = 1000


def compute_risk_score(calibrated_pd: float) -> int:
    """calibrated_pd (0-1) -> risk_score (0-RISK_SCORE_SCALE), rounded to the
    nearest integer. Deterministic, monotonically increasing, bounded.
    """
    if calibrated_pd is None:
        raise ValueError("compute_risk_score requires a non-None calibrated_pd")
    clipped = min(max(calibrated_pd, 0.0), 1.0)
    return int(round(clipped * RISK_SCORE_SCALE))


def compute_risk_band(calibrated_pd: float) -> str:
    """calibrated_pd (0-1) -> one of RISK_BAND_ORDER, via RISK_BAND_THRESHOLDS.
    The upper bound of each band is exclusive except for the last (Very High),
    which is inclusive of 1.0.
    """
    if calibrated_pd is None:
        raise ValueError("compute_risk_band requires a non-None calibrated_pd")
    for band in RISK_BAND_ORDER:
        low, high = RISK_BAND_THRESHOLDS[band]
        if band == RISK_BAND_ORDER[-1]:
            if calibrated_pd >= low:
                return band
        elif low <= calibrated_pd < high:
            return band
    # Only reachable if calibrated_pd < 0, which compute_risk_score also guards against.
    raise ValueError(f"calibrated_pd={calibrated_pd} is out of the expected [0, 1] range")


@dataclass
class RiskScoringResult:
    """The structured, centralized prediction contract (Milestone 5 § 3).
    Every field src.ml.predict.predict_applicant() historically returned is
    preserved (see that module's thin wrapper around score_applicant() below)
    — this dataclass is the new authoritative shape; predict_applicant() maps
    it to a dict with the old keys included for backward compatibility.
    """
    applicant_id: Optional[Union[int, str]]
    raw_probability: float
    calibrated_probability: Optional[float]
    calibration_method: Optional[str]
    risk_score: Optional[int]
    risk_band: str
    model_version: str
    feature_version: str
    features_used: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        """Built manually (not via dataclasses.asdict()) so features_used (a
        DataFrame) is kept as the same object rather than dataclasses.asdict()
        deep-copying it — a live DataFrame isn't a nested dataclass/list/dict
        container, and deep-copying it here would be pure overhead.
        """
        return {
            "applicant_id": self.applicant_id,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "calibration_method": self.calibration_method,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "features_used": self.features_used,
        }


def score_applicant(applicant: Union[Dict, pd.DataFrame]) -> RiskScoringResult:
    """The single, centralized risk-scoring path (Milestone 5 § 2/§ 15):
    feature engineering -> LightGBM -> raw probability -> calibration ->
    calibrated PD -> risk score -> risk band. Returns a RiskScoringResult;
    src.ml.predict.predict_applicant() wraps this for backward-compatible
    dict-shaped output (adding cost_optimal_decision/optimal_threshold, which
    remain on raw_probability — see this module's docstring and Milestone 6).

    Risk score AND risk band are both computed from calibrated_probability
    when a calibrator is available. If no calibrator artifact exists,
    calibration_method="none" and BOTH risk_score and risk_band fall back to
    raw_probability — explicitly, not silently: a caller can always tell which
    happened from calibration_method.
    """
    from src.data.preprocessor import build_aggregate_features, engineer_features
    from src.ml.predict import load_calibrator, load_model

    bundle = load_model()
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = pd.DataFrame([applicant]) if isinstance(applicant, dict) else applicant.copy()
    applicant_id = None
    if "SK_ID_CURR" in df.columns and len(df) > 0:
        raw_id = df["SK_ID_CURR"].iloc[0]
        applicant_id = raw_id.item() if hasattr(raw_id, "item") else raw_id

    df = build_aggregate_features(df)
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None
    df_features = df[numeric_cols + categorical_cols]

    raw_probability = float(pipeline.predict_proba(df_features)[:, 1][0])

    calibrator = load_calibrator()
    if calibrator is not None:
        calibrated_probability = float(calibrator.transform([raw_probability])[0])
        calibration_method = calibrator.method
        pd_for_scoring = calibrated_probability
    else:
        # Explicit fallback (Milestone 5 § 5/§ 12 of the spec) — never silently
        # present an uncalibrated number as calibrated PD.
        calibrated_probability = None
        calibration_method = "none"
        pd_for_scoring = raw_probability
        log.warning(
            "No probability calibrator found — risk_score/risk_band fall back to "
            "raw_probability. Run `python -m src.ml.calibration` to enable calibrated scoring."
        )

    risk_score = compute_risk_score(pd_for_scoring)
    risk_band_value = compute_risk_band(pd_for_scoring)

    model_version = _model_version()
    feature_version = _feature_version()

    return RiskScoringResult(
        applicant_id=applicant_id,
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        calibration_method=calibration_method,
        risk_score=risk_score,
        risk_band=risk_band_value,
        model_version=model_version,
        feature_version=feature_version,
        features_used=df_features,
    )


def _model_version() -> str:
    """Minimal, honest model-version proxy: the deployed model artifact's
    content fingerprint (same md5-12 scheme src.ml.calibration already uses to
    detect model drift relative to a calibrator). NOT a formal semantic
    versioning scheme — no model_metadata.json exists yet (that's Milestone
    12's job) — this is deliberately just enough for Milestone 12 to build on,
    per the Milestone 5 spec's "do not build the full versioning system yet."
    """
    from src.ml.calibration import _model_artifact_fingerprint
    from src.ml.predict import MODEL_PATH

    if not os.path.exists(MODEL_PATH):
        return "unknown"
    return f"lightgbm-{_model_artifact_fingerprint(MODEL_PATH)}"


def _feature_version() -> str:
    """Same feature-version identifier src.ml.model_comparison and
    src.ml.calibration already use — imported from calibration to avoid a
    third independent copy of the literal string.
    """
    from src.ml.calibration import FEATURE_VERSION

    return FEATURE_VERSION


RISK_SCORING_REPORT_PATH = os.path.join("reports", "risk_scoring_report.md")


def _old_risk_band(raw_probability, low_threshold: float, medium_threshold: float) -> str:
    """The PRE-Milestone-5 risk_band() logic (src/utils/helpers.py), reproduced
    here (not imported) purely so generate_risk_scoring_report() can compute a
    like-for-like "what would the old system have said" comparison without
    resurrecting the old raw-probability-based call path in predict.py itself.
    """
    if raw_probability < low_threshold:
        return "Low"
    if raw_probability < medium_threshold:
        return "Medium"
    return "High"


def generate_risk_scoring_report(output_path: str = RISK_SCORING_REPORT_PATH) -> dict:
    """Real end-to-end evidence for the risk-band methodology (Milestone 5 § 11):
    band definitions, population coverage, mean/median calibrated PD, observed
    default rate per band, a monotonicity check, and an old-vs-new comparison
    (raw-probability bands from Milestones 1-4 vs. calibrated-PD bands here) —
    computed on the real held-out validation split, not fabricated.
    """
    import joblib
    import numpy as np

    from src.ml.predict import CALIBRATOR_PATH, MODEL_PATH
    from src.ml.calibration import ProbabilityCalibrator
    from src.ml.train import _prepare_data
    from src.utils.config import settings
    from src.utils.helpers import ensure_dir

    df, X_train, X_val, y_train, y_val, numeric_cols, categorical_cols = _prepare_data(
        use_bureau_features=True, test_size=0.2, random_state=42
    )
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    raw_proba = pipeline.predict_proba(X_val)[:, 1]

    calibrator = ProbabilityCalibrator.load(CALIBRATOR_PATH)
    if calibrator is None:
        raise RuntimeError(
            "No calibrator found at "
            f"{CALIBRATOR_PATH} — run `python -m src.ml.calibration` before generating "
            "the risk-scoring report, since the new bands are defined on calibrated PD."
        )
    calibrated_proba = calibrator.transform(raw_proba)
    y_val_arr = y_val.reset_index(drop=True).to_numpy()

    # ---- New (calibrated-PD) bands: real per-band statistics ----
    new_bands = np.array([compute_risk_band(p) for p in calibrated_proba])
    band_rows = []
    for band in RISK_BAND_ORDER:
        mask = new_bands == band
        count = int(mask.sum())
        band_pd = calibrated_proba[mask]
        band_rows.append({
            "band": band,
            "pd_range": RISK_BAND_THRESHOLDS[band],
            "count": count,
            "pct_of_population": count / len(new_bands) if len(new_bands) else 0.0,
            "mean_pd": float(band_pd.mean()) if count else None,
            "median_pd": float(np.median(band_pd)) if count else None,
            "observed_default_rate": float(y_val_arr[mask].mean()) if count else None,
            "score_range": (compute_risk_score(band_pd.min()), compute_risk_score(band_pd.max())) if count else None,
        })

    # ---- Monotonicity check ----
    mean_pds = [r["mean_pd"] for r in band_rows if r["mean_pd"] is not None]
    observed_rates = [r["observed_default_rate"] for r in band_rows if r["observed_default_rate"] is not None]
    pd_monotonic = all(mean_pds[i] < mean_pds[i + 1] for i in range(len(mean_pds) - 1))
    rate_monotonic = all(observed_rates[i] < observed_rates[i + 1] for i in range(len(observed_rates) - 1))

    # ---- Old (raw-probability) bands, for comparison ----
    old_bands = np.array([
        _old_risk_band(p, settings.risk_threshold_low, settings.risk_threshold_medium) for p in raw_proba
    ])
    old_band_order = ["Low", "Medium", "High"]
    old_distribution = {b: int((old_bands == b).sum()) for b in old_band_order}
    new_distribution = {b: int((new_bands == b).sum()) for b in RISK_BAND_ORDER}

    crosstab = {}
    for old_b in old_band_order:
        crosstab[old_b] = {new_b: int(((old_bands == old_b) & (new_bands == new_b)).sum()) for new_b in RISK_BAND_ORDER}

    # "Materially reclassified": old Low -> new High/Very High, or old High -> new Low.
    # (old Medium has no single unambiguous "opposite" band, so it's excluded from
    # this specific count — the full crosstab above is the complete picture.)
    old_low_now_high = crosstab["Low"]["High"] + crosstab["Low"]["Very High"]
    old_high_now_low = crosstab["High"]["Low"]
    n_val = len(X_val)

    report = {
        "validation_sample_size": n_val,
        "overall_observed_default_rate": float(y_val_arr.mean()),
        "bands": band_rows,
        "pd_monotonic": pd_monotonic,
        "rate_monotonic": rate_monotonic,
        "old_distribution": old_distribution,
        "new_distribution": new_distribution,
        "crosstab": crosstab,
        "old_low_now_high_or_very_high": old_low_now_high,
        "old_high_now_low": old_high_now_low,
    }

    _write_risk_scoring_report(report, output_path)
    return report


def _write_risk_scoring_report(report: dict, output_path: str) -> None:
    from src.utils.helpers import ensure_dir

    lines = [
        "# Risk Scoring Report (Milestone 5)",
        "",
        f"Validation sample size: {report['validation_sample_size']:,}",
        f"Overall observed default rate: {report['overall_observed_default_rate']:.4f}",
        "",
        "## Band definitions (calibrated PD)",
        "",
        "| Band | PD range | Count | % of population | Mean PD | Median PD | "
        "Observed default rate | Risk score range |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["bands"]:
        low, high = r["pd_range"]
        score_range = f"{r['score_range'][0]}-{r['score_range'][1]}" if r["score_range"] else "n/a"
        lines.append(
            f"| {r['band']} | [{low:.2f}, {high:.2f}) | {r['count']:,} | "
            f"{r['pct_of_population']:.1%} | {r['mean_pd']:.4f} | {r['median_pd']:.4f} | "
            f"{r['observed_default_rate']:.4f} | {score_range} |"
        )

    lines += [
        "",
        "## Monotonicity check",
        "",
        f"- Mean calibrated PD strictly increasing across bands: **{report['pd_monotonic']}**",
        f"- Observed default rate strictly increasing across bands: **{report['rate_monotonic']}**",
        "",
        "## Old (raw-probability) vs. new (calibrated-PD) band comparison",
        "",
        "Old bands (Milestones 1-4): raw LightGBM probability against "
        "RISK_THRESHOLD_LOW/MEDIUM (Low/Medium/High, 3 bands). "
        "New bands (this milestone): calibrated PD against the empirically-derived "
        "thresholds above (Low/Moderate/High/Very High, 4 bands). The two schemes "
        "use different numbers of bands, so there is no 1:1 relabeling — the full "
        "crosstab below is the complete picture.",
        "",
        "**Old band distribution**:",
        "",
    ]
    for b, count in report["old_distribution"].items():
        lines.append(f"- {b}: {count:,} ({count / report['validation_sample_size']:.1%})")
    lines += ["", "**New band distribution**:", ""]
    for b, count in report["new_distribution"].items():
        lines.append(f"- {b}: {count:,} ({count / report['validation_sample_size']:.1%})")

    lines += ["", "**Crosstab (rows = old band, columns = new band, values = applicant count)**:", ""]
    header = "| Old \\ New | " + " | ".join(RISK_BAND_ORDER) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(RISK_BAND_ORDER))
    for old_b, row in report["crosstab"].items():
        lines.append(f"| {old_b} | " + " | ".join(str(row[new_b]) for new_b in RISK_BAND_ORDER) + " |")

    lines += [
        "",
        "**Material reclassifications** (crossing from one extreme to the other):",
        "",
        f"- Old \"Low\" -> new \"High\"/\"Very High\": {report['old_low_now_high_or_very_high']:,} applicants",
        f"- Old \"High\" -> new \"Low\": {report['old_high_now_low']:,} applicants",
        "",
        "## Interpretation",
        "",
        "Calibration substantially compressed the probability scale (raw LightGBM "
        "probabilities are systematically overconfident — see reports/calibration_report.md), "
        "so a direct numeric comparison of old vs. new band assignments reflects that scale "
        "change, not a change in the model's underlying risk ranking (ROC-AUC/PR-AUC are "
        "unchanged by calibration, per Milestone 4). The new bands are recalibrated to be "
        "meaningful on the calibrated PD scale specifically, which the old fixed thresholds "
        "(designed around the old, overconfident raw scale) were not.",
        "",
        "## Limitations",
        "",
        "- These bands are application-defined risk categories for this project, evaluated on "
        "one validation snapshot of one dataset — NOT official banking, FICO, or "
        "credit-bureau risk grades.",
        "- Business decision thresholds (Approve/Decline, cost-optimal thresholds) are a "
        "SEPARATE concept from these risk bands and are not redesigned here — see "
        "src.ml.predict.cost_optimal_decision, still raw-probability-based pending Milestone 6.",
        "- Thresholds were chosen from a random 80/20 validation split; no temporal validation "
        "has been performed, and thresholds could shift under a time-based split or after "
        "retraining on new data.",
        "",
    ]

    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"Wrote risk scoring report to {output_path}")


if __name__ == "__main__":
    generate_risk_scoring_report()
