"""Fairness audit: demographic parity difference and equal opportunity difference
across CODE_GENDER and NAME_EDUCATION_TYPE, computed on the same stratified 80/20
validation split as src/ml/train.py, using the already-saved model. Reporting only
— this module never trains or modifies models/credit_risk_model.joblib.

No new dependencies: pandas/numpy/matplotlib only, same as the rest of the ML stack.

Definitions used (standard fairness-audit terminology):
- "Predicted high-risk" = predicted probability of default >= RISK_THRESHOLD_MEDIUM
  (the same threshold app.py/predict.py use for the "High" risk band). Treated as
  the model's unfavorable/positive-for-scrutiny outcome, consistent with fairness
  literature framing the outcome being predicted (here, default risk) as the
  "positive" class regardless of whether it's a good or bad outcome for the subject.
- "Approved" = NOT predicted high-risk (Low or Medium band) — this project has no
  separate underwriting/approval model, so risk band is used as the approval proxy.
- Demographic parity difference = max(predicted high-risk rate) - min(...) across
  groups of one sensitive attribute. 0 = identical rates across groups.
- Equal opportunity difference = max(true positive rate) - min(...) across groups,
  where true positive rate = P(predicted high-risk | actually defaulted) — i.e. does
  the model catch actual defaulters equally well across groups, not just predict
  "high risk" at equal rates overall.
- Per-group flag: a group's predicted-high-risk rate differs from the *overall*
  validation-set rate by more than DISPARITY_FLAG_THRESHOLD (5 percentage points).
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import build_joined_dataset
from src.data.preprocessor import engineer_features
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")

# The two sensitive attributes this audit covers. Both are already part of the
# model's own trained categorical_cols (see bundle["categorical_cols"]), so they
# come along for free in X_val without a separate data split.
SENSITIVE_ATTRIBUTES = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

DISPARITY_FLAG_THRESHOLD = 0.05  # 5 percentage points, per the task spec

# Below this group size, a disparity figure is more likely sample-size noise than a
# real finding (e.g. CODE_GENDER's "XNA" category has n=2 in the real dataset) — the
# group still gets computed and flagged normally (never silently dropped/reweighted),
# but callers (the Streamlit tab, the CLI __main__ below) surface a caveat rather than
# presenting a small-N group's number with the same confidence as a large one.
MIN_RELIABLE_GROUP_SIZE = 30


def _get_validation_split(df: pd.DataFrame = None, test_size: float = 0.2, random_state: int = 42):
    """Reconstruct the same stratified 80/20 split src.ml.train.train() used, against
    the already-saved model — never retrains.

    No module-level caching here deliberately: this always re-reads
    models/credit_risk_model.joblib fresh, so it can never hand back a stale pipeline
    after a retrain. app.py's own cache (_compute_fairness_audit, keyed on the model
    file's mtime) is where the ~40s aggregation cost actually gets amortized across
    Streamlit reruns — caching it again here with no mtime key would risk serving a
    pipeline from before the most recent retrain.

    df: optionally a pre-loaded, already-joined applicant dataframe (e.g. app.py's
    get_sample_data(), which falls back to synthetic data when the real CSV is
    missing). If not given, loads the real dataset directly via build_joined_dataset()
    — the CLI (__main__ below) uses this path and expects the real data to be present,
    same as src.ml.train's own CLI.
    """
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    if df is None:
        df = build_joined_dataset(is_train=True)
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None

    y = df["TARGET"]
    X = df[numeric_cols + categorical_cols]
    _, X_val, _, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return pipeline, X_val, y_val


def compute_group_metrics(attribute: str, threshold: float = None, df: pd.DataFrame = None) -> pd.DataFrame:
    """Per-group table: count, actual default rate, mean predicted risk score,
    predicted high-risk rate, approval rate, true positive rate, and whether this
    group's predicted-high-risk rate deviates from the overall rate by more than
    DISPARITY_FLAG_THRESHOLD.
    """
    if attribute not in SENSITIVE_ATTRIBUTES:
        raise ValueError(f"{attribute!r} is not an audited attribute ({SENSITIVE_ATTRIBUTES})")

    pipeline, X_val, y_val = _get_validation_split(df=df)
    threshold = settings.risk_threshold_medium if threshold is None else threshold

    y_proba = pipeline.predict_proba(X_val)[:, 1]
    predicted_high_risk = (y_proba >= threshold).astype(int)

    frame = pd.DataFrame({
        "group": X_val[attribute].astype(str).values,
        "y_true": y_val.values,
        "y_proba": y_proba,
        "predicted_high_risk": predicted_high_risk,
    })

    overall_high_risk_rate = frame["predicted_high_risk"].mean()

    rows = []
    for group, g in frame.groupby("group"):
        n = len(g)
        actual_positives = g[g["y_true"] == 1]
        true_positive_rate = (
            actual_positives["predicted_high_risk"].mean() if len(actual_positives) > 0 else np.nan
        )
        predicted_high_risk_rate = g["predicted_high_risk"].mean()
        rows.append({
            "group": group,
            "n": n,
            "actual_default_rate": g["y_true"].mean(),
            "mean_predicted_risk_score": g["y_proba"].mean(),
            "predicted_high_risk_rate": predicted_high_risk_rate,
            "approval_rate": 1 - predicted_high_risk_rate,
            "true_positive_rate": true_positive_rate,
            "disparity_vs_overall": predicted_high_risk_rate - overall_high_risk_rate,
        })

    result = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    result["flagged"] = result["disparity_vs_overall"].abs() > DISPARITY_FLAG_THRESHOLD
    return result


def demographic_parity_difference(group_metrics: pd.DataFrame) -> float:
    """max - min of predicted_high_risk_rate across groups. 0 = identical rates."""
    rates = group_metrics["predicted_high_risk_rate"].dropna()
    return float(rates.max() - rates.min()) if len(rates) > 1 else 0.0


def equal_opportunity_difference(group_metrics: pd.DataFrame) -> float:
    """max - min of true_positive_rate (recall on actual defaulters) across groups.
    0 = the model catches actual defaulters equally well regardless of group.
    """
    rates = group_metrics["true_positive_rate"].dropna()
    return float(rates.max() - rates.min()) if len(rates) > 1 else 0.0


def run_audit(attribute: str, threshold: float = None, df: pd.DataFrame = None) -> dict:
    """Full audit result for one sensitive attribute."""
    group_metrics = compute_group_metrics(attribute, threshold, df=df)
    dpd = demographic_parity_difference(group_metrics)
    eod = equal_opportunity_difference(group_metrics)
    flagged_groups = group_metrics.loc[group_metrics["flagged"], "group"].tolist()

    return {
        "attribute": attribute,
        "group_metrics": group_metrics,
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eod,
        "flagged_groups": flagged_groups,
    }


def plot_approval_rate_by_group(group_metrics: pd.DataFrame, attribute: str):
    """Bar chart of approval rate (%) by group, flagged groups highlighted in a
    different color. Returns a matplotlib Figure — caller is responsible for
    displaying/closing it (e.g. via st.pyplot then plt.close in the Streamlit tab).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C44E52" if flagged else "#4C72B0" for flagged in group_metrics["flagged"]]
    ax.bar(group_metrics["group"], group_metrics["approval_rate"] * 100, color=colors)
    ax.set_ylabel("Approval rate (%)")
    ax.set_title(f"Approval rate by {attribute}")
    ax.set_xticks(range(len(group_metrics)))
    ax.set_xticklabels(group_metrics["group"], rotation=30, ha="right")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    for attr in SENSITIVE_ATTRIBUTES:
        result = run_audit(attr)
        print(f"\n=== Fairness audit: {attr} ===")
        gm = result["group_metrics"].copy()
        for col in ["actual_default_rate", "mean_predicted_risk_score", "predicted_high_risk_rate",
                    "approval_rate", "true_positive_rate", "disparity_vs_overall"]:
            gm[col] = (gm[col] * 100).round(2)
        print(gm.to_string(index=False))
        print(f"Demographic parity difference: {result['demographic_parity_difference']:.4f}")
        print(f"Equal opportunity difference:  {result['equal_opportunity_difference']:.4f}")
        if result["flagged_groups"]:
            print(f"FLAGGED (>{DISPARITY_FLAG_THRESHOLD:.0%} disparity vs. overall): {result['flagged_groups']}")
        else:
            print(f"No group exceeds the {DISPARITY_FLAG_THRESHOLD:.0%} disparity threshold.")

        small_groups = result["group_metrics"].loc[
            result["group_metrics"]["n"] < MIN_RELIABLE_GROUP_SIZE, "group"
        ].tolist()
        if small_groups:
            print(
                f"CAUTION: {small_groups} have fewer than {MIN_RELIABLE_GROUP_SIZE} validation "
                "rows — their disparity figures are more likely sample-size noise than a "
                "reliable finding."
            )
