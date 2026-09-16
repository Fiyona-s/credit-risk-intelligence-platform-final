"""Applicant-level SHAP explainability for the deployed LightGBM model (Milestone 7).

## SHAP explanation semantics — verified empirically, not assumed

`shap.TreeExplainer(model)` is used with its default `model_output="raw"` — NOT
reconfigured to explain probability. Confirmed directly against a real applicant
(SK_ID_CURR=100002, see reports/shap_explainability_report.md for the full
verification):

    explainer.model_output          == "raw"
    explainer.expected_value        == -0.6238532645115576   (a single scalar; can be
                                                                negative, so this is NOT
                                                                a probability)
    base_value + sum(shap_values)   == 1.625874401494976
    model.predict(X, raw_score=True)== 1.6258744               (exact match)
    sigmoid(base_value + sum(shap)) == 0.8356036890313358
    pipeline.predict_proba(X)[:,1]  == 0.83560369               (exact match)

Conclusion: SHAP values and the base/expected value live on the model's RAW
MARGIN (log-odds) scale — the pre-sigmoid decision function output — not on the
probability scale directly. `raw_probability = sigmoid(base_value +
sum(shap_values))`, a NONLINEAR transform, so individual SHAP values do not
correspond one-to-one to changes in raw_probability; only the reconstructed
TOTAL (base + sum), passed through sigmoid, recovers raw_probability exactly.

SHAP_EXPLANATION_TARGET below names this explicitly so no caller can present a
SHAP contribution as if it directly explained a probability, calibrated PD,
risk band, or business decision:

    Applicant -> Feature Engineering -> LightGBM -> Raw Model Output (margin)
        -> SHAP explains THIS
    Raw Model Output -> sigmoid -> raw_probability
        -> sigmoid Calibration -> calibrated_probability
        -> Risk Scoring -> risk_score / risk_band
        -> Decision Optimization -> Approve/Decline
        -> SHAP explains NONE of these downstream transformations directly;
           it explains why the model produced the raw margin that ultimately
           feeds into all of them.

## Consistency with prediction

`explain_applicant_full()` calls `src.ml.predict.predict_applicant()` internally
and reuses its `features_used` DataFrame verbatim as the SHAP input — the exact
same engineered/aligned feature matrix predict_applicant() fed to
`pipeline.predict_proba()`. This is not merely "the same code path" but the
literal same DataFrame object, so SHAP cannot silently drift onto a different
feature representation than the one that actually produced the prediction
being explained (see tests/test_shap_explainability.py's consistency test).
"""
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.feature_registry import FEATURE_REGISTRY
from src.data.preprocessor import build_aggregate_features, engineer_features, get_output_feature_names
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")

# See module docstring for the empirical verification behind this label.
SHAP_EXPLANATION_TARGET = "raw_model_margin"

_FEATURE_REGISTRY_BY_NAME = {entry["name"]: entry for entry in FEATURE_REGISTRY}

_explainer_cache = None
_bundle_cache = None


def _load():
    global _explainer_cache, _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = joblib.load(MODEL_PATH)
        model = _bundle_cache["pipeline"].named_steps["model"]
        _explainer_cache = shap.TreeExplainer(model)
        log.info(f"SHAP TreeExplainer initialized (model_output={_explainer_cache.model_output})")
    return _bundle_cache, _explainer_cache


def _business_meaning(source_feature: str) -> Optional[str]:
    """Milestone 1 feature registry lookup. Returns None — a safe, explicit
    "not registered" signal the caller must handle — rather than fabricating an
    interpretation for a feature the registry doesn't cover (e.g. a raw
    application-table passthrough column like AMT_INCOME_TOTAL, which the
    registry intentionally excludes — see src/data/feature_registry.py's own
    docstring on what it does and doesn't cover).
    """
    entry = _FEATURE_REGISTRY_BY_NAME.get(source_feature)
    return entry["business_meaning"] if entry else None


def _split_onehot_feature_name(encoded_name: str, categorical_cols: List[str]) -> Optional[Tuple[str, str]]:
    """"NAME_CONTRACT_TYPE_Cash loans" -> ("NAME_CONTRACT_TYPE", "Cash loans").

    Matches encoded_name against each known categorical source column name as a
    prefix, longest source name first (so e.g. "NAME_CONTRACT_TYPE" doesn't
    accidentally match before a hypothetical longer overlapping name). Returns
    None if encoded_name isn't a one-hot column at all (a numeric passthrough
    feature) — this relies on src.data.preprocessor.get_output_feature_names()
    now being called WITH categorical_cols (Milestone 7 fix — previously it
    produced meaningless "x0_Cash loans"-style names with no source-column
    context, which this function could not have reliably reversed).
    """
    for source in sorted(categorical_cols, key=len, reverse=True):
        prefix = f"{source}_"
        if encoded_name.startswith(prefix):
            return source, encoded_name[len(prefix):]
    return None


def _describe_feature(
    encoded_name: str, categorical_cols: List[str], raw_row: pd.Series, transformed_value: float
) -> dict:
    """One encoded model feature -> business-readable description: source
    feature, category (if one-hot; None for numeric), a display value, and a
    business_meaning with a safe fallback (never a fabricated interpretation)
    when the source isn't in the feature registry.
    """
    split = _split_onehot_feature_name(encoded_name, categorical_cols)
    if split is not None:
        source_feature, category = split
        meaning = _business_meaning(source_feature)
        return {
            "source_feature": source_feature,
            "category": category,
            "display_value": category,
            "business_meaning": meaning if meaning else f"Category of {source_feature.replace('_', ' ').title()}",
            "is_categorical": True,
        }
    display_val = raw_row[encoded_name] if encoded_name in raw_row.index else round(float(transformed_value), 4)
    if hasattr(display_val, "item"):
        display_val = display_val.item()
    meaning = _business_meaning(encoded_name)
    return {
        "source_feature": encoded_name,
        "category": None,
        "display_value": display_val,
        "business_meaning": meaning if meaning else encoded_name.replace("_", " ").title(),
        "is_categorical": False,
    }


@dataclass
class ApplicantExplanation:
    """Structured, business-readable applicant-level explanation (Milestone 7).

    See this module's docstring for exactly what shap_explanation_target means
    and what SHAP does/doesn't explain. calibrated_probability/risk_score/
    decision/etc. are carried through from src.ml.predict.predict_applicant()
    for display alongside the explanation — SHAP itself does not compute or
    attribute those downstream quantities.
    """
    applicant_id: Optional[Union[int, str]]
    raw_probability: float
    calibrated_probability: Optional[float]
    risk_score: Optional[int]
    risk_band: str
    decision: Optional[str]
    decision_threshold: Optional[float]
    shap_explanation_target: str
    shap_base_value: float
    shap_model_output: float
    top_risk_increasing: List[dict]
    top_risk_reducing: List[dict]
    all_contributions: List[dict] = field(default_factory=list)
    model_version: Optional[str] = None
    feature_version: Optional[str] = None
    calibration_method: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def explain_applicant_full(applicant: Union[Dict, pd.DataFrame], top_n: int = 5) -> ApplicantExplanation:
    """The Milestone 7 structured explanation. Calls src.ml.predict.predict_applicant()
    for raw/calibrated probability, risk score/band, and decision (one source of
    truth — never recomputed independently here), then attaches the SHAP
    breakdown of the underlying raw model margin, ranked by |SHAP value| and
    split into risk-increasing (SHAP > 0) vs. risk-reducing (SHAP < 0) drivers.
    """
    from src.ml.predict import predict_applicant  # lazy: avoid import-time coupling

    prediction = predict_applicant(applicant)

    bundle, explainer = _load()
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    preprocessor = pipeline.named_steps["preprocessor"]

    df = prediction["features_used"]  # identical DataFrame predict_applicant() scored — see module docstring
    X_transformed = preprocessor.transform(df)
    feature_names = get_output_feature_names(preprocessor, numeric_cols, categorical_cols)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # some shap/LightGBM version combos return [class0, class1]
        shap_values = shap_values[1]
    row_shap = np.asarray(shap_values)[0]

    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    base_value = float(base_value)
    model_output = base_value + float(row_shap.sum())

    raw_row = df.iloc[0]
    all_contributions = []
    for idx, fname in enumerate(feature_names):
        shap_value = float(row_shap[idx])
        desc = _describe_feature(fname, categorical_cols, raw_row, X_transformed[0][idx])
        all_contributions.append({
            "feature": fname,
            **desc,
            "shap_value": shap_value,
            "direction": "increases_risk" if shap_value > 0 else "reduces_risk",
            "impact": abs(shap_value),
        })

    ranked = sorted(all_contributions, key=lambda c: c["impact"], reverse=True)
    top_increasing = [c for c in ranked if c["shap_value"] > 0][:top_n]
    top_reducing = [c for c in ranked if c["shap_value"] < 0][:top_n]

    # From predict_applicant() itself (src.ml.risk_scoring computes this from the
    # ORIGINAL applicant input before it's trimmed to numeric_cols+categorical_cols
    # for scoring — features_used here never contains SK_ID_CURR, since that's an
    # excluded id column, not a model feature).
    applicant_id = prediction.get("applicant_id")

    return ApplicantExplanation(
        applicant_id=applicant_id,
        raw_probability=prediction["raw_probability"],
        calibrated_probability=prediction["calibrated_probability"],
        risk_score=prediction["risk_score"],
        risk_band=prediction["risk_band"],
        decision=prediction["decision"],
        decision_threshold=prediction["decision_threshold"],
        shap_explanation_target=SHAP_EXPLANATION_TARGET,
        shap_base_value=base_value,
        shap_model_output=model_output,
        top_risk_increasing=top_increasing,
        top_risk_reducing=top_reducing,
        all_contributions=ranked,
        model_version=prediction["model_version"],
        feature_version=prediction["feature_version"],
        calibration_method=prediction["calibration_method"],
    )


def _plain_language(feature: str, shap_value: float, feature_value) -> str:
    direction = "increased" if shap_value > 0 else "decreased"
    magnitude = abs(shap_value)
    readable = feature.replace("_", " ").title()
    return f"{readable} = {feature_value} {direction} risk (impact {magnitude:.3f})"


def _top_shap_contributions(applicant: Dict, top_n: int) -> List[dict]:
    """PRESERVED for backward compatibility with existing callers
    (explain_applicant/explain_applicant_structured, used by the Streamlit
    Explainability tab and api.py's /predict). Same computation
    explain_applicant_full() now does internally, kept as a separate simpler
    path so those two existing call sites' output shape doesn't change.
    {feature, value, shap_value, direction, impact}, sorted by |shap_value|
    descending. feature names now come from the Milestone 7 naming fix
    (get_output_feature_names with categorical_cols), so a categorical
    feature's name is e.g. "NAME_CONTRACT_TYPE_Cash loans" instead of the
    previous, uninformative "x0_Cash loans".
    """
    bundle, explainer = _load()
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = pd.DataFrame([applicant])
    df = build_aggregate_features(df)
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None
    df = df[numeric_cols + categorical_cols]

    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(df)
    feature_names = get_output_feature_names(preprocessor, numeric_cols, categorical_cols)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # binary classifiers sometimes return [class0, class1]
        shap_values = shap_values[1]
    row_shap = np.asarray(shap_values)[0]

    order = np.argsort(-np.abs(row_shap))[:top_n]
    raw_row = df.iloc[0]

    contributions = []
    for idx in order:
        fname = feature_names[idx]
        display_val = raw_row[fname] if fname in raw_row.index else round(float(X_transformed[0][idx]), 3)
        shap_value = float(row_shap[idx])
        contributions.append({
            "feature": fname,
            "value": display_val.item() if hasattr(display_val, "item") else display_val,
            "shap_value": shap_value,
            "direction": "increased" if shap_value > 0 else "decreased",
            "impact": abs(shap_value),
        })
    return contributions


def explain_applicant(applicant: Dict, top_n: int = 5) -> List[str]:
    """Return the top_n features driving this applicant's prediction, as plain-language strings."""
    contributions = _top_shap_contributions(applicant, top_n)
    return [
        _plain_language(c["feature"], c["shap_value"], c["value"])
        for c in contributions
    ]


def explain_applicant_structured(applicant: Dict, top_n: int = 3) -> List[dict]:
    """Same computation as explain_applicant(), but returns structured, JSON-serializable
    dicts instead of pre-formatted strings — used by api.py's /predict endpoint.
    """
    return _top_shap_contributions(applicant, top_n)


def compute_global_shap_values(df: pd.DataFrame, max_rows: int = 300, random_state: int = 42):
    """Compute SHAP values for a sample of applicants, for a global summary/beeswarm plot.

    Returns (shap_values, X_transformed, feature_names) so the caller (app.py's
    Explainability tab) can pass them straight to shap.summary_plot without this
    module taking a matplotlib dependency itself.
    """
    bundle, explainer = _load()
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = df.copy()
    df = build_aggregate_features(df)
    df = engineer_features(df)
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None
    df = df[numeric_cols + categorical_cols]

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state)

    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(df)
    feature_names = get_output_feature_names(preprocessor, numeric_cols, categorical_cols)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    log.info(f"Computed global SHAP values for {len(df)} sampled applicants")
    return np.asarray(shap_values), X_transformed, feature_names


if __name__ == "__main__":
    from src.data.loader import build_joined_dataset

    df = build_joined_dataset(is_train=True)
    sample = df.iloc[0].to_dict()
    explanation = explain_applicant_full(sample)
    print(f"Raw probability: {explanation.raw_probability:.4f}")
    print(f"Calibrated probability: {explanation.calibrated_probability}")
    print(f"Risk band: {explanation.risk_band}")
    print(f"Decision: {explanation.decision}")
    print(f"SHAP explanation target: {explanation.shap_explanation_target}")
    print("\nTop risk-increasing drivers:")
    for c in explanation.top_risk_increasing:
        print(f"  {c['source_feature']} ({c.get('category')}) = {c['display_value']} "
              f"| SHAP={c['shap_value']:.4f} | {c['business_meaning']}")
    print("\nTop risk-reducing drivers:")
    for c in explanation.top_risk_reducing:
        print(f"  {c['source_feature']} ({c.get('category')}) = {c['display_value']} "
              f"| SHAP={c['shap_value']:.4f} | {c['business_meaning']}")
