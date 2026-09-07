"""Per-prediction SHAP explanations: which features drove one applicant's risk score, in plain language."""
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocessor import build_aggregate_features, engineer_features, get_output_feature_names
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")

_explainer_cache = None
_bundle_cache = None


def _load():
    global _explainer_cache, _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = joblib.load(MODEL_PATH)
        model = _bundle_cache["pipeline"].named_steps["model"]
        _explainer_cache = shap.TreeExplainer(model)
        log.info("SHAP TreeExplainer initialized")
    return _bundle_cache, _explainer_cache


def _plain_language(feature: str, shap_value: float, feature_value) -> str:
    direction = "increased" if shap_value > 0 else "decreased"
    magnitude = abs(shap_value)
    readable = feature.replace("_", " ").title()
    return f"{readable} = {feature_value} {direction} risk (impact {magnitude:.3f})"


def explain_applicant(applicant: Dict, top_n: int = 5) -> List[str]:
    """Return the top_n features driving this applicant's prediction, as plain-language strings."""
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
    feature_names = get_output_feature_names(preprocessor, numeric_cols)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # binary classifiers sometimes return [class0, class1]
        shap_values = shap_values[1]
    row_shap = np.asarray(shap_values)[0]

    order = np.argsort(-np.abs(row_shap))[:top_n]
    raw_row = df.iloc[0]

    explanations = []
    for idx in order:
        fname = feature_names[idx]
        base_col = fname.split("_")[0] if fname not in raw_row.index else fname
        display_val = raw_row[fname] if fname in raw_row.index else round(float(X_transformed[0][idx]), 3)
        explanations.append(_plain_language(fname, row_shap[idx], display_val))

    return explanations


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
    feature_names = get_output_feature_names(preprocessor, numeric_cols)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    log.info(f"Computed global SHAP values for {len(df)} sampled applicants")
    return np.asarray(shap_values), X_transformed, feature_names


if __name__ == "__main__":
    from src.data.loader import build_joined_dataset

    df = build_joined_dataset(is_train=True)
    sample = df.iloc[0].to_dict()
    for line in explain_applicant(sample):
        print(line)
