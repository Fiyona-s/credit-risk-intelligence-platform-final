"""Inference: given applicant features, output a default-probability risk score and a risk band."""
import os
from typing import Dict, Union

import joblib
import pandas as pd

from src.data.preprocessor import build_aggregate_features, engineer_features
from src.utils.config import settings
from src.utils.helpers import risk_band
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python -m src.ml.train` first."
            )
        _model_cache = joblib.load(MODEL_PATH)
        log.info(f"Loaded model from {MODEL_PATH}")
    return _model_cache


def predict_applicant(applicant: Union[Dict, pd.DataFrame]) -> dict:
    """Score a single applicant (dict of raw feature values or a one-row DataFrame)."""
    bundle = load_model()
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    df = pd.DataFrame([applicant]) if isinstance(applicant, dict) else applicant.copy()
    # Same aggregation code path as training (src.data.preprocessor.build_aggregate_features),
    # so a real applicant's SK_ID_CURR pulls their actual bureau/previous-application
    # history instead of falling back to imputed defaults for those columns.
    df = build_aggregate_features(df)
    df = engineer_features(df)

    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = None
    df = df[numeric_cols + categorical_cols]

    probability = float(pipeline.predict_proba(df)[:, 1][0])
    band = risk_band(probability, settings.risk_threshold_low, settings.risk_threshold_medium)

    return {
        "probability_of_default": probability,
        "risk_band": band,
        "features_used": df,
    }


if __name__ == "__main__":
    from src.data.loader import build_joined_dataset

    df = build_joined_dataset(is_train=True)
    sample = df.iloc[0].to_dict()
    result = predict_applicant(sample)
    print(f"Probability of default: {result['probability_of_default']:.4f}")
    print(f"Risk band: {result['risk_band']}")
    print(f"Actual TARGET: {sample['TARGET']}")
