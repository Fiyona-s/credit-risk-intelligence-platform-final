"""Extracts simple if/then business rules from the trained model.

Approach: train a shallow decision-tree surrogate to mimic the LightGBM
model's predictions (not the ground-truth TARGET). A shallow tree trained on
the model's own outputs approximates its decision boundary in a form a credit
policy analyst can read directly, without needing any ML background.
"""
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(settings.models_dir, "credit_risk_model.joblib")


def derive_rules(X_sample: pd.DataFrame, max_depth: int = 4, sample_size: int = 20000) -> List[str]:
    """Fit a shallow surrogate tree on (features -> model's predicted class) and render it as rules."""
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    X_sample = X_sample.copy()
    for col in numeric_cols + categorical_cols:
        if col not in X_sample.columns:
            X_sample[col] = None
    X = X_sample[numeric_cols + categorical_cols]
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    model_predictions = model.predict(X_transformed)

    from src.data.preprocessor import get_output_feature_names
    feature_names = get_output_feature_names(preprocessor, numeric_cols)

    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight="balanced")
    surrogate.fit(X_transformed, model_predictions)

    tree_text = export_text(surrogate, feature_names=feature_names, max_depth=max_depth)
    rules = _tree_text_to_business_rules(tree_text)

    log.info(f"Derived {len(rules)} business rules from surrogate tree (depth={max_depth})")
    return rules


def _tree_text_to_business_rules(tree_text: str) -> List[str]:
    """Convert sklearn's export_text output into readable if/then sentences."""
    rules = []
    path_conditions: List[str] = []

    for line in tree_text.splitlines():
        if not line.strip():
            continue
        # export_text prefixes each depth level with one "|" (either "|   " or "|---"),
        # so the count of "|" characters equals depth + 1.
        depth = line.count("|") - 1
        stripped = line.strip("|- ").strip()

        if not stripped:
            continue

        if stripped.startswith("class:"):
            predicted_class = stripped.split(":")[1].strip()
            label = "High Risk (likely default)" if predicted_class == "1" else "Low Risk (likely repay)"
            conditions = " AND ".join(path_conditions[:depth])
            rules.append(f"IF {conditions} THEN predicted outcome = {label}")
        else:
            path_conditions = path_conditions[:depth] + [stripped]

    return rules


if __name__ == "__main__":
    from src.data.loader import build_joined_dataset
    from src.data.preprocessor import engineer_features

    df = build_joined_dataset(is_train=True)
    df = engineer_features(df)
    rules = derive_rules(df)
    for r in rules[:15]:
        print(r)
