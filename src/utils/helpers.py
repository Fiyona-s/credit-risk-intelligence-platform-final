"""Small shared helpers used across ml/, explainability/, and app.py."""
import os


def risk_band(probability: float, low_threshold: float, medium_threshold: float) -> str:
    """Map a default probability to a business-readable risk band."""
    if probability < low_threshold:
        return "Low"
    if probability < medium_threshold:
        return "Medium"
    return "High"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
