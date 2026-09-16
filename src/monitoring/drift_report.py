"""Model/data monitoring via Evidently: data drift (feature distributions) and
target drift (TARGET distribution), comparing a saved training-time reference
snapshot against a current batch of applicants.

API note: this uses evidently>=0.7 (the currently-installed/pinned version). Older
Evidently releases (<=0.4) had a different API built around
`evidently.dashboard.Dashboard` with `DataDriftTab`/`CatTargetDriftTab`, or
`evidently.report.Report` with `metric_preset=[DataDriftPreset(), TargetDriftPreset()]`
— none of which exist in 0.7.x. The modern equivalent used here: build `Dataset`
objects with an explicit `DataDefinition`, run a `Report(metrics=[DataDriftPreset()])`
against (a) the feature columns for "data drift" and (b) the TARGET column alone for
"target drift" — two separate reports/HTML files, matching the spirit of the
original DataDriftReport/TargetDriftReport split.
"""
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from src.data.preprocessor import get_feature_columns
from src.utils.logger import get_logger

log = get_logger(__name__)

REPORTS_DIR = "reports"
REFERENCE_DATA_PATH = os.path.join(REPORTS_DIR, "reference_data.parquet")

DATA_DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "data_drift_report.html")
TARGET_DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "target_drift_report.html")


def _split_columns_for_evidently(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Evidently's DataDefinition wants numeric vs. categorical columns declared explicitly.
    Boolean columns (e.g. HAS_BUREAU_HISTORY) aren't picked up by preprocessor.py's
    get_feature_columns() numeric check (np.number excludes bool) and aren't object
    dtype either, so they'd silently fall through both lists — route them into
    categorical here instead of dropping them from the drift check.
    """
    bool_cols = [c for c in df.columns if df[c].dtype == bool and c not in numeric_cols and c not in categorical_cols]
    return numeric_cols, categorical_cols + bool_cols


def _align_columns(reference: pd.DataFrame, current: pd.DataFrame, columns: List[str]) -> List[str]:
    """Only compare columns present in both dataframes — the current batch (e.g. the
    synthetic fallback dataset, or a smaller live batch) may not have every column
    the reference training set had.
    """
    aligned = [c for c in columns if c in reference.columns and c in current.columns]
    missing = set(columns) - set(aligned)
    if missing:
        log.warning(f"Dropping {len(missing)} column(s) from the drift check, absent from one side: {sorted(missing)}")
    return aligned


def generate_data_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str = DATA_DRIFT_REPORT_PATH
):
    """Feature-level drift: compares the distribution of every numeric/categorical
    feature column (TARGET and ID columns excluded) between reference and current.
    Saves HTML to output_path and returns the Evidently snapshot object (its
    .dict()/.json() carries the per-column drift scores — see get_top_drifted_features()).
    """
    numeric_cols, categorical_cols = get_feature_columns(reference)
    numeric_cols, categorical_cols = _split_columns_for_evidently(reference, numeric_cols, categorical_cols)

    numeric_cols = _align_columns(reference, current, numeric_cols)
    categorical_cols = _align_columns(reference, current, categorical_cols)
    all_cols = numeric_cols + categorical_cols

    data_definition = DataDefinition(numerical_columns=numeric_cols, categorical_columns=categorical_cols)
    ref_dataset = Dataset.from_pandas(reference[all_cols], data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(current[all_cols], data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    snapshot.save_html(output_path)
    log.info(f"Saved data drift report to {output_path} ({len(all_cols)} columns compared)")
    return snapshot


def generate_target_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output_path: str = TARGET_DRIFT_REPORT_PATH
) -> Optional[object]:
    """TARGET-only drift: has the distribution of TARGET itself shifted between
    reference and current (e.g. the current batch's default rate looks different
    from what the model was trained on)? Returns None if either side lacks TARGET
    (e.g. a live scoring batch with no known outcome yet — this is expected, not
    an error: target drift can only be checked once outcomes are known).
    """
    if "TARGET" not in reference.columns or "TARGET" not in current.columns:
        log.warning("TARGET not present in reference and/or current data — skipping target drift report "
                    "(expected for a live scoring batch with unknown outcomes yet).")
        return None

    data_definition = DataDefinition(categorical_columns=["TARGET"])
    ref_dataset = Dataset.from_pandas(reference[["TARGET"]], data_definition=data_definition)
    cur_dataset = Dataset.from_pandas(current[["TARGET"]], data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    snapshot.save_html(output_path)
    log.info(f"Saved target drift report to {output_path}")
    return snapshot


def get_top_drifted_features(snapshot, top_n: int = 10) -> pd.DataFrame:
    """Parse an Evidently snapshot's per-column ValueDrift metrics into a ranked
    DataFrame: feature, p_value, drifted (p_value < 0.05), sorted by p_value
    ascending drift severity (most-drifted first).

    Evidently auto-selects a different test method per column depending on its
    type/cardinality — e.g. "K-S p_value" or "Z-test p_value" for some columns,
    "Jensen-Shannon distance"/"Wasserstein distance"/PSI for others. These are NOT
    interchangeable: p-value-style methods signal drift when the value is BELOW
    the threshold (smaller p-value = more evidence of a real difference), while
    distance/divergence-style methods signal drift when the value is ABOVE the
    threshold (larger distance = more different). Treating every method as a flat
    "value < 0.05" p-value check — an earlier version of this function did — was
    confirmed wrong against real data: TARGET's own drift check used "Jensen-Shannon
    distance" (value ~3e-7, threshold 0.1), which is NOT drifted, but a blanket
    "< 0.05" check happened to flag it as drifted anyway, for the wrong reason.
    The method name and threshold are read directly out of Evidently's own
    metric_name string here rather than assumed, so this stays correct as
    Evidently changes its auto-selected method per column/dataset.
    """
    rows = []
    for metric in snapshot.dict().get("metrics", []):
        name = metric.get("metric_name", "")
        match = re.match(r"ValueDrift\(column=(.*?),method=(.*?),threshold=([\d.eE+-]+)\)", name)
        if not match:
            continue
        column, method, threshold_str = match.groups()
        value = metric.get("value")
        if not isinstance(value, (int, float)):
            continue
        threshold = float(threshold_str)
        is_p_value_method = "p_value" in method.lower()
        drifted = value < threshold if is_p_value_method else value > threshold
        # Sort key: for p-value methods, more negative (smaller) value = more severe;
        # for distance methods, larger value = more severe. Comparing severity scores
        # *across* different methods is inherently approximate (different methods,
        # different scales) — good enough for "most drifted first" ranking, not a
        # precise cross-method ordering.
        severity = -value if is_p_value_method else value
        rows.append({
            "feature": column, "method": method, "value": value,
            "threshold": threshold, "drifted": drifted, "_severity": severity,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("_severity", ascending=False).drop(columns="_severity").reset_index(drop=True)
    return result.head(top_n)


def save_reference_snapshot(df: pd.DataFrame, path: str = REFERENCE_DATA_PATH) -> None:
    """Save the training dataframe as the reference distribution for future drift checks."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    log.info(f"Saved reference data snapshot ({df.shape[0]} rows, {df.shape[1]} cols) to {path}")


def load_reference_snapshot(path: str = REFERENCE_DATA_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def run_drift_monitoring(
    reference: pd.DataFrame, current: pd.DataFrame, reports_dir: str = REPORTS_DIR
) -> dict:
    """Full monitoring pass: both reports saved to reports_dir, plus a ranked
    top-drifted-features table for a quick summary (e.g. the Streamlit Monitoring tab).
    """
    data_drift_path = os.path.join(reports_dir, "data_drift_report.html")
    target_drift_path = os.path.join(reports_dir, "target_drift_report.html")

    data_drift_snapshot = generate_data_drift_report(reference, current, data_drift_path)
    target_drift_snapshot = generate_target_drift_report(reference, current, target_drift_path)

    top_drifted = get_top_drifted_features(data_drift_snapshot)

    target_drifted = None
    if target_drift_snapshot is not None:
        target_rows = get_top_drifted_features(target_drift_snapshot, top_n=1)
        target_drifted = bool(target_rows.iloc[0]["drifted"]) if len(target_rows) else None

    return {
        "data_drift_report_path": data_drift_path,
        "target_drift_report_path": target_drift_path,
        "top_drifted_features": top_drifted,
        "target_drifted": target_drifted,
    }


if __name__ == "__main__":
    import argparse

    from src.data.loader import build_joined_dataset
    from src.data.preprocessor import engineer_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", default=REFERENCE_DATA_PATH,
        help="Path to the reference_data.parquet snapshot (default: reports/reference_data.parquet, "
             "written by `python -m src.ml.train --monitor`)",
    )
    args = parser.parse_args()

    reference_df = load_reference_snapshot(args.reference)
    if reference_df is None:
        raise SystemExit(
            f"No reference snapshot found at {args.reference}. Run `python -m src.ml.train --monitor` first."
        )

    current_df = build_joined_dataset(is_train=True)
    current_df = engineer_features(current_df)

    result = run_drift_monitoring(reference_df, current_df)
    print(f"\nData drift report: {result['data_drift_report_path']}")
    print(f"Target drift report: {result['target_drift_report_path']}")
    print(f"Target drifted: {result['target_drifted']}")
    print("\nTop drifted features:")
    print(result["top_drifted_features"].to_string(index=False))
