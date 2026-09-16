"""Unit tests for the Milestone 2 data-validation module (src/data/validation.py).

All fixtures are small synthetic DataFrames — no real Home Credit CSVs required.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import validation as v


def _valid_synthetic_df(n=50, seed=0):
    """A well-formed synthetic applicant-level dataset: unique IDs, no TARGET
    leakage, no infinities, no constant columns, features weakly correlated
    with TARGET — should pass every check cleanly.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1),
        "TARGET": rng.integers(0, 2, size=n),
        "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, size=n),
        "AMT_CREDIT": rng.uniform(10000, 900000, size=n),
        "CREDIT_INCOME_RATIO": rng.uniform(0.1, 5.0, size=n),
        "CODE_GENDER": rng.choice(["M", "F"], size=n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary"], size=n),
    })


# ---- Test 1: TARGET cannot appear in predictor features ---------------------

def test_target_leakage_detected_when_present():
    result = v.check_target_not_in_features(["AMT_INCOME_TOTAL", "TARGET", "CODE_GENDER"])
    assert result.status == v.ValidationStatus.FAIL


def test_target_leakage_passes_when_absent():
    result = v.check_target_not_in_features(["AMT_INCOME_TOTAL", "CODE_GENDER"])
    assert result.status == v.ValidationStatus.PASS


# ---- Test 2: duplicate applicant IDs are detected ---------------------------

def test_duplicate_applicant_ids_detected():
    df = pd.DataFrame({"SK_ID_CURR": [1, 2, 2, 3], "AMT_INCOME_TOTAL": [1.0, 2.0, 2.0, 3.0]})
    result = v.check_applicant_id_uniqueness(df)
    assert result.status == v.ValidationStatus.FAIL
    assert result.stats["duplicate_rows"] == 1


def test_unique_applicant_ids_pass():
    df = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "AMT_INCOME_TOTAL": [1.0, 2.0, 3.0]})
    result = v.check_applicant_id_uniqueness(df)
    assert result.status == v.ValidationStatus.PASS


# ---- Test 3: historical joins cannot silently create multiple rows ---------

def test_row_count_change_across_join_fails():
    result = v.check_row_count_preserved(before=100, after=150, step_name="bureau_join")
    assert result.status == v.ValidationStatus.FAIL


def test_row_count_preserved_across_join_passes():
    result = v.check_row_count_preserved(before=100, after=100, step_name="bureau_join")
    assert result.status == v.ValidationStatus.PASS


def test_aggregate_table_grain_detects_many_to_many():
    # Two rows share SK_ID_CURR=1 -> not one-row-per-applicant.
    agg = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "BUREAU_COUNT": [3, 3, 1]})
    result = v.check_aggregate_table_grain(agg)
    assert result.status == v.ValidationStatus.FAIL


def test_aggregate_table_grain_passes_when_one_row_per_id():
    agg = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "BUREAU_COUNT": [3, 0, 1]})
    result = v.check_aggregate_table_grain(agg)
    assert result.status == v.ValidationStatus.PASS


# ---- Test 4: infinite values are detected -----------------------------------

def test_infinite_values_detected():
    df = pd.DataFrame({"RATIO_COL": [1.0, np.inf, 3.0], "OTHER": [1.0, 2.0, 3.0]})
    result = v.check_infinite_values(df, ["RATIO_COL", "OTHER"])
    assert result.status == v.ValidationStatus.FAIL
    assert "RATIO_COL" in result.stats["offending_columns"]


def test_no_infinite_values_passes():
    df = pd.DataFrame({"RATIO_COL": [1.0, 2.0, 3.0]})
    result = v.check_infinite_values(df, ["RATIO_COL"])
    assert result.status == v.ValidationStatus.PASS


# ---- Test 5: constant features are detected ---------------------------------

def test_constant_feature_detected():
    df = pd.DataFrame({"CONST": [5, 5, 5, 5], "VARIES": [1, 2, 3, 4]})
    result = v.check_constant_features(df, ["CONST", "VARIES"])
    assert result.status == v.ValidationStatus.WARNING
    assert "CONST" in result.stats["constant_columns"]
    assert "VARIES" not in result.stats["constant_columns"]


def test_no_constant_features_passes():
    df = pd.DataFrame({"VARIES": [1, 2, 3, 4]})
    result = v.check_constant_features(df, ["VARIES"])
    assert result.status == v.ValidationStatus.PASS


# ---- Test 6: feature registry mismatches are detected -----------------------

def test_registry_mismatch_detected_when_feature_not_generated():
    generated_columns = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL"]  # missing all registered features
    result = v.check_registry_matches_generated_columns(generated_columns)
    assert result.status == v.ValidationStatus.FAIL
    assert len(result.stats["missing_from_output"]) > 0


def test_registry_matches_when_all_features_present():
    from src.data.feature_registry import get_registered_feature_names
    generated_columns = ["SK_ID_CURR", "TARGET"] + get_registered_feature_names()
    result = v.check_registry_matches_generated_columns(generated_columns)
    assert result.status == v.ValidationStatus.PASS


def test_registry_no_duplicates_on_real_registry():
    result = v.check_registry_no_duplicates()
    assert result.status == v.ValidationStatus.PASS


def test_registry_entries_complete_on_real_registry():
    result = v.check_registry_entries_complete()
    assert result.status == v.ValidationStatus.PASS


# ---- Test 7: a synthetic leakage feature is correctly identified -----------

def test_synthetic_leakage_feature_flagged_by_correlation_check():
    n = 200
    rng = np.random.default_rng(1)
    target = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "TARGET": target,
        # A blatant leak: directly derived from TARGET (plus a hair of noise so
        # it isn't a literal copy, which would be immediately obvious anyway).
        "LEAKY_FEATURE": target * 100 + rng.normal(0, 0.01, size=n),
        "LEGITIMATE_FEATURE": rng.uniform(0, 1, size=n),  # unrelated to TARGET
    })
    result = v.check_suspicious_target_correlation(df, ["LEAKY_FEATURE", "LEGITIMATE_FEATURE"])
    assert result.status == v.ValidationStatus.WARNING
    assert "LEAKY_FEATURE" in result.stats["suspicious_columns"]
    assert "LEGITIMATE_FEATURE" not in result.stats["suspicious_columns"]


def test_no_suspicious_correlation_on_unrelated_features():
    n = 200
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "TARGET": rng.integers(0, 2, size=n),
        "UNRELATED": rng.uniform(0, 1, size=n),
    })
    result = v.check_suspicious_target_correlation(df, ["UNRELATED"])
    assert result.status == v.ValidationStatus.PASS


# ---- Test 8: validation distinguishes PASS / WARNING / FAIL -----------------

def test_summarize_distinguishes_statuses():
    results = [
        v.ValidationResult("a", v.ValidationStatus.PASS, "ok"),
        v.ValidationResult("b", v.ValidationStatus.WARNING, "hmm"),
        v.ValidationResult("c", v.ValidationStatus.WARNING, "hmm2"),
    ]
    summary = v.summarize(results)
    assert summary["overall_status"] == "WARNING"
    assert summary["passed"] == 1
    assert summary["warnings"] == 2
    assert summary["failed"] == 0


def test_summarize_overall_fail_when_any_failure_present():
    results = [
        v.ValidationResult("a", v.ValidationStatus.PASS, "ok"),
        v.ValidationResult("b", v.ValidationStatus.FAIL, "bad"),
    ]
    summary = v.summarize(results)
    assert summary["overall_status"] == "FAIL"
    assert summary["failed"] == 1


# ---- Test 9: critical validation failure prevents training ------------------

def test_raise_if_failed_raises_on_critical_failure():
    results = [
        v.ValidationResult("target_leakage_direct", v.ValidationStatus.FAIL, "TARGET leaked"),
    ]
    with pytest.raises(v.DataValidationError):
        v.raise_if_failed(results)


def test_raise_if_failed_does_not_raise_on_warning_only():
    results = [
        v.ValidationResult("missingness", v.ValidationStatus.WARNING, "some missing"),
        v.ValidationResult("target_leakage_direct", v.ValidationStatus.PASS, "clean"),
    ]
    v.raise_if_failed(results)  # must not raise


# ---- Test 10: a valid synthetic dataset passes validation -------------------

def test_valid_synthetic_dataset_passes_full_suite():
    df = _valid_synthetic_df()
    numeric_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "CREDIT_INCOME_RATIO"]
    categorical_cols = ["CODE_GENDER", "NAME_EDUCATION_TYPE"]

    results = [
        v.check_target_not_in_features(numeric_cols + categorical_cols),
        v.check_applicant_id_uniqueness(df),
        v.check_exact_duplicate_rows(df),
        v.check_infinite_values(df, numeric_cols),
        v.check_missingness(df, numeric_cols + categorical_cols),
        v.check_constant_features(df, numeric_cols),
        v.check_categorical_cardinality(df, categorical_cols),
        v.check_unexpected_dtypes(df, numeric_cols, categorical_cols),
    ]
    statuses = {r.check_name: r.status for r in results}
    for name, status in statuses.items():
        assert status != v.ValidationStatus.FAIL, f"{name} unexpectedly failed on a clean synthetic dataset"

    summary = v.summarize(results)
    assert summary["failed"] == 0


# ---- generate_markdown_report ------------------------------------------------

def test_generate_markdown_report_writes_file(tmp_path):
    results = [
        v.ValidationResult("check_a", v.ValidationStatus.PASS, "all good", {"n": 10}),
        v.ValidationResult("check_b", v.ValidationStatus.WARNING, "hmm", {"cols": ["x"]}),
    ]
    output_path = str(tmp_path / "report.md")
    text = v.generate_markdown_report(results, output_path)

    assert "Data Validation Report" in text
    assert "check_a" in text
    assert "check_b" in text
    with open(output_path) as f:
        saved = f.read()
    assert saved == text
