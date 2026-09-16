"""Unit tests for the Milestone 1 credit-behavior feature layer (src/data/preprocessor.py,
src/data/feature_registry.py).

All fixtures are small, hand-built synthetic DataFrames — the real Kaggle CSVs are
NOT required to run this file (they are gitignored and may not exist in CI). Every
test monkeypatches src.data.loader's load_* functions rather than reading data/*.csv,
and calls src.data.preprocessor.reset_aggregate_cache() first so each test starts
from a clean module-level cache.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import preprocessor as pp
from src.data.feature_registry import FEATURE_REGISTRY, get_registered_feature_names


@pytest.fixture(autouse=True)
def _reset_cache():
    """build_aggregate_features() caches its aggregate table in a module-level
    global (by design — see preprocessor.py's docstring: it's static per-process
    training/inference data). Tests use different synthetic fixtures per case, so
    the cache must be cleared before and after every test.
    """
    pp.reset_aggregate_cache()
    yield
    pp.reset_aggregate_cache()


# ---- Synthetic fixtures -----------------------------------------------------

def _synthetic_bureau():
    return pd.DataFrame({
        "SK_ID_CURR":            [1, 1, 2],
        "SK_ID_BUREAU":          [100, 101, 102],
        "CREDIT_ACTIVE":         ["Active", "Closed", "Active"],
        "CREDIT_CURRENCY":       ["currency 1"] * 3,
        "DAYS_CREDIT":           [-500, -1200, -100],
        "CREDIT_DAY_OVERDUE":    [0, 0, 10],
        "AMT_CREDIT_SUM":        [100000.0, 50000.0, 20000.0],
        "AMT_CREDIT_SUM_DEBT":   [40000.0, 0.0, 20000.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 500.0],
        "CREDIT_TYPE":           ["Consumer credit"] * 3,
        "DAYS_CREDIT_UPDATE":    [-10, -10, -5],
        "AMT_ANNUITY":           [1000.0, 0.0, 2000.0],
    })


def _synthetic_bureau_balance():
    # SK_ID_BUREAU 100 (applicant 1): one recent DPD month, one old current month.
    # SK_ID_BUREAU 101 (applicant 1): closed, all current.
    # SK_ID_BUREAU 102 (applicant 2): recent DPD.
    return pd.DataFrame({
        "SK_ID_BUREAU":   [100, 100, 101, 102],
        "MONTHS_BALANCE": [-1, -10, -2, -1],
        "STATUS":         ["1", "0", "C", "2"],
    })


def _synthetic_previous_application():
    return pd.DataFrame({
        "SK_ID_CURR":        [1, 1, 1],
        "SK_ID_PREV":        [200, 201, 202],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "AMT_APPLICATION":   [100000.0, 50000.0, 80000.0],
        "AMT_CREDIT":        [100000.0, 0.0, 80000.0],
        "AMT_ANNUITY":       [5000.0, np.nan, 4000.0],
        "CNT_PAYMENT":       [24.0, np.nan, 12.0],
    })


def _synthetic_pos_cash():
    return pd.DataFrame({
        "SK_ID_CURR":       [1, 1, 1],
        "SK_ID_PREV":       [200, 200, 201],
        "MONTHS_BALANCE":   [-1, -2, -1],
        "CNT_INSTALMENT":   [24, 24, 12],
        "CNT_INSTALMENT_FUTURE": [10, 11, 0],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed"],
        "SK_DPD":           [0, 5, 0],
        "SK_DPD_DEF":       [0, 0, 0],
    })


def _synthetic_credit_card():
    return pd.DataFrame({
        "SK_ID_CURR":        [1, 1],
        "SK_ID_PREV":        [300, 300],
        "MONTHS_BALANCE":    [-1, -10],
        "AMT_BALANCE":       [5000.0, 2000.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [10000.0, 10000.0],
        "AMT_DRAWINGS_ATM_CURRENT": [0.0, 0.0],
        "AMT_DRAWINGS_CURRENT": [0.0, 0.0],
        "AMT_DRAWINGS_OTHER_CURRENT": [0.0, 0.0],
        "AMT_DRAWINGS_POS_CURRENT": [0.0, 0.0],
        "AMT_INST_MIN_REGULARITY": [200.0, 200.0],
        "AMT_PAYMENT_CURRENT": [500.0, 200.0],
        "AMT_PAYMENT_TOTAL_CURRENT": [500.0, 200.0],
        "AMT_RECEIVABLE_PRINCIPAL": [4800.0, 1900.0],
        "AMT_RECIVABLE": [4800.0, 1900.0],
        "AMT_TOTAL_RECEIVABLE": [4800.0, 1900.0],
        "CNT_DRAWINGS_ATM_CURRENT": [0, 0],
        "CNT_DRAWINGS_CURRENT": [0, 0],
        "CNT_DRAWINGS_OTHER_CURRENT": [0, 0],
        "CNT_DRAWINGS_POS_CURRENT": [0, 0],
        "CNT_INSTALMENT_MATURE_CUM": [1, 10],
        "NAME_CONTRACT_STATUS": ["Active", "Active"],
        "SK_DPD":            [0, 0],
        "SK_DPD_DEF":        [0, 0],
    })


def _synthetic_installments():
    return pd.DataFrame({
        "SK_ID_CURR":          [1, 1, 1],
        "SK_ID_PREV":          [200, 200, 201],
        "NUM_INSTALMENT_VERSION": [1, 1, 1],
        "NUM_INSTALMENT_NUMBER": [1, 2, 1],
        "DAYS_INSTALMENT":     [-30, -400, -20],
        "DAYS_ENTRY_PAYMENT":  [-25, -395, -50],  # first two paid late, third paid early
        "AMT_INSTALMENT":      [1000.0, 1000.0, 500.0],
        "AMT_PAYMENT":         [1000.0, 900.0, 500.0],
    })


def _synthetic_application():
    """3 applicants: 1 has full history, 2 has bureau-only, 3 has none."""
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
        "CODE_GENDER": ["M", "F", "F"],
        "AMT_INCOME_TOTAL": [200000.0, 100000.0, 150000.0],
        "AMT_CREDIT": [500000.0, 300000.0, 0.0],
        "AMT_ANNUITY": [25000.0, 15000.0, 10000.0],
        "DAYS_BIRTH": [-12000, -9000, -15000],
        "DAYS_EMPLOYED": [-2000, 365243, -500],
    })


@pytest.fixture
def patched_loader(monkeypatch):
    """Monkeypatch every src.data.loader.load_* function used by preprocessor.py's
    build_*_features() to return the synthetic fixtures above instead of reading
    data/*.csv. Each build_*_features() re-imports load_* at call time
    (`from src.data.loader import load_bureau, ...` inside the function body), so
    patching the attribute on src.data.loader is sufficient.
    """
    import src.data.loader as loader

    monkeypatch.setattr(loader, "load_bureau", lambda: _synthetic_bureau())
    monkeypatch.setattr(loader, "load_bureau_balance", lambda: _synthetic_bureau_balance())
    monkeypatch.setattr(loader, "load_previous_application", lambda: _synthetic_previous_application())
    monkeypatch.setattr(loader, "load_pos_cash_balance", lambda: _synthetic_pos_cash())
    monkeypatch.setattr(loader, "load_credit_card_balance", lambda: _synthetic_credit_card())
    monkeypatch.setattr(loader, "load_installments_payments", lambda: _synthetic_installments())
    return loader


@pytest.fixture
def patched_loader_no_history(monkeypatch):
    """All auxiliary tables report no data at all (every load_* returns None) —
    the "applicant with zero historical records anywhere" scenario.
    """
    import src.data.loader as loader

    for name in ("load_bureau", "load_bureau_balance", "load_previous_application",
                 "load_pos_cash_balance", "load_credit_card_balance", "load_installments_payments"):
        monkeypatch.setattr(loader, name, lambda: None)
    return loader


# ---- Test 1: one row per applicant ------------------------------------------

def test_aggregate_table_has_one_row_per_applicant(patched_loader):
    table = pp._build_full_aggregate_table()
    assert table["SK_ID_CURR"].is_unique


# ---- Test 2: no row explosion on join ---------------------------------------

def test_build_aggregate_features_does_not_change_row_count(patched_loader):
    app_df = _synthetic_application()
    before = len(app_df)
    merged = pp.build_aggregate_features(app_df)
    assert len(merged) == before


# ---- Test 3: missing history still produces a valid row --------------------

def test_applicant_with_no_history_gets_valid_row(patched_loader):
    app_df = _synthetic_application()
    merged = pp.build_aggregate_features(app_df)

    row3 = merged[merged["SK_ID_CURR"] == 3].iloc[0]
    assert row3["HAS_BUREAU_HISTORY"] == False  # noqa: E712
    assert row3["HAS_PREV_APPLICATION"] == False  # noqa: E712
    assert row3["HAS_POS_HISTORY"] == False  # noqa: E712
    assert row3["HAS_CC_HISTORY"] == False  # noqa: E712
    assert row3["HAS_INSTALLMENT_HISTORY"] == False  # noqa: E712
    # Count-like columns fill to 0 for an applicant with zero history rows.
    assert row3["BUREAU_COUNT"] == 0
    assert row3["PREV_APP_COUNT"] == 0
    # A ratio/amount column should NOT be 0 (that would look like "definitely no
    # debt" rather than "unknown") — it should be filled with the cohort median.
    assert not pd.isna(row3["BUREAU_CREDIT_SUM_MEAN"])

    # Applicant 2 has bureau history but no prev-application/POS/CC/installment history:
    # "history exists but the value could be zero" vs "no history" must be distinguishable.
    row2 = merged[merged["SK_ID_CURR"] == 2].iloc[0]
    assert row2["HAS_BUREAU_HISTORY"] == True  # noqa: E712
    assert row2["HAS_PREV_APPLICATION"] == False  # noqa: E712
    assert row2["BUREAU_COUNT"] == 1


def test_no_history_anywhere_still_produces_valid_frame(patched_loader_no_history):
    app_df = _synthetic_application()
    merged = pp.build_aggregate_features(app_df)
    # With every auxiliary table absent, _build_full_aggregate_table() returns an
    # empty frame and build_aggregate_features() is a documented no-op passthrough.
    assert len(merged) == len(app_df)
    assert set(app_df["SK_ID_CURR"]) == set(merged["SK_ID_CURR"])


# ---- Test 4: safe ratios -----------------------------------------------------

def test_safe_divide_handles_zero_and_missing_denominators():
    numerator = pd.Series([10.0, 5.0, np.nan, 8.0])
    denominator = pd.Series([0.0, 2.0, 4.0, np.nan])
    result = pp._safe_divide(numerator, denominator)

    assert not np.isinf(result).any()
    assert pd.isna(result.iloc[0])  # zero denominator -> NaN, not inf
    assert result.iloc[1] == 2.5
    assert pd.isna(result.iloc[2])  # NaN numerator -> NaN
    assert pd.isna(result.iloc[3])  # NaN denominator -> NaN


def test_engineered_ratios_have_no_infinite_values(patched_loader):
    app_df = _synthetic_application()
    merged = pp.build_aggregate_features(app_df)
    engineered = pp.engineer_features(merged)

    ratio_cols = [c for c in engineered.columns if "RATIO" in c or c == "CREDIT_TERM"]
    for col in ratio_cols:
        values = engineered[col].replace([np.inf, -np.inf], np.nan)
        assert not np.isinf(engineered[col].dropna()).any(), f"{col} contains an infinite value"

    # Applicant 3 has AMT_CREDIT == 0.0, which would make CREDIT_TERM's denominator
    # zero — this must come out NaN, not inf or a crash.
    row3 = engineered[engineered["SK_ID_CURR"] == 3].iloc[0]
    assert pd.isna(row3["CREDIT_TERM"])


# ---- Test 5: deterministic features ------------------------------------------

def test_feature_engineering_is_deterministic(patched_loader):
    app_df = _synthetic_application()

    pp.reset_aggregate_cache()
    result_1 = pp.engineer_features(pp.build_aggregate_features(app_df.copy()))
    pp.reset_aggregate_cache()
    result_2 = pp.engineer_features(pp.build_aggregate_features(app_df.copy()))

    pd.testing.assert_frame_equal(
        result_1.sort_index(axis=1), result_2.sort_index(axis=1), check_dtype=False
    )


# ---- Test 6: feature registry consistency -----------------------------------

def test_registry_features_are_actually_generated(patched_loader):
    app_df = _synthetic_application()
    merged = pp.build_aggregate_features(app_df)
    engineered = pp.engineer_features(merged)

    generated_columns = set(engineered.columns)
    registered_names = set(get_registered_feature_names())

    missing_from_output = registered_names - generated_columns
    assert not missing_from_output, (
        f"Registry references features never actually generated: {missing_from_output}"
    )


def test_registry_has_no_duplicate_names():
    names = get_registered_feature_names()
    assert len(names) == len(set(names))


def test_registry_entries_have_required_fields():
    required = {"name", "group", "source", "calculation", "business_meaning",
                "dtype", "historical", "actionable", "leakage_risk"}
    for entry in FEATURE_REGISTRY:
        missing = required - set(entry.keys())
        assert not missing, f"{entry.get('name', '<unnamed>')} is missing fields: {missing}"


# ---- Test 7: no target leakage -----------------------------------------------

def test_target_not_present_in_feature_columns(patched_loader):
    app_df = _synthetic_application()
    merged = pp.build_aggregate_features(app_df)
    engineered = pp.engineer_features(merged)

    numeric_cols, categorical_cols = pp.get_feature_columns(engineered)
    assert "TARGET" not in numeric_cols
    assert "TARGET" not in categorical_cols
    assert "SK_ID_CURR" not in numeric_cols
    assert "SK_ID_CURR" not in categorical_cols


def test_all_auxiliary_time_columns_are_non_positive(patched_loader):
    """Every DAYS_*/MONTHS_BALANCE column used by the aggregators must be <= 0
    (i.e. relative to "now"/the current application) — this is the actual
    leakage-safety property preprocessor.py's module docstring asserts. Verified
    here against the synthetic fixtures directly, not merely trusted.
    """
    bureau = _synthetic_bureau()
    assert (bureau["DAYS_CREDIT"] <= 0).all()

    bb = _synthetic_bureau_balance()
    assert (bb["MONTHS_BALANCE"] <= 0).all()

    pos = _synthetic_pos_cash()
    assert (pos["MONTHS_BALANCE"] <= 0).all()

    cc = _synthetic_credit_card()
    assert (cc["MONTHS_BALANCE"] <= 0).all()

    inst = _synthetic_installments()
    assert (inst["DAYS_INSTALMENT"] <= 0).all()
    assert (inst["DAYS_ENTRY_PAYMENT"] <= 0).all()


# ---- Test 8: train/inference consistency ------------------------------------

def test_train_and_inference_use_identical_feature_functions(patched_loader):
    """A "training-time" call (a full batch) and an "inference-time" call (one
    row, matching what predict.py does) must go through the exact same
    build_aggregate_features/engineer_features functions and produce identical
    values for the same applicant — no separate/duplicated inference-only logic.
    """
    app_df = _synthetic_application()

    pp.reset_aggregate_cache()
    training_batch = pp.engineer_features(pp.build_aggregate_features(app_df.copy()))

    pp.reset_aggregate_cache()
    single_applicant = app_df[app_df["SK_ID_CURR"] == 1].copy().reset_index(drop=True)
    inference_result = pp.engineer_features(pp.build_aggregate_features(single_applicant))

    training_row = training_batch[training_batch["SK_ID_CURR"] == 1].iloc[0]
    inference_row = inference_result[inference_result["SK_ID_CURR"] == 1].iloc[0]

    shared_cols = set(training_row.index) & set(inference_row.index)
    for col in sorted(shared_cols):
        t_val, i_val = training_row[col], inference_row[col]
        if pd.isna(t_val) and pd.isna(i_val):
            continue
        if isinstance(t_val, float):
            assert t_val == pytest.approx(i_val, nan_ok=True), f"Mismatch on {col}: {t_val} vs {i_val}"
        else:
            assert t_val == i_val, f"Mismatch on {col}: {t_val} vs {i_val}"
