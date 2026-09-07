"""Cleaning, imputation, encoding, and feature engineering for the Home Credit dataset.

Exposes a single sklearn-compatible Pipeline (`build_preprocessing_pipeline`) so
train.py, predict.py, and shap_explainer.py all transform data identically.

build_aggregate_features() is the single shared code path (used by both
src.ml.train and src.ml.predict via src.data.loader.build_joined_dataset) that
rolls up the six auxiliary Home Credit history tables to one row per
SK_ID_CURR and left-joins them onto the applicant table. Aggregates are
computed once per process and cached, since they only depend on the
(static) auxiliary CSVs, not on the applicant rows being scored.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)

# DAYS_EMPLOYED has a known Kaggle data-quality anomaly: unemployed/retired
# applicants are encoded as 365243 ("365243 days employed") instead of NaN.
DAYS_EMPLOYED_ANOMALY = 365243

# bureau_balance.csv STATUS is an ordinal delinquency bucket: '0' = current,
# '1'..'5' = escalating days-past-due severity, 'C' = closed, 'X' = unknown.
# All DAYS_*/MONTHS_BALANCE columns in every auxiliary table are <= 0, i.e.
# relative to the *current* application and therefore historical by
# construction — using them as features carries no target leakage.
_BUREAU_BALANCE_STATUS_ORDINAL = {"C": -1, "X": -1, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

_aggregate_cache: Optional[pd.DataFrame] = None
_ratio_medians_cache: Optional[dict] = None


def _aggregate_bureau() -> Optional[pd.DataFrame]:
    from src.data.loader import load_bureau, load_bureau_balance

    bureau = load_bureau()
    if bureau is None:
        return None

    bb = load_bureau_balance()
    if bb is not None:
        bb = bb.copy()
        bb["STATUS_ORD"] = bb["STATUS"].map(_BUREAU_BALANCE_STATUS_ORDINAL)
        bb["IS_DPD_MONTH"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_WORST_STATUS=("STATUS_ORD", "max"),
            BB_DPD_MONTHS=("IS_DPD_MONTH", "sum"),
        ).reset_index()
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    else:
        bureau = bureau.copy()
        bureau["BB_WORST_STATUS"] = np.nan
        bureau["BB_DPD_MONTHS"] = np.nan

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Closed").sum()),
        BUREAU_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_MAX=("AMT_CREDIT_SUM", "max"),
        BUREAU_DEBT_SUM_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_DEBT_SUM_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_DEBT_SUM_MAX=("AMT_CREDIT_SUM_DEBT", "max"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_OVERDUE_MEAN=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_OVERDUE_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_BAL_WORST_STATUS=("BB_WORST_STATUS", "max"),
        BUREAU_BAL_DPD_MONTHS=("BB_DPD_MONTHS", "sum"),
    ).reset_index()
    log.info(f"Aggregated bureau.csv (+ bureau_balance.csv) -> {agg.shape[0]} applicants")
    return agg


def _aggregate_previous_application() -> Optional[pd.DataFrame]:
    from src.data.loader import load_previous_application

    prev = load_previous_application()
    if prev is None:
        return None

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APP_APPROVAL_RATE=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").mean()),
        PREV_APP_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
        PREV_APP_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_APP_AMT_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_APP_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_APP_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_APP_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_APP_AMT_ANNUITY_MAX=("AMT_ANNUITY", "max"),
        PREV_APP_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    ).reset_index()
    log.info(f"Aggregated previous_application.csv -> {agg.shape[0]} applicants")
    return agg


def _aggregate_pos_cash() -> Optional[pd.DataFrame]:
    from src.data.loader import load_pos_cash_balance

    pos = load_pos_cash_balance()
    if pos is None:
        return None

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "nunique"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()
    log.info(f"Aggregated POS_CASH_balance.csv -> {agg.shape[0]} applicants")
    return agg


def _aggregate_credit_card() -> Optional[pd.DataFrame]:
    from src.data.loader import load_credit_card_balance

    cc = load_credit_card_balance()
    if cc is None:
        return None

    cc = cc.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "nunique"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
    ).reset_index()
    log.info(f"Aggregated credit_card_balance.csv -> {agg.shape[0]} applicants")
    return agg


def _aggregate_installments() -> Optional[pd.DataFrame]:
    from src.data.loader import load_installments_payments

    inst = load_installments_payments()
    if inst is None:
        return None

    inst = inst.copy()
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["UNDERPAY_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTALL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INSTALL_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INSTALL_UNDERPAY_RATIO_MEAN=("UNDERPAY_RATIO", "mean"),
        INSTALL_UNDERPAY_RATIO_MIN=("UNDERPAY_RATIO", "min"),
        INSTALL_LATE_COUNT=("IS_LATE", "sum"),
    ).reset_index()
    log.info(f"Aggregated installments_payments.csv -> {agg.shape[0]} applicants")
    return agg


_COUNT_COL_PREFIXES = ("BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT",
                        "BUREAU_BAL_DPD_MONTHS", "PREV_APP_COUNT", "PREV_APP_REFUSED_COUNT",
                        "POS_COUNT", "CC_COUNT", "INSTALL_LATE_COUNT")


def _build_full_aggregate_table() -> pd.DataFrame:
    """Build (once per process) the one-row-per-SK_ID_CURR table of all auxiliary aggregates."""
    global _ratio_medians_cache

    pieces = [
        _aggregate_bureau(),
        _aggregate_previous_application(),
        _aggregate_pos_cash(),
        _aggregate_credit_card(),
        _aggregate_installments(),
    ]
    pieces = [p for p in pieces if p is not None]

    if not pieces:
        log.warning("No auxiliary tables found; aggregate features will be all-default")
        return pd.DataFrame(columns=["SK_ID_CURR"])

    table = pieces[0]
    for p in pieces[1:]:
        table = table.merge(p, on="SK_ID_CURR", how="outer")

    table["HAS_BUREAU_HISTORY"] = table.get("BUREAU_COUNT", pd.Series(dtype=float)).notna()
    table["HAS_PREV_APPLICATION"] = table.get("PREV_APP_COUNT", pd.Series(dtype=float)).notna()

    # Sensible fill: 0 for counts (no history == zero events, distinguished from
    # "unknown" by the HAS_* flags above), median for ratio/amount columns so a
    # missing aggregate doesn't look like an extreme (zero) value to the model.
    ratio_medians = {}
    for col in table.columns:
        if col in ("SK_ID_CURR", "HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION"):
            continue
        if col.startswith(_COUNT_COL_PREFIXES):
            table[col] = table[col].fillna(0)
        else:
            median = table[col].median()
            ratio_medians[col] = median
            table[col] = table[col].fillna(median)

    _ratio_medians_cache = ratio_medians
    log.info(f"Built full auxiliary-table aggregate: {table.shape[0]} applicants, {table.shape[1] - 1} features")
    return table


def build_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join the bureau/previous-application/POS/credit-card/installments aggregates onto df.

    df must have a SK_ID_CURR column (one row per applicant, train or test — or a single
    applicant row at inference time). The aggregate table itself is computed once per
    process and cached, since it depends only on the static auxiliary CSVs.
    """
    global _aggregate_cache

    if _aggregate_cache is None:
        _aggregate_cache = _build_full_aggregate_table()

    if _aggregate_cache.empty or "SK_ID_CURR" not in df.columns:
        return df

    before_rows = df.shape[0]
    merged = df.merge(_aggregate_cache, on="SK_ID_CURR", how="left")
    assert merged.shape[0] == before_rows, (
        f"build_aggregate_features changed row count: {before_rows} -> {merged.shape[0]}"
    )

    # Applicants with no bureau/previous-application row at all (not in the cache
    # merge output -> NaN here) still need the same fill/flag treatment.
    if "HAS_BUREAU_HISTORY" in merged.columns:
        merged["HAS_BUREAU_HISTORY"] = merged["HAS_BUREAU_HISTORY"].fillna(False)
    if "HAS_PREV_APPLICATION" in merged.columns:
        merged["HAS_PREV_APPLICATION"] = merged["HAS_PREV_APPLICATION"].fillna(False)
    for col, median in (_ratio_medians_cache or {}).items():
        if col in merged.columns:
            fill = 0 if col.startswith(_COUNT_COL_PREFIXES) else median
            merged[col] = merged[col].fillna(fill)

    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed ratio features known to matter for this dataset."""
    df = df.copy()

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # Credit-to-income and annuity-to-income ratios are consistently among the
    # strongest predictors in public Home Credit analyses — they capture
    # affordability better than either raw amount alone.
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

    if "BUREAU_CREDIT_SUM_SUM" in df.columns and "BUREAU_DEBT_SUM_SUM" in df.columns:
        df["BUREAU_DEBT_CREDIT_RATIO"] = (
            df["BUREAU_DEBT_SUM_SUM"] / df["BUREAU_CREDIT_SUM_SUM"].replace(0, np.nan)
        )

    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split feature columns (excluding TARGET and id columns) into numeric and categorical."""
    exclude = {"TARGET", "SK_ID_CURR"}
    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and c not in categorical_cols
    ]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Median-impute numeric features, most-frequent-impute + one-hot-encode categoricals."""
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    log.info(f"Built preprocessing pipeline: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical cols")
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str]) -> List[str]:
    """Recover human-readable feature names after ColumnTransformer transformation, for SHAP/rules."""
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out())
    return numeric_cols + cat_names
