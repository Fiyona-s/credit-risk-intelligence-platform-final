"""Cleaning, imputation, encoding, and credit-behavior feature engineering for the
Home Credit dataset.

Exposes a single sklearn-compatible Pipeline (`build_preprocessing_pipeline`) so
train.py, predict.py, and shap_explainer.py all transform data identically.

Conceptual flow (this is the credit-behavior feature layer described in
BASELINE.md's Milestone 1 upgrade):

    raw application data
        -> historical-table aggregation      (build_bureau_features, build_previous_
                                                application_features, build_pos_cash_
                                                features, build_credit_card_features,
                                                build_installment_features)
        -> build_aggregate_features()         (left-join all of the above onto the
                                                applicant row, one row per SK_ID_CURR)
        -> engineer_features()                (second-order/ratio features that need
                                                both applicant-level and aggregate
                                                columns, e.g. credit-to-income)
        -> build_preprocessing_pipeline()      (impute + one-hot-encode -> final matrix)

build_aggregate_features() remains the single shared code path (used by
src.ml.train, src.ml.predict, src.explainability.shap_explainer, and
src.rules.rule_derivation via src.data.loader.build_joined_dataset) that rolls
up the six auxiliary Home Credit history tables to one row per SK_ID_CURR and
left-joins them onto the applicant table — unchanged in this milestone.
engineer_features() is likewise unchanged as the single call site every other
module imports; new ratio features were added inside it rather than via a
second, parallel function, so training/inference consistency (src.ml.train,
src.ml.predict, src.explainability.shap_explainer, src.rules.rule_derivation,
src.fairness.audit, src.monitoring.drift_report, scripts.generate_model_card,
app.py all call these same two functions) required no caller changes.

See src/data/feature_registry.py for the metadata (source table, business
meaning, actionable/leakage classification) behind every engineered feature
this module produces.
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

# Recency windows for the limited set of recency-aware features below (§ Module 5
# of the Milestone 1 spec). Deliberately small and fixed, not a general temporal
# framework — this is a feature-engineering upgrade, not a time-series model.
_RECENT_MONTHS_WINDOW = 6  # for tables keyed by MONTHS_BALANCE (bureau_balance, credit_card_balance)
_RECENT_DAYS_WINDOW = 180  # ~6 months, for installments_payments (keyed by DAYS_INSTALMENT)

_aggregate_cache: Optional[pd.DataFrame] = None
_column_fill_cache: Optional[dict] = None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator/denominator with zero/near-zero and missing denominators
    mapped to NaN rather than +-inf, and any residual +-inf (e.g. a non-zero-but-tiny
    denominator producing an overflow) also mapped to NaN. Used by every ratio feature
    in this module instead of ad hoc `.replace(0, np.nan)` calls, so "invalid ratio" is
    handled the same way everywhere: as missing information for the trained pipeline's
    imputer to fill, never as an infinite or otherwise unbounded value.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build_bureau_features() -> Optional[pd.DataFrame]:
    """Aggregate bureau.csv (+ bureau_balance.csv) to one row per SK_ID_CURR.

    Feature group: credit_history. See src/data/feature_registry.py for the full
    per-feature metadata (this docstring covers the aggregation logic, not the
    business-meaning catalogue).

    Columns used (confirmed present in the real Kaggle bureau.csv): SK_ID_CURR,
    SK_ID_BUREAU, CREDIT_ACTIVE, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT,
    AMT_CREDIT_SUM_OVERDUE, CREDIT_DAY_OVERDUE, DAYS_CREDIT. bureau_balance.csv:
    SK_ID_BUREAU, MONTHS_BALANCE, STATUS.
    """
    from src.data.loader import load_bureau, load_bureau_balance

    bureau = load_bureau()
    if bureau is None:
        return None

    bureau = bureau.copy()
    # Historical bureau utilization at the individual credit-line level (debt
    # outstanding / total credit granted on that line), averaged per applicant
    # below — distinct from BUREAU_DEBT_CREDIT_RATIO (computed in engineer_features
    # from the applicant-level *sums*); this is the per-line mean instead.
    bureau["_UTILIZATION"] = _safe_divide(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
    bureau["_IS_OVERDUE_ACCOUNT"] = (bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype(int)
    # DAYS_CREDIT is "days before this application that the bureau credit was
    # opened" (always <= 0, historical by construction — see module docstring
    # above) — negate for a positive "days of credit history" duration.
    bureau["_CREDIT_DURATION_DAYS"] = -bureau["DAYS_CREDIT"]

    bb = load_bureau_balance()
    if bb is not None:
        bb = bb.copy()
        bb["STATUS_ORD"] = bb["STATUS"].map(_BUREAU_BALANCE_STATUS_ORDINAL)
        bb["IS_DPD_MONTH"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_WORST_STATUS=("STATUS_ORD", "max"),
            BB_DPD_MONTHS=("IS_DPD_MONTH", "sum"),
        ).reset_index()

        # Recency-aware: delinquency in just the last _RECENT_MONTHS_WINDOW months of
        # bureau-balance history, vs. BB_DPD_MONTHS's all-time count above. Vectorized
        # boolean-mask + groupby, no per-row loop.
        bb_recent = bb[bb["MONTHS_BALANCE"] >= -_RECENT_MONTHS_WINDOW]
        bb_recent_agg = bb_recent.groupby("SK_ID_BUREAU").agg(
            BB_DPD_MONTHS_RECENT_6M=("IS_DPD_MONTH", "sum"),
        ).reset_index()
        bb_agg = bb_agg.merge(bb_recent_agg, on="SK_ID_BUREAU", how="left")

        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    else:
        bureau["BB_WORST_STATUS"] = np.nan
        bureau["BB_DPD_MONTHS"] = np.nan
        bureau["BB_DPD_MONTHS_RECENT_6M"] = np.nan

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda s: (s == "Closed").sum()),
        BUREAU_OVERDUE_ACCOUNT_COUNT=("_IS_OVERDUE_ACCOUNT", "sum"),
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
        BUREAU_UTILIZATION_MEAN=("_UTILIZATION", "mean"),
        BUREAU_CREDIT_DURATION_MEAN=("_CREDIT_DURATION_DAYS", "mean"),
        BUREAU_BAL_WORST_STATUS=("BB_WORST_STATUS", "max"),
        BUREAU_BAL_DPD_MONTHS=("BB_DPD_MONTHS", "sum"),
        BUREAU_BAL_DPD_MONTHS_RECENT_6M=("BB_DPD_MONTHS_RECENT_6M", "sum"),
    ).reset_index()

    # Ratios that only make sense at the applicant level (count of active loans
    # relative to total loans, not per-credit-line) — computed after the groupby.
    agg["BUREAU_ACTIVE_RATIO"] = _safe_divide(agg["BUREAU_ACTIVE_COUNT"], agg["BUREAU_COUNT"])
    agg["BUREAU_DELINQUENCY_RATE"] = _safe_divide(agg["BUREAU_OVERDUE_ACCOUNT_COUNT"], agg["BUREAU_COUNT"])

    log.info(f"Aggregated bureau.csv (+ bureau_balance.csv) -> {agg.shape[0]} applicants")
    return agg


def build_previous_application_features() -> Optional[pd.DataFrame]:
    """Aggregate previous_application.csv to one row per SK_ID_CURR.

    Feature group: previous_application_behavior. Columns used (confirmed present):
    SK_ID_CURR, SK_ID_PREV, NAME_CONTRACT_STATUS, AMT_APPLICATION, AMT_CREDIT,
    AMT_ANNUITY, CNT_PAYMENT.

    Previous-credit-relative-to-income ratios (PREV_CREDIT_INCOME_RATIO,
    PREV_APPLICATION_INCOME_RATIO) are NOT computed here — AMT_INCOME_TOTAL only
    exists on the applicant table, not this auxiliary one, so those ratios are
    computed in engineer_features() after the two are joined (same pattern
    already used for BUREAU_DEBT_CREDIT_RATIO).
    """
    from src.data.loader import load_previous_application

    prev = load_previous_application()
    if prev is None:
        return None

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APP_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
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

    agg["PREV_APP_REJECTION_RATE"] = _safe_divide(agg["PREV_APP_REFUSED_COUNT"], agg["PREV_APP_COUNT"])

    log.info(f"Aggregated previous_application.csv -> {agg.shape[0]} applicants")
    return agg


def build_pos_cash_features() -> Optional[pd.DataFrame]:
    """Aggregate POS_CASH_balance.csv to one row per SK_ID_CURR.

    Feature group: pos_cash_behavior. Columns used (confirmed present): SK_ID_CURR,
    SK_ID_PREV, CNT_INSTALMENT_FUTURE, NAME_CONTRACT_STATUS, SK_DPD, SK_DPD_DEF.

    "Delinquency duration" (a continuous days-past-due count) is not present in
    this table — SK_DPD is a per-month-snapshot DPD value, not a running duration
    — so POS_SK_DPD_MEAN/MAX (already existing) are the closest reliable proxy;
    a true delinquency-duration feature would require reconstructing per-loan
    time series from MONTHS_BALANCE, out of scope for this feature-layer
    milestone (see BASELINE.md's "recency, not time-series modeling" guidance).
    """
    from src.data.loader import load_pos_cash_balance

    pos = load_pos_cash_balance()
    if pos is None:
        return None

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "nunique"),
        POS_ACTIVE_COUNT=("NAME_CONTRACT_STATUS", lambda s: s.isin(["Active"]).sum()),
        POS_COMPLETED_COUNT=("NAME_CONTRACT_STATUS", lambda s: s.isin(["Completed"]).sum()),
        POS_OVERDUE_COUNT=("SK_DPD", lambda s: (s > 0).sum()),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()
    log.info(f"Aggregated POS_CASH_balance.csv -> {agg.shape[0]} applicants")
    return agg


def build_credit_card_features() -> Optional[pd.DataFrame]:
    """Aggregate credit_card_balance.csv to one row per SK_ID_CURR.

    Feature group: credit_card_behavior. Columns used (confirmed present):
    SK_ID_CURR, SK_ID_PREV, MONTHS_BALANCE, AMT_BALANCE, AMT_CREDIT_LIMIT_ACTUAL,
    AMT_PAYMENT_TOTAL_CURRENT, SK_DPD.

    CC_UTILIZATION_* (balance / credit limit) doubles as the "balance-to-limit
    ratio" requested by the spec — one feature, not two identical ones under
    different names.
    """
    from src.data.loader import load_credit_card_balance

    cc = load_credit_card_balance()
    if cc is None:
        return None

    cc = cc.copy()
    cc["UTILIZATION"] = _safe_divide(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["_PAYMENT_BALANCE_RATIO"] = _safe_divide(cc["AMT_PAYMENT_TOTAL_CURRENT"], cc["AMT_BALANCE"])
    cc["_IS_OVERDUE"] = (cc["SK_DPD"].fillna(0) > 0).astype(int)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "nunique"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_LIMIT_MAX=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_PAYMENT_BALANCE_RATIO_MEAN=("_PAYMENT_BALANCE_RATIO", "mean"),
        CC_OVERDUE_COUNT=("_IS_OVERDUE", "sum"),
        CC_SK_DPD_MEAN=("SK_DPD", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
        CC_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        CC_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()

    # Recency-aware: mean utilization in just the last _RECENT_MONTHS_WINDOW months,
    # vs. CC_UTILIZATION_MEAN's all-time average above.
    cc_recent = cc[cc["MONTHS_BALANCE"] >= -_RECENT_MONTHS_WINDOW]
    cc_recent_agg = cc_recent.groupby("SK_ID_CURR").agg(
        CC_UTILIZATION_MEAN_RECENT_6M=("UTILIZATION", "mean"),
    ).reset_index()
    agg = agg.merge(cc_recent_agg, on="SK_ID_CURR", how="left")

    log.info(f"Aggregated credit_card_balance.csv -> {agg.shape[0]} applicants")
    return agg


def build_installment_features() -> Optional[pd.DataFrame]:
    """Aggregate installments_payments.csv to one row per SK_ID_CURR.

    Feature group: installment_repayment_behavior. Columns used (confirmed
    present): SK_ID_CURR, SK_ID_PREV, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT,
    AMT_INSTALMENT, AMT_PAYMENT.

    INSTALL_COMPLETION_RATIO is a proxy (share of scheduled installments with a
    recorded payment amount) rather than a true "loan completed" flag — this
    table has no explicit completion/status column, only per-installment
    payment records, so completion is inferred from whether AMT_PAYMENT is
    present, not asserted directly.
    """
    from src.data.loader import load_installments_payments

    inst = load_installments_payments()
    if inst is None:
        return None

    inst = inst.copy()
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["UNDERPAY_RATIO"] = _safe_divide(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"])
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)
    inst["_PAYMENT_SHORTFALL"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["_IS_PAID"] = inst["AMT_PAYMENT"].notna().astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        INSTALL_COUNT=("SK_ID_PREV", "count"),
        INSTALL_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INSTALL_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INSTALL_UNDERPAY_RATIO_MEAN=("UNDERPAY_RATIO", "mean"),
        INSTALL_UNDERPAY_RATIO_MIN=("UNDERPAY_RATIO", "min"),
        INSTALL_LATE_COUNT=("IS_LATE", "sum"),
        INSTALL_COMPLETION_RATIO=("_IS_PAID", "mean"),
        INSTALL_PAYMENT_SHORTFALL_MEAN=("_PAYMENT_SHORTFALL", "mean"),
    ).reset_index()

    agg["INSTALL_LATE_RATIO"] = _safe_divide(agg["INSTALL_LATE_COUNT"], agg["INSTALL_COUNT"])

    # Recency-aware: late-payment count in just the last _RECENT_DAYS_WINDOW days
    # (DAYS_INSTALMENT, not MONTHS_BALANCE, is this table's only time axis).
    inst_recent = inst[inst["DAYS_INSTALMENT"] >= -_RECENT_DAYS_WINDOW]
    inst_recent_agg = inst_recent.groupby("SK_ID_CURR").agg(
        INSTALL_LATE_COUNT_RECENT_6M=("IS_LATE", "sum"),
    ).reset_index()
    agg = agg.merge(inst_recent_agg, on="SK_ID_CURR", how="left")

    log.info(f"Aggregated installments_payments.csv -> {agg.shape[0]} applicants")
    return agg


# NOTE on features considered but NOT implemented (documented per the Milestone 1
# spec's requirement to explain schema-driven omissions, not silently skip them):
#   - True bureau "credit history duration" as (first credit ever opened -> now):
#     implemented as BUREAU_CREDIT_DURATION_MEAN (mean age of bureau credit lines)
#     rather than a min/max span, since DAYS_CREDIT_ENDDATE is frequently NaN for
#     active/ongoing credits, making a reliable start-to-end span unavailable for
#     a large share of rows.
#   - POS_CASH "delinquency duration" (continuous days-past-due streak length):
#     would require reconstructing a per-loan MONTHS_BALANCE time series and is
#     out of scope for a feature-engineering (not time-series) milestone — see
#     build_pos_cash_features()'s docstring.
#   - Credit-card "drawings"-based features (AMT_DRAWINGS_ATM/POS_CURRENT etc.):
#     present in the schema but overlap heavily with AMT_BALANCE/UTILIZATION in
#     what they capture about spending behavior; skipped to avoid adding
#     features without a distinct credit-risk interpretation, per the spec's
#     "do not blindly generate features" instruction.

_COUNT_COL_PREFIXES = (
    "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT", "BUREAU_OVERDUE_ACCOUNT_COUNT",
    "BUREAU_BAL_DPD_MONTHS", "PREV_APP_COUNT", "PREV_APP_APPROVED_COUNT", "PREV_APP_REFUSED_COUNT",
    "POS_COUNT", "POS_ACTIVE_COUNT", "POS_COMPLETED_COUNT", "POS_OVERDUE_COUNT",
    "CC_COUNT", "CC_OVERDUE_COUNT", "INSTALL_COUNT", "INSTALL_LATE_COUNT",
)


def _build_full_aggregate_table() -> pd.DataFrame:
    """Build (once per process) the one-row-per-SK_ID_CURR table of all auxiliary aggregates."""
    global _column_fill_cache

    pieces = [
        build_bureau_features(),
        build_previous_application_features(),
        build_pos_cash_features(),
        build_credit_card_features(),
        build_installment_features(),
    ]
    pieces = [p for p in pieces if p is not None]

    if not pieces:
        log.warning("No auxiliary tables found; aggregate features will be all-default")
        return pd.DataFrame(columns=["SK_ID_CURR"])

    table = pieces[0]
    for p in pieces[1:]:
        table = table.merge(p, on="SK_ID_CURR", how="outer")

    # Explicit "no history at all" indicators, kept separate from "history exists
    # but the value happens to be zero" (e.g. a bureau record with zero overdue
    # accounts vs. no bureau record at all) — set from row presence in the raw
    # aggregate merge (.notna() before the fill step below touches anything),
    # not from any filled value.
    table["HAS_BUREAU_HISTORY"] = table.get("BUREAU_COUNT", pd.Series(dtype=float)).notna()
    table["HAS_PREV_APPLICATION"] = table.get("PREV_APP_COUNT", pd.Series(dtype=float)).notna()
    table["HAS_POS_HISTORY"] = table.get("POS_COUNT", pd.Series(dtype=float)).notna()
    table["HAS_CC_HISTORY"] = table.get("CC_COUNT", pd.Series(dtype=float)).notna()
    table["HAS_INSTALLMENT_HISTORY"] = table.get("INSTALL_COUNT", pd.Series(dtype=float)).notna()

    # Sensible fill: 0 for counts (no history == zero events, distinguished from
    # "unknown" by the HAS_* flags above), median for ratio/amount columns so a
    # missing aggregate doesn't look like an extreme (zero) value to the model.
    #
    # column_fill_values records the fill used for EVERY aggregate column (not just
    # ratio/amount ones) — this matters for an applicant who appears in NONE of the
    # five auxiliary tables (a real, if rare, case in a large dataset, and always
    # true for a brand-new applicant scored via the API with no bureau/previous-app
    # history at all): such an applicant is entirely absent from `table`, so their
    # row is produced later by build_aggregate_features()'s left-join, which leaves
    # every aggregate column NaN, count columns included. An earlier version of this
    # function only recorded medians for non-count columns here, so a wholly-absent
    # applicant's count columns (BUREAU_COUNT, PREV_APP_COUNT, etc.) were silently
    # left as NaN instead of 0 — caught by tests/test_feature_engineering.py's
    # "applicant with no history anywhere" case, not by inspection.
    column_fill_values = {}
    _has_flags = ("HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION", "HAS_POS_HISTORY",
                  "HAS_CC_HISTORY", "HAS_INSTALLMENT_HISTORY")
    for col in table.columns:
        if col == "SK_ID_CURR" or col in _has_flags:
            continue
        if col.startswith(_COUNT_COL_PREFIXES):
            fill = 0
        else:
            fill = table[col].median()
        column_fill_values[col] = fill
        table[col] = table[col].fillna(fill)

    _column_fill_cache = column_fill_values
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

    # Applicants with no bureau/previous-application/POS/credit-card/installment row
    # at all (not in the cache merge output -> NaN here) still need the same
    # fill/flag treatment.
    for flag_col in ("HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION", "HAS_POS_HISTORY",
                     "HAS_CC_HISTORY", "HAS_INSTALLMENT_HISTORY"):
        if flag_col in merged.columns:
            merged[flag_col] = merged[flag_col].fillna(False)
    for col, fill_value in (_column_fill_cache or {}).items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(fill_value)

    return merged


def get_aggregate_table_for_validation() -> Tuple[pd.DataFrame, tuple]:
    """Expose the cached aggregate table + count-column-prefix tuple to
    src.data.validation.check_aggregate_fill_contamination(), which needs both to
    measure how much the fill values used by build_aggregate_features() would
    change if computed on a train-only subset instead of the full population.
    Not for general use — build_aggregate_features() remains the only supported
    way to actually apply these aggregates to an applicant DataFrame.
    """
    global _aggregate_cache
    if _aggregate_cache is None:
        _aggregate_cache = _build_full_aggregate_table()
    return _aggregate_cache, _COUNT_COL_PREFIXES


def reset_aggregate_cache() -> None:
    """Clear the module-level aggregate cache/medians. Only needed when the underlying
    auxiliary CSVs change within the same process (e.g. between test cases using
    different synthetic fixtures) — normal training/inference runs are one-process,
    one-dataset, and never need this.
    """
    global _aggregate_cache, _column_fill_cache
    _aggregate_cache = None
    _column_fill_cache = None


_ENGINEER_FEATURES_INPUT_COLS = ["DAYS_EMPLOYED", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_BIRTH"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed ratio features known to matter for this dataset.

    Second-order/business ratios that need columns from *both* the applicant
    table and the joined aggregate table (e.g. previous-credit-relative-to-
    income) live here rather than in the individual build_*_features()
    functions above, since AMT_INCOME_TOTAL only exists on the applicant row.

    Tolerates a partial applicant record missing one or more of the base columns
    below (e.g. api.py's /predict accepting an arbitrary JSON body, or a manual
    Risk Prediction form field left blank) — any missing base column is filled
    with NaN first, so the ratios below come out NaN (correctly imputed later by
    the trained pipeline) instead of raising a KeyError. build_joined_dataset()'s
    normal path always has every one of these columns already, so this is a no-op
    there.
    """
    df = df.copy()
    for col in _ENGINEER_FEATURES_INPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)

    # Credit-to-income and annuity-to-income ratios are consistently among the
    # strongest predictors in public Home Credit analyses — they capture
    # affordability better than either raw amount alone.
    df["CREDIT_INCOME_RATIO"] = _safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["DAYS_EMPLOYED_RATIO"] = _safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    if "BUREAU_CREDIT_SUM_SUM" in df.columns and "BUREAU_DEBT_SUM_SUM" in df.columns:
        df["BUREAU_DEBT_CREDIT_RATIO"] = _safe_divide(df["BUREAU_DEBT_SUM_SUM"], df["BUREAU_CREDIT_SUM_SUM"])
    if "BUREAU_DEBT_SUM_SUM" in df.columns:
        df["BUREAU_DEBT_INCOME_RATIO"] = _safe_divide(df["BUREAU_DEBT_SUM_SUM"], df["AMT_INCOME_TOTAL"])
    if "PREV_APP_AMT_CREDIT_MEAN" in df.columns:
        df["PREV_CREDIT_INCOME_RATIO"] = _safe_divide(df["PREV_APP_AMT_CREDIT_MEAN"], df["AMT_INCOME_TOTAL"])
    if "PREV_APP_AMT_APPLICATION_MEAN" in df.columns:
        df["PREV_APPLICATION_INCOME_RATIO"] = _safe_divide(
            df["PREV_APP_AMT_APPLICATION_MEAN"], df["AMT_INCOME_TOTAL"]
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


def get_output_feature_names(
    preprocessor: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str] = None
) -> List[str]:
    """Recover human-readable feature names after ColumnTransformer transformation, for SHAP/rules.

    Milestone 7 fix: get_feature_names_out() was previously called with no
    input_features, so sklearn fell back to generic "x0_...", "x1_..." prefixes
    (e.g. "x0_Cash loans" instead of "NAME_CONTRACT_TYPE_Cash loans") — a real
    readability bug, not a cosmetic one, since it directly blocked mapping a
    one-hot column back to its source feature/category for applicant
    explanations (src.explainability.shap_explainer). categorical_cols is
    optional and defaults to the encoder's own recorded feature_names_in_ (the
    categorical columns it was actually fit on) when not supplied, so existing
    call sites that only pass numeric_cols keep working unchanged.
    """
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    if categorical_cols is None:
        categorical_cols = list(getattr(cat_encoder, "feature_names_in_", []))
    cat_names = list(cat_encoder.get_feature_names_out(categorical_cols)) if categorical_cols else list(
        cat_encoder.get_feature_names_out()
    )
    return numeric_cols + cat_names
