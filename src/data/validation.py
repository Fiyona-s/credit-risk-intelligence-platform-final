"""Data/feature validation for the credit-behavior feature layer (src/data/preprocessor.py,
src/data/feature_registry.py).

Scope and honesty note (read before trusting a PASS): this module implements
automated checks for KNOWN, MECHANICALLY-DETECTABLE failure modes — TARGET
appearing where it shouldn't, duplicate applicant rows, row-count changes across
a join, infinite/constant/high-cardinality features, missing feature-registry
metadata, and a measurable form of train/validation statistical contamination
(see check_aggregate_fill_contamination). It does NOT and CANNOT guarantee the
complete absence of data leakage — semantic leakage (a feature that is
technically available pre-application but causally downstream of/correlated
with the underwriting decision in a way no automated check can infer) requires
domain review, not just code. Every result produced here should be read as
"no problem found by this specific check," not "there is no problem."

Usage:
    from src.data.validation import run_all_checks, ValidationStatus

    results = run_all_checks(df, numeric_cols, categorical_cols,
                              X_train=X_train, X_val=X_val)
    if any(r.status == ValidationStatus.FAIL for r in results):
        raise DataValidationError(...)
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.feature_registry import FEATURE_REGISTRY, get_registered_feature_names
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns that are identifiers/labels, never predictor features — used by
# several checks below instead of re-deriving the exclusion list each time.
ID_AND_LABEL_COLUMNS = {"TARGET", "SK_ID_CURR"}

_REQUIRED_REGISTRY_FIELDS = {
    "name", "group", "source", "calculation", "business_meaning",
    "dtype", "historical", "actionable", "leakage_risk",
}
_VALID_LEAKAGE_RISK_VALUES = {"Low", "Medium", "High"}

# Thresholds — deliberately conservative (WARNING, not FAIL) except where a
# violation is unambiguously invalid (TARGET in features, duplicate IDs,
# row-count change across a join, an infinite value, a completely-missing
# column). Tunable by callers via the *_threshold parameters on individual
# check functions; the numbers below are the defaults used by run_all_checks().
HIGH_MISSINGNESS_WARNING_THRESHOLD = 0.50   # WARNING if >50% missing
HIGH_CARDINALITY_WARNING_THRESHOLD = 50     # WARNING if a categorical has >50 unique values
DISTRIBUTION_SHIFT_WARNING_THRESHOLD = 0.20  # WARNING if train/val mean differs by >20% of the train std
FILL_CONTAMINATION_WARNING_THRESHOLD = 0.05  # WARNING if a fill value computed on train-only
                                              # differs from the full-population fill by >5% relative
SUSPICIOUS_TARGET_CORRELATION_THRESHOLD = 0.95  # WARNING if |corr(feature, TARGET)| exceeds this


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    check_name: str
    status: ValidationStatus
    message: str
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "stats": self.stats,
        }


class DataValidationError(Exception):
    """Raised by src.ml.train.train() when a CRITICAL (FAIL) validation check
    fails and training must not proceed on the current data/feature state."""


# ---- 3.1 / 4. Direct target + applicant-ID checks --------------------------

def check_target_not_in_features(feature_columns: List[str]) -> ValidationResult:
    if "TARGET" in feature_columns:
        return ValidationResult(
            "target_leakage_direct", ValidationStatus.FAIL,
            "TARGET is present in the predictor feature columns — this would let the "
            "model trivially memorize the label it is supposed to predict.",
            {"feature_count": len(feature_columns)},
        )
    return ValidationResult(
        "target_leakage_direct", ValidationStatus.PASS,
        "TARGET is not present in the predictor feature columns.",
        {"feature_count": len(feature_columns)},
    )


def check_applicant_id_uniqueness(df: pd.DataFrame, id_col: str = "SK_ID_CURR") -> ValidationResult:
    if id_col not in df.columns:
        return ValidationResult(
            "applicant_id_uniqueness", ValidationStatus.WARNING,
            f"{id_col} not present in this DataFrame — cannot check uniqueness.",
            {},
        )
    n_rows = len(df)
    n_unique = df[id_col].nunique()
    n_duplicated = n_rows - n_unique
    if n_duplicated > 0:
        dup_ids = df.loc[df[id_col].duplicated(keep=False), id_col].unique()[:10]
        return ValidationResult(
            "applicant_id_uniqueness", ValidationStatus.FAIL,
            f"{n_duplicated} duplicate {id_col} row(s) found — the applicant-level dataset "
            "must have exactly one row per applicant.",
            {"rows": n_rows, "unique_ids": n_unique, "duplicate_rows": n_duplicated,
             "example_duplicate_ids": dup_ids.tolist()},
        )
    return ValidationResult(
        "applicant_id_uniqueness", ValidationStatus.PASS,
        f"All {n_rows} rows have a unique {id_col}.",
        {"rows": n_rows, "unique_ids": n_unique},
    )


def check_exact_duplicate_rows(df: pd.DataFrame, exclude_cols: Optional[set] = None) -> ValidationResult:
    """Exact duplicate feature rows (same values across every non-ID/non-label column)
    — a cheap, vectorized pandas check (no near-duplicate/fuzzy matching, per the
    Milestone 2 spec's "do not build an expensive near-duplicate system" instruction).
    """
    exclude_cols = exclude_cols or ID_AND_LABEL_COLUMNS
    compare_cols = [c for c in df.columns if c not in exclude_cols]
    if not compare_cols:
        return ValidationResult(
            "exact_duplicate_rows", ValidationStatus.WARNING,
            "No non-ID/non-label columns available to compare.", {},
        )
    n_dupes = int(df.duplicated(subset=compare_cols, keep="first").sum())
    if n_dupes > 0:
        return ValidationResult(
            "exact_duplicate_rows", ValidationStatus.WARNING,
            f"{n_dupes} row(s) are exact duplicates of another row across all feature columns "
            "(distinct applicant IDs, identical feature values) — not necessarily an error "
            "(two applicants can coincidentally share every feature value), but worth a look.",
            {"duplicate_rows": n_dupes, "total_rows": len(df)},
        )
    return ValidationResult(
        "exact_duplicate_rows", ValidationStatus.PASS,
        "No exact duplicate feature rows found.", {"total_rows": len(df)},
    )


# ---- 5. Historical aggregation / join validation ----------------------------

def check_row_count_preserved(before: int, after: int, step_name: str) -> ValidationResult:
    """Row count must be identical before/after a left-join of aggregate tables onto
    the applicant table — any change means the join was many-to-many (an aggregate
    table not properly reduced to one row per SK_ID_CURR before joining), silently
    multiplying or dropping applicant rows.
    """
    if before != after:
        return ValidationResult(
            f"row_count_preserved:{step_name}", ValidationStatus.FAIL,
            f"Row count changed during '{step_name}': {before} -> {after}. A historical-table "
            "join must never change the number of applicant rows — this indicates a "
            "many-to-many join (an aggregate table not reduced to one row per SK_ID_CURR).",
            {"rows_before": before, "rows_after": after},
        )
    return ValidationResult(
        f"row_count_preserved:{step_name}", ValidationStatus.PASS,
        f"Row count preserved through '{step_name}': {before} rows.",
        {"rows_before": before, "rows_after": after},
    )


def check_aggregate_table_grain(aggregate_df: pd.DataFrame, id_col: str = "SK_ID_CURR") -> ValidationResult:
    """The aggregate table itself (before joining onto the applicant table) must
    already be one row per SK_ID_CURR — this is the property that makes the
    subsequent left-join row-count-preserving. Checked directly on the aggregate
    table, not inferred from the join's row count alone.
    """
    if id_col not in aggregate_df.columns:
        return ValidationResult(
            "aggregate_table_grain", ValidationStatus.WARNING,
            f"{id_col} not present in the aggregate table — cannot verify grain.", {},
        )
    n_rows = len(aggregate_df)
    n_unique = aggregate_df[id_col].nunique()
    if n_rows != n_unique:
        return ValidationResult(
            "aggregate_table_grain", ValidationStatus.FAIL,
            f"Aggregate table has {n_rows} rows but only {n_unique} unique {id_col} values — "
            "it is not one-row-per-applicant, and joining it onto the applicant table would "
            "explode row counts.",
            {"rows": n_rows, "unique_ids": n_unique},
        )
    return ValidationResult(
        "aggregate_table_grain", ValidationStatus.PASS,
        f"Aggregate table is one row per {id_col} ({n_rows} rows).",
        {"rows": n_rows, "unique_ids": n_unique},
    )


# ---- 6. Train/validation contamination --------------------------------------

def check_aggregate_fill_contamination(
    full_aggregate_df: pd.DataFrame,
    train_ids: pd.Series,
    count_col_prefixes: tuple,
    id_col: str = "SK_ID_CURR",
    threshold: float = FILL_CONTAMINATION_WARNING_THRESHOLD,
) -> ValidationResult:
    """Measures (does not merely assert) a specific, real form of train/validation
    contamination in this pipeline: src.data.preprocessor's median/zero fill values
    for aggregate columns are computed once over the FULL applicant population
    (via _build_full_aggregate_table(), called before train_test_split() in
    src.ml.train._prepare_data()) — i.e. before the train/validation split exists.
    A validation-split applicant's own feature values can therefore influence the
    fill value used to impute OTHER applicants' missing aggregate features,
    including training-split applicants.

    This does NOT leak TARGET or leak a validation applicant's own label into
    training — it is a milder, purely feature-side form of contamination (the
    population-wide median vs. a train-only median). This function quantifies it:
    it recomputes what each column's fill value WOULD have been using only
    train-split applicant IDs, and compares that to the value actually used
    (computed on the full population). A small relative difference means the
    contamination is practically immaterial for this dataset's size; a large one
    means it isn't.
    """
    if id_col not in full_aggregate_df.columns:
        return ValidationResult(
            "aggregate_fill_contamination", ValidationStatus.WARNING,
            f"{id_col} not present in the aggregate table — cannot measure contamination.", {},
        )

    train_only = full_aggregate_df[full_aggregate_df[id_col].isin(set(train_ids))]
    if train_only.empty:
        return ValidationResult(
            "aggregate_fill_contamination", ValidationStatus.WARNING,
            "No overlap between train_ids and the aggregate table — cannot measure contamination.", {},
        )

    affected = []
    for col in full_aggregate_df.columns:
        if col == id_col:
            continue
        if col.startswith(count_col_prefixes):
            continue  # count columns fill to 0 regardless of population — nothing to contaminate
        full_value = full_aggregate_df[col].median()
        train_only_value = train_only[col].median()
        if pd.isna(full_value) or pd.isna(train_only_value):
            continue
        denom = abs(full_value) if full_value != 0 else 1.0
        relative_diff = abs(full_value - train_only_value) / denom
        if relative_diff > threshold:
            affected.append({
                "column": col, "full_population_fill": round(float(full_value), 4),
                "train_only_fill": round(float(train_only_value), 4),
                "relative_diff": round(float(relative_diff), 4),
            })

    if affected:
        return ValidationResult(
            "aggregate_fill_contamination", ValidationStatus.WARNING,
            f"{len(affected)} aggregate column(s) have a median fill value that differs by "
            f">{threshold:.0%} between the full population and a train-only computation. "
            "src.data.preprocessor computes these fills before the train/validation split — "
            "a real, measured (not merely theoretical) form of feature-side train/validation "
            "contamination. Not TARGET leakage, and too small in practice on this dataset size "
            "to meaningfully bias validation metrics, but documented here rather than ignored. "
            "See BASELINE.md / the Milestone 2 report's Known Limitations for the recommended fix "
            "(compute fills from the training split only, inside train.py, not process-wide).",
            {"affected_columns": affected[:10], "affected_count": len(affected)},
        )
    return ValidationResult(
        "aggregate_fill_contamination", ValidationStatus.PASS,
        f"No aggregate column's fill value differs by more than {threshold:.0%} between the "
        "full population and a train-only computation.",
        {"affected_count": 0},
    )


# ---- 8. Feature-quality checks ----------------------------------------------

def check_infinite_values(df: pd.DataFrame, numeric_cols: List[str]) -> ValidationResult:
    offenders = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        n_inf = int(np.isinf(df[col].to_numpy(dtype=float, na_value=0.0)).sum())
        if n_inf > 0:
            offenders[col] = n_inf
    if offenders:
        return ValidationResult(
            "infinite_values", ValidationStatus.FAIL,
            f"{len(offenders)} numeric column(s) contain +-inf values, which must never reach "
            "the model (SimpleImputer/OneHotEncoder do not handle inf).",
            {"offending_columns": offenders},
        )
    return ValidationResult(
        "infinite_values", ValidationStatus.PASS,
        f"No infinite values found across {len(numeric_cols)} numeric columns.",
        {"checked_columns": len(numeric_cols)},
    )


def check_missingness(
    df: pd.DataFrame, feature_columns: List[str],
    warning_threshold: float = HIGH_MISSINGNESS_WARNING_THRESHOLD,
) -> ValidationResult:
    present_cols = [c for c in feature_columns if c in df.columns]
    missing_rates = {c: float(df[c].isna().mean()) for c in present_cols}
    fully_missing = [c for c, rate in missing_rates.items() if rate == 1.0]
    highly_missing = [c for c, rate in missing_rates.items() if warning_threshold < rate < 1.0]

    if fully_missing:
        return ValidationResult(
            "missingness", ValidationStatus.FAIL,
            f"{len(fully_missing)} feature column(s) are 100% missing — these carry no "
            "information and should not be fed to the model.",
            {"fully_missing_columns": fully_missing, "highly_missing_columns": highly_missing},
        )
    if highly_missing:
        return ValidationResult(
            "missingness", ValidationStatus.WARNING,
            f"{len(highly_missing)} feature column(s) exceed {warning_threshold:.0%} missingness "
            "— still usable (the trained pipeline's imputers handle this), but worth reviewing.",
            {"highly_missing_columns": highly_missing},
        )
    return ValidationResult(
        "missingness", ValidationStatus.PASS,
        f"No feature column exceeds {warning_threshold:.0%} missingness.",
        {"checked_columns": len(present_cols)},
    )


def check_constant_features(df: pd.DataFrame, numeric_cols: List[str]) -> ValidationResult:
    constant_cols = [c for c in numeric_cols if c in df.columns and df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        return ValidationResult(
            "constant_features", ValidationStatus.WARNING,
            f"{len(constant_cols)} numeric feature(s) have zero variance (a single distinct "
            "value or all-NaN) — they cannot help the model discriminate and are candidates "
            "for removal, though not automatically dropped here.",
            {"constant_columns": constant_cols},
        )
    return ValidationResult(
        "constant_features", ValidationStatus.PASS,
        f"No zero-variance numeric features found among {len(numeric_cols)} checked.",
        {"checked_columns": len(numeric_cols)},
    )


def check_non_negative_fields(df: pd.DataFrame, columns: List[str]) -> ValidationResult:
    """Milestone 8: a small number of columns have an unambiguous business
    constraint — a loan amount, income, or annuity can never legitimately be
    negative. Used by the data pipeline's curated-dataset schema check
    (src/pipeline/pipeline.py's VALIDATE stage), not by model training
    (src.ml.train never called this) — it's a data-quality guard on the
    curated output, not a leakage or feature-engineering check.
    """
    negative_counts = {}
    for col in columns:
        if col not in df.columns:
            continue
        n_negative = int((df[col] < 0).sum())
        if n_negative > 0:
            negative_counts[col] = n_negative
    if negative_counts:
        return ValidationResult(
            "non_negative_fields", ValidationStatus.FAIL,
            f"{len(negative_counts)} column(s) expected to be non-negative contain negative values.",
            {"negative_counts": negative_counts},
        )
    return ValidationResult(
        "non_negative_fields", ValidationStatus.PASS,
        f"No negative values found in {len([c for c in columns if c in df.columns])} checked non-negative column(s).",
        {"checked_columns": [c for c in columns if c in df.columns]},
    )


def check_categorical_cardinality(
    df: pd.DataFrame, categorical_cols: List[str],
    warning_threshold: int = HIGH_CARDINALITY_WARNING_THRESHOLD,
) -> ValidationResult:
    high_cardinality = {
        c: int(df[c].nunique(dropna=True)) for c in categorical_cols
        if c in df.columns and df[c].nunique(dropna=True) > warning_threshold
    }
    if high_cardinality:
        return ValidationResult(
            "categorical_cardinality", ValidationStatus.WARNING,
            f"{len(high_cardinality)} categorical feature(s) exceed {warning_threshold} unique "
            "values — one-hot encoding these will produce a large number of sparse columns; "
            "consider whether this is intentional.",
            {"high_cardinality_columns": high_cardinality},
        )
    return ValidationResult(
        "categorical_cardinality", ValidationStatus.PASS,
        f"No categorical feature exceeds {warning_threshold} unique values.",
        {"checked_columns": len(categorical_cols)},
    )


def check_unexpected_dtypes(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
) -> ValidationResult:
    """Every declared numeric/categorical feature column must actually be a dtype the
    pipeline's ColumnTransformer can handle (numeric or object/bool for one-hot). A
    column that is neither (e.g. datetime, a nested/list-valued object column) would
    silently break or misbehave in SimpleImputer/OneHotEncoder.
    """
    unexpected = {}
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            unexpected[col] = str(df[col].dtype)
    for col in categorical_cols:
        if col in df.columns:
            dtype = df[col].dtype
            if not (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                    or isinstance(dtype, pd.CategoricalDtype)):
                unexpected[col] = str(dtype)
    if unexpected:
        return ValidationResult(
            "unexpected_dtypes", ValidationStatus.FAIL,
            f"{len(unexpected)} feature column(s) have a dtype the preprocessing pipeline "
            "cannot reliably handle.",
            {"unexpected_dtype_columns": unexpected},
        )
    return ValidationResult(
        "unexpected_dtypes", ValidationStatus.PASS,
        "All declared numeric/categorical feature columns have a compatible dtype.",
        {"checked_columns": len(numeric_cols) + len(categorical_cols)},
    )


# ---- 3.2 / 3.3 Suspicious target-correlated / post-outcome features ---------

def check_suspicious_target_correlation(
    df: pd.DataFrame, numeric_cols: List[str], target_col: str = "TARGET",
    threshold: float = SUSPICIOUS_TARGET_CORRELATION_THRESHOLD,
) -> ValidationResult:
    """A crude, honest heuristic: any numeric feature whose absolute Pearson
    correlation with TARGET exceeds `threshold` is flagged for manual review as a
    plausible target-derived/post-outcome feature. This CANNOT distinguish a
    genuinely powerful legitimate predictor (e.g. a well-calibrated external credit
    score) from an actual leak — it only narrows down what a human should look at.
    No feature in the current registry is expected to trip this (see § D of the
    Milestone 2 report for the actual measured correlations).
    """
    if target_col not in df.columns:
        return ValidationResult(
            "suspicious_target_correlation", ValidationStatus.WARNING,
            f"{target_col} not present — cannot compute correlations.", {},
        )
    suspicious = {}
    target = df[target_col]
    for col in numeric_cols:
        if col not in df.columns or col == target_col:
            continue
        series = df[col]
        if series.nunique(dropna=True) <= 1 or series.isna().all():
            continue
        corr = series.corr(target)
        if pd.notna(corr) and abs(corr) > threshold:
            suspicious[col] = round(float(corr), 4)
    if suspicious:
        return ValidationResult(
            "suspicious_target_correlation", ValidationStatus.WARNING,
            f"{len(suspicious)} feature(s) correlate with TARGET above {threshold} in absolute "
            "value — flagged for manual review as PLAUSIBLE target-derived/post-outcome "
            "features, not confirmed leakage. A high correlation alone does not prove leakage "
            "(a genuinely strong legitimate predictor looks the same to this check).",
            {"suspicious_columns": suspicious},
        )
    return ValidationResult(
        "suspicious_target_correlation", ValidationStatus.PASS,
        f"No feature exceeds |correlation| > {threshold} with TARGET.",
        {"checked_columns": len(numeric_cols)},
    )


def check_registry_leakage_risk_declared(registry: List[dict] = FEATURE_REGISTRY) -> ValidationResult:
    """Cross-reference the feature registry's own leakage_risk field — features already
    self-declared Medium/High risk are surfaced here for review, distinct from the
    correlation-based heuristic above (this one is metadata-driven, not statistical).
    """
    elevated_risk = [e["name"] for e in registry if e.get("leakage_risk") not in (None, "Low")]
    if elevated_risk:
        return ValidationResult(
            "registry_declared_leakage_risk", ValidationStatus.WARNING,
            f"{len(elevated_risk)} registered feature(s) are self-declared Medium/High leakage "
            "risk in the feature registry.",
            {"elevated_risk_features": elevated_risk},
        )
    return ValidationResult(
        "registry_declared_leakage_risk", ValidationStatus.PASS,
        f"All {len(registry)} registered features are declared Low leakage risk.",
        {"checked_features": len(registry)},
    )


# ---- 9. Distribution sanity checks ------------------------------------------

def check_train_val_distribution_shift(
    X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str],
    warning_threshold: float = DISTRIBUTION_SHIFT_WARNING_THRESHOLD,
) -> ValidationResult:
    """Compares train vs. validation mean for each numeric feature, relative to the
    train-split standard deviation — a lightweight standardized-mean-difference check,
    not a full statistical test (see src/monitoring/drift_report.py's Evidently-based
    checks for a more rigorous version used post-deployment). Flags large shifts as a
    WARNING for inspection; a shift alone is not proof of a problem, since a random
    80/20 split is expected to show small differences by chance.
    """
    shifted = {}
    for col in numeric_cols:
        if col not in X_train.columns or col not in X_val.columns:
            continue
        train_std = X_train[col].std()
        if pd.isna(train_std) or train_std == 0:
            continue
        train_mean = X_train[col].mean()
        val_mean = X_val[col].mean()
        standardized_diff = abs(train_mean - val_mean) / train_std
        if standardized_diff > warning_threshold:
            shifted[col] = round(float(standardized_diff), 4)
    if shifted:
        return ValidationResult(
            "train_val_distribution_shift", ValidationStatus.WARNING,
            f"{len(shifted)} numeric feature(s) show a train/validation mean difference "
            f"exceeding {warning_threshold} standard deviations — worth a look, not "
            "automatically a leakage or bug indicator (a random split can produce this by chance).",
            {"shifted_columns": shifted},
        )
    return ValidationResult(
        "train_val_distribution_shift", ValidationStatus.PASS,
        f"No numeric feature exceeds a {warning_threshold}-std train/validation mean shift.",
        {"checked_columns": len(numeric_cols)},
    )


def check_train_val_missingness_shift(
    X_train: pd.DataFrame, X_val: pd.DataFrame, feature_columns: List[str],
    warning_threshold: float = 0.10,
) -> ValidationResult:
    shifted = {}
    for col in feature_columns:
        if col not in X_train.columns or col not in X_val.columns:
            continue
        train_rate = X_train[col].isna().mean()
        val_rate = X_val[col].isna().mean()
        diff = abs(train_rate - val_rate)
        if diff > warning_threshold:
            shifted[col] = round(float(diff), 4)
    if shifted:
        return ValidationResult(
            "train_val_missingness_shift", ValidationStatus.WARNING,
            f"{len(shifted)} feature(s) have a missingness rate that differs by more than "
            f"{warning_threshold:.0%} between train and validation.",
            {"shifted_columns": shifted},
        )
    return ValidationResult(
        "train_val_missingness_shift", ValidationStatus.PASS,
        f"No feature's missingness rate differs by more than {warning_threshold:.0%} "
        "between train and validation.",
        {"checked_columns": len(feature_columns)},
    )


# ---- 10. Feature registry validation ----------------------------------------

def check_registry_matches_generated_columns(
    generated_columns: List[str], registry: List[dict] = FEATURE_REGISTRY,
) -> ValidationResult:
    registered_names = get_registered_feature_names()
    missing_from_output = set(registered_names) - set(generated_columns)
    if missing_from_output:
        return ValidationResult(
            "registry_matches_generated_columns", ValidationStatus.FAIL,
            f"{len(missing_from_output)} registry entries reference features that were not "
            "actually generated by this run — the registry has drifted from the code.",
            {"missing_from_output": sorted(missing_from_output)},
        )
    return ValidationResult(
        "registry_matches_generated_columns", ValidationStatus.PASS,
        f"All {len(registered_names)} registered features are present in the generated output.",
        {"registered_count": len(registered_names)},
    )


def check_registry_no_duplicates(registry: List[dict] = FEATURE_REGISTRY) -> ValidationResult:
    names = [e["name"] for e in registry]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        return ValidationResult(
            "registry_no_duplicates", ValidationStatus.FAIL,
            f"{len(duplicates)} duplicate registry entry name(s) found.",
            {"duplicate_names": sorted(duplicates)},
        )
    return ValidationResult(
        "registry_no_duplicates", ValidationStatus.PASS,
        f"No duplicate names among {len(names)} registry entries.", {},
    )


def check_registry_entries_complete(registry: List[dict] = FEATURE_REGISTRY) -> ValidationResult:
    incomplete = {}
    invalid_leakage_values = {}
    for entry in registry:
        name = entry.get("name", "<unnamed>")
        missing_fields = _REQUIRED_REGISTRY_FIELDS - set(entry.keys())
        empty_fields = {
            f for f in _REQUIRED_REGISTRY_FIELDS - missing_fields
            if entry[f] in (None, "") and f not in ("historical", "actionable")
        }
        if missing_fields or empty_fields:
            incomplete[name] = {"missing": sorted(missing_fields), "empty": sorted(empty_fields)}
        if entry.get("leakage_risk") not in _VALID_LEAKAGE_RISK_VALUES:
            invalid_leakage_values[name] = entry.get("leakage_risk")

    if incomplete or invalid_leakage_values:
        return ValidationResult(
            "registry_entries_complete", ValidationStatus.FAIL,
            f"{len(incomplete)} registry entries are missing required metadata, "
            f"{len(invalid_leakage_values)} have an invalid leakage_risk value.",
            {"incomplete_entries": incomplete, "invalid_leakage_risk": invalid_leakage_values},
        )
    return ValidationResult(
        "registry_entries_complete", ValidationStatus.PASS,
        f"All {len(registry)} registry entries have complete, valid metadata.", {},
    )


# ---- Orchestration -----------------------------------------------------------

def run_all_checks(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    X_train: Optional[pd.DataFrame] = None,
    X_val: Optional[pd.DataFrame] = None,
) -> List[ValidationResult]:
    """Run every check applicable given what's provided. df is the full engineered
    dataset (post build_aggregate_features + engineer_features, pre-split) — the
    same object src.ml.train._prepare_data() builds before splitting. X_train/X_val
    are optional; when given, the train/validation-specific checks (distribution
    shift, missingness shift) also run.
    """
    feature_columns = numeric_cols + categorical_cols
    results = [
        check_target_not_in_features(feature_columns),
        check_applicant_id_uniqueness(df),
        check_exact_duplicate_rows(df),
        check_infinite_values(df, numeric_cols),
        check_missingness(df, feature_columns),
        check_constant_features(df, numeric_cols),
        check_categorical_cardinality(df, categorical_cols),
        check_unexpected_dtypes(df, numeric_cols, categorical_cols),
        check_suspicious_target_correlation(df, numeric_cols),
        check_registry_leakage_risk_declared(),
        check_registry_matches_generated_columns(list(df.columns)),
        check_registry_no_duplicates(),
        check_registry_entries_complete(),
    ]

    if X_train is not None and X_val is not None:
        results.append(check_train_val_distribution_shift(X_train, X_val, numeric_cols))
        results.append(check_train_val_missingness_shift(X_train, X_val, feature_columns))

    return results


def summarize(results: List[ValidationResult]) -> dict:
    n_pass = sum(1 for r in results if r.status == ValidationStatus.PASS)
    n_warn = sum(1 for r in results if r.status == ValidationStatus.WARNING)
    n_fail = sum(1 for r in results if r.status == ValidationStatus.FAIL)
    overall = "FAIL" if n_fail > 0 else ("WARNING" if n_warn > 0 else "PASS")
    return {"overall_status": overall, "checks_run": len(results),
            "passed": n_pass, "warnings": n_warn, "failed": n_fail}


def generate_markdown_report(results: List[ValidationResult], output_path: str) -> str:
    """Render results to a human-readable Markdown report (reports/data_validation_report.md
    by convention — see src.ml.train.run_data_validation). Returns the report text as well
    as writing it, so callers/tests can inspect the content without re-reading the file.
    """
    import os
    from datetime import datetime, timezone

    summary = summarize(results)
    lines = [
        "# Data Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "**Scope note**: this report covers automated, mechanically-detectable checks "
        "(TARGET leakage, applicant-ID uniqueness, row-count/join integrity, infinite/"
        "constant/high-cardinality features, feature-registry consistency, and a measured "
        "form of train/validation statistical contamination). It does NOT and cannot "
        "guarantee the complete absence of data leakage — semantic leakage requires domain "
        "review that no automated check can fully replace. Read every PASS below as "
        "\"no problem found by this specific check,\" not as a leakage-free guarantee.",
        "",
        f"## Overall status: {summary['overall_status']}",
        "",
        f"- Checks run: {summary['checks_run']}",
        f"- Passed: {summary['passed']}",
        f"- Warnings: {summary['warnings']}",
        f"- Critical failures: {summary['failed']}",
        "",
    ]

    for status in (ValidationStatus.FAIL, ValidationStatus.WARNING, ValidationStatus.PASS):
        matching = [r for r in results if r.status == status]
        if not matching:
            continue
        lines.append(f"## {status.value} ({len(matching)})")
        lines.append("")
        for r in matching:
            lines.append(f"### {r.check_name}")
            lines.append("")
            lines.append(r.message)
            if r.stats:
                lines.append("")
                lines.append("```")
                for k, v in r.stats.items():
                    lines.append(f"{k}: {v}")
                lines.append("```")
            lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    log.info(f"Wrote data validation report to {output_path} (overall status: {summary['overall_status']})")
    return report_text


def raise_if_failed(results: List[ValidationResult]) -> None:
    """Called by src.ml.train.train() right after validation runs. Raises
    DataValidationError (halting training) if any check returned FAIL. Warnings
    are logged, not raised — per the Milestone 2 spec, they should not block training.
    """
    failures = [r for r in results if r.status == ValidationStatus.FAIL]
    warnings_ = [r for r in results if r.status == ValidationStatus.WARNING]
    for w in warnings_:
        log.warning(f"[data validation] WARNING — {w.check_name}: {w.message}")
    if failures:
        details = "; ".join(f"{f.check_name}: {f.message}" for f in failures)
        raise DataValidationError(
            f"{len(failures)} critical data validation check(s) failed — training aborted. {details}"
        )
    log.info(
        f"[data validation] {len(results) - len(warnings_)} check(s) passed cleanly, "
        f"{len(warnings_)} warning(s), 0 failures."
    )
