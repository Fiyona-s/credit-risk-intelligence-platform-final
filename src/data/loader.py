"""Loads the Home Credit Default Risk tables from data/.

The main application tables are loaded directly here. The six auxiliary
history tables (bureau, bureau_balance, previous_application, POS_CASH_balance,
credit_card_balance, installments_payments) are also loaded here as raw
per-row tables; aggregating them to one row per SK_ID_CURR is handled by
src.data.preprocessor.build_aggregate_features so that exact aggregation
logic runs identically at train and inference time.
"""
import os
from typing import List, Optional

import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

REQUIRED_FILES = [
    "application_train.csv",
    "application_test.csv",
]
OPTIONAL_FILES = [
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]


def _path(filename: str) -> str:
    return os.path.join(settings.data_dir, filename)


def check_data_available() -> List[str]:
    """Return list of missing required files (empty list = ready to load)."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(_path(f))]
    return missing


def load_application_train() -> pd.DataFrame:
    df = pd.read_csv(_path("application_train.csv"))
    log.info(f"Loaded application_train.csv: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_application_test() -> pd.DataFrame:
    df = pd.read_csv(_path("application_test.csv"))
    log.info(f"Loaded application_test.csv: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def _load_optional(filename: str) -> Optional[pd.DataFrame]:
    path = _path(filename)
    if not os.path.exists(path):
        log.warning(f"{filename} not found, skipping features derived from it")
        return None
    df = pd.read_csv(path)
    log.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_bureau() -> Optional[pd.DataFrame]:
    """Raw bureau.csv: one row per external credit line per applicant (SK_ID_CURR -> many)."""
    return _load_optional("bureau.csv")


def load_bureau_balance() -> Optional[pd.DataFrame]:
    """Raw bureau_balance.csv: monthly balance/status history per bureau credit (SK_ID_BUREAU -> many)."""
    return _load_optional("bureau_balance.csv")


def load_previous_application() -> Optional[pd.DataFrame]:
    """Raw previous_application.csv: applicant's previous loan applications with this lender."""
    return _load_optional("previous_application.csv")


def load_pos_cash_balance() -> Optional[pd.DataFrame]:
    """Raw POS_CASH_balance.csv: monthly balance snapshots of previous POS/cash loans."""
    return _load_optional("POS_CASH_balance.csv")


def load_credit_card_balance() -> Optional[pd.DataFrame]:
    """Raw credit_card_balance.csv: monthly balance snapshots of previous credit cards."""
    return _load_optional("credit_card_balance.csv")


def load_installments_payments() -> Optional[pd.DataFrame]:
    """Raw installments_payments.csv: repayment history on previous credits."""
    return _load_optional("installments_payments.csv")


def build_joined_dataset(is_train: bool = True, use_bureau_features: bool = True) -> pd.DataFrame:
    """Load the main application table and left-join the auxiliary-table aggregates onto it.

    use_bureau_features=True (default) joins the bureau/previous-application/POS/credit-card/
    installments aggregates built by src.data.preprocessor.build_aggregate_features. Set to
    False to train/predict on application-level features only, e.g. for an ablation comparison.
    """
    base = load_application_train() if is_train else load_application_test()
    before_rows = base.shape[0]

    if use_bureau_features:
        from src.data.preprocessor import build_aggregate_features
        base = build_aggregate_features(base)
        assert base.shape[0] == before_rows, (
            f"Row count changed after joining aggregates: {before_rows} -> {base.shape[0]} "
            "(aggregate tables must be one row per SK_ID_CURR before the left-join)"
        )

    log.info(f"Final joined dataset ({'train' if is_train else 'test'}): {base.shape}")
    return base


def load_from_postgres() -> pd.DataFrame:
    """Pull the Postgres-hosted applicants table (loaded via scripts/load_dataset_to_postgres.py)
    as a DataFrame, for use by app.py's get_sample_data(). Raises RuntimeError on any failure
    (unset POSTGRES_URL, unreachable database, missing table) so callers can catch it and fall
    through to the synthetic fallback, mirroring query_runner.py's own fallback chain.

    Modeled directly on query_runner.py::_try_load_from_postgres — same attach pattern, same
    pg_db.public.applicants table. Note this table only has the ~21 chatbot-facing columns
    (query_runner.ALLOWED_COLUMNS), not the full 166-column joined dataset used for training —
    enough for EDA and a richer-than-synthetic fallback elsewhere, but not full model feature
    parity (no EXT_SOURCE_1/2/3, etc.), so Risk Prediction/Model Evaluation/Explainability run
    in the same degraded-but-real-data-backed mode they already tolerate for the synthetic
    fallback today, via the existing "if col not in df.columns: df[col] = None" pattern.
    """
    import concurrent.futures
    import duckdb

    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is not set")

    # connect_timeout is passed through too, but is NOT sufficient on its own — observed in
    # production (Render) that DuckDB's postgres extension can still hang past it on some
    # network paths (e.g. a host that accepts TCP but the extension's own connect/handshake
    # logic doesn't honor connect_timeout the way libpq's CLI does). The hard wall-clock
    # timeout below via a background thread is the real guarantee: if the attempt hasn't
    # returned within timeout_seconds, we give up and move on regardless of what the
    # underlying (possibly still-blocked) network call eventually does. The abandoned
    # thread may leak in the background until the call finally errors or the process
    # restarts — an acceptable tradeoff against hanging every page load indefinitely.
    sep = "&" if "?" in settings.postgres_url else "?"
    timed_url = f"{settings.postgres_url}{sep}connect_timeout=5"
    escaped_url = timed_url.replace("'", "''")

    def _attempt() -> pd.DataFrame:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute("INSTALL postgres")
            con.execute("LOAD postgres")
            con.execute(f"ATTACH '{escaped_url}' AS pg_db (TYPE postgres, READ_ONLY)")
            return con.execute("SELECT * FROM pg_db.public.applicants").df()
        finally:
            con.close()

    # NOT a `with ThreadPoolExecutor() as pool:` block: that context manager calls
    # shutdown(wait=True) on exit unconditionally, which would block waiting for the
    # hung worker thread anyway — silently defeating the whole point of this timeout.
    # shutdown(wait=False) here lets this function return immediately; the abandoned
    # thread keeps running (and leaking its socket/connection) in the background.
    timeout_seconds = 8
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_attempt)
    try:
        df = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as e:
        pool.shutdown(wait=False)
        raise RuntimeError(
            f"Postgres load timed out after {timeout_seconds}s (hard wall-clock limit, "
            "connect_timeout alone wasn't enough on this network path)"
        ) from e
    except Exception as e:
        pool.shutdown(wait=False)
        raise RuntimeError(f"Postgres load failed: {e}") from e

    pool.shutdown(wait=False)
    log.info(f"Loaded {len(df)} rows from Postgres (pg_db.public.applicants)")
    return df


if __name__ == "__main__":
    missing = check_data_available()
    if missing:
        log.error(f"Missing required files in {settings.data_dir}/: {missing}")
        log.error("Download the Home Credit Default Risk dataset from Kaggle and place CSVs in data/.")
    else:
        df = build_joined_dataset(is_train=True)
        print(df.head())
        print(f"\nTARGET distribution:\n{df['TARGET'].value_counts(normalize=True)}")
