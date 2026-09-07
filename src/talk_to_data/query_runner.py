"""Loads applicant data into DuckDB and safely validates + executes generated SQL.

Safety: only SELECT statements are allowed, only against the known "applicants"
table/columns, with a row-limit safeguard. This is the last line of defense
against a hallucinated or malicious query before anything touches the database.
"""
import os
from typing import Tuple

import duckdb
import pandas as pd
import sqlparse
from sqlparse.tokens import DML, Name, Keyword

# SQL functions/keywords that can appear as bare names but aren't column references.
_SAFE_NAMES = {
    "AVG", "SUM", "COUNT", "MIN", "MAX", "AS", "AND", "OR", "NOT", "IN", "IS",
    "NULL", "LIKE", "BETWEEN", "ASC", "DESC", "DISTINCT", "ROUND", "CAST",
}

from src.talk_to_data.prompt_templates import SCHEMA_DESCRIPTION
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ALLOWED_TABLE = "applicants"
ALLOWED_COLUMNS = {
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "OCCUPATION_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_DEBT_SUM_SUM", "HAS_BUREAU_HISTORY",
    "PREV_APP_COUNT", "PREV_APP_APPROVAL_RATE", "PREV_APP_REFUSED_COUNT", "HAS_PREV_APPLICATION",
    "INSTALL_LATE_COUNT",
}

SYNTHETIC_DATA_PATH = os.path.join("data", "sample_applicants.csv")

_connection = None
_data_source = None  # "real" or "synthetic", set by _load_applicants_table


def get_connection() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        _connection = duckdb.connect(database=":memory:")
        _load_applicants_table(_connection)
    return _connection


def get_data_source() -> str:
    """Return "real", "synthetic", or None (not loaded yet) — for the UI fallback banner."""
    get_connection()  # ensure loaded
    return _data_source


def _load_df_into_table(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, source_label: str) -> None:
    cols = [c for c in ALLOWED_COLUMNS if c in df.columns]
    con.register("applicants_df", df[cols])
    con.execute(f"CREATE TABLE {ALLOWED_TABLE} AS SELECT * FROM applicants_df")
    log.info(
        f"Loaded {len(df)} rows into DuckDB table '{ALLOWED_TABLE}' "
        f"(source: {source_label}) with columns {cols}"
    )


def _try_load_from_postgres(con: duckdb.DuckDBPyConnection) -> bool:
    """Attempt to serve applicants live from settings.postgres_url via DuckDB's postgres extension.

    Returns True on success (table/view created, _data_source set to "real"), False on any
    failure — network unreachable, INSTALL/LOAD failing, bad connection string, auth failure,
    missing table, etc. Never raises: a broad except is intentional here, since this is an
    optional path and any failure must fall through cleanly to the synthetic fallback.
    """
    global _data_source

    # Standard SQL string-literal escaping (DuckDB's ATTACH doesn't support parameter
    # binding for the connection string — confirmed: it's parsed as a literal, not an
    # expression position).
    #
    # connect_timeout is critical here: without it, an unreachable-but-not-actively-refusing
    # Postgres host (e.g. a suspended/expired database that accepts TCP but never completes
    # the handshake) can hang the connection attempt for the OS-level TCP timeout (60s+),
    # which blocks the whole Streamlit render and gets killed by the platform's own request
    # timeout (observed in production as a 502 Bad Gateway) instead of falling through to the
    # synthetic fallback like every other failure mode here already does.
    sep = "&" if "?" in settings.postgres_url else "?"
    timed_url = f"{settings.postgres_url}{sep}connect_timeout=5"
    escaped_url = timed_url.replace("'", "''")

    try:
        con.execute("INSTALL postgres")
        con.execute("LOAD postgres")
        con.execute(f"ATTACH '{escaped_url}' AS pg_db (TYPE postgres, READ_ONLY)")
    except Exception as e:
        log.warning(
            f"POSTGRES_URL is set but connecting failed ({e}) — falling back to synthetic "
            "demo data. Check the connection string and that Postgres is reachable."
        )
        return False

    try:
        # Prefer a view: queries hit Postgres live, nothing duplicated into DuckDB's memory.
        con.execute(f"CREATE VIEW {ALLOWED_TABLE} AS SELECT * FROM pg_db.public.applicants")
        row_count = con.execute(f"SELECT COUNT(*) FROM {ALLOWED_TABLE}").fetchone()[0]
        log.info(f"Attached Postgres live view '{ALLOWED_TABLE}' -> pg_db.public.applicants ({row_count} rows)")
    except Exception as view_err:
        log.warning(f"CREATE VIEW against Postgres failed ({view_err}), falling back to a one-time table copy")
        try:
            con.execute(f"CREATE TABLE {ALLOWED_TABLE} AS SELECT * FROM pg_db.public.applicants")
            row_count = con.execute(f"SELECT COUNT(*) FROM {ALLOWED_TABLE}").fetchone()[0]
            log.info(f"Copied {row_count} rows from Postgres into DuckDB table '{ALLOWED_TABLE}'")
        except Exception as copy_err:
            log.warning(
                f"POSTGRES_URL is set and reachable but reading pg_db.public.applicants failed "
                f"({copy_err}) — falling back to synthetic demo data. Has "
                "scripts/load_dataset_to_postgres.py been run against this database?"
            )
            return False

    _data_source = "real"
    return True


def _load_applicants_table(con: duckdb.DuckDBPyConnection) -> None:
    global _data_source
    from src.data.loader import build_joined_dataset

    try:
        df = build_joined_dataset(is_train=True)
        _data_source = "real"
        _load_df_into_table(con, df, "real, local CSV")
        return
    except FileNotFoundError:
        pass  # fall through to Postgres, then the synthetic fallback below

    if settings.postgres_url and _try_load_from_postgres(con):
        return

    log.warning(
        "application_train.csv not found — falling back to synthetic demo data "
        f"({SYNTHETIC_DATA_PATH}). Place the real Kaggle dataset in data/ for accurate results."
    )
    df = pd.read_csv(SYNTHETIC_DATA_PATH, comment="#")
    _data_source = "synthetic"
    _load_df_into_table(con, df, "synthetic")


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """Raise SQLValidationError if the SQL is unsafe; otherwise return a cleaned, limited statement."""
    sql = sql.strip().strip(";").strip()
    if not sql:
        raise SQLValidationError("Empty SQL statement")

    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed")

    statement = parsed[0]
    first_token = statement.token_first(skip_cm=True)
    if first_token is None or first_token.ttype is not DML or first_token.value.upper() != "SELECT":
        raise SQLValidationError("Only SELECT statements are allowed")

    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "ATTACH", "COPY", "EXPORT"}
    upper_sql = sql.upper()
    for word in forbidden:
        if word in upper_sql:
            raise SQLValidationError(f"Forbidden keyword detected: {word}")

    if ALLOWED_TABLE not in sql:
        raise SQLValidationError(f"Query must reference the '{ALLOWED_TABLE}' table")

    # Referenced-column check: reject any bare name token that isn't a known
    # column, the table name, or a recognized SQL function/keyword
    # (best-effort token scan, not a full parser, but catches hallucinated columns).
    known_lower = {c.lower() for c in ALLOWED_COLUMNS} | {ALLOWED_TABLE.lower()}
    unknown = _extract_unknown_names(statement, known_lower)
    if unknown:
        raise SQLValidationError(f"Unknown column(s) referenced: {sorted(unknown)}")

    if "LIMIT" not in upper_sql:
        sql = f"{sql} LIMIT {settings.max_query_rows}"

    return sql


def _extract_unknown_names(statement, known_lower: set) -> set:
    """Flatten the parsed statement and collect bare Name tokens that aren't recognized.

    Aliases defined with "AS <alias>" are collected first so later references
    to that alias (e.g. in ORDER BY) aren't mistaken for a hallucinated column.
    """
    tokens = list(statement.flatten())

    aliases = set()
    for i, tok in enumerate(tokens):
        if tok.ttype is Keyword and tok.value.upper() == "AS":
            next_non_ws = next((t for t in tokens[i + 1:] if not t.is_whitespace), None)
            if next_non_ws is not None and next_non_ws.ttype is Name:
                aliases.add(next_non_ws.value.lower())

    unknown = set()
    for i, tok in enumerate(tokens):
        if tok.ttype is not Name:
            continue
        value = tok.value
        if value.upper() in _SAFE_NAMES or value.lower() in known_lower or value.lower() in aliases:
            continue
        prev_non_ws = next((t for t in reversed(tokens[:i]) if not t.is_whitespace), None)
        if prev_non_ws is not None and prev_non_ws.ttype is Keyword and prev_non_ws.value.upper() == "AS":
            continue
        unknown.add(value)
    return unknown


def run_query(sql: str) -> Tuple[pd.DataFrame, str]:
    """Validate then execute SQL. Returns (result_dataframe, cleaned_sql). Raises SQLValidationError on failure."""
    clean_sql = validate_sql(sql)
    con = get_connection()
    result = con.execute(clean_sql).fetchdf()
    log.info(f"Executed query, {len(result)} rows returned: {clean_sql}")
    return result, clean_sql


if __name__ == "__main__":
    df, sql = run_query("SELECT AVG(AMT_INCOME_TOTAL) as avg_income FROM applicants WHERE TARGET = 1")
    print(sql)
    print(df)
