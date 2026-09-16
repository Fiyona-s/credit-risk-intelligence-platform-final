"""Milestone 8: LOAD stage — writes the curated applicant-level dataset into a
PostgreSQL analytical table (`applicant_features`), separate from the chatbot's
existing `applicants` table (src/talk_to_data/query_runner.py, populated by the
pre-existing scripts/load_dataset_to_postgres.py) — different table, different
purpose, no shared schema, per this milestone's "do not redesign the existing
application's unrelated database schema" constraint.

Uses the same mechanism the existing chatbot-loading script already uses —
DuckDB's `postgres` extension — rather than adding a new DB driver dependency
(psycopg2/sqlalchemy) for a second, parallel way of talking to Postgres.

Idempotency: CREATE OR REPLACE TABLE is a full-refresh load, not an incremental
append. Running the pipeline twice with the same (or different) run_id against
the same database_url replaces the table's contents wholesale — never
duplicates rows, never appends. This is the appropriate, non-over-engineered
choice for a ~307k-row batch table (see the report's § H): a proper incremental
upsert would need a durable notion of "what changed since last run," which this
project's batch/snapshot data model doesn't have (the source CSVs are a single
static Kaggle snapshot, not a continuously-updated feed).
"""
from typing import Callable, Optional

import pandas as pd

from src.data.cloud_storage import retry_with_backoff
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_TABLE_NAME = "applicant_features"

# Postgres reserves plenty of column-name-safe characters, but keep this
# explicit and simple rather than trusting arbitrary column names.
_PANDAS_TO_POSTGRES_TYPE = {
    "int64": "BIGINT",
    "int32": "INTEGER",
    "float64": "DOUBLE PRECISION",
    "float32": "REAL",
    "bool": "BOOLEAN",
    "object": "TEXT",
}


def _postgres_type_for(dtype) -> str:
    return _PANDAS_TO_POSTGRES_TYPE.get(str(dtype), "TEXT")


def build_create_table_sql(df: pd.DataFrame, table_name: str = DEFAULT_TABLE_NAME) -> str:
    """Pure function (no DB connection) — builds the DDL for documentation/
    testing purposes. The actual load uses DuckDB's `CREATE OR REPLACE TABLE
    ... AS SELECT` (which infers types itself from the DataFrame), so this
    function's output is illustrative/testable rather than executed verbatim —
    see load_dataframe_to_postgres()'s docstring for why.
    """
    columns_sql = ",\n    ".join(f'"{col}" {_postgres_type_for(dtype)}' for col, dtype in df.dtypes.items())
    return f'CREATE TABLE "{table_name}" (\n    {columns_sql}\n);'


def load_dataframe_to_postgres(
    df: pd.DataFrame, database_url: str, table_name: str = DEFAULT_TABLE_NAME,
    id_column: str = "SK_ID_CURR", duckdb_connect: Optional[Callable] = None,
) -> dict:
    """Loads df into Postgres as `table_name`, full-refresh (CREATE OR REPLACE
    TABLE), then adds a PRIMARY KEY on id_column for uniqueness + fast lookups.

    duckdb_connect is injectable (defaults to duckdb.connect) so tests can pass
    a mock connection factory without a real Postgres instance — see
    tests/test_data_pipeline.py's mocked-load tests.

    Retried (via src.data.cloud_storage.retry_with_backoff, reused rather than
    reimplemented) on connection-level failures only — a bad database_url or a
    SQL/type error is NOT retried (retrying a deterministic error just wastes
    time before failing the same way again); see the report's § I.
    """
    import duckdb as duckdb_module

    connect_fn = duckdb_connect or duckdb_module.connect

    def _do_load():
        con = connect_fn(database=":memory:")
        try:
            con.register("curated_df", df)
            con.execute("INSTALL postgres")
            con.execute("LOAD postgres")
            escaped_url = database_url.replace("'", "''")
            con.execute(f"ATTACH '{escaped_url}' AS pg_db (TYPE postgres)")
            con.execute(f'CREATE OR REPLACE TABLE pg_db.public."{table_name}" AS SELECT * FROM curated_df')
            if id_column in df.columns:
                try:
                    con.execute(
                        f'ALTER TABLE pg_db.public."{table_name}" ADD PRIMARY KEY ("{id_column}")'
                    )
                except Exception as pk_error:
                    # Non-fatal: the load itself succeeded; a primary key failure
                    # (e.g. duplicate values slipping past validation) is logged,
                    # not silently swallowed, but doesn't roll back a successful load.
                    log.warning(f"[LOAD] Could not add PRIMARY KEY on {id_column}: {pk_error}")
            row_count = con.execute(f'SELECT COUNT(*) FROM pg_db.public."{table_name}"').fetchone()[0]
            return row_count
        finally:
            con.close()

    row_count = retry_with_backoff(_do_load, max_retries=3)
    log.info(f"[LOAD] Loaded {row_count:,} rows into Postgres table '{table_name}'")
    return {"table_name": table_name, "row_count": row_count}
