"""One-time load of the real joined applicant dataset into Postgres, for the chatbot's
optional live-Postgres path (src/talk_to_data/query_runner.py).

*** RUN MANUALLY, ONCE, from a machine that has the real Kaggle CSVs in data/. ***
This script is NOT run as part of `docker-compose up`, is NOT run automatically
anywhere in this repo, and its output — a Postgres database — is the only place
the real dataset lives outside of Kaggle itself and your local data/ folder. The
dataset is never committed to git and never baked into the Docker image; only the
Postgres connection string (POSTGRES_URL) is configured on the deployed instance.

Prerequisite: the real Kaggle CSVs already present locally in data/ (same
precondition as `python -m src.ml.train`).

Usage:
    python scripts/load_dataset_to_postgres.py --connection-string "postgresql://user:pass@host:5432/dbname"
    # or, with POSTGRES_URL already set in the environment:
    python scripts/load_dataset_to_postgres.py

No new Python dependency is added for this — DuckDB's `postgres` extension
(auto-installed via `INSTALL postgres` at runtime, requires outbound network
access once) handles both the read (application CSVs, via the existing loader)
and the write (into Postgres) without psycopg2/sqlalchemy.
"""
import argparse
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import build_joined_dataset  # noqa: E402
from src.talk_to_data.query_runner import ALLOWED_COLUMNS  # noqa: E402


def load(connection_string: str) -> None:
    print("Building the joined dataset from local CSVs (application_train.csv + auxiliary tables)...")
    df = build_joined_dataset(is_train=True)

    cols = [c for c in ALLOWED_COLUMNS if c in df.columns]
    filtered = df[cols]
    print(f"Filtered to the chatbot-facing schema: {len(cols)} columns (of {df.shape[1]} in the full joined dataset).")

    con = duckdb.connect(database=":memory:")
    con.register("filtered_df", filtered)

    escaped = connection_string.replace("'", "''")
    print("Installing/loading DuckDB's postgres extension...")
    con.execute("INSTALL postgres")
    con.execute("LOAD postgres")

    print("Attaching the target Postgres database...")
    con.execute(f"ATTACH '{escaped}' AS pg_db (TYPE postgres)")

    print("Writing pg_db.public.applicants (CREATE OR REPLACE TABLE)...")
    con.execute("CREATE OR REPLACE TABLE pg_db.public.applicants AS SELECT * FROM filtered_df")

    row_count = con.execute("SELECT COUNT(*) FROM pg_db.public.applicants").fetchone()[0]
    col_info = con.execute("DESCRIBE pg_db.public.applicants").fetchall()

    print(f"\nDone. Loaded {row_count:,} rows into pg_db.public.applicants.")
    print(f"Columns ({len(col_info)}):")
    for name, dtype, *_ in col_info:
        print(f"  {name:<28s} {dtype}")

    con.execute("DETACH pg_db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-string",
        default=os.getenv("POSTGRES_URL", ""),
        help="Postgres connection string (defaults to the POSTGRES_URL env var if not passed).",
    )
    args = parser.parse_args()

    if not args.connection_string:
        parser.error(
            "No connection string given. Pass --connection-string or set POSTGRES_URL."
        )

    load(args.connection_string)
