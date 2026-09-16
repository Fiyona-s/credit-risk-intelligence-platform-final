"""Milestone 8: the batch data pipeline orchestrator.

    INGEST -> VALIDATE -> TRANSFORM -> CURATE -> LOAD -> WRITE METADATA

A deterministic Python function pipeline, not a DAG scheduler — see
reports/data_engineering_cloud_report.md § F for why Airflow/similar was not
introduced (this project's single nightly-scale batch job needs none of a
scheduler's retry/backfill/dependency-graph machinery; cron + this script is
the entire "orchestration platform" a project at this scale actually needs).

VALIDATE is deliberately two-tier, matching what the reused validation
functions actually check:
  1. Raw-file completeness (src.pipeline.ingest — are the required CSVs
     present at all) gates whether TRANSFORM runs.
  2. The full Milestone 2 data-quality/leakage suite
     (src.data.validation.run_all_checks, reused verbatim, NOT reimplemented)
     runs on the TRANSFORMED/curated DataFrame — this is where checks like
     "TARGET not in features" or "registry matches generated columns"
     structurally have to run, since they need the joined+engineered columns
     to exist. A FAIL here stops the pipeline before CURATE/LOAD; WARNINGs are
     recorded in the run manifest but do not stop the run (identical semantics
     to src.ml.train's own use of the same functions).

TRANSFORM calls src.data.loader.build_joined_dataset() and
src.data.preprocessor.engineer_features() — the EXACT SAME functions
src.ml.train/src.ml.predict use. This pipeline does not reimplement or fork
feature engineering; it is a second consumer of the same source of truth.

Idempotency: every stage's output path is a deterministic function of run_id
(curated/applicant_features/run_id=<RUN_ID>/part-000.parquet,
metadata/run_<RUN_ID>.json) — rerunning the same run_id overwrites those exact
files rather than accumulating duplicates. The Postgres LOAD stage is a
full-refresh (CREATE OR REPLACE TABLE), independent of run_id, so any
successful run leaves Postgres in a consistent "latest run's data" state,
never with duplicated or partially-appended rows — see
src/pipeline/postgres_loader.py's docstring.
"""
import argparse
import time
from datetime import datetime, timezone
from typing import Optional

from src.data.preprocessor import engineer_features, get_feature_columns
from src.data.validation import (
    check_non_negative_fields,
    run_all_checks,
    summarize,
)
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# The canonical, business-meaningful non-negative fields in the curated
# dataset (Milestone 8 § 9) — a small, deliberately narrow list, not every
# numeric column (many engineered ratios can legitimately be negative, e.g.
# BUREAU_DEBT_CREDIT_RATIO variants are not in scope here; only raw monetary
# amounts, which can never be negative by construction in this domain).
NON_NEGATIVE_FIELDS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]


def generate_run_id() -> str:
    """Deterministic-enough default run_id: date + a short random-free counter
    would need external state, so this uses a timestamp with second precision
    — unique in practice for a manually-triggered batch pipeline, and the
    caller can always pass --run-id explicitly for a fully deterministic,
    reproducible identifier (e.g. in a test or a scheduled cron invocation
    that wants the same run_id if rerun on failure).
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_pipeline(
    source: str = "local", run_id: Optional[str] = None,
    data_dir: Optional[str] = None, dry_run: bool = False,
    load_to_postgres: bool = True,
) -> dict:
    """Executes the full INGEST -> VALIDATE -> TRANSFORM -> CURATE -> LOAD ->
    WRITE METADATA pipeline. Returns the run manifest as a dict.

    dry_run=True (S3 mode only, meaningfully) verifies configuration/
    connectivity (src.data.cloud_storage.S3Client.dry_run_check) and returns
    immediately WITHOUT reading, transforming, or writing any data — see
    § 16 of the milestone spec.
    """
    run_id = run_id or generate_run_id()
    start_time = time.monotonic()
    log.info(f"=== Pipeline run {run_id} starting (source={source}, dry_run={dry_run}) ===")

    if dry_run:
        return _dry_run(source, run_id)

    from src.pipeline.ingest import ingest

    # ---- INGEST ----
    ingestion = ingest(source=source, run_id=run_id, data_dir=data_dir)
    if not ingestion.is_complete:
        log.error(f"[PIPELINE] Missing required input file(s): {ingestion.missing_required_files} — stopping "
                   "before VALIDATE/TRANSFORM.")
        manifest = _build_and_write_manifest(
            run_id=run_id, source=source, ingestion=ingestion,
            validation_status="FAIL", validation_warnings=0, validation_failures=1,
            output_rows=0, curated_path=None, postgres_result=None,
            duration_seconds=time.monotonic() - start_time,
        )
        return manifest.to_dict()

    # ---- TRANSFORM (the existing feature-engineering pipeline, unchanged) ----
    from src.data.loader import build_joined_dataset

    log.info("[TRANSFORM] Building applicant aggregates via src.data.loader.build_joined_dataset() "
              "(the same function src.ml.train/src.ml.predict use — not duplicated here)")
    df = build_joined_dataset(is_train=True)
    df = engineer_features(df)
    log.info(f"[TRANSFORM] Curated rows: {len(df):,}, columns: {df.shape[1]}")

    # ---- VALIDATE (Milestone 2 suite, reused verbatim, on the transformed data) ----
    numeric_cols, categorical_cols = get_feature_columns(df)
    results = run_all_checks(df, numeric_cols, categorical_cols)
    results.append(check_non_negative_fields(df, NON_NEGATIVE_FIELDS))
    summary = summarize(results)
    log.info(f"[VALIDATE] status={summary['overall_status']} passed={summary['passed']} "
              f"warnings={summary['warnings']} failures={summary['failed']}")

    if summary["overall_status"] == "FAIL":
        log.error("[PIPELINE] Validation FAILED — stopping before CURATE/LOAD. See failures above.")
        manifest = _build_and_write_manifest(
            run_id=run_id, source=source, ingestion=ingestion,
            validation_status="FAIL", validation_warnings=summary["warnings"],
            validation_failures=summary["failed"], output_rows=0, curated_path=None,
            postgres_result=None, duration_seconds=time.monotonic() - start_time,
        )
        return manifest.to_dict()

    # ---- CURATE (Parquet) ----
    from src.pipeline.curate import stage_curated_to_s3, write_curated_parquet

    curated_path = write_curated_parquet(df, run_id=run_id)
    s3_staged = False
    if source == "s3" and settings.has_s3:
        stage_curated_to_s3(curated_path, run_id=run_id)
        s3_staged = True

    # ---- LOAD (Postgres) ----
    postgres_result = None
    if load_to_postgres and settings.has_database:
        from src.pipeline.postgres_loader import load_dataframe_to_postgres

        try:
            postgres_result = load_dataframe_to_postgres(df, settings.database_url)
        except Exception as e:
            log.error(f"[LOAD] Postgres load failed: {e} — curated Parquet output is still valid; "
                      "the pipeline does not roll back CURATE for a downstream LOAD failure.")
    elif load_to_postgres:
        log.info("[LOAD] DATABASE_URL not configured — skipping Postgres load "
                  "(curated Parquet output is unaffected).")

    # ---- WRITE METADATA ----
    manifest = _build_and_write_manifest(
        run_id=run_id, source=source, ingestion=ingestion,
        validation_status=summary["overall_status"], validation_warnings=summary["warnings"],
        validation_failures=summary["failed"], output_rows=len(df), curated_path=curated_path,
        postgres_result=postgres_result, duration_seconds=time.monotonic() - start_time,
        s3_staged=s3_staged,
    )

    log.info(f"=== Pipeline run {run_id} completed in {manifest.duration_seconds:.1f}s ===")
    return manifest.to_dict()


def _dry_run(source: str, run_id: str) -> dict:
    if source != "s3":
        log.info("[DRY RUN] source=local requires no external configuration — nothing to check.")
        return {"run_id": run_id, "source": source, "dry_run": True, "status": "ok_no_cloud_config_needed"}

    from src.data.cloud_storage import S3Client

    client = S3Client.from_settings()
    result = client.dry_run_check()
    log.info(f"[DRY RUN] S3 configuration check: {result}")
    return {"run_id": run_id, "source": source, "dry_run": True, **result}


def _build_and_write_manifest(
    run_id, source, ingestion, validation_status, validation_warnings, validation_failures,
    output_rows, curated_path, postgres_result, duration_seconds, s3_staged=False,
):
    from src.ml.calibration import FEATURE_VERSION
    from src.pipeline.metadata import PIPELINE_VERSION, build_run_manifest, stage_metadata_to_s3, write_metadata_local

    manifest = build_run_manifest(
        run_id=run_id, source=source, input_files=len(ingestion.discovered_files),
        input_rows=ingestion.total_input_rows, output_rows=output_rows,
        validation_status=validation_status, feature_version=FEATURE_VERSION,
        output_format="parquet", pipeline_version=PIPELINE_VERSION,
        curated_path=curated_path,
        postgres_table=postgres_result["table_name"] if postgres_result else None,
        postgres_row_count=postgres_result["row_count"] if postgres_result else None,
        s3_staged=s3_staged, validation_warnings=validation_warnings,
        validation_failures=validation_failures, duration_seconds=duration_seconds,
    )
    write_metadata_local(manifest)
    if source == "s3":
        stage_metadata_to_s3(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "s3"], default="local")
    parser.add_argument("--run-id", default=None, help="Deterministic run identifier; auto-generated if omitted")
    parser.add_argument("--data-dir", default=None, help="Override settings.data_dir for local CSV input")
    parser.add_argument("--dry-run", action="store_true",
                         help="Verify configuration only (S3 mode); never reads/writes data")
    parser.add_argument("--skip-postgres", action="store_true", help="Run CURATE but skip the Postgres LOAD stage")
    args = parser.parse_args()

    import json

    from src.data.cloud_storage import S3NotConfiguredError

    try:
        result = run_pipeline(
            source=args.source, run_id=args.run_id, data_dir=args.data_dir,
            dry_run=args.dry_run, load_to_postgres=not args.skip_postgres,
        )
    except S3NotConfiguredError as e:
        # A clean, actionable CLI message instead of a raw traceback — this is
        # a configuration problem the user can immediately fix, not a bug.
        print(json.dumps({"status": "error", "error_type": "S3NotConfigured", "message": str(e)}, indent=2))
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
