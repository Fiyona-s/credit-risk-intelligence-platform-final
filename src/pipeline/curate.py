"""Milestone 8: CURATE stage — writes the transformed applicant-level dataset to
Parquet, locally and (in S3 mode) to the S3 curated/ zone.

Partition scheme: `curated/applicant_features/run_id=<RUN_ID>/part-000.parquet`

Partitioning by run_id (not by e.g. TARGET, education, or an arbitrary date) is
the only partition key that's actually justified at this project's scale:
  - The full applicant-level curated table is ~307k rows x ~194 columns — a
    single Parquet file (tens of MB) reads back in well under a second; there
    is no query-pruning benefit to slicing it further (unlike a multi-billion-
    row warehouse table where partitioning by date lets a query skip most
    files).
  - What run_id partitioning DOES buy: every pipeline run's output is isolated
    under its own prefix, so re-running the pipeline with a new run_id can
    never silently overwrite or corrupt a previous run's curated output, and
    reproducing "what did the pipeline produce on run X" is just reading that
    one partition — this is the actual, load-bearing reason for the scheme,
    not partitioning for its own sake.
Partitioning by e.g. NAME_EDUCATION_TYPE or CODE_GENDER was considered and
rejected: it would fragment the dataset into many small files for no query
benefit at this scale, and — separately — baking a sensitive/fairness-audited
attribute into the physical storage layout is a bad practice regardless of
scale.
"""
import os
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

CURATED_LOCAL_DIR = os.path.join("curated_data", "applicant_features")


def curated_partition_path(run_id: str, base_dir: Optional[str] = None) -> str:
    # base_dir defaults to None, resolved to the module-level CURATED_LOCAL_DIR
    # *inside* the function body rather than as a parameter default — a
    # `base_dir: str = CURATED_LOCAL_DIR` default binds at function-DEFINITION
    # time, so a test monkeypatching curate.CURATED_LOCAL_DIR afterward would
    # have no effect on callers (like write_curated_parquet below) that don't
    # pass base_dir explicitly. Same bug class caught and fixed in
    # src.ml.model_comparison._append_experiment_log and
    # src.ml.calibration._model_artifact_fingerprint in earlier milestones.
    base_dir = base_dir if base_dir is not None else CURATED_LOCAL_DIR
    return os.path.join(base_dir, f"run_id={run_id}")


def write_curated_parquet(df: pd.DataFrame, run_id: str, base_dir: Optional[str] = None) -> str:
    """Writes df to <base_dir>/run_id=<run_id>/part-000.parquet locally.
    Idempotent by construction: re-running with the SAME run_id overwrites this
    exact file (a deterministic path, not an append) — see
    src/pipeline/pipeline.py's docstring for the full idempotency strategy.
    """
    partition_dir = curated_partition_path(run_id, base_dir)
    ensure_dir(partition_dir)
    output_path = os.path.join(partition_dir, "part-000.parquet")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path)

    size_bytes = os.path.getsize(output_path)
    log.info(f"[CURATE] Wrote {len(df):,} rows x {len(df.columns)} cols to {output_path} "
              f"({size_bytes / 1_048_576:.2f} MB)")
    return output_path


def stage_curated_to_s3(local_parquet_path: str, run_id: str) -> Optional[str]:
    """Uploads the local curated Parquet file to the S3 curated/ zone. Only
    called in S3 mode (src.pipeline.pipeline); returns None (no-op) if S3
    isn't configured, rather than raising, since CURATE's local write already
    succeeded and staging to S3 is an additive step, not the load-bearing one.
    """
    if not settings.has_s3:
        return None
    from src.data.cloud_storage import S3Client

    client = S3Client.from_settings()
    key = f"curated/applicant_features/run_id={run_id}/part-000.parquet"
    full_key = client.upload_file(local_parquet_path, key)
    log.info(f"[CURATE] Staged curated Parquet to s3://{client.bucket}/{full_key}")
    return full_key


def read_curated_parquet(run_id: str, base_dir: Optional[str] = None) -> pd.DataFrame:
    """Read back a previously-curated partition — used by tests and by anyone
    wanting to inspect a specific run's output without re-running the pipeline.

    Note (discovered while testing, not a bug): because the file path follows
    Hive-style partition naming (run_id=<RUN_ID>/part-000.parquet), pyarrow's
    read path auto-detects `run_id` as a partition column and adds it back
    onto the returned DataFrame even though it was never one of the columns
    originally written — a useful, free lineage signal (every curated row can
    report which run produced it), not a schema mismatch.
    """
    path = os.path.join(curated_partition_path(run_id, base_dir), "part-000.parquet")
    return pq.read_table(path).to_pandas()
