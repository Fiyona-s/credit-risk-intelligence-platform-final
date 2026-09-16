"""Milestone 8: batch ingestion stage — identifies input files, reads them, records
row counts and source metadata, and (in S3 mode) stages the raw files into the
S3 raw/ zone. Does NOT parse/join/engineer anything — that's the TRANSFORM stage
(src/pipeline/pipeline.py), which calls src.data.loader/src.data.preprocessor,
the existing single source of truth for feature engineering, unchanged here.

Two source modes, both supported so the pipeline (and the whole project) keeps
working without any cloud credentials:
    source="local"  Read data/*.csv directly (the existing REQUIRED_FILES/
                     OPTIONAL_FILES lists from src.data.loader — reused, not
                     redefined). Default. No AWS credentials needed.
    source="s3"      Same local read (S3 is not yet a live data store to READ
                     from for this project — the Kaggle dataset lives in data/
                     or nowhere, per the existing, unchanged synthetic-fallback
                     design), but additionally stages each discovered file to
                     s3://bucket/prefix/raw/<run_id>/<filename> as an immutable
                     upload, giving the pipeline a durable, versioned copy of
                     exactly what was ingested for this run — the "raw/" zone's
                     purpose (see src/data/cloud_storage.py's docstring).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.data.loader import OPTIONAL_FILES, REQUIRED_FILES
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ALL_EXPECTED_FILES = REQUIRED_FILES + OPTIONAL_FILES


@dataclass
class IngestionResult:
    source: str
    run_id: str
    discovered_files: List[str]
    missing_required_files: List[str]
    row_counts: Dict[str, int] = field(default_factory=dict)
    staged_s3_keys: List[str] = field(default_factory=list)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def total_input_rows(self) -> int:
        return sum(self.row_counts.values())

    @property
    def is_complete(self) -> bool:
        return len(self.missing_required_files) == 0


def discover_input_files(data_dir: str = None) -> List[str]:
    """Which of the expected Home Credit files are actually present in data_dir."""
    data_dir = data_dir or settings.data_dir
    return [f for f in ALL_EXPECTED_FILES if os.path.exists(os.path.join(data_dir, f))]


def ingest(source: str, run_id: str, data_dir: str = None, stage_to_s3: bool = None) -> IngestionResult:
    """Discover, read, and (optionally) stage the raw input files for one pipeline run.

    stage_to_s3 defaults to (source == "s3") — i.e. s3 mode stages by default,
    local mode never touches S3 at all. Can be forced False even in s3 mode
    (used by --dry-run, which must never write anything).
    """
    data_dir = data_dir or settings.data_dir
    if stage_to_s3 is None:
        stage_to_s3 = (source == "s3")

    log.info(f"[INGEST] run_id={run_id} source={source} data_dir={data_dir}")

    discovered = discover_input_files(data_dir)
    missing_required = [f for f in REQUIRED_FILES if f not in discovered]
    log.info(f"[INGEST] Files discovered: {len(discovered)} of {len(ALL_EXPECTED_FILES)} expected")
    if missing_required:
        log.warning(f"[INGEST] Missing required file(s): {missing_required}")

    tables: Dict[str, pd.DataFrame] = {}
    row_counts: Dict[str, int] = {}
    for filename in discovered:
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path)
        table_name = filename.replace(".csv", "")
        tables[table_name] = df
        row_counts[filename] = len(df)
        log.info(f"[INGEST] {filename} rows: {len(df):,}")

    staged_keys: List[str] = []
    if stage_to_s3:
        from src.data.cloud_storage import S3Client

        client = S3Client.from_settings()
        for filename in discovered:
            key = f"raw/{run_id}/{filename}"
            full_key = client.upload_file(os.path.join(data_dir, filename), key)
            staged_keys.append(full_key)
        log.info(f"[INGEST] Staged {len(staged_keys)} file(s) to S3 raw/ zone under run_id={run_id}")

    return IngestionResult(
        source=source, run_id=run_id, discovered_files=discovered,
        missing_required_files=missing_required, row_counts=row_counts,
        staged_s3_keys=staged_keys, tables=tables,
    )
