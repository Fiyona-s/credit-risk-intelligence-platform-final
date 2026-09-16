"""Milestone 8: run metadata / data lineage manifest.

One JSON manifest per pipeline run, written to metadata/run_<RUN_ID>.json
locally, and to the S3 metadata/ zone in S3 mode. Answers "where did this
feature dataset come from" — source, input files/rows, validation outcome,
output rows/format, and the exact feature/pipeline version used — without
ever storing secrets (no connection strings, no credentials, no API keys;
verified by test_data_pipeline.py's test_secrets_not_written_to_metadata).
"""
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

PIPELINE_VERSION = "v1.0"
METADATA_LOCAL_DIR = "metadata"


@dataclass
class RunManifest:
    run_id: str
    source: str
    input_files: int
    input_rows: int
    output_rows: int
    validation_status: str
    feature_version: str
    output_format: str
    pipeline_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    curated_path: Optional[str] = None
    postgres_table: Optional[str] = None
    postgres_row_count: Optional[int] = None
    s3_staged: bool = False
    validation_warnings: int = 0
    validation_failures: int = 0
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_run_manifest(**kwargs) -> RunManifest:
    return RunManifest(**kwargs)


def write_metadata_local(manifest: RunManifest, base_dir: Optional[str] = None) -> str:
    # See src.pipeline.curate.curated_partition_path's comment for why this
    # resolves the module-level default inside the body, not as a parameter default.
    base_dir = base_dir if base_dir is not None else METADATA_LOCAL_DIR
    ensure_dir(base_dir)
    path = os.path.join(base_dir, f"run_{manifest.run_id}.json")
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, default=str)
    log.info(f"[METADATA] Wrote run manifest to {path}")
    return path


def stage_metadata_to_s3(manifest: RunManifest) -> Optional[str]:
    """Uploads the manifest JSON to the S3 metadata/ zone. No-op (returns None)
    if S3 isn't configured — metadata is always written locally regardless."""
    from src.utils.config import settings

    if not settings.has_s3:
        return None
    from src.data.cloud_storage import S3Client

    client = S3Client.from_settings()
    key = f"metadata/run_{manifest.run_id}.json"
    body = json.dumps(manifest.to_dict(), indent=2, default=str).encode("utf-8")
    full_key = client.upload_bytes(body, key)
    log.info(f"[METADATA] Staged run manifest to s3://{client.bucket}/{full_key}")
    return full_key
