"""Milestone 8: thin S3 object-storage abstraction for the data pipeline (src/pipeline/*).

Zone layout under s3://<S3_BUCKET>/<S3_PREFIX>/:

    raw/        Original source files (application_train.csv + the six auxiliary
                tables), staged verbatim from data/. Treated as IMMUTABLE — never
                overwritten in place; a re-ingest of the same run_id re-uploads
                the same bytes, a genuinely new dataset would use a new run_id.
    validated/  A small marker/manifest per run recording that the raw inputs
                passed src.data.validation's checks (not a copy of the data
                itself — duplicating ~2GB of CSVs a second time for a pass/fail
                flag would be wasteful; see reports/data_engineering_cloud_report.md
                § E for the reasoning).
    curated/    The transformed, applicant-level Parquet dataset
                (curated/applicant_features/run_id=<RUN_ID>/part-000.parquet).
    features/   Reserved for a future ML-ready feature export distinct from the
                curated analytical table, if the two ever need to diverge (e.g.
                a feature-store-style point-in-time table). Not populated by the
                current pipeline — curated/ already IS the applicant-level
                feature table for this project's scale; see the report for why
                a separate features/ zone isn't populated yet rather than
                populating it just to fill the directory.
    metadata/   One run-manifest JSON per pipeline run (src/pipeline/metadata.py).

Credentials come ENTIRELY from boto3's normal credential chain (environment
variables, ~/.aws/credentials, an attached IAM role, etc.) — never hard-coded,
never read from this project's own .env beyond AWS_REGION/S3_BUCKET/S3_PREFIX
(which are configuration, not secrets). Constructing settings.Settings() never
requires AWS credentials to be present; only actually calling an S3Client
method does, and only when S3_BUCKET is configured (settings.has_s3).
"""
import time
from dataclasses import dataclass
from typing import List, Optional

from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Bounded retry for transient network/throttling failures only — see
# src/pipeline/pipeline.py's docstring and the report's § I for the exact
# boundary between "retried" and "fails immediately."
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0


class S3NotConfiguredError(Exception):
    """Raised when an S3Client method is called but settings.S3_BUCKET is unset.
    Distinct from a boto3/network error — this is a configuration problem the
    caller should surface clearly, not retry."""


def _is_not_found_error(e: Exception) -> bool:
    """True if e represents a genuine "object does not exist" response (a real
    botocore.exceptions.ClientError with Error.Code 404/NoSuchKey/NotFound, or
    a generic exception whose message says as much) — as opposed to a
    transient network/throttling error, which should be retried instead.
    """
    response = getattr(e, "response", None)
    if response:
        error_code = response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return True
    message = str(e).lower()
    return "404" in message or "not found" in message or "nosuchkey" in message


def retry_with_backoff(fn, max_retries: int = DEFAULT_MAX_RETRIES, backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS):
    """Bounded retry with linear backoff, for transient S3 errors (throttling,
    timeouts, connection resets) ONLY. Does not catch/retry S3NotConfiguredError
    or a botocore ClientError for a genuine permissions/not-found problem — see
    callers below for the exact exception handling.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below if retries exhausted
            last_exc = e
            if attempt < max_retries:
                log.warning(f"S3 operation failed (attempt {attempt}/{max_retries}): {e} — retrying")
                time.sleep(backoff_seconds * attempt)
            else:
                log.error(f"S3 operation failed after {max_retries} attempts: {e}")
    raise last_exc


@dataclass
class S3Client:
    """Thin wrapper around boto3's S3 client, scoped to this project's bucket/
    prefix. Every method raises S3NotConfiguredError immediately (no retry) if
    S3_BUCKET is unset — a configuration problem, not a transient failure.
    """
    bucket: str
    prefix: str
    region: str

    def __post_init__(self):
        self._client = None  # lazy: only construct a real boto3 client when first used

    @classmethod
    def from_settings(cls) -> "S3Client":
        return cls(bucket=settings.s3_bucket, prefix=settings.s3_prefix.strip("/"), region=settings.aws_region)

    def _require_configured(self):
        if not self.bucket:
            raise S3NotConfiguredError(
                "S3_BUCKET is not configured — set S3_BUCKET (and optionally S3_PREFIX/AWS_REGION) "
                "in .env to enable S3 mode. Local mode (DATA_SOURCE=local) needs none of this."
            )

    def _get_client(self):
        if self._client is None:
            import boto3  # lazy import: processes that never touch S3 don't pay this cost

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload_file(self, local_path: str, key: str) -> str:
        """Upload a local file to s3://bucket/prefix/key. Returns the full key.
        Retried on transient failure (network/throttling)."""
        self._require_configured()
        full_key = self._full_key(key)
        client = self._get_client()
        retry_with_backoff(lambda: client.upload_file(local_path, self.bucket, full_key))
        log.info(f"Uploaded {local_path} -> s3://{self.bucket}/{full_key}")
        return full_key

    def upload_bytes(self, data: bytes, key: str) -> str:
        """Upload in-memory bytes (e.g. a small JSON manifest) without a temp file."""
        self._require_configured()
        full_key = self._full_key(key)
        client = self._get_client()
        retry_with_backoff(lambda: client.put_object(Bucket=self.bucket, Key=full_key, Body=data))
        log.info(f"Uploaded {len(data)} bytes -> s3://{self.bucket}/{full_key}")
        return full_key

    def download_file(self, key: str, local_path: str) -> None:
        self._require_configured()
        full_key = self._full_key(key)
        client = self._get_client()
        retry_with_backoff(lambda: client.download_file(self.bucket, full_key, local_path))
        log.info(f"Downloaded s3://{self.bucket}/{full_key} -> {local_path}")

    def read_bytes(self, key: str) -> bytes:
        self._require_configured()
        full_key = self._full_key(key)
        client = self._get_client()

        def _read():
            response = client.get_object(Bucket=self.bucket, Key=full_key)
            return response["Body"].read()

        return retry_with_backoff(_read)

    def exists(self, key: str) -> bool:
        """True/False for whether the object exists — NOT retried on a genuine
        404 (that's an expected, deterministic answer, not a transient
        failure); still retried on a network-level error reaching S3 at all.

        The 404-vs-transient check happens INSIDE the retried callable, not
        after retry_with_backoff already exhausted its attempts — a version
        of this that let a 404 propagate out of the retried function and only
        classified it in an outer except (as an earlier version of this method
        did) would retry a deterministic "not found" answer up to
        max_retries times before finally recognizing it, wasting the entire
        backoff delay on a case that was never going to change. Caught by a
        test using a fake S3 client that raises a plain Exception (not a real
        botocore.exceptions.ClientError) for a 404 — which also means the
        error-code detection below tolerates both a real ClientError's
        `.response` structure and a generic exception whose message merely
        mentions "404" or "not found", for robustness against exactly that
        kind of test double as well as real-world variation.
        """
        self._require_configured()
        full_key = self._full_key(key)
        client = self._get_client()

        def _head():
            try:
                client.head_object(Bucket=self.bucket, Key=full_key)
                return True
            except Exception as e:
                if _is_not_found_error(e):
                    return False
                raise  # a genuine transient/other error — let retry_with_backoff retry it

        return retry_with_backoff(_head)

    def list_objects(self, prefix: str = "") -> List[str]:
        """Keys (full, including self.prefix) under prefix, non-recursive-paginated
        (handles >1000 objects via boto3's paginator)."""
        self._require_configured()
        full_prefix = self._full_key(prefix)
        client = self._get_client()

        def _list():
            keys = []
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys

        return retry_with_backoff(_list)

    def dry_run_check(self) -> dict:
        """Verify configuration + connectivity WITHOUT modifying any cloud data —
        backs `python -m src.pipeline.pipeline --source s3 --dry-run`. Checks
        that the bucket is reachable and the credential chain resolves, via a
        single lightweight head_bucket call (no object read/write).
        """
        self._require_configured()
        client = self._get_client()
        try:
            retry_with_backoff(lambda: client.head_bucket(Bucket=self.bucket), max_retries=1)
            return {"configured": True, "bucket_reachable": True, "bucket": self.bucket,
                     "prefix": self.prefix, "region": self.region}
        except Exception as e:
            return {"configured": True, "bucket_reachable": False, "bucket": self.bucket,
                     "prefix": self.prefix, "region": self.region, "error": str(e)}
