"""Central config loaded from .env. Import `settings` everywhere instead of calling os.getenv directly."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-latest"))

    risk_threshold_low: float = field(default_factory=lambda: _float("RISK_THRESHOLD_LOW", 0.10))
    risk_threshold_medium: float = field(default_factory=lambda: _float("RISK_THRESHOLD_MEDIUM", 0.30))

    # Cost-sensitive threshold selection (src/ml/train.py's post-fit threshold sweep,
    # src/ml/predict.py's cost-optimal decision, app.py's Risk Prediction tab).
    # Business framing: missing a defaulter (a false negative — approving someone who
    # actually defaults) is assumed costlier than wrongly flagging a good applicant
    # (a false positive — extra manual review), hence FN_COST > FP_COST by default.
    fp_cost: float = field(default_factory=lambda: _float("FP_COST", 1.0))
    fn_cost: float = field(default_factory=lambda: _float("FN_COST", 5.0))

    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    models_dir: str = field(default_factory=lambda: os.getenv("MODELS_DIR", "models"))

    max_query_rows: int = field(default_factory=lambda: _int("MAX_QUERY_ROWS", 200))

    # Optional, deployment-only: a Postgres connection string (e.g. Render's free Postgres
    # add-on) pre-loaded with the real dataset via scripts/load_dataset_to_postgres.py.
    # Lets src/talk_to_data/query_runner.py serve the chatbot from real data on a deployed
    # instance (where the real Kaggle CSV is never present, correctly gitignored) without
    # ever committing the dataset — same "optional but recommended" spirit as
    # groq_api_key/gemini_api_key above. Leave unset to use the real local CSV if present,
    # else the synthetic fallback (default behavior, no setup needed).
    postgres_url: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", ""))

    # Milestone 8 — data engineering / cloud platform layer (src/pipeline/*,
    # src/data/cloud_storage.py). Entirely optional and additive: the existing
    # application (Streamlit/FastAPI/chatbot) never reads these, so a fresh
    # checkout with none of this configured is completely unaffected — the
    # same "must work without cloud credentials" principle already established
    # for postgres_url/groq_api_key/gemini_api_key above.
    #
    # data_source selects the pipeline's ingestion mode: "local" (read data/*.csv
    # directly, the default, no cloud credentials needed) or "s3" (also stage/read
    # from the configured S3 bucket). Never affects the Streamlit/FastAPI app.
    data_source: str = field(default_factory=lambda: os.getenv("DATA_SOURCE", "local"))
    aws_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: os.getenv("S3_PREFIX", "credit-risk-platform"))
    # A DELIBERATELY SEPARATE setting from postgres_url above — postgres_url
    # backs the chatbot's optional live-Postgres read path only (an existing,
    # narrower-scoped feature); database_url is this milestone's own analytical
    # `applicant_features` table target. The two may point at the same physical
    # Postgres instance if the user wants, but are configured and reasoned
    # about independently, so neither feature's behavior is coupled to the other.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_s3(self) -> bool:
        return bool(self.s3_bucket)

    @property
    def has_database(self) -> bool:
        return bool(self.database_url)


settings = Settings()
