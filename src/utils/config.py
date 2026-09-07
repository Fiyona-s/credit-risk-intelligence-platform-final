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

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()
