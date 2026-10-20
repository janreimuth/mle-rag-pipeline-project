"""Configuration loaded from environment variables for local or live runs."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class ProviderConfig:
    """OpenAI-compatible generation provider details."""

    name: str
    api_key: str = field(repr=False)
    base_url: str
    model: str


@dataclass(frozen=True)
class Settings:
    """All configurable inputs for building and evaluating the pipeline."""

    data_dir: Path
    evaluation_file: Path
    storage_path: Path | None
    embedding_model: str
    top_k: int
    min_score: float
    provider: ProviderConfig
    embedding_revision: str | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1.")
        if not 0 <= self.min_score <= 1:
            raise ValueError("MIN_SCORE must be a finite number between 0 and 1.")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Load local defaults while allowing explicit environment overrides."""

        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        load_dotenv(root / ".env")
        provider_name = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        providers = {
            "groq": ProviderConfig(
                "groq",
                os.getenv("GROQ_API_KEY", ""),
                "https://api.groq.com/openai/v1",
                os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            ),
            "mistral": ProviderConfig(
                "mistral",
                os.getenv("MISTRAL_API_KEY", ""),
                "https://api.mistral.ai/v1",
                os.getenv("MISTRAL_MODEL", "ministral-14b-latest"),
            ),
        }
        if provider_name not in providers:
            raise ValueError("LLM_PROVIDER must be 'groq' or 'mistral'.")
        provider = providers[provider_name]
        storage_value = os.getenv("VECTOR_STORAGE_PATH", "")
        storage_path = root / storage_value if storage_value else None
        model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        revision = os.getenv("EMBEDDING_REVISION") or (
            DEFAULT_EMBEDDING_REVISION if model == DEFAULT_EMBEDDING_MODEL else None
        )
        return cls(
            data_dir=root / "data" / "knowledge_base",
            evaluation_file=root / "data" / "evaluation" / "questions.json",
            storage_path=storage_path,
            embedding_model=model,
            top_k=int(os.getenv("TOP_K", "3")),
            min_score=float(os.getenv("MIN_SCORE", "0.30")),
            provider=provider,
            embedding_revision=revision,
        )
