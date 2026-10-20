"""Shared lightweight pipeline fixtures for behavioral tests."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from solution.rag_project.config import Settings
from solution.rag_project.embeddings import HashEmbedder
from solution.rag_project.generation import DryRunGenerator
from solution.rag_project.pipeline import RAGPipeline


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never use developer credentials or local .env files in automated tests."""

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    for name in (
        "LLM_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_REVISION",
        "TOP_K",
        "MIN_SCORE",
        "VECTOR_STORAGE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def smoke_pipeline() -> Iterator[RAGPipeline]:
    """Build an in-memory pipeline without downloading the dense model."""

    settings = replace(
        Settings.from_env(project_root=Path(__file__).parents[2]),
        embedding_model="hash-smoke-test-only",
        embedding_revision=None,
        # Lexical smoke scores are not calibrated to the dense baseline threshold.
        min_score=0.35,
    )
    pipeline = RAGPipeline(
        settings, embedder=HashEmbedder(), generator=DryRunGenerator()
    )
    try:
        yield pipeline
    finally:
        pipeline.close()
