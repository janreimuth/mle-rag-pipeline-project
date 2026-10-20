"""Configuration and local vector-store lifecycle regressions."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solution.rag_project.config import Settings
from solution.rag_project.embeddings import HashEmbedder
from solution.rag_project.models import Chunk
from solution.rag_project.pipeline import RAGPipeline
from solution.rag_project.store import VectorStore


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_PROVIDER", "mistal"),
        ("TOP_K", "0"),
        ("MIN_SCORE", "nan"),
        ("MIN_SCORE", "inf"),
        ("MIN_SCORE", "-0.1"),
        ("MIN_SCORE", "1.1"),
    ],
)
def test_invalid_configuration_fails_early(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_storage_path_is_project_relative_and_credentials_are_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VECTOR_STORAGE_PATH", "storage/qdrant")
    monkeypatch.setenv("GROQ_API_KEY", "test-private-key")
    settings = Settings.from_env(tmp_path)
    assert settings.storage_path == tmp_path / "storage/qdrant"
    assert "test-private-key" not in repr(settings)


def test_custom_models_do_not_inherit_the_minilm_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "another/model")
    assert Settings.from_env().embedding_revision is None


def test_invalid_rebuild_preserves_the_existing_index() -> None:
    store = VectorStore(2)
    chunk = Chunk(
        chunk_id="one-0", document_id="one", title="One", text="Evidence.", position=0
    )
    try:
        store.replace([chunk], np.array([[1.0, 0.0]], dtype=np.float32))
        for vectors in (
            np.zeros((0, 2), dtype=np.float32),
            np.array([[float("nan"), 1.0]], dtype=np.float32),
        ):
            with pytest.raises(ValueError):
                store.replace([chunk], vectors)
            assert store.search([1.0, 0.0], 1)[0].chunk_id == chunk.chunk_id
    finally:
        store.close()


def test_persistent_collection_can_be_reopened_after_close(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), storage_path=tmp_path / "qdrant")
    for _ in range(2):
        pipeline = RAGPipeline(settings, embedder=HashEmbedder())
        try:
            assert pipeline.chunks
            assert pipeline.retrieve("invoice")[0]
        finally:
            pipeline.close()
