"""Embedding implementations for production-like runs and inexpensive tests."""

import hashlib
import re
from typing import Protocol

import numpy as np

VectorMatrix = np.ndarray[tuple[int, int], np.dtype[np.float32]]


class Embedder(Protocol):
    """Minimal embedding contract required by the vector index."""

    @property
    def dimension(self) -> int:
        """Return vector dimensionality."""

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Encode texts as normalized vectors."""


class SentenceTransformerEmbedder:
    """Dense semantic embeddings backed by an evaluated local model."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        # Smoke tests do not need to import the dense model runtime.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, revision=revision, device="cpu")
        probe = self._model.encode(
            ["embedding dimension probe"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        probe_matrix = np.asarray(probe, dtype=np.float32)
        if probe_matrix.ndim != 2 or probe_matrix.shape[1] == 0:
            raise ValueError("The configured embedding model has no fixed dimension.")
        # The vector store dimension must match the exact matrix this model returns.
        self._dimension = int(probe_matrix.shape[1])

    @property
    def dimension(self) -> int:
        """Return the model output size required by the vector store."""

        return self._dimension

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Generate unit-length dense vectors for cosine similarity."""

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Deterministic lexical vectorizer used only for fast tests and smoke runs."""

    def __init__(self, dimension: int = 2048) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the fixed test-vector dimension."""

        return self._dimension

    def encode(self, texts: list[str]) -> VectorMatrix:
        """Hash visible terms into normalized vectors without network downloads."""

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "big") % self.dimension
                matrix[row, column] += 1.0
            length = float(np.linalg.norm(matrix[row]))
            if length:
                matrix[row] /= length
        return matrix
