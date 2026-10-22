"""Step 3: embed chunks and queries.

SentenceTransformerEmbedder is the real embedder for the pipeline.
HashEmbedder is a deterministic, offline stand-in with the same interface,
meant for tests and debugging (lexical overlap only, no semantics).
"""

from __future__ import annotations

import hashlib
import re
import warnings

import numpy as np

from models import Chunk

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_to_text(chunk: Chunk) -> str:
    """The exact text that gets embedded; the title gives every chunk its topic."""
    return f"{chunk.title}\n\n{chunk.text}"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,       # None = auto (cuda/mps/cpu)
        batch_size: int = 64,
        query_prefix: str = "",          # e.g. "query: " for e5 models
        document_prefix: str = "",       # e.g. "passage: " for e5 models
        revision: str | None = None,     # pin a model commit for reproducibility
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text)["input_ids"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.astype(np.float32, copy=False)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class HashEmbedder:
    """Feature-hashing bag-of-words embedder: deterministic, instant, no download."""

    _TOKEN = re.compile(r"\w+")

    def __init__(self, dim: int = 1024, max_tokens: int = 10_000) -> None:
        self.dim = dim
        self.model_name = f"hash-embedder-{dim}"
        self.query_prefix = ""
        self.document_prefix = ""
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self._TOKEN.findall(text))

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in self._TOKEN.findall(text.lower()):
            # md5, not hash(): Python's hash() is randomized per process
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(self.document_prefix + t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(self.query_prefix + text)


Embedder = SentenceTransformerEmbedder | HashEmbedder


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> np.ndarray:
    """Return an (n_chunks, dim) float32 matrix, row i belonging to chunks[i]."""
    texts = [chunk_to_text(c) for c in chunks]
    truncated = [
        c.chunk_id
        for c, t in zip(chunks, texts)
        if embedder.count_tokens(embedder.document_prefix + t) > embedder.max_tokens
    ]
    if truncated:
        warnings.warn(
            f"{len(truncated)} chunk(s) exceed {embedder.max_tokens} tokens and will be "
            f"truncated; lower MAX_CHARS. First few: {truncated[:5]}"
        )
    return embedder.embed_documents(texts)