"""Step 4: retrieve the top-k chunks for a question, or abstain.

A question gets no hits (= abstain) when even the best chunk scores below
min_score. The default threshold was calibrated on data/eval/calibration.json
with all-MiniLM-L6-v2 and is only meaningful for that model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from models import Chunk
from embedding import DEFAULT_MODEL, Embedder, embed_chunks

DEFAULT_K = 3
# v2 (24 calibration cases): the threshold only rejects clearly off-topic questions.
# Lowest answerable case 0.336 (cal13), highest far off-topic 0.252 (cal06); cal04
# (0.290) sits in between. Near-topic unanswerable questions (cal05, cal17-21) score
# 0.35-0.55 and are left to the LLM to decline. v1 was 0.41 (6 cases).
DEFAULT_MIN_SCORE = 0.30
CALIBRATED_MODEL = DEFAULT_MODEL  # the model DEFAULT_MIN_SCORE belongs to


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    hits: list[Hit]         # hits >= min_score, best first; empty means abstain
    top_score: float        # best score overall, also when below the cutoff
    min_score: float

    @property
    def abstain(self) -> bool:
        return not self.hits


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    hits: list[Hit]         # top-k, best first, if top_score >= min_score; else empty (abstain)
    top_score: float        # best score overall, also when below the cutoff
    min_score: float
 
    @property
    def abstain(self) -> bool:
        return not self.hits
 
 
class VectorStore(Protocol):
    """Where chunk vectors live: InMemoryStore (numpy) or QdrantStore (qdrant_store.py)."""
 
    def __len__(self) -> int: ...
 
    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        """Top-k chunks by cosine similarity, best first."""
        ...
 
 
class InMemoryStore:
    """Brute-force numpy search over L2-normalized vectors."""
 
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"Got {vectors.shape[0]} vectors for {len(chunks)} chunks")
        self.chunks = chunks
        self.vectors = vectors
 
    def __len__(self) -> int:
        return len(self.chunks)
 
    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        scores = self.vectors @ query_vector
        # stable sort on -score: ties keep the original chunk order
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    
class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        k: int = DEFAULT_K,
        min_score: float | None = None,
        vectors: np.ndarray | None = None,  # precomputed embed_chunks(chunks, embedder)
        store: VectorStore | None = None,   # default: InMemoryStore over the chunks
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever without chunks")
        if k < 1:
            raise ValueError("k must be at least 1")
        if min_score is None:
            if embedder.model_name != CALIBRATED_MODEL:
                warnings.warn(
                    f"DEFAULT_MIN_SCORE was calibrated for {CALIBRATED_MODEL}, not "
                    f"{embedder.model_name}; pass min_score explicitly"
                )
            min_score = DEFAULT_MIN_SCORE

        self.chunks = chunks
        self.embedder = embedder
        self.k = k
        self.min_score = min_score
        if store is None:
            if vectors is None:
                vectors = embed_chunks(chunks, embedder)
            store = InMemoryStore(chunks, vectors)
        self.store = store

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        """Top-k chunks by cosine similarity, best first, without a cutoff."""
        k = min(k or self.k, len(self.store))
        return self.store.search(self.embedder.embed_query(question), k)

    def retrieve(
        self, question: str, k: int | None = None, min_score: float | None = None
    ) -> RetrievalResult:
        """Top-k chunks if the best one reaches min_score, else no hits (abstain).
 
        min_score is a question-level gate (is this question on-topic at all?),
        not a per-chunk filter: once a question passes, all top-k chunks go to
        the LLM, so weaker second sources of multi-source questions survive.
        """
        threshold = self.min_score if min_score is None else min_score
        candidates = self.search(question, k)
        on_topic = candidates[0].score >= threshold
        return RetrievalResult(
            question=question,
            hits=candidates if on_topic else [],
            top_score=candidates[0].score,
            min_score=threshold,
        )