"""Qdrant in local mode: no server, data in a folder (or in memory).

    store = QdrantStore.open(chunks, embedder, path=None)        # in memory
    store = QdrantStore.open(chunks, embedder, path=".qdrant")   # persisted
    retriever = Retriever(chunks, embedder, store=store)

Each collection name ends with a fingerprint of the chunks and the embedding
model. An unchanged corpus reuses the saved collection without re-embedding;
any change (documents, chunking, model) creates a fresh one and removes stale
ones. Note: a local Qdrant folder can be opened by only one process at a time.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from models import Chunk
from embedding import Embedder, embed_chunks
from retrieval import Hit

DEFAULT_COLLECTION = "knowledge_base"
_ID_NAMESPACE = uuid.UUID("6f1c2a52-6c3e-4f4e-9a55-0d1b7e3c9a10")  # fixed: stable point IDs


def fingerprint(chunks: list[Chunk], model_name: str) -> str:
    """Changes whenever the chunks or the embedding model change."""
    h = hashlib.sha256(model_name.encode())
    for c in chunks:
        for part in (c.chunk_id, c.title, c.text):
            h.update(b"\0" + part.encode())
    return h.hexdigest()[:12]


def point_id(chunk_id: str) -> str:
    """Qdrant needs integer or UUID ids; derive a stable UUID from the chunk ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def open(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        path: str | Path | None = None,
        prefix: str = DEFAULT_COLLECTION,
        vectors: np.ndarray | None = None,  # precomputed embed_chunks(chunks, embedder)
    ) -> QdrantStore:
        client = QdrantClient(path=str(path)) if path else QdrantClient(location=":memory:")
        collection = f"{prefix}_{fingerprint(chunks, embedder.model_name)}"

        # Remove collections of this prefix built from older corpora or models.
        for existing in client.get_collections().collections:
            if existing.name.startswith(f"{prefix}_") and existing.name != collection:
                client.delete_collection(existing.name)

        store = cls(client, collection)
        if client.collection_exists(collection) and len(store) == len(chunks):
            return store  # unchanged corpus: reuse, no embedding needed

        if vectors is None:
            vectors = embed_chunks(chunks, embedder)
        store._build(chunks, vectors)
        return store

    def _build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"Got {vectors.shape[0]} vectors for {len(chunks)} chunks")
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "position": position,  # original order, for stable ties
                    },
                )
                for position, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
            wait=True,
        )

    def __len__(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        points = self.client.query_points(
            self.collection, query=query_vector.tolist(), limit=k, with_payload=True
        ).points
        # Same tie-breaking as InMemoryStore: higher score first, then original order.
        points = sorted(points, key=lambda p: (-p.score, p.payload["position"]))
        return [
            Hit(
                Chunk(
                    chunk_id=p.payload["chunk_id"],
                    document_id=p.payload["document_id"],
                    title=p.payload["title"],
                    text=p.payload["text"],
                ),
                float(p.score),
            )
            for p in points
        ]

    def close(self) -> None:
        self.client.close()
