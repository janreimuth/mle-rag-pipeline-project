"""Qdrant-backed vector storage with source metadata retained in payloads."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from .embeddings import VectorMatrix
from .models import Chunk, SearchHit


class VectorStore:
    """Index and search chunks in local Qdrant using cosine similarity."""

    collection_name = "northstar_support"

    def __init__(self, dimension: int, storage_path: Path | None = None) -> None:
        self.dimension = dimension
        self.client = (
            QdrantClient(path=str(storage_path))
            if storage_path is not None
            else QdrantClient(":memory:")
        )

    def replace(self, chunks: list[Chunk], vectors: VectorMatrix) -> None:
        """Rebuild the collection so source changes cannot leave stale chunks."""

        # Reject invalid rebuilds before deleting an existing persistent index.
        if not chunks or vectors.shape != (len(chunks), self.dimension):
            raise ValueError("Provide one correctly sized vector per nonempty chunk.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors must contain only finite values.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique.")
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension, distance=models.Distance.COSINE
            ),
        )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def close(self) -> None:
        """Release the local storage lock before another pipeline opens it."""

        self.client.close()

    def search(self, query_vector: list[float], limit: int) -> list[SearchHit]:
        """Return scored chunks reconstructed from Qdrant payload metadata."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        results: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchHit(
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload["document_id"]),
                    title=str(payload["title"]),
                    text=str(payload["text"]),
                    position=int(payload["position"]),
                    score=float(point.score),
                )
            )
        return results
