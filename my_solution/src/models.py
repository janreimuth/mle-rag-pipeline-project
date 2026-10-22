from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    path: Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<document_id>#<n>", used for citations
    document_id: str
    title: str
    text: str