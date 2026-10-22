from __future__ import annotations
 
import re
from dataclasses import dataclass
from models import Document
from pathlib import Path

from models import Document  # adjust to wherever load_documents/Document live
 


def load_documents(source_dir: Path) -> list[Document]:
    """Load Markdown documents and derive stable citation IDs from filenames."""
    documents: list[Document] = []
    for path in sorted(source_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8-sig").strip().splitlines()

        # Skip optional YAML front matter
        if lines and lines[0].strip() == "---":
            try:
                end = lines.index("---", 1)
            except ValueError:
                raise ValueError(f"Unterminated front matter in {path}")
            lines = lines[end + 1:]
            while lines and not lines[0].strip():
                lines.pop(0)

        if not lines or not lines[0].startswith("# "):
            first = lines[0][:60] if lines else "<empty>"
            raise ValueError(f"Expected '# Title' as first line in {path}, got: {first!r}")

        title = lines[0].removeprefix("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not title or not body:
            raise ValueError(f"Expected a nonempty title and body in {path}")

        documents.append(Document(document_id=path.stem, title=title, text=body, path=path))

    if not documents:
        raise ValueError(f"No Markdown documents found in {source_dir}")
    return documents


"""Step 2: split documents into retrieval-sized chunks with stable citation IDs."""



@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<document_id>#<n>", used for citations
    document_id: str
    title: str
    text: str


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than max_chars on sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    parts: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


MAX_CHARS = 1000      # whole KB articles fit; stays under MiniLM's 256-token limit
OVERLAP_CHARS = 300   # carry the previous paragraph over only if it is this short


def chunk_document(
    doc: Document, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ~max_chars characters.

    When a document splits, each new chunk starts with the previous chunk's
    last paragraph if that paragraph is <= overlap_chars and still fits.
    """
    paragraphs = [
        piece
        for block in doc.text.split("\n\n")
        if block.strip()
        for piece in split_long_paragraph(block.strip(), max_chars)
    ]
    groups: list[list[str]] = [[]]
    size = 0
    for para in paragraphs:
        if groups[-1] and size + len(para) > max_chars:
            carry = groups[-1][-1]
            if len(carry) <= overlap_chars and len(carry) + 2 + len(para) <= max_chars:
                groups.append([carry])
                size = len(carry) + 2
            else:
                groups.append([])
                size = 0
        groups[-1].append(para)
        size += len(para) + 2  # account for the "\n\n" joiner
    return [
        Chunk(f"{doc.document_id}#{i}", doc.document_id, doc.title, "\n\n".join(g))
        for i, g in enumerate(groups, start=1)
    ]


def chunk_documents(
    docs: list[Document], max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    chunks = [c for doc in docs for c in chunk_document(doc, max_chars, overlap_chars)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs; check for duplicate document IDs")
    return chunks
