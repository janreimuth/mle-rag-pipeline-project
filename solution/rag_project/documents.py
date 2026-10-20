"""Document loading and paragraph-aware chunk construction."""

from pathlib import Path

from .models import Chunk, Document


def load_documents(source_dir: Path) -> list[Document]:
    """Load Markdown documents and derive stable citation IDs from filenames."""

    documents: list[Document] = []
    for path in sorted(source_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8").strip()
        lines = raw_text.splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"Expected a Markdown title in {path}")
        title = lines[0].removeprefix("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not title or not body:
            raise ValueError(f"Expected a nonempty title and body in {path}")
        documents.append(
            Document(document_id=path.stem, title=title, text=body, path=path)
        )
    if not documents:
        raise ValueError(f"No Markdown documents found in {source_dir}")
    return documents


def _window_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split an overlong paragraph while retaining context near boundaries."""

    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words")
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, max(len(words) - overlap_words, 1), step)
        if words[start : start + max_words]
    ]


def chunk_documents(
    documents: list[Document], max_words: int = 90, overlap_words: int = 15
) -> list[Chunk]:
    """Create citable chunks while keeping short policy paragraphs intact."""

    if not isinstance(max_words, int) or not isinstance(overlap_words, int):
        raise TypeError("Chunk size and overlap must be integer word counts.")
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError("Use max_words > 0 and 0 <= overlap_words < max_words.")
    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = [
            paragraph.replace("\n", " ").strip()
            for paragraph in document.text.split("\n\n")
            if paragraph.strip()
        ]
        position = 0
        for paragraph in paragraphs:
            for passage in _window_text(paragraph, max_words, overlap_words):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-{position}",
                        document_id=document.document_id,
                        title=document.title,
                        text=passage,
                        position=position,
                    )
                )
                position += 1
    return chunks
