"""Tests for source loading, chunk metadata, and evidence isolation."""

from pathlib import Path

import pytest

from solution.rag_project.documents import chunk_documents, load_documents
from solution.rag_project.models import Document
from solution.rag_project.safety import unsafe_instruction


def test_chunks_keep_source_identity_and_isolate_malicious_example() -> None:
    """Security example text should be filterable without removing its policy."""

    data_dir = Path(__file__).parents[2] / "data" / "knowledge_base"
    chunks = chunk_documents(load_documents(data_dir))
    safety_chunks = [
        chunk for chunk in chunks if chunk.document_id == "retrieved-content-safety"
    ]

    assert len(safety_chunks) == 3
    assert all(
        chunk.chunk_id.startswith("retrieved-content-safety-")
        for chunk in safety_chunks
    )
    assert sum(unsafe_instruction(chunk.text) for chunk in safety_chunks) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_windows_are_rejected_even_for_short_documents(
    size: int, overlap: int
) -> None:
    document = Document(
        document_id="short", title="Short", text="One word.", path=Path("short.md")
    )
    with pytest.raises(ValueError):
        chunk_documents([document], max_words=size, overlap_words=overlap)


def test_window_tail_adds_new_content() -> None:
    document = Document(
        document_id="long",
        title="Long",
        text=" ".join(map(str, range(151))),
        path=Path("long.md"),
    )
    chunks = chunk_documents([document], max_words=90, overlap_words=15)
    assert len(chunks) == 2
    assert chunks[-1].text.split()[-1] == "150"
    assert [chunk.position for chunk in chunks] == [0, 1]


@pytest.mark.parametrize("content", ["", "# Title", "Missing title\n\nBody"])
def test_invalid_sources_have_actionable_errors(tmp_path: Path, content: str) -> None:
    (tmp_path / "bad.md").write_text(content)
    with pytest.raises(ValueError, match="bad.md"):
        load_documents(tmp_path)
