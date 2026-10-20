"""Typed records shared by ingestion, retrieval, generation, and evaluation."""

from pathlib import Path

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One approved Markdown source loaded from the project corpus."""

    document_id: str
    title: str
    text: str
    path: Path


class Chunk(BaseModel):
    """One independently retrievable passage with citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    position: int


class SearchHit(Chunk):
    """A retrieved chunk paired with its cosine similarity score."""

    score: float


class Answer(BaseModel):
    """Final answer contract returned by both dry and live generation."""

    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    status: str
    abstention_reason: str | None = None
    safety_events: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    """One labeled question used to measure system behavior."""

    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    should_abstain: bool


class EvaluationRow(BaseModel):
    """Per-question evidence retained for the generated report."""

    id: str
    question: str
    expected_sources: list[str]
    should_abstain: bool
    retrieved_sources: list[str]
    first_relevant_rank: int | None
    top_score: float | None
    status: str
    abstention_reason: str | None = None
    answer: str
    citations: list[str]
    safety_events: list[str]
    relevance: float | None
    citation_correct: bool | None
    faithful: bool | None
    decision_correct: bool


class EvaluationSummary(BaseModel):
    """Aggregate metrics and inspectable rows from one evaluation run."""

    retrieval_recall_at_k: float | None
    retrieval_hit_rate_at_k: float | None
    mean_reciprocal_rank: float | None
    answer_relevance: float | None
    citation_correctness: float | None
    faithfulness: float | None
    decision_accuracy: float
    rows: list[EvaluationRow]
