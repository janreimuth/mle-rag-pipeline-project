"""End-to-end retrieval, safety, generation, and citation validation."""

from .config import Settings
from .documents import chunk_documents, load_documents
from .embeddings import Embedder, SentenceTransformerEmbedder
from .generation import (
    FALLBACK_TEXT,
    DryRunGenerator,
    Generator,
    ProviderGenerator,
    citations_are_valid,
    citations_in,
    is_fallback,
)
from .models import Answer, SearchHit
from .safety import filter_retrieved_evidence, unsafe_instruction
from .store import VectorStore


class RAGPipeline:
    """Runnable assistant pipeline built from explicit replaceable layers."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_revision
        )
        self.generator = generator or self._generator_from_settings()
        documents = load_documents(settings.data_dir)
        self.chunks = chunk_documents(documents)
        self.store = VectorStore(self.embedder.dimension, settings.storage_path)
        try:
            vectors = self.embedder.encode(
                [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks]
            )
            self.store.replace(self.chunks, vectors)
        except Exception:
            self.store.close()
            raise

    def close(self) -> None:
        """Release database and provider resources after a command or test."""

        self.store.close()
        if isinstance(self.generator, ProviderGenerator):
            self.generator.client.close()

    def _generator_from_settings(self) -> Generator:
        """Use provider generation only when an explicit API key is configured."""

        if self.settings.provider.api_key:
            return ProviderGenerator(self.settings.provider)
        return DryRunGenerator()

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[SearchHit], list[str]]:
        """Retrieve evidence and remove redirect instructions before generation."""

        limit = self.settings.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1.")
        vector = self.embedder.encode([question])[0].tolist()
        # Fetch extra candidates so a removed unsafe chunk need not empty context.
        raw_hits = self.store.search(vector, limit=limit + 2)
        safe_hits, events = filter_retrieved_evidence(raw_hits)
        return safe_hits[:limit], events

    def accepted_evidence(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Keep only hits that pass the evidence threshold."""

        return [
            hit
            for hit in hits
            if hit.score > 0 and hit.score >= self.settings.min_score
        ]

    def answer(self, question: str) -> Answer:
        """Return a grounded answer or a visible blocked/abstained outcome."""

        return self.answer_with_evidence(question)[0]

    def answer_with_evidence(
        self, question: str
    ) -> tuple[Answer, list[SearchHit] | None, list[SearchHit]]:
        """Return the answer with its retrieved and accepted hits.

        Retrieved hits are None when the question is rejected before retrieval.
        """

        if not question.strip():
            answer = Answer(
                question=question,
                answer=FALLBACK_TEXT,
                status="abstained",
                abstention_reason="empty_question",
            )
            return answer, None, []
        if unsafe_instruction(question):
            answer = Answer(
                question=question,
                answer="Request blocked by input safety policy.",
                status="blocked_input",
                safety_events=["blocked_user_instruction"],
            )
            return answer, None, []
        hits, events = self.retrieve(question)
        evidence_hits = self.accepted_evidence(hits)
        answer = self._answer_from_hits(question, hits, evidence_hits, events)
        return answer, hits, evidence_hits

    def _answer_from_hits(
        self,
        question: str,
        hits: list[SearchHit],
        evidence_hits: list[SearchHit],
        events: list[str],
    ) -> Answer:
        """Generate from accepted evidence and validate citations."""

        retrieved_sources = list(dict.fromkeys(hit.document_id for hit in hits))
        if not evidence_hits:
            return Answer(
                question=question,
                answer=FALLBACK_TEXT,
                retrieved_sources=retrieved_sources,
                status="abstained",
                abstention_reason="no_accepted_evidence",
                safety_events=events,
            )
        # Do not pass weak neighboring chunks to generation after accepting a hit.
        answer_text = self.generator.generate(question, evidence_hits)
        if is_fallback(answer_text):
            return Answer(
                question=question,
                answer=FALLBACK_TEXT,
                retrieved_sources=retrieved_sources,
                status="abstained",
                abstention_reason="generator_fallback",
                safety_events=events,
            )
        if not citations_are_valid(answer_text, evidence_hits):
            return Answer(
                question=question,
                answer=FALLBACK_TEXT,
                retrieved_sources=retrieved_sources,
                status="invalid_citations",
                safety_events=events + ["invalid_citations_rejected"],
            )
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations_in(answer_text),
            retrieved_sources=retrieved_sources,
            status="answered",
            safety_events=events,
        )
