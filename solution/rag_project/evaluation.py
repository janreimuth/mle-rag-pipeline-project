"""Evaluation metrics and report generation for the complete RAG path."""

import json
from pathlib import Path

from .generation import CITATION_PATTERN, DryRunGenerator
from .models import EvaluationCase, EvaluationRow, EvaluationSummary, SearchHit
from .pipeline import RAGPipeline


def load_cases(path: Path) -> list[EvaluationCase]:
    """Read the labeled benchmark without changing its expected outcomes."""

    records = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(record) for record in records]


def _rank_of_expected(retrieved: list[str], expected: list[str]) -> int | None:
    """Return the first one-based relevant rank, or no rank when missed."""

    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in expected:
            return rank
    return None


def _term_coverage(answer: str, required_terms: list[str]) -> float:
    """Measure whether expected key facts appear in a supported answer."""

    if not required_terms:
        return 1.0
    lowered = answer.lower()
    matches = sum(term.lower() in lowered for term in required_terms)
    return matches / len(required_terms)


def _literal_support(answer: str, citations: list[str], hits: list[SearchHit]) -> bool:
    """Verify copied evidence, not semantic entailment of arbitrary paraphrases."""

    content = " ".join(CITATION_PATTERN.sub("", answer).split()).casefold()
    return bool(content) and any(
        content
        in " ".join(
            " ".join(hit.text.split()).casefold()
            for hit in sorted(hits, key=lambda hit: hit.position)
            if hit.document_id == document_id
        )
        for document_id in citations
    )


def evaluate(pipeline: RAGPipeline, cases: list[EvaluationCase]) -> EvaluationSummary:
    """Run retrieval and final answers through transparent deterministic rubrics."""

    if not cases:
        raise ValueError("The evaluation set must contain at least one case.")
    if any(not case.expected_sources for case in cases if not case.should_abstain):
        raise ValueError("Supported cases need at least one expected source.")
    rows: list[EvaluationRow] = []
    for case in cases:
        # Score the same hits the answer was built from.
        final_answer, hits, evidence = pipeline.answer_with_evidence(case.question)
        if hits is None:
            # Blocked questions skip retrieval; ranking metrics still need it.
            hits, _ = pipeline.retrieve(case.question)
        # k counts chunks: duplicate document IDs must not compress their ranks.
        retrieved = [hit.document_id for hit in hits]
        first_rank = _rank_of_expected(retrieved, case.expected_sources)
        expected_citations = set(case.expected_sources)
        cited_evidence = (
            final_answer.status == "answered"
            and bool(final_answer.citations)
            and set(final_answer.citations).issubset(
                {hit.document_id for hit in evidence}
            )
        )
        # Abstention belongs to decision accuracy, not answer-claim verification.
        faithful: bool | None = None
        if final_answer.status == "answered":
            if not cited_evidence:
                faithful = False
            elif _literal_support(
                final_answer.answer, final_answer.citations, evidence
            ):
                faithful = True
            elif isinstance(pipeline.generator, DryRunGenerator):
                faithful = False
        # Answer metrics score supported cases; decision accuracy owns abstention.
        relevance: float | None = None
        citation_correct: bool | None = None
        if not case.should_abstain:
            relevance = (
                _term_coverage(final_answer.answer, case.required_terms)
                if final_answer.status == "answered"
                else 0.0
            )
            citation_correct = cited_evidence and set(final_answer.citations).issubset(
                expected_citations
            )
        decision_correct = (
            final_answer.status == "abstained"
            if case.should_abstain
            else final_answer.status == "answered"
        )
        rows.append(
            EvaluationRow(
                id=case.id,
                question=case.question,
                expected_sources=case.expected_sources,
                should_abstain=case.should_abstain,
                retrieved_sources=retrieved,
                first_relevant_rank=first_rank,
                top_score=hits[0].score if hits else None,
                status=final_answer.status,
                abstention_reason=final_answer.abstention_reason,
                answer=final_answer.answer,
                citations=final_answer.citations,
                safety_events=final_answer.safety_events,
                relevance=relevance,
                citation_correct=citation_correct,
                faithful=faithful,
                decision_correct=decision_correct,
            )
        )
    supported_rows = [
        row for row, case in zip(rows, cases, strict=True) if not case.should_abstain
    ]
    answered_rows = [row for row in rows if row.status == "answered"]
    recall = (
        sum(
            len(set(row.retrieved_sources) & set(row.expected_sources))
            / len(set(row.expected_sources))
            for row in supported_rows
        )
        / len(supported_rows)
        if supported_rows
        else None
    )
    hit_rate = (
        sum(row.first_relevant_rank is not None for row in supported_rows)
        / len(supported_rows)
        if supported_rows
        else None
    )
    mrr = (
        sum(
            0.0 if row.first_relevant_rank is None else 1.0 / row.first_relevant_rank
            for row in supported_rows
        )
        / len(supported_rows)
        if supported_rows
        else None
    )
    return EvaluationSummary(
        retrieval_recall_at_k=recall,
        retrieval_hit_rate_at_k=hit_rate,
        mean_reciprocal_rank=mrr,
        answer_relevance=(
            sum(row.relevance or 0.0 for row in supported_rows) / len(supported_rows)
            if supported_rows
            else None
        ),
        citation_correctness=(
            sum(row.citation_correct is True for row in supported_rows)
            / len(supported_rows)
            if supported_rows
            else None
        ),
        faithfulness=(
            sum(row.faithful is True for row in answered_rows) / len(answered_rows)
            if answered_rows and all(row.faithful is not None for row in answered_rows)
            else None
        ),
        decision_accuracy=sum(row.decision_correct for row in rows) / len(rows),
        rows=rows,
    )


def _format_sources(sources: list[str]) -> str:
    """Produce compact source display for report tables."""

    return ", ".join(sources) if sources else "-"


def _format_relevance(value: float | None) -> str:
    """Show per-question relevance, which unsupported cases do not receive."""

    return f"{value:.2f}" if value is not None else "n/a"


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "not rated"


def render_report(pipeline: RAGPipeline, summary: EvaluationSummary) -> str:
    """Create a reviewable Markdown report with results and recommendations."""

    settings = pipeline.settings
    misses = [
        row
        for row in summary.rows
        if (
            row.should_abstain
            and not row.decision_correct
            or not row.should_abstain
            and (
                row.first_relevant_rank is None
                or (row.relevance or 0.0) < 1.0
                or row.citation_correct is not True
                or row.faithful is not True
            )
        )
    ]
    fragile = sorted(
        (
            row
            for row in summary.rows
            if (
                not row.should_abstain
                and row.top_score is not None
                and row.status == "answered"
            )
        ),
        key=lambda row: row.top_score or 0.0,
    )[:2]
    supported_scored = [
        row
        for row in summary.rows
        if not row.should_abstain and row.top_score is not None
    ]
    unsupported_scored = [
        row for row in summary.rows if row.should_abstain and row.top_score is not None
    ]
    lowest_supported = min(
        supported_scored, key=lambda row: row.top_score or 0.0, default=None
    )
    highest_unsupported = max(
        unsupported_scored, key=lambda row: row.top_score or 0.0, default=None
    )
    threshold_note = "Both supported and unsupported scored cases are needed to assess this threshold."
    if lowest_supported is not None and highest_unsupported is not None:
        low = lowest_supported.top_score
        high = highest_unsupported.top_score
        separates = (
            high is not None and low is not None and high < settings.min_score <= low
        )
        threshold_note = (
            f"The weakest supported top hit is `{lowest_supported.id}` at `{low:.3f}`, "
            f"while the strongest unsupported top hit is `{highest_unsupported.id}` "
            f"at `{high:.3f}`. The current threshold "
            f"{'separates' if separates else 'does not separate'} those observed scores."
        )
    lines = [
        "# Reference Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- Question set: `{settings.evaluation_file.name}`",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- Embedding revision: `{settings.embedding_revision or 'not pinned (smoke/custom)'}`",
        "- Vector store: `Qdrant` local mode with cosine similarity",
        f"- Retrieval: `top_k={settings.top_k}`, `min_score={settings.min_score:.2f}`",
        "- Ranking metrics use safety-filtered chunks before the evidence threshold.",
        f"- Generation: `{pipeline.generator.mode}`",
        "",
        "## Metric Rubric",
        "",
        "- `recall@k`: mean fraction of all labeled relevant documents retrieved per supported question.",
        "- `HitRate@k`: fraction of supported questions with at least one relevant retrieved document.",
        "- `MRR@k`: reciprocal rank of the first relevant chunk, averaged over supported questions; repeated sources keep their chunk positions.",
        "- `answer relevance`: literal required-term coverage on supported questions, not a semantic relevance judge; a false abstention receives 0. Unsupported questions are scored by decision accuracy instead.",
        "- `citation correctness`: supported questions are answered with citations to expected sources in accepted evidence; a false abstention is incorrect. Unsupported questions are scored by decision accuracy instead.",
        "- `faithfulness`: literal support in cited evidence for answered cases only, not semantic entailment. Live paraphrases need manual review and remain unrated; abstentions are excluded.",
        "- `decision accuracy`: answers supported questions and abstains on unsupported questions, averaged over all questions. Only `answered` and `abstained` count as decisions; `blocked_input` and `invalid_citations` are scored as incorrect for every question.",
        "- Ranking metrics, answer relevance, and citation correctness are not rated when no supported cases exist. Faithfulness is not aggregated when there are no answers or any answer needs manual review.",
        "",
        "## Results",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Retrieval recall@{settings.top_k} | {_format_metric(summary.retrieval_recall_at_k)} |",
        f"| Retrieval HitRate@{settings.top_k} | {_format_metric(summary.retrieval_hit_rate_at_k)} |",
        f"| Mean reciprocal rank@{settings.top_k} | {_format_metric(summary.mean_reciprocal_rank)} |",
        f"| Answer relevance | {_format_metric(summary.answer_relevance)} |",
        f"| Citation correctness | {_format_metric(summary.citation_correctness)} |",
        f"| Faithfulness (literal support only) | {_format_metric(summary.faithfulness)} |",
        f"| Decision accuracy | {summary.decision_accuracy:.3f} |",
        "",
        "## Per-Question Evidence",
        "",
        "| ID | Expected source | Retrieved chunk source order | Status | Relevance | Citations correct | Literal support | Safety event |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in summary.rows:
        support_label = (
            "n/a"
            if row.status != "answered"
            else "review"
            if row.faithful is None
            else "yes"
            if row.faithful
            else "no"
        )
        citation_label = (
            "n/a"
            if row.citation_correct is None
            else "yes"
            if row.citation_correct
            else "no"
        )
        status_label = row.status
        if row.abstention_reason:
            status_label += f" ({row.abstention_reason})"
        lines.append(
            f"| {row.id} | {_format_sources(row.expected_sources)} | "
            f"{_format_sources(row.retrieved_sources)} | {status_label} | "
            f"{_format_relevance(row.relevance)} | {citation_label} | "
            f"{support_label} | "
            f"{_format_sources(row.safety_events)} |"
        )
    lines.extend(["", "## Failure And Fragility Analysis", ""])
    if misses:
        for row in misses:
            lines.append(
                f"- `{row.id}` requires inspection: status `{row.status}`, "
                f"retrieved `{_format_sources(row.retrieved_sources)}`, "
                f"relevance `{_format_relevance(row.relevance)}`, faithful `{row.faithful}`."
            )
    else:
        lines.append(
            "- No labeled case fails the deterministic rubric. This is a small, "
            "curated benchmark and does not prove production reliability."
        )
    for row in fragile:
        lines.append(
            f"- Fragile success candidate `{row.id}` has top score "
            f"`{row.top_score:.3f}`; paraphrase expansion should test whether "
            "the selected threshold remains defensible."
        )
    lines.extend(
        [
            "",
            "## Threshold Decision",
            "",
            (
                f"The reference pipeline uses `min_score={settings.min_score:.2f}`. "
                f"{threshold_note} These are diagnostics for this question set, "
                "not evidence used to select settings. Calibrate the threshold "
                "together with the generator's abstention rule on `calibration.json`, "
                "then freeze both before the final `questions.json` evaluation. "
                "See `solution/README.md` for the default reference calibration rationale."
            ),
            "",
            "## Prioritized Recommendations",
            "",
            "1. Expand the benchmark with paraphrases and multi-source questions before changing the threshold.",
            "2. Compare the embedding model and chunking configuration using the same labeled cases.",
            "3. Run a separately reviewed live-generation evaluation and inspect citation failures before operational use.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(pipeline: RAGPipeline, output_path: Path) -> EvaluationSummary:
    """Execute the benchmark and write both Markdown and machine-readable evidence."""

    cases = load_cases(pipeline.settings.evaluation_file)
    summary = evaluate(pipeline, cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(pipeline, summary), encoding="utf-8")
    json_output = output_path.with_suffix(".json")
    json_output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary
