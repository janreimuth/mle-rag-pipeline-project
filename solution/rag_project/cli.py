"""Command-line entry points for building, querying, and evaluating the solution."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import click

from .config import Settings
from .embeddings import HashEmbedder
from .evaluation import write_report
from .pipeline import RAGPipeline


def build_pipeline(hash_smoke: bool, min_score: float | None) -> RAGPipeline:
    """Build the default semantic pipeline or a clearly labeled smoke variant."""

    settings = Settings.from_env()
    if min_score is not None:
        settings = replace(settings, min_score=min_score)
    if hash_smoke:
        settings = replace(
            settings, embedding_model="hash-smoke-test-only", embedding_revision=None
        )
        return RAGPipeline(settings, embedder=HashEmbedder())
    return RAGPipeline(settings)


@click.group()
@click.option(
    "--hash-smoke",
    is_flag=True,
    help="Use a deterministic lexical encoder for quick tests, not final results.",
)
@click.option("--min-score", type=float, help="Override the evidence threshold.")
@click.pass_context
def main(context: click.Context, hash_smoke: bool, min_score: float | None) -> None:
    """Run the Northstar RAG reference implementation."""

    try:
        context.obj = build_pipeline(hash_smoke, min_score)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    context.call_on_close(context.obj.close)


@main.command()
@click.pass_obj
def index(pipeline: RAGPipeline) -> None:
    """Print the chunk-count summary and a sample chunk with its metadata."""

    click.echo(
        f"Indexed {len(pipeline.chunks)} chunks with "
        f"{pipeline.settings.embedding_model} in Qdrant local mode."
    )
    for document_id, count in Counter(
        chunk.document_id for chunk in pipeline.chunks
    ).items():
        click.echo(f"  {document_id}: {count} chunks")
    click.echo("Sample chunk:")
    click.echo(json.dumps(pipeline.chunks[0].model_dump(mode="json"), indent=2))


@main.command()
@click.argument("question")
@click.pass_obj
def ask(pipeline: RAGPipeline, question: str) -> None:
    """Answer one question and expose citations and safety outcomes."""

    result = pipeline.answer(question)
    click.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@main.command()
@click.option(
    "--cases",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Question JSON file; defaults to the final evaluation set.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Report path; custom question sets default to storage/<name>_evaluation.md.",
)
@click.pass_obj
def evaluate(pipeline: RAGPipeline, cases: Path | None, output: Path | None) -> None:
    """Run the complete labeled benchmark and write an evaluation report."""

    if cases is not None:
        pipeline.settings = replace(pipeline.settings, evaluation_file=cases)
    # A calibration run must not overwrite the checked-in final reference report.
    output = output or (
        Path("storage") / f"{cases.stem}_evaluation.md"
        if cases is not None
        else Path("solution/reports/reference_evaluation.md")
    )
    summary = write_report(pipeline, output)
    click.echo(f"Wrote evaluation report to {output}")
    recall = (
        f"{summary.retrieval_recall_at_k:.3f}"
        if summary.retrieval_recall_at_k is not None
        else "not-rated"
    )
    faithfulness = (
        f"{summary.faithfulness:.3f}"
        if summary.faithfulness is not None
        else "not-rated"
    )
    click.echo(
        "recall@k="
        f"{recall} "
        f"faithfulness={faithfulness} "
        f"decision_accuracy={summary.decision_accuracy:.3f}"
    )
