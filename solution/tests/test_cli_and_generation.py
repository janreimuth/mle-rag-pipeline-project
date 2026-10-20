"""CLI lifecycle and provider-contract tests without real API requests."""

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from openai import OpenAI

from solution.rag_project.cli import main
from solution.rag_project.config import Settings
from solution.rag_project.generation import FALLBACK_TEXT, ProviderGenerator
from solution.rag_project.pipeline import RAGPipeline


def test_cli_commands_close_persistent_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VECTOR_STORAGE_PATH", str(tmp_path / "qdrant"))
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(main, ["--hash-smoke", "index"])
        assert result.exit_code == 0, result.output
        assert "Indexed" in result.output
        assert "account-mfa: " in result.output
        assert '"document_id": "account-mfa"' in result.output
        assert '"position": 0' in result.output
    result = runner.invoke(
        main, ["--hash-smoke", "evaluate", "--output", str(tmp_path / "report.md")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.json").is_file()
    trace = json.loads((tmp_path / "report.json").read_text())
    assert all("abstention_reason" in row for row in trace["rows"])


@pytest.mark.parametrize("use_calibration", [False, True])
def test_cli_keeps_calibration_and_final_reports_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_calibration: bool
) -> None:
    root = Path(__file__).parents[2]
    reference = root / "solution/reports/reference_evaluation.md"
    original = reference.read_bytes()
    monkeypatch.chdir(tmp_path)
    arguments = ["--hash-smoke", "evaluate"]
    if use_calibration:
        arguments.extend(["--cases", str(root / "data/evaluation/calibration.json")])
        report = tmp_path / "storage/calibration_evaluation.md"
        filename, count = "calibration.json", 6
    else:
        report = tmp_path / "solution/reports/reference_evaluation.md"
        filename, count = "questions.json", 16
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code == 0, result.output
    assert f"Question set: `{filename}`" in report.read_text()
    assert len(json.loads(report.with_suffix(".json").read_text())["rows"]) == count
    assert reference.read_bytes() == original


def test_cli_rejects_missing_question_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["--hash-smoke", "evaluate", "--cases", str(tmp_path / "missing.json")]
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output


@pytest.mark.parametrize("provider_name", ["groq", "mistral"])
@pytest.mark.parametrize("content", ["Evidence. [billing-refunds]", None])
def test_provider_uses_the_configured_endpoint_and_bounded_requests(
    smoke_pipeline: RAGPipeline,
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    content: str | None,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider_name)
    monkeypatch.setenv(f"{provider_name.upper()}_API_KEY", "fake-review-key")
    provider = Settings.from_env().provider
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "review-only",
                "object": "chat.completion",
                "created": 0,
                "model": provider.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "solution.rag_project.generation.OpenAI",
            lambda **kwargs: OpenAI(**kwargs, http_client=client),
        )
        generator = ProviderGenerator(provider)
        hits, _ = smoke_pipeline.retrieve("invoice refund")
        assert generator.generate("Refund?", hits) == (content or FALLBACK_TEXT)
        assert generator.client.timeout == 30
        assert generator.client.max_retries == 1
        smoke_pipeline.generator = generator
        result = smoke_pipeline.answer(
            "What invoice number and payment references are needed for duplicate payments?"
        )
        assert result.status == ("answered" if content else "abstained")
        assert result.abstention_reason == (None if content else "generator_fallback")
    assert str(requests[0].url) == provider.base_url + "/chat/completions"
    body = json.loads(requests[0].content)
    assert body["model"] == provider.model
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "different service or request type" in body["messages"][0]["content"]


def test_cli_configuration_errors_are_actionable() -> None:
    result = CliRunner().invoke(main, ["--hash-smoke", "--min-score", "nan", "index"])
    assert result.exit_code != 0
    assert "MIN_SCORE" in result.output
    assert "Traceback" not in result.output
