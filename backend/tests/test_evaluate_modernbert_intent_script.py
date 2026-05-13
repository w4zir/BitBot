from __future__ import annotations

import json

import pytest

from testing.scripts import evaluate_modernbert_intent as script


class _FakeClassifier:
    def __init__(self, outputs: list[tuple[str, float]]) -> None:
        self._outputs = outputs
        self._idx = 0

    def classify(self, _: str):
        category, confidence = self._outputs[self._idx]
        self._idx += 1
        return type("Result", (), {"category": category, "confidence": confidence})()


def test_compute_multiclass_metrics_basic() -> None:
    metrics = script._compute_multiclass_metrics(
        gold=["refund", "refund", "payment"],
        pred=["refund", "payment", "payment"],
    )
    assert metrics["count"] == 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["micro"]["f1"] == 2 / 3
    assert len(metrics["per_label"]) == 2


def test_run_evaluation_uses_intent_only_when_category_correct(monkeypatch) -> None:
    rows = [
        script.DatasetRow(
            index=0,
            line_number=1,
            text="refund question",
            gold_category="refund",
            gold_intent="get_refund",
        ),
        script.DatasetRow(
            index=1,
            line_number=2,
            text="payment issue",
            gold_category="payment",
            gold_intent="payment_issue",
        ),
    ]
    classifier = _FakeClassifier(outputs=[("refund", 0.91), ("refund", 0.52)])

    def _fake_intent_node(state: dict):
        _ = state
        return {"intent": "get_refund", "assistant_metadata": {"intent_classifier": "llm"}}

    monkeypatch.setattr(script, "_classify_intent_node", _fake_intent_node)
    monkeypatch.setattr(script, "get_intents_for_category", lambda _category: ["get_refund"])
    records, summary = script._run_evaluation(rows=rows, classifier=classifier)  # type: ignore[arg-type]

    assert len(records) == 2
    assert records[0]["intent_classifier"]["evaluated"] is True
    assert records[1]["intent_classifier"]["evaluated"] is False
    assert summary["counts"]["category_correct_examples"] == 1
    assert summary["counts"]["intent_evaluated_examples"] == 1
    assert summary["metrics"]["intent_on_category_correct"]["count"] == 1


def test_run_evaluation_skips_intent_when_allowed_intents_missing(monkeypatch) -> None:
    rows = [
        script.DatasetRow(
            index=0,
            line_number=1,
            text="refund question",
            gold_category="refund",
            gold_intent="get_refund",
        ),
    ]
    classifier = _FakeClassifier(outputs=[("refund", 0.93)])

    monkeypatch.setattr(script, "get_intents_for_category", lambda _category: [])

    records, summary = script._run_evaluation(rows=rows, classifier=classifier)  # type: ignore[arg-type]

    assert records[0]["intent_classifier"]["evaluated"] is False
    assert "No DB allowed intents configured" in records[0]["intent_classifier"]["error"]
    assert records[0]["intent_classifier"]["metadata"]["allowed_intents"] == []
    assert summary["counts"]["intent_evaluated_examples"] == 0
    assert summary["metrics"]["intent_on_category_correct"]["count"] == 0


def test_run_evaluation_rejects_intent_outside_allowed_list(monkeypatch) -> None:
    rows = [
        script.DatasetRow(
            index=0,
            line_number=1,
            text="refund question",
            gold_category="refund",
            gold_intent="get_refund",
        ),
    ]
    classifier = _FakeClassifier(outputs=[("refund", 0.97)])

    monkeypatch.setattr(script, "get_intents_for_category", lambda _category: ["get_refund"])
    monkeypatch.setattr(
        script,
        "_classify_intent_node",
        lambda _state: {"intent": "refund_general", "assistant_metadata": {"intent_classifier": "llm"}},
    )

    records, summary = script._run_evaluation(rows=rows, classifier=classifier)  # type: ignore[arg-type]

    assert records[0]["intent_classifier"]["evaluated"] is True
    assert records[0]["intent_classifier"]["intent"] == ""
    assert "outside allowed DB intents" in records[0]["intent_classifier"]["error"]
    assert records[0]["intent_classifier"]["is_intent_correct"] is False
    assert summary["counts"]["intent_evaluated_examples"] == 1
    assert summary["metrics"]["intent_on_category_correct"]["count"] == 1


def test_write_results_orders_summary_metadata_examples(tmp_path) -> None:
    records = [{"type": "example", "index": 0, "line_number": 1, "text": "hello"}]
    summary_payload = {"counts": {"total_examples": 1}, "metrics": {"category": {"count": 1}}}
    metadata_payload = {"timestamp_utc": "2026-01-01T00:00:00+00:00", "config": {"intent_model": "x"}}

    output_file = script._write_results(
        output_dir=tmp_path,
        records=records,
        summary_payload=summary_payload,
        metadata_payload=metadata_payload,
    )

    with output_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert lines[0]["type"] == "summary"
    assert lines[1]["type"] == "metadata"
    assert lines[2]["type"] == "example"


def test_load_rows_skip_invalid(tmp_path) -> None:
    data_file = tmp_path / "dataset.jsonl"
    with data_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"text": "hello", "category": "refund", "intent": "get_refund"}) + "\n")
        f.write("\n")
        f.write(json.dumps({"text": "", "category": "refund", "intent": "get_refund"}) + "\n")

    rows, skipped = script._load_rows(data_file, skip_invalid=True)
    assert len(rows) == 1
    assert skipped == 2


def test_build_parser_validates_max_limit() -> None:
    parser = script._build_parser()
    args = parser.parse_args(["--max-limit", "2"])
    assert args.max_limit == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-limit", "0"])


def test_build_parser_accepts_ollama_url() -> None:
    parser = script._build_parser()
    args = parser.parse_args(["--ollama-url", "http://127.0.0.1:11434"])
    assert args.ollama_url == "http://127.0.0.1:11434"


def test_warn_if_ollama_unreachable_emits_warning(monkeypatch, capsys) -> None:
    class _FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = (exc_type, exc, tb)
            return False

        def get(self, _url):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(script.httpx, "Client", _FailingClient)
    script._warn_if_ollama_unreachable(provider="ollama", model="llama3.2")
    captured = capsys.readouterr()
    assert "WARNING: Ollama appears unreachable" in captured.err


def test_warn_if_ollama_model_missing_emits_warning(monkeypatch, capsys) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "some_other_model:latest"}]}

    class _OkClient:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = (exc_type, exc, tb)
            return False

        def get(self, _url):
            return _FakeResponse()

    monkeypatch.setattr(script.httpx, "Client", _OkClient)
    script._warn_if_ollama_unreachable(provider="ollama", model="llama3.2")
    captured = capsys.readouterr()
    assert "WARNING: Configured Ollama model not listed" in captured.err


def test_warn_if_ollama_unreachable_skips_non_ollama_provider(capsys) -> None:
    script._warn_if_ollama_unreachable(provider="cerebras", model="llama3.2")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_if_ollama_unreachable_skips_vllm_provider(capsys) -> None:
    script._warn_if_ollama_unreachable(provider="vllm", model="dummy")
    captured = capsys.readouterr()
    assert captured.err == ""
