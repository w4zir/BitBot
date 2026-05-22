from __future__ import annotations

import json

import pytest

from testing.scripts import evaluate_category_n_intent as script


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

    monkeypatch.setattr(
        "backend.agent.issue_graph._classify_intent_node",
        _fake_intent_node,
    )
    monkeypatch.setattr(
        "backend.db.intents_repo.get_intents_for_category",
        lambda _category: ["get_refund"],
    )
    records, summary = script._run_evaluation(
        rows=rows,
        classifier=classifier,  # type: ignore[arg-type]
        category_only=False,
    )

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

    monkeypatch.setattr(
        "backend.db.intents_repo.get_intents_for_category",
        lambda _category: [],
    )

    records, summary = script._run_evaluation(
        rows=rows,
        classifier=classifier,  # type: ignore[arg-type]
        category_only=False,
    )

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

    monkeypatch.setattr(
        "backend.db.intents_repo.get_intents_for_category",
        lambda _category: ["get_refund"],
    )
    monkeypatch.setattr(
        "backend.agent.issue_graph._classify_intent_node",
        lambda _state: {
            "intent": "refund_general",
            "assistant_metadata": {"intent_classifier": "llm"},
        },
    )

    records, summary = script._run_evaluation(
        rows=rows,
        classifier=classifier,  # type: ignore[arg-type]
        category_only=False,
    )

    assert records[0]["intent_classifier"]["evaluated"] is True
    assert records[0]["intent_classifier"]["intent"] == ""
    assert "outside allowed DB intents" in records[0]["intent_classifier"]["error"]
    assert records[0]["intent_classifier"]["is_intent_correct"] is False
    assert summary["counts"]["intent_evaluated_examples"] == 1
    assert summary["metrics"]["intent_on_category_correct"]["count"] == 1


def test_run_evaluation_category_only_skips_intent() -> None:
    rows = [
        script.DatasetRow(
            index=0,
            line_number=1,
            text="refund question",
            gold_category="refund",
            gold_intent="get_refund",
        ),
    ]
    classifier = _FakeClassifier(outputs=[("refund", 0.91)])

    records, summary = script._run_evaluation(
        rows=rows,
        classifier=classifier,  # type: ignore[arg-type]
        category_only=True,
    )

    assert "intent_classifier" not in records[0]
    assert "intent_on_category_correct" not in summary["metrics"]
    assert summary["counts"]["category_correct_examples"] == 1
    assert "intent_evaluated_examples" not in summary["counts"]


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

    rows, skipped, input_format, label2id_file = script._load_rows(
        data_file,
        skip_invalid=True,
        label2id_path=None,
        category_only=True,
    )
    assert len(rows) == 1
    assert skipped == 2
    assert input_format == "category_intent"
    assert label2id_file is None


def test_load_rows_category_only_allows_missing_intent(tmp_path) -> None:
    data_file = tmp_path / "dataset.jsonl"
    with data_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"text": "hello", "category": "refund"}) + "\n")

    rows, skipped, input_format, _ = script._load_rows(
        data_file,
        skip_invalid=False,
        label2id_path=None,
        category_only=True,
    )
    assert len(rows) == 1
    assert rows[0].gold_intent == ""
    assert skipped == 0
    assert input_format == "category_intent"


def test_load_rows_training_format_with_label2id(tmp_path) -> None:
    label2id_path = tmp_path / "label2id.json"
    label2id_path.write_text(
        json.dumps({"CONTACT": 0, "REFUND": 1}, indent=2) + "\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "test.jsonl"
    with data_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"text": "refund please", "label": 1}) + "\n")

    rows, skipped, input_format, resolved = script._load_rows(
        data_file,
        skip_invalid=False,
        label2id_path=label2id_path,
        category_only=True,
    )
    assert len(rows) == 1
    assert rows[0].gold_category == "refund"
    assert rows[0].gold_label_id == 1
    assert skipped == 0
    assert input_format == "training"
    assert resolved == str(label2id_path.resolve())


def test_load_rows_training_only_rejects_intent_mode(tmp_path) -> None:
    label2id_path = tmp_path / "label2id.json"
    label2id_path.write_text(json.dumps({"REFUND": 0}) + "\n", encoding="utf-8")
    data_file = tmp_path / "test.jsonl"
    data_file.write_text(
        json.dumps({"text": "refund please", "label": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Intent evaluation requires"):
        script._load_rows(
            data_file,
            skip_invalid=False,
            label2id_path=label2id_path,
            category_only=False,
        )


def test_build_parser_validates_max_limit() -> None:
    parser = script._build_parser()
    args = parser.parse_args(["--max-limit", "2"])
    assert args.max_limit == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-limit", "0"])


def test_build_parser_accepts_category_only_and_verbose_debug() -> None:
    parser = script._build_parser()
    args = parser.parse_args(["--category-only", "--verbose-debug"])
    assert args.category_only is True
    assert args.verbose_debug is True


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
