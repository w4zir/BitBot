from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from testing.scripts import evaluate_modernbert_category as script
from testing.scripts.evaluate_category_n_intent import DatasetRow


def _write_fake_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def test_resolve_model_dir_from_cli(tmp_path, monkeypatch) -> None:
    model_dir = _write_fake_model_dir(tmp_path)
    monkeypatch.delenv("MODERNBERT_MODEL_DIR", raising=False)
    resolved = script._resolve_model_dir(cli_value=str(model_dir))
    assert resolved == model_dir.resolve()


def test_resolve_model_dir_from_env(tmp_path, monkeypatch) -> None:
    model_dir = _write_fake_model_dir(tmp_path)
    monkeypatch.setenv("MODERNBERT_MODEL_DIR", str(model_dir))
    resolved = script._resolve_model_dir(cli_value="")
    assert resolved == model_dir.resolve()


def test_resolve_model_dir_cli_overrides_env(tmp_path, monkeypatch) -> None:
    cli_dir = _write_fake_model_dir(tmp_path / "cli")
    env_dir = _write_fake_model_dir(tmp_path / "env")
    monkeypatch.setenv("MODERNBERT_MODEL_DIR", str(env_dir))
    resolved = script._resolve_model_dir(cli_value=str(cli_dir))
    assert resolved == cli_dir.resolve()


def test_resolve_model_dir_missing_raises(monkeypatch) -> None:
    monkeypatch.delenv("MODERNBERT_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="Model directory is required"):
        script._resolve_model_dir(cli_value="")


def test_resolve_label2id_path_for_eval_prefers_dataset_dir(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    input_file = tmp_path / "other" / "data.jsonl"
    input_file.parent.mkdir(parents=True)
    input_file.write_text("{}\n", encoding="utf-8")
    label2id_path = dataset_dir / "label2id.json"
    label2id_path.write_text(json.dumps({"REFUND": 0}), encoding="utf-8")

    resolved = script._resolve_label2id_path_for_eval(
        input_file=input_file,
        dataset_dir=dataset_dir,
        explicit="",
    )
    assert resolved == label2id_path.resolve()


def test_resolve_input_file_uses_dataset_test_jsonl(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    test_file = dataset_dir / "test.jsonl"
    test_file.write_text("{}\n", encoding="utf-8")

    resolved = script._resolve_input_file(cli_value="", dataset_dir=dataset_dir)
    assert resolved == test_file.resolve()


def test_resolve_input_path_single_file(tmp_path) -> None:
    data_file = tmp_path / "test.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")

    resolved = script._resolve_input_path(cli_value=str(data_file))
    assert resolved == data_file.resolve()


def test_resolve_input_path_rejects_non_jsonl(tmp_path) -> None:
    data_file = tmp_path / "test.txt"
    data_file.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be .jsonl"):
        script._resolve_input_path(cli_value=str(data_file))


def test_resolve_dataset_dir_from_input_parent(tmp_path) -> None:
    data_file = tmp_path / "bitext_category" / "test.jsonl"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("{}\n", encoding="utf-8")

    resolved = script._resolve_dataset_dir(cli_value="", input_file=data_file)
    assert resolved == data_file.parent.resolve()


def test_resolve_effective_input_path_prefers_flag() -> None:
    assert script._resolve_effective_input_path(
        positional="/tmp/a.jsonl",
        optional_flag="/tmp/b.jsonl",
    ) == "/tmp/b.jsonl"
    assert script._resolve_effective_input_path(
        positional="/tmp/a.jsonl",
        optional_flag="",
    ) == "/tmp/a.jsonl"


def test_build_evaluation_records_summary() -> None:
    rows = [
        DatasetRow(
            index=0,
            line_number=1,
            text="refund please",
            gold_category="refund",
            gold_intent="",
            gold_label_id=1,
        ),
        DatasetRow(
            index=1,
            line_number=2,
            text="payment issue",
            gold_category="payment",
            gold_intent="",
            gold_label_id=0,
        ),
    ]
    id2label = {0: "PAYMENT", 1: "REFUND"}
    pred_ids = np.array([1, 0], dtype=np.int64)
    confidences = np.array([0.9, 0.8], dtype=np.float64)

    records, summary = script._build_evaluation_records(
        rows=rows,
        pred_ids=pred_ids,
        confidences=confidences,
        id2label=id2label,
    )

    assert len(records) == 2
    assert records[0]["modernbert"]["category"] == "refund"
    assert records[0]["modernbert"]["is_category_correct"] is True
    assert records[1]["modernbert"]["category"] == "payment"
    assert records[1]["modernbert"]["is_category_correct"] is True
    assert summary["counts"]["category_correct_examples"] == 2
    assert summary["metrics"]["category"]["accuracy"] == 1.0


def test_category_to_label_id_normalizes_keys() -> None:
    mapping = script._category_to_label_id({"REFUND": 1, "no_issue": 11})
    assert mapping["refund"] == 1
    assert mapping["no_issue"] == 11


def test_build_parser_accepts_positional_input_and_output_dir() -> None:
    parser = script._build_parser()
    args = parser.parse_args(
        [
            "/tmp/test.jsonl",
            "-o",
            "/tmp/results",
        ]
    )
    assert args.input_file == "/tmp/test.jsonl"
    assert args.output_dir == "/tmp/results"


def test_build_parser_accepts_model_and_input_flags() -> None:
    parser = script._build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "/tmp/model",
            "-i",
            "/tmp/test.jsonl",
            "--batch-size",
            "4",
        ]
    )
    assert args.model_dir == "/tmp/model"
    assert args.input_file_flag == "/tmp/test.jsonl"
    assert args.batch_size == 4
