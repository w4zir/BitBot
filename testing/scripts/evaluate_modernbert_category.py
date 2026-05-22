from __future__ import annotations

"""Evaluate ModernBERT category classification locally via Hugging Face.

Supports two labeled JSONL formats:

- **category_intent**: ``{"text", "category", "intent"}`` (e.g. ``data/raw/simulated/category_intent.jsonl``)
- **training**: ``{"text", "label": <int>}`` with ``label2id.json`` (e.g. ``training/data/bitext_category/test.jsonl``)

Loads a fine-tuned checkpoint from ``--model-dir`` or ``MODERNBERT_MODEL_DIR`` and runs batched
``Trainer.predict`` (same stack as ``training/experiments/src/train_multiclass_modernbert.py``).

Writes metrics and per-sample records to ``testing/results/modernbert_category/run_YYYYMMDD_HHMMSS.jsonl``.

Prerequisites::

    pip install -r training/requirements-train.txt

Usage (from repository root)::

    python testing/scripts/evaluate_modernbert_category.py \\
        training/data/bitext_category/test.jsonl \\
        -o testing/results/modernbert_category

    python testing/scripts/evaluate_modernbert_category.py \\
        --model-dir training/models/bitext_multiclass_finetuned_20260415T120000Z/winner \\
        -i training/data/bitext_category/test.jsonl \\
        --output-dir testing/results/my_run

    python testing/scripts/evaluate_modernbert_category.py \\
        --dataset-dir training/data/bitext_category \\
        -o testing/results/modernbert_category

    python testing/scripts/evaluate_modernbert_category.py --help
"""

import argparse
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.repo_dotenv import load_repo_dotenv

from testing.scripts.evaluate_category_n_intent import (
    DatasetRow,
    _compute_multiclass_metrics,
    _load_id2label,
    _load_rows,
    _normalize_category,
    _positive_int,
    _print_progress,
    _print_sample_debug,
    _print_terminal_summary,
    _resolve_label2id_path,
    _write_results,
)

DEFAULT_DATASET_DIR = REPO_ROOT / "training" / "data" / "bitext_category"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing" / "results" / "modernbert_category"


def _has_local_checkpoint_files(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    if (model_dir / "model.safetensors").is_file():
        return True
    if (model_dir / "pytorch_model.bin").is_file():
        return True
    if list(model_dir.glob("model-*.safetensors")):
        return True
    return False


def _trainer_tokenizer_kwargs(tokenizer: Any) -> dict[str, Any]:
    from transformers import Trainer

    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _resolve_model_dir(*, cli_value: str) -> Path:
    raw = cli_value.strip() or os.getenv("MODERNBERT_MODEL_DIR", "").strip()
    if not raw:
        raise ValueError(
            "Model directory is required: pass --model-dir or set MODERNBERT_MODEL_DIR in .env"
        )
    model_dir = Path(raw).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    if not _has_local_checkpoint_files(model_dir):
        raise ValueError(
            f"Model directory is missing Hugging Face weights (config.json + weights): {model_dir}"
        )
    return model_dir


def _resolve_label2id_path_for_eval(
    *,
    input_file: Path,
    dataset_dir: Path,
    explicit: str,
) -> Path:
    if explicit.strip():
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"label2id file does not exist: {path}")
        return path

    sibling = _resolve_label2id_path(input_file=input_file, explicit="")
    if sibling is not None:
        return sibling

    dataset_candidate = (dataset_dir / "label2id.json").resolve()
    if dataset_candidate.is_file():
        return dataset_candidate

    raise ValueError(
        "label2id.json is required to map model outputs to category labels; "
        "pass --label2id-file or place label2id.json next to the input file or under --dataset-dir"
    )


def _resolve_input_path(*, cli_value: str) -> Path | None:
    if not cli_value.strip():
        return None
    path = Path(cli_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Input file must be .jsonl: {path}")
    return path


def _resolve_dataset_dir(*, cli_value: str | None, input_file: Path | None) -> Path:
    if cli_value and cli_value.strip():
        return Path(cli_value).expanduser().resolve()
    if input_file is not None:
        return input_file.parent.resolve()
    return DEFAULT_DATASET_DIR.resolve()


def _resolve_input_file(*, cli_value: str, dataset_dir: Path) -> Path:
    explicit = _resolve_input_path(cli_value=cli_value)
    if explicit is not None:
        return explicit

    default_test = (dataset_dir / "test.jsonl").resolve()
    if default_test.is_file():
        return default_test

    raise ValueError(
        "No input file specified and default test.jsonl was not found; "
        f"pass INPUT_FILE (positional), --input-file/-i, or add {default_test}"
    )


def _category_to_label_id(label2id: dict[str, int]) -> dict[str, int]:
    return {_normalize_category(name): label_id for name, label_id in label2id.items()}


def _row_gold_label_id(row: DatasetRow, category_to_id: dict[str, int]) -> int:
    if row.gold_label_id is not None:
        return row.gold_label_id
    label_id = category_to_id.get(row.gold_category)
    if label_id is None:
        raise ValueError(
            f"Line {row.line_number}: gold category '{row.gold_category}' is not in label2id mapping"
        )
    return label_id


def _run_local_evaluation(
    *,
    rows: list[DatasetRow],
    model: Any,
    tokenizer: Any,
    id2label: dict[int, str],
    label2id: dict[str, int],
    batch_size: int,
    max_length: int,
    verbose_debug: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    category_to_id = _category_to_label_id(label2id)
    hf_rows = [
        {"text": row.text, "label": _row_gold_label_id(row, category_to_id)} for row in rows
    ]
    eval_ds = Dataset.from_list(hf_rows)

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    eval_tok = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )

    with tempfile.TemporaryDirectory(prefix="modernbert_category_eval_") as tmp_out:
        training_args = TrainingArguments(
            output_dir=tmp_out,
            per_device_eval_batch_size=batch_size,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            **_trainer_tokenizer_kwargs(tokenizer),
        )
        predict_out = trainer.predict(eval_tok)

    logits = np.asarray(predict_out.predictions, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, len(label2id))
    probs = _softmax(logits)
    pred_ids = np.argmax(logits, axis=-1)
    confidences = probs[np.arange(len(pred_ids)), pred_ids]

    return _build_evaluation_records(
        rows=rows,
        pred_ids=pred_ids,
        confidences=confidences,
        id2label=id2label,
        verbose_debug=verbose_debug,
    )


def _build_evaluation_records(
    *,
    rows: list[DatasetRow],
    pred_ids: np.ndarray,
    confidences: np.ndarray,
    id2label: dict[int, str],
    verbose_debug: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    category_gold: list[str] = []
    category_pred: list[str] = []
    category_correct_total = 0
    total = len(rows)

    for processed, (row, pred_id, confidence) in enumerate(
        zip(rows, pred_ids, confidences, strict=True), start=1
    ):
        pred_id_int = int(pred_id)
        if pred_id_int not in id2label:
            raise ValueError(f"Model predicted invalid label id {pred_id_int}")
        predicted_category = _normalize_category(id2label[pred_id_int])
        category_confidence = float(confidence)
        is_category_correct = predicted_category == row.gold_category
        if is_category_correct:
            category_correct_total += 1

        category_gold.append(row.gold_category)
        category_pred.append(predicted_category)

        record: dict[str, Any] = {
            "type": "example",
            "index": row.index,
            "line_number": row.line_number,
            "text": row.text,
            "gold": {
                "category": row.gold_category,
                "intent": row.gold_intent,
            },
            "modernbert": {
                "category": predicted_category,
                "confidence": category_confidence,
                "is_category_correct": is_category_correct,
                "error": "",
            },
        }
        if row.gold_label_id is not None:
            record["gold_label_id"] = row.gold_label_id
        records.append(record)
        _print_progress(processed=processed, total=total)
        if verbose_debug:
            _print_sample_debug(
                record=record,
                processed=processed,
                total=total,
                category_only=True,
            )

    if total > 0:
        print()

    summary: dict[str, Any] = {
        "counts": {
            "total_examples": len(rows),
            "category_correct_examples": category_correct_total,
        },
        "metrics": {
            "category": _compute_multiclass_metrics(category_gold, category_pred),
        },
    }
    return records, summary


def _load_label2id_dict(path: Path) -> dict[str, int]:
    id2label = _load_id2label(path)
    return {name: label_id for label_id, name in id2label.items()}


def _resolve_effective_input_path(*, positional: str, optional_flag: str) -> str:
    flag_value = str(optional_flag or "").strip()
    if flag_value:
        return flag_value
    return str(positional or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ModernBERT category classification locally (Hugging Face) "
            "against a labeled JSONL dataset file."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="",
        metavar="INPUT_FILE",
        help=(
            "JSONL file to evaluate (e.g. training/data/bitext_category/test.jsonl). "
            "When omitted, uses <dataset-dir>/test.jsonl."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help=(
            "HF checkpoint directory (e.g. training/models/.../winner). "
            "Defaults to MODERNBERT_MODEL_DIR from .env when omitted."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help=(
            "Folder for label2id.json fallback and default test.jsonl. "
            f"Defaults to the input file's directory, or {DEFAULT_DATASET_DIR}."
        ),
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default="",
        dest="input_file_flag",
        help="JSONL file to evaluate (overrides positional INPUT_FILE when set).",
    )
    parser.add_argument(
        "--label2id-file",
        default="",
        help="Path to label2id.json (default: sibling of input, then <dataset-dir>/label2id.json).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for run_YYYYMMDD_HHMMSS.jsonl results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device eval batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid/empty rows instead of failing fast.",
    )
    parser.add_argument(
        "--verbose-debug",
        action="store_true",
        help="Print detailed per-sample evaluation output during the run.",
    )
    parser.add_argument(
        "--max-limit",
        type=_positive_int,
        default=None,
        help="Optional positive integer to limit number of dataset rows evaluated.",
    )
    return parser


def main() -> int:
    load_repo_dotenv(REPO_ROOT)
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    effective_input_path = _resolve_effective_input_path(
        positional=str(args.input_file or ""),
        optional_flag=str(args.input_file_flag or ""),
    )

    try:
        model_dir = _resolve_model_dir(cli_value=str(args.model_dir or ""))
        input_file_explicit = _resolve_input_path(cli_value=effective_input_path)
        dataset_dir = _resolve_dataset_dir(
            cli_value=str(args.dataset_dir or ""),
            input_file=input_file_explicit,
        )
        input_file = _resolve_input_file(
            cli_value=effective_input_path,
            dataset_dir=dataset_dir,
        )
        label2id_path = _resolve_label2id_path_for_eval(
            input_file=input_file,
            dataset_dir=dataset_dir,
            explicit=str(args.label2id_file or ""),
        )
        id2label = _load_id2label(label2id_path)
        label2id = _load_label2id_dict(label2id_path)

        rows, skipped_invalid_rows, input_format, _ = _load_rows(
            input_file=input_file,
            skip_invalid=bool(args.skip_invalid),
            label2id_path=label2id_path,
            category_only=True,
        )
        pre_limit_total_rows = len(rows)
        max_limit = args.max_limit
        if max_limit is not None:
            rows = rows[:max_limit]

        total_to_evaluate = len(rows)
        print(
            f"Total samples to evaluate: {total_to_evaluate} "
            f"(loaded={pre_limit_total_rows}, skipped_invalid={skipped_invalid_rows}, "
            f"input_format={input_format}, model_dir={model_dir})"
        )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        num_labels = len(label2id)
        if int(getattr(model.config, "num_labels", num_labels)) != num_labels:
            raise ValueError(
                f"Model num_labels={model.config.num_labels} does not match "
                f"label2id ({num_labels} labels in {label2id_path})"
            )

        records, raw_summary = _run_local_evaluation(
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            batch_size=max(1, int(args.batch_size)),
            max_length=max(1, int(args.max_length)),
            verbose_debug=bool(args.verbose_debug),
        )
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        parser.error(str(exc))
        return 2

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    summary_payload = {**raw_summary}
    metadata_payload = {
        "timestamp_utc": timestamp_utc,
        "config": {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
            "input_format": input_format,
            "label2id_file": str(label2id_path),
            "model_dir": str(model_dir),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
            "skip_invalid": bool(args.skip_invalid),
            "verbose_debug": bool(args.verbose_debug),
            "max_limit": args.max_limit,
            "input_rows_before_limit": pre_limit_total_rows,
            "skipped_invalid_rows": skipped_invalid_rows,
            "num_labels": num_labels,
        },
    }
    output_file = _write_results(
        output_dir=output_dir,
        records=records,
        summary_payload=summary_payload,
        metadata_payload=metadata_payload,
    )

    _print_terminal_summary(
        summary_payload=summary_payload,
        metadata_payload=metadata_payload,
        output_file=output_file,
        category_only=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
