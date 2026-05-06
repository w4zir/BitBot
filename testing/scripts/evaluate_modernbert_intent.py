from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent.issue_graph import _classify_intent_node
from backend.db.intents_repo import get_intents_for_category
from backend.rag.query_classifier import QueryClassifier


DEFAULT_INPUT_FILE = REPO_ROOT / "data" / "raw" / "simulated" / "category_intent.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing" / "results" / "modernbert_n_intent"


def _load_project_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class DatasetRow:
    index: int
    line_number: int
    text: str
    gold_category: str
    gold_intent: str


def _normalize_category(label: str) -> str:
    return str(label or "").strip().lower()


def _normalize_intent(label: str) -> str:
    return str(label or "").strip()


def _load_rows(input_file: Path, skip_invalid: bool) -> tuple[list[DatasetRow], int]:
    if not input_file.exists() or not input_file.is_file():
        raise ValueError(f"Input file does not exist or is not a file: {input_file}")
    if input_file.suffix.lower() != ".jsonl":
        raise ValueError(f"Input file must be .jsonl: {input_file}")

    rows: list[DatasetRow] = []
    skipped = 0
    with input_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(f"Line {line_number} is empty")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(f"Line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(f"Line {line_number} must be a JSON object")

            text = str(payload.get("text") or "").strip()
            gold_category = _normalize_category(str(payload.get("category") or ""))
            gold_intent = _normalize_intent(str(payload.get("intent") or ""))
            if not text or not gold_category or not gold_intent:
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(
                    f"Line {line_number} must include non-empty text/category/intent fields"
                )

            rows.append(
                DatasetRow(
                    index=len(rows),
                    line_number=line_number,
                    text=text,
                    gold_category=gold_category,
                    gold_intent=gold_intent,
                )
            )
    if not rows:
        raise ValueError("No valid rows found in dataset")
    return rows, skipped


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _warn_if_ollama_unreachable(*, provider: str, model: str) -> None:
    if provider != "ollama":
        return
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    tags_url = f"{base_url}/api/tags"
    timeout_seconds = 3.0
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(tags_url)
            response.raise_for_status()
            payload = response.json() or {}
    except Exception as exc:  # noqa: BLE001
        print(
            "WARNING: Ollama appears unreachable "
            f"(provider={provider}, model={model}, url={tags_url}): {exc}",
            file=sys.stderr,
        )
        return

    models = payload.get("models") or []
    available_model_names = {str(item.get("name") or "").strip() for item in models}
    model_is_available = any(
        name == model or name.startswith(f"{model}:") for name in available_model_names if name
    )
    if model and not model_is_available:
        print(
            "WARNING: Configured Ollama model not listed by server "
            f"(model={model}, url={tags_url}, available_count={len(available_model_names)}).",
            file=sys.stderr,
        )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_multiclass_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("Gold/pred lengths must match")
    total = len(gold)
    if total == 0:
        return {
            "count": 0,
            "accuracy": 0.0,
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_label": [],
        }

    labels = sorted(set(gold) | set(pred))
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    tp_counts: Counter[str] = Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp_counts[g] += 1

    per_label: list[dict[str, Any]] = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in labels:
        tp = int(tp_counts[label])
        support = int(gold_counts[label])
        pred_count = int(pred_counts[label])
        fp = pred_count - tp
        fn = support - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

        per_label.append(
            {
                "label": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    total_tp = sum(int(tp_counts[label]) for label in labels)
    total_fp = sum(int(pred_counts[label]) - int(tp_counts[label]) for label in labels)
    total_fn = sum(int(gold_counts[label]) - int(tp_counts[label]) for label in labels)
    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = (
        _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    label_count = len(labels)
    return {
        "count": total,
        "accuracy": _safe_div(correct, total),
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "macro": {
            "precision": _safe_div(macro_precision, label_count),
            "recall": _safe_div(macro_recall, label_count),
            "f1": _safe_div(macro_f1, label_count),
        },
        "weighted": {
            "precision": _safe_div(weighted_precision, total),
            "recall": _safe_div(weighted_recall, total),
            "f1": _safe_div(weighted_f1, total),
        },
        "per_label": per_label,
    }


def _print_sample_progress(*, record: dict[str, Any], processed: int, total: int) -> None:
    gold = dict(record.get("gold") or {})
    modernbert = dict(record.get("modernbert") or {})
    intent_classifier = dict(record.get("intent_classifier") or {})
    text = str(record.get("text") or "")
    line_number = record.get("line_number")

    print("=" * 80)
    print(f"Sample {processed}/{total} (line {line_number})")
    print(f"Text: {text}")
    print(f"Actual labels: category={gold.get('category', '')}, intent={gold.get('intent', '')}")
    print(
        "ModernBERT: "
        f"category={modernbert.get('category', '')}, "
        f"confidence={float(modernbert.get('confidence') or 0.0):.4f}, "
        f"correct={bool(modernbert.get('is_category_correct'))}"
    )
    modernbert_error = str(modernbert.get("error") or "")
    if modernbert_error:
        print(f"ModernBERT error: {modernbert_error}")

    intent_evaluated = bool(intent_classifier.get("evaluated"))
    intent_label = str(intent_classifier.get("intent") or "")
    intent_correct = intent_classifier.get("is_intent_correct")
    print(
        "Intent classifier: "
        f"evaluated={intent_evaluated}, "
        f"intent={intent_label if intent_label else '(empty)'}, "
        f"correct={intent_correct if intent_correct is not None else 'n/a'}"
    )
    intent_error = str(intent_classifier.get("error") or "")
    if intent_error:
        print(f"Intent classifier error: {intent_error}")
    print("=" * 80)


def _print_terminal_summary(
    *,
    summary_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    output_file: Path,
) -> None:
    counts = dict(summary_payload.get("counts") or {})
    metrics = dict(summary_payload.get("metrics") or {})
    category_metrics = dict(metrics.get("category") or {})
    intent_metrics = dict(metrics.get("intent_on_category_correct") or {})
    category_macro = dict(category_metrics.get("macro") or {})
    category_weighted = dict(category_metrics.get("weighted") or {})
    intent_macro = dict(intent_metrics.get("macro") or {})
    intent_weighted = dict(intent_metrics.get("weighted") or {})
    config = dict(metadata_payload.get("config") or {})

    print("\n" + "#" * 80)
    print("Evaluation summary")
    print("#" * 80)
    print(
        "Counts: "
        f"total_examples={counts.get('total_examples', 0)}, "
        f"category_correct_examples={counts.get('category_correct_examples', 0)}, "
        f"intent_evaluated_examples={counts.get('intent_evaluated_examples', 0)}, "
        f"skipped_invalid_rows={config.get('skipped_invalid_rows', 0)}"
    )
    print(
        "Category metrics: "
        f"accuracy={float(category_metrics.get('accuracy') or 0.0):.4f}, "
        f"macro_f1={float(category_macro.get('f1') or 0.0):.4f}, "
        f"weighted_f1={float(category_weighted.get('f1') or 0.0):.4f}"
    )
    print(
        "Intent metrics (on category-correct examples): "
        f"accuracy={float(intent_metrics.get('accuracy') or 0.0):.4f}, "
        f"macro_f1={float(intent_macro.get('f1') or 0.0):.4f}, "
        f"weighted_f1={float(intent_weighted.get('f1') or 0.0):.4f}"
    )
    print(f"Output file: {output_file}")
    print("#" * 80)


def _run_evaluation(
    *,
    rows: list[DatasetRow],
    classifier: QueryClassifier,
    verbose_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    category_gold: list[str] = []
    category_pred: list[str] = []
    intent_gold: list[str] = []
    intent_pred: list[str] = []
    category_correct_total = 0
    intent_eval_total = 0

    allowed_intents_cache: dict[str, list[str]] = {}

    total = len(rows)
    for processed, row in enumerate(rows, start=1):
        predicted_category = "unknown"
        category_confidence = 0.0
        category_error = ""
        try:
            category_out = classifier.classify(row.text)
            predicted_category = _normalize_category(category_out.category)
            category_confidence = float(category_out.confidence)
        except Exception as exc:  # noqa: BLE001
            category_error = str(exc)

        is_category_correct = predicted_category == row.gold_category
        if is_category_correct:
            category_correct_total += 1

        category_gold.append(row.gold_category)
        category_pred.append(predicted_category)

        predicted_intent = ""
        intent_error = ""
        intent_metadata: dict[str, Any] = {}
        is_intent_correct = None
        intent_evaluated = False
        if is_category_correct:
            if predicted_category not in allowed_intents_cache:
                try:
                    cached_allowed = get_intents_for_category(predicted_category)
                except Exception:  # noqa: BLE001
                    cached_allowed = []
                allowed_intents_cache[predicted_category] = [
                    _normalize_intent(item) for item in cached_allowed if _normalize_intent(item)
                ]
            allowed_intents = allowed_intents_cache.get(predicted_category, [])
            intent_metadata["allowed_intents"] = allowed_intents
            if not allowed_intents:
                intent_error = (
                    f"No DB allowed intents configured for category '{predicted_category}'; "
                    "skipping intent evaluation."
                )
            else:
                intent_evaluated = True
                intent_eval_total += 1
                try:
                    state: dict[str, Any] = {
                        "text": row.text,
                        "category": predicted_category,
                        "confidence": category_confidence,
                        "messages": [{"role": "user", "content": row.text, "metadata": {"source": "eval"}}],
                        "assistant_metadata": {},
                        "issue_locked": False,
                    }
                    intent_out = _classify_intent_node(state)
                    predicted_intent = _normalize_intent(str(intent_out.get("intent") or ""))
                    intent_metadata.update(dict(intent_out.get("assistant_metadata") or {}))
                except Exception as exc:  # noqa: BLE001
                    intent_error = str(exc)
                    predicted_intent = ""

                if predicted_intent and predicted_intent not in allowed_intents:
                    intent_error = (
                        f"Predicted intent '{predicted_intent}' is outside allowed DB intents "
                        f"for category '{predicted_category}'."
                    )
                    predicted_intent = ""

                is_intent_correct = predicted_intent == row.gold_intent
                intent_gold.append(row.gold_intent)
                intent_pred.append(predicted_intent)

        record = {
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
                "error": category_error,
            },
            "intent_classifier": {
                "evaluated": intent_evaluated,
                "intent": predicted_intent,
                "is_intent_correct": is_intent_correct,
                "error": intent_error,
                "metadata": intent_metadata,
            },
        }
        records.append(record)
        if verbose_progress:
            _print_sample_progress(record=record, processed=processed, total=total)

    summary = {
        "counts": {
            "total_examples": len(rows),
            "category_correct_examples": category_correct_total,
            "intent_evaluated_examples": intent_eval_total,
        },
        "metrics": {
            "category": _compute_multiclass_metrics(category_gold, category_pred),
            "intent_on_category_correct": _compute_multiclass_metrics(intent_gold, intent_pred),
        },
    }
    return records, summary


def _write_results(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    summary_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"run_{ts}.json"
    payload = {
        "summary": {"type": "summary", **summary_payload},
        "metadata": {"type": "metadata", **metadata_payload},
        "records": records,
    }
    with output_file.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ModernBERT category predictions and production intent classifier output "
            "against a labeled JSONL dataset."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help=f"Dataset JSONL file (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for run_YYYYMMDD_HHMMSS.json output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--bentoml-url",
        default="",
        help="Optional override for CLASSIFIER_BENTOML_URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        default=None,
        type=float,
        help="Optional override for CLASSIFIER_BENTOML_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid/empty rows instead of failing fast.",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print per-sample progress details during evaluation.",
    )
    parser.add_argument(
        "--max-limit",
        type=_positive_int,
        default=None,
        help="Optional positive integer to limit number of dataset rows evaluated.",
    )
    parser.add_argument(
        "--ollama-url",
        default="",
        help="Optional override for OLLAMA_BASE_URL (used for intent LLM and preflight).",
    )
    parser.add_argument(
        "--postgres-host",
        default="",
        help="Optional override for POSTGRES_HOST (used by DB-backed intent lookup).",
    )
    return parser


def main() -> int:
    _load_project_env()
    parser = _build_parser()
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    bentoml_url = str(args.bentoml_url or "").strip()
    timeout_seconds = args.timeout_seconds
    effective_bentoml_url = bentoml_url or os.getenv("CLASSIFIER_BENTOML_URL", "").strip()
    intent_model_provider = os.getenv("INTENT_MODEL_PROVIDER", "ollama").strip().lower()
    intent_model = os.getenv("INTENT_MODEL", "llama3.2").strip()
    ollama_url_cli = str(args.ollama_url or "").strip()
    postgres_host_cli = str(args.postgres_host or "").strip()
    if ollama_url_cli:
        os.environ["OLLAMA_BASE_URL"] = ollama_url_cli.rstrip("/")
    if postgres_host_cli:
        os.environ["POSTGRES_HOST"] = postgres_host_cli
    effective_ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    effective_postgres_host = os.getenv("POSTGRES_HOST", "").strip()
    if timeout_seconds is None:
        timeout_seconds = float(os.getenv("CLASSIFIER_BENTOML_TIMEOUT_SECONDS", "5"))

    try:
        rows, skipped_invalid_rows = _load_rows(input_file=input_file, skip_invalid=bool(args.skip_invalid))
        pre_limit_total_rows = len(rows)
        max_limit = args.max_limit
        if max_limit is not None:
            rows = rows[:max_limit]
        _warn_if_ollama_unreachable(provider=intent_model_provider, model=intent_model)
        classifier = QueryClassifier(
            endpoint=effective_bentoml_url or None,
            timeout_seconds=timeout_seconds,
        )
        records, raw_summary = _run_evaluation(
            rows=rows,
            classifier=classifier,
            verbose_progress=bool(args.verbose_progress),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    summary_payload = {**raw_summary}
    metadata_payload = {
        "timestamp_utc": timestamp_utc,
        "config": {
            "input_file": str(input_file),
            "output_dir": str(output_dir),
            "bentoml_url": effective_bentoml_url,
            "timeout_seconds": timeout_seconds,
            "skip_invalid": bool(args.skip_invalid),
            "verbose_progress": bool(args.verbose_progress),
            "max_limit": args.max_limit,
            "input_rows_before_limit": pre_limit_total_rows,
            "skipped_invalid_rows": skipped_invalid_rows,
            "intent_model_provider": intent_model_provider,
            "intent_model": intent_model,
            "ollama_base_url": effective_ollama_base_url,
            "postgres_host": effective_postgres_host,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
