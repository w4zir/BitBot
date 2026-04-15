#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Fine-tune a ModernBERT checkpoint for multiclass text classification on JSONL.

Expected layout under ``--dataset-dir``:
  train.jsonl, eval.jsonl, test.jsonl, label2id.json

JSONL schema (one object per line):
  {"text": "<user utterance>", "label": <int>}
  Labels are integer ids matching ``label2id.json``.

The base checkpoint is stored at ``training/models/modernbert-base-zeroshot-v2.0``;
if missing, it is downloaded from Hugging Face ``MoritzLaurer/ModernBERT-base-zeroshot-v2.0``.

Checkpoints and logs go under a timestamped directory by default, for example
``training/models/bitext_multiclass_finetuned_<UTC_TIMESTAMP>/``. The best model
(by eval ``f1_macro``) is written to ``.../winner/`` after training.

Local (from repo root)::

    pip install -r training/requirements-train.txt
    python training/experiments/src/train_multiclass_modernbert.py \\
        --dataset-dir training/data/bitext --num-epochs 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for metrics. Install with: pip install scikit-learn"
    ) from e

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

_LOG = logging.getLogger(__name__)

# training/experiments/src -> training/
_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent

HF_ZEROSHOT_MODEL_ID = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
_LOCAL_BASE_MODEL_DIR = _TRAINING_ROOT / "models" / "modernbert-base-zeroshot-v2.0"
_DEFAULT_DATASET_DIR = _TRAINING_ROOT / "data" / "bitext"
_DEFAULT_OUTPUT_DIR = _TRAINING_ROOT / "models" / "bitext_multiclass_finetuned"
_DEFAULT_RESULTS_DIR = _EXPERIMENTS_ROOT / "results"


def _training_args_eval_kwargs() -> dict[str, Any]:
    """HF renamed evaluation_strategy -> eval_strategy; support transformers 4.40 through current."""
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        return {"eval_strategy": "steps"}
    if "evaluation_strategy" in sig.parameters:
        return {"evaluation_strategy": "steps"}
    raise RuntimeError(
        "Unsupported transformers: TrainingArguments has neither eval_strategy nor evaluation_strategy."
    )


def _trainer_tokenizer_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Trainer uses processing_class in recent versions; older releases used tokenizer."""
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _has_local_checkpoint_files(model_dir: Path) -> bool:
    """True if directory looks like a usable HF model folder."""
    if not (model_dir / "config.json").is_file():
        return False
    if (model_dir / "model.safetensors").is_file():
        return True
    if (model_dir / "pytorch_model.bin").is_file():
        return True
    # sharded safetensors
    if list(model_dir.glob("model-*.safetensors")):
        return True
    return False


def ensure_local_base_model(
    local_dir: Path,
    hub_model_id: str = HF_ZEROSHOT_MODEL_ID,
) -> Path:
    """Download tokenizer + model to ``local_dir`` if not already present."""
    local_dir = local_dir.resolve()
    if _has_local_checkpoint_files(local_dir):
        _LOG.info("Using existing local base model at %s", local_dir)
        return local_dir

    _LOG.info(
        "Local base model not found or incomplete at %s; downloading %s",
        local_dir,
        hub_model_id,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        hub_model_id,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(local_dir))
    model.save_pretrained(str(local_dir))
    _LOG.info("Saved base model to %s", local_dir)
    return local_dir


def load_label_maps(path: Path) -> tuple[int, dict[int, str], dict[str, int]]:
    """Load label2id.json; return num_labels, id2label (int -> str), label2id (str -> int)."""
    if not path.is_file():
        raise FileNotFoundError(f"label2id file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected non-empty JSON object (string -> int)")
    label2id: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(f"{path}: label keys must be strings, got {type(k)}")
        label2id[k] = int(v)
    ids = sorted(label2id.values())
    expected = list(range(len(label2id)))
    if ids != expected:
        raise ValueError(
            f"{path}: label ids must be contiguous 0..{len(label2id) - 1}, got {ids[:8]}... "
            f"(len={len(ids)})"
        )
    num_labels = len(label2id)
    id2label: dict[int, str] = {int(v): k for k, v in label2id.items()}
    return num_labels, id2label, label2id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ModernBERT (multiclass) on train/eval/test JSONL under a dataset folder."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET_DIR,
        help=f"Folder containing train.jsonl, eval.jsonl, test.jsonl, label2id.json (default: {_DEFAULT_DATASET_DIR}).",
    )
    p.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Override path to train.jsonl (default: <dataset-dir>/train.jsonl).",
    )
    p.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Override path to eval.jsonl (default: <dataset-dir>/eval.jsonl).",
    )
    p.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Override path to test.jsonl (default: <dataset-dir>/test.jsonl).",
    )
    p.add_argument(
        "--label2id-file",
        type=Path,
        default=None,
        help="Override path to label2id.json (default: <dataset-dir>/label2id.json).",
    )
    p.add_argument(
        "--local-base-model-dir",
        type=Path,
        default=_LOCAL_BASE_MODEL_DIR,
        help=f"Directory for the ModernBERT zeroshot v2 base weights (default: {_LOCAL_BASE_MODEL_DIR}).",
    )
    p.add_argument(
        "--hub-model-id",
        type=str,
        default=HF_ZEROSHOT_MODEL_ID,
        help=f"Hugging Face model id used only when the local base is missing (default: {HF_ZEROSHOT_MODEL_ID}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Base directory name for this run; a UTC timestamp is appended by default "
            f"(e.g. {_DEFAULT_OUTPUT_DIR}_20250415T120000Z) so each finetuning run has its own folder."
        ),
    )
    p.add_argument(
        "--no-output-timestamp",
        action="store_true",
        help="Use --output-dir exactly as given (no UTC timestamp suffix on the folder name).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Directory for experiment summary JSON (default: {_DEFAULT_RESULTS_DIR}).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="modernbert-multiclass",
        help="Short label used in the experiment results filename.",
    )
    p.add_argument("--num-epochs", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size.")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides num_epochs.")
    p.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Run evaluation on eval.jsonl every N steps (default: 50).",
    )
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--fp16", action="store_true", help="Use fp16 (CUDA).")
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 on CUDA when available (default: on). Use --no-bf16 to disable.",
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="HF TrainingArguments optim name (default: adamw_torch_fused).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory (recommended on small GPUs).",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping on eval loss.",
    )
    return p.parse_args()


def _validate_jsonl_schema(
    rows: list[dict[str, Any]], path: Path, valid_label_ids: set[int]
) -> None:
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {i+1}: expected object, got {type(row)}")
        if "text" not in row or "label" not in row:
            raise ValueError(f"{path}: line {i+1}: missing 'text' or 'label'")
        if not isinstance(row["text"], str):
            raise ValueError(f"{path}: line {i+1}: 'text' must be string")
        lab = int(row["label"])
        if lab not in valid_label_ids:
            raise ValueError(
                f"{path}: line {i+1}: 'label' must be one of label2id values, got {lab!r}"
            )


def _load_jsonl_as_dataset(path: Path, valid_label_ids: set[int]) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    _validate_jsonl_schema(raw, path, valid_label_ids)
    slim = [{"text": r["text"], "label": int(r["label"])} for r in raw]
    return Dataset.from_list(slim)


def _load_datasets(
    train_path: Path, eval_path: Path, valid_label_ids: set[int]
) -> tuple[Dataset, Dataset]:
    """Load via datasets when possible; fall back to manual JSONL for Colab edge cases."""
    try:
        ds_dict = load_dataset(
            "json",
            data_files={"train": str(train_path), "validation": str(eval_path)},
        )
        train_ds = ds_dict["train"]
        eval_ds = ds_dict["validation"]
        train_list = train_ds.to_list()
        eval_list = eval_ds.to_list()
        _validate_jsonl_schema(train_list, train_path, valid_label_ids)
        _validate_jsonl_schema(eval_list, eval_path, valid_label_ids)
        train_ds = Dataset.from_list(
            [{"text": r["text"], "label": int(r["label"])} for r in train_list]
        )
        eval_ds = Dataset.from_list(
            [{"text": r["text"], "label": int(r["label"])} for r in eval_list]
        )
        return train_ds, eval_ds
    except Exception:
        _LOG.debug("load_dataset json failed, using manual JSONL reader", exc_info=True)
        return (
            _load_jsonl_as_dataset(train_path, valid_label_ids),
            _load_jsonl_as_dataset(eval_path, valid_label_ids),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def build_compute_metrics_fn(num_labels: int) -> Any:
    label_list = list(range(num_labels))

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        preds = np.argmax(logits, axis=-1)
        probs = _softmax(logits) if logits.shape[-1] >= 2 else None

        out: dict[str, float] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_macro": float(
                precision_score(labels, preds, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "precision_weighted": float(
                precision_score(labels, preds, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(labels, preds, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, preds)),
        }

        uniq = np.unique(labels)
        if probs is not None and len(uniq) > 1 and logits.shape[-1] == num_labels:
            try:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        labels,
                        probs,
                        multi_class="ovr",
                        average="macro",
                        labels=label_list,
                    )
                )
            except ValueError:
                out["roc_auc_ovr_macro"] = float("nan")
            try:
                y_bin = label_binarize(labels, classes=label_list)
                out["pr_auc_macro"] = float(
                    average_precision_score(y_bin, probs, average="macro")
                )
            except (TypeError, ValueError):
                out["pr_auc_macro"] = float("nan")
        else:
            out["roc_auc_ovr_macro"] = float("nan")
            out["pr_auc_macro"] = float("nan")

        return out

    return compute_metrics


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (float, int)):
        return x
    if isinstance(x, np.generic):
        return x.item()
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def _try_import_version(mod_name: str) -> str | None:
    try:
        mod = __import__(mod_name)
        return str(getattr(mod, "__version__", None))
    except Exception:
        return None


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    dataset_dir = args.dataset_dir.resolve()
    train_path = (args.train_file or dataset_dir / "train.jsonl").resolve()
    eval_path = (args.eval_file or dataset_dir / "eval.jsonl").resolve()
    test_path = (args.test_file or dataset_dir / "test.jsonl").resolve()
    label2id_path = (args.label2id_file or dataset_dir / "label2id.json").resolve()

    num_labels, id2label, label2id = load_label_maps(label2id_path)
    valid_ids = set(label2id.values())

    output_base = args.output_dir.resolve()
    if args.no_output_timestamp:
        output_dir = output_base
    else:
        run_ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_base.parent / f"{output_base.name}_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    local_base = ensure_local_base_model(
        args.local_base_model_dir.resolve(),
        hub_model_id=args.hub_model_id,
    )
    model_load_path = str(local_base)

    _LOG.info("Dataset dir=%s", dataset_dir)
    _LOG.info("Loading data: train=%s eval=%s test=%s", train_path, eval_path, test_path)
    _LOG.info("num_labels=%s label2id=%s", num_labels, label2id_path)
    train_ds, eval_ds = _load_datasets(train_path, eval_path, valid_ids)
    test_ds = _load_jsonl_as_dataset(test_path, valid_ids)
    _LOG.info(
        "Train size=%s eval size=%s test size=%s",
        len(train_ds),
        len(eval_ds),
        len(test_ds),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_load_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    eval_tok = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    )
    test_tok = test_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=test_ds.column_names,
        desc="Tokenizing test",
    )

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())
    eval_steps = max(1, args.eval_steps)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        **_training_args_eval_kwargs(),
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=max(1, min(100, eval_steps)),
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    callbacks = []
    if not args.no_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    compute_metrics_fn = build_compute_metrics_fn(num_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks or None,
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    winner_dir = output_dir / "winner"
    _LOG.info("Training; artifacts -> %s (best model will be saved under %s)", output_dir, winner_dir)
    _LOG.info("Eval every %s steps on %s", eval_steps, eval_path)
    train_result = trainer.train()
    _LOG.info("Train loss=%s", getattr(train_result, "training_loss", None))

    eval_metrics = trainer.evaluate()
    _LOG.info("Final eval metrics: %s", eval_metrics)
    print("\n=== Final eval (eval.jsonl) ===")
    print(json.dumps(eval_metrics, indent=2))

    predict_eval = trainer.predict(eval_tok)
    pred_logits_eval = np.asarray(predict_eval.predictions, dtype=np.float64)
    pred_labels_eval = np.asarray(predict_eval.label_ids, dtype=np.int64)
    pred_ids_eval = np.argmax(pred_logits_eval, axis=-1)
    cm_eval = confusion_matrix(
        pred_labels_eval, pred_ids_eval, labels=list(range(num_labels))
    )

    predict_test = trainer.predict(test_tok)
    pred_logits_test = np.asarray(predict_test.predictions, dtype=np.float64)
    pred_labels_test = np.asarray(predict_test.label_ids, dtype=np.int64)
    test_metrics = compute_metrics_fn((predict_test.predictions, predict_test.label_ids))
    pred_ids_test = np.argmax(pred_logits_test, axis=-1)
    cm_test = confusion_matrix(
        pred_labels_test, pred_ids_test, labels=list(range(num_labels))
    )

    _LOG.info("Final test metrics: %s", test_metrics)
    print("\n=== Final test (test.jsonl) ===")
    print(json.dumps(test_metrics, indent=2))

    # load_best_model_at_end=True ensures the in-memory model is the eval-f1_macro winner.
    winner_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(winner_dir))
    tokenizer.save_pretrained(str(winner_dir))
    _LOG.info("Saved final winner (best checkpoint) to %s", winner_dir)

    train_out_metrics = getattr(train_result, "metrics", None)
    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    reproducibility: dict[str, Any] = {
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _try_import_version("torch"),
        "transformers": _try_import_version("transformers"),
        "datasets": _try_import_version("datasets"),
    }

    artifact_metrics_path = output_dir / "metrics.json"
    artifact_payload: dict[str, Any] = {
        "hub_model_id": args.hub_model_id,
        "local_base_model_dir": str(local_base),
        "model_load_path": model_load_path,
        "finetuned_winner_dir": str(winner_dir),
        "label2id_file": str(label2id_path),
        "label2id": label2id,
        "num_labels": num_labels,
        "dataset_dir": str(dataset_dir),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "test_file": str(test_path),
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "test_samples": len(test_ds),
        "training_args": _json_safe(training_args.to_dict()),
        "train_runtime": _json_safe(train_out_metrics) if train_out_metrics else None,
        "eval_metrics_final": _json_safe(dict(eval_metrics)),
        "test_metrics_final": _json_safe(test_metrics),
        "confusion_matrix_eval": cm_eval.tolist(),
        "confusion_matrix_test": cm_test.tolist(),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "reproducibility": reproducibility,
    }
    artifact_metrics_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    _LOG.info("Wrote %s", artifact_metrics_path)

    ds_slug = dataset_dir.name.replace(" ", "_")
    safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.run_name)[:80]
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_name = f"{safe_run}_{ds_slug}_multiclass_{ts}.json"
    results_path = results_dir / results_name

    experiment_record: dict[str, Any] = {
        "experiment": {
            "name": args.run_name,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        },
        "model": {
            "hub_model_id": args.hub_model_id,
            "local_base_model_dir": str(local_base),
            "finetuned_output_dir": str(output_dir),
            "finetuned_winner_dir": str(winner_dir),
        },
        "data": {
            "dataset_dir": str(dataset_dir),
            "train_file": str(train_path),
            "eval_file": str(eval_path),
            "test_file": str(test_path),
            "label2id_file": str(label2id_path),
            "label2id": label2id,
            "num_labels": num_labels,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "test_samples": len(test_ds),
        },
        "training": {
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "eval_steps": eval_steps,
            "seed": args.seed,
            "training_args": _json_safe(training_args.to_dict()),
        },
        "metrics": {
            "train": _json_safe(train_out_metrics) if train_out_metrics else None,
            "eval_final": _json_safe(dict(eval_metrics)),
            "test_final": _json_safe(test_metrics),
            "confusion_matrix_eval": cm_eval.tolist(),
            "confusion_matrix_test": cm_test.tolist(),
        },
        "reproducibility": reproducibility,
    }
    results_path.write_text(json.dumps(experiment_record, indent=2), encoding="utf-8")
    _LOG.info("Wrote experiment results %s", results_path)
    print(f"\nExperiment summary saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
