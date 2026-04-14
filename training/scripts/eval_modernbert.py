#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Evaluate a fine-tuned ModernBERT classifier on JSONL and print the same metrics as train_modernbert.py.

Uses the same schema and ``compute_metrics`` as ``training/scripts/train_modernbert.py`` (accuracy,
precision/recall/F1 binary/macro/weighted, MCC, confusion counts, ROC-AUC, PR-AUC).

With ``--compare-with-base``, downloads (once) ModernBERT zeroshot v2 to ``training/models`` if missing,
evaluates the base model on the same data, and reports finetuned vs base metrics plus deltas.

Example (from repo root)::

    python training/scripts/eval_modernbert.py \\
        --checkpoint modernbert-large-is-issue \\
        --data-file training/data/train.jsonl

    python training/scripts/eval_modernbert.py \\
        --checkpoint modernbert-large-is-issue \\
        --data-file training/data/test.jsonl \\
        --metrics-json training/data/test_metrics.json

    python training/scripts/eval_modernbert.py \\
        --checkpoint training/models/my_run \\
        --data-file training/data/test.jsonl \\
        --compare-with-base
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Same directory as train_modernbert.py — import shared metric + data helpers.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_modernbert import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    _load_jsonl_as_dataset,
    _trainer_tokenizer_kwargs,
    build_compute_metrics_fn,
)

_LOG = logging.getLogger(__name__)

_REPO_TRAINING = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _REPO_TRAINING / "data" / "samples" / "bitext_binary_mini" / "eval.jsonl"
_DEFAULT_CHECKPOINT = _REPO_TRAINING / "models" / "modernbert_finetuned"
_DEFAULT_BASE_MODEL_ID = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
_DEFAULT_BASE_MODEL_DIR = _REPO_TRAINING / "models" / "modernbert-base-zeroshot-v2.0"

# Keys to skip when computing finetuned - base deltas (HF eval meta, not task metrics).
_DELTA_SKIP_KEYS = frozenset(
    {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "epoch"}
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ModernBERT issue classifier on JSONL (same metrics as training)."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Directory with saved model + tokenizer (default: modernbert-large-is-issue).",
    )
    p.add_argument(
        "--data-file",
        type=Path,
        default=_DEFAULT_DATA,
        help="JSONL with text/label rows (default: training/data/train.jsonl).",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Use fp16 (CUDA).")
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 on CUDA when available (default: on). Use --no-bf16 to disable.",
    )
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="If set, write metrics (and metadata) to this JSON file.",
    )
    p.add_argument(
        "--compare-with-base",
        action="store_true",
        help="Also evaluate ModernBERT zeroshot v2 (cached under training/models) and print deltas vs finetuned.",
    )
    p.add_argument(
        "--base-model-id",
        type=str,
        default=_DEFAULT_BASE_MODEL_ID,
        help=f"Hugging Face model id for base zeroshot (default: {_DEFAULT_BASE_MODEL_ID}).",
    )
    p.add_argument(
        "--base-model-dir",
        type=Path,
        default=_DEFAULT_BASE_MODEL_DIR,
        help="Local directory to cache the base model (default: training/models/modernbert-base-zeroshot-v2.0).",
    )
    return p.parse_args()


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


def _ensure_base_model_local(base_model_id: str, base_model_dir: Path) -> Path:
    """Use cached base under base_model_dir if present; otherwise download and save there."""
    base_model_dir = base_model_dir.resolve()
    config_path = base_model_dir / "config.json"
    if config_path.is_file():
        _LOG.info("Using cached base model at %s", base_model_dir)
        return base_model_dir

    base_model_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("Downloading base model %s -> %s", base_model_id, base_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.save_pretrained(str(base_model_dir))
    tokenizer.save_pretrained(str(base_model_dir))
    return base_model_dir


def _run_eval(
    model_source: str,
    ds: Any,
    *,
    max_length: int,
    eval_batch_size: int,
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
    base_init: bool,
    eval_desc: str,
) -> tuple[dict[str, Any], str | None]:
    """Tokenize ``ds``, run Trainer.evaluate, return (metrics dict, config name_or_path)."""
    if base_init:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

    def tokenize_fn(batch: dict[str, list]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    eval_tok = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc=eval_desc,
    )

    tmpdir = tempfile.mkdtemp(prefix="modernbert-eval-")
    try:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=eval_batch_size,
            seed=seed,
            data_seed=seed,
            report_to="none",
            fp16=use_fp16,
            bf16=use_bf16,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_tok,
            compute_metrics=build_compute_metrics_fn(),
            **_trainer_tokenizer_kwargs(tokenizer),
        )
        metrics = trainer.evaluate()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    name_or_path = getattr(model.config, "name_or_path", None)
    return dict(metrics), name_or_path


def _metric_deltas(
    finetuned_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
) -> dict[str, float]:
    """Per-key difference finetuned - base for numeric task metrics shared by both."""
    out: dict[str, float] = {}
    for k in finetuned_metrics:
        if k in _DELTA_SKIP_KEYS:
            continue
        if k not in base_metrics:
            continue
        fv, bv = finetuned_metrics[k], base_metrics[k]
        if isinstance(fv, bool) or isinstance(bv, bool):
            continue
        if isinstance(fv, (int, float, np.floating, np.integer)) and isinstance(
            bv, (int, float, np.floating, np.integer)
        ):
            try:
                out[str(k)] = float(fv) - float(bv)
            except (TypeError, ValueError):
                continue
    return out


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()

    ckpt = args.checkpoint.resolve()
    data_path = args.data_file.resolve()

    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    _LOG.info("Loading data: %s", data_path)
    ds = _load_jsonl_as_dataset(data_path)
    _LOG.info("Samples: %s", len(ds))

    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())

    _LOG.info("Evaluating finetuned checkpoint=%s", ckpt)
    metrics, ft_name_or_path = _run_eval(
        str(ckpt),
        ds,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        base_init=False,
        eval_desc="Tokenizing finetuned",
    )
    _LOG.info("Metrics: %s", metrics)

    if not args.compare_with_base:
        payload = {
            "checkpoint": str(ckpt),
            "data_file": str(data_path),
            "num_samples": len(ds),
            "config_name_or_path": ft_name_or_path,
            "metrics": _json_safe(dict(metrics)),
        }
        if args.metrics_json is not None:
            out = args.metrics_json.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _LOG.info("Wrote %s", out)
        print(json.dumps(metrics, indent=2))
        return 0

    base_dir = _ensure_base_model_local(args.base_model_id, args.base_model_dir)
    _LOG.info("Evaluating base model at %s", base_dir)
    base_metrics, base_name_or_path = _run_eval(
        str(base_dir),
        ds,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        base_init=True,
        eval_desc="Tokenizing base",
    )
    _LOG.info("Base metrics: %s", base_metrics)

    deltas = _metric_deltas(metrics, base_metrics)
    compare_payload = {
        "finetuned": _json_safe(dict(metrics)),
        "base": _json_safe(dict(base_metrics)),
        "deltas_finetuned_minus_base": _json_safe(deltas),
        "base_model_id": args.base_model_id,
        "base_model_dir": str(base_dir),
    }

    payload = {
        "checkpoint": str(ckpt),
        "data_file": str(data_path),
        "num_samples": len(ds),
        "config_name_or_path": ft_name_or_path,
        "metrics": _json_safe(dict(metrics)),
        "base_model_id": args.base_model_id,
        "base_model_dir": str(base_dir),
        "base_config_name_or_path": base_name_or_path,
        "base_metrics": _json_safe(dict(base_metrics)),
        "metric_deltas_finetuned_minus_base": _json_safe(deltas),
        "compare": compare_payload,
    }

    if args.metrics_json is not None:
        out = args.metrics_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _LOG.info("Wrote %s", out)

    print(json.dumps(compare_payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
