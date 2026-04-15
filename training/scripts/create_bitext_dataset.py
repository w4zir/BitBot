#!/usr/bin/env python3
"""
Combine Hugging Face Bitext with synthetic no-issue JSON and emit train/eval/test JSONL.

Modes (--mode):
  - binary: Bitext rows -> label "issue", synthetic -> "no_issue"; JSONL uses 0=no_issue, 1=issue.
  - category: Bitext rows use HF `category`; synthetic -> "no_issue"; JSONL uses integer ids from label2id.json.
  - intent: Bitext rows use HF `intent`; synthetic -> "no_issue"; JSONL uses integer ids from label2id.json.

Synthetic: only samples with is_issue=false from files matching --synthetic-glob under --input-dir,
  text = user_message, label = "no_issue".

Defaults: 70% / 15% / 15% train/eval/test (stratified), configurable via --train-ratio and --eval-ratio.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

Mode = Literal["binary", "category", "intent"]

DEFAULT_HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DEFAULT_HF_SPLIT = "train"
_REPO_TRAINING = Path(__file__).resolve().parent.parent
_REPO_ROOT = _REPO_TRAINING.parent
DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "raw" / "synthetic" / "no_issue"
DEFAULT_SYNTHETIC_GLOB = "no_issue_*.json"
DEFAULT_OUTPUT_DIR = _REPO_TRAINING / "data" / "bitext"
DEFAULT_SEED = 42

BINARY_NO_ISSUE = "no_issue"
BINARY_ISSUE = "issue"


def normalize_text_key(text: str) -> str:
    """Normalize for deduplication: strip, lowercase, collapse whitespace."""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_samples_from_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load samples from one JSON file. Returns (samples, counters)."""
    stats = {"malformed_root": 0, "missing_samples": 0}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to read JSON: {path}: {e}") from e

    if not isinstance(raw, dict):
        stats["malformed_root"] = 1
        return [], stats

    samples = raw.get("samples")
    if not isinstance(samples, list):
        stats["missing_samples"] = 1
        return [], stats

    out: list[dict[str, Any]] = []
    for obj in samples:
        if not isinstance(obj, dict):
            continue
        out.append({**obj, "_source_file": path.name})
    return out, stats


def row_from_no_issue_sample(
    sample: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Build row for synthetic no-issue only (is_issue must be False)."""
    msg = sample.get("user_message")
    if msg is None or not isinstance(msg, str) or not msg.strip():
        return None, "missing_or_empty_user_message"

    flag = sample.get("is_issue")
    if not isinstance(flag, bool):
        return None, "invalid_is_issue"
    if flag is not False:
        return None, "skipped_issue_sample"

    row: dict[str, Any] = {
        "text": msg.strip(),
        "source_file": sample.get("_source_file"),
    }
    return row, None


def load_bitext_rows(
    dataset_name: str,
    split: str,
    skip_reasons: Counter[str],
) -> list[dict[str, Any]]:
    """Load Bitext from Hugging Face: instruction, intent, category."""
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )
    ds = load_dataset(dataset_name, split=split)
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(ds):
        row_dict = dict(raw)
        instruction = row_dict.get("instruction")
        intent = row_dict.get("intent")
        category = row_dict.get("category")

        if instruction is None or not isinstance(instruction, str) or not instruction.strip():
            skip_reasons["bitext_missing_or_empty_instruction"] += 1
            continue
        if intent is None or not isinstance(intent, str) or not intent.strip():
            skip_reasons["bitext_missing_or_empty_intent"] += 1
            continue
        if category is None or not isinstance(category, str) or not category.strip():
            skip_reasons["bitext_missing_or_empty_category"] += 1
            continue

        rows.append(
            {
                "text": instruction.strip(),
                "intent": intent.strip(),
                "category": category.strip(),
                "bitext_index": idx,
            }
        )
    return rows


def apply_mode_labels(
    bitext_rows: list[dict[str, Any]],
    synth_rows: list[dict[str, Any]],
    mode: Mode,
) -> list[dict[str, Any]]:
    """Assign final string `label` per row for the given mode."""
    out: list[dict[str, Any]] = []
    for r in bitext_rows:
        if mode == "binary":
            lab = BINARY_ISSUE
        elif mode == "category":
            lab = r["category"]
        else:
            lab = r["intent"]
        out.append({"text": r["text"], "label": lab})
    for r in synth_rows:
        out.append({"text": r["text"], "label": BINARY_NO_ISSUE})
    return out


def dedupe_by_text(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep first occurrence per normalized text key."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dup = 0
    for r in rows:
        key = normalize_text_key(r["text"])
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, dup


def label_distribution_str(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(str(r["label"]) for r in rows)
    return {k: c[k] for k in sorted(c.keys())}


def label_distribution_int_multiclass(
    rows: list[dict[str, Any]], label2id: dict[str, int]
) -> dict[str, int]:
    c: Counter[int] = Counter()
    for r in rows:
        lab = str(r["label"])
        c[label2id[lab]] += 1
    return {str(k): c[k] for k in sorted(c.keys())}


def label_distribution_int_binary(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[int] = Counter()
    for r in rows:
        c[int(r["label"])] += 1
    return {str(k): c[k] for k in sorted(c.keys())}


def build_label2id(all_labels_sorted: list[str]) -> dict[str, int]:
    """Deterministic mapping: sorted label -> 0..n-1."""
    return {lab: i for i, lab in enumerate(all_labels_sorted)}


def stratified_train_eval_test_split_by_label(
    rows: list[dict[str, Any]],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified 3-way split by string label (train / eval / test)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if train_ratio + eval_ratio >= 1.0:
        raise ValueError("train_ratio + eval_ratio must be < 1 so test has a positive share")

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        lab = str(r["label"])
        by_label[lab].append(r)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for _label in sorted(by_label.keys()):
        items = by_label[_label][:]
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            train.extend(items)
            continue

        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        n_test = n - n_train - n_eval

        if n_train <= 0:
            n_train = 1
            n_test = n - n_train - n_eval
            if n_test < 0:
                n_eval = max(0, n_eval + n_test)
                n_test = 0
        elif n_train >= n:
            n_train = max(1, n - 1)
            n_eval = min(n_eval, max(0, n - n_train - 1))
            n_test = n - n_train - n_eval

        if n_test < 0:
            n_eval = max(0, n_eval + n_test)
            n_test = 0

        train.extend(items[:n_train])
        eval_rows.extend(items[n_train : n_train + n_eval])
        test.extend(items[n_train + n_eval :])

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    rng.shuffle(test)
    return train, eval_rows, test


def map_rows_to_binary_int(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace string label with int: no_issue=0, issue=1."""
    out: list[dict[str, Any]] = []
    for r in rows:
        s = str(r["label"])
        b = 0 if s == BINARY_NO_ISSUE else 1
        out.append({"text": r["text"], "label": b})
    return out


def stratified_train_eval_test_split_by_binary_int(
    rows: list[dict[str, Any]],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified 3-way split by integer label 0 or 1."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if train_ratio + eval_ratio >= 1.0:
        raise ValueError("train_ratio + eval_ratio must be < 1 so test has a positive share")

    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)

    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for _label in sorted(by_label.keys()):
        items = by_label[_label][:]
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue
        if n == 1:
            train.extend(items)
            continue

        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        n_test = n - n_train - n_eval

        if n_train <= 0:
            n_train = 1
            n_test = n - n_train - n_eval
            if n_test < 0:
                n_eval = max(0, n_eval + n_test)
                n_test = 0
        elif n_train >= n:
            n_train = max(1, n - 1)
            n_eval = min(n_eval, max(0, n - n_train - 1))
            n_test = n - n_train - n_eval

        if n_test < 0:
            n_eval = max(0, n_eval + n_test)
            n_test = 0

        train.extend(items[:n_train])
        eval_rows.extend(items[n_train : n_train + n_eval])
        test.extend(items[n_train + n_eval :])

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    rng.shuffle(test)
    return train, eval_rows, test


def write_jsonl_multiclass(
    path: Path, rows: list[dict[str, Any]], label2id: dict[str, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            lab = str(r["label"])
            line = json.dumps(
                {"text": r["text"], "label": label2id[lab]},
                ensure_ascii=False,
            )
            f.write(line + "\n")


def write_jsonl_binary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(
                {"text": r["text"], "label": int(r["label"])},
                ensure_ascii=False,
            )
            f.write(line + "\n")


def slim_row_for_json(r: dict[str, Any]) -> dict[str, str]:
    return {"text": r["text"], "label": str(r["label"])}


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [slim_row_for_json(r) for r in rows]
    path.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Bitext + synthetic no_issue train/eval/test JSONL (binary, category, or intent).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("binary", "category", "intent"),
        required=True,
        help="Dataset type: binary (issue vs no_issue), category (HF category + no_issue), intent (HF intent + no_issue).",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset id for Bitext",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default=DEFAULT_HF_SPLIT,
        help="Dataset split to load (default: train)",
    )
    parser.add_argument(
        "--input-dir",
        "--synthetic-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        dest="input_dir",
        help="Directory containing synthetic no-issue JSON files (no_issue_*.json by default). "
        "Default: <repo>/data/raw/synthetic/no_issue. --synthetic-dir is an alias.",
    )
    parser.add_argument(
        "--synthetic-glob",
        type=str,
        default=DEFAULT_SYNTHETIC_GLOB,
        help="Glob pattern for synthetic no-issue JSON files under --input-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.jsonl, eval.jsonl, test.jsonl, dataset_stats.json, optional dataset_full.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target fraction per class/label for training (stratified)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Target fraction per class/label for eval; test gets remainder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling and split",
    )
    parser.add_argument(
        "--bitext-only",
        action="store_true",
        help="Load only Hugging Face Bitext (no synthetic no_issue JSON files).",
    )
    parser.add_argument(
        "--write-dataset-full",
        action="store_true",
        help="Also write dataset_full.json (deduped rows with string labels for this mode).",
    )
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    mode: Mode = args.mode  # type: ignore[assignment]
    synthetic_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files: list[Path] = []
    synth_rows: list[dict[str, Any]] = []
    root_issues = 0

    if not args.bitext_only:
        if not synthetic_dir.is_dir():
            print(f"Error: input directory does not exist: {synthetic_dir}", file=sys.stderr)
            return 1

        json_files = sorted(synthetic_dir.glob(args.synthetic_glob))
        if not json_files:
            print(
                f"Error: no files matching {args.synthetic_glob!r} in {synthetic_dir}",
                file=sys.stderr,
            )
            return 1

    skip_reasons: Counter[str] = Counter()
    bitext_rows = load_bitext_rows(args.hf_dataset, args.hf_split, skip_reasons)

    if not args.bitext_only:
        synth_samples: list[dict[str, Any]] = []
        for jf in json_files:
            samples, st = load_samples_from_file(jf)
            root_issues += st["malformed_root"] + st["missing_samples"]
            synth_samples.extend(samples)

        for s in synth_samples:
            row, reason = row_from_no_issue_sample(s)
            if row is None:
                skip_reasons[reason or "unknown"] += 1
                continue
            synth_rows.append(row)

    rows = apply_mode_labels(bitext_rows, synth_rows, mode)
    rows_before_dedup = len(rows)
    rows, n_dup = dedupe_by_text(rows)

    if not rows:
        print("Error: no rows after loading and validation", file=sys.stderr)
        return 1

    test_ratio_effective = 1.0 - args.train_ratio - args.eval_ratio
    try:
        if mode == "binary":
            rows_bin = map_rows_to_binary_int(rows)
            train_rows, eval_rows, test_rows = stratified_train_eval_test_split_by_binary_int(
                rows_bin, args.train_ratio, args.eval_ratio, args.seed
            )
        else:
            train_rows, eval_rows, test_rows = stratified_train_eval_test_split_by_label(
                rows, args.train_ratio, args.eval_ratio, args.seed
            )
    except ValueError as e:
        print(f"Error: invalid split ratios: {e}", file=sys.stderr)
        return 1

    all_labels = sorted({str(r["label"]) for r in rows})
    label2id = build_label2id(all_labels)

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    stats_path = output_dir / "dataset_stats.json"

    if mode == "binary":
        write_jsonl_binary(train_path, train_rows)
        write_jsonl_binary(eval_path, eval_rows)
        write_jsonl_binary(test_path, test_rows)
    else:
        write_jsonl_multiclass(train_path, train_rows, label2id)
        write_jsonl_multiclass(eval_path, eval_rows, label2id)
        write_jsonl_multiclass(test_path, test_rows, label2id)

    stats: dict[str, Any] = {
        "mode": mode,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "bitext_only": bool(args.bitext_only),
        "input_dir": str(synthetic_dir.resolve()) if not args.bitext_only else None,
        "synthetic_glob": args.synthetic_glob,
        "synthetic_files": [p.name for p in json_files],
        "output_dir": str(output_dir.resolve()),
        "bitext_rows_loaded": len(bitext_rows),
        "synthetic_rows_valid": len(synth_rows),
        "rows_before_dedup": rows_before_dedup,
        "rows_after_dedup": len(rows),
        "duplicates_removed": n_dup,
        "skipped_rows": dict(skip_reasons),
        "skipped_total": int(sum(skip_reasons.values())),
        "malformed_or_missing_samples_key_files": root_issues,
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "test_ratio_target": test_ratio_effective,
        "seed": args.seed,
        "num_labels": len(all_labels),
        "class_distribution_all": label_distribution_str(rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "test_size": len(test_rows),
    }

    if mode == "binary":
        stats["binary_mapping"] = {BINARY_NO_ISSUE: 0, BINARY_ISSUE: 1}
        stats["binary_class_distribution_all"] = label_distribution_int_binary(map_rows_to_binary_int(rows))
        stats["binary_class_distribution_train"] = label_distribution_int_binary(train_rows)
        stats["binary_class_distribution_eval"] = label_distribution_int_binary(eval_rows)
        stats["binary_class_distribution_test"] = label_distribution_int_binary(test_rows)
    else:
        stats["label2id"] = label2id
        stats["class_distribution_train"] = label_distribution_str(train_rows)
        stats["class_distribution_eval"] = label_distribution_str(eval_rows)
        stats["class_distribution_test"] = label_distribution_str(test_rows)
        stats["label_id_distribution_train"] = label_distribution_int_multiclass(train_rows, label2id)
        stats["label_id_distribution_eval"] = label_distribution_int_multiclass(eval_rows, label2id)
        stats["label_id_distribution_test"] = label_distribution_int_multiclass(test_rows, label2id)

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if mode != "binary":
        label2id_path = output_dir / "label2id.json"
        label2id_path.write_text(
            json.dumps(label2id, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        label2id_path = None

    if args.write_dataset_full:
        full_path = output_dir / "dataset_full.json"
        write_json_array(full_path, rows)
    else:
        full_path = None

    total = len(rows)
    print(f"Bitext multi-mode dataset build complete (mode={mode})")
    print(f"  HF:     {args.hf_dataset} (split={args.hf_split}) -> {len(bitext_rows)} rows")
    if not args.bitext_only:
        print(f"  Synth:  {synthetic_dir} ({len(json_files)} files) -> {len(synth_rows)} no_issue rows")
    else:
        print("  Synth:  (skipped --bitext-only)")
    print(f"  Rows:   {total} after dedup ({rows_before_dedup} before, {n_dup} dupes removed)")
    print(f"  Split:  train={len(train_rows)}, eval={len(eval_rows)}, test={len(test_rows)}")
    if mode != "binary":
        print(f"  Labels: {len(all_labels)} unique -> label2id.json")
    else:
        print("  Binary: 0=no_issue, 1=issue")
    print(f"  Skipped: {sum(skip_reasons.values())} {dict(skip_reasons) if skip_reasons else ''}")
    print(f"  Wrote: {train_path}")
    print(f"  Wrote: {eval_path}")
    print(f"  Wrote: {test_path}")
    print(f"  Wrote: {stats_path}")
    if label2id_path:
        print(f"  Wrote: {label2id_path}")
    if full_path:
        print(f"  Wrote: {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
