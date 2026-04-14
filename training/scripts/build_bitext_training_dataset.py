#!/usr/bin/env python3
"""
Split dataset_full.json into train/eval/test JSONL with integer labels.

Reads label2id.json (string label -> int) produced by create_bitext_dataset.py.
Each output line: {"text": "...", "label": <int>} for use with sequence classifiers.

Default split ratios: 0.7 train, 0.15 eval, 0.15 test (stratified by string label).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATASET_FULL = Path(__file__).resolve().parent.parent / "data" / "bitext" / "dataset_full.json"
DEFAULT_LABEL2ID = Path(__file__).resolve().parent.parent / "data" / "bitext" / "label2id.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bitext"
DEFAULT_SEED = 42


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


def label_distribution_str(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(str(r["label"]) for r in rows)
    return {k: c[k] for k in sorted(c.keys())}


def label_distribution_int(rows: list[dict[str, Any]], label2id: dict[str, int]) -> dict[str, int]:
    c: Counter[int] = Counter()
    for r in rows:
        lab = str(r["label"])
        c[label2id[lab]] += 1
    return {str(k): c[k] for k in sorted(c.keys())}


def load_dataset_full(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected JSON array")
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: item {i} is not an object")
        if "text" not in obj or "label" not in obj:
            raise ValueError(f"{path}: item {i} missing 'text' or 'label'")
        if not isinstance(obj["text"], str):
            raise ValueError(f"{path}: item {i} 'text' must be string")
        out.append({"text": obj["text"], "label": str(obj["label"])})
    return out


def load_label2id(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected JSON object mapping label -> id")
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            k = str(k)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{path}: label {k!r} -> id must be int, got {type(v).__name__}")
        out[k] = int(v)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]], label2id: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            lab = str(r["label"])
            line = json.dumps(
                {"text": r["text"], "label": label2id[lab]},
                ensure_ascii=False,
            )
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build train/eval/test JSONL from dataset_full.json + label2id.json.",
    )
    parser.add_argument(
        "--dataset-full",
        type=Path,
        default=DEFAULT_DATASET_FULL,
        help="Path to dataset_full.json",
    )
    parser.add_argument(
        "--label2id",
        type=Path,
        default=DEFAULT_LABEL2ID,
        help="Path to label2id.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.jsonl, eval.jsonl, test.jsonl, dataset_stats.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target fraction per label for training (stratified)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Target fraction per label for eval; test gets remainder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling and split",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_full.resolve()
    label2id_path = args.label2id.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_path.is_file():
        print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
        return 1
    if not label2id_path.is_file():
        print(f"Error: label2id file not found: {label2id_path}", file=sys.stderr)
        return 1

    rows = load_dataset_full(dataset_path)
    label2id = load_label2id(label2id_path)

    if not rows:
        print("Error: no rows in dataset_full.json", file=sys.stderr)
        return 1

    missing = sorted({str(r["label"]) for r in rows} - set(label2id.keys()))
    if missing:
        print(
            f"Error: labels in dataset not found in label2id.json: {missing[:20]}"
            + (" ..." if len(missing) > 20 else ""),
            file=sys.stderr,
        )
        return 1

    extra = sorted(set(label2id.keys()) - {str(r["label"]) for r in rows})
    if extra:
        # Warn but continue — mapping may include reserved labels for future data
        print(
            f"Warning: label2id.json has labels not present in dataset ({len(extra)}); continuing.",
            file=sys.stderr,
        )

    train_rows, eval_rows, test_rows = stratified_train_eval_test_split_by_label(
        rows, args.train_ratio, args.eval_ratio, args.seed
    )
    test_ratio_effective = 1.0 - args.train_ratio - args.eval_ratio

    stats: dict[str, Any] = {
        "dataset_full": str(dataset_path),
        "label2id_file": str(label2id_path),
        "output_dir": str(output_dir),
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "test_ratio_target": test_ratio_effective,
        "seed": args.seed,
        "num_rows": len(rows),
        "num_labels_in_mapping": len(label2id),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "test_size": len(test_rows),
        "class_distribution_all": label_distribution_str(rows),
        "class_distribution_train": label_distribution_str(train_rows),
        "class_distribution_eval": label_distribution_str(eval_rows),
        "class_distribution_test": label_distribution_str(test_rows),
        "label_id_distribution_train": label_distribution_int(train_rows, label2id),
        "label_id_distribution_eval": label_distribution_int(eval_rows, label2id),
        "label_id_distribution_test": label_distribution_int(test_rows, label2id),
    }

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    stats_path = output_dir / "dataset_split_stats.json"

    write_jsonl(train_path, train_rows, label2id)
    write_jsonl(eval_path, eval_rows, label2id)
    write_jsonl(test_path, test_rows, label2id)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tr, ev, te = len(train_rows), len(eval_rows), len(test_rows)
    denom = tr + ev + te
    ratio_str = f"{tr / denom:.1%} / {ev / denom:.1%} / {te / denom:.1%}" if denom else "n/a"

    print("Bitext training JSONL build complete")
    print(f"  Rows: {len(rows)}  Split: train={tr}, eval={ev}, test={te} ({ratio_str})")
    print(f"  Wrote: {train_path}")
    print(f"  Wrote: {eval_path}")
    print(f"  Wrote: {test_path}")
    print(f"  Wrote: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
