#!/usr/bin/env python3
"""
Split bitext dataset_full.json into train/eval/test JSONL with binary labels.

Input schema:
  - dataset_full.json as an array of {"text": "...", "label": "<string>"}

Binary mapping:
  - label == "no_issue" -> 0
  - any other label      -> 1

Output schema (JSONL):
  {"text": "...", "label": 0|1}

Default split ratios: 0.7 train, 0.15 eval, 0.15 test (stratified by binary label).
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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bitext_binary_issue"
DEFAULT_SEED = 42
NO_ISSUE_LABEL = "no_issue"


def map_label_to_binary(label: str) -> int:
    return 0 if label == NO_ISSUE_LABEL else 1


def stratified_train_eval_test_split_by_binary_label(
    rows: list[dict[str, Any]],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified 3-way split by binary label (train / eval / test)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if train_ratio + eval_ratio >= 1.0:
        raise ValueError("train_ratio + eval_ratio must be < 1 so test has a positive share")

    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[int(r["binary_label"])].append(r)

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


def original_label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(str(r["original_label"]) for r in rows)
    return {k: c[k] for k in sorted(c.keys())}


def binary_label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter(int(r["binary_label"]) for r in rows)
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

        original_label = str(obj["label"])
        out.append(
            {
                "text": obj["text"],
                "original_label": original_label,
                "binary_label": map_label_to_binary(original_label),
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(
                {"text": r["text"], "label": int(r["binary_label"])},
                ensure_ascii=False,
            )
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build binary issue train/eval/test JSONL from bitext dataset_full.json.",
    )
    parser.add_argument(
        "--dataset-full",
        type=Path,
        default=DEFAULT_DATASET_FULL,
        help="Path to bitext-style dataset_full.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.jsonl, eval.jsonl, test.jsonl, dataset_split_stats.json",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Target fraction per binary class for training (stratified)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Target fraction per binary class for eval; test gets remainder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for shuffling and split",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_full.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_path.is_file():
        print(f"Error: dataset file not found: {dataset_path}", file=sys.stderr)
        return 1

    rows = load_dataset_full(dataset_path)
    if not rows:
        print("Error: no rows in dataset_full.json", file=sys.stderr)
        return 1

    train_rows, eval_rows, test_rows = stratified_train_eval_test_split_by_binary_label(
        rows, args.train_ratio, args.eval_ratio, args.seed
    )
    test_ratio_effective = 1.0 - args.train_ratio - args.eval_ratio

    stats: dict[str, Any] = {
        "dataset_full": str(dataset_path),
        "output_dir": str(output_dir),
        "binary_mapping": {NO_ISSUE_LABEL: 0, "__other_labels__": 1},
        "train_ratio": args.train_ratio,
        "eval_ratio": args.eval_ratio,
        "test_ratio_target": test_ratio_effective,
        "seed": args.seed,
        "num_rows": len(rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "test_size": len(test_rows),
        "original_label_distribution_all": original_label_distribution(rows),
        "original_label_distribution_train": original_label_distribution(train_rows),
        "original_label_distribution_eval": original_label_distribution(eval_rows),
        "original_label_distribution_test": original_label_distribution(test_rows),
        "binary_class_distribution_all": binary_label_distribution(rows),
        "binary_class_distribution_train": binary_label_distribution(train_rows),
        "binary_class_distribution_eval": binary_label_distribution(eval_rows),
        "binary_class_distribution_test": binary_label_distribution(test_rows),
    }

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    test_path = output_dir / "test.jsonl"
    stats_path = output_dir / "dataset_split_stats.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(test_path, test_rows)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tr, ev, te = len(train_rows), len(eval_rows), len(test_rows)
    denom = tr + ev + te
    ratio_str = f"{tr / denom:.1%} / {ev / denom:.1%} / {te / denom:.1%}" if denom else "n/a"

    print("Binary issue training JSONL build complete")
    print(f"  Rows: {len(rows)}  Split: train={tr}, eval={ev}, test={te} ({ratio_str})")
    print(f"  Binary labels: 0={NO_ISSUE_LABEL}, 1=issue")
    print(f"  Wrote: {train_path}")
    print(f"  Wrote: {eval_path}")
    print(f"  Wrote: {test_path}")
    print(f"  Wrote: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
