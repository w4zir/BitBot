#!/usr/bin/env python3
"""
Combine Hugging Face Bitext (instruction + intent) with synthetic no-issue JSON
into a multiclass dataset with string labels.

Writes canonical artifacts only:
  - dataset_full.json (array of {text, label})
  - dataset_stats.json
  - label2id.json (string label -> integer id for training)

Use build_bitext_training_dataset.py to split into train/eval/test JSONL with integer labels.

Bitext rows: text = instruction, label = intent (from HF dataset).
Synthetic: only samples with is_issue=false from no_issue_*.json, label = "no_issue".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

DEFAULT_HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DEFAULT_HF_SPLIT = "train"
_REPO_TRAINING = Path(__file__).resolve().parent.parent
# Committed sample synthetic no-issue JSON (see training/data/samples/synthetic_no_issue/)
DEFAULT_SYNTHETIC_DIR = _REPO_TRAINING / "data" / "samples" / "synthetic_no_issue"
DEFAULT_SYNTHETIC_GLOB = "no_issue_*.json"
DEFAULT_OUTPUT_DIR = _REPO_TRAINING / "data" / "bitext"


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

    label = "no_issue"
    row: dict[str, Any] = {
        "text": msg.strip(),
        "label": label,
        "source_file": sample.get("_source_file"),
    }
    return row, None


def load_bitext_rows(
    dataset_name: str,
    split: str,
    skip_reasons: Counter[str],
) -> list[dict[str, Any]]:
    """Load Bitext from Hugging Face; instruction -> text, intent -> label."""
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

        if instruction is None or not isinstance(instruction, str) or not instruction.strip():
            skip_reasons["bitext_missing_or_empty_instruction"] += 1
            continue
        if intent is None or not isinstance(intent, str) or not intent.strip():
            skip_reasons["bitext_missing_or_empty_intent"] += 1
            continue

        rows.append(
            {
                "text": instruction.strip(),
                "label": intent.strip(),
                "bitext_index": idx,
            }
        )
    return rows


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


def slim_row_for_json(r: dict[str, Any]) -> dict[str, str]:
    """Public schema: text + label only."""
    return {"text": r["text"], "label": str(r["label"])}


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = [slim_row_for_json(r) for r in rows]
    path.write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_label2id(all_labels_sorted: list[str]) -> dict[str, int]:
    """Deterministic mapping: sorted label -> 0..n-1."""
    return {lab: i for i, lab in enumerate(all_labels_sorted)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Bitext + no_issue dataset_full.json, dataset_stats.json, label2id.json."
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
        "--synthetic-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_DIR,
        help="Directory containing no_issue_*.json files",
    )
    parser.add_argument(
        "--synthetic-glob",
        type=str,
        default=DEFAULT_SYNTHETIC_GLOB,
        help="Glob pattern for synthetic no-issue JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for dataset_full.json, dataset_stats.json, label2id.json",
    )
    parser.add_argument(
        "--bitext-only",
        action="store_true",
        help="Load only Hugging Face Bitext (no synthetic no_issue JSON files).",
    )
    args = parser.parse_args()

    synthetic_dir: Path = args.synthetic_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files: list[Path] = []
    synth_rows: list[dict[str, Any]] = []
    root_issues = 0

    if not args.bitext_only:
        if not synthetic_dir.is_dir():
            print(f"Error: synthetic directory does not exist: {synthetic_dir}", file=sys.stderr)
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

    rows: list[dict[str, Any]] = bitext_rows + synth_rows
    rows_before_dedup = len(rows)
    rows, n_dup = dedupe_by_text(rows)

    if not rows:
        print("Error: no rows after loading and validation", file=sys.stderr)
        return 1

    all_labels = sorted({str(r["label"]) for r in rows})
    label2id = build_label2id(all_labels)

    stats: dict[str, Any] = {
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "bitext_only": bool(args.bitext_only),
        "synthetic_dir": str(synthetic_dir.resolve()) if not args.bitext_only else None,
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
        "num_labels": len(all_labels),
        "class_distribution_all": label_distribution_str(rows),
        "note": "Use build_bitext_training_dataset.py for train/eval/test JSONL splits.",
    }

    full_path = output_dir / "dataset_full.json"
    stats_path = output_dir / "dataset_stats.json"
    label2id_path = output_dir / "label2id.json"

    write_json_array(full_path, rows)
    label2id_path.write_text(
        json.dumps(label2id, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total = len(rows)
    print("Bitext dataset build complete" + ("" if args.bitext_only else " (+ synthetic no_issue)"))
    print(f"  HF:     {args.hf_dataset} (split={args.hf_split}) -> {len(bitext_rows)} rows")
    if not args.bitext_only:
        print(f"  Synth:  {synthetic_dir} ({len(json_files)} files) -> {len(synth_rows)} no_issue rows")
    else:
        print("  Synth:  (skipped --bitext-only)")
    print(f"  Rows:   {total} after dedup ({rows_before_dedup} before, {n_dup} dupes removed)")
    print(f"  Labels: {len(all_labels)} unique -> label2id.json")
    print(f"  Skipped: {sum(skip_reasons.values())} {dict(skip_reasons) if skip_reasons else ''}")
    print(f"  Wrote: {full_path}")
    print(f"  Wrote: {label2id_path}")
    print(f"  Wrote: {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
