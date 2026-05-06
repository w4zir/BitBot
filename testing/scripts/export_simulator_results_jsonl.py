from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be a .json file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not files:
            raise ValueError(f"No .json files found in directory: {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


def _extract_rows_from_result(result_obj: dict) -> Iterable[dict[str, str]]:
    scenarios = result_obj.get("scenarios")
    if not isinstance(scenarios, list):
        return

    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue

        category = str(scenario.get("category") or "").strip()
        intent = str(scenario.get("intent") or "").strip()
        if not category or not intent:
            continue

        trace = scenario.get("trace")
        if not isinstance(trace, list):
            continue

        for turn in trace:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("user_message") or "").strip()
            if not text:
                continue
            yield {"text": text, "category": category, "intent": intent}


def export_jsonl(input_path: Path, output_path: Path, dedupe: bool) -> tuple[int, int]:
    files = _iter_input_files(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_duplicates = 0
    seen: set[tuple[str, str, str]] = set()

    with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
        for src in files:
            with src.open("r", encoding="utf-8") as in_f:
                payload = json.load(in_f)
            if not isinstance(payload, dict):
                continue

            for row in _extract_rows_from_result(payload):
                key = (row["text"], row["category"], row["intent"])
                if dedupe and key in seen:
                    skipped_duplicates += 1
                    continue
                if dedupe:
                    seen.add(key)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    return written, skipped_duplicates


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export simulator result JSON file(s) into JSONL rows with "
            "text=user_message and gold category/intent from each scenario."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a simulator result .json file or a directory containing .json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSONL file path.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate rows by exact (text, category, intent).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        written, skipped_duplicates = export_jsonl(
            input_path=input_path,
            output_path=output_path,
            dedupe=bool(args.dedupe),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2

    dedupe_msg = f", duplicates_skipped={skipped_duplicates}" if args.dedupe else ""
    print(f"Export complete: rows_written={written}{dedupe_msg}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
