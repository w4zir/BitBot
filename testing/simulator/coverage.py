from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.agent.procedures import load_blueprints
from backend.db.intents_repo import get_intents_for_category
from testing.simulator.config import KnownGapsFileConfig, SeedConfig


@dataclass
class CoverageReport:
    total_intents: int
    covered: int
    known_gaps: int
    unexpected_gaps: int
    rows: list[dict[str, str | int]]

    def to_dict(self) -> dict[str, int]:
        return {
            "total_intents": self.total_intents,
            "covered": self.covered,
            "known_gaps": self.known_gaps,
            "unexpected_gaps": self.unexpected_gaps,
        }


def build_coverage_report(
    seeds: list[SeedConfig],
    gaps_file: Path,
) -> CoverageReport:
    seed_pairs: dict[tuple[str, str], int] = {}
    for seed in seeds:
        key = (seed.category.strip().lower(), seed.intent.strip().lower())
        seed_pairs[key] = seed_pairs.get(key, 0) + 1

    known_gap_pairs = _read_known_gaps(gaps_file)

    supported_pairs = _load_supported_pairs()
    rows: list[dict[str, str | int]] = []
    covered = 0
    known_gaps = 0
    unexpected_gaps = 0

    for category, intent in sorted(supported_pairs):
        count = seed_pairs.get((category, intent), 0)
        if count > 0:
            status = "covered"
            covered += 1
        elif (category, intent) in known_gap_pairs:
            status = "known_gap"
            known_gaps += 1
        else:
            status = "gap"
            unexpected_gaps += 1
        rows.append(
            {
                "category": category,
                "intent": intent,
                "seeds": count,
                "status": status,
            }
        )

    return CoverageReport(
        total_intents=len(supported_pairs),
        covered=covered,
        known_gaps=known_gaps,
        unexpected_gaps=unexpected_gaps,
        rows=rows,
    )


def render_coverage_table(report: CoverageReport) -> str:
    lines = [
        "Category Coverage Report",
        "========================",
        f"{'category':<20} {'intent':<32} {'seeds':<6} status",
        f"{'-' * 20} {'-' * 32} {'-' * 6} {'-' * 10}",
    ]
    for row in report.rows:
        status = str(row["status"])
        if status == "covered":
            status = "OK"
        elif status == "known_gap":
            status = "KNOWN GAP"
        else:
            status = "GAP"
        lines.append(
            f"{row['category']:<20} {row['intent']:<32} {row['seeds']:<6} {status}"
        )
    lines.append("")
    lines.append(
        f"Coverage: {report.covered}/{report.total_intents} intents covered "
        f"({(100.0 * report.covered / report.total_intents) if report.total_intents else 0:.1f}%)"
    )
    lines.append(f"Known gaps: {report.known_gaps}")
    lines.append(f"Unexpected gaps: {report.unexpected_gaps}")
    return "\n".join(lines)


def _load_supported_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    blueprints = load_blueprints()
    for bp in blueprints.values():
        category = bp.category.strip().lower()
        intent = bp.intent.strip().lower()
        if category and intent:
            pairs.add((category, intent))

    categories = {cat for cat, _ in pairs}
    for category in categories:
        for intent in get_intents_for_category(category):
            pair = (category, intent.strip().lower())
            if pair[1]:
                pairs.add(pair)
    return pairs


def _read_known_gaps(gaps_file: Path) -> set[tuple[str, str]]:
    if not gaps_file.exists():
        return set()
    raw = yaml.safe_load(gaps_file.read_text(encoding="utf-8")) or {}
    parsed = KnownGapsFileConfig.model_validate(raw)
    return {
        (gap.category.strip().lower(), gap.intent.strip().lower())
        for gap in parsed.known_gaps
        if gap.category and gap.intent
    }
