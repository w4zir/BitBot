from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testing.simulator.coverage import CoverageReport
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.trace import ConversationTrace


def write_run_artifact(
    *,
    run_id: str,
    suite_path: str,
    db_snapshot: str,
    agent_url: str,
    coverage: CoverageReport,
    traces: list[ConversationTrace],
    structural_results: dict[str, StructuralResult],
    policy_results: dict[str, PolicyResult],
    output_dir: Path,
    started_at: datetime,
) -> Path:
    completed_at = datetime.now(timezone.utc)
    scenarios: list[dict[str, Any]] = []
    per_category: dict[str, dict[str, Any]] = {}

    passed = 0
    structural_failures = 0
    policy_failures = 0

    for trace in traces:
        seed_id = str(trace.scenario.get("seed_id") or "")
        structural = structural_results[seed_id]
        policy = policy_results[seed_id]
        if structural.passed and policy.passed:
            passed += 1
        if not structural.passed:
            structural_failures += 1
        if not policy.passed:
            policy_failures += 1

        category = str(trace.scenario.get("category") or "unknown")
        cat = per_category.setdefault(
            category,
            {
                "count": 0,
                "resolved": 0,
                "escalated": 0,
                "total_turns": 0,
                "total_latency_ms": 0.0,
            },
        )
        cat["count"] += 1
        cat["resolved"] += 1 if trace.final_outcome_status == "resolved" else 0
        cat["escalated"] += 1 if trace.final_outcome_status == "pending_escalation" else 0
        cat["total_turns"] += len(trace.turns)
        cat["total_latency_ms"] += trace.total_latency_ms

        scenarios.append(
            {
                "seed_id": seed_id,
                "entity_id": (
                    trace.scenario.get("entity", {}).get("order_id")
                    or trace.scenario.get("entity", {}).get("user_id")
                    or trace.scenario.get("entity", {}).get("account_email")
                ),
                "persona_id": trace.scenario.get("persona_id"),
                "turns": len(trace.turns),
                "final_outcome_status": trace.final_outcome_status,
                "expected_outcome": trace.scenario.get("expected_outcome"),
                "structural": asdict(structural),
                "policy": asdict(policy),
                "llm_judge": None,
                "regression": None,
                "trace": [asdict(turn) for turn in trace.turns],
            }
        )

    per_category_summary: dict[str, dict[str, Any]] = {}
    for category, cat in per_category.items():
        count = cat["count"] or 1
        per_category_summary[category] = {
            "resolution_rate": cat["resolved"] / count,
            "escalation_rate": cat["escalated"] / count,
            "avg_turns": cat["total_turns"] / count,
            "avg_latency_ms": cat["total_latency_ms"] / count,
        }

    artifact = {
        "run_id": run_id,
        "suite": suite_path,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "db_snapshot": db_snapshot,
        "agent_url": agent_url,
        "coverage": coverage.to_dict(),
        "summary": {
            "total_scenarios": len(traces),
            "passed": passed,
            "failed": len(traces) - passed,
            "structural_failures": structural_failures,
            "policy_failures": policy_failures,
            "llm_judge_failures": 0,
            "regressions": 0,
        },
        "per_category": per_category_summary,
        "scenarios": scenarios,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{completed_at.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def render_console_summary(
    traces: list[ConversationTrace],
    structural_results: dict[str, StructuralResult],
    policy_results: dict[str, PolicyResult],
) -> str:
    total = len(traces)
    if total == 0:
        return "No scenarios were executed."
    passed = 0
    lines = ["Run Summary", "==========="]
    for trace in traces:
        seed_id = str(trace.scenario.get("seed_id") or "")
        structural = structural_results[seed_id]
        policy = policy_results[seed_id]
        scenario_passed = structural.passed and policy.passed
        if scenario_passed:
            passed += 1
        lines.append(
            f"- {seed_id}: {'PASS' if scenario_passed else 'FAIL'} "
            f"(outcome={trace.final_outcome_status}, turns={len(trace.turns)})"
        )
        if not structural.passed:
            lines.extend([f"    structural: {failure}" for failure in structural.failures])
        if not policy.passed:
            lines.extend([f"    policy: {failure}" for failure in policy.failures])
    lines.append("")
    lines.append(f"Passed: {passed}/{total}")
    return "\n".join(lines)
