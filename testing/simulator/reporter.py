from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from testing.simulator.coverage import CoverageReport
from testing.simulator.evaluators.llm_judge import LlmJudgeResult
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
    llm_judge_results: dict[str, LlmJudgeResult | None],
    output_dir: Path,
    started_at: datetime,
    skipped_scenarios: list[dict[str, Any]] | None = None,
) -> Path:
    completed_at = datetime.now(timezone.utc)
    skipped = list(skipped_scenarios or [])
    scenarios: list[dict[str, Any]] = []
    per_category: dict[str, dict[str, Any]] = {}

    passed = 0
    structural_failures = 0
    policy_failures = 0
    llm_judge_failures = 0

    for trace in traces:
        scenario_key = str(trace.scenario.get("run_scenario_id") or trace.scenario.get("seed_id") or "")
        seed_id = str(trace.scenario.get("seed_id") or "")
        structural = structural_results[scenario_key]
        policy = policy_results[scenario_key]
        llm_judge = llm_judge_results.get(scenario_key)
        judge_passed = llm_judge.passed if llm_judge is not None else True
        if structural.passed and policy.passed and judge_passed:
            passed += 1
        if not structural.passed:
            structural_failures += 1
        if not policy.passed:
            policy_failures += 1
        if llm_judge is not None and not llm_judge.passed:
            llm_judge_failures += 1

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
                "llm_judge": asdict(llm_judge) if llm_judge is not None else None,
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
            "llm_judge_failures": llm_judge_failures,
            "scenarios_skipped": len(skipped),
            "regressions": 0,
        },
        "per_category": per_category_summary,
        "scenarios": scenarios,
        "skipped_scenarios": skipped,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{completed_at.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def render_console_summary(
    traces: list[ConversationTrace],
    structural_results: dict[str, StructuralResult],
    policy_results: dict[str, PolicyResult],
    llm_judge_results: dict[str, LlmJudgeResult | None],
    skipped_scenarios: list[dict[str, Any]] | None = None,
) -> str:
    skipped = list(skipped_scenarios or [])
    total = len(traces)
    if total == 0 and not skipped:
        return "No scenarios were executed."
    passed = 0
    lines = ["Run Summary", "==========="]
    if total == 0:
        lines.append("(no completed scenarios)")
    for trace in traces:
        scenario_key = str(trace.scenario.get("run_scenario_id") or trace.scenario.get("seed_id") or "")
        seed_id = str(trace.scenario.get("seed_id") or "")
        structural = structural_results[scenario_key]
        policy = policy_results[scenario_key]
        llm_judge = llm_judge_results.get(scenario_key)
        scenario_passed = structural.passed and policy.passed and (
            llm_judge.passed if llm_judge is not None else True
        )
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
        if llm_judge is not None and not llm_judge.passed:
            lines.extend([f"    llm_judge: {failure}" for failure in llm_judge.failures])
    lines.append("")
    lines.append(f"Passed: {passed}/{total}")
    if skipped:
        lines.append("")
        lines.append(f"Skipped ({len(skipped)})")
        lines.append("----------")
        for item in skipped:
            key = str(item.get("scenario_key") or item.get("seed_id") or "")
            err = str(item.get("error") or "")
            lines.append(f"- {key}: {err}")
    return "\n".join(lines)
