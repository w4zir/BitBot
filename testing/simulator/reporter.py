from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from testing.simulator.coverage import CoverageReport
from testing.simulator.evaluators.llm_judge import LlmJudgeResult
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.trace import ConversationTrace

ARTIFACT_SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "agent_debug_trace"

_SENSITIVE_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {"authorization", "api_key", "apikey", "password", "secret", "token", "bearer"}
)
_SAFE_TOKEN_KEYS: frozenset[str] = frozenset(
    {"token_usage", "total_tokens", "input_tokens", "output_tokens", "cache_tokens"}
)

_ASSISTANT_METADATA_KEYS: tuple[str, ...] = (
    "category",
    "intent",
    "problem_to_solve",
    "procedure_id",
    "confidence",
    "outcome_status",
    "eligibility_ok",
    "validation_ok",
    "validation_missing",
    "validation_wait_count",
    "validation_wait_limit",
    "failure_reasons",
    "tool_error",
    "step_error",
    "pending_human_action",
    "escalation_decision",
)


def _is_sensitive_key(key: str) -> bool:
    lower = key.strip().lower().replace("-", "_")
    if lower in _SAFE_TOKEN_KEYS or lower.endswith("_tokens"):
        return False
    if lower in _SENSITIVE_KEY_FRAGMENTS:
        return True
    if lower == "token" or lower.endswith("_token"):
        return True
    return any(part in lower for part in _SENSITIVE_KEY_FRAGMENTS if part != "token")


def redact_sensitive_for_console(value: Any) -> Any:
    """Return a deep copy of JSON-like structures with obvious secret keys redacted."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            sk = str(k)
            if _is_sensitive_key(sk):
                out[sk] = "<redacted>"
            else:
                out[sk] = redact_sensitive_for_console(v)
        return out
    if isinstance(value, list):
        return [redact_sensitive_for_console(item) for item in value]
    return value


def _pretty_json(obj: Any) -> str:
    safe = redact_sensitive_for_console(obj)
    return json.dumps(safe, indent=2, ensure_ascii=False, default=str)


def _issue_label(index: int, total_planned: int | None) -> str:
    if total_planned is None:
        return f"Issue {index}"
    return f"Issue {index}/{total_planned}"


def _issue_separator() -> str:
    return "-" * 72


def _resolve_git_sha() -> str | None:
    env_sha = os.getenv("GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _synthesized_nodes_from_stage_metadata(stage_metadata: dict[str, Any] | None) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for node_name, raw_meta in dict(stage_metadata or {}).items():
        if not isinstance(node_name, str) or not isinstance(raw_meta, dict):
            continue
        steps = raw_meta.get("steps")
        if not isinstance(steps, list):
            steps = [
                {
                    "step_id": str(raw_meta.get("step_id") or ""),
                    "step_type": str(raw_meta.get("step_type") or "node_operation"),
                    "timestamp": raw_meta.get("ts"),
                    "state": dict(raw_meta.get("state_context") or {}),
                    "context": {"details": {k: v for k, v in raw_meta.items() if k not in {"steps", "state_context"}}},
                }
            ]
        nodes[node_name] = {"steps": [item for item in steps if isinstance(item, dict)]}
    return nodes


def _resolve_nodes(turn: Any) -> dict[str, Any]:
    trace = dict(getattr(turn, "agent_trace", {}) or {})
    nodes = trace.get("nodes") if isinstance(trace.get("nodes"), dict) else {}
    if nodes:
        return dict(nodes)
    raw_meta = _raw_assistant_metadata_from_turn(turn)
    agent_trace = raw_meta.get("agent_trace")
    if isinstance(agent_trace, dict):
        nested_nodes = agent_trace.get("nodes")
        if isinstance(nested_nodes, dict) and nested_nodes:
            return dict(nested_nodes)
    return _synthesized_nodes_from_stage_metadata(getattr(turn, "stage_metadata", {}) or {})


def _raw_assistant_metadata_from_turn(turn: Any) -> dict[str, Any]:
    payload = getattr(turn, "response_payload", None)
    if isinstance(payload, dict):
        meta = payload.get("assistant_metadata")
        if isinstance(meta, dict):
            return dict(meta)
    return {}


def _select_assistant_metadata(meta: dict[str, Any], turn: Any) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in _ASSISTANT_METADATA_KEYS:
        if key in meta:
            selected[key] = meta[key]
    if "validation_missing" not in selected:
        missing = getattr(turn, "validation_missing", None)
        if missing:
            selected["validation_missing"] = list(missing)
    if "eligibility_ok" not in selected:
        eligibility = getattr(turn, "eligibility_ok", None)
        if eligibility is not None:
            selected["eligibility_ok"] = eligibility
    return selected


def _prune_llm_call(llm_call: Any) -> dict[str, Any] | None:
    if not isinstance(llm_call, dict):
        return None
    pruned = {
        key: llm_call[key]
        for key in ("provider", "model", "parsed_output", "attempts", "error")
        if key in llm_call
    }
    return pruned or None


def _prune_node_step(step: dict[str, Any], *, step_no: int) -> dict[str, Any]:
    pruned: dict[str, Any] = {
        "step_no": step_no,
        "step_id": step.get("step_id"),
        "step_type": step.get("step_type"),
        "timestamp": step.get("timestamp"),
        "details": step.get("details") if isinstance(step.get("details"), dict) else {},
    }
    llm_call = _prune_llm_call(step.get("llm_call"))
    if llm_call:
        pruned["llm_call"] = llm_call
    tool_call = step.get("tool_call")
    if isinstance(tool_call, dict) and tool_call:
        pruned["tool_call"] = dict(tool_call)
    return pruned


def _prune_nodes(nodes: dict[str, Any]) -> dict[str, Any]:
    pruned: dict[str, Any] = {}
    for node_name, node_data in nodes.items():
        if not isinstance(node_data, dict):
            continue
        steps = node_data.get("steps")
        if not isinstance(steps, list):
            continue
        pruned_steps = [
            _prune_node_step(step, step_no=index)
            for index, step in enumerate(steps, start=1)
            if isinstance(step, dict)
        ]
        if pruned_steps:
            pruned[node_name] = {"steps": pruned_steps}
    return pruned


def _format_token_usage(turn: Any) -> dict[str, Any]:
    return {
        "input_tokens": getattr(turn, "input_tokens", None),
        "output_tokens": getattr(turn, "output_tokens", None),
        "cache_tokens": getattr(turn, "cache_tokens", None),
        "total_tokens": getattr(turn, "total_tokens", None),
    }


def _compact_request_payload(turn: Any) -> dict[str, Any]:
    payload = getattr(turn, "request_payload", None)
    if not isinstance(payload, dict):
        return {
            "text": str(getattr(turn, "user_message", "") or ""),
            "full_flow": True,
            "session_id": None,
        }
    return {
        "text": str(payload.get("text") or getattr(turn, "user_message", "") or ""),
        "full_flow": bool(payload.get("full_flow", True)),
        "session_id": payload.get("session_id"),
    }


def _compact_response_payload(turn: Any, assistant_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "assistant_reply": str(getattr(turn, "agent_response", "") or ""),
        "assistant_metadata": assistant_metadata,
    }


def _policy_check_results(turn: Any, raw_meta: dict[str, Any]) -> list[Any]:
    checks = raw_meta.get("policy_check_results")
    if isinstance(checks, list) and checks:
        return checks
    context_data = getattr(turn, "context_data", None)
    if isinstance(context_data, dict):
        ctx_checks = context_data.get("policy_check_results")
        if isinstance(ctx_checks, list):
            return ctx_checks
    return []


def _format_turn_trace(turn: Any) -> dict[str, Any]:
    raw_meta = _raw_assistant_metadata_from_turn(turn)
    assistant_metadata = _select_assistant_metadata(raw_meta, turn)
    nodes = _prune_nodes(_resolve_nodes(turn))
    context_data = getattr(turn, "context_data", None)
    context_summary = getattr(turn, "context_summary", None)
    policy_constraints = getattr(turn, "policy_constraints", None)
    output_validation = getattr(turn, "output_validation", None)
    agent_state = getattr(turn, "agent_state", None)
    policy_checks = _policy_check_results(turn, raw_meta)

    trace: dict[str, Any] = {
        "turn_number": int(getattr(turn, "turn_number", 0) or 0),
        "request_started_at": getattr(turn, "request_started_at", None),
        "response_received_at": getattr(turn, "response_received_at", None),
        "latency_ms": float(getattr(turn, "latency_ms", 0.0) or 0.0),
        "user_message": str(getattr(turn, "user_message", "") or ""),
        "agent_response": str(getattr(turn, "agent_response", "") or ""),
        "outcome_status": str(getattr(turn, "outcome_status", "") or ""),
        "category": getattr(turn, "category", None) or assistant_metadata.get("category"),
        "intent": str(getattr(turn, "intent", "") or assistant_metadata.get("intent") or ""),
        "procedure_id": str(getattr(turn, "procedure_id", "") or assistant_metadata.get("procedure_id") or ""),
        "request_payload": _compact_request_payload(turn),
        "response_payload": _compact_response_payload(turn, assistant_metadata),
        "assistant_metadata": assistant_metadata,
        "token_usage": _format_token_usage(turn),
        "nodes": nodes,
    }
    if isinstance(agent_state, dict) and agent_state:
        trace["agent_state"] = agent_state
    if isinstance(context_data, dict) and context_data:
        trace["context_data"] = context_data
    if isinstance(context_summary, dict) and context_summary:
        trace["context_summary"] = context_summary
    if isinstance(policy_constraints, dict) and policy_constraints:
        trace["policy_constraints"] = policy_constraints
    if policy_checks:
        trace["policy_check_results"] = policy_checks
    if isinstance(output_validation, dict) and output_validation:
        trace["output_validation"] = output_validation
    return trace


def _turn_like_from_exchange(
    *,
    turn_number: int,
    request_payload: dict[str, Any] | None,
    response_payload: dict[str, Any] | None,
) -> SimpleNamespace:
    req = request_payload or {}
    resp = response_payload or {}
    meta = resp.get("assistant_metadata") if isinstance(resp.get("assistant_metadata"), dict) else {}
    agent_trace = meta.get("agent_trace") if isinstance(meta.get("agent_trace"), dict) else {}
    return SimpleNamespace(
        turn_number=turn_number,
        user_message=str(req.get("text") or ""),
        agent_response=str(resp.get("assistant_reply") or ""),
        outcome_status=str(meta.get("outcome_status") or ""),
        procedure_id=str(meta.get("procedure_id") or ""),
        validation_missing=list(meta.get("validation_missing") or []),
        eligibility_ok=meta.get("eligibility_ok") if isinstance(meta.get("eligibility_ok"), bool) else None,
        policy_constraints=meta.get("policy_constraints") if isinstance(meta.get("policy_constraints"), dict) else None,
        context_data=meta.get("context_data") if isinstance(meta.get("context_data"), dict) else None,
        confidence=meta.get("confidence"),
        category=meta.get("category"),
        intent=meta.get("intent"),
        agent_state=meta.get("agent_state") if isinstance(meta.get("agent_state"), dict) else None,
        stage_metadata=meta.get("stage_metadata") if isinstance(meta.get("stage_metadata"), dict) else None,
        agent_trace=agent_trace,
        output_validation=meta.get("output_validation") if isinstance(meta.get("output_validation"), dict) else None,
        context_summary=meta.get("context_summary") if isinstance(meta.get("context_summary"), dict) else None,
        validation_wait_count=meta.get("validation_wait_count"),
        validation_wait_limit=meta.get("validation_wait_limit"),
        request_started_at=None,
        response_received_at=None,
        request_payload={
            "text": req.get("text"),
            "full_flow": req.get("full_flow", True),
            "session_id": req.get("session_id"),
        },
        response_payload=resp,
        input_tokens=None,
        output_tokens=None,
        cache_tokens=None,
        total_tokens=None,
        latency_ms=0.0,
    )


def _derive_outcome_status_reason(
    trace: ConversationTrace,
    structural: StructuralResult,
    policy: PolicyResult,
) -> str:
    if structural.failures:
        return str(structural.failures[0])
    if policy.failures:
        return str(policy.failures[0])
    expected = trace.scenario.get("expected_outcome")
    if expected and trace.final_outcome_status != expected:
        return (
            f"Expected outcome '{expected}' but got '{trace.final_outcome_status}'."
        )
    return f"Terminated by {trace.terminated_by} with outcome {trace.final_outcome_status}."


def _extract_procedure_snapshot(turns: list[Any]) -> dict[str, Any] | None:
    for turn in reversed(turns):
        nodes = _resolve_nodes(turn)
        fetch_procedure = nodes.get("fetch_procedure") or {}
        for step in fetch_procedure.get("steps", []):
            if not isinstance(step, dict):
                continue
            details = step.get("details") if isinstance(step.get("details"), dict) else {}
            procedure_id = details.get("procedure_id") or getattr(turn, "procedure_id", None)
            if procedure_id:
                snapshot: dict[str, Any] = {"procedure_id": str(procedure_id)}
                if details.get("procedure_path"):
                    snapshot["procedure_path"] = details["procedure_path"]
                if details.get("procedure_version"):
                    snapshot["procedure_version"] = details["procedure_version"]
                return snapshot
        procedure_id = getattr(turn, "procedure_id", None)
        if procedure_id:
            return {"procedure_id": str(procedure_id)}
    return None


def _extract_policy_snapshot(turns: list[Any]) -> dict[str, Any] | None:
    for turn in reversed(turns):
        nodes = _resolve_nodes(turn)
        policy_load = nodes.get("policy_load") or {}
        for step in policy_load.get("steps", []):
            if not isinstance(step, dict):
                continue
            details = step.get("details") if isinstance(step.get("details"), dict) else {}
            policy_path = details.get("policy_constraints_path")
            if policy_path:
                snapshot: dict[str, Any] = {"policy_path": str(policy_path)}
                if details.get("policy_schema_version"):
                    snapshot["policy_schema_version"] = details["policy_schema_version"]
                return snapshot
    return None


def _entity_id_from_scenario(scenario: dict[str, Any]) -> Any:
    entity = scenario.get("entity") if isinstance(scenario.get("entity"), dict) else {}
    return (
        entity.get("order_id")
        or entity.get("user_id")
        or entity.get("account_email")
    )


def _format_scenario(
    *,
    trace: ConversationTrace,
    structural: StructuralResult,
    policy: PolicyResult,
    llm_judge: LlmJudgeResult | None,
    scenario_no: int,
) -> dict[str, Any]:
    scenario = trace.scenario
    scenario_key = str(scenario.get("run_scenario_id") or scenario.get("seed_id") or "")
    procedure_snapshot = _extract_procedure_snapshot(trace.turns)
    policy_snapshot = _extract_policy_snapshot(trace.turns)
    entity = scenario.get("entity") if isinstance(scenario.get("entity"), dict) else {}

    formatted: dict[str, Any] = {
        "run_scenario_id": scenario_key,
        "scenario_no": scenario_no,
        "seed_id": str(scenario.get("seed_id") or ""),
        "entity_id": _entity_id_from_scenario(scenario),
        "session_id": trace.session_id,
        "persona_id": scenario.get("persona_id"),
        "category": scenario.get("category"),
        "intent": scenario.get("intent"),
        "expected_outcome": scenario.get("expected_outcome"),
        "final_outcome_status": trace.final_outcome_status,
        "outcome_status_reason": _derive_outcome_status_reason(trace, structural, policy),
        "terminated_by": trace.terminated_by,
        "seed_snapshot": scenario.get("seed_snapshot") or {},
        "persona_snapshot": scenario.get("persona_snapshot") or {},
        "entity_snapshot": dict(entity),
        "evaluation": {
            "structural": asdict(structural),
            "policy": asdict(policy),
            "llm_judge": asdict(llm_judge) if llm_judge is not None else None,
            "regression": None,
        },
        "metrics": {
            "turns": len(trace.turns),
            "total_latency_ms": trace.total_latency_ms,
            "total_tokens_used": trace.total_tokens_used,
        },
        "trace": [_format_turn_trace(turn) for turn in trace.turns],
    }
    if procedure_snapshot:
        formatted["procedure_snapshot"] = procedure_snapshot
    if policy_snapshot:
        formatted["policy_snapshot"] = policy_snapshot
    return formatted


def _format_skipped_scenario(item: dict[str, Any]) -> dict[str, Any]:
    scenario_key = str(item.get("run_scenario_id") or item.get("scenario_key") or "")
    formatted: dict[str, Any] = {
        "run_scenario_id": scenario_key,
        "seed_id": str(item.get("seed_id") or ""),
        "reason": str(item.get("reason") or item.get("error") or ""),
    }
    for key, value in item.items():
        if key in {"run_scenario_id", "scenario_key", "seed_id", "reason", "error"}:
            continue
        formatted[key] = value
    return formatted


class SimulatorConsoleReporter:
    """Incremental stdout reporter for simulator runs (scenario progress + LLM/agent exchanges)."""

    def __init__(self, *, file: Any = None) -> None:
        self._file = file

    def _print(self, text: str = "") -> None:
        print(text, file=self._file)

    def start_scenario(
        self,
        *,
        index: int,
        total_planned: int | None,
        scenario_key: str,
        seed: Any,
        persona_id: str = "",
    ) -> None:
        seed_id = str(getattr(seed, "seed_id", "") or "")
        pid = str(persona_id or getattr(seed, "persona_id", "") or "")
        intent = str(getattr(seed, "intent", "") or "")
        self._print("")
        self._print(_issue_separator())
        self._print(f"{_issue_label(index, total_planned)}: {scenario_key}")
        self._print(
            f"Seed: {seed_id} | Persona: {pid} | Intent: {intent}",
        )

    def skip_scenario(
        self,
        *,
        index: int,
        total_planned: int | None,
        scenario_key: str,
        error: str,
    ) -> None:
        self._print("")
        self._print(f"{_issue_label(index, total_planned)} SKIP: {scenario_key}")
        self._print(f"  {error}")

    def finish_scenario(
        self,
        *,
        index: int,
        total_planned: int | None,
        scenario_key: str,
        trace: ConversationTrace,
        structural: StructuralResult,
        policy: PolicyResult,
        llm_judge: LlmJudgeResult | None,
    ) -> None:
        judge_ok = llm_judge.passed if llm_judge is not None else True
        scenario_passed = structural.passed and policy.passed and judge_ok
        status = "PASS" if scenario_passed else "FAIL"
        self._print("")
        self._print(
            f"{_issue_label(index, total_planned)} complete: {status} "
            f"scenario_key={scenario_key} outcome={trace.final_outcome_status} "
            f"turns={len(trace.turns)}",
        )

    def persona_exchange(
        self,
        *,
        mode: str,
        turn_number: int,
        attempt: int,
        messages: list[dict[str, Any]],
        raw_response: str,
    ) -> None:
        """Persona LLM traffic is intentionally not printed (terminal stays minimal)."""
        _ = (mode, turn_number, attempt, messages, raw_response)

    def agent_exchange(
        self,
        *,
        turn_number: int,
        request_payload: dict[str, Any] | None,
        response_payload: dict[str, Any] | None,
    ) -> None:
        turn_like = _turn_like_from_exchange(
            turn_number=turn_number,
            request_payload=request_payload,
            response_payload=response_payload,
        )
        turn_json = _format_turn_trace(turn_like)
        self._print("")
        self._print(f"[Agent Turn] {turn_number}")
        self._print(_pretty_json(turn_json))


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
    run_error: dict[str, Any] | None = None,
    run_loop_status: str | None = None,
    environment_config: dict[str, Any] | None = None,
    git_sha: str | None = None,
) -> Path:
    completed_at = datetime.now(timezone.utc)
    skipped = [_format_skipped_scenario(item) for item in list(skipped_scenarios or [])]
    scenarios: list[dict[str, Any]] = []
    per_category: dict[str, dict[str, Any]] = {}

    passed = 0
    structural_failures = 0
    policy_failures = 0
    llm_judge_failures = 0

    for scenario_no, trace in enumerate(traces, start=1):
        scenario_key = str(trace.scenario.get("run_scenario_id") or trace.scenario.get("seed_id") or "")
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
            _format_scenario(
                trace=trace,
                structural=structural,
                policy=policy,
                llm_judge=llm_judge,
                scenario_no=scenario_no,
            )
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

    resolved_git_sha = git_sha if git_sha is not None else _resolve_git_sha()
    environment: dict[str, Any] = {
        "agent_url": agent_url,
        "db_snapshot": db_snapshot,
        "config": dict(environment_config or {}),
    }
    if resolved_git_sha:
        environment["git_sha"] = resolved_git_sha

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": run_id,
        "suite": suite_path,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "run_loop_status": run_loop_status or "completed",
        "run_error": run_error,
        "environment": environment,
        "coverage": coverage.to_dict(),
        "summary": {
            "total_scenarios": len(traces),
            "passed": passed,
            "failed": len(traces) - passed,
            "skipped": len(skipped),
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
            key = str(item.get("scenario_key") or item.get("run_scenario_id") or item.get("seed_id") or "")
            err = str(item.get("reason") or item.get("error") or "")
            lines.append(f"- {key}: {err}")
    return "\n".join(lines)
