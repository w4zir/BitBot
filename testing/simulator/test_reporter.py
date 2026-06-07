from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from testing.simulator.coverage import CoverageReport
from testing.simulator.runner import _planned_scenario_count
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.reporter import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_TYPE,
    SimulatorConsoleReporter,
    redact_sensitive_for_console,
    write_run_artifact,
)
from testing.simulator.trace import ConversationTrace, TurnRecord


def test_redact_sensitive_for_console_nested() -> None:
    data = {
        "ok": "visible",
        "nested": {"api_key": "secret123", "Authorization": "Bearer x"},
    }
    red = redact_sensitive_for_console(data)
    assert red["ok"] == "visible"
    assert red["nested"]["api_key"] == "<redacted>"
    assert red["nested"]["Authorization"] == "<redacted>"


def test_issue_label_in_output() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    rep.start_scenario(
        index=2,
        total_planned=5,
        scenario_key="seed_a#2",
        seed=type("S", (), {"seed_id": "seed_a", "intent": "cancel"})(),
        persona_id="p1",
    )
    out = buf.getvalue()
    assert "Issue 2/5" in out
    assert "seed_a#2" in out


def test_issue_label_forever_style() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    rep.start_scenario(
        index=7,
        total_planned=None,
        scenario_key="x#7",
        seed=type("S", (), {"seed_id": "x", "intent": "i"})(),
        persona_id="p",
    )
    out = buf.getvalue()
    assert "Issue 7: x#7" in out
    assert "Issue 7/7" not in out


def test_finish_scenario_line() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    trace = ConversationTrace(
        scenario={"seed_id": "s1", "run_scenario_id": "s1#1"},
        session_id="sess",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="u",
                agent_response="a",
                outcome_status="resolved",
                procedure_id="p1",
            )
        ],
        final_outcome_status="resolved",
        terminated_by="resolved",
        total_latency_ms=1.0,
    )
    rep.finish_scenario(
        index=1,
        total_planned=1,
        scenario_key="s1#1",
        trace=trace,
        structural=StructuralResult(passed=True, checks={}, failures=[]),
        policy=PolicyResult(passed=True, checks={}, failures=[]),
        llm_judge=None,
    )
    text = buf.getvalue()
    assert "complete: PASS" in text
    assert "turns=1" in text


def test_agent_exchange_outputs_structured_turn_json() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    rep.agent_exchange(
        turn_number=1,
        request_payload={"text": "hi", "token": "abc", "full_flow": True},
        response_payload={
            "assistant_reply": "ok",
            "session_id": "s",
            "assistant_metadata": {
                "outcome_status": "resolved",
                "intent": "order_status",
                "procedure_id": "order_status",
                "agent_trace": {
                    "nodes": {
                        "classify_intent": {
                            "steps": [
                                {
                                    "step_id": "step_1",
                                    "step_type": "llm_call",
                                    "llm_call": {
                                        "provider": "ollama",
                                        "model": "test-model",
                                        "messages": [{"role": "system", "content": "secret prompt"}],
                                        "raw_response": '{"intent":"order_status"}',
                                        "parsed_output": {"intent": "order_status"},
                                        "attempts": 1,
                                    },
                                }
                            ]
                        }
                    }
                },
            },
        },
    )
    text = buf.getvalue()
    assert "[Agent Turn] 1" in text
    assert '"user_message": "hi"' in text
    assert '"agent_response": "ok"' in text
    assert '"outcome_status": "resolved"' in text
    assert '"intent": "order_status"' in text
    assert '"procedure_id": "order_status"' in text
    assert '"classify_intent"' in text
    assert '"request_payload"' in text
    assert '"assistant_metadata"' in text
    assert '"parsed_output"' in text
    assert "secret prompt" not in text
    assert "raw_response" not in text
    assert "messages" not in text
    assert '"token": "abc"' not in text


def _sample_trace() -> ConversationTrace:
    return ConversationTrace(
        scenario={
            "run_scenario_id": "seed_a#1",
            "seed_id": "seed_a",
            "category": "order",
            "intent": "order_status",
            "expected_outcome": "resolved",
            "persona_id": "policy_prober",
            "entity": {"entity_type": "order", "order_id": 42},
            "seed_snapshot": {"seed_id": "seed_a"},
            "persona_snapshot": {"persona_id": "policy_prober"},
        },
        session_id="sess-42",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="Where is my order?",
                agent_response="Please share your order id.",
                outcome_status="needs_more_data",
                procedure_id="order_status",
                category="order",
                intent="order_status",
                validation_missing=["order_id"],
                eligibility_ok=True,
                policy_constraints={"eligible": True, "variables": {}, "validation_results": {}},
                agent_state={"stage": "validate_required", "validation_ok": False},
                context_data={"policy_docs": []},
                request_started_at="2026-06-06T09:00:00+00:00",
                response_received_at="2026-06-06T09:00:01+00:00",
                request_payload={"text": "Where is my order?", "full_flow": True, "session_id": None},
                response_payload={
                    "assistant_reply": "Please share your order id.",
                    "assistant_metadata": {
                        "outcome_status": "needs_more_data",
                        "intent": "order_status",
                        "procedure_id": "order_status",
                        "validation_ok": False,
                        "validation_missing": ["order_id"],
                        "policy_check_results": [{"check": "eligibility", "passed": True}],
                        "agent_trace": {
                            "nodes": {
                                "fetch_procedure": {
                                    "steps": [
                                        {
                                            "step_id": "fetch_procedure_1",
                                            "step_type": "node_operation",
                                            "details": {"procedure_id": "order_status"},
                                        }
                                    ]
                                },
                                "policy_load": {
                                    "steps": [
                                        {
                                            "step_id": "policy_load_1",
                                            "step_type": "node_operation",
                                            "details": {
                                                "policy_constraints_path": "order/order_status.yaml",
                                                "policy_schema_version": "1.0",
                                            },
                                        }
                                    ]
                                },
                            }
                        },
                    },
                },
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                latency_ms=1200.5,
            )
        ],
        final_outcome_status="needs_more_data",
        terminated_by="max_turns",
        total_latency_ms=1200.5,
        total_tokens_used=15,
    )


def test_write_run_artifact_emits_template_envelope(tmp_path: Path) -> None:
    trace = _sample_trace()
    coverage = CoverageReport(
        total_intents=10,
        covered=8,
        known_gaps=1,
        unexpected_gaps=1,
        rows=[],
    )
    started_at = datetime(2026, 6, 6, 9, 0, 0, tzinfo=timezone.utc)
    artifact_path = write_run_artifact(
        run_id="test_run",
        suite_path="testing/simulator/suites/smoke.yaml",
        db_snapshot="live",
        agent_url="http://localhost:8000/classify",
        coverage=coverage,
        traces=[trace],
        structural_results={
            "seed_a#1": StructuralResult(
                passed=False,
                checks={"outcome_status_match": False},
                failures=["Expected outcome 'resolved' but got 'needs_more_data'."],
            )
        },
        policy_results={"seed_a#1": PolicyResult(passed=True, checks={}, failures=[])},
        llm_judge_results={"seed_a#1": None},
        output_dir=tmp_path,
        started_at=started_at,
        skipped_scenarios=[
            {
                "scenario_key": "seed_b#2",
                "seed_id": "seed_b",
                "error": "persona generation failed",
            }
        ],
        environment_config={
            "max_turns": 6,
            "timeout_seconds": 120.0,
            "model_provider": "ollama",
            "model": "llama3.2",
        },
        git_sha="abc123",
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert artifact["environment"]["agent_url"] == "http://localhost:8000/classify"
    assert artifact["environment"]["db_snapshot"] == "live"
    assert artifact["environment"]["git_sha"] == "abc123"
    assert artifact["environment"]["config"]["max_turns"] == 6
    assert artifact["summary"]["skipped"] == 1
    assert artifact["summary"]["scenarios_skipped"] == 1

    scenario = artifact["scenarios"][0]
    assert scenario["session_id"] == "sess-42"
    assert scenario["terminated_by"] == "max_turns"
    assert scenario["entity_snapshot"]["order_id"] == 42
    assert scenario["evaluation"]["structural"]["passed"] is False
    assert scenario["metrics"]["turns"] == 1
    assert scenario["metrics"]["total_latency_ms"] == 1200.5
    assert scenario["procedure_snapshot"]["procedure_id"] == "order_status"
    assert scenario["policy_snapshot"]["policy_path"] == "order/order_status.yaml"
    assert "Expected outcome" in scenario["outcome_status_reason"]

    turn = scenario["trace"][0]
    assert turn["latency_ms"] == 1200.5
    assert "validation_missing" not in turn
    assert turn["assistant_metadata"]["validation_missing"] == ["order_id"]
    assert turn["policy_check_results"] == [{"check": "eligibility", "passed": True}]
    assert turn["token_usage"]["total_tokens"] == 15
    assert "assistant_reply" in turn["response_payload"]
    assert "session_id" not in turn["response_payload"]
    assert "messages" not in json.dumps(turn["nodes"])
    assert "raw_response" not in json.dumps(turn["nodes"])

    skipped = artifact["skipped_scenarios"][0]
    assert skipped["run_scenario_id"] == "seed_b#2"
    assert skipped["seed_id"] == "seed_b"
    assert skipped["reason"] == "persona generation failed"


def test_persona_exchange_is_silent() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    rep.persona_exchange(
        mode="opening",
        turn_number=0,
        attempt=1,
        messages=[{"role": "system", "content": "secret"}],
        raw_response="{}",
    )
    assert buf.getvalue() == ""


def test_planned_scenario_count() -> None:
    assert _planned_scenario_count(forever=True, randomize=False, iterations=1, num_selected_scenarios=5) is None
    assert _planned_scenario_count(forever=False, randomize=True, iterations=3, num_selected_scenarios=10) == 3
    assert _planned_scenario_count(forever=False, randomize=False, iterations=2, num_selected_scenarios=4) == 8


