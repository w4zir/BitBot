from __future__ import annotations

import io

import pytest

from testing.simulator.runner import _planned_scenario_count
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.reporter import (
    SimulatorConsoleReporter,
    redact_sensitive_for_console,
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
        seed=type("S", (), {"seed_id": "seed_a", "persona_id": "p1", "intent": "cancel"})(),
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
        seed=type("S", (), {"seed_id": "x", "persona_id": "p", "intent": "i"})(),
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


def test_agent_exchange_shows_text_only_and_plain_reply() -> None:
    buf = io.StringIO()
    rep = SimulatorConsoleReporter(file=buf)
    rep.agent_exchange(
        turn_number=1,
        request_payload={"text": "hi", "token": "abc", "full_flow": True},
        response_payload={"assistant_reply": "ok", "session_id": "s"},
    )
    text = buf.getvalue()
    assert "[Agent Request] turn 1" in text
    assert '"text": "hi"' in text
    assert "token" not in text
    assert "session_id" not in text
    assert "[Agent Response] turn 1" in text
    assert text.strip().endswith("ok")


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


