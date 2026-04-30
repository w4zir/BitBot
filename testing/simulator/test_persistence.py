from __future__ import annotations

from dataclasses import dataclass

from testing.simulator.evaluators.llm_judge import LlmJudgeResult
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.persistence import SimulatorPersistence
from testing.simulator.trace import ConversationTrace, TurnRecord


@dataclass
class _FakeCursor:
    statements: list[str]
    _fetchone_values: list[tuple]

    def execute(self, sql, params=None):  # noqa: ANN001
        self.statements.append(sql)

    def fetchone(self):  # noqa: ANN001
        if self._fetchone_values:
            return self._fetchone_values.pop(0)
        return ("00000000-0000-4000-8000-000000000999",)

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):  # noqa: ANN001
        return self._cursor

    def __enter__(self):  # noqa: ANN001
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


def _trace() -> ConversationTrace:
    turn = TurnRecord(
        turn_number=1,
        user_message="hello",
        agent_response="hi",
        outcome_status="resolved",
        procedure_id="p1",
        latency_ms=20.0,
    )
    return ConversationTrace(
        scenario={"seed_id": "seed1", "category": "order", "intent": "cancel_order", "expected_outcome": "resolved"},
        session_id="",
        turns=[turn],
        final_outcome_status="resolved",
        terminated_by="resolved",
        total_latency_ms=20.0,
    )


def test_persistence_writes_simulation_tables(monkeypatch) -> None:
    statements: list[str] = []
    cursor = _FakeCursor(statements=statements, _fetchone_values=[("run-uuid",), ("scenario-uuid",), ("turn-uuid",)])

    def _fake_get_connection():
        return _FakeConn(cursor)

    monkeypatch.setattr("testing.simulator.persistence.postgres_configured", lambda: True)
    monkeypatch.setattr("testing.simulator.persistence.get_connection", _fake_get_connection)

    store = SimulatorPersistence(enabled=True)
    store.start_run(
        run_id="run1",
        suite_name="smoke.yaml",
        db_snapshot="live",
        baseline_ref=None,
        run_metadata={"randomize": False},
    )
    store.record_coverage({"total_intents": 2, "covered": 2, "known_gaps": 0, "unexpected_gaps": 0})
    store.record_scenario(
        trace=_trace(),
        structural=StructuralResult(passed=True, checks={}, failures=[]),
        policy=PolicyResult(passed=True, checks={}, failures=[]),
        llm_judge=LlmJudgeResult(
            passed=True,
            scores={"tone": 4.0},
            rationales={"tone": "ok"},
            thresholds={"tone": 3.0},
            failures=[],
            provider="ollama",
            model="llama3.2",
        ),
    )
    store.complete_run(summary={"exit_code": 0}, status="completed")

    sql_joined = "\n".join(statements)
    assert "INSERT INTO simulation_runs" in sql_joined
    assert "INSERT INTO simulation_scenarios" in sql_joined
    assert "INSERT INTO simulation_turns" in sql_joined
    assert "INSERT INTO simulation_messages" in sql_joined
    assert "INSERT INTO simulation_evaluations" in sql_joined
    assert "INSERT INTO simulation_llm_judgements" in sql_joined
    assert "INSERT INTO simulation_training_examples" in sql_joined
