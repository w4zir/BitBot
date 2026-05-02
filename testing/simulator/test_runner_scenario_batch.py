from __future__ import annotations

import pytest

from testing.simulator.config import (
    DbFilterConfig,
    DefaultsConfig,
    PersonaConfig,
    ScenarioRunConfig,
    SeedConfig,
    SuiteConfig,
)
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.persona import PersonaGenerationError
from testing.simulator.runner import _run_scenario_batch
from testing.simulator.trace import ConversationTrace, TurnRecord


def _seed(seed_id: str) -> SeedConfig:
    return SeedConfig(
        seed_id=seed_id,
        category="order",
        intent="cancel_order",
        persona_id="p1",
        db_filter=DbFilterConfig(entity_type="order"),
    )


def _scenario_for_seed(seed_id: str) -> ScenarioInstance:
    return ScenarioInstance(
        seed_id=seed_id,
        category="order",
        intent="cancel_order",
        difficulty="easy",
        persona_id="p1",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="order_cancel",
        adversarial_flags=[],
        entity={
            "entity_type": "order",
            "order_id": "ORD-123",
            "status": "processing",
        },
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )


def _persona_cfg() -> PersonaConfig:
    return PersonaConfig(
        persona_id="p1",
        display_name="P1",
        vocabulary="simple",
        patience="medium",
        cooperation_level="cooperative",
        escalation_tendency="low",
        typical_message_length="medium",
        traits=["test"],
    )


def _minimal_trace(scenario: ScenarioInstance) -> ConversationTrace:
    return ConversationTrace(
        scenario=scenario.to_dict(),
        session_id="",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="cancel ORD-123",
                agent_response="Order ORD-123 has been cancelled.",
                outcome_status="resolved",
                procedure_id="order_cancel",
                category="order",
                intent="cancel_order",
            )
        ],
        final_outcome_status="resolved",
        terminated_by="resolved",
        total_latency_ms=1.0,
    )


def test_run_scenario_batch_skips_persona_error_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "testing.simulator.runner.evaluate_structural",
        lambda *_a, **_k: StructuralResult(passed=True, checks={}, failures=[]),
    )
    monkeypatch.setattr(
        "testing.simulator.runner.evaluate_policy",
        lambda *_a, **_k: PolicyResult(passed=True, checks={}, failures=[]),
    )

    class FakeHydrator:
        def hydrate(self, seed: SeedConfig) -> ScenarioInstance:
            return _scenario_for_seed(seed.seed_id)

    class FakeDriver:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, scenario: ScenarioInstance, persona) -> ConversationTrace:  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                raise PersonaGenerationError("simulated empty message")
            return _minimal_trace(scenario)

    class FakePersistence:
        def __init__(self) -> None:
            self.skipped: list[dict] = []
            self.recorded: list[dict] = []

        def record_skipped_scenario(self, **kwargs) -> None:
            self.skipped.append(kwargs)

        def record_scenario(self, **kwargs) -> None:
            self.recorded.append(kwargs)

    suite = SuiteConfig(
        run_id="r1",
        scenarios=[ScenarioRunConfig(seed_id="s1"), ScenarioRunConfig(seed_id="s2")],
        defaults=DefaultsConfig(eval_targets=["structural"]),
    )
    seed1, seed2 = _seed("s1"), _seed("s2")
    personas = {"p1": _persona_cfg()}
    persistence = FakePersistence()

    traces, _structural, _policy, _llm_judge, skipped, interrupted = _run_scenario_batch(
        indexed_plan=[
            (1, (ScenarioRunConfig(seed_id="s1"), seed1)),
            (2, (ScenarioRunConfig(seed_id="s2"), seed2)),
        ],
        hydrator=FakeHydrator(),
        driver=FakeDriver(),
        personas=personas,
        suite=suite,
        persistence=persistence,  # type: ignore[arg-type]
    )

    assert not interrupted
    assert len(skipped) == 1
    assert skipped[0]["seed_id"] == "s1"
    assert len(traces) == 1
    assert traces[0].scenario.get("seed_id") == "s2"
    assert len(persistence.skipped) == 1
    assert persistence.skipped[0]["error_type"] == "PersonaGenerationError"
    assert len(persistence.recorded) == 1
