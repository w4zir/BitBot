from __future__ import annotations

import json

import pytest

from testing.simulator.config import PersonaConfig
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.persona import PersonaEngine


def _persona(*, cooperation_level: str = "cooperative", patience: str = "medium") -> PersonaConfig:
    return PersonaConfig(
        persona_id="p1",
        display_name="P1",
        vocabulary="simple",
        patience=patience,
        cooperation_level=cooperation_level,  # type: ignore[arg-type]
        escalation_tendency="low",
        typical_message_length="medium",
        traits=["test"],
    )


def _scenario(
    *,
    multi_issue: bool = False,
    adversarial_flags: list[str] | None = None,
) -> ScenarioInstance:
    return ScenarioInstance(
        seed_id="seed1",
        category="order",
        intent="cancel_order",
        difficulty="medium",
        persona_id="p1",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="order_cancel",
        adversarial_flags=adversarial_flags or [],
        entity={
            "entity_type": "order",
            "order_id": "ORD-123",
            "status": "processing",
        },
        secondary_entity={"entity_type": "order", "order_id": "ORD-999"} if multi_issue else None,
        multi_issue=multi_issue,
        secondary_category="order" if multi_issue else None,
        secondary_intent="order_status" if multi_issue else None,
    )


def test_opening_requires_grounded_order_id(monkeypatch: pytest.MonkeyPatch) -> None:
    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr(
        "testing.simulator.persona.chat_completion",
        lambda **_: '{"message":"I need to cancel my order.", "stop": false}',
    )

    with pytest.raises(RuntimeError, match="opening must include hydrated order_id"):
        persona.generate_opening()


def test_missing_order_id_response_must_include_hydrated_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona = PersonaEngine(
        persona=_persona(cooperation_level="cooperative"),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr(
        "testing.simulator.persona.chat_completion",
        lambda **_: '{"message":"Sure, here are the details.", "stop": false}',
    )

    with pytest.raises(RuntimeError, match="must include hydrated order_id"):
        persona.generate_response(
            agent_message="Please provide your order id.",
            turn_number=1,
            conversation_history=[],
            agent_metadata={"validation_missing": ["order_id"], "outcome_status": "needs_more_info"},
        )


def test_resistant_persona_can_challenge_first_missing_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona = PersonaEngine(
        persona=_persona(cooperation_level="resistant"),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr(
        "testing.simulator.persona.chat_completion",
        lambda **_: '{"message":"Why do you need that first?", "stop": false}',
    )

    message = persona.generate_response(
        agent_message="Please provide your order id.",
        turn_number=1,
        conversation_history=[],
        agent_metadata={"validation_missing": ["order_id"], "outcome_status": "needs_more_info"},
    )
    assert message == "Why do you need that first?"


def test_force_directives_for_secondary_issue_and_human_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def _fake_chat_completion(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        captured.append(payload)
        return '{"message":"Please connect me to a human. Also, issue with order ORD-999.", "stop": false}'

    persona = PersonaEngine(
        persona=_persona(patience="low"),
        scenario=_scenario(multi_issue=True),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr("testing.simulator.persona.chat_completion", _fake_chat_completion)

    persona.generate_response(
        agent_message="Can I help with anything else?",
        turn_number=3,
        conversation_history=[],
        agent_metadata={"outcome_status": "in_progress"},
    )
    assert captured
    directives = captured[0]["directives"]
    assert any("secondary order id 'ORD-999'" in item for item in directives)
    assert any("transfer to a human agent" in item for item in directives)
