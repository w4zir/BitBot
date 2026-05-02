from __future__ import annotations

import json

import pytest

from testing.simulator.config import PersonaConfig
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.persona import PersonaEngine, PersonaGenerationError


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

    with pytest.raises(PersonaGenerationError, match="opening must include hydrated order_id"):
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

    with pytest.raises(PersonaGenerationError, match="must include hydrated order_id"):
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


def test_opening_includes_style_and_anti_template_directives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    def _fake_chat_completion(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        captured.append(payload)
        return '{"message":"Please check order ORD-123 status for me.", "stop": false}'

    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr("testing.simulator.persona.chat_completion", _fake_chat_completion)

    opening = persona.generate_opening()
    assert opening
    assert captured
    payload = captured[0]
    directives = payload["directives"]
    assert any("opening style profile" in item for item in directives)
    assert any("Avoid overused support openers" in item for item in directives)
    assert any("order id 'ORD-123'" in item for item in directives)
    assert isinstance(payload.get("style_profile"), dict)
    assert payload["style_profile"].get("opening_style")


def test_opening_retries_when_template_like_starter_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    calls = {"count": 0}

    def _fake_chat_completion(**kwargs):
        calls["count"] += 1
        payload = json.loads(kwargs["messages"][1]["content"])
        captured.append(payload)
        if calls["count"] == 1:
            return '{"message":"Hi there! I was hoping you could help me check order ORD-123.", "stop": false}'
        return '{"message":"Need an update on ORD-123 status.", "stop": false}'

    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr("testing.simulator.persona.chat_completion", _fake_chat_completion)

    opening = persona.generate_opening()
    assert opening == "Need an update on ORD-123 status."
    assert calls["count"] == 2
    assert len(captured) == 2
    retry_directives = captured[1]["directives"]
    assert any("previous attempt sounded formulaic" in item for item in retry_directives)


def test_generation_options_are_forwarded_to_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: list[dict] = []

    def _fake_chat_completion(**kwargs):
        captured_kwargs.append(kwargs)
        return '{"message":"Please check ORD-123.", "stop": false}'

    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
        llm_temperature=0.7,
        llm_top_p=0.9,
        llm_repeat_penalty=1.1,
    )
    monkeypatch.setattr("testing.simulator.persona.chat_completion", _fake_chat_completion)

    persona.generate_opening()
    assert captured_kwargs
    kwargs = captured_kwargs[0]
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["top_p"] == pytest.approx(0.9)
    assert kwargs["repeat_penalty"] == pytest.approx(1.1)


def test_empty_persona_message_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _fake_chat_completion(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"message":"", "stop": false}'
        return '{"message":"Please cancel order ORD-123.", "stop": false}'

    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    monkeypatch.setattr("testing.simulator.persona.chat_completion", _fake_chat_completion)

    opening = persona.generate_opening()
    assert opening == "Please cancel order ORD-123."
    assert calls["n"] == 2


def test_empty_persona_message_twice_raises_persona_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "testing.simulator.persona.chat_completion",
        lambda **_kwargs: '{"message":"   ", "stop": false}',
    )
    persona = PersonaEngine(
        persona=_persona(),
        scenario=_scenario(),
        llm_provider="ollama",
        llm_model="llama3.2",
        llm_timeout_seconds=30.0,
    )
    with pytest.raises(PersonaGenerationError, match="empty 'message'"):
        persona.generate_opening()
