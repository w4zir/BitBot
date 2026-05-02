from __future__ import annotations

from typing import Any

from testing.simulator.driver import ConversationDriver
from testing.simulator.hydrator import ScenarioInstance


def _minimal_scenario() -> ScenarioInstance:
    return ScenarioInstance(
        seed_id="s1",
        category="order",
        intent="cancel_order",
        difficulty="easy",
        persona_id="p1",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="order_cancel",
        adversarial_flags=[],
        entity={"entity_type": "order", "order_id": "ORD-1", "status": "processing"},
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )


class _OneTurnPersona:
    def generate_opening(self) -> str:
        return "Cancel ORD-1 please"

    def generate_response(
        self,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict[str, str]],
        agent_metadata: dict[str, Any],
    ) -> str | None:
        return None


def test_conversation_driver_emits_agent_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    class Sink:
        def agent_exchange(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    driver = ConversationDriver(
        agent_url="http://127.0.0.1:9/classify",
        max_turns=3,
        event_sink=Sink(),
    )

    def fake_post(_self: ConversationDriver, text: str, session_id: str | None) -> dict[str, Any]:
        assert "ORD-1" in text
        return {
            "session_id": "sess-1",
            "assistant_reply": "Done.",
            "assistant_metadata": {"outcome_status": "resolved"},
        }

    monkeypatch.setattr(ConversationDriver, "_post_classify", fake_post)

    trace = driver.run(_minimal_scenario(), _OneTurnPersona())
    assert len(trace.turns) == 1
    assert len(captured) == 1
    assert captured[0]["turn_number"] == 1
    req = captured[0]["request_payload"]
    assert isinstance(req, dict)
    assert req.get("text") == "Cancel ORD-1 please"
    resp = captured[0]["response_payload"]
    assert resp.get("assistant_reply") == "Done."


def test_conversation_driver_no_sink_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = ConversationDriver(agent_url="http://127.0.0.1:9/classify", max_turns=1, event_sink=None)

    monkeypatch.setattr(
        ConversationDriver,
        "_post_classify",
        lambda *_a, **_k: {
            "assistant_reply": "ok",
            "assistant_metadata": {"outcome_status": "resolved"},
        },
    )
    trace = driver.run(_minimal_scenario(), _OneTurnPersona())
    assert len(trace.turns) == 1
