from __future__ import annotations

from backend.agent import procedures


def test_blueprints_load() -> None:
    procedures.load_blueprints.cache_clear()
    data = procedures.load_blueprints()
    assert "no_issue_chat" in data
    assert "order_cancel" in data


def test_intent_inference_scoped_to_category() -> None:
    procedures.load_blueprints.cache_clear()
    intent = procedures.infer_intent_from_text(category="order", text="please cancel this order")
    assert intent == "cancel_order"


def test_blueprints_validate() -> None:
    procedures.load_blueprints.cache_clear()
    errors = procedures.validate_blueprints()
    assert errors == []
