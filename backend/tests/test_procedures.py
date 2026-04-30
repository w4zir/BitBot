from __future__ import annotations

from backend.agent import procedures


def test_canonical_intent_track_order_maps_to_order_status() -> None:
    assert procedures.canonical_procedure_intent("order", "track_order") == "order_status"
    procedures.load_blueprints.cache_clear()
    bp = procedures.get_blueprint_with_fallback_chain("order", "track_order")
    assert bp is not None
    assert bp.id == "order_status"
    assert bp.intent == "order_status"


def test_blueprints_load() -> None:
    procedures.load_blueprints.cache_clear()
    data = procedures.load_blueprints()
    assert "order_cancel" in data
    assert "get_refund" in data
    assert "change_order_shipping_address" in data
    assert "get_product_info" in data
    assert "order_status" in data


def test_category_intents_are_loaded() -> None:
    procedures.load_blueprints.cache_clear()
    order_intents = procedures.get_category_intents("order")
    assert any(bp.intent == "cancel_order" for bp in order_intents)
    refund_intents = procedures.get_category_intents("refund")
    assert any(bp.intent == "get_refund" for bp in refund_intents)


def test_blueprints_validate() -> None:
    procedures.load_blueprints.cache_clear()
    errors = procedures.validate_blueprints()
    assert errors == []
