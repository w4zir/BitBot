from __future__ import annotations

from backend.agent import procedures


def test_blueprints_load() -> None:
    procedures.load_blueprints.cache_clear()
    data = procedures.load_blueprints()
    assert "order_cancel" in data
    assert "get_refund" in data
    assert "change_shipping_address" in data
    assert "product_info" in data
    assert "product_price" in data
    assert "product_availability" in data
    assert "order_status" in data
    assert "payment_issue" in data
    assert "check_payment_methods" in data
    assert "payment_track_refund" in data
    assert "check_invoice" in data
    assert "subscription_status" in data
    assert "unsubscribe" in data
    assert "contact_human_agent" in data
    assert "delivery_period" in data
    assert "complaint" in data


def test_category_intents_are_loaded() -> None:
    procedures.load_blueprints.cache_clear()
    order_intents = procedures.get_category_intents("order")
    assert any(bp.intent == "cancel_order" for bp in order_intents)
    refund_intents = procedures.get_category_intents("refund")
    assert any(bp.intent == "get_refund" for bp in refund_intents)


def test_track_order_no_longer_maps_to_order_status() -> None:
    procedures.load_blueprints.cache_clear()
    bp = procedures.get_blueprint_with_fallback_chain("order", "track_order")
    assert bp is None


def test_blueprints_validate() -> None:
    procedures.load_blueprints.cache_clear()
    errors = procedures.validate_blueprints()
    assert errors == []
