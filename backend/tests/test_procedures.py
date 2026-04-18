from __future__ import annotations

from backend.agent import procedures


def test_blueprints_load() -> None:
    procedures.load_blueprints.cache_clear()
    data = procedures.load_blueprints()
    assert "no_issue_chat" in data
    assert "order_cancel" in data
    assert "get_refund" in data
    assert "change_order_shipping_address" in data
    assert "get_product_info" in data


def test_intent_inference_scoped_to_category() -> None:
    procedures.load_blueprints.cache_clear()
    intent = procedures.infer_intent_from_text(category="order", text="please cancel this order")
    assert intent == "cancel_order"


def test_intent_inference_refund_policy_vs_request() -> None:
    procedures.load_blueprints.cache_clear()
    assert (
        procedures.infer_intent_from_text(
            category="refund",
            text="I need a refund my item arrived damaged",
        )
        == "get_refund"
    )


def test_intent_inference_shipping_and_product() -> None:
    procedures.load_blueprints.cache_clear()
    assert (
        procedures.infer_intent_from_text(
            category="shipping",
            text="please change my shipping address",
        )
        == "change_order_shipping_address"
    )
    assert (
        procedures.infer_intent_from_text(
            category="product",
            text="what is the price and is it in stock",
        )
        == "get_product_info"
    )


def test_blueprints_validate() -> None:
    procedures.load_blueprints.cache_clear()
    errors = procedures.validate_blueprints()
    assert errors == []
