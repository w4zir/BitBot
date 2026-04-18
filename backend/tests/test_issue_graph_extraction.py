"""Tests for order-id extraction helpers in issue_graph."""

from __future__ import annotations

from backend.agent.issue_graph import _extract_order_id_from_conversation


def test_extract_order_id_prefers_first_user_message() -> None:
    msgs = [
        {"role": "user", "content": "Refund for ORD-11111 please"},
        {"role": "user", "content": "Actually use ORD-22222 instead"},
    ]
    assert _extract_order_id_from_conversation(msgs, None) == "ORD-11111"


def test_extract_order_id_falls_back_to_text_when_messages_empty() -> None:
    assert (
        _extract_order_id_from_conversation([], "What is status of ORD-99999?")
        == "ORD-99999"
    )


def test_extract_order_id_skips_non_user_roles() -> None:
    msgs = [
        {"role": "assistant", "content": "ORD-00001 is fake"},
        {"role": "user", "content": "My order ORD-54321"},
    ]
    assert _extract_order_id_from_conversation(msgs, None) == "ORD-54321"
