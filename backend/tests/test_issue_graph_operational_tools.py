from __future__ import annotations

from backend.agent.issue_graph import _structured_executor_node


def test_cancel_order_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.cancel_order_record",
        lambda order_id: {"ok": True, "order_id": order_id, "status": "cancelled"},
    )
    state = {
        "text": "cancel ORD-12345",
        "messages": [{"role": "user", "content": "please cancel ORD-12345"}],
        "todo_list": [{"id": "cancel", "type": "tool_call", "tool": "cancel_order"}],
        "current_step_index": 0,
        "context_data": {"order_id_extracted": "ORD-12345"},
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["cancel_succeeded"] is True
    assert out["context_data"]["order_status"] == "cancelled"


def test_create_refund_request_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.create_refund_request",
        lambda order_id, reason: {
            "ok": True,
            "refund_id": 99,
            "order_id": order_id,
            "decision": "pending",
        },
    )
    state = {
        "text": "refund ORD-12345",
        "messages": [{"role": "user", "content": "refund this item"}],
        "todo_list": [{"id": "refund", "type": "tool_call", "tool": "create_refund_request"}],
        "current_step_index": 0,
        "context_data": {"order_id_extracted": "ORD-12345", "refund_reason": "defective"},
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["refund_request_created"] is True
    assert out["context_data"]["refund_request_id"] == 99


def test_update_shipping_address_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.update_shipping_address_record",
        lambda order_id, new_address: {
            "ok": True,
            "order_id": order_id,
            "shipping_address": {"line": new_address},
        },
    )
    state = {
        "text": "change address for ORD-12345",
        "messages": [{"role": "user", "content": "new address is 101 New St"}],
        "todo_list": [
            {"id": "change_address", "type": "tool_call", "tool": "update_shipping_address"}
        ],
        "current_step_index": 0,
        "context_data": {"order_id_extracted": "ORD-12345", "new_address": "101 New St"},
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["shipping_address_updated"] is True
    assert out["context_data"]["shipping_address"]["line"] == "101 New St"
