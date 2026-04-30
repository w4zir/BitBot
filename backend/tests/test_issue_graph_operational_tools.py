from __future__ import annotations

from backend.agent.issue_graph import (
    _build_agent_state_snapshot,
    _outcome_validator_node,
    _structured_executor_node,
)


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


def test_order_cancel_confirm_cancelled_reply_is_deterministic() -> None:
    state = {
        "procedure_id": "order_cancel",
        "text": "cancel my order",
        "messages": [{"role": "user", "content": "cancel ORD-2019"}],
        "todo_list": [{"id": "confirm_cancelled", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {
            "order_id_extracted": "ORD-2019",
            "order_data": {"order_id": "ORD-2019", "status": "processing"},
            "order_status": "cancelled",
            "cancel_succeeded": True,
            "cancel_reason": "",
        },
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "cancelled successfully" in reply
    assert "unable to process the cancellation" not in reply
    assert "could not cancel" not in reply


def test_order_cancel_failure_reply_uses_cancel_reason() -> None:
    state = {
        "procedure_id": "order_cancel",
        "text": "cancel my order",
        "messages": [{"role": "user", "content": "cancel ORD-2019"}],
        "todo_list": [{"id": "cancellation_not_allowed", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {
            "order_id_extracted": "ORD-2019",
            "cancel_succeeded": False,
            "cancel_reason": "order_delivered",
        },
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "could not cancel order ord-2019" in reply
    assert "order delivered" in reply


def test_order_cancel_fallback_success_copy_does_not_say_already() -> None:
    state = {
        "procedure_id": "order_cancel",
        "text": "cancel my order",
        "messages": [{"role": "user", "content": "cancel ORD-2014"}],
        "todo_list": [{"id": "cancellation_not_allowed", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {
            "order_id_extracted": "ORD-2014",
            "cancel_succeeded": True,
            "cancel_reason": "",
        },
        "assistant_metadata": {},
    }

    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "has been cancelled successfully" in reply
    assert "already been cancelled successfully" not in reply


def test_legacy_validate_required_data_step_is_not_supported() -> None:
    state = {
        "todo_list": [{"id": "legacy_validate", "type": "validate_required_data"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert "unsupported procedure step" in str(out.get("final_response") or "").lower()
    assert "step_error" in (out.get("assistant_metadata") or {})


def test_outcome_validator_adds_output_validation_for_cancel(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_order_status",
        lambda order_id: {"order_id": order_id, "status": "cancelled"},
    )
    state = {
        "intent": "cancel_order",
        "context_data": {"order_id_extracted": "ORD-12345", "cancel_succeeded": True},
        "assistant_metadata": {},
        "validation_ok": True,
        "todo_list": [],
        "current_step_index": 0,
    }
    out = _outcome_validator_node(state)
    checks = (out.get("output_validation") or {}).get("checks") or {}
    assert "order_cancel_db_verification" in checks
    assert checks["order_cancel_db_verification"]["valid"] is True


def test_order_status_before_after_are_db_backed(monkeypatch) -> None:
    statuses: dict[str, str] = {"ORD-2010": "processing"}

    def _get_order_status(order_id: str):
        return {"order_id": order_id, "status": statuses.get(order_id, "")}

    def _cancel_order(order_id: str):
        statuses[order_id] = "cancelled"
        return {"ok": True, "order_id": order_id, "status": "cancelled"}

    monkeypatch.setattr("backend.agent.issue_graph.get_order_status", _get_order_status)
    monkeypatch.setattr("backend.agent.issue_graph.cancel_order_record", _cancel_order)

    state = {
        "intent": "cancel_order",
        "procedure_id": "order_cancel",
        "text": "cancel ORD-2010",
        "messages": [{"role": "user", "content": "cancel ORD-2010"}],
        "todo_list": [
            {"id": "lookup_order", "type": "tool_call", "tool": "check_order_status"},
            {"id": "cancel_order", "type": "tool_call", "tool": "cancel_order"},
        ],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
        "validation_ok": True,
    }

    out1 = _structured_executor_node(state)
    assert out1["context_data"]["order_status_before"] == "processing"

    out2 = _structured_executor_node(out1)
    out3 = _outcome_validator_node({**out2, "outcome_status": "resolved"})
    snap = _build_agent_state_snapshot(out3)
    assert snap["order_status_before"] == "processing"
    assert snap["order_status_after"] == "cancelled"


def test_stage_metadata_has_state_context_without_policy_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_order_status",
        lambda order_id: {"order_id": order_id, "status": "processing", "total_amount": 12.0},
    )
    state = {
        "text": "track ORD-4444",
        "messages": [{"role": "user", "content": "track ORD-4444"}],
        "category": "order",
        "intent": "order_status",
        "procedure_id": "order_status",
        "todo_list": [{"id": "lookup_order", "type": "tool_call", "tool": "check_order_status"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
        "policy_constraints": {"policy_doc_names": ["Order Policy"], "raw_chunks": ["secret body"]},
        "validation_missing": [],
        "validation_wait_count": 0,
        "validation_wait_limit": 5,
    }

    out = _structured_executor_node(state)
    stage = (out.get("stage_metadata") or {}).get("structured_executor") or {}
    state_context = stage.get("state_context") or {}
    assert isinstance(state_context, dict)
    assert state_context.get("context_data", {}).get("order_status_before") == "processing"
    assert state_context.get("policy", {}).get("policy_doc_names") == ["Order Policy"]
