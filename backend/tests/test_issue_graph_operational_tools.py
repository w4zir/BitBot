from __future__ import annotations

from backend.agent.issue_graph import (
    _build_agent_state_snapshot,
    _classify_intent_node,
    _collect_failure_reasons,
    _outcome_validator_node,
    _policy_load_node,
    _structured_executor_node,
    build_agent_trace,
)


def test_cancel_order_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.cancel_order_record",
        lambda order_id, **_kwargs: {"ok": True, "order_id": order_id, "status": "cancelled"},
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
        lambda order_id, reason, **_kwargs: {
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
        lambda order_id, new_address, **_kwargs: {
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

    def _cancel_order(order_id: str, **_kwargs):
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


def test_stage_metadata_compact_step_details(monkeypatch) -> None:
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
    steps = stage.get("steps") or []
    assert steps
    assert isinstance(steps[0].get("details"), dict)
    assert steps[0]["details"]["step_id"] == "lookup_order"
    assert steps[0]["details"]["step_type"] == "tool_call"
    assert "state_context" not in stage


def test_build_agent_trace_includes_structured_executor_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_order_status",
        lambda order_id: {"order_id": order_id, "status": "processing", "total_amount": 25.0},
    )
    state = {
        "text": "status ORD-9000",
        "messages": [{"role": "user", "content": "status ORD-9000"}],
        "category": "order",
        "intent": "order_status",
        "procedure_id": "order_status",
        "todo_list": [{"id": "lookup_order", "type": "tool_call", "tool": "check_order_status"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
        "validation_missing": [],
        "validation_wait_count": 0,
        "validation_wait_limit": 5,
    }
    out = _structured_executor_node(state)
    trace = build_agent_trace(
        agent_state=_build_agent_state_snapshot(out),
        stage_metadata=out.get("stage_metadata"),
    )
    nodes = trace.get("nodes") or {}
    structured = nodes.get("structured_executor") or {}
    steps = structured.get("steps") or []
    assert steps
    assert steps[0].get("step_id") == "lookup_order"
    assert steps[0].get("step_type") == "tool_call"
    assert isinstance(steps[0].get("details"), dict)


def test_classify_intent_stage_keeps_llm_messages(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_intent_definitions_for_category",
        lambda _category: [{"intent_name": "order_status", "description": "Track order status"}],
    )
    monkeypatch.setattr(
        "backend.agent.issue_graph.chat_completion",
        lambda **_kwargs: '{"intent":"order_status","problem_to_solve":"Track order"}',
    )
    state = {
        "category": "order",
        "confidence": 0.9,
        "messages": [{"role": "user", "content": "where is order ORD-9"}],
        "assistant_metadata": {},
        "stage_metadata": {},
    }
    out = _classify_intent_node(state)
    stage = (out.get("stage_metadata") or {}).get("classify_intent") or {}
    steps = stage.get("steps") or []
    assert steps
    llm_call = steps[0].get("llm_call") or {}
    assert llm_call.get("provider")
    assert llm_call.get("model")
    assert isinstance(llm_call.get("messages"), list)


def test_order_status_share_status_reply_is_deterministic() -> None:
    state = {
        "procedure_id": "order_status",
        "text": "status ORD-123",
        "messages": [{"role": "user", "content": "status ORD-123"}],
        "todo_list": [{"id": "share_status", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {
            "order_id_extracted": "ORD-123",
            "order_found": True,
            "order_status": "shipped",
        },
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "ord-123" in reply
    assert "shipped" in reply
    assert "could not find" not in reply


def test_order_status_not_found_reply_is_deterministic() -> None:
    state = {
        "procedure_id": "order_status",
        "text": "status ORD-999",
        "messages": [{"role": "user", "content": "status ORD-999"}],
        "todo_list": [{"id": "order_not_found", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {"order_id_extracted": "ORD-999", "order_found": False},
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "could not find" in reply


def test_order_status_not_found_step_reports_status_when_order_found() -> None:
    """If branch id is wrong but context says order was found, still report factual status."""
    state = {
        "procedure_id": "order_status",
        "text": "status ORD-1",
        "messages": [{"role": "user", "content": "status ORD-1"}],
        "todo_list": [{"id": "order_not_found", "type": "llm_response"}],
        "current_step_index": 0,
        "context_data": {
            "order_id_extracted": "ORD-1",
            "order_found": True,
            "order_status": "delivered",
        },
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    reply = str(out.get("final_response") or "").lower()
    assert out["current_step_index"] == 1
    assert "delivered" in reply
    assert "could not find" not in reply


def test_policy_load_reads_yaml_by_intent() -> None:
    state = {
        "text": "I need a refund for order ORD-8194",
        "category": "refund",
        "intent": "get_refund",
        "problem_to_solve": "Start refund process",
        "context_data": {"order_status": "delivered"},
        "policy_constraints": None,
    }

    out = _policy_load_node(state)
    context_data = out.get("context_data") or {}
    constraints = out.get("policy_constraints") or {}
    assert context_data.get("policy_constraints_path") == "refund/get_refund.yaml"
    assert context_data.get("policy_schema_version") == "1.0"
    assert context_data.get("policy_eligible") is True
    assert constraints.get("eligible") is True
    assert constraints.get("reason") == ""


def test_collect_failure_reasons_dedupes_and_skips_ok() -> None:
    state = {
        "assistant_metadata": {"tool_error": "Unknown tool 'bad_tool'"},
        "context_data": {"policy_ineligibility_reason": ""},
        "policy_constraints": {"eligible": True},
        "policy_check_results": [
            {"passed": False, "reason": "We could not find that order in our system.", "source": "procedure_gate"},
            {"passed": True, "reason": "ok", "source": "procedure_gate"},
        ],
        "output_validation": {"checks": {"x": {"valid": False, "reason": "Missing cancellation outcome in context."}}},
    }
    reasons = _collect_failure_reasons(state)
    blob = " ".join(reasons)
    assert "Unknown tool" in blob
    assert any("could not find" in r.lower() for r in reasons)
    assert any("missing cancellation" in r.lower() for r in reasons)


def test_logic_gate_failure_reason_survives_in_policy_check_results() -> None:
    state = {
        "procedure_id": "order_cancel",
        "text": "cancel",
        "messages": [{"role": "user", "content": "cancel ORD-1"}],
        "todo_list": [
            {
                "id": "branch_order_found",
                "type": "logic_gate",
                "condition": {
                    "op": "eq",
                    "field": "order_found",
                    "value": True,
                    "failure_reason": "We could not find that order in our system.",
                },
                "on_true": "confirm_cancelled",
                "on_false": "cancellation_not_allowed",
            },
            {"id": "cancellation_not_allowed", "type": "llm_response", "message": "deny"},
            {"id": "confirm_cancelled", "type": "llm_response", "message": "ok"},
        ],
        "current_step_index": 0,
        "context_data": {"order_found": False},
        "policy_constraints": {},
        "policy_check_results": [],
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    checks = out.get("policy_check_results") or []
    failed = [c for c in checks if c.get("passed") is False]
    assert failed
    assert "could not find" in str(failed[-1].get("reason") or "").lower()


def test_outcome_validator_appends_failure_summary_for_tool_error() -> None:
    state = {
        "intent": "order_status",
        "context_data": {},
        "assistant_metadata": {"tool_error": "Unknown tool 'x'"},
        "validation_ok": True,
        "todo_list": [],
        "current_step_index": 0,
        "final_response": "I could not run a required backend tool for this request.",
    }
    out = _outcome_validator_node(state)
    assert out.get("outcome_status") == "tool_error"
    fr = str(out.get("final_response") or "")
    assert "could not complete or confirm" in fr.lower()
    assert "unknown tool" in fr.lower()
    assert "failure_reasons" in (out.get("context_data") or {})
    assert "Unknown tool" in (out.get("context_data") or {}).get("failure_reasons", [""])[0]


def test_refund_logic_gate_records_policy_check_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.create_refund_request",
        lambda order_id, reason, **_kwargs: {
            "ok": True,
            "refund_id": 1210,
            "order_id": order_id,
            "decision": "pending",
        },
    )
    state = {
        "procedure_id": "get_refund",
        "text": "I need to start the refund process for order ORD-8194.",
        "messages": [{"role": "user", "content": "I need to start the refund process for order ORD-8194."}],
        "todo_list": [
            {"id": "assess_eligibility_and_refund", "type": "tool_call", "tool": "create_refund_request"},
            {
                "id": "branch_refund_created",
                "type": "logic_gate",
                "condition": {"op": "eq", "field": "refund_request_created", "value": True},
                "on_true": "confirm_refund_submitted",
                "on_false": "refund_not_created",
            },
            {"id": "confirm_refund_submitted", "type": "llm_response", "message": "confirm refund"},
            {"id": "refund_not_created", "type": "llm_response", "message": "deny"},
        ],
        "current_step_index": 0,
        "context_data": {"order_status": "delivered"},
        "policy_constraints": {},
        "policy_check_results": [],
        "assistant_metadata": {},
    }

    out1 = _structured_executor_node(state)
    out2 = _structured_executor_node(out1)
    out3 = _structured_executor_node(out2)

    context_data = out3.get("context_data") or {}
    assert context_data.get("refund_request_created") is True
    assert context_data.get("refund_request_id") == 1210
    checks = out2.get("policy_check_results") or []
    assert checks
    assert checks[-1]["source"] == "procedure_gate"


def test_product_price_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_product_price_record",
        lambda _name: {"sku": "SKU-9", "name": "Widget", "price": 12.5},
    )
    state = {
        "text": "price for widget",
        "messages": [{"role": "user", "content": "widget"}],
        "todo_list": [{"id": "lookup_price", "type": "tool_call", "tool": "product_price_lookup"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["product_found"] is True
    assert out["context_data"]["product_price"]["price"] == 12.5


def test_unsubscribe_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.unsubscribe_subscription",
        lambda _email, **_kwargs: {"ok": True, "subscription_status": "unsubscribed"},
    )
    state = {
        "text": "unsubscribe subscription_holder@example.com",
        "messages": [{"role": "user", "content": "unsubscribe subscription_holder@example.com"}],
        "todo_list": [{"id": "unsubscribe", "type": "tool_call", "tool": "unsubscribe_subscription"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["unsubscribe_succeeded"] is True
    assert out["context_data"]["subscription_status"] == "unsubscribed"


def test_delivery_period_tool_step_updates_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_delivery_period",
        lambda _ref: {
            "tracking_id": "TRK-X",
            "order_id": "ORD-1",
            "promised_delivery_at": "2026-05-10T10:00:00+00:00",
            "actual_delivery_at": None,
            "order_status": "shipped",
        },
    )
    state = {
        "text": "delivery period for TRK-X",
        "messages": [{"role": "user", "content": "TRK-X"}],
        "todo_list": [{"id": "delivery_period", "type": "tool_call", "tool": "delivery_period_lookup"}],
        "current_step_index": 0,
        "context_data": {},
        "assistant_metadata": {},
    }
    out = _structured_executor_node(state)
    assert out["current_step_index"] == 1
    assert out["context_data"]["delivery_info_found"] is True
    assert out["context_data"]["delivery_period"]["tracking_id"] == "TRK-X"
