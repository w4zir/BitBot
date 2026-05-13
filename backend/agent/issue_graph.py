"""Procedure-driven LangGraph: category -> intent -> procedure execution."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from backend.agent.procedures import (
    as_dict,
    get_blueprint_by_category_intent,
    get_blueprint_with_fallback_chain,
    load_blueprints,
)
from backend.agent.policy_constraints import (
    PolicyCheckResult,
    PolicyConstraints,
    load_policy_constraints_for_intent,
)
from backend.db.intents_repo import get_intent_definitions_for_category
from backend.db.orders_repo import (
    cancel_order as cancel_order_record,
    get_order_status,
    update_shipping_address as update_shipping_address_record,
)
from backend.db.postgres import postgres_configured
from backend.db.delivery_repo import get_delivery_period
from backend.db.invoices_repo import get_invoice
from backend.db.payments_repo import (
    get_payment,
    list_payment_methods,
)
from backend.db.products_repo import (
    get_product_availability as get_product_availability_record,
    get_product_info as get_product_info_record,
    get_product_price as get_product_price_record,
    lookup_product,
)
from backend.db.refunds_repo import create_refund_request, get_refund_context, get_refund_tracking
from backend.db.subscriptions_repo import (
    get_subscription,
    unsubscribe_subscription,
)
from backend.db.support_repo import create_support_ticket
from backend.llm.providers import chat_completion, extract_json_object
from backend.rag.query_classifier import ClassificationResult, get_query_classifier
from backend.rag.required_fields import normalize_category_key

logger = logging.getLogger(__name__)


class IssueGraphState(TypedDict, total=False):
    text: str
    session_id: str
    messages: list[dict[str, Any]]
    issue_locked: bool
    category: str
    intent: str
    confidence: float
    problem_to_solve: str
    procedure_id: str
    todo_list: list[dict[str, Any]]
    current_step_index: int
    context_data: dict[str, Any]
    validation_ok: bool | None
    validation_missing: list[str]
    eligibility_ok: bool | None
    specialist_agent_id: str
    tool_registry_scope: str
    procedure_namespace: str
    policy_constraints: dict[str, Any] | None
    policy_check_results: list[dict[str, Any]]
    outcome_status: str | None
    escalation_bundle: dict[str, Any] | None
    final_response: str | None
    assistant_metadata: dict[str, Any]
    stage_metadata: dict[str, Any]
    agent_state: dict[str, Any]
    validation_wait_count: int
    validation_wait_limit: int
    output_validation: dict[str, Any]
    context_summary: dict[str, Any]
    classify_intent_attempts: int
    policy_load_attempts: int
    executor_turn_count: int
    enable_persistent_wait_interrupt: bool


DEFAULT_MAX_NODE_TURNS = 20


def _max_node_turns() -> int:
    raw = os.getenv("AGENT_MAX_NODE_TURNS", str(DEFAULT_MAX_NODE_TURNS)).strip()
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_MAX_NODE_TURNS
    return max(1, val)


MAX_NODE_TURNS = _max_node_turns()


def _validation_wait_limit() -> int:
    raw = os.getenv("AGENT_VALIDATION_MAX_USER_WAITS", str(MAX_NODE_TURNS)).strip()
    try:
        val = int(raw)
    except ValueError:
        return MAX_NODE_TURNS
    return max(1, val)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_name_from_state(state: IssueGraphState) -> str:
    return str((state.get("agent_state") or {}).get("stage") or "unknown_stage")


def _with_stage_metadata(
    state: IssueGraphState,
    stage_name: str,
    details: dict[str, Any] | None = None,
    *,
    llm_call: dict[str, Any] | None = None,
) -> IssueGraphState:
    stage_metadata = dict(state.get("stage_metadata") or {})
    ts = _utc_now_iso()
    detail_map = {k: v for k, v in dict(details or {}).items() if v is not None}
    llm_payload = dict(llm_call or {}) if isinstance(llm_call, dict) else None
    step_entry = {
        "timestamp": ts,
        "step_id": str(detail_map.get("step_id") or ""),
        "step_type": str(detail_map.get("step_type") or ("llm_call" if llm_payload else "node_operation")),
        "details": detail_map,
    }
    if llm_payload:
        step_entry["llm_call"] = llm_payload
    existing = stage_metadata.get(stage_name)
    prior_steps = []
    if isinstance(existing, dict):
        raw_steps = existing.get("steps")
        if isinstance(raw_steps, list):
            prior_steps = [item for item in raw_steps if isinstance(item, dict)]
    stage_metadata[stage_name] = {
        "timestamp": ts,
        "steps": [*prior_steps, step_entry],
    }
    return {
        **state,
        "stage_metadata": stage_metadata,
        "agent_state": {
            **dict(state.get("agent_state") or {}),
            "stage": stage_name,
        },
    }


def _compact_context_data(context_data: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "order_id_extracted",
        "order_status",
        "order_found",
        "cancel_succeeded",
        "cancel_reason",
        "order_age_hours",
        "refund_request_created",
        "refund_request_reason",
        "refund_request_id",
        "shipping_address_updated",
        "policy_constraints_path",
        "policy_schema_version",
        "policy_eligible",
        "policy_ineligibility_reason",
        "policy_doc_names",
        "policy_check_results",
        "failure_reasons",
        "tool_call",
        "order_status_before",
        "order_status_after",
        "payment_found",
        "payment_issue_ticket_created",
        "refund_tracking_found",
        "invoice_found",
        "subscription_found",
        "unsubscribe_succeeded",
        "handoff_created",
        "complaint_created",
        "delivery_info_found",
        "product_found",
    }
    return {k: v for k, v in context_data.items() if k in keep_keys}


def _is_internal_gate_reason(reason: str) -> bool:
    r = (reason or "").strip().lower()
    return bool(
        r.startswith("condition_failed:")
        or r.startswith("unsupported_operator")
        or r.startswith("condition_eval_error")
    )


def _collect_failure_reasons(state: dict[str, Any]) -> list[str]:
    """Customer-facing / LLM-facing failure strings from policy, procedure gates, tools, and validation."""
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        t = (s or "").strip()
        if not t or t.lower() == "ok":
            return
        if t in seen:
            return
        seen.add(t)
        out.append(t)

    meta = dict(state.get("assistant_metadata") or {})
    add(str(meta.get("tool_error") or ""))
    add(str(meta.get("step_error") or ""))

    ctx = dict(state.get("context_data") or {})
    add(str(ctx.get("policy_ineligibility_reason") or ""))

    pc = dict(state.get("policy_constraints") or {})
    if not bool(pc.get("eligible", True)):
        add(str(pc.get("reason") or "").strip())
        if not pc.get("reason"):
            add(str(pc.get("default_ineligible_reason") or "").strip())

    for chk in state.get("policy_check_results") or []:
        if not isinstance(chk, dict):
            continue
        if chk.get("passed"):
            continue
        r = str(chk.get("reason") or "").strip()
        if r and not _is_internal_gate_reason(r):
            add(r)

    ov = dict(state.get("output_validation") or {})
    checks = ov.get("checks")
    if isinstance(checks, dict):
        for _cid, item in checks.items():
            if not isinstance(item, dict):
                continue
            if item.get("valid"):
                continue
            r = str(item.get("reason") or "").strip()
            if r and r.lower() != "ok":
                add(r)
    return out


def _merge_failure_reasons_into_state(state: dict[str, Any]) -> dict[str, Any]:
    reasons = _collect_failure_reasons(state)
    ctx = dict(state.get("context_data") or {})
    ctx["failure_reasons"] = reasons
    meta = dict(state.get("assistant_metadata") or {})
    meta["failure_reasons"] = reasons
    return {**state, "context_data": ctx, "assistant_metadata": meta}


def _maybe_augment_blocked_final_response(*, outcome_status: str, final_response: str, reasons: list[str]) -> str | None:
    """Append explicit failure reasons when outcome is blocked and the reply omits them."""
    if outcome_status not in {"unresolvable", "tool_error", "step_error"}:
        return None
    if not reasons:
        return None
    summary = "; ".join(reasons)
    if not summary.strip():
        return None
    fr = (final_response or "").strip()
    if summary.lower() in fr.lower():
        return fr if fr else None
    tail = "We could not complete or confirm this request because: " + summary
    if not fr:
        return tail
    return f"{fr}\n\n{tail}"


def _build_agent_state_snapshot(state: IssueGraphState) -> dict[str, Any]:
    idx = int(state.get("current_step_index") or 0)
    context = dict(state.get("context_data") or {})
    return {
        "stage": str((state.get("agent_state") or {}).get("stage") or "unknown_stage"),
        "category": str(state.get("category") or ""),
        "intent": str(state.get("intent") or ""),
        "problem_to_solve": str(state.get("problem_to_solve") or ""),
        "procedure_id": str(state.get("procedure_id") or ""),
        "validation_ok": state.get("validation_ok"),
        "validation_missing": list(state.get("validation_missing") or []),
        "eligibility_ok": state.get("eligibility_ok"),
        "outcome_status": state.get("outcome_status"),
        "order_status_before": context.get("order_status_before"),
        "order_status_after": context.get("order_status_after"),
        "current_step_index": idx,
        "validation_wait_count": int(state.get("validation_wait_count") or 0),
        "validation_wait_limit": int(state.get("validation_wait_limit") or _validation_wait_limit()),
        "classify_intent_attempts": int(state.get("classify_intent_attempts") or 0),
        "policy_load_attempts": int(state.get("policy_load_attempts") or 0),
        "executor_turn_count": int(state.get("executor_turn_count") or 0),
        "max_node_turns": MAX_NODE_TURNS,
    }


def build_agent_trace(
    *,
    agent_state: dict[str, Any] | None,
    stage_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    for node_name, node_raw in dict(stage_metadata or {}).items():
        if not isinstance(node_name, str):
            continue
        if not isinstance(node_raw, dict):
            continue
        steps_raw = node_raw.get("steps")
        step_dicts: list[dict[str, Any]] = []
        if isinstance(steps_raw, list):
            step_dicts = [item for item in steps_raw if isinstance(item, dict)]
        if not step_dicts:
            synthesized_details = {
                k: v
                for k, v in node_raw.items()
                if k not in {"steps"}
            }
            step_dicts = [
                {
                    "timestamp": node_raw.get("timestamp"),
                    "step_id": str(synthesized_details.get("step_id") or ""),
                    "step_type": str(synthesized_details.get("step_type") or ""),
                    "details": synthesized_details,
                }
            ]
        formatted_steps: list[dict[str, Any]] = []
        for index, step in enumerate(step_dicts, start=1):
            details = dict(step.get("details") or {})
            step_id = str(step.get("step_id") or details.get("step_id") or f"{node_name}_{index}")
            step_type = str(step.get("step_type") or details.get("step_type") or "node_operation")
            formatted = {
                "step_id": step_id,
                "step_type": step_type,
                "timestamp": step.get("timestamp") or node_raw.get("timestamp"),
                "details": details,
            }
            llm_call = step.get("llm_call")
            if isinstance(llm_call, dict):
                formatted["llm_call"] = llm_call
            formatted_steps.append(formatted)
        nodes[node_name] = {"steps": formatted_steps}
    return {
        "state": dict(agent_state or {}),
        "nodes": nodes,
    }


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_duration_hours(
    *,
    start: Any,
    end: Any,
    op: str,
    threshold_hours: Any,
) -> dict[str, Any]:
    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end) or datetime.now(timezone.utc)
    threshold = _as_float(threshold_hours)
    if start_dt is None or threshold is None:
        return {"valid": False, "actual_value": None, "reason": "invalid_duration_inputs"}
    elapsed_hours = (end_dt - start_dt).total_seconds() / 3600.0
    checks = {
        "lte": elapsed_hours <= threshold,
        "lt": elapsed_hours < threshold,
        "gte": elapsed_hours >= threshold,
        "gt": elapsed_hours > threshold,
        "eq": abs(elapsed_hours - threshold) < 1e-9,
    }
    return {
        "valid": bool(checks.get(op, False)),
        "actual_value": elapsed_hours,
        "reason": f"duration_hours_{op}_{threshold}",
    }


def _validate_set_membership(*, value: Any, allowed_values: Any) -> dict[str, Any]:
    allowed = {str(v).strip().lower() for v in (allowed_values or []) if str(v).strip()}
    probe = str(value or "").strip().lower()
    return {
        "valid": bool(probe and probe in allowed),
        "actual_value": probe,
        "reason": "value_not_in_allow_set" if probe not in allowed else "ok",
        "set_difference": sorted({probe} - allowed) if probe else [],
    }


def _validate_arithmetic(*, lhs: Any, rhs: Any, op: str) -> dict[str, Any]:
    left = _as_float(lhs)
    right = _as_float(rhs)
    if left is None or right is None:
        return {"valid": False, "actual_value": None, "reason": "invalid_numeric_inputs"}
    checks = {
        "lte": left <= right,
        "lt": left < right,
        "gte": left >= right,
        "gt": left > right,
        "eq": abs(left - right) < 1e-9,
    }
    return {
        "valid": bool(checks.get(op, False)),
        "actual_value": left,
        "reason": f"arithmetic_{op}_{right}",
    }


def _value_from_path(path: str, *, context_data: dict[str, Any], policy_constraints: dict[str, Any]) -> Any:
    raw = (path or "").strip()
    if not raw:
        return None
    if raw.startswith("policy."):
        cursor: Any = policy_constraints
        parts = raw.split(".")[1:]
    elif raw.startswith("context."):
        cursor = context_data
        parts = raw.split(".")[1:]
    else:
        cursor = context_data
        parts = raw.split(".")
    for part in parts:
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        else:
            return None
    return cursor


def _build_policy_check_result(
    *,
    check_id: str,
    passed: bool,
    reason: str,
    actual_value: Any,
    expected_value: Any,
    source: str,
    condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return PolicyCheckResult(
        check_id=check_id,
        passed=passed,
        reason=reason,
        actual_value=actual_value,
        expected_value=expected_value,
        source=source,
        condition=dict(condition or {}),
    ).model_dump()


def _classify_category_node(state: IssueGraphState) -> IssueGraphState:
    text = state.get("text") or ""
    if state.get("issue_locked"):
        return _with_stage_metadata(
            {**state, "text": text},
            "classify_category",
            {"issue_locked": True, "category": state.get("category"), "confidence": state.get("confidence")},
        )
    qc = get_query_classifier()
    result: ClassificationResult = qc.classify(text)
    out: IssueGraphState = {
        **state,
        "text": text,
        "category": result.category,
        "confidence": result.confidence,
    }
    return _with_stage_metadata(
        out,
        "classify_category",
        {"issue_locked": False, "category": result.category, "confidence": result.confidence},
    )


def _category_confidence_threshold() -> float:
    raw = os.getenv("CATEGORY_CONFIDENCE_THRESHOLD", "0.5").strip()
    try:
        val = float(raw)
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, val))


def _no_issue_direct_node(state: IssueGraphState) -> IssueGraphState:
    provider = os.getenv("NO_ISSUE_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("NO_ISSUE_MODEL", "llama3.2").strip()
    system = os.getenv(
        "NO_ISSUE_SYSTEM_PROMPT",
        "You are a helpful assistant for a commerce chatbot. Reply concisely and helpfully.",
    ).strip()

    llm_messages: list[dict[str, str]] = []
    if system:
        llm_messages.append({"role": "system", "content": system})
    llm_messages.extend(_messages_for_llm(state.get("messages") or []))

    meta = dict(state.get("assistant_metadata") or {})
    meta["branch"] = "no_issue_direct"
    meta["model_provider"] = provider
    meta["model"] = model
    llm_error = ""
    try:
        reply = chat_completion(provider=provider, model=model, messages=llm_messages)
    except Exception as e:  # noqa: BLE001
        reply = f"(Model error: {e})"
        meta["error"] = str(e)
        llm_error = str(e)

    out: IssueGraphState = {
        **state,
        "category": "no_issue",
        "intent": "no_issue_chat",
        "problem_to_solve": "",
        "procedure_id": "no_issue_chat",
        "todo_list": [],
        "current_step_index": 0,
        "final_response": reply,
        "assistant_metadata": meta,
    }
    return _with_stage_metadata(
        out,
        "no_issue_direct",
        {"model_provider": provider, "model": model, "response_generated": bool(reply)},
        llm_call={
            "provider": provider,
            "model": model,
            "messages": llm_messages,
            "raw_response": reply,
            "error": llm_error,
        },
    )


def _generate_ineligibility_response(
    *,
    reason: str,
    messages: list[dict[str, Any]],
    text: str,
    failure_reasons: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    provider = os.getenv("NO_ISSUE_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("NO_ISSUE_MODEL", "llama3.2").strip()
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*(failure_reasons or []), reason]:
        t = (item or "").strip()
        if t and t not in seen:
            seen.add(t)
            merged.append(t)
    reasons_json = json.dumps(merged, ensure_ascii=False)
    system = (
        "You are a customer support assistant. Explain policy ineligibility clearly and politely, "
        "and provide a concise next step when possible. Do not mention internal system details. "
        "Ground your explanation in the provided policy ineligibility reasons JSON array; "
        "prefer those strings when explaining why the request cannot be completed."
    )
    transcript = json.dumps(_messages_for_llm(messages), ensure_ascii=False)
    user_prompt = (
        f"Latest user message: {text or '(empty)'}\n"
        f"Policy ineligibility reasons (JSON array): {reasons_json}\n"
        f"Primary summary reason: {reason or 'Not eligible under current policy.'}\n"
        f"Conversation transcript JSON: {transcript}\n"
        "Write one concise customer-facing response."
    )
    try:
        raw = chat_completion(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (
            raw,
            {
                "provider": provider,
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                "raw_response": raw,
            },
        )
    except Exception:  # noqa: BLE001
        fallback_reason = (
            "; ".join(merged).strip()
            if merged
            else (reason.strip() or "this request is not eligible under our current policy.")
        )
        fallback = (
            "I am sorry, but I cannot complete that request because "
            f"{fallback_reason} Please let me know if you want help with an alternative next step."
        )
        return (
            fallback,
            {
                "provider": provider,
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                "raw_response": fallback,
                "error": "model_error",
            },
        )


def _messages_for_llm(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        out.append({"role": role, "content": content})
    return out


def _user_messages_from_session(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for msg in messages:
        if str(msg.get("role") or "").strip().lower() != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            out.append(content)
    return out


def _load_allowed_intent_definitions(category: str) -> list[dict[str, str]]:
    if not postgres_configured():
        return []
    try:
        return get_intent_definitions_for_category(category)
    except Exception:  # noqa: BLE001
        return []


def _classify_intent_node(state: IssueGraphState) -> IssueGraphState:
    category = normalize_category_key(state.get("category") or "unknown")
    meta = dict(state.get("assistant_metadata") or {})
    if state.get("issue_locked"):
        meta["intent_classifier"] = "session_locked"
        meta["intent_candidates"] = [state.get("intent") or ""]
        out_locked: IssueGraphState = {
            **state,
            "intent": (state.get("intent") or "").strip(),
            "problem_to_solve": str(state.get("problem_to_solve") or "").strip(),
            "assistant_metadata": meta,
        }
        return _with_stage_metadata(
            out_locked,
            "classify_intent",
            {"intent_classifier": "session_locked", "intent": out_locked.get("intent")},
        )

    provider = os.getenv("INTENT_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("INTENT_MODEL", "llama3.2").strip()
    confidence = float(state.get("confidence") or 0.0)
    user_messages = _user_messages_from_session(state.get("messages") or [])
    messages_json = json.dumps(user_messages, ensure_ascii=False)
    allowed_intent_definitions = _load_allowed_intent_definitions(category)
    allowed_intents = [
        str(item.get("intent_name") or "").strip()
        for item in allowed_intent_definitions
        if str(item.get("intent_name") or "").strip()
    ]
    intent_definition_lookup = {
        intent_name: str(item.get("description") or "").strip()
        for item in allowed_intent_definitions
        for intent_name in [str(item.get("intent_name") or "").strip()]
        if intent_name
    }
    if allowed_intents:
        intents_bullets = "\n".join(
            f"- {item}: {intent_definition_lookup[item]}"
            if intent_definition_lookup.get(item)
            else f"- {item}"
            for item in allowed_intents
        )
        system_prompt = (
            "You classify a customer support session into:\n"
            "(1) a stable procedure intent\n"
            "(2) a concise problem to solve\n\n"
            "Respond ONLY with JSON:\n"
            '{"intent":"snake_case_label","problem_to_solve":"one sentence"}\n\n'
            "Rules:\n"
            "\n"
            "Intent selection:\n"
            "- Must be one of the allowed intents listed below\n"
            "- Select intent based on the intent descriptions, not just label names\n"
            "- Prefer the earliest dominant user goal unless a clear shift occurs\n"
            "- If multiple issues exist, choose the primary blocking issue\n"
            "- Prefer specific intents over general ones when possible\n"
            f"- If no intent clearly fits, use {category}_general\n\n"
            "Ambiguity handling:\n"
            "- If multiple intents seem valid, choose the most specific matching intent\n"
            "- Prefer specific intents over general ones when confidence is sufficient\n"
            "- Do NOT invent new intents\n\n"
            "Problem to solve:\n"
            "- Describe the user's concrete goal or issue (not symptoms)\n"
            "- Match the user's problem to the closest semantic definition\n"
            "- Use one clear sentence\n"
            '- Avoid vague phrasing (e.g., "needs help")\n'
            '- Prefer action-oriented phrasing (e.g., "resolve duplicate charge on account")\n\n'
            "Output constraints:\n"
            "- intent must be snake_case and match exactly one allowed label\n"
            "- problem_to_solve must be <= 20 words\n"
            "- No extra text outside JSON\n\n"
            f"Allowed intents for category '{category}':\n{intents_bullets}"
        )
    else:
        system_prompt = (
            "You classify a customer support session into:\n"
            "(1) a stable procedure intent\n"
            "(2) a concise problem to solve\n\n"
            "Respond ONLY with JSON:\n"
            '{"intent":"snake_case_label","problem_to_solve":"one sentence"}\n\n'
            "Rules:\n"
            "- Use ONLY the provided category, confidence score, and user messages\n"
            f"- If no intent clearly fits, use {category}_general\n"
            "- problem_to_solve must be one clear sentence and <= 20 words\n"
            "- No extra text outside JSON"
        )
    user_prompt = (
        f"Category: {category}\n"
        f"Category probability: {confidence:.6f}\n"
        f"User messages (chronological JSON array): {messages_json}"
    )
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    data: dict[str, Any] = {}
    attempt_count = 0
    llm_error = ""
    raw = ""
    for attempt_count in range(1, MAX_NODE_TURNS + 1):
        try:
            raw = chat_completion(provider=provider, model=model, messages=llm_messages)
            candidate = extract_json_object(raw)
            if isinstance(candidate, dict):
                data = candidate
            intent_candidate = str(data.get("intent") or "").strip()
            problem_candidate = str(data.get("problem_to_solve") or "").strip()
            if intent_candidate or problem_candidate:
                break
        except Exception as e:  # noqa: BLE001
            meta["intent_classifier_error"] = str(e)
            llm_error = str(e)

    intent = str(data.get("intent") or f"{category}_general").strip()
    if not intent:
        intent = f"{category}_general"
    if allowed_intents and intent not in allowed_intents:
        intent = f"{category}_general"
    problem_to_solve = str(data.get("problem_to_solve") or "").strip()

    meta["intent_classifier"] = "llm"
    meta["intent_model_provider"] = provider
    meta["intent_model"] = model
    meta["intent_allowed_list_used"] = bool(allowed_intents)
    meta["intent_definitions_used"] = bool(allowed_intent_definitions)
    meta["intent_candidates"] = allowed_intents if allowed_intents else [intent]
    out: IssueGraphState = {
        **state,
        "intent": intent,
        "problem_to_solve": problem_to_solve,
        "classify_intent_attempts": attempt_count,
        "assistant_metadata": meta,
    }
    return _with_stage_metadata(
        out,
        "classify_intent",
        {
            "intent_classifier": "llm",
            "intent": intent,
            "problem_to_solve": problem_to_solve,
            "allowed_intents_count": len(allowed_intents),
            "attempts": attempt_count,
        },
        llm_call={
            "provider": provider,
            "model": model,
            "messages": llm_messages,
            "raw_response": raw,
            "parsed_output": data,
            "attempts": attempt_count,
            "error": llm_error,
        },
    )


def _fetch_procedure_node(state: IssueGraphState) -> IssueGraphState:
    category = normalize_category_key(state.get("category") or "unknown")
    intent = (state.get("intent") or "").strip()
    bp = get_blueprint_with_fallback_chain(category, intent)
    if bp is None:
        text = str(state.get("text") or "").lower()
        inferred: tuple[str, str] | None = None
        if "refund" in text:
            inferred = ("refund", "get_refund")
        elif "cancel" in text:
            inferred = ("order", "cancel_order")
        elif _ORDER_NUMBER_RE.search(text) and any(k in text for k in ("status", "track", "where is")):
            inferred = ("order", "order_status")
        if inferred is not None:
            bp = get_blueprint_by_category_intent(*inferred)
    if bp is None:
        out_missing: IssueGraphState = {
            **state,
            "procedure_id": "",
            "todo_list": [],
            "current_step_index": 0,
            "final_response": "I could not map this request to a procedure.",
        }
        return _with_stage_metadata(out_missing, "fetch_procedure", {"procedure_found": False})
    out: IssueGraphState = {
        **state,
        "procedure_id": bp.id,
        "intent": bp.intent,
        "todo_list": [as_dict(step) for step in bp.steps],
        "current_step_index": 0,
        "context_data": dict(state.get("context_data") or {}),
    }
    return _with_stage_metadata(
        out,
        "fetch_procedure",
        {"procedure_found": True, "procedure_id": bp.id, "todo_count": len(out.get("todo_list") or [])},
    )


def _specialist_router_node(state: IssueGraphState) -> IssueGraphState:
    category = normalize_category_key(state.get("category") or "unknown")
    intent = str(state.get("intent") or "").strip()
    specialist = f"{category}_agent" if category and category != "unknown" else "general_agent"
    out: IssueGraphState = {
        **state,
        "specialist_agent_id": specialist,
        "tool_registry_scope": specialist,
        "procedure_namespace": f"{category}:{intent or 'general'}",
    }
    return _with_stage_metadata(
        out,
        "specialist_router",
        {
            "specialist_agent_id": specialist,
            "tool_registry_scope": specialist,
            "procedure_namespace": out.get("procedure_namespace"),
        },
    )


def _policy_load_node(state: IssueGraphState) -> IssueGraphState:
    category = str(state.get("category") or "").strip().lower()
    intent = str(state.get("intent") or "").strip().lower()
    context = dict(state.get("context_data") or {})

    constraints_model: PolicyConstraints = load_policy_constraints_for_intent(category, intent)
    constraints = constraints_model.model_dump()
    policy_path = f"{category or 'unknown'}/{intent or 'unknown'}.yaml"
    load_error = str((constraints.get("metadata") or {}).get("load_error") or "").strip()
    eligible = not bool(load_error)
    reason = "" if eligible else str(constraints.get("default_ineligible_reason") or load_error)
    check = _build_policy_check_result(
        check_id="policy_constraints_loaded",
        passed=eligible,
        reason="" if eligible else load_error,
        actual_value=policy_path,
        expected_value="existing constraints artifact",
        source="policy_loader",
    )
    constraints["eligible"] = eligible
    constraints["reason"] = reason

    out: IssueGraphState = {
        **state,
        "policy_load_attempts": 1,
        "policy_constraints": constraints,
        "policy_check_results": [check],
        "context_data": {
            **context,
            "policy_constraints_path": policy_path,
            "policy_schema_version": constraints.get("schema_version"),
            "policy_doc_names": list(constraints.get("policy_doc_names") or []),
            "policy_eligible": eligible,
            "policy_ineligibility_reason": reason,
            "policy_check_results": [check],
        },
    }
    return _with_stage_metadata(
        out,
        "policy_load",
        {
            "policy_constraints_path": policy_path,
            "policy_schema_version": constraints.get("schema_version"),
            "policy_eligible": eligible,
            "policy_ineligibility_reason": reason,
            "policy_checks_count": 1,
        },
    )


def _build_missing_prompts(required_fields: list[dict[str, Any]], missing_names: list[str]) -> str:
    name_to_prompt: dict[str, str] = {}
    for field in required_fields:
        name = str(field.get("name") or "").strip().lower()
        prompt = str(field.get("prompt") or "").strip()
        if name and prompt:
            name_to_prompt[name] = prompt
    lines = [name_to_prompt[m.lower()] for m in missing_names if m.lower() in name_to_prompt]
    if not lines:
        return "Please provide the missing details so we can help."
    return "\n".join(lines)


def _evaluate_policy_rules(
    *,
    policy_constraints: dict[str, Any],
    context_data: dict[str, Any],
    phase: str,
) -> tuple[bool, str, list[dict[str, Any]]]:
    rules = list(policy_constraints.get("eligibility_rules") or [])
    results: list[dict[str, Any]] = []
    if not rules:
        return True, "", results
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        applies_to = str(rule.get("applies_to") or "runtime").strip().lower()
        if applies_to not in {phase, "any"}:
            continue
        eval_out = _evaluate_condition(rule, context_data=context_data, policy_constraints=policy_constraints)
        check = _build_policy_check_result(
            check_id=str(rule.get("id") or "policy_rule"),
            passed=bool(eval_out.get("passed")),
            reason=str(eval_out.get("reason") or ""),
            actual_value=eval_out.get("actual_value"),
            expected_value=eval_out.get("expected_value"),
            source="policy_rule",
            condition=rule,
        )
        results.append(check)
        if not check["passed"]:
            fallback_reason = str(rule.get("failure_reason") or "").strip()
            return False, fallback_reason or str(check["reason"]), results
    return True, "", results


def _validate_required_data_node(state: IssueGraphState) -> IssueGraphState:
    bp = get_blueprint_by_category_intent(state.get("category") or "", state.get("intent") or "")
    if bp is None:
        return _with_stage_metadata(state, "validate_required", {"blueprint_found": False})
    required = [as_dict(x) for x in bp.required_data]
    if not required:
        return _with_stage_metadata(
            {**state, "validation_ok": True, "validation_missing": []},
            "validate_required",
            {"required_fields_count": 0, "validation_ok": True},
        )

    provider = os.getenv("VALIDATION_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("VALIDATION_MODEL", "llama3.2").strip()
    category = str(state.get("category") or "unknown")
    transcript = json.dumps(state.get("messages") or [], ensure_ascii=False)
    req = json.dumps(required, ensure_ascii=False)
    sys_prompt = (
        "You validate whether the user provided all REQUIRED information for a support case.\n"
        f"Category: {category}.\n"
        f"Required fields definition (JSON): {req}\n"
        "Each required field has a name. Decide if the conversation contains a plausible value for each.\n"
        'Reply with ONLY a JSON object: {"valid": true|false, "missing_field_names": ["name1", ...], "notes": "short"}\n'
        "missing_field_names must use the exact field names from the required_fields list."
    )
    user_prompt = f"Transcript (JSON messages): {transcript}"
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = ""
    last_err: Exception | None = None
    validation_attempts = 0
    for validation_attempts in range(1, MAX_NODE_TURNS + 1):
        try:
            raw = chat_completion(provider=provider, model=model, messages=msgs)
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        out_err: IssueGraphState = {
            **state,
            "validation_ok": False,
            "validation_missing": [],
            "final_response": f"Validation could not run: {last_err}",
            "assistant_metadata": {
                **dict(state.get("assistant_metadata") or {}),
                "branch": "validate",
                "error": str(last_err),
            },
        }
        return _with_stage_metadata(
            out_err,
            "validate_required",
            {
                "required_fields_count": len(required),
                "validation_ok": False,
                "error": str(last_err),
                "attempts": validation_attempts,
            },
            llm_call={
                "provider": provider,
                "model": model,
                "messages": msgs,
                "raw_response": raw,
                "attempts": validation_attempts,
                "error": str(last_err),
            },
        )
    data = extract_json_object(raw)
    llm_valid = bool(data.get("valid"))
    missing = data.get("missing_field_names") or data.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []
    missing_strs = [str(x) for x in missing if x]
    extracted_fields = data.get("extracted_fields")
    if not isinstance(extracted_fields, dict):
        extracted_fields = {}
    context_data = dict(state.get("context_data") or {})

    required_names = [str(item.get("name") or "").strip() for item in required if str(item.get("name") or "").strip()]
    required_names_lc = {name.lower() for name in required_names}
    for field_name in required_names:
        field_lc = field_name.lower()
        if field_lc == "order_id":
            strict_order_id = _extract_order_id_from_conversation(state.get("messages") or [], state.get("text"))
            if strict_order_id:
                extracted_fields[field_name] = strict_order_id
                extracted_fields[field_lc] = strict_order_id

    recomputed_missing: list[str] = []
    for field_name in required_names:
        field_lc = field_name.lower()
        if field_lc == "order_id":
            oid = str(extracted_fields.get(field_name) or extracted_fields.get(field_lc) or "").strip().upper()
            if not _ORDER_ID_ONLY_RE.match(oid):
                recomputed_missing.append(field_name)
    for candidate in missing_strs:
        field_lc = candidate.lower().strip()
        if field_lc == "order_id":
            oid = str(
                extracted_fields.get("order_id")
                or extracted_fields.get("ORDER_ID")
                or extracted_fields.get("Order_Id")
                or ""
            ).strip().upper()
            if _ORDER_ID_ONLY_RE.match(oid):
                continue
        if field_lc and field_lc in required_names_lc and all(field_lc != x.lower() for x in recomputed_missing):
            recomputed_missing.append(next(x for x in required_names if x.lower() == field_lc))
    if not missing_strs and not llm_valid:
        # Recover from malformed validator output by checking deterministic fields first.
        # If the schema has non-deterministic fields, keep validation blocked.
        if required_names_lc == {"order_id"} and not recomputed_missing:
            missing_strs = []
        elif required_names_lc == {"order_id"} and recomputed_missing:
            missing_strs = recomputed_missing
        elif recomputed_missing:
            missing_strs = recomputed_missing
        else:
            missing_strs = required_names
    else:
        missing_strs = recomputed_missing
    valid = not missing_strs

    assistant_reply: str | None = None
    if not valid:
        assistant_reply = _build_missing_prompts(required, missing_strs)
    else:
        assistant_reply = str(data.get("assistant_reply") or "")

    meta = {
        **dict(state.get("assistant_metadata") or {}),
        "branch": "validate_required_data",
        "model_provider": provider,
        "model": model,
        "validation_notes": data.get("notes"),
        "validation_attempts": validation_attempts,
        "validation_llm_valid": llm_valid,
    }
    out: IssueGraphState = {
        **state,
        "validation_ok": valid,
        "validation_missing": missing_strs,
        "final_response": assistant_reply or state.get("final_response"),
        "context_data": {
            **context_data,
            **{str(k): v for k, v in extracted_fields.items() if k},
        },
        "assistant_metadata": meta,
    }
    if not valid:
        out["current_step_index"] = len(state.get("todo_list") or [])
    return _with_stage_metadata(
        out,
        "validate_required",
        {
            "required_fields_count": len(required),
            "validation_ok": valid,
            "validation_missing": missing_strs,
        },
        llm_call={
            "provider": provider,
            "model": model,
            "messages": msgs,
            "raw_response": raw,
            "parsed_output": data,
            "attempts": validation_attempts,
        },
    )


def _data_and_eligibility_validator_node(state: IssueGraphState) -> IssueGraphState:
    validated = _validate_required_data_node(state)
    policy_constraints = dict(validated.get("policy_constraints") or {})
    context_data = dict(validated.get("context_data") or {})
    loaded_checks = list(validated.get("policy_check_results") or [])
    ineligibility_llm_call: dict[str, Any] | None = None
    rules_ok, rules_reason, rule_checks = _evaluate_policy_rules(
        policy_constraints=policy_constraints,
        context_data=context_data,
        phase="pre_validation",
    )
    eligibility_ok = bool(policy_constraints.get("eligible", True)) and rules_ok
    reason = (
        rules_reason
        or str(policy_constraints.get("reason") or "").strip()
        or str(policy_constraints.get("default_ineligible_reason") or "").strip()
    )
    wait_limit = int(validated.get("validation_wait_limit") or _validation_wait_limit())
    wait_count = int(validated.get("validation_wait_count") or 0)
    out: IssueGraphState = {
        **validated,
        "eligibility_ok": eligibility_ok,
        "policy_check_results": [*loaded_checks, *rule_checks],
        "validation_wait_limit": wait_limit,
        "policy_constraints": {
            **policy_constraints,
            "eligible": eligibility_ok,
            "reason": "" if eligibility_ok else reason,
        },
        "context_data": {
            **context_data,
            "policy_eligible": eligibility_ok,
            "policy_ineligibility_reason": "" if eligibility_ok else reason,
            "policy_check_results": [*loaded_checks, *rule_checks],
        },
    }
    if validated.get("validation_ok") is False:
        wait_count += 1
        out["validation_wait_count"] = wait_count
        out["outcome_status"] = "needs_more_data"
        meta = dict(out.get("assistant_metadata") or {})
        if wait_count >= wait_limit:
            out["final_response"] = (
                "I still do not have the required details to proceed. "
                "I am escalating this to a human support agent."
            )
            out["outcome_status"] = "pending_escalation"
            meta["validation_wait_limit_reached"] = True
        elif bool(out.get("enable_persistent_wait_interrupt")):
            meta["validation_interrupt_pending"] = True
        out["assistant_metadata"] = meta
    elif not eligibility_ok:
        reason_list = _collect_failure_reasons(out)
        ineligible_response, ineligibility_llm_call = _generate_ineligibility_response(
            reason=reason,
            messages=validated.get("messages") or [],
            text=str(validated.get("text") or ""),
            failure_reasons=reason_list,
        )
        out["final_response"] = ineligible_response
        out["outcome_status"] = "policy_ineligible"
        out["validation_wait_count"] = 0
    else:
        out["validation_wait_count"] = 0
    return _with_stage_metadata(
        _merge_failure_reasons_into_state(out),
        "validate_required",
        {
            "validation_ok": out.get("validation_ok"),
            "eligibility_ok": eligibility_ok,
            "policy_checks_count": len(out.get("policy_check_results") or []),
            "policy_ineligibility_reason": "" if eligibility_ok else reason,
            "validation_wait_count": out.get("validation_wait_count"),
            "validation_wait_limit": wait_limit,
        },
        llm_call=ineligibility_llm_call,
    )


_ORDER_NUMBER_RE = re.compile(r"\b(ORD-[A-Z0-9]+)\b", re.IGNORECASE)
_ORDER_ID_ONLY_RE = re.compile(r"^\s*ORD-[A-Z0-9]+\s*$", re.IGNORECASE)
_TXN_NUMBER_RE = re.compile(r"\b(TXN-[A-Z0-9]+)\b", re.IGNORECASE)
_INVOICE_ID_RE = re.compile(r"\b(INV-[A-Z0-9-]+)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
_TRACKING_RE = re.compile(r"\b(TRK-[A-Z0-9-]+)\b", re.IGNORECASE)
_ESCALATION_DECISION_RE = re.compile(r"\b(accept|reject)\b", re.IGNORECASE)
_USER_RESOLUTION_CONFIRM_RE = re.compile(
    r"(?i)\b("
    r"problem\s+solved|"
    r"issue\s+resolved|"
    r"all\s+set|"
    r"no\s+more\s+(help|questions)|"
    r"(it'?s|that'?s)\s+(resolved|fixed|sorted)|"
    r"(yes[, ]+)?(that'?s|this\s+is)\s+(all|fixed|resolved|sorted)|"
    r"(thanks|thank\s+you)[,.]?\s*(that'?s|this\s+is)\s+enough"
    r")\b"
)


def get_category_for_stored_intent(intent: str) -> str | None:
    """Resolve blueprint category from a stored intent name (for locked sessions)."""
    it = (intent or "").strip().lower()
    if not it:
        return None
    for bp in load_blueprints().values():
        if bp.intent.lower() == it:
            return bp.category
    return None


def user_confirms_resolution(text: str) -> bool:
    """Heuristic: user explicitly signals their issue is resolved."""
    return bool(_USER_RESOLUTION_CONFIRM_RE.search((text or "").strip()))


def graph_suggests_session_resolved(state: IssueGraphState) -> bool:
    """
    True when the procedure finished successfully without pending human action or escalation
    to a live agent.
    """
    meta = dict(state.get("assistant_metadata") or {})
    if meta.get("pending_human_action"):
        return False
    if meta.get("escalation_decision") == "accept":
        return False
    if state.get("validation_ok") is False:
        return False
    if meta.get("tool_error") or meta.get("step_error"):
        return False
    if meta.get("branch") == "validate" and meta.get("error"):
        return False
    fr = str(state.get("final_response") or "")
    if "could not map this request to a procedure" in fr.lower():
        return False

    todo = state.get("todo_list") or []
    idx = int(state.get("current_step_index") or 0)
    if not todo:
        cat = str(state.get("category") or "").strip().lower()
        if cat == "no_issue" and state.get("final_response"):
            return True
        return False
    return idx >= len(todo)


def _extract_order_id_from_conversation(
    messages: list[dict[str, Any]] | None, text: str | None = None
) -> str | None:
    """First ORD-… token in chronological user messages wins; then current `text` if no match."""
    for m in messages or []:
        if str(m.get("role")) != "user":
            continue
        content = str(m.get("content") or "")
        mo = _ORDER_NUMBER_RE.search(content)
        if mo:
            return mo.group(1).upper()
    if text:
        mo = _ORDER_NUMBER_RE.search(str(text))
        if mo:
            return mo.group(1).upper()
    return None


def _extract_product_name_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages or []):
        if str(m.get("role")) != "user":
            continue
        text = str(m.get("content") or "").strip()
        if text:
            return text
    return None


def _extract_transaction_id(messages: list[dict[str, Any]], text: str | None = None) -> str | None:
    for m in messages or []:
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        mo = _TXN_NUMBER_RE.search(str(m.get("content") or ""))
        if mo:
            return mo.group(1).upper()
    if text:
        mo = _TXN_NUMBER_RE.search(str(text))
        if mo:
            return mo.group(1).upper()
    return None


def _extract_invoice_id(messages: list[dict[str, Any]], text: str | None = None) -> str | None:
    for m in messages or []:
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        mo = _INVOICE_ID_RE.search(str(m.get("content") or ""))
        if mo:
            return mo.group(1).upper()
    if text:
        mo = _INVOICE_ID_RE.search(str(text))
        if mo:
            return mo.group(1).upper()
    return None


def _extract_account_email(messages: list[dict[str, Any]], text: str | None = None) -> str | None:
    for m in messages or []:
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        mo = _EMAIL_RE.search(str(m.get("content") or ""))
        if mo:
            return mo.group(1).lower()
    if text:
        mo = _EMAIL_RE.search(str(text))
        if mo:
            return mo.group(1).lower()
    return None


def _extract_order_or_tracking(messages: list[dict[str, Any]], text: str | None = None) -> str | None:
    for m in messages or []:
        if str(m.get("role") or "").strip().lower() != "user":
            continue
        content = str(m.get("content") or "")
        trk = _TRACKING_RE.search(content)
        if trk:
            return trk.group(1).upper()
        ord_m = _ORDER_NUMBER_RE.search(content)
        if ord_m:
            return ord_m.group(1).upper()
    if text:
        trk = _TRACKING_RE.search(str(text))
        if trk:
            return trk.group(1).upper()
        ord_m = _ORDER_NUMBER_RE.search(str(text))
        if ord_m:
            return ord_m.group(1).upper()
    return None


def _extract_latest_user_message(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages or []):
        if str(m.get("role")) != "user":
            continue
        text = str(m.get("content") or "").strip()
        if text:
            return text
    return None


def _extract_escalation_decision(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages or []):
        if str(m.get("role")) != "user":
            continue
        text = str(m.get("content") or "")
        mo = _ESCALATION_DECISION_RE.search(text)
        if mo:
            return mo.group(1).lower()
    return None


def _extract_order_status_hint(state: IssueGraphState) -> str:
    context = dict(state.get("context_data") or {})
    raw = str(context.get("order_status") or "").strip().lower()
    if raw:
        return raw
    text = str(state.get("text") or "").strip().lower()
    mo = re.search(r"\bstatus\s*(?:is|=|:)\s*([a-z_]+)\b", text)
    if mo:
        return mo.group(1).strip().lower()
    return ""


def _check_order_status(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    """Populate context for order status from DB-backed repository."""
    oid = _extract_order_id_from_conversation(
        state.get("messages") or [], state.get("text")
    )
    tool_name = str(step.get("tool") or "check_order_status")
    base: dict[str, Any] = {
        "order_lookup_tool": tool_name,
        "order_id_extracted": oid,
    }
    if not oid:
        return {**base, "order_found": False, "order_status": None}
    row = get_order_status(oid)
    if not row:
        return {**base, "order_found": False, "order_status": None}
    prior_context = dict(state.get("context_data") or {})
    prior_status_before = prior_context.get("order_status_before")
    status_now = row.get("status")
    order_date = row.get("order_date")
    order_age_hours: float | None = None
    parsed_order_date = _parse_iso_datetime(order_date)
    if parsed_order_date is not None:
        order_age_hours = (datetime.now(timezone.utc) - parsed_order_date).total_seconds() / 3600.0
    return {
        **base,
        "order_found": True,
        "order_status": status_now,
        "order_status_before": prior_status_before if prior_status_before is not None else status_now,
        "order_age_hours": order_age_hours,
        "order_total_amount": row.get("total_amount"),
        "order_data": row,
    }


def _lookup_product_info(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "product_catalog_lookup")
    product_name = _extract_product_name_from_messages(state.get("messages") or [])
    base: dict[str, Any] = {"tool_call": tool_name, "product_name_extracted": product_name}
    if not product_name:
        return {**base, "product_found": False, "product": None}
    product = lookup_product(product_name)
    if not product:
        return {**base, "product_found": False, "product": None}
    return {**base, "product_found": True, "product": product}


def _lookup_product_info_only(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "product_info_lookup")
    product_name = _extract_product_name_from_messages(state.get("messages") or [])
    base: dict[str, Any] = {"tool_call": tool_name, "product_name_extracted": product_name}
    if not product_name:
        return {**base, "product_found": False, "product_info": None}
    product = get_product_info_record(product_name)
    if not product:
        return {**base, "product_found": False, "product_info": None}
    return {**base, "product_found": True, "product_info": product}


def _lookup_product_price(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "product_price_lookup")
    product_name = _extract_product_name_from_messages(state.get("messages") or [])
    base: dict[str, Any] = {"tool_call": tool_name, "product_name_extracted": product_name}
    if not product_name:
        return {**base, "product_found": False, "product_price": None}
    payload = get_product_price_record(product_name)
    if not payload:
        return {**base, "product_found": False, "product_price": None}
    return {**base, "product_found": True, "product_price": payload}


def _lookup_product_availability(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "product_availability_lookup")
    product_name = _extract_product_name_from_messages(state.get("messages") or [])
    base: dict[str, Any] = {"tool_call": tool_name, "product_name_extracted": product_name}
    if not product_name:
        return {**base, "product_found": False, "product_availability": None}
    payload = get_product_availability_record(product_name)
    if not payload:
        return {**base, "product_found": False, "product_availability": None}
    return {**base, "product_found": True, "product_availability": payload}


def _lookup_refund_context(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "refund_context_lookup")
    oid = _extract_order_id_from_conversation(
        state.get("messages") or [], state.get("text")
    )
    base: dict[str, Any] = {"tool_call": tool_name, "order_id_extracted": oid}
    if not oid:
        return {**base, "refund_context_found": False}
    payload = get_refund_context(oid)
    if not payload:
        return {**base, "refund_context_found": False}
    return {**base, "refund_context_found": True, **payload}


def _lookup_payment(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "payment_lookup")
    context = dict(state.get("context_data") or {})
    tx = str(context.get("transaction_id") or "").strip().upper()
    if not tx:
        tx = _extract_transaction_id(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "transaction_id": tx}
    if not tx:
        return {**base, "payment_found": False, "payment": None}
    payment = get_payment(tx)
    if not payment:
        return {**base, "payment_found": False, "payment": None}
    return {**base, "payment_found": True, "payment": payment}


def _list_payment_methods_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "list_payment_methods")
    return {
        "tool_call": tool_name,
        "payment_methods": list_payment_methods(),
    }


def _track_refund_payment_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "payment_refund_status")
    context = dict(state.get("context_data") or {})
    oid = str(context.get("order_id") or "").strip().upper()
    if not oid:
        oid = _extract_order_id_from_conversation(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "order_id": oid}
    if not oid:
        return {**base, "refund_tracking_found": False}
    payload = get_refund_tracking(oid)
    if not payload.get("found"):
        return {**base, "refund_tracking_found": False, "refund_tracking_reason": payload.get("reason")}
    return {**base, "refund_tracking_found": True, **payload}


def _check_invoice_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "check_invoice_value")
    context = dict(state.get("context_data") or {})
    invoice_id = str(context.get("invoice_id") or "").strip().upper()
    if not invoice_id:
        invoice_id = _extract_invoice_id(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "invoice_id": invoice_id}
    if not invoice_id:
        return {**base, "invoice_found": False}
    payload = get_invoice(invoice_id)
    if not payload:
        return {**base, "invoice_found": False}
    return {**base, "invoice_found": True, "invoice": payload}


def _get_subscription_status_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "get_subscription_status")
    context = dict(state.get("context_data") or {})
    email = str(context.get("account_email") or "").strip().lower()
    if not email:
        email = _extract_account_email(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "account_email": email}
    if not email:
        return {**base, "subscription_found": False}
    payload = get_subscription(email)
    if not payload:
        return {**base, "subscription_found": False}
    return {**base, "subscription_found": True, "subscription": payload}


def _unsubscribe_subscription_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "unsubscribe_subscription")
    context = dict(state.get("context_data") or {})
    email = str(context.get("account_email") or "").strip().lower()
    if not email:
        email = _extract_account_email(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "account_email": email}
    result = unsubscribe_subscription(email, update_source="agent")
    return {
        **base,
        "unsubscribe_succeeded": bool(result.get("ok")),
        "unsubscribe_reason": str(result.get("reason") or ""),
        "subscription_status": result.get("subscription_status"),
    }


def _create_payment_issue_ticket_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "create_payment_issue_ticket")
    context = dict(state.get("context_data") or {})
    tx = str(context.get("transaction_id") or "").strip().upper()
    if not tx:
        tx = _extract_transaction_id(state.get("messages") or [], state.get("text")) or ""
    payment = get_payment(tx) if tx else None
    latest_msg = _extract_latest_user_message(state.get("messages") or []) or str(state.get("text") or "")
    result = create_support_ticket(
        issue_type="payment",
        payload={
            "transaction_id": tx,
            "payment": payment or {},
            "issue_summary": latest_msg,
        },
        routing_result="payment_issue_review",
        validation_passed=bool(payment),
        update_source="agent",
    )
    return {
        "tool_call": tool_name,
        "transaction_id": tx,
        "payment_found": bool(payment),
        "payment_issue_ticket_created": bool(result.get("ok")),
        "payment_issue_ticket_id": result.get("ticket_id"),
    }


def _create_contact_handoff_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "create_contact_handoff")
    summary = _extract_latest_user_message(state.get("messages") or []) or str(state.get("text") or "")
    result = create_support_ticket(
        issue_type="contact",
        payload={"summary": summary, "session_id": str(state.get("session_id") or "")},
        routing_result="human_agent_queue",
        update_source="agent",
    )
    return {
        "tool_call": tool_name,
        "handoff_created": bool(result.get("ok")),
        "handoff_ticket_id": result.get("ticket_id"),
    }


def _create_complaint_ticket_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "create_complaint_ticket")
    complaint = _extract_latest_user_message(state.get("messages") or []) or str(state.get("text") or "")
    result = create_support_ticket(
        issue_type="feedback",
        payload={"complaint": complaint},
        routing_result="complaint_queue",
        update_source="agent",
    )
    return {
        "tool_call": tool_name,
        "complaint_created": bool(result.get("ok")),
        "complaint_ticket_id": result.get("ticket_id"),
    }


def _get_delivery_period_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "delivery_period_lookup")
    context = dict(state.get("context_data") or {})
    ref = str(context.get("order_or_tracking") or "").strip().upper()
    if not ref:
        ref = _extract_order_or_tracking(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "order_or_tracking": ref}
    if not ref:
        return {**base, "delivery_info_found": False}
    payload = get_delivery_period(ref)
    if not payload:
        return {**base, "delivery_info_found": False}
    return {**base, "delivery_info_found": True, "delivery_period": payload}


def _cancel_order_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "cancel_order")
    oid = str((state.get("context_data") or {}).get("order_id_extracted") or "")
    if not oid:
        oid = _extract_order_id_from_conversation(state.get("messages") or [], state.get("text")) or ""
    base: dict[str, Any] = {"tool_call": tool_name, "order_id_extracted": oid}
    result = cancel_order_record(oid, update_source="agent")
    return {
        **base,
        "cancel_succeeded": bool(result.get("ok")),
        "cancel_reason": str(result.get("reason") or ""),
        "order_status": result.get("status") or (state.get("context_data") or {}).get("order_status"),
    }


def _create_refund_request_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "create_refund_request")
    context = dict(state.get("context_data") or {})
    oid = str(context.get("order_id_extracted") or "")
    if not oid:
        oid = _extract_order_id_from_conversation(state.get("messages") or [], state.get("text")) or ""
    reason = str(context.get("refund_reason") or _extract_latest_user_message(state.get("messages") or []) or "")
    base: dict[str, Any] = {
        "tool_call": tool_name,
        "order_id_extracted": oid,
        "refund_reason": reason,
    }
    result = create_refund_request(oid, reason, update_source="agent")
    return {
        **base,
        "refund_request_created": bool(result.get("ok")),
        "refund_request_reason": str(result.get("reason") or ""),
        "refund_request_id": result.get("refund_id"),
        "refund_decision": result.get("decision"),
    }


def _update_shipping_address_tool(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "update_shipping_address")
    context = dict(state.get("context_data") or {})
    oid = str(context.get("order_id_extracted") or "")
    if not oid:
        oid = _extract_order_id_from_conversation(state.get("messages") or [], state.get("text")) or ""
    new_address = str(context.get("new_address") or _extract_latest_user_message(state.get("messages") or []) or "")
    base: dict[str, Any] = {
        "tool_call": tool_name,
        "order_id_extracted": oid,
        "new_address": new_address,
    }
    result = update_shipping_address_record(oid, new_address, update_source="agent")
    return {
        **base,
        "shipping_address_updated": bool(result.get("ok")),
        "shipping_address_update_reason": str(result.get("reason") or ""),
        "shipping_address": result.get("shipping_address") or {"line": new_address},
    }


def _handle_interrupt_step(step: dict[str, Any], state: IssueGraphState, idx: int, todo: list[dict[str, Any]]) -> IssueGraphState:
    msg = str(step.get("message") or "Human approval required.")
    decision = _extract_escalation_decision(state.get("messages") or [])
    action_id = str(step.get("action_id") or f"{state.get('session_id', '')}:{step.get('id', idx)}:{uuid.uuid4().hex[:8]}")
    if decision == "accept":
        accept_msg = str(step.get("on_accept_message") or "Thanks. We have escalated your case to a human agent.")
        return {
            **state,
            "assistant_metadata": {
                **dict(state.get("assistant_metadata") or {}),
                "escalation_decision": "accept",
                "action_id": action_id,
                "step_id": step.get("id"),
            },
            "final_response": accept_msg,
            "current_step_index": len(todo),
        }
    if decision == "reject":
        reject_msg = str(step.get("on_reject_message") or "Understood. We will not escalate this request.")
        return {
            **state,
            "assistant_metadata": {
                **dict(state.get("assistant_metadata") or {}),
                "escalation_decision": "reject",
                "action_id": action_id,
                "step_id": step.get("id"),
            },
            "final_response": reject_msg,
            "current_step_index": len(todo),
        }
    meta = dict(state.get("assistant_metadata") or {})
    meta.update(
        {
            "pending_human_action": True,
            "action_type": str(step.get("action_type") or "escalation"),
            "action_id": action_id,
            "decision_required": ["accept", "reject"],
            "step_id": step.get("id"),
        }
    )
    return {**state, "assistant_metadata": meta, "final_response": msg, "current_step_index": len(todo)}


_CONDITION_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "in": lambda a, b: a in b,
    "exists": lambda a, _: a is not None,
}


def _evaluate_condition(
    condition: dict[str, Any],
    *,
    context_data: dict[str, Any],
    policy_constraints: dict[str, Any],
) -> dict[str, Any]:
    try:
        op = str(condition["op"])
        field = str(condition["field"])
        lhs = _value_from_path(field, context_data=context_data, policy_constraints=policy_constraints)
        rhs = (
            _value_from_path(
                str(condition.get("value_from") or ""),
                context_data=context_data,
                policy_constraints=policy_constraints,
            )
            if condition.get("value_from")
            else condition.get("value")
        )
        predicate = _CONDITION_OPS.get(op)
        if predicate is None:
            return {"passed": False, "reason": f"unsupported_operator:{op}", "actual_value": lhs, "expected_value": rhs}
        passed = bool(predicate(lhs, rhs))
        reason = "ok" if passed else str(condition.get("failure_reason") or f"condition_failed:{field}:{op}")
        return {"passed": passed, "reason": reason, "actual_value": lhs, "expected_value": rhs}
    except Exception as exc:  # noqa: BLE001
        return {
            "passed": False,
            "reason": f"condition_eval_error:{exc}",
            "actual_value": None,
            "expected_value": condition.get("value"),
        }


def _draft_response(state: IssueGraphState, step: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    provider = os.getenv("NO_ISSUE_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("NO_ISSUE_MODEL", "llama3.2").strip()
    system = os.getenv(
        "NO_ISSUE_SYSTEM_PROMPT",
        "You are a helpful assistant for a commerce chatbot. Reply concisely and helpfully.",
    ).strip()
    msgs = _messages_for_llm(state.get("messages") or [])
    summary = json.dumps(state.get("context_data") or {}, ensure_ascii=False)
    validation_summary = json.dumps(state.get("output_validation") or {}, ensure_ascii=False)
    failure_reasons = _collect_failure_reasons(state)
    failure_summary = json.dumps(failure_reasons, ensure_ascii=False)
    step_msg = str(step.get("message") or "").strip()
    user_prompt = (
        "Use only the resolved deterministic facts in context_data, policy_check_results, and output_validation. "
        "Do not decide eligibility or invent constraints.\n"
        "If the customer's request could not be completed or was blocked, explain why using the "
        f"failure_reasons JSON array below (prefer these exact customer-facing strings when they apply).\n"
        f"failure_reasons: {failure_summary}\n\n"
        f"Procedure context: {summary}\n"
        f"Validation context: {validation_summary}"
    )
    if step_msg:
        user_prompt = f"{step_msg}\n\n{user_prompt}"
    llm_messages: list[dict[str, str]] = []
    if system:
        llm_messages.append({"role": "system", "content": system})
    llm_messages.extend(msgs)
    llm_messages.append({"role": "user", "content": user_prompt})
    try:
        raw = chat_completion(provider=provider, model=model, messages=llm_messages)
        return (
            raw,
            {
                "provider": provider,
                "model": model,
                "messages": llm_messages,
                "raw_response": raw,
            },
        )
    except Exception as e:  # noqa: BLE001
        fallback = f"(Model error: {e})"
        return (
            fallback,
            {
                "provider": provider,
                "model": model,
                "messages": llm_messages,
                "raw_response": fallback,
                "error": str(e),
            },
        )


def _draft_order_cancel_terminal_response(state: IssueGraphState, step: dict[str, Any]) -> str | None:
    """Deterministic terminal replies for cancellation procedure outcomes."""
    if str(state.get("procedure_id") or "") != "order_cancel":
        return None
    step_id = str(step.get("id") or "")
    context = dict(state.get("context_data") or {})
    order_id = str(context.get("order_id_extracted") or "your order").strip() or "your order"
    cancel_succeeded = bool(context.get("cancel_succeeded"))
    cancel_reason = str(context.get("cancel_reason") or "").strip()

    if step_id == "confirm_cancelled":
        if not cancel_succeeded:
            return None
        return (
            f"Your order {order_id} has been cancelled successfully.\n\n"
            "Any eligible refund will be processed to your original payment method based on your payment provider timeline."
        )

    if step_id == "cancellation_not_allowed":
        if cancel_succeeded:
            return (
                f"Your order {order_id} has been cancelled successfully.\n\n"
                "No further cancellation action is needed."
            )
        reason = cancel_reason.replace("_", " ").strip() or "the cancellation could not be completed"
        return (
            f"I could not cancel order {order_id} because {reason}.\n\n"
            "If you want, I can help you with the next best option (for example, checking refund eligibility or escalating to support)."
        )

    if step_id == "order_not_found":
        return (
            "I could not find that order in our system.\n\n"
            "Please double-check the order number and share it again (example: ORD-12345)."
        )
    return None


def _draft_order_status_terminal_response(state: IssueGraphState, step: dict[str, Any]) -> str | None:
    """Deterministic terminal replies for order status lookup (no LLM paraphrase)."""
    if str(state.get("procedure_id") or "") != "order_status":
        return None
    step_id = str(step.get("id") or "")
    context = dict(state.get("context_data") or {})
    order_id = str(context.get("order_id_extracted") or "your order").strip() or "your order"
    order_found = bool(context.get("order_found"))
    status_raw = str(context.get("order_status") or "").strip()
    status_disp = status_raw.replace("_", " ").strip().lower() or "available"

    if step_id == "share_status":
        if not order_found:
            return (
                "I could not find that order in our system.\n\n"
                "Please double-check the order number and share it again (example: ORD-12345)."
            )
        return f"Order {order_id} is currently {status_disp}."

    if step_id == "order_not_found":
        if order_found:
            return f"Order {order_id} is currently {status_disp}."
        return (
            "I could not find that order in our system.\n\n"
            "Please double-check the order number and share it again (example: ORD-12345)."
        )
    return None


def _jump_to_step(state: IssueGraphState, next_step_id: str) -> IssueGraphState:
    todo = state.get("todo_list") or []
    for idx, item in enumerate(todo):
        if str(item.get("id")) == next_step_id:
            return {**state, "current_step_index": idx}
    return {**state, "current_step_index": len(todo)}


def _structured_executor_node(state: IssueGraphState) -> IssueGraphState:
    todo = state.get("todo_list") or []
    idx = int(state.get("current_step_index") or 0)
    turn_count = int(state.get("executor_turn_count") or 0) + 1
    if idx >= len(todo):
        return _with_stage_metadata(
            {**state, "executor_turn_count": turn_count},
            "structured_executor",
            {"done": True, "current_step_index": idx, "executor_turn_count": turn_count},
        )
    step = todo[idx]
    context = dict(state.get("context_data") or {})
    step_type = str(step.get("type") or "")
    tool_dispatch = {
        "check_order_status": _check_order_status,
        "product_catalog_lookup": _lookup_product_info,
        "product_info_lookup": _lookup_product_info_only,
        "product_price_lookup": _lookup_product_price,
        "product_availability_lookup": _lookup_product_availability,
        "refund_context_lookup": _lookup_refund_context,
        "payment_lookup": _lookup_payment,
        "list_payment_methods": _list_payment_methods_tool,
        "payment_refund_status": _track_refund_payment_tool,
        "check_invoice_value": _check_invoice_tool,
        "get_subscription_status": _get_subscription_status_tool,
        "unsubscribe_subscription": _unsubscribe_subscription_tool,
        "create_payment_issue_ticket": _create_payment_issue_ticket_tool,
        "create_contact_handoff": _create_contact_handoff_tool,
        "create_complaint_ticket": _create_complaint_ticket_tool,
        "delivery_period_lookup": _get_delivery_period_tool,
        "cancel_order": _cancel_order_tool,
        "create_refund_request": _create_refund_request_tool,
        "update_shipping_address": _update_shipping_address_tool,
    }

    if step_type == "tool_call":
        tool_name = str(step.get("tool") or "unknown_tool")
        context["tool_call"] = tool_name
        runner = tool_dispatch.get(tool_name)
        if not runner:
            meta = dict(state.get("assistant_metadata") or {})
            meta["tool_error"] = f"Unknown tool '{tool_name}'"
            return _with_stage_metadata(
                {
                    **state,
                    "assistant_metadata": meta,
                    "final_response": "I could not run a required backend tool for this request.",
                    "current_step_index": len(todo),
                    "context_data": context,
                    "executor_turn_count": turn_count,
                },
                "structured_executor",
                {
                    "step_id": step.get("id"),
                    "step_type": step_type,
                    "error": meta["tool_error"],
                    "executor_turn_count": turn_count,
                },
            )
        context.update(runner(step, state))
    elif step_type == "logic_gate":
        cond = step.get("condition") or {}
        gate_eval = _evaluate_condition(
            cond,
            context_data=context,
            policy_constraints=dict(state.get("policy_constraints") or {}),
        )
        branch = bool(gate_eval.get("passed"))
        target = str(step.get("on_true") if branch else step.get("on_false") or "")
        policy_checks = list(state.get("policy_check_results") or [])
        policy_checks.append(
            _build_policy_check_result(
                check_id=str(step.get("id") or "logic_gate"),
                passed=branch,
                reason=str(gate_eval.get("reason") or ""),
                actual_value=gate_eval.get("actual_value"),
                expected_value=gate_eval.get("expected_value"),
                source="procedure_gate",
                condition=cond if isinstance(cond, dict) else {},
            )
        )
        return _with_stage_metadata(
            _jump_to_step(
                {
                    **state,
                    "context_data": {
                        **context,
                        "policy_check_results": policy_checks,
                    },
                    "policy_check_results": policy_checks,
                    "executor_turn_count": turn_count,
                },
                target,
            ),
            "structured_executor",
            {
                "step_id": step.get("id"),
                "step_type": step_type,
                "gate_passed": branch,
                "gate_reason": gate_eval.get("reason"),
                "gate_actual_value": gate_eval.get("actual_value"),
                "gate_expected_value": gate_eval.get("expected_value"),
                "branch_target": target,
                "executor_turn_count": turn_count,
            },
        )
    elif step_type == "policy_check":
        phase = str(step.get("policy_phase") or "runtime").strip().lower() or "runtime"
        selected_rule_ids = {str(x).strip() for x in (step.get("policy_rules") or []) if str(x).strip()}
        constraints = dict(state.get("policy_constraints") or {})
        if selected_rule_ids:
            scoped_rules: list[dict[str, Any]] = []
            for rule in list(constraints.get("eligibility_rules") or []):
                if not isinstance(rule, dict):
                    continue
                if str(rule.get("id") or "").strip() in selected_rule_ids:
                    scoped_rules.append(rule)
            scoped_constraints = {**constraints, "eligibility_rules": scoped_rules}
        else:
            scoped_constraints = constraints
        rules_ok, rules_reason, rule_checks = _evaluate_policy_rules(
            policy_constraints=scoped_constraints,
            context_data=context,
            phase=phase,
        )
        normalized_checks = [{**check, "source": "procedure_policy_check"} for check in rule_checks]
        combined_checks = [*list(state.get("policy_check_results") or []), *normalized_checks]
        eligibility_ok = bool(constraints.get("eligible", True)) and rules_ok
        reason = (
            rules_reason
            or str(constraints.get("reason") or "").strip()
            or str(constraints.get("default_ineligible_reason") or "").strip()
        )
        target = str(step.get("on_true") if eligibility_ok else step.get("on_false") or "")
        return _with_stage_metadata(
            _jump_to_step(
                {
                    **state,
                    "context_data": {
                        **context,
                        "policy_eligible": eligibility_ok,
                        "policy_ineligibility_reason": "" if eligibility_ok else reason,
                        "policy_check_results": combined_checks,
                    },
                    "policy_constraints": {
                        **constraints,
                        "eligible": eligibility_ok,
                        "reason": "" if eligibility_ok else reason,
                    },
                    "policy_check_results": combined_checks,
                    "executor_turn_count": turn_count,
                },
                target,
            ),
            "structured_executor",
            {
                "step_id": step.get("id"),
                "step_type": step_type,
                "policy_phase": phase,
                "rules_evaluated": len(normalized_checks),
                "policy_eligible": eligibility_ok,
                "policy_reason": "" if eligibility_ok else reason,
                "branch_target": target,
                "executor_turn_count": turn_count,
            },
        )
    elif step_type == "interrupt":
        return _with_stage_metadata(
            _handle_interrupt_step(step, {**state, "context_data": context, "executor_turn_count": turn_count}, idx, todo),
            "structured_executor",
            {"step_id": step.get("id"), "step_type": step_type, "executor_turn_count": turn_count},
        )
    elif step_type == "llm_response":
        reasons = _collect_failure_reasons({**state, "context_data": context})
        context["failure_reasons"] = reasons
        state_with_ctx = {**state, "context_data": context}
        deterministic_reply = _draft_order_cancel_terminal_response(state_with_ctx, step)
        if deterministic_reply is None:
            deterministic_reply = _draft_order_status_terminal_response(state_with_ctx, step)
        llm_call: dict[str, Any] | None = None
        if deterministic_reply is None:
            reply, llm_call = _draft_response(state, step)
        else:
            reply = deterministic_reply
        return _with_stage_metadata(
            {
            **state,
            "context_data": context,
            "final_response": reply,
            "current_step_index": idx + 1,
            "executor_turn_count": turn_count,
            },
            "structured_executor",
            {"step_id": step.get("id"), "step_type": step_type, "executor_turn_count": turn_count},
            llm_call=llm_call,
        )
    else:
        meta = dict(state.get("assistant_metadata") or {})
        meta["step_error"] = f"Unknown step type '{step_type}'"
        return _with_stage_metadata(
            {
            **state,
            "context_data": context,
            "assistant_metadata": meta,
            "final_response": "I hit an unsupported procedure step and cannot continue safely.",
            "current_step_index": len(todo),
            "executor_turn_count": turn_count,
            },
            "structured_executor",
            {
                "step_id": step.get("id"),
                "step_type": step_type,
                "error": meta["step_error"],
                "executor_turn_count": turn_count,
            },
        )

    return _with_stage_metadata(
        {**state, "context_data": context, "current_step_index": idx + 1, "executor_turn_count": turn_count},
        "structured_executor",
        {"step_id": step.get("id"), "step_type": step_type, "executor_turn_count": turn_count},
    )


def _run_output_validation(state: IssueGraphState) -> dict[str, Any]:
    context = dict(state.get("context_data") or {})
    checks: dict[str, Any] = {}
    policy_constraints = dict(state.get("policy_constraints") or {})
    bp = get_blueprint_by_category_intent(state.get("category") or "", state.get("intent") or "")
    for assertion in list((as_dict(x) for x in (bp.expected_outcomes if bp else []))):
        check_id = str(assertion.get("id") or "expected_outcome")
        eval_out = _evaluate_condition(assertion, context_data=context, policy_constraints=policy_constraints)
        checks[check_id] = {
            "valid": bool(eval_out.get("passed")),
            "reason": str(eval_out.get("reason") or ""),
            "actual_value": eval_out.get("actual_value"),
            "expected_value": eval_out.get("expected_value"),
        }

    order_id = str(context.get("order_id_extracted") or "").strip()
    if order_id and bool(context.get("cancel_succeeded")):
        db_row = get_order_status(order_id)
        db_status = str((db_row or {}).get("status") or "").strip().lower()
        checks["order_cancel_db_verification"] = {
            "valid": bool(db_status == "cancelled"),
            "reason": "ok" if db_status == "cancelled" else "order_status_not_cancelled_after_cancel_succeeded",
            "expected_value": "cancelled",
            "actual_value": db_status,
            "order_id": order_id,
        }
    all_valid = all(bool(item.get("valid")) for item in checks.values()) if checks else True
    return {"checks": checks, "all_valid": all_valid}


def _build_context_summary(state: IssueGraphState) -> dict[str, Any]:
    context = _compact_context_data(dict(state.get("context_data") or {}))
    return {
        "session_id": str(state.get("session_id") or ""),
        "category": str(state.get("category") or ""),
        "intent": str(state.get("intent") or ""),
        "problem_to_solve": str(state.get("problem_to_solve") or ""),
        "procedure_id": str(state.get("procedure_id") or ""),
        "outcome_status": str(state.get("outcome_status") or ""),
        "validation_missing": list(state.get("validation_missing") or []),
        "context_data": context,
        "failure_reasons": list(_collect_failure_reasons(state)),
    }


def _with_final_order_status(state: IssueGraphState) -> IssueGraphState:
    context = dict(state.get("context_data") or {})
    order_id = str(context.get("order_id_extracted") or "").strip()
    if not order_id:
        return state
    row = get_order_status(order_id)
    if not row:
        return state
    return {
        **state,
        "context_data": {
            **context,
            "order_status_after": row.get("status"),
        },
    }


def _outcome_validator_node(state: IssueGraphState) -> IssueGraphState:
    state = _with_final_order_status(state)
    status = state.get("outcome_status")
    if status in {"needs_more_data", "policy_ineligible", "pending_escalation"}:
        out_terminal: IssueGraphState = {
            **state,
            "output_validation": _run_output_validation(state),
        }
        out_terminal = _merge_failure_reasons_into_state(out_terminal)  # type: ignore[assignment]
        out_terminal["context_summary"] = _build_context_summary(out_terminal)  # type: ignore[index]
        return _with_stage_metadata(out_terminal, "outcome_validator", {"outcome_status": status})
    meta = dict(state.get("assistant_metadata") or {})
    if meta.get("pending_human_action") or meta.get("escalation_decision") == "accept":
        status = "pending_escalation"
    elif meta.get("tool_error"):
        status = "tool_error"
    elif meta.get("step_error"):
        status = "step_error"
    elif graph_suggests_session_resolved(state):
        status = "resolved"
    else:
        status = "unresolvable"
    out: IssueGraphState = {**state, "outcome_status": status}
    output_validation = _run_output_validation(out)
    if not output_validation.get("all_valid"):
        out["outcome_status"] = "unresolvable"
    out["output_validation"] = output_validation
    out = _merge_failure_reasons_into_state(out)  # type: ignore[assignment]
    reasons = _collect_failure_reasons(out)
    augmented = _maybe_augment_blocked_final_response(
        outcome_status=str(out.get("outcome_status") or ""),
        final_response=str(out.get("final_response") or ""),
        reasons=reasons,
    )
    if augmented is not None:
        out["final_response"] = augmented
    out["context_summary"] = _build_context_summary(out)
    return _with_stage_metadata(
        out,
        "outcome_validator",
        {
            "outcome_status": out.get("outcome_status"),
            "output_validation_all_valid": bool(output_validation.get("all_valid")),
        },
    )


def _human_escalation_node(state: IssueGraphState) -> IssueGraphState:
    bundle = {
        "session_id": str(state.get("session_id") or ""),
        "category": str(state.get("category") or ""),
        "intent": str(state.get("intent") or ""),
        "problem_to_solve": str(state.get("problem_to_solve") or ""),
        "transcript": state.get("messages") or [],
        "context_data": dict(state.get("context_data") or {}),
        "context_summary": dict(state.get("context_summary") or {}),
        "policy_constraints": state.get("policy_constraints") or {},
        "procedure_id": str(state.get("procedure_id") or ""),
        "last_step_id": (
            (state.get("todo_list") or [])[max(0, int(state.get("current_step_index") or 0) - 1)].get("id")
            if state.get("todo_list")
            else None
        ),
        "outcome_status": str(state.get("outcome_status") or "unresolvable"),
        "reason": "Escalated by outcome validator",
    }
    meta = dict(state.get("assistant_metadata") or {})
    meta["escalated"] = True
    return _with_stage_metadata(
        {**state, "escalation_bundle": bundle, "assistant_metadata": meta},
        "human_escalation",
        {"escalated": True, "outcome_status": state.get("outcome_status")},
    )


def _should_continue(state: IssueGraphState) -> Literal["continue", "end"]:
    if int(state.get("executor_turn_count") or 0) >= MAX_NODE_TURNS:
        return "end"
    todo = state.get("todo_list") or []
    idx = int(state.get("current_step_index") or 0)
    if idx >= len(todo):
        return "end"
    return "continue"


def _await_user_input_node(state: IssueGraphState) -> IssueGraphState:
    payload = {
        "type": "needs_more_data",
        "validation_missing": list(state.get("validation_missing") or []),
        "prompt": str(state.get("final_response") or ""),
        "validation_wait_count": int(state.get("validation_wait_count") or 0),
        "validation_wait_limit": int(state.get("validation_wait_limit") or _validation_wait_limit()),
    }
    resumed = interrupt(payload)
    updates: dict[str, Any] = {}
    if isinstance(resumed, dict):
        if "text" in resumed:
            updates["text"] = str(resumed.get("text") or "")
        if isinstance(resumed.get("messages"), list):
            updates["messages"] = resumed.get("messages") or []
    return _with_stage_metadata(
        {**state, **updates},
        "await_user_input",
        {
            "validation_wait_count": payload["validation_wait_count"],
            "validation_wait_limit": payload["validation_wait_limit"],
            "missing_count": len(payload["validation_missing"]),
        },
    )


def _route_after_category(state: IssueGraphState) -> Literal["no_issue_direct", "classify_intent"]:
    category = normalize_category_key(state.get("category") or "unknown")
    confidence = float(state.get("confidence") or 0.0)
    if category == "no_issue" or confidence < _category_confidence_threshold():
        return "no_issue_direct"
    return "classify_intent"


def _route_after_validation(
    state: IssueGraphState,
) -> Literal["structured_executor", "outcome_validator", "await_user_input", "end"]:
    if str(state.get("outcome_status") or "") in {"pending_escalation", "policy_ineligible"}:
        return "outcome_validator"
    if state.get("validation_ok") is False:
        if (
            bool(state.get("enable_persistent_wait_interrupt"))
            and int(state.get("validation_wait_count") or 0) < int(state.get("validation_wait_limit") or _validation_wait_limit())
        ):
            return "await_user_input"
        return "end"
    return "structured_executor"


def _route_after_outcome(state: IssueGraphState) -> Literal["human_escalation", "end"]:
    if str(state.get("outcome_status") or "") in {
        "pending_escalation",
        "unresolvable",
        "tool_error",
        "step_error",
        "policy_ineligible",
    }:
        return "human_escalation"
    return "end"

def build_issue_classification_graph(*, checkpointer: Any | None = None):
    g: StateGraph[IssueGraphState] = StateGraph(IssueGraphState)
    g.add_node("classify_category", _classify_category_node)
    g.add_node("no_issue_direct", _no_issue_direct_node)
    g.add_node("classify_intent", _classify_intent_node)
    g.add_node("specialist_router", _specialist_router_node)
    g.add_node("fetch_procedure", _fetch_procedure_node)
    g.add_node("policy_load", _policy_load_node)
    g.add_node("validate_required", _data_and_eligibility_validator_node)
    g.add_node("await_user_input", _await_user_input_node)
    g.add_node("structured_executor", _structured_executor_node)
    g.add_node("outcome_validator", _outcome_validator_node)
    g.add_node("human_escalation", _human_escalation_node)
    g.set_entry_point("classify_category")
    g.add_conditional_edges(
        "classify_category",
        _route_after_category,
        {"no_issue_direct": "no_issue_direct", "classify_intent": "classify_intent"},
    )
    g.add_edge("no_issue_direct", END)
    g.add_edge("classify_intent", "specialist_router")
    g.add_edge("specialist_router", "fetch_procedure")
    g.add_edge("fetch_procedure", "policy_load")
    g.add_edge("policy_load", "validate_required")
    g.add_conditional_edges(
        "validate_required",
        _route_after_validation,
        {
            "structured_executor": "structured_executor",
            "outcome_validator": "outcome_validator",
            "await_user_input": "await_user_input",
            "end": END,
        },
    )
    g.add_edge("await_user_input", "validate_required")
    g.add_conditional_edges(
        "structured_executor",
        _should_continue,
        {"continue": "structured_executor", "end": "outcome_validator"},
    )
    g.add_conditional_edges(
        "outcome_validator",
        _route_after_outcome,
        {"human_escalation": "human_escalation", "end": END},
    )
    g.add_edge("human_escalation", END)
    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


_COMPILED = None


def get_issue_classification_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_issue_classification_graph()
    return _COMPILED


def run_conversation_graph(
    *,
    text: str,
    session_id: str,
    messages: list[dict[str, Any]],
    issue_locked: bool = False,
    locked_category: str | None = None,
    locked_intent: str | None = None,
    locked_problem_to_solve: str | None = None,
    locked_confidence: float | None = None,
    initial_validation_wait_count: int = 0,
    initial_validation_wait_limit: int | None = None,
) -> dict[str, Any]:
    graph = get_issue_classification_graph()
    cat0 = "unknown"
    intent0 = ""
    problem0 = ""
    conf0 = 0.0
    if issue_locked and locked_intent:
        cat0 = normalize_category_key(locked_category or get_category_for_stored_intent(locked_intent) or "unknown")
        intent0 = str(locked_intent).strip()
        problem0 = str(locked_problem_to_solve or "").strip()
        conf0 = float(locked_confidence) if locked_confidence is not None else 1.0

    wait_limit = initial_validation_wait_limit or _validation_wait_limit()
    out = graph.invoke(
        {
            "text": text or "",
            "session_id": session_id or "",
            "messages": messages,
            "issue_locked": bool(issue_locked and locked_intent),
            "category": cat0,
            "intent": intent0,
            "problem_to_solve": problem0,
            "confidence": conf0,
            "procedure_id": "",
            "todo_list": [],
            "current_step_index": 0,
            "context_data": {},
            "validation_ok": None,
            "validation_missing": [],
            "eligibility_ok": None,
            "specialist_agent_id": "",
            "tool_registry_scope": "",
            "procedure_namespace": "",
            "policy_constraints": None,
            "policy_check_results": [],
            "outcome_status": None,
            "escalation_bundle": None,
            "final_response": None,
            "assistant_metadata": {},
            "stage_metadata": {},
            "agent_state": {"stage": "classify_category"},
            "validation_wait_count": max(0, int(initial_validation_wait_count)),
            "validation_wait_limit": max(1, int(wait_limit)),
            "output_validation": {},
            "context_summary": {},
            "classify_intent_attempts": 0,
            "policy_load_attempts": 0,
            "executor_turn_count": 0,
            "enable_persistent_wait_interrupt": False,
        }
    )
    resolved_by_graph = graph_suggests_session_resolved(out)  # type: ignore[arg-type]
    context_data = _compact_context_data(dict(out.get("context_data") or {}))
    policy_constraints = dict(out.get("policy_constraints") or {})
    agent_state = _build_agent_state_snapshot(out)  # type: ignore[arg-type]
    stage_metadata = dict(out.get("stage_metadata") or {})
    agent_trace = build_agent_trace(agent_state=agent_state, stage_metadata=stage_metadata)
    assistant_metadata = {
        **(out.get("assistant_metadata") or {}),
        "outcome_status": out.get("outcome_status"),
        "specialist_agent_id": out.get("specialist_agent_id"),
        "agent_state": agent_state,
        "agent_trace": agent_trace,
        "validation_wait_count": out.get("validation_wait_count"),
        "validation_wait_limit": out.get("validation_wait_limit"),
        "failure_reasons": _collect_failure_reasons(out),  # type: ignore[arg-type]
        "tool_error": (out.get("assistant_metadata") or {}).get("tool_error"),
        "step_error": (out.get("assistant_metadata") or {}).get("step_error"),
        "pending_human_action": (out.get("assistant_metadata") or {}).get("pending_human_action"),
        "escalation_decision": (out.get("assistant_metadata") or {}).get("escalation_decision"),
    }
    return {
        "text": out.get("text", ""),
        "category": str(out.get("category", "unknown")),
        "intent": str(out.get("intent", "")),
        "problem_to_solve": str(out.get("problem_to_solve", "")),
        "confidence": float(out.get("confidence", 0.0)),
        "procedure_id": str(out.get("procedure_id", "")),
        "validation_ok": out.get("validation_ok"),
        "validation_missing": list(out.get("validation_missing") or []),
        "assistant_reply": out.get("final_response"),
        "assistant_metadata": assistant_metadata,
        "context_data": context_data,
        "policy_constraints": policy_constraints,
        "policy_check_results": list(out.get("policy_check_results") or []),
        "agent_state": agent_state,
        "stage_metadata": stage_metadata,
        "agent_trace": agent_trace,
        "output_validation": dict(out.get("output_validation") or {}),
        "context_summary": dict(out.get("context_summary") or {}),
        "validation_wait_count": int(out.get("validation_wait_count") or 0),
        "validation_wait_limit": int(out.get("validation_wait_limit") or _validation_wait_limit()),
        "eligibility_ok": out.get("eligibility_ok"),
        "session_resolved_by_graph": resolved_by_graph,
    }
