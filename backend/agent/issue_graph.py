"""Procedure-driven LangGraph: category -> intent -> procedure execution."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from backend.agent.procedures import (
    as_dict,
    get_blueprint_by_category_intent,
    get_fallback_blueprint,
    load_blueprints,
)
from backend.db.intents_repo import get_intents_for_category
from backend.db.orders_repo import get_order_status
from backend.db.postgres import postgres_configured
from backend.db.products_repo import lookup_product
from backend.db.refunds_repo import get_refund_context
from backend.llm.providers import chat_completion, extract_json_object
from backend.rag.policy_retriever import search_policy_docs
from backend.rag.query_classifier import ClassificationResult, get_query_classifier
from backend.rag.required_fields import normalize_category_key


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
    final_response: str | None
    assistant_metadata: dict[str, Any]


def _classify_category_node(state: IssueGraphState) -> IssueGraphState:
    text = state.get("text") or ""
    if state.get("issue_locked"):
        return {**state, "text": text}
    qc = get_query_classifier()
    result: ClassificationResult = qc.classify(text)
    return {
        **state,
        "text": text,
        "category": result.category,
        "confidence": result.confidence,
    }


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
    try:
        reply = chat_completion(provider=provider, model=model, messages=llm_messages)
    except Exception as e:  # noqa: BLE001
        reply = f"(Model error: {e})"
        meta["error"] = str(e)

    return {
        **state,
        "intent": "no_issue_chat",
        "problem_to_solve": "",
        "procedure_id": "no_issue_chat",
        "todo_list": [],
        "current_step_index": 0,
        "final_response": reply,
        "assistant_metadata": meta,
    }


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


def _load_allowed_intents(category: str) -> list[str]:
    if not postgres_configured():
        return []
    try:
        return get_intents_for_category(category)
    except Exception:  # noqa: BLE001
        return []


def _classify_intent_node(state: IssueGraphState) -> IssueGraphState:
    category = normalize_category_key(state.get("category") or "unknown")
    meta = dict(state.get("assistant_metadata") or {})
    if state.get("issue_locked"):
        meta["intent_classifier"] = "session_locked"
        meta["intent_candidates"] = [state.get("intent") or ""]
        return {
            **state,
            "intent": (state.get("intent") or "").strip(),
            "problem_to_solve": str(state.get("problem_to_solve") or "").strip(),
            "assistant_metadata": meta,
        }

    provider = os.getenv("INTENT_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("INTENT_MODEL", "llama3.2").strip()
    confidence = float(state.get("confidence") or 0.0)
    user_messages = _user_messages_from_session(state.get("messages") or [])
    messages_json = json.dumps(user_messages, ensure_ascii=False)
    allowed_intents = _load_allowed_intents(category)
    if allowed_intents:
        intents_bullets = "\n".join(f"- {item}" for item in allowed_intents)
        system_prompt = (
            "You classify a customer support session into a stable procedure intent and summarize "
            "the user's problem to solve.\n"
            "Use ONLY the provided category, confidence score, and user messages.\n"
            "Respond with ONLY a JSON object shaped as:\n"
            '{"intent":"snake_case_short_label","problem_to_solve":"one concise sentence"}\n'
            "Rules:\n"
            "- intent must be stable for the full session.\n"
            "- intent MUST be one of the allowed intents listed below.\n"
            f"- if none fits, use {category}_general.\n"
            "- problem_to_solve should capture the concrete user problem for this session.\n"
            "- Keep both values concise and deterministic.\n\n"
            f"Allowed intents for category '{category}':\n{intents_bullets}"
        )
    else:
        system_prompt = (
            "You classify a customer support session into a stable procedure intent and summarize "
            "the user's problem to solve.\n"
            "Use ONLY the provided category, confidence score, and user messages.\n"
            "Respond with ONLY a JSON object shaped as:\n"
            '{"intent":"snake_case_short_label","problem_to_solve":"one concise sentence"}\n'
            "Rules:\n"
            "- intent must be stable for the full session.\n"
            "- problem_to_solve should capture the concrete user problem for this session.\n"
            "- Keep both values concise and deterministic."
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
    try:
        raw = chat_completion(provider=provider, model=model, messages=llm_messages)
        data = extract_json_object(raw)
    except Exception as e:  # noqa: BLE001
        data = {}
        meta["intent_classifier_error"] = str(e)

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
    meta["intent_candidates"] = allowed_intents if allowed_intents else [intent]
    return {
        **state,
        "intent": intent,
        "problem_to_solve": problem_to_solve,
        "assistant_metadata": meta,
    }


def _fetch_procedure_node(state: IssueGraphState) -> IssueGraphState:
    category = normalize_category_key(state.get("category") or "unknown")
    intent = (state.get("intent") or "").strip()
    bp = get_blueprint_by_category_intent(category, intent)
    if bp is None:
        bp = get_fallback_blueprint(category) or get_fallback_blueprint("unknown")
    if bp is None:
        return {
            **state,
            "procedure_id": "",
            "todo_list": [],
            "current_step_index": 0,
            "final_response": "I could not map this request to a procedure.",
        }
    return {
        **state,
        "procedure_id": bp.id,
        "intent": bp.intent,
        "todo_list": [as_dict(step) for step in bp.steps],
        "current_step_index": 0,
        "context_data": dict(state.get("context_data") or {}),
    }


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


def _validate_required_data_node(state: IssueGraphState) -> IssueGraphState:
    bp = get_blueprint_by_category_intent(state.get("category") or "", state.get("intent") or "")
    if bp is None:
        return state
    required = [as_dict(x) for x in bp.required_data]
    if not required:
        return {**state, "validation_ok": True, "validation_missing": []}

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
    try:
        raw = chat_completion(provider=provider, model=model, messages=msgs)
    except Exception as e:  # noqa: BLE001
        return {
            **state,
            "validation_ok": False,
            "validation_missing": [],
            "final_response": f"Validation could not run: {e}",
            "assistant_metadata": {
                **dict(state.get("assistant_metadata") or {}),
                "branch": "validate",
                "error": str(e),
            },
        }
    data = extract_json_object(raw)
    valid = bool(data.get("valid"))
    missing = data.get("missing_field_names") or data.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []
    missing_strs = [str(x) for x in missing if x]

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
    }
    out: IssueGraphState = {
        **state,
        "validation_ok": valid,
        "validation_missing": missing_strs,
        "final_response": assistant_reply or state.get("final_response"),
        "assistant_metadata": meta,
    }
    if not valid:
        out["current_step_index"] = len(state.get("todo_list") or [])
    return out


_ORDER_NUMBER_RE = re.compile(r"\b(ORD-[A-Z0-9]+)\b", re.IGNORECASE)
_ORDER_ID_ONLY_RE = re.compile(r"^\s*ORD-[A-Z0-9]+\s*$", re.IGNORECASE)
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


def _extract_escalation_decision(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages or []):
        if str(m.get("role")) != "user":
            continue
        text = str(m.get("content") or "")
        mo = _ESCALATION_DECISION_RE.search(text)
        if mo:
            return mo.group(1).lower()
    return None


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
    return {
        **base,
        "order_found": True,
        "order_status": row.get("status"),
        "order_total_amount": row.get("total_amount"),
        "order_data": row,
    }


def _retrieve_policy(step: dict[str, Any], state: IssueGraphState) -> dict[str, Any]:
    tool_name = str(step.get("tool") or "policy_search")
    text = str(state.get("text") or "").strip()
    problem_to_solve = str(state.get("problem_to_solve") or "").strip()
    category = str(state.get("category") or "").strip().replace("_", " ")
    intent = str(state.get("intent") or "").strip().replace("_", " ")

    query_parts: list[str] = []
    if problem_to_solve:
        query_parts.append(problem_to_solve)
    if text and not _ORDER_ID_ONLY_RE.match(text):
        query_parts.append(text)
    if category:
        query_parts.append(category)
    if intent and intent.lower() != category.lower():
        query_parts.append(intent)

    deduped_parts: list[str] = []
    seen: set[str] = set()
    for part in query_parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_parts.append(part)

    query = " ".join(deduped_parts).strip() or text or "policy"
    docs = search_policy_docs(query)
    return {
        "policy_found": bool(docs),
        "policy_tool": tool_name,
        "policy_query": query,
        "retrieved_docs": docs,
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


def _evaluate_condition(condition: str, context_data: dict[str, Any]) -> bool:
    try:
        return bool(eval(condition, {"__builtins__": {}}, dict(context_data)))  # noqa: S307
    except Exception:  # noqa: BLE001
        return False


def _draft_response(state: IssueGraphState, step: dict[str, Any]) -> str:
    provider = os.getenv("NO_ISSUE_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("NO_ISSUE_MODEL", "llama3.2").strip()
    system = os.getenv(
        "NO_ISSUE_SYSTEM_PROMPT",
        "You are a helpful assistant for a commerce chatbot. Reply concisely and helpfully.",
    ).strip()
    msgs = _messages_for_llm(state.get("messages") or [])
    summary = json.dumps(state.get("context_data") or {}, ensure_ascii=False)
    step_msg = str(step.get("message") or "").strip()
    user_prompt = f"Procedure context: {summary}"
    if step_msg:
        user_prompt = f"{step_msg}\n\n{user_prompt}"
    llm_messages: list[dict[str, str]] = []
    if system:
        llm_messages.append({"role": "system", "content": system})
    llm_messages.extend(msgs)
    llm_messages.append({"role": "user", "content": user_prompt})
    try:
        return chat_completion(provider=provider, model=model, messages=llm_messages)
    except Exception as e:  # noqa: BLE001
        return f"(Model error: {e})"


def _jump_to_step(state: IssueGraphState, next_step_id: str) -> IssueGraphState:
    todo = state.get("todo_list") or []
    for idx, item in enumerate(todo):
        if str(item.get("id")) == next_step_id:
            return {**state, "current_step_index": idx}
    return {**state, "current_step_index": len(todo)}


def _structured_executor_node(state: IssueGraphState) -> IssueGraphState:
    todo = state.get("todo_list") or []
    idx = int(state.get("current_step_index") or 0)
    if idx >= len(todo):
        return state
    step = todo[idx]
    context = dict(state.get("context_data") or {})
    step_type = str(step.get("type") or "")
    tool_dispatch = {
        "check_order_status": _check_order_status,
        "product_catalog_lookup": _lookup_product_info,
        "refund_context_lookup": _lookup_refund_context,
    }

    if step_type == "retrieval":
        context.update(_retrieve_policy(step, state))
    elif step_type == "validate_required_data":
        # Required-data validation is already handled by graph node; keep compatibility if present in YAML.
        context["validate_required_data"] = True
    elif step_type == "tool_call":
        tool_name = str(step.get("tool") or "unknown_tool")
        context["tool_call"] = tool_name
        runner = tool_dispatch.get(tool_name)
        if not runner:
            meta = dict(state.get("assistant_metadata") or {})
            meta["tool_error"] = f"Unknown tool '{tool_name}'"
            return {
                **state,
                "assistant_metadata": meta,
                "final_response": "I could not run a required backend tool for this request.",
                "current_step_index": len(todo),
            }
        context.update(runner(step, state))
    elif step_type == "logic_gate":
        cond = str(step.get("condition") or "False")
        branch = _evaluate_condition(cond, context)
        target = str(step.get("on_true") if branch else step.get("on_false") or "")
        return _jump_to_step({**state, "context_data": context}, target)
    elif step_type == "interrupt":
        return _handle_interrupt_step(step, {**state, "context_data": context}, idx, todo)
    elif step_type == "llm_response":
        reply = _draft_response(state, step)
        return {
            **state,
            "context_data": context,
            "final_response": reply,
            "current_step_index": idx + 1,
        }
    else:
        meta = dict(state.get("assistant_metadata") or {})
        meta["step_error"] = f"Unknown step type '{step_type}'"
        return {
            **state,
            "context_data": context,
            "assistant_metadata": meta,
            "final_response": "I hit an unsupported procedure step and cannot continue safely.",
            "current_step_index": len(todo),
        }

    return {**state, "context_data": context, "current_step_index": idx + 1}


def _should_continue(state: IssueGraphState) -> Literal["continue", "end"]:
    todo = state.get("todo_list") or []
    idx = int(state.get("current_step_index") or 0)
    if idx >= len(todo):
        return "end"
    return "continue"


def _route_after_category(state: IssueGraphState) -> Literal["no_issue_direct", "classify_intent"]:
    category = normalize_category_key(state.get("category") or "unknown")
    if category == "no_issue":
        return "no_issue_direct"
    return "classify_intent"

def build_issue_classification_graph():
    g: StateGraph[IssueGraphState] = StateGraph(IssueGraphState)
    g.add_node("classify_category", _classify_category_node)
    g.add_node("no_issue_direct", _no_issue_direct_node)
    g.add_node("classify_intent", _classify_intent_node)
    g.add_node("fetch_procedure", _fetch_procedure_node)
    g.add_node("validate_required", _validate_required_data_node)
    g.add_node("structured_executor", _structured_executor_node)
    g.set_entry_point("classify_category")
    g.add_conditional_edges(
        "classify_category",
        _route_after_category,
        {"no_issue_direct": "no_issue_direct", "classify_intent": "classify_intent"},
    )
    g.add_edge("no_issue_direct", END)
    g.add_edge("classify_intent", "fetch_procedure")
    g.add_edge("fetch_procedure", "validate_required")
    g.add_edge("validate_required", "structured_executor")
    g.add_conditional_edges(
        "structured_executor",
        _should_continue,
        {"continue": "structured_executor", "end": END},
    )
    return g.compile()


_COMPILED = None


def get_issue_classification_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_issue_classification_graph()
    return _COMPILED


def run_issue_classification(text: str) -> dict[str, Any]:
    """Backward-compatible: Bento classification only (no LLM branches)."""
    qc = get_query_classifier()
    result: ClassificationResult = qc.classify(text or "")
    return {
        "text": text or "",
        "category": result.category,
        "intent": "",
        "confidence": result.confidence,
        "procedure_id": "",
        "validation_ok": None,
        "validation_missing": [],
        "assistant_reply": None,
        "assistant_metadata": {},
    }


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
            "final_response": None,
            "assistant_metadata": {},
        }
    )
    resolved_by_graph = graph_suggests_session_resolved(out)  # type: ignore[arg-type]
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
        "assistant_metadata": out.get("assistant_metadata") or {},
        "session_resolved_by_graph": resolved_by_graph,
    }
