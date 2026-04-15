"""
LangGraph: classify (Bento) -> route -> no_issue LLM or validation LLM -> END.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from backend.llm.providers import chat_completion, extract_json_object
from backend.rag.query_classifier import ClassificationResult, get_query_classifier
from backend.rag.required_fields import (
    build_missing_prompts,
    get_category_spec,
    normalize_category_key,
)


class IssueGraphState(TypedDict, total=False):
    text: str
    session_id: str
    messages: list[dict[str, Any]]
    category: str
    confidence: float
    validation_ok: bool | None
    validation_missing: list[str]
    assistant_reply: str | None
    assistant_metadata: dict[str, Any]


def _classify_node(state: IssueGraphState) -> IssueGraphState:
    qc = get_query_classifier()
    text = state.get("text") or ""
    result: ClassificationResult = qc.classify(text)
    return {
        **state,
        "text": text,
        "category": result.category,
        "confidence": result.confidence,
    }


def _route_after_classify(state: IssueGraphState) -> Literal["no_issue", "validate"]:
    cat = normalize_category_key(state.get("category") or "")
    if cat == "no_issue":
        return "no_issue"
    return "validate"


def _messages_for_llm(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        out.append({"role": role, "content": content})
    return out


def _no_issue_node(state: IssueGraphState) -> IssueGraphState:
    provider = os.getenv("NO_ISSUE_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("NO_ISSUE_MODEL", "llama3.2").strip()
    system = os.getenv(
        "NO_ISSUE_SYSTEM_PROMPT",
        "You are a helpful assistant for a commerce chatbot. Reply concisely and helpfully.",
    ).strip()
    msgs = _messages_for_llm(state.get("messages") or [])
    llm_messages: list[dict[str, str]] = []
    if system:
        llm_messages.append({"role": "system", "content": system})
    llm_messages.extend(msgs)
    try:
        reply = chat_completion(provider=provider, model=model, messages=llm_messages)
    except Exception as e:  # noqa: BLE001
        reply = f"(Model error: {e})"
    meta = {
        "branch": "no_issue",
        "model_provider": provider,
        "model": model,
    }
    return {
        **state,
        "validation_ok": None,
        "validation_missing": [],
        "assistant_reply": reply or "I'm here to help.",
        "assistant_metadata": meta,
    }


def _validation_node(state: IssueGraphState) -> IssueGraphState:
    provider = os.getenv("VALIDATION_MODEL_PROVIDER", "ollama").strip().lower()
    model = os.getenv("VALIDATION_MODEL", "llama3.2").strip()
    category = str(state.get("category") or "unknown")
    spec = get_category_spec(category)
    transcript = json.dumps(state.get("messages") or [], ensure_ascii=False)
    if spec:
        req = json.dumps(spec.get("required_fields") or [], ensure_ascii=False)
        display = str(spec.get("display_name") or category)
        sys_prompt = (
            "You validate whether the user provided all REQUIRED information for a support case.\n"
            f"Category: {category} ({display}).\n"
            f"Required fields definition (JSON): {req}\n"
            "Each required field has a name. Decide if the conversation contains a plausible value for each.\n"
            'Reply with ONLY a JSON object: {"valid": true|false, "missing_field_names": ["name1", ...], "notes": "short"}\n'
            "missing_field_names must use the exact field names from the required_fields list."
        )
    else:
        sys_prompt = (
            "You validate whether the user message is specific enough to open a support ticket.\n"
            f"Category: {category}.\n"
            f"Transcript (JSON): {transcript}\n"
            'Reply with ONLY JSON: {"valid": true|false, "missing_field_names": [], "assistant_reply": "If invalid, a short question to the user."}'
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
            "assistant_reply": f"Validation could not run: {e}",
            "assistant_metadata": {"branch": "validate", "error": str(e)},
        }
    data = extract_json_object(raw)
    valid = bool(data.get("valid"))
    missing = data.get("missing_field_names") or data.get("missing_fields") or []
    if not isinstance(missing, list):
        missing = []
    missing_strs = [str(x) for x in missing if x]

    assistant_reply: str | None = None
    if spec and not valid:
        assistant_reply = build_missing_prompts(spec, missing_strs)
        if not assistant_reply.strip():
            assistant_reply = (
                data.get("assistant_reply")
                or data.get("reply")
                or "Please provide the missing details so we can help."
            )
    elif not spec and not valid:
        assistant_reply = str(
            data.get("assistant_reply") or data.get("reply") or "Could you share more details?"
        )
    elif valid:
        assistant_reply = str(
            data.get("assistant_reply")
            or data.get("reply")
            or f"Thanks — we have what we need for your {category} request."
        )

    meta = {
        "branch": "validate",
        "model_provider": provider,
        "model": model,
        "validation_notes": data.get("notes"),
    }
    return {
        **state,
        "validation_ok": valid,
        "validation_missing": missing_strs,
        "assistant_reply": assistant_reply,
        "assistant_metadata": meta,
    }


def build_issue_classification_graph():
    g: StateGraph[IssueGraphState] = StateGraph(IssueGraphState)
    g.add_node("classify", _classify_node)
    g.add_node("no_issue", _no_issue_node)
    g.add_node("validate", _validation_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", _route_after_classify, {"no_issue": "no_issue", "validate": "validate"})
    g.add_edge("no_issue", END)
    g.add_edge("validate", END)
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
        "confidence": result.confidence,
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
) -> dict[str, Any]:
    graph = get_issue_classification_graph()
    out = graph.invoke(
        {
            "text": text or "",
            "session_id": session_id or "",
            "messages": messages,
            "category": "unknown",
            "confidence": 0.0,
            "validation_ok": None,
            "validation_missing": [],
            "assistant_reply": None,
            "assistant_metadata": {},
        }
    )
    return {
        "text": out.get("text", ""),
        "category": str(out.get("category", "unknown")),
        "confidence": float(out.get("confidence", 0.0)),
        "validation_ok": out.get("validation_ok"),
        "validation_missing": list(out.get("validation_missing") or []),
        "assistant_reply": out.get("assistant_reply"),
        "assistant_metadata": out.get("assistant_metadata") or {},
    }
