"""Persistent LangGraph runner with SQLite checkpoints."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from backend.agent.issue_graph import (
    _build_agent_state_snapshot,
    _compact_context_data,
    _validation_wait_limit,
    build_issue_classification_graph,
    get_category_for_stored_intent,
    graph_suggests_session_resolved,
)
from backend.rag.required_fields import normalize_category_key

_PERSISTENT_GRAPHS: dict[str, Any] = {}
_PERSISTENT_CONNS: dict[str, sqlite3.Connection] = {}


def _persistent_db_path() -> str:
    raw = os.getenv("AGENT_CHECKPOINT_DB", "agent_checkpoints.db").strip()
    return raw or "agent_checkpoints.db"


def build_persistent_issue_graph(db_path: str | None = None):
    path = db_path or _persistent_db_path()
    if path in _PERSISTENT_GRAPHS:
        return _PERSISTENT_GRAPHS[path]
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_issue_classification_graph(checkpointer=saver)
    _PERSISTENT_CONNS[path] = conn
    _PERSISTENT_GRAPHS[path] = graph
    return graph


def run_persistent_conversation(
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
    db_path: str | None = None,
) -> dict[str, Any]:
    graph = build_persistent_issue_graph(db_path=db_path)
    config = {"configurable": {"thread_id": session_id or "default"}}
    snapshot = graph.get_state(config)
    has_pending_state = bool(getattr(snapshot, "next", ()))

    if has_pending_state:
        out = graph.invoke(
            Command(
                resume={
                    "text": text or "",
                    "messages": messages or [],
                }
            ),
            config=config,
        )
    else:
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
                "enable_persistent_wait_interrupt": True,
            },
            config=config,
        )

    resolved_by_graph = graph_suggests_session_resolved(out)  # type: ignore[arg-type]
    context_data = _compact_context_data(dict(out.get("context_data") or {}))
    policy_constraints = dict(out.get("policy_constraints") or {})
    agent_state = _build_agent_state_snapshot(out)  # type: ignore[arg-type]
    stage_metadata = dict(out.get("stage_metadata") or {})
    assistant_metadata = {
        **(out.get("assistant_metadata") or {}),
        "outcome_status": out.get("outcome_status"),
        "specialist_agent_id": out.get("specialist_agent_id"),
        "agent_state": agent_state,
        "stage_metadata": stage_metadata,
        "validation_wait_count": out.get("validation_wait_count"),
        "validation_wait_limit": out.get("validation_wait_limit"),
        "output_validation": dict(out.get("output_validation") or {}),
        "context_summary": dict(out.get("context_summary") or {}),
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
        "agent_state": agent_state,
        "stage_metadata": stage_metadata,
        "output_validation": dict(out.get("output_validation") or {}),
        "context_summary": dict(out.get("context_summary") or {}),
        "validation_wait_count": int(out.get("validation_wait_count") or 0),
        "validation_wait_limit": int(out.get("validation_wait_limit") or _validation_wait_limit()),
        "eligibility_ok": out.get("eligibility_ok"),
        "outcome_status": out.get("outcome_status"),
        "session_resolved_by_graph": resolved_by_graph,
    }
