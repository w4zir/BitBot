from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from backend.agent.issue_graph import (
    run_conversation_graph,
    user_confirms_resolution,
)
from backend.db.messages_repo import (
    append_message,
    create_session,
    get_session,
    get_session_issue_state,
    list_messages,
    mark_session_resolved,
    update_session_active_issue,
)
from backend.db.postgres import postgres_configured
from backend.rag.query_classifier import get_query_classifier

router = APIRouter(tags=["classification"])


class ChatMessage(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str = Field(default="", description="User utterance to classify.")
    session_id: Optional[str] = Field(default=None, description="Existing session UUID.")
    full_flow: bool = Field(
        default=False,
        description="If true, run session + LangGraph branches (no_issue/validation) and persist history.",
    )


class SessionIssue(BaseModel):
    """Authoritative active issue for this chat session (Postgres-backed)."""

    intent: str = ""
    user_request: str = ""
    problem_to_solve: str = ""
    is_resolved: bool = False


class ClassifyResponse(BaseModel):
    session_id: Optional[str] = None
    text: str
    category: str
    intent: str = ""
    confidence: float
    procedure_id: str = ""
    validation_ok: Optional[bool] = None
    validation_missing: list[str] = Field(default_factory=list)
    assistant_reply: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    assistant_metadata: dict[str, Any] = Field(default_factory=dict)
    session_issue: SessionIssue = Field(default_factory=SessionIssue)


def _strip_messages(rows: list[dict[str, Any]]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for r in rows:
        ca = r.get("created_at")
        created_at = str(ca) if ca is not None else None
        out.append(
            ChatMessage(
                role=str(r.get("role") or "user"),
                content=str(r.get("content") or ""),
                metadata=r.get("metadata") if isinstance(r.get("metadata"), dict) else {},
                created_at=created_at,
            )
        )
    return out


@router.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Classify the latest user message (Bento/ModernBERT). With `full_flow=true`, run LangGraph
    branches and persist hybrid session history in Postgres.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if not req.full_flow:
        qc = get_query_classifier()
        result = qc.classify(text)
        return ClassifyResponse(
            session_id=None,
            text=text,
            category=result.category,
            intent="",
            confidence=result.confidence,
            procedure_id="",
            validation_ok=None,
            validation_missing=[],
            assistant_reply=None,
            messages=[],
            assistant_metadata={},
            session_issue=SessionIssue(),
        )

    if not postgres_configured():
        raise HTTPException(
            status_code=503,
            detail="Postgres not configured; set POSTGRES_HOST (and related env) for full_flow.",
        )

    session_id = (req.session_id or "").strip() or create_session()
    if req.session_id and not get_session(session_id):
        raise HTTPException(status_code=404, detail="session_id not found")

    pre = get_session_issue_state(session_id) or {}
    had_resolved = pre.get("resolved_at") is not None
    issue_locked = bool(pre.get("intent") and not had_resolved)

    append_message(session_id, "user", text, metadata={"source": "user"})

    history_rows = list_messages(session_id)
    messages_for_graph: list[dict[str, Any]] = []
    for m in history_rows:
        messages_for_graph.append(
            {
                "role": m["role"],
                "content": m["content"],
                "metadata": m.get("metadata") or {},
            }
        )

    # Explicit user confirmation ends the active issue without re-running the graph.
    if issue_locked and user_confirms_resolution(text):
        mark_session_resolved(session_id)
        assistant = (
            "Great — I've marked this issue as resolved. "
            "Let me know if you need help with anything else."
        )
        assistant_meta = {
            "category": pre.get("issue_category") or "",
            "intent": str(pre.get("intent") or ""),
            "procedure_id": "",
            "confidence": float(pre.get("issue_confidence") or 0.0),
            "problem_to_solve": str(pre.get("problem_to_solve") or ""),
            "resolution": "user_confirmed",
        }
        append_message(
            session_id,
            "assistant",
            assistant,
            metadata=assistant_meta,
        )
        final_rows = list_messages(session_id)
        ur = str(pre.get("user_request") or "")
        problem_to_solve = str(pre.get("problem_to_solve") or "")
        return ClassifyResponse(
            session_id=session_id,
            text=text,
            category=str(pre.get("issue_category") or ""),
            intent=str(pre.get("intent") or ""),
            confidence=float(pre.get("issue_confidence") or 0.0),
            procedure_id="",
            validation_ok=None,
            validation_missing=[],
            assistant_reply=assistant,
            messages=_strip_messages(final_rows),
            assistant_metadata=assistant_meta,
            session_issue=SessionIssue(
                intent=str(pre.get("intent") or ""),
                user_request=ur,
                problem_to_solve=problem_to_solve,
                is_resolved=True,
            ),
        )

    graph_out = run_conversation_graph(
        text=text,
        session_id=session_id,
        messages=messages_for_graph,
        issue_locked=issue_locked,
        locked_category=str(pre.get("issue_category") or "") if issue_locked else None,
        locked_intent=str(pre.get("intent") or "") if issue_locked else None,
        locked_problem_to_solve=str(pre.get("problem_to_solve") or "") if issue_locked else None,
        locked_confidence=float(pre["issue_confidence"])
        if issue_locked and pre.get("issue_confidence") is not None
        else None,
    )

    cat = graph_out.get("category") or "unknown"
    conf = float(graph_out.get("confidence") or 0.0)
    intent = str(graph_out.get("intent") or "")
    problem_to_solve = str(graph_out.get("problem_to_solve") or "")
    procedure_id = str(graph_out.get("procedure_id") or "")
    val_ok = graph_out.get("validation_ok")
    val_missing = list(graph_out.get("validation_missing") or [])
    assistant = graph_out.get("assistant_reply")
    meta = graph_out.get("assistant_metadata") or {}
    resolved_by_graph = bool(graph_out.get("session_resolved_by_graph"))

    assistant_meta: dict[str, Any] = {
        "category": cat,
        "intent": intent,
        "problem_to_solve": problem_to_solve,
        "procedure_id": procedure_id,
        "confidence": conf,
        **(meta if isinstance(meta, dict) else {}),
    }
    if val_ok is not None:
        assistant_meta["validation_ok"] = val_ok
        assistant_meta["validation_missing"] = val_missing

    if assistant:
        append_message(
            session_id,
            "assistant",
            assistant,
            metadata=assistant_meta,
        )

    # Persist / update session-level issue tracking (new or post-resolution issue only).
    if not issue_locked:
        update_session_active_issue(
            session_id,
            intent=intent,
            user_request=text,
            problem_to_solve=problem_to_solve,
            issue_category=str(cat),
            issue_confidence=conf,
        )
    if resolved_by_graph:
        mark_session_resolved(session_id)

    final_rows = list_messages(session_id)
    post = get_session_issue_state(session_id) or {}

    return ClassifyResponse(
        session_id=session_id,
        text=text,
        category=str(cat),
        intent=intent,
        confidence=conf,
        procedure_id=procedure_id,
        validation_ok=val_ok if isinstance(val_ok, bool) else None,
        validation_missing=val_missing,
        assistant_reply=assistant,
        messages=_strip_messages(final_rows),
        assistant_metadata=assistant_meta if isinstance(assistant_meta, dict) else {},
        session_issue=SessionIssue(
            intent=str(post.get("intent") or ""),
            user_request=str(post.get("user_request") or ""),
            problem_to_solve=str(post.get("problem_to_solve") or ""),
            is_resolved=post.get("resolved_at") is not None,
        ),
    )
