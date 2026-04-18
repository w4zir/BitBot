from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.db.messages_repo import append_message, get_session, list_messages
from backend.db.postgres import get_connection

router = APIRouter(prefix="/escalations", tags=["escalations"])


class EscalationDecisionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    decision: Literal["accept", "reject"]


@router.post("/decision")
async def escalation_decision(req: EscalationDecisionRequest) -> dict:
    if not get_session(req.session_id):
        raise HTTPException(status_code=404, detail="session_id not found")
    rows = list_messages(req.session_id)
    target = None
    for item in reversed(rows):
        md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if md.get("pending_human_action") and md.get("action_id") == req.action_id:
            target = md
            break
    if target is None:
        raise HTTPException(status_code=404, detail="pending action not found")

    if req.decision == "accept":
        reply = "Thanks for confirming. I have escalated your case to a human support agent."
    else:
        reply = "Understood. I will not escalate this case right now."

    appended = append_message(
        req.session_id,
        "assistant",
        reply,
        metadata={
            "action_id": req.action_id,
            "decision": req.decision,
            "pending_human_action": False,
        },
    )

    if req.decision == "accept":
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET escalated = TRUE, updated_at = NOW() WHERE id = %s",
                    (req.session_id,),
                )

    created_at = appended.get("created_at") if isinstance(appended, dict) else None
    return {
        "ok": True,
        "assistant_reply": reply,
        "decision": req.decision,
        "created_at": created_at,
    }

