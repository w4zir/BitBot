from __future__ import annotations

import json
from typing import Any

from backend.db.postgres import get_connection

_ALLOWED_UPDATE_SOURCES = {"human", "agent", "system"}


def _normalize_update_source(update_source: str) -> str:
    source = (update_source or "").strip().lower()
    return source if source in _ALLOWED_UPDATE_SOURCES else "system"


def create_support_ticket(
    *,
    issue_type: str,
    payload: dict[str, Any],
    routing_result: str,
    user_id: int | None = None,
    validation_passed: bool = True,
    update_source: str = "system",
) -> dict[str, Any]:
    itype = (issue_type or "").strip().lower()
    if not itype:
        return {"ok": False, "reason": "missing_issue_type"}
    source = _normalize_update_source(update_source)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO support_tickets (
                    issue_type, user_id, payload_json, validation_passed, routing_result, update_date, update_source
                )
                VALUES (%s, %s, %s::jsonb, %s, %s, NOW(), %s)
                RETURNING ticket_id
                """,
                (itype, user_id, json.dumps(payload or {}), validation_passed, routing_result, source),
            )
            row = cur.fetchone()
    return {"ok": True, "ticket_id": row[0] if row else None}
