from __future__ import annotations

import uuid
from typing import Any, Optional

from psycopg2.extras import Json

from backend.db.postgres import get_connection


def create_session(*, user_id: Optional[str] = None, company_id: Optional[str] = None) -> str:
    sid = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id, user_id, company_id)
                VALUES (%s, %s, %s)
                """,
                (sid, user_id, company_id),
            )
    return sid


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, company_id, created_at,
                       intent, user_request, problem_to_solve, issue_category, issue_confidence,
                       resolved_at, escalated
                FROM sessions WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": str(row[0]),
                "user_id": row[1],
                "company_id": row[2],
                "created_at": row[3],
                "intent": row[4],
                "user_request": row[5],
                "problem_to_solve": row[6],
                "issue_category": row[7],
                "issue_confidence": row[8],
                "resolved_at": row[9],
                "escalated": row[10],
            }


def get_session_issue_state(session_id: str) -> Optional[dict[str, Any]]:
    """Return session row fields needed for active issue locking (or None if missing)."""
    return get_session(session_id)


def update_session_active_issue(
    session_id: str,
    *,
    intent: str,
    user_request: str,
    problem_to_solve: str,
    issue_category: str,
    issue_confidence: float,
) -> None:
    """Set the locked procedure intent and canonical user request; clears resolution timestamp."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET intent = %s,
                    user_request = %s,
                    problem_to_solve = %s,
                    issue_category = %s,
                    issue_confidence = %s,
                    resolved_at = NULL,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (intent, user_request, problem_to_solve, issue_category, issue_confidence, session_id),
            )


def mark_session_resolved(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET resolved_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (session_id,),
            )


def list_messages(session_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, metadata, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for role, content, metadata, created_at in rows:
        md = metadata if isinstance(metadata, dict) else {}
        out.append(
            {
                "role": role,
                "content": content,
                "metadata": md,
                "created_at": created_at.isoformat() if created_at else None,
            }
        )
    return out


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    mid = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE id = %s",
                (session_id,),
            )
            cur.execute(
                """
                INSERT INTO messages (id, session_id, role, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING created_at
                """,
                (mid, session_id, role, content, Json(metadata or {})),
            )
            row = cur.fetchone()
            created_at = row[0] if row else None
    ca_iso = created_at.isoformat() if created_at is not None else None
    return {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": ca_iso,
    }
