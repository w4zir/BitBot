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
                "SELECT id, user_id, company_id, created_at FROM sessions WHERE id = %s",
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
            }


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
                """,
                (mid, session_id, role, content, Json(metadata or {})),
            )
    return {
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
