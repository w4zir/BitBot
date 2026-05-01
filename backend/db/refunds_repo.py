from __future__ import annotations

from typing import Any

from backend.db.orders_repo import get_order_status
from backend.db.postgres import get_connection

_ALLOWED_UPDATE_SOURCES = {"human", "agent", "system"}


def _normalize_update_source(update_source: str) -> str:
    source = (update_source or "").strip().lower()
    return source if source in _ALLOWED_UPDATE_SOURCES else "system"


def get_refund_context(order_id: str) -> dict[str, Any] | None:
    oid = (order_id or "").strip().upper()
    if not oid:
        return None
    order = get_order_status(oid)
    if not order:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT decision, decision_reason
                FROM refund_requests
                WHERE order_id = %s
                ORDER BY requested_at DESC NULLS LAST, refund_id DESC
                LIMIT 1
                """,
                (oid,),
            )
            row = cur.fetchone()
    return {
        "refund_last_decision": row[0] if row else None,
        "refund_last_decision_reason": row[1] if row else None,
        "refund_order_status": order.get("status"),
        "refund_order_total_amount": order.get("total_amount"),
    }


def create_refund_request(
    order_id: str,
    reason: str,
    *,
    update_source: str = "system",
) -> dict[str, Any]:
    oid = (order_id or "").strip().upper()
    note = (reason or "").strip()
    if not oid:
        return {"ok": False, "reason": "missing_order_id"}
    if not note:
        return {"ok": False, "reason": "missing_refund_reason"}
    source = _normalize_update_source(update_source)

    order = get_order_status(oid)
    if not order:
        return {"ok": False, "reason": "order_not_found"}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refund_requests (
                    order_id, reason, requested_at, decision, decision_reason, update_date, update_source
                )
                VALUES (%s, %s, NOW(), 'pending', NULL, NOW(), %s)
                RETURNING refund_id
                """,
                (oid, note, source),
            )
            row = cur.fetchone()

    return {
        "ok": True,
        "refund_id": row[0] if row else None,
        "order_id": oid,
        "decision": "pending",
    }
