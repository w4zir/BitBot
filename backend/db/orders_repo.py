from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def get_order_status(order_number: str) -> dict[str, Any] | None:
    order_id = (order_number or "").strip().upper()
    if not order_id:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT order_number, status, total_amount, estimated_delivery
                FROM orders
                WHERE order_number = %s
                """,
                (order_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "order_number": row[0],
        "status": row[1],
        "total_amount": float(row[2]) if row[2] is not None else None,
        "estimated_delivery": row[3].isoformat() if row[3] else None,
    }
