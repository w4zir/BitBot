from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def get_order_status(order_id: str) -> dict[str, Any] | None:
    """Load order row by primary key `order_id` (e.g. ORD-1001).

    `estimated_delivery` is derived from `shipments.promised_delivery_at` when present.
    """
    oid = (order_id or "").strip().upper()
    if not oid:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    o.order_id,
                    o.status,
                    o.total_amount,
                    (
                        SELECT s.promised_delivery_at
                        FROM shipments s
                        WHERE s.order_id = o.order_id
                        ORDER BY s.promised_delivery_at DESC NULLS LAST
                        LIMIT 1
                    ) AS estimated_delivery
                FROM orders o
                WHERE o.order_id = %s
                """,
                (oid,),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "order_id": row[0],
        "status": row[1],
        "total_amount": float(row[2]) if row[2] is not None else None,
        "estimated_delivery": row[3].isoformat() if row[3] else None,
    }
