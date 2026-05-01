from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def get_delivery_period(order_or_tracking: str) -> dict[str, Any] | None:
    ref = (order_or_tracking or "").strip().upper()
    if not ref:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.tracking_id,
                    s.order_id,
                    s.shipping_tier,
                    s.promised_delivery_at,
                    s.actual_delivery_at,
                    s.delay_reason,
                    o.status
                FROM shipments s
                JOIN orders o ON o.order_id = s.order_id
                WHERE s.tracking_id = %s OR s.order_id = %s
                ORDER BY s.promised_delivery_at DESC NULLS LAST
                LIMIT 1
                """,
                (ref, ref),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "tracking_id": row[0],
        "order_id": row[1],
        "shipping_tier": row[2],
        "promised_delivery_at": row[3].isoformat() if row[3] else None,
        "actual_delivery_at": row[4].isoformat() if row[4] else None,
        "delay_reason": row[5],
        "order_status": row[6],
    }
