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
                    o.order_date,
                    o.shipping_address_line,
                    o.shipping_city,
                    o.shipping_postal_code,
                    o.shipping_country,
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
        "order_date": row[3].isoformat() if row[3] else None,
        "shipping_address": {
            "line": row[4],
            "city": row[5],
            "postal_code": row[6],
            "country": row[7],
        },
        "estimated_delivery": row[8].isoformat() if row[8] else None,
    }


def cancel_order(order_id: str) -> dict[str, Any]:
    oid = (order_id or "").strip().upper()
    if not oid:
        return {"ok": False, "reason": "missing_order_id"}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM orders WHERE order_id = %s", (oid,))
            row = cur.fetchone()
            if not row:
                return {"ok": False, "reason": "order_not_found"}
            status = str(row[0] or "").strip().lower()
            if status in {"cancelled", "delivered"}:
                return {"ok": False, "reason": f"order_{status}"}
            cur.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = %s",
                (oid,),
            )
    return {"ok": True, "order_id": oid, "status": "cancelled"}


def update_shipping_address(order_id: str, new_address: str) -> dict[str, Any]:
    oid = (order_id or "").strip().upper()
    raw = (new_address or "").strip()
    if not oid:
        return {"ok": False, "reason": "missing_order_id"}
    if not raw:
        return {"ok": False, "reason": "missing_new_address"}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM orders WHERE order_id = %s", (oid,))
            row = cur.fetchone()
            if not row:
                return {"ok": False, "reason": "order_not_found"}
            status = str(row[0] or "").strip().lower()
            if status in {"delivered", "cancelled"}:
                return {"ok": False, "reason": f"order_{status}"}

            cur.execute(
                """
                UPDATE orders
                SET shipping_address_line = %s
                WHERE order_id = %s
                """,
                (raw, oid),
            )
    return {
        "ok": True,
        "order_id": oid,
        "shipping_address": {"line": raw},
    }
