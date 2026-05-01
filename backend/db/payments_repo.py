from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def get_payment(transaction_id: str) -> dict[str, Any] | None:
    tx = (transaction_id or "").strip().upper()
    if not tx:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT transaction_id, order_id, amount, method, payment_status, charged_at
                FROM payments
                WHERE transaction_id = %s
                """,
                (tx,),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "transaction_id": row[0],
        "order_id": row[1],
        "amount": float(row[2]) if row[2] is not None else None,
        "method": row[3],
        "payment_status": row[4],
        "charged_at": row[5].isoformat() if row[5] else None,
    }


def list_payment_methods() -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT method
                FROM payments
                WHERE method IS NOT NULL AND method <> ''
                ORDER BY method
                """
            )
            rows = cur.fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def get_refund_tracking(transaction_id: str) -> dict[str, Any]:
    payment = get_payment(transaction_id)
    if not payment:
        return {"found": False, "reason": "transaction_not_found"}

    order_id = str(payment.get("order_id") or "").strip()
    refund_row: tuple[Any, ...] | None = None
    if order_id:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT refund_id, decision, decision_reason, requested_at
                    FROM refund_requests
                    WHERE order_id = %s
                    ORDER BY requested_at DESC NULLS LAST, refund_id DESC
                    LIMIT 1
                    """,
                    (order_id,),
                )
                refund_row = cur.fetchone()

    return {
        "found": True,
        "transaction_id": payment.get("transaction_id"),
        "order_id": order_id,
        "payment_status": payment.get("payment_status"),
        "refund_id": refund_row[0] if refund_row else None,
        "refund_decision": refund_row[1] if refund_row else None,
        "refund_decision_reason": refund_row[2] if refund_row else None,
        "refund_requested_at": refund_row[3].isoformat() if refund_row and refund_row[3] else None,
    }
