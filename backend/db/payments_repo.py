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
