from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def get_invoice(invoice_id: str) -> dict[str, Any] | None:
    iid = (invoice_id or "").strip().upper()
    if not iid:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT invoice_id, user_id, order_id, account_email, amount, issued_at, status
                FROM invoices
                WHERE invoice_id = %s
                """,
                (iid,),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "invoice_id": row[0],
        "user_id": row[1],
        "order_id": row[2],
        "account_email": row[3],
        "amount": float(row[4]) if row[4] is not None else None,
        "issued_at": row[5].isoformat() if row[5] else None,
        "status": row[6],
    }
