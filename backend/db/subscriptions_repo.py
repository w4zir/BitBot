from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection

_ALLOWED_UPDATE_SOURCES = {"human", "agent", "system"}


def _normalize_update_source(update_source: str) -> str:
    source = (update_source or "").strip().lower()
    return source if source in _ALLOWED_UPDATE_SOURCES else "system"


def get_subscription(account_email: str) -> dict[str, Any] | None:
    email = (account_email or "").strip().lower()
    if not email:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT account_email, plan, next_renewal_at, last_charge_at, subscription_status
                FROM subscription_accounts
                WHERE lower(account_email) = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None
    return {
        "account_email": row[0],
        "plan": row[1],
        "next_renewal_at": row[2].isoformat() if row[2] else None,
        "last_charge_at": row[3].isoformat() if row[3] else None,
        "subscription_status": row[4],
    }


def unsubscribe_subscription(account_email: str, *, update_source: str = "system") -> dict[str, Any]:
    email = (account_email or "").strip().lower()
    if not email:
        return {"ok": False, "reason": "missing_account_email"}
    source = _normalize_update_source(update_source)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT account_email, subscription_status
                FROM subscription_accounts
                WHERE lower(account_email) = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return {"ok": False, "reason": "subscription_not_found"}
            status = str(row[1] or "").strip().lower()
            if status in {"unsubscribed", "cancelled"}:
                return {"ok": False, "reason": "already_unsubscribed", "account_email": row[0]}
            cur.execute(
                """
                UPDATE subscription_accounts
                SET subscription_status = 'unsubscribed',
                    update_date = NOW(),
                    update_source = %s
                WHERE lower(account_email) = %s
                """,
                (source, email),
            )
    return {"ok": True, "account_email": row[0], "subscription_status": "unsubscribed"}
