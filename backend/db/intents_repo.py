from __future__ import annotations

from backend.db.postgres import get_connection, postgres_configured


def get_intents_for_category(category: str) -> list[str]:
    """Return active intents for the given category from Postgres taxonomy tables."""
    name = (category or "").strip()
    if not name or not postgres_configured():
        return []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT intent_name
                    FROM category_intents
                    WHERE category_name = %s
                      AND is_active = TRUE
                    ORDER BY intent_name
                    """,
                    (name,),
                )
                rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        return []

    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
