from __future__ import annotations

from backend.db.postgres import get_connection, postgres_configured


def get_intents_for_category(category: str) -> list[str]:
    """Return active intents for the given category from Postgres taxonomy tables."""
    return [item["intent_name"] for item in get_intent_definitions_for_category(category)]


def get_intent_definitions_for_category(category: str) -> list[dict[str, str]]:
    """Return active intents and one-line descriptions for a category."""
    name = (category or "").strip()
    if not name or not postgres_configured():
        return []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT intent_name, description
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

    out: list[dict[str, str]] = []
    for row in rows:
        if not row:
            continue
        intent_name = str(row[0]).strip()
        if not intent_name:
            continue
        description = ""
        if len(row) > 1 and row[1] is not None:
            description = str(row[1]).strip()
        out.append({"intent_name": intent_name, "description": description})
    return out
