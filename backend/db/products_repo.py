from __future__ import annotations

from typing import Any

from backend.db.postgres import get_connection


def lookup_product(product_name: str) -> dict[str, Any] | None:
    name = (product_name or "").strip()
    if not name:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sku, name, price, is_available, metadata
                FROM products
                WHERE name ILIKE %s
                ORDER BY
                    CASE WHEN lower(name) = lower(%s) THEN 0 ELSE 1 END,
                    name ASC
                LIMIT 1
                """,
                (f"%{name}%", name),
            )
            row = cur.fetchone()
            if not row:
                return None
    metadata = row[4] if isinstance(row[4], dict) else {}
    return {
        "sku": row[0],
        "name": row[1],
        "price": float(row[2]) if row[2] is not None else None,
        "is_available": bool(row[3]),
        "metadata": metadata,
    }
