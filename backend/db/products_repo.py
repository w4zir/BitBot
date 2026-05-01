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
                SELECT sku, name, company, description, price, is_available, metadata
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
    metadata = row[6] if isinstance(row[6], dict) else {}
    return {
        "sku": row[0],
        "name": row[1],
        "company": row[2],
        "description": row[3],
        "price": float(row[4]) if row[4] is not None else None,
        "is_available": bool(row[5]),
        "metadata": metadata,
    }


def get_product_info(product_name: str) -> dict[str, Any] | None:
    product = lookup_product(product_name)
    if not product:
        return None
    return {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "company": product.get("company"),
        "description": product.get("description"),
    }


def get_product_price(product_name: str) -> dict[str, Any] | None:
    product = lookup_product(product_name)
    if not product:
        return None
    return {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "price": product.get("price"),
    }


def get_product_availability(product_name: str) -> dict[str, Any] | None:
    product = lookup_product(product_name)
    if not product:
        return None
    return {
        "sku": product.get("sku"),
        "name": product.get("name"),
        "is_available": bool(product.get("is_available")),
    }
