from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.db.postgres import get_connection, postgres_configured

from testing.simulator.config import DbFilterConfig, SeedConfig


class HydrationError(RuntimeError):
    """Raised when a scenario cannot be hydrated with real entities."""


@dataclass
class ScenarioInstance:
    seed_id: str
    category: str
    intent: str
    description: str
    expected_outcome: str
    expected_procedure_id: str | None
    persona_id: str
    cooperation_level: str
    entity: dict[str, Any]
    seed_snapshot: dict[str, Any] = field(default_factory=dict)
    persona_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "category": self.category,
            "intent": self.intent,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "expected_procedure_id": self.expected_procedure_id,
            "persona_id": self.persona_id,
            "cooperation_level": self.cooperation_level,
            "entity": dict(self.entity),
            "seed_snapshot": dict(self.seed_snapshot),
            "persona_snapshot": dict(self.persona_snapshot),
        }


class ScenarioHydrator:
    def __init__(self) -> None:
        if not postgres_configured():
            raise HydrationError("Postgres is not configured for simulator hydration.")

    def hydrate(self, seed: SeedConfig) -> ScenarioInstance:
        entity = self._query_entity(seed.db_filter)
        return ScenarioInstance(
            seed_id=seed.seed_id,
            category=seed.category,
            intent=seed.intent,
            description=seed.description,
            expected_outcome=seed.expected_outcome,
            expected_procedure_id=seed.expected_procedure_id,
            persona_id="",
            cooperation_level="cooperative",
            entity=entity,
            seed_snapshot=seed.model_dump(mode="json"),
            persona_snapshot={},
        )

    def _query_entity(self, db_filter: DbFilterConfig) -> dict[str, Any]:
        if db_filter.entity_type == "order":
            return self._query_order(db_filter)
        if db_filter.entity_type == "user":
            return self._query_user(db_filter)
        if db_filter.entity_type == "subscription":
            return self._query_subscription(db_filter)
        raise HydrationError(f"Unsupported entity_type: {db_filter.entity_type}")

    def _query_order(self, db_filter: DbFilterConfig) -> dict[str, Any]:
        predicates: list[str] = []
        params: list[Any] = []

        if db_filter.order_status:
            predicates.append("o.status = ANY(%s)")
            params.append(db_filter.order_status)

        age_minutes = _normalize_range(db_filter.order_age_minutes)
        if age_minutes is not None:
            min_dt = datetime.now(timezone.utc) - timedelta(minutes=age_minutes[1])
            max_dt = datetime.now(timezone.utc) - timedelta(minutes=age_minutes[0])
            predicates.append("o.order_date BETWEEN %s AND %s")
            params.extend([min_dt, max_dt])

        age_days = _normalize_range(db_filter.order_age_days)
        if age_days is not None:
            min_dt = datetime.now(timezone.utc) - timedelta(days=age_days[1])
            max_dt = datetime.now(timezone.utc) - timedelta(days=age_days[0])
            predicates.append("o.order_date BETWEEN %s AND %s")
            params.extend([min_dt, max_dt])

        where_sql = " AND ".join(predicates) if predicates else "TRUE"
        query = f"""
            SELECT
                o.order_id,
                o.user_id,
                o.order_date,
                o.status,
                o.total_amount,
                o.shipping_address_line,
                o.shipping_city,
                o.shipping_postal_code,
                o.shipping_country
            FROM orders o
            WHERE {where_sql}
            ORDER BY RANDOM()
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                row = cur.fetchone()

        if not row:
            raise HydrationError(
                f"No order entity matched filter: status={db_filter.order_status!r}, "
                f"order_age_minutes={db_filter.order_age_minutes!r}, "
                f"order_age_days={db_filter.order_age_days!r}"
            )
        return {
            "entity_type": "order",
            "order_id": row[0],
            "user_id": row[1],
            "order_date": row[2].isoformat() if row[2] else None,
            "status": row[3],
            "total_amount": float(row[4]) if row[4] is not None else None,
            "shipping_address_line": row[5],
            "shipping_city": row[6],
            "shipping_postal_code": row[7],
            "shipping_country": row[8],
        }

    def _query_user(self, db_filter: DbFilterConfig) -> dict[str, Any]:
        predicates: list[str] = []
        params: list[Any] = []
        if db_filter.user_status:
            predicates.append("u.status = ANY(%s)")
            params.append(db_filter.user_status)
        where_sql = " AND ".join(predicates) if predicates else "TRUE"
        query = f"""
            SELECT u.user_id, u.email, u.status, u.created_at
            FROM users u
            WHERE {where_sql}
            ORDER BY RANDOM()
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                row = cur.fetchone()
        if not row:
            raise HydrationError(f"No user entity matched filter: {db_filter.user_status!r}")
        return {
            "entity_type": "user",
            "user_id": row[0],
            "email": row[1],
            "status": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
        }

    def _query_subscription(self, db_filter: DbFilterConfig) -> dict[str, Any]:
        predicates: list[str] = []
        params: list[Any] = []
        if db_filter.subscription_status:
            predicates.append("s.subscription_status = ANY(%s)")
            params.append(db_filter.subscription_status)
        if db_filter.subscription_plan:
            predicates.append("s.plan = ANY(%s)")
            params.append(db_filter.subscription_plan)
        where_sql = " AND ".join(predicates) if predicates else "TRUE"
        query = f"""
            SELECT
                s.account_email,
                s.plan,
                s.next_renewal_at,
                s.last_charge_at,
                s.subscription_status
            FROM subscription_accounts s
            WHERE {where_sql}
            ORDER BY RANDOM()
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                row = cur.fetchone()
        if not row:
            raise HydrationError(
                "No subscription entity matched filter: "
                f"status={db_filter.subscription_status!r}, "
                f"plan={db_filter.subscription_plan!r}"
            )
        return {
            "entity_type": "subscription",
            "account_email": row[0],
            "plan": row[1],
            "next_renewal_at": row[2].isoformat() if row[2] else None,
            "last_charge_at": row[3].isoformat() if row[3] else None,
            "subscription_status": row[4],
        }


def _normalize_range(values: list[int]) -> tuple[int, int] | None:
    if not values:
        return None
    if len(values) != 2:
        raise HydrationError(f"Expected [min,max] range with two entries, got: {values!r}")
    lo, hi = int(values[0]), int(values[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi
