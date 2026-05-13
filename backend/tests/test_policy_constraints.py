from __future__ import annotations

from backend.agent.policy_constraints import (
    clear_policy_constraints_cache,
    load_policy_constraints_for_intent,
)


def test_load_policy_constraints_by_intent() -> None:
    clear_policy_constraints_cache()
    model = load_policy_constraints_for_intent("order", "cancel_order")
    assert model.category == "order"
    assert model.intent == "cancel_order"
    assert model.time_limits.get("order_age_hours_max") == 24
    assert model.eligibility_rules
    first = model.eligibility_rules[0]
    assert getattr(first, "failure_reason", "").strip(), "eligibility rules should include customer-facing failure_reason"


def test_missing_policy_constraints_fail_closed() -> None:
    clear_policy_constraints_cache()
    model = load_policy_constraints_for_intent("unknown", "missing_intent")
    assert model.auto_resolvable is False
    assert "not found" in model.default_ineligible_reason.lower()
