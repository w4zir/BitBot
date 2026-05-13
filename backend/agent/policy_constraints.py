from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


ComparisonOp = Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "exists"]


class PolicyRule(BaseModel):
    id: str
    description: str = ""
    field: str
    op: ComparisonOp
    value: Any | None = None
    value_from: str | None = None
    failure_reason: str = ""
    applies_to: str = "runtime"


class PolicyCheckResult(BaseModel):
    check_id: str
    passed: bool
    reason: str = ""
    actual_value: Any | None = None
    expected_value: Any | None = None
    source: str = "policy"
    condition: dict[str, Any] = Field(default_factory=dict)


class PolicyConstraints(BaseModel):
    schema_version: str = "1.0"
    category: str
    intent: str
    policy_doc_names: list[str] = Field(default_factory=list)
    source_query: str = ""
    auto_resolvable: bool = True
    requires_evidence: bool = False
    default_ineligible_reason: str = "The request does not satisfy policy constraints."
    time_limits: dict[str, float] = Field(default_factory=dict)
    eligibility_rules: list[PolicyRule] = Field(default_factory=list)
    required_conditions: list[PolicyRule] = Field(default_factory=list)
    escalation_conditions: list[PolicyRule] = Field(default_factory=list)
    response_guidance: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def policy_constraints_dir() -> Path:
    raw = os.getenv("POLICY_CONSTRAINTS_DIR", "").strip()
    if raw:
        return Path(raw)
    root = Path(__file__).resolve().parents[1]
    return root / "policy_constraints"


def _intent_key(category: str, intent: str) -> str:
    return f"{(category or '').strip().lower()}::{(intent or '').strip().lower()}"


@lru_cache(maxsize=1)
def _load_all_constraints() -> dict[str, PolicyConstraints]:
    base = policy_constraints_dir()
    out: dict[str, PolicyConstraints] = {}
    if not base.is_dir():
        return out
    for path in sorted(base.glob("*/*.yaml")):
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model = PolicyConstraints.model_validate(raw)
        out[_intent_key(model.category, model.intent)] = model
    return out


def clear_policy_constraints_cache() -> None:
    _load_all_constraints.cache_clear()


def build_fail_closed_constraints(category: str, intent: str, reason: str) -> PolicyConstraints:
    return PolicyConstraints(
        category=(category or "").strip().lower() or "unknown",
        intent=(intent or "").strip().lower() or "unknown",
        auto_resolvable=False,
        default_ineligible_reason=reason.strip() or "Missing or invalid policy constraints artifact.",
        metadata={"load_error": reason},
    )


def load_policy_constraints_for_intent(category: str, intent: str) -> PolicyConstraints:
    key = _intent_key(category, intent)
    model = _load_all_constraints().get(key)
    if model is not None:
        return model
    reason = (
        "Policy constraints artifact not found for "
        f"category='{(category or '').strip().lower()}' intent='{(intent or '').strip().lower()}'."
    )
    return build_fail_closed_constraints(category, intent, reason)
