from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


ComparisonOp = Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "exists", "nin", "contains"]

_CANON_OPS: frozenset[str] = frozenset(
    {"eq", "neq", "gt", "gte", "lt", "lte", "in", "exists", "nin", "contains"}
)

_OP_ALIASES: dict[str, str] = {
    "equals": "eq",
    "equal": "eq",
    "==": "eq",
    "not_equals": "neq",
    "not_equal": "neq",
    "!=": "neq",
    "greater_than": "gt",
    "less_than": "lt",
    "greater_or_equal": "gte",
    "less_or_equal": "lte",
    "not_in": "nin",
    "notin": "nin",
    "one_of": "in",
    "member_of": "in",
    "substring": "contains",
    "includes": "contains",
    "like": "contains",
    "present": "exists",
    "is_set": "exists",
}


class PolicyRule(BaseModel):
    id: str
    description: str = ""
    field: str
    op: ComparisonOp
    value: Any | None = None
    value_from: str | None = None
    failure_reason: str = ""
    applies_to: str = "runtime"

    @field_validator("op", mode="before")
    @classmethod
    def _normalize_op(cls, v: Any) -> str:
        if v is None:
            return "eq"
        key = str(v).strip().lower().replace(" ", "_")
        if key in _CANON_OPS:
            return key
        mapped = _OP_ALIASES.get(key)
        if mapped is not None:
            return mapped
        raise ValueError(f"unsupported policy rule op: {v!r}")


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

    @field_validator("time_limits", mode="before")
    @classmethod
    def _coerce_time_limits(cls, v: Any) -> dict[str, Any]:
        if v is None or not isinstance(v, dict):
            return {}
        return v

    @field_validator("response_guidance", mode="before")
    @classmethod
    def _coerce_response_guidance(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [f"{k}: {val}" for k, val in v.items()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


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
