from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


StepType = Literal["tool_call", "logic_gate", "interrupt", "llm_response"]


class RequiredDataField(BaseModel):
    name: str
    prompt: str
    validation: str | None = None


class ConditionSpec(BaseModel):
    op: str
    field: str
    value: Any | None = None
    value_from: str | None = None
    failure_reason: str = ""


class OutcomeAssertion(BaseModel):
    id: str
    field: str
    op: str
    value: Any | None = None
    value_from: str | None = None
    failure_reason: str = ""


class ProcedureStep(BaseModel):
    id: str
    type: StepType
    tool: str | None = None
    required_data: list[str] = Field(default_factory=list)
    condition: ConditionSpec | None = None
    on_true: str | None = None
    on_false: str | None = None
    message: str | None = None
    action_type: str | None = None
    action_id: str | None = None
    on_accept_message: str | None = None
    on_reject_message: str | None = None


class ProcedureBlueprint(BaseModel):
    id: str
    category: str
    intent: str
    keywords: list[str] = Field(default_factory=list)
    required_data: list[RequiredDataField] = Field(default_factory=list)
    steps: list[ProcedureStep]
    policy_required: bool = True
    expected_outcomes: list[OutcomeAssertion] = Field(default_factory=list)
    fallback_response: str | None = None


def procedures_dir() -> Path:
    raw = os.getenv("PROCEDURES_DIR", "").strip()
    if raw:
        return Path(raw)
    root = Path(__file__).resolve().parents[1]
    return root / "procedures"


@lru_cache(maxsize=1)
def load_blueprints() -> dict[str, ProcedureBlueprint]:
    out: dict[str, ProcedureBlueprint] = {}
    base = procedures_dir()
    if not base.is_dir():
        return out
    for path in sorted(base.glob("*.yaml")):
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        bp = ProcedureBlueprint.model_validate(data)
        out[bp.id] = bp
    return out


def get_blueprint_by_category_intent(category: str, intent: str) -> ProcedureBlueprint | None:
    cat = (category or "").strip().lower()
    it = (intent or "").strip().lower()
    for bp in load_blueprints().values():
        if bp.category.lower() == cat and bp.intent.lower() == it:
            return bp
    return None


def get_fallback_blueprint(category: str) -> ProcedureBlueprint | None:
    cat = (category or "").strip().lower()
    for bp in load_blueprints().values():
        if bp.category.lower() == cat and bp.intent.lower().endswith("_general"):
            return bp
    for bp in load_blueprints().values():
        if bp.id == "general_research":
            return bp
    return None


def get_blueprint_with_fallback_chain(category: str, intent: str) -> ProcedureBlueprint | None:
    """
    Resolve a blueprint using the deterministic fallback chain:
    1) (category, intent)
    2) (category, *_general fallback)
    3) (unknown, *_general fallback)
    """
    direct = get_blueprint_by_category_intent(category, intent)
    if direct is not None:
        return direct
    return get_fallback_blueprint(category) or get_fallback_blueprint("unknown")


def get_category_intents(category: str) -> list[ProcedureBlueprint]:
    cat = (category or "").strip().lower()
    return [bp for bp in load_blueprints().values() if bp.category.lower() == cat]


def validate_blueprints() -> list[str]:
    errors: list[str] = []
    blueprints = load_blueprints()
    for bp in blueprints.values():
        if len({s.id for s in bp.steps}) != len(bp.steps):
            errors.append(f"{bp.id} has duplicate step ids")
        step_ids = {s.id for s in bp.steps}
        if bp.policy_required:
            policy_path = policy_constraints_dir() / bp.category.lower() / f"{bp.intent.lower()}.yaml"
            if not policy_path.is_file():
                errors.append(f"{bp.id} missing policy constraints file: {policy_path}")
        for step in bp.steps:
            if step.type == "logic_gate":
                if not isinstance(step.condition, ConditionSpec):
                    errors.append(f"{bp.id}:{step.id} condition must be an object")
                    continue
                if not step.condition.op or not step.condition.field:
                    errors.append(f"{bp.id}:{step.id} condition missing op/field")
                    continue
                if step.condition.value is not None and step.condition.value_from:
                    errors.append(f"{bp.id}:{step.id} condition cannot set both value and value_from")
                if not step.on_true or not step.on_false:
                    errors.append(f"{bp.id}:{step.id} missing on_true/on_false")
                    continue
                if step.on_true not in step_ids:
                    errors.append(f"{bp.id}:{step.id} on_true={step.on_true} unknown")
                if step.on_false not in step_ids:
                    errors.append(f"{bp.id}:{step.id} on_false={step.on_false} unknown")
        for outcome in bp.expected_outcomes:
            if bool(outcome.value is not None and outcome.value_from):
                errors.append(f"{bp.id}:expected_outcome:{outcome.id} cannot set both value and value_from")
    return errors


def policy_constraints_dir() -> Path:
    raw = os.getenv("POLICY_CONSTRAINTS_DIR", "").strip()
    if raw:
        return Path(raw)
    root = Path(__file__).resolve().parents[1]
    return root / "policy_constraints"


def as_dict(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump()
