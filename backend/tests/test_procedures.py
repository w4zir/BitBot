from __future__ import annotations

from backend.agent import procedures
from backend.agent.policy_constraints import clear_policy_constraints_cache


def test_blueprints_load() -> None:
    procedures.load_blueprints.cache_clear()
    data = procedures.load_blueprints()
    assert "order_cancel" in data
    assert "get_refund" in data
    assert "change_shipping_address" in data
    assert "product_info" in data
    assert "product_price" in data
    assert "product_availability" in data
    assert "order_status" in data
    assert "payment_issue" in data
    assert "check_payment_methods" in data
    assert "track_refund" in data
    assert "check_invoice" in data
    assert "subscription_status" in data
    assert "unsubscribe" in data
    assert "contact_human_agent" in data
    assert "delivery_period" in data
    assert "complaint" in data


def test_category_intents_are_loaded() -> None:
    procedures.load_blueprints.cache_clear()
    order_intents = procedures.get_category_intents("order")
    assert any(bp.intent == "cancel_order" for bp in order_intents)
    refund_intents = procedures.get_category_intents("refund")
    assert any(bp.intent == "get_refund" for bp in refund_intents)


def test_track_order_no_longer_maps_to_order_status() -> None:
    procedures.load_blueprints.cache_clear()
    bp = procedures.get_blueprint_with_fallback_chain("order", "track_order")
    assert bp is None


def test_logic_gate_condition_failure_reason_preserved_on_blueprint() -> None:
    procedures.load_blueprints.cache_clear()
    bp = procedures.load_blueprints()["order_cancel"]
    step = next(s for s in bp.steps if s.id == "branch_order_within_cancel_window")
    assert step.condition is not None
    assert "expired" in (step.condition.failure_reason or "").lower()


def test_blueprints_validate() -> None:
    procedures.load_blueprints.cache_clear()
    errors = procedures.validate_blueprints()
    assert errors == []


def test_order_cancel_blueprint_includes_policy_check_step() -> None:
    procedures.load_blueprints.cache_clear()
    bp = procedures.load_blueprints()["order_cancel"]
    step = next(s for s in bp.steps if s.id == "check_policy_constraints")
    assert step.type == "policy_check"
    assert step.on_true
    assert step.on_false


def test_validate_blueprints_rejects_policy_check_missing_branches(tmp_path, monkeypatch) -> None:
    proc_dir = tmp_path / "procedures"
    policy_dir = tmp_path / "policy_constraints"
    (policy_dir / "test").mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    (proc_dir / "bad_policy_check.yaml").write_text(
        "\n".join(
            [
                "id: bad_policy_check",
                "category: test",
                "intent: bad_policy_check",
                "policy_required: true",
                "steps:",
                "  - id: check_policy",
                "    type: policy_check",
                "  - id: done",
                "    type: llm_response",
                "    message: done",
            ]
        ),
        encoding="utf-8",
    )
    (policy_dir / "test" / "bad_policy_check.yaml").write_text(
        "\n".join(
            [
                "schema_version: '1.0'",
                "category: test",
                "intent: bad_policy_check",
                "eligibility_rules:",
                "  - id: known_rule",
                "    field: value_present",
                "    op: eq",
                "    value: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PROCEDURES_DIR", str(proc_dir))
    monkeypatch.setenv("POLICY_CONSTRAINTS_DIR", str(policy_dir))
    procedures.load_blueprints.cache_clear()
    clear_policy_constraints_cache()
    errors = procedures.validate_blueprints()
    assert any("missing on_true/on_false" in err for err in errors)


def test_validate_blueprints_rejects_unknown_policy_rule_reference(tmp_path, monkeypatch) -> None:
    proc_dir = tmp_path / "procedures"
    policy_dir = tmp_path / "policy_constraints"
    (policy_dir / "test").mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    (proc_dir / "bad_policy_rule.yaml").write_text(
        "\n".join(
            [
                "id: bad_policy_rule",
                "category: test",
                "intent: bad_policy_rule",
                "policy_required: true",
                "steps:",
                "  - id: check_policy",
                "    type: policy_check",
                "    policy_rules: [unknown_rule]",
                "    on_true: done",
                "    on_false: done",
                "  - id: done",
                "    type: llm_response",
                "    message: done",
            ]
        ),
        encoding="utf-8",
    )
    (policy_dir / "test" / "bad_policy_rule.yaml").write_text(
        "\n".join(
            [
                "schema_version: '1.0'",
                "category: test",
                "intent: bad_policy_rule",
                "eligibility_rules:",
                "  - id: known_rule",
                "    field: value_present",
                "    op: eq",
                "    value: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PROCEDURES_DIR", str(proc_dir))
    monkeypatch.setenv("POLICY_CONSTRAINTS_DIR", str(policy_dir))
    procedures.load_blueprints.cache_clear()
    clear_policy_constraints_cache()
    errors = procedures.validate_blueprints()
    assert any("policy_rule=unknown_rule unknown" in err for err in errors)
