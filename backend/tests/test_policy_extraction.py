from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_policy_constraints.py"
_SPEC = importlib.util.spec_from_file_location("extract_policy_constraints", _MODULE_PATH)
assert _SPEC and _SPEC.loader
extract_policy_constraints = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract_policy_constraints)


class _Blueprint:
    def __init__(self, category: str, intent: str) -> None:
        self.category = category
        self.intent = intent


def test_extraction_writes_yaml_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        extract_policy_constraints,
        "load_blueprints",
        lambda: {"x": _Blueprint("order", "cancel_order")},
    )
    monkeypatch.setattr(
        extract_policy_constraints,
        "search_policy_docs",
        lambda _q: [{"id": "doc-1", "title": "Order Cancellation Policy", "content": "Cancel within 24 hours."}],
    )
    monkeypatch.setattr(
        extract_policy_constraints,
        "chat_completion",
        lambda **_kwargs: '{"time_limits":{"order_age_hours_max":24},"eligibility_rules":[]}',
    )
    monkeypatch.setattr(extract_policy_constraints, "policy_constraints_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["extract_policy_constraints.py"],
    )

    rc = extract_policy_constraints.main()
    assert rc == 0
    target = tmp_path / "order" / "cancel_order.yaml"
    assert target.is_file()
    payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    assert payload["category"] == "order"
    assert payload["intent"] == "cancel_order"
    assert payload["time_limits"]["order_age_hours_max"] == 24
