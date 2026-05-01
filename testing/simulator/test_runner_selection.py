from __future__ import annotations

from testing.simulator.config import (
    DbFilterConfig,
    ScenarioRunConfig,
    SeedConfig,
    SuiteConfig,
)
from testing.simulator.runner import _build_arg_parser, _iter_execution_plan, _select_scenarios


def _seed(seed_id: str, *, category: str = "order", intent: str = "cancel_order", persona: str = "p1") -> SeedConfig:
    return SeedConfig(
        seed_id=seed_id,
        category=category,
        intent=intent,
        persona_id=persona,
        db_filter=DbFilterConfig(entity_type="order"),
    )


def _suite() -> SuiteConfig:
    return SuiteConfig(
        run_id="r1",
        scenarios=[
            ScenarioRunConfig(seed_id="s1"),
            ScenarioRunConfig(seed_id="s2"),
        ],
    )


def test_select_scenarios_applies_filters() -> None:
    suite = _suite()
    seeds_by_id = {
        "s1": _seed("s1", category="order", intent="cancel_order", persona="persona_a"),
        "s2": _seed("s2", category="refund", intent="get_refund", persona="persona_b"),
    }
    selected = _select_scenarios(
        suite=suite,
        seeds_by_id=seeds_by_id,
        seed_override=None,
        category_filters=["refund"],
        difficulty_filters=[],
        persona_filters=["persona_b"],
        intent_filters=["get_refund"],
    )
    assert len(selected) == 1
    assert selected[0][1].seed_id == "s2"


def test_iter_execution_plan_deterministic_iterations() -> None:
    selected = [
        (ScenarioRunConfig(seed_id="s1"), _seed("s1")),
        (ScenarioRunConfig(seed_id="s2"), _seed("s2")),
    ]
    planned = list(
        _iter_execution_plan(
            selected_scenarios=selected,
            randomize=False,
            iterations=2,
            forever=False,
        )
    )
    assert [item[1].seed_id for item in planned] == ["s1", "s2", "s1", "s2"]


def test_suite_defaults_enable_db_persistence() -> None:
    suite = _suite()
    assert suite.defaults.persist_db is True


def test_cli_no_persist_db_override() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(["--suite", "testing/simulator/suites/regression.yaml", "--no-persist-db"])
    assert args.persist_db is False
