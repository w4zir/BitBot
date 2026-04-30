from __future__ import annotations

from testing.simulator.evaluators.structural import evaluate_structural
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace, TurnRecord


def _order_status_scenario() -> ScenarioInstance:
    return ScenarioInstance(
        seed_id="order_status_seed",
        category="order",
        intent="order_status",
        difficulty="easy",
        persona_id="polite_first_timer",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="order_status",
        adversarial_flags=[],
        entity={"order_id": "ORD-1", "status": "shipped"},
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )


def _trace_final(
    *,
    agent_response: str,
    procedure_id: str | None,
    context_data: dict,
) -> ConversationTrace:
    return ConversationTrace(
        scenario={},
        session_id="s1",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="status please",
                agent_response=agent_response,
                outcome_status="resolved",
                procedure_id=procedure_id,
                context_data=context_data,
                confidence=0.99,
                category="order",
                intent="order_status",
            )
        ],
        final_outcome_status="resolved",
        terminated_by="end",
        total_latency_ms=1.0,
    )


def test_structural_order_status_denies_existing_order_fails() -> None:
    scenario = _order_status_scenario()
    trace = _trace_final(
        agent_response="We could not find an order with ID ORD-1 in our system.",
        procedure_id="order_status",
        context_data={
            "order_found": True,
            "order_status": "shipped",
            "order_id_extracted": "ORD-1",
        },
    )
    result = evaluate_structural(trace, scenario, max_turns=8)
    assert result.checks.get("order_status_reply_matches_context") is False
    assert result.passed is False


def test_structural_order_status_consistent_reply_passes() -> None:
    scenario = _order_status_scenario()
    trace = _trace_final(
        agent_response="Order ORD-1 is currently shipped.",
        procedure_id="order_status",
        context_data={
            "order_found": True,
            "order_status": "shipped",
            "order_id_extracted": "ORD-1",
        },
    )
    result = evaluate_structural(trace, scenario, max_turns=8)
    assert result.checks.get("order_status_reply_matches_context") is True
    assert result.passed is True


def test_structural_other_procedure_skips_order_status_guard() -> None:
    scenario = ScenarioInstance(
        seed_id="cancel",
        category="order",
        intent="cancel_order",
        difficulty="easy",
        persona_id="p",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="order_cancel",
        adversarial_flags=[],
        entity={"status": "processing"},
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )
    trace = ConversationTrace(
        scenario={},
        session_id="s1",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="cancel",
                agent_response="We could not find an order.",
                outcome_status="resolved",
                procedure_id="order_cancel",
                context_data={"order_found": True},
                confidence=0.99,
                category="order",
                intent="cancel_order",
            )
        ],
        final_outcome_status="resolved",
        terminated_by="end",
        total_latency_ms=1.0,
    )
    result = evaluate_structural(trace, scenario, max_turns=8)
    assert result.checks.get("order_status_reply_matches_context") is True
