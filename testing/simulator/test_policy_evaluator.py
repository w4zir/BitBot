from __future__ import annotations

from testing.simulator.evaluators.policy import evaluate_policy
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace, TurnRecord


def _refund_scenario() -> ScenarioInstance:
    return ScenarioInstance(
        seed_id="refund_delivered_easy",
        category="refund",
        intent="get_refund",
        difficulty="medium",
        persona_id="polite_first_timer",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="get_refund",
        adversarial_flags=[],
        entity={"order_id": "ORD-9352", "status": "delivered"},
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )


def _policy_turn(
    *,
    agent_response: str,
    eligibility_ok: bool,
    pc_eligible: bool,
    ctx_eligible: bool,
) -> TurnRecord:
    return TurnRecord(
        turn_number=1,
        user_message="refund please",
        agent_response=agent_response,
        outcome_status="resolved",
        procedure_id="get_refund",
        context_data={
            "policy_doc_names": ["Global Returns & Refund Policy"],
            "policy_eligible": ctx_eligible,
        },
        policy_constraints={
            "eligible": pc_eligible,
            "reason": "",
            "variables": {},
            "validation_results": {},
        },
        eligibility_ok=eligibility_ok,
        category="refund",
        intent="get_refund",
    )


def test_policy_eligibility_cross_field_mismatch_fails() -> None:
    scenario = _refund_scenario()
    trace = ConversationTrace(
        scenario={},
        session_id="s1",
        turns=[
            _policy_turn(
                agent_response="Refund submitted.",
                eligibility_ok=True,
                pc_eligible=True,
                ctx_eligible=False,
            )
        ],
        final_outcome_status="resolved",
        terminated_by="end",
        total_latency_ms=1.0,
    )
    result = evaluate_policy(trace, scenario)
    assert result.checks["policy_eligibility_fields_consistent"] is False
    assert result.passed is False


def test_policy_eligibility_cross_field_match_passes() -> None:
    scenario = _refund_scenario()
    trace = ConversationTrace(
        scenario={},
        session_id="s1",
        turns=[
            _policy_turn(
                agent_response="Refund submitted.",
                eligibility_ok=True,
                pc_eligible=False,
                ctx_eligible=False,
            )
        ],
        final_outcome_status="resolved",
        terminated_by="end",
        total_latency_ms=1.0,
    )
    result = evaluate_policy(trace, scenario)
    assert result.checks["policy_eligibility_fields_consistent"] is True


def test_policy_skips_consistency_when_context_eligible_missing() -> None:
    scenario = _refund_scenario()
    trace = ConversationTrace(
        scenario={},
        session_id="s1",
        turns=[
            TurnRecord(
                turn_number=1,
                user_message="refund please",
                agent_response="Refund submitted.",
                outcome_status="resolved",
                procedure_id="get_refund",
                context_data={"policy_doc_names": ["Policy A"]},
                policy_constraints={
                    "eligible": True,
                    "variables": {},
                    "validation_results": {},
                },
                eligibility_ok=True,
                category="refund",
                intent="get_refund",
            )
        ],
        final_outcome_status="resolved",
        terminated_by="end",
        total_latency_ms=1.0,
    )
    result = evaluate_policy(trace, scenario)
    assert result.checks["policy_eligibility_fields_consistent"] is True
