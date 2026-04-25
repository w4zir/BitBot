from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace


@dataclass
class PolicyResult:
    passed: bool
    checks: dict[str, bool]
    failures: list[str]


def evaluate_policy(trace: ConversationTrace, scenario: ScenarioInstance) -> PolicyResult:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    final_turn = trace.turns[-1] if trace.turns else None

    context_data = final_turn.context_data if final_turn else {}
    checks["policy_docs_retrieved"] = bool((context_data or {}).get("retrieved_docs"))
    if not checks["policy_docs_retrieved"]:
        failures.append("No retrieved policy documents were captured in context_data.")

    expected_eligibility = _expected_eligibility(scenario)
    if expected_eligibility is None:
        checks["eligibility_correct"] = True
    else:
        checks["eligibility_correct"] = final_turn is not None and final_turn.eligibility_ok == expected_eligibility
        if not checks["eligibility_correct"]:
            failures.append(
                f"Eligibility mismatch. expected={expected_eligibility}, actual={final_turn.eligibility_ok if final_turn else None}."
            )

    has_boundary = "eligibility_boundary" in scenario.adversarial_flags
    checks["boundary_handling"] = not has_boundary or (final_turn is not None and final_turn.eligibility_ok is not None)
    if not checks["boundary_handling"]:
        failures.append("Boundary scenario did not produce a deterministic eligibility signal.")

    ineligible = trace.final_outcome_status == "policy_ineligible"
    final_agent_message = (final_turn.agent_response if final_turn else "").lower()
    checks["ineligible_explanation_present"] = (
        not ineligible
        or "because" in final_agent_message
        or "reason" in final_agent_message
        or "cannot" in final_agent_message
    )
    if not checks["ineligible_explanation_present"]:
        failures.append("Policy-ineligible outcome did not include an explanation in assistant response.")

    return PolicyResult(
        passed=all(checks.values()),
        checks=checks,
        failures=failures,
    )


def _expected_eligibility(scenario: ScenarioInstance) -> bool | None:
    if scenario.intent == "cancel_order":
        status = str(scenario.entity.get("status") or "").strip().lower()
        return status not in {"cancelled", "delivered"}
    if scenario.intent == "get_refund":
        # Current implementation accepts refund submissions if order exists and reason is present.
        return True
    return None
