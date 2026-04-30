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
    policy_constraints = final_turn.policy_constraints if final_turn else {}
    checks["policy_docs_retrieved"] = bool((context_data or {}).get("policy_doc_names")) or bool(
        (policy_constraints or {}).get("policy_doc_names")
    )
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

    variables_ok = isinstance((policy_constraints or {}).get("variables"), dict)
    validation_results_ok = isinstance((policy_constraints or {}).get("validation_results"), dict)
    checks["policy_variables_shape"] = variables_ok and validation_results_ok
    if not checks["policy_variables_shape"]:
        failures.append("Policy constraints did not include JSON variables and validation_results maps.")

    pc_con = dict(policy_constraints) if isinstance(policy_constraints, dict) else {}
    pc_elig = pc_con.get("eligible")
    ctx = dict(context_data) if isinstance(context_data, dict) else {}
    ctx_elig = ctx.get("policy_eligible") if "policy_eligible" in ctx else None
    checks["policy_eligibility_fields_consistent"] = True
    if (
        pc_elig is not None
        and ctx_elig is not None
        and bool(pc_elig) != bool(ctx_elig)
    ):
        checks["policy_eligibility_fields_consistent"] = False
        failures.append(
            f"policy_constraints.eligible ({pc_elig!r}) disagrees with "
            f"context_data.policy_eligible ({ctx_elig!r})."
        )

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
