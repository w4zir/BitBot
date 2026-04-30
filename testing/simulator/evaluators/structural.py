from __future__ import annotations

import os
import re
from dataclasses import dataclass

from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace, TurnRecord


@dataclass
class StructuralResult:
    passed: bool
    checks: dict[str, bool]
    failures: list[str]


_RESPONSE_DENIES_ORDER_EXISTS_MARKERS = (
    "could not find an order",
    "could not find that order",
    "couldn't find an order",
    "couldn't find that order",
    "cannot find an order",
    "can't find an order",
    "unable to find an order",
    "unable to find that order",
    "no order with",
    "not find this order",
    "not find that order",
)

_ORDER_STATUS_TOKEN_RE = re.compile(
    r"\b(shipped|delivered|processing|pending|cancelled|canceled)\b",
    re.IGNORECASE,
)


def _response_denies_order_exists(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _RESPONSE_DENIES_ORDER_EXISTS_MARKERS)


def _response_mentions_known_order_status(agent_response: str) -> bool:
    return bool(_ORDER_STATUS_TOKEN_RE.search(agent_response))


def _evaluate_order_status_response_context_consistency(
    final_turn: TurnRecord | None,
    failures: list[str],
) -> bool:
    """Fail when user-facing reply contradicts tool-backed order_found for order_status."""
    if final_turn is None:
        return True
    if str(final_turn.procedure_id or "").strip().lower() != "order_status":
        return True
    ctx = final_turn.context_data or {}
    if "order_found" not in ctx:
        return True
    order_found = bool(ctx["order_found"])
    msg = str(final_turn.agent_response or "")
    if order_found and _response_denies_order_exists(msg):
        failures.append(
            "order_status contradiction: assistant said order was not found but "
            "context_data.order_found is true."
        )
        return False
    if (
        not order_found
        and _response_mentions_known_order_status(msg)
        and not _response_denies_order_exists(msg)
    ):
        failures.append(
            "order_status contradiction: assistant asserted a fulfillment status "
            "but context_data.order_found is false."
        )
        return False
    return True


def evaluate_structural(
    trace: ConversationTrace,
    scenario: ScenarioInstance,
    *,
    max_turns: int,
) -> StructuralResult:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    final_turn = trace.turns[-1] if trace.turns else None
    final_outcome = trace.final_outcome_status
    threshold = _category_confidence_threshold()

    checks["outcome_status_match"] = final_outcome == scenario.expected_outcome
    if not checks["outcome_status_match"]:
        failures.append(
            f"Expected outcome '{scenario.expected_outcome}' but got '{final_outcome}'."
        )

    expected_procedure = scenario.expected_procedure_id
    actual_procedure = final_turn.procedure_id if final_turn else None
    checks["procedure_id_match"] = (
        True if not expected_procedure else actual_procedure == expected_procedure
    )
    if not checks["procedure_id_match"]:
        failures.append(
            f"Expected procedure '{expected_procedure}' but got '{actual_procedure}'."
        )

    checks["no_unexpected_escalation"] = (
        scenario.expected_outcome == "pending_escalation"
        or not any(turn.escalation_bundle for turn in trace.turns)
    )
    if not checks["no_unexpected_escalation"]:
        failures.append("Unexpected escalation bundle present for non-escalation scenario.")

    checks["validation_resolved"] = (
        final_outcome != "resolved"
        or (final_turn is not None and len(final_turn.validation_missing) == 0)
    )
    if not checks["validation_resolved"]:
        failures.append("Final turn still had missing required validation fields.")

    confidence_ok = True
    for turn in trace.turns:
        if turn.turn_number >= 2 and turn.confidence is not None and turn.confidence < threshold:
            confidence_ok = False
            break
    checks["classification_confidence"] = confidence_ok
    if not confidence_ok:
        failures.append(
            f"Classification confidence dropped below threshold {threshold} on turn >= 2."
        )

    issue_lock_ok = True
    locked_category: str | None = None
    locked_intent: str | None = None
    for turn in trace.turns:
        if not turn.issue_locked:
            continue
        if locked_category is None:
            locked_category = turn.category
        elif turn.category != locked_category:
            issue_lock_ok = False
            break
        if locked_intent is None:
            locked_intent = turn.intent
        elif turn.intent != locked_intent:
            issue_lock_ok = False
            break
    checks["issue_lock_respected"] = issue_lock_ok
    if not issue_lock_ok:
        failures.append("Category/intent changed while issue lock was active.")

    checks["max_turns_not_breached"] = len(trace.turns) <= max_turns
    if not checks["max_turns_not_breached"]:
        failures.append(f"Conversation exceeded max_turns={max_turns}.")

    wait_limit_ok = True
    for turn in trace.turns:
        if (
            turn.validation_wait_count is not None
            and turn.validation_wait_limit is not None
            and turn.validation_wait_count > turn.validation_wait_limit
        ):
            wait_limit_ok = False
            break
    checks["validation_wait_limit_respected"] = wait_limit_ok
    if not wait_limit_ok:
        failures.append("Validation wait counter exceeded configured limit.")

    order_ctx_ok = _evaluate_order_status_response_context_consistency(final_turn, failures)
    checks["order_status_reply_matches_context"] = order_ctx_ok

    return StructuralResult(
        passed=all(checks.values()),
        checks=checks,
        failures=failures,
    )


def _category_confidence_threshold() -> float:
    raw = os.getenv("CATEGORY_CONFIDENCE_THRESHOLD", "0.5").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.5
    return max(0.0, min(1.0, value))
