from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnRecord:
    turn_number: int
    user_message: str
    agent_response: str
    outcome_status: str
    procedure_id: str | None
    validation_missing: list[str] = field(default_factory=list)
    eligibility_ok: bool | None = None
    escalation_bundle: dict[str, Any] | None = None
    policy_constraints: dict[str, Any] | None = None
    context_data: dict[str, Any] | None = None
    confidence: float | None = None
    category: str | None = None
    intent: str | None = None
    issue_locked: bool | None = None
    agent_state: dict[str, Any] | None = None
    stage_metadata: dict[str, Any] | None = None
    output_validation: dict[str, Any] | None = None
    context_summary: dict[str, Any] | None = None
    validation_wait_count: int | None = None
    validation_wait_limit: int | None = None
    latency_ms: float = 0.0


@dataclass
class ConversationTrace:
    scenario: dict[str, Any]
    session_id: str
    turns: list[TurnRecord]
    final_outcome_status: str
    terminated_by: str
    total_latency_ms: float
    total_tokens_used: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "session_id": self.session_id,
            "turns": [asdict(turn) for turn in self.turns],
            "final_outcome_status": self.final_outcome_status,
            "terminated_by": self.terminated_by,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens_used": self.total_tokens_used,
        }
