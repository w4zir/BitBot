from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from testing.simulator.config import PersonaConfig
from testing.simulator.hydrator import ScenarioInstance


@dataclass
class PersonaEngine:
    persona: PersonaConfig
    scenario: ScenarioInstance
    _asks_for_human: bool = False
    _missing_prompt_count: int = 0
    _introduced_secondary_issue: bool = False
    _state: dict[str, Any] = field(default_factory=dict)

    def generate_opening(self) -> str:
        entity = self.scenario.entity
        order_id = str(entity.get("order_id") or "").strip()
        status = str(entity.get("status") or "").strip()
        if self.scenario.intent == "cancel_order":
            if self.persona.vocabulary == "informal":
                return f"hey, can you cancel my order {order_id}? it is still {status}."
            return f"Hi, I need help canceling order {order_id}. Current status is {status}."
        if self.scenario.intent == "get_refund":
            return (
                f"I want a refund for order {order_id}. "
                "The item was not acceptable and I need help with next steps."
            )
        if self.scenario.intent == "order_status":
            return f"Can you check the status for my order {order_id}?"
        return f"I need help with my order {order_id}."

    def generate_response(
        self,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict],
        agent_metadata: dict,
    ) -> str | None:
        _ = conversation_history
        outcome = str(agent_metadata.get("outcome_status") or "").strip().lower()
        missing = list(agent_metadata.get("validation_missing") or [])
        order_id = str(self.scenario.entity.get("order_id") or "").strip()

        if outcome in {"resolved", "policy_ineligible", "pending_escalation"}:
            return None

        if missing:
            self._missing_prompt_count += 1
            if self.persona.cooperation_level == "resistant" and self._missing_prompt_count <= 1:
                return "why do you need that first?"
            if "order_id" in missing:
                return f"My order id is {order_id}."
            return "Sure, here are the details you requested."

        if self.scenario.multi_issue and not self._introduced_secondary_issue and turn_number >= 2:
            self._introduced_secondary_issue = True
            second = self.scenario.secondary_entity or {}
            second_order_id = str(second.get("order_id") or "").strip()
            if second_order_id:
                return f"Also, I have another issue with order {second_order_id}."

        if self.persona.patience == "low" and turn_number >= 3 and not self._asks_for_human:
            self._asks_for_human = True
            return "This is taking too long. Can you transfer me to a human agent?"

        if "need help" in agent_message.lower():
            return "Yes, please continue."
        return "Okay, thanks."
