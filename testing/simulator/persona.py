from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from backend.llm.providers import chat_completion, extract_json_object
from testing.simulator.config import PersonaConfig
from testing.simulator.hydrator import ScenarioInstance


class PersonaGenerationError(RuntimeError):
    """Raised when the simulator persona LLM fails after built-in retries; runner may skip the scenario."""


@dataclass
class PersonaEngine:
    persona: PersonaConfig
    scenario: ScenarioInstance
    llm_provider: str
    llm_model: str
    llm_timeout_seconds: float
    llm_temperature: float | None = None
    llm_top_p: float | None = None
    llm_repeat_penalty: float | None = None
    event_sink: Any | None = None
    _asks_for_human: bool = False
    _missing_prompt_count: int = 0
    _introduced_secondary_issue: bool = False
    _state: dict[str, Any] = field(default_factory=dict)

    def generate_opening(self) -> str:
        message, stop = self._generate_message(
            mode="opening",
            agent_message="",
            turn_number=0,
            conversation_history=[],
            agent_metadata={},
            force_challenge_missing=False,
            force_secondary_issue=False,
            force_ask_human=False,
            force_distinct_opening=False,
        )
        if self._should_retry_opening(message):
            message, stop = self._generate_message(
                mode="opening",
                agent_message="",
                turn_number=0,
                conversation_history=[],
                agent_metadata={},
                force_challenge_missing=False,
                force_secondary_issue=False,
                force_ask_human=False,
                force_distinct_opening=True,
            )
        if stop:
            raise PersonaGenerationError(
                "Simulator persona generation returned stop=true for opening message."
            )
        return message

    def generate_response(
        self,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict],
        agent_metadata: dict,
    ) -> str | None:
        outcome = str(agent_metadata.get("outcome_status") or "").strip().lower()
        missing = list(agent_metadata.get("validation_missing") or [])

        if outcome in {"resolved", "policy_ineligible", "pending_escalation"}:
            return None

        force_challenge_missing = False
        if missing:
            self._missing_prompt_count += 1
            force_challenge_missing = (
                self.persona.cooperation_level == "resistant" and self._missing_prompt_count <= 1
            )

        force_secondary_issue = False
        if self.scenario.multi_issue and not self._introduced_secondary_issue and turn_number >= 2:
            second = self.scenario.secondary_entity or {}
            second_order_id = str(second.get("order_id") or "").strip()
            if second_order_id:
                force_secondary_issue = True

        force_ask_human = False
        if self.persona.patience == "low" and turn_number >= 3 and not self._asks_for_human:
            force_ask_human = True

        message, stop = self._generate_message(
            mode="response",
            agent_message=agent_message,
            turn_number=turn_number,
            conversation_history=conversation_history,
            agent_metadata=agent_metadata,
            force_challenge_missing=force_challenge_missing,
            force_secondary_issue=force_secondary_issue,
            force_ask_human=force_ask_human,
            force_distinct_opening=False,
        )
        if force_secondary_issue:
            self._introduced_secondary_issue = True
        if force_ask_human:
            self._asks_for_human = True
        if stop:
            return None
        return message

    def _generate_message(
        self,
        *,
        mode: str,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict],
        agent_metadata: dict,
        force_challenge_missing: bool,
        force_secondary_issue: bool,
        force_ask_human: bool,
        force_distinct_opening: bool,
    ) -> tuple[str, bool]:
        directives: list[str] = []
        missing = list(agent_metadata.get("validation_missing") or [])
        order_id = str(self.scenario.entity.get("order_id") or "").strip()
        secondary_order_id = str((self.scenario.secondary_entity or {}).get("order_id") or "").strip()
        style_profile = self._style_profile()

        if mode == "opening":
            directives.append("Write a realistic first customer message to start this support conversation.")
            directives.append(
                f"Use this opening style profile: {style_profile['opening_style']} "
                f"(tone: {style_profile['tone_hint']}, pacing: {style_profile['pacing_hint']}, "
                f"length: {style_profile['length_hint']})."
            )
            directives.append(
                "Avoid overused support openers and template starts. Do not open with phrases like "
                "'Hi there! I was hoping you could help', 'I was wondering if you could', "
                "'Hope you're doing well', or similarly padded preambles."
            )
            directives.append("Vary sentence structure and wording across runs.")
            if force_distinct_opening:
                directives.append(
                    "Your previous attempt sounded formulaic. Rewrite with a distinctly different opener "
                    "and sentence structure while preserving scenario facts."
                )
            directives.append("Return stop=false.")
            if order_id and not self._allows_missing_data_opening():
                directives.append(f"You must explicitly mention order id '{order_id}' in the message.")
        else:
            directives.append("Write the next customer reply in the ongoing support conversation.")
            if force_challenge_missing:
                directives.append(
                    "The user is resistant right now. Ask why the requested info is needed first and do not provide requested details yet."
                )
            elif "order_id" in missing and order_id:
                directives.append(
                    f"The assistant requested missing order_id. You must include order id '{order_id}' in this response."
                )
            elif missing:
                directives.append("Provide the missing details the assistant asked for.")
            if force_secondary_issue:
                directives.append(
                    f"Introduce a second issue now and explicitly mention secondary order id '{secondary_order_id}'."
                )
            if force_ask_human:
                directives.append("The user has low patience. Ask to transfer to a human agent now.")
            directives.append("Set stop=true only if the user would naturally end the conversation now.")

        payload = {
            "mode": mode,
            "persona": self.persona.model_dump(mode="json"),
            "scenario": self.scenario.to_dict(),
            "turn_number": turn_number,
            "agent_message": agent_message,
            "conversation_history": conversation_history,
            "agent_metadata": agent_metadata,
            "state": {
                "asks_for_human": self._asks_for_human,
                "missing_prompt_count": self._missing_prompt_count,
                "introduced_secondary_issue": self._introduced_secondary_issue,
            },
            "style_profile": style_profile,
            "directives": directives,
            "required_output_schema": {
                "message": "string",
                "stop": "boolean",
            },
        }
        user_prompt = json.dumps(payload, ensure_ascii=False)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _PERSONA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_reason = ""
        for attempt in range(2):
            try:
                raw = chat_completion(
                    provider=self.llm_provider,
                    model=self.llm_model,
                    messages=messages,
                    timeout_seconds=self.llm_timeout_seconds,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    repeat_penalty=self.llm_repeat_penalty,
                )
            except Exception as exc:  # noqa: BLE001
                last_reason = f"Simulator persona LLM request failed: {exc}"
                if self.event_sink is not None:
                    self.event_sink.persona_exchange(
                        mode=mode,
                        turn_number=turn_number,
                        attempt=attempt + 1,
                        messages=messages,
                        raw_response=f"<request failed> {exc!r}",
                    )
                continue

            if self.event_sink is not None:
                self.event_sink.persona_exchange(
                    mode=mode,
                    turn_number=turn_number,
                    attempt=attempt + 1,
                    messages=messages,
                    raw_response=raw if isinstance(raw, str) else repr(raw),
                )

            parsed = extract_json_object(raw)
            if not isinstance(parsed, dict):
                last_reason = "Simulator persona LLM returned non-object payload."
                continue

            message = str(parsed.get("message") or "").strip()
            if not message:
                last_reason = "Simulator persona LLM returned empty 'message'."
                continue

            stop = bool(parsed.get("stop"))

            try:
                if mode == "opening":
                    self._validate_opening_message(message)
                else:
                    self._validate_response_message(
                        message=message,
                        missing=missing,
                        force_challenge_missing=force_challenge_missing,
                    )
            except RuntimeError as exc:
                last_reason = str(exc)
                continue

            return message, stop

        raise PersonaGenerationError(
            f"Simulator persona LLM failed after 2 attempts (mode={mode!r}): {last_reason}"
        )

    def _validate_opening_message(self, message: str) -> None:
        order_id = str(self.scenario.entity.get("order_id") or "").strip()
        entity_type = str(self.scenario.entity.get("entity_type") or "").strip().lower()
        if entity_type == "order" and order_id and not self._allows_missing_data_opening():
            if order_id.lower() not in message.lower():
                raise RuntimeError(
                    "Simulator persona opening must include hydrated order_id for order scenarios."
                )

    def _validate_response_message(
        self,
        *,
        message: str,
        missing: list[str],
        force_challenge_missing: bool,
    ) -> None:
        if force_challenge_missing:
            return
        if "order_id" not in missing:
            return
        order_id = str(self.scenario.entity.get("order_id") or "").strip()
        if order_id and order_id.lower() not in message.lower():
            raise RuntimeError(
                "Simulator persona response must include hydrated order_id when assistant requests it."
            )

    def _allows_missing_data_opening(self) -> bool:
        flags = [str(item).strip().lower() for item in self.scenario.adversarial_flags]
        return any("missing_data" in flag for flag in flags)

    def _style_profile(self) -> dict[str, str]:
        cached = self._state.get("style_profile")
        if isinstance(cached, dict) and cached:
            return cached
        profile = random.choice(_OPENING_STYLE_PROFILES)
        self._state["style_profile"] = profile
        return profile

    def _should_retry_opening(self, message: str) -> bool:
        msg = message.strip().lower()
        if not msg:
            return False
        return any(re.match(pattern, msg) for pattern in _BANNED_OPENING_PATTERNS)


_OPENING_STYLE_PROFILES: list[dict[str, str]] = [
    {
        "opening_style": "brief_direct",
        "tone_hint": "neutral and concise",
        "pacing_hint": "straight to the issue",
        "length_hint": "short",
    },
    {
        "opening_style": "casual",
        "tone_hint": "relaxed and friendly",
        "pacing_hint": "natural and conversational",
        "length_hint": "medium",
    },
    {
        "opening_style": "urgent",
        "tone_hint": "time-sensitive and focused",
        "pacing_hint": "fast with minimal filler",
        "length_hint": "short",
    },
    {
        "opening_style": "formal",
        "tone_hint": "polite and structured",
        "pacing_hint": "measured and precise",
        "length_hint": "medium",
    },
    {
        "opening_style": "frustrated",
        "tone_hint": "mildly frustrated but cooperative",
        "pacing_hint": "direct with clear pain point",
        "length_hint": "medium",
    },
    {
        "opening_style": "confused",
        "tone_hint": "uncertain and seeking clarity",
        "pacing_hint": "hesitant but actionable",
        "length_hint": "medium",
    },
    {
        "opening_style": "minimal_context",
        "tone_hint": "sparse details at first",
        "pacing_hint": "very brief",
        "length_hint": "short",
    },
]

_BANNED_OPENING_PATTERNS: tuple[str, ...] = (
    r"^hi there!?[, ]+i was hoping you could help",
    r"^i was hoping you could help",
    r"^i was wondering if you could",
    r"^hope you(?:'| a)re doing well",
)


_PERSONA_SYSTEM_PROMPT = """You are simulating a realistic e-commerce customer in a support chat.
Use the provided persona and scenario context exactly.
Return JSON only with this shape:
{
  "message": "customer utterance",
  "stop": false
}
Rules:
- Keep the message natural and conversational.
- Respect persona vocabulary, patience, and cooperation traits.
- Keep details grounded in provided entity/scenario data only.
- Never invent IDs, statuses, or policy facts not present in the input.
"""
