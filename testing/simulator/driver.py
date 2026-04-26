from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from typing import Any, Protocol

from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace, TurnRecord


TERMINAL_OUTCOMES = {
    "resolved",
    "policy_ineligible",
    "tool_error",
    "step_error",
    "unresolvable",
    "pending_escalation",
}


class PersonaLike(Protocol):
    def generate_opening(self) -> str: ...

    def generate_response(
        self,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict[str, str]],
        agent_metadata: dict[str, Any],
    ) -> str | None: ...


@dataclass
class ConversationDriver:
    agent_url: str
    max_turns: int
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        self.agent_url = _normalize_loopback_url(self.agent_url)

    def run(self, scenario: ScenarioInstance, persona: PersonaLike) -> ConversationTrace:
        session_id: str | None = None
        history: list[dict[str, str]] = []
        turns: list[TurnRecord] = []
        total_latency_ms = 0.0
        terminated_by = "max_turns"
        outcome_status = "unresolvable"

        user_message = persona.generate_opening()
        for turn_number in range(1, self.max_turns + 1):
            started = time.perf_counter()
            response_payload = self._post_classify(user_message, session_id)
            response_session_id = str(response_payload.get("session_id") or "").strip()
            if response_session_id:
                session_id = response_session_id
            latency_ms = (time.perf_counter() - started) * 1000.0
            total_latency_ms += latency_ms

            assistant_message = str(response_payload.get("assistant_reply") or "")
            assistant_metadata = response_payload.get("assistant_metadata") or {}
            if not isinstance(assistant_metadata, dict):
                assistant_metadata = {}

            outcome_status = str(assistant_metadata.get("outcome_status") or "unresolvable")
            turn = TurnRecord(
                turn_number=turn_number,
                user_message=user_message,
                agent_response=assistant_message,
                outcome_status=outcome_status,
                procedure_id=_string_or_none(
                    assistant_metadata.get("procedure_id") or response_payload.get("procedure_id")
                ),
                validation_missing=list(assistant_metadata.get("validation_missing") or []),
                eligibility_ok=_bool_or_none(assistant_metadata.get("eligibility_ok")),
                escalation_bundle=_dict_or_none(assistant_metadata.get("escalation_bundle")),
                policy_constraints=_dict_or_none(assistant_metadata.get("policy_constraints")),
                context_data=_dict_or_none(assistant_metadata.get("context_data")),
                confidence=_float_or_none(assistant_metadata.get("confidence")),
                category=_string_or_none(assistant_metadata.get("category")),
                intent=_string_or_none(assistant_metadata.get("intent")),
                issue_locked=_bool_or_none(assistant_metadata.get("issue_locked")),
                latency_ms=latency_ms,
            )
            turns.append(turn)
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})

            if outcome_status in TERMINAL_OUTCOMES:
                terminated_by = "escalated" if outcome_status == "pending_escalation" else "resolved"
                break

            next_user_message = persona.generate_response(
                agent_message=assistant_message,
                turn_number=turn_number,
                conversation_history=history,
                agent_metadata=assistant_metadata,
            )
            if next_user_message is None or next_user_message.strip() == "[RESOLVED]":
                terminated_by = "persona_accepted"
                break
            user_message = next_user_message
        else:
            terminated_by = "max_turns"

        return ConversationTrace(
            scenario=scenario.to_dict(),
            session_id=session_id or "",
            turns=turns,
            final_outcome_status=outcome_status,
            terminated_by=terminated_by,
            total_latency_ms=total_latency_ms,
            total_tokens_used=None,
        )

    def _post_classify(self, text: str, session_id: str | None) -> dict[str, Any]:
        payload_obj: dict[str, Any] = {
            "text": text,
            "full_flow": True,
        }
        if session_id:
            payload_obj["session_id"] = session_id
        payload = json.dumps(payload_obj).encode("utf-8")
        request = urllib.request.Request(
            self.agent_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Simulator request failed ({exc.code}) for session {session_id}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Simulator could not connect to agent URL '{self.agent_url}': {exc.reason}"
            ) from exc
        data = json.loads(body)
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected non-object response from /classify")
        return data


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_loopback_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != "localhost":
        return url
    host = "127.0.0.1"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host))
