from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

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
    event_sink: Any | None = None

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
            request_session_id = session_id
            request_started_at = datetime.now(timezone.utc)
            response_payload = self._post_classify(user_message, session_id)
            response_received_at = datetime.now(timezone.utc)
            response_session_id = str(response_payload.get("session_id") or "").strip()
            if response_session_id:
                session_id = response_session_id
            latency_ms = (time.perf_counter() - started) * 1000.0
            total_latency_ms += latency_ms

            assistant_message = str(response_payload.get("assistant_reply") or "")
            assistant_metadata = response_payload.get("assistant_metadata") or {}
            if not isinstance(assistant_metadata, dict):
                assistant_metadata = {}
            token_usage = _extract_token_usage(response_payload, assistant_metadata)

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
                agent_state=_dict_or_none(assistant_metadata.get("agent_state")),
                stage_metadata=_dict_or_none(assistant_metadata.get("stage_metadata")),
                output_validation=_dict_or_none(assistant_metadata.get("output_validation")),
                context_summary=_dict_or_none(assistant_metadata.get("context_summary")),
                validation_wait_count=_int_or_none(assistant_metadata.get("validation_wait_count")),
                validation_wait_limit=_int_or_none(assistant_metadata.get("validation_wait_limit")),
                request_started_at=request_started_at.isoformat(),
                response_received_at=response_received_at.isoformat(),
                request_payload={
                    "text": user_message,
                    "full_flow": True,
                    "session_id": request_session_id,
                },
                response_payload=response_payload,
                input_tokens=token_usage.get("input_tokens"),
                output_tokens=token_usage.get("output_tokens"),
                cache_tokens=token_usage.get("cache_tokens"),
                total_tokens=token_usage.get("total_tokens"),
                latency_ms=latency_ms,
            )
            turns.append(turn)
            if self.event_sink is not None:
                self.event_sink.agent_exchange(
                    turn_number=turn_number,
                    request_payload=turn.request_payload,
                    response_payload=turn.response_payload,
                )
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
            total_tokens_used=_sum_turn_tokens(turns),
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


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_token_usage(
    response_payload: dict[str, Any], assistant_metadata: dict[str, Any]
) -> dict[str, int | None]:
    usage = _dict_or_none(response_payload.get("usage")) or _dict_or_none(
        assistant_metadata.get("usage")
    )
    token_usage = _dict_or_none(response_payload.get("token_usage")) or _dict_or_none(
        assistant_metadata.get("token_usage")
    )
    llm_usage = _dict_or_none(assistant_metadata.get("llm_usage"))
    source = usage or token_usage or llm_usage or {}
    input_tokens = _int_or_none(source.get("input_tokens") or source.get("prompt_tokens"))
    output_tokens = _int_or_none(source.get("output_tokens") or source.get("completion_tokens"))
    cache_tokens = _int_or_none(source.get("cache_tokens"))
    total_tokens = _int_or_none(source.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_tokens": cache_tokens,
        "total_tokens": total_tokens,
    }


def _sum_turn_tokens(turns: list[TurnRecord]) -> int | None:
    totals = [turn.total_tokens for turn in turns if turn.total_tokens is not None]
    if not totals:
        return None
    return int(sum(totals))


def _normalize_loopback_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != "localhost":
        return url
    host = "127.0.0.1"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host))
