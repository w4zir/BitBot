from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from backend.llm.providers import extract_json_object
from testing.simulator.config import LlmJudgeThresholdsConfig
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace

DIMENSIONS = (
    "tone",
    "completeness",
    "groundedness",
    "escalation_appropriateness",
    "resolution_clarity",
)


@dataclass
class LlmJudgeResult:
    passed: bool
    scores: dict[str, float]
    rationales: dict[str, str]
    thresholds: dict[str, float]
    failures: list[str]
    provider: str
    model: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: dict[str, Any] | None = None


def evaluate_llm_judge(
    *,
    trace: ConversationTrace,
    scenario: ScenarioInstance,
    provider: str,
    model: str,
    thresholds: LlmJudgeThresholdsConfig,
) -> LlmJudgeResult:
    prompt = _build_prompt(trace, scenario)
    failures: list[str] = []
    scores: dict[str, float] = {}
    rationales: dict[str, str] = {}
    thresholds_map = {
        "tone": float(thresholds.tone),
        "completeness": float(thresholds.completeness),
        "groundedness": float(thresholds.groundedness),
        "escalation_appropriateness": float(thresholds.escalation_appropriateness),
        "resolution_clarity": float(thresholds.resolution_clarity),
    }
    try:
        content, usage, latency_ms = _chat_completion_with_usage(
            provider=provider,
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        parsed = extract_json_object(content)
        if not parsed:
            failures.append("LLM judge returned invalid JSON.")
            return LlmJudgeResult(
                passed=False,
                scores=scores,
                rationales=rationales,
                thresholds=thresholds_map,
                failures=failures,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_tokens=usage.get("cache_tokens"),
                total_tokens=usage.get("total_tokens"),
                raw_response=None,
            )

        for name in DIMENSIONS:
            block = parsed.get(name) if isinstance(parsed, dict) else None
            if not isinstance(block, dict):
                failures.append(f"Missing judge dimension '{name}'.")
                continue
            rationale = str(block.get("rationale") or "").strip()
            try:
                score = float(block.get("score"))
            except (TypeError, ValueError):
                failures.append(f"Invalid score for '{name}'.")
                continue
            scores[name] = score
            rationales[name] = rationale
            if score < thresholds_map[name]:
                failures.append(
                    f"LLM judge score below threshold for '{name}' ({score:.2f} < {thresholds_map[name]:.2f})."
                )

        return LlmJudgeResult(
            passed=not failures,
            scores=scores,
            rationales=rationales,
            thresholds=thresholds_map,
            failures=failures,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_tokens=usage.get("cache_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_response=parsed if isinstance(parsed, dict) else None,
        )
    except Exception as exc:  # noqa: BLE001
        return LlmJudgeResult(
            passed=False,
            scores=scores,
            rationales=rationales,
            thresholds=thresholds_map,
            failures=[f"LLM judge request failed: {exc}"],
            provider=provider,
            model=model,
        )


_SYSTEM_PROMPT = """You are evaluating a customer support conversation.
For each dimension, provide concise rationale then a 1-5 score.
Return JSON only.
Expected shape:
{
  "tone": {"rationale":"...", "score": 1-5},
  "completeness": {"rationale":"...", "score": 1-5},
  "groundedness": {"rationale":"...", "score": 1-5},
  "escalation_appropriateness": {"rationale":"...", "score": 1-5},
  "resolution_clarity": {"rationale":"...", "score": 1-5}
}
"""


def _build_prompt(trace: ConversationTrace, scenario: ScenarioInstance) -> str:
    turns = []
    for turn in trace.turns:
        turns.append(f"User: {turn.user_message}")
        turns.append(f"Assistant: {turn.agent_response}")
    policy_doc_names: list[str] = []
    if trace.turns and isinstance(trace.turns[-1].context_data, dict):
        names = trace.turns[-1].context_data.get("policy_doc_names")
        if isinstance(names, list):
            policy_doc_names = [str(item) for item in names]
    policy_docs_text = ", ".join(policy_doc_names) if policy_doc_names else "(none)"
    entity_summary = json.dumps(scenario.entity, ensure_ascii=False)
    final_response = trace.turns[-1].agent_response if trace.turns else ""
    return (
        "## Conversation\n"
        + "\n".join(turns)
        + "\n\n## Agent final response\n"
        + final_response
        + "\n\n## Context available to agent\n"
        + f"Policy docs retrieved: {policy_docs_text}\n"
        + f"Entity data: {entity_summary}\n"
        + f"Outcome status: {trace.final_outcome_status}\n"
    )


def _chat_completion_with_usage(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, int | None], float]:
    p = (provider or os.getenv("SIMULATOR_LLM_PROVIDER", "ollama")).strip().lower()
    if p == "ollama":
        return _ollama_chat(model=model, messages=messages)
    if p == "cerebras":
        return _cerebras_chat(model=model, messages=messages)
    raise ValueError(f"Unsupported LLM provider for simulator judge: {provider!r}")


def _timeout_seconds() -> float:
    return float(os.getenv("SIMULATOR_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120")))


def _ollama_chat(*, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int | None], float]:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    started = time.perf_counter()
    with httpx.Client(timeout=_timeout_seconds()) as client:
        response = client.post(f"{base}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json() or {}
    latency_ms = (time.perf_counter() - started) * 1000.0
    msg = data.get("message") or {}
    content = str(msg.get("content") or data.get("response") or "").strip()
    usage = {
        "input_tokens": _int_or_none(data.get("prompt_eval_count")),
        "output_tokens": _int_or_none(data.get("eval_count")),
        "cache_tokens": _int_or_none(data.get("prompt_eval_cache_count")),
        "total_tokens": _sum_tokens(
            _int_or_none(data.get("prompt_eval_count")),
            _int_or_none(data.get("eval_count")),
        ),
    }
    return content, usage, latency_ms


def _cerebras_chat(*, model: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int | None], float]:
    api_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY must be set for cerebras provider")
    base = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1").rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("CEREBRAS_TEMPERATURE", "0.2")),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = time.perf_counter()
    with httpx.Client(timeout=_timeout_seconds()) as client:
        response = client.post(f"{base}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json() or {}
    latency_ms = (time.perf_counter() - started) * 1000.0
    choices = data.get("choices") or []
    msg = (choices[0] if choices else {}).get("message") or {}
    usage_data = data.get("usage") or {}
    usage = {
        "input_tokens": _int_or_none(usage_data.get("prompt_tokens")),
        "output_tokens": _int_or_none(usage_data.get("completion_tokens")),
        "cache_tokens": _int_or_none(usage_data.get("cache_tokens")),
        "total_tokens": _int_or_none(usage_data.get("total_tokens")),
    }
    return str(msg.get("content") or "").strip(), usage, latency_ms


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_tokens(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)
