from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


def _ollama_base() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _cerebras_base() -> str:
    return os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1").rstrip("/")


def _cerebras_api_key() -> str:
    return os.getenv("CEREBRAS_API_KEY", "").strip()


def chat_completion(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
) -> str:
    """Run a chat completion. Provider is `ollama` or `cerebras`."""
    p = (provider or "").strip().lower()
    if p == "ollama":
        return _ollama_chat(
            model=model,
            messages=messages,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
    if p == "cerebras":
        return _cerebras_chat(
            model=model,
            messages=messages,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def _timeout(timeout_seconds: float | None) -> float:
    if timeout_seconds is not None:
        return float(timeout_seconds)
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float | None,
    temperature: float | None,
    top_p: float | None,
    repeat_penalty: float | None,
) -> str:
    url = f"{_ollama_base()}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    options: dict[str, float] = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if top_p is not None:
        options["top_p"] = float(top_p)
    if repeat_penalty is not None:
        options["repeat_penalty"] = float(repeat_penalty)
    if options:
        payload["options"] = options
    with httpx.Client(timeout=_timeout(timeout_seconds)) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json() or {}
    msg = data.get("message") or {}
    content = msg.get("content")
    if content is not None:
        return str(content).strip()
    # Fallback for older shapes
    if "response" in data:
        return str(data["response"]).strip()
    return ""


def _cerebras_chat(
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float | None,
    temperature: float | None,
    top_p: float | None,
    repeat_penalty: float | None,
) -> str:
    key = _cerebras_api_key()
    if not key:
        raise ValueError("CEREBRAS_API_KEY must be set for cerebras provider")
    url = f"{_cerebras_base()}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": (
            float(temperature)
            if temperature is not None
            else float(os.getenv("CEREBRAS_TEMPERATURE", "0.2"))
        ),
    }
    if top_p is not None:
        payload["top_p"] = float(top_p)
    _ = repeat_penalty
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=_timeout(timeout_seconds)) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json() or {}
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from an LLM reply."""
    t = (text or "").strip()
    if not t:
        return {}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
