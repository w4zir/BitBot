from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "")


def resolve_vllm_target(model: str) -> tuple[str, str]:
    """Return ``(api_base, served_model_name)`` for a logical model id."""
    requested = (model or "").strip()
    base = _env("VLLM_API_BASE", "http://localhost:8001/v1").rstrip("/")
    served = _env("VLLM_SERVED_NAME", "") or _env("VLLM_MODEL", "")
    hf_model = _env("VLLM_MODEL", "")

    if not requested:
        return base, served

    norm_req = _normalize(requested)
    if norm_req in {_normalize(served), _normalize(hf_model)}:
        return base, served

    return base, requested
