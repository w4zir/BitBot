from __future__ import annotations

import pytest

from backend.llm.vllm_routing import resolve_vllm_target


def test_resolve_vllm_target_maps_served_name_and_hf_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://vllm:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "google/gemma-4-E4B-it")
    monkeypatch.setenv("VLLM_SERVED_NAME", "gemma4:e4b")

    base, served = resolve_vllm_target("gemma4:e4b")
    assert base == "http://vllm:8000/v1"
    assert served == "gemma4:e4b"

    base2, served2 = resolve_vllm_target("google/gemma-4-E4B-it")
    assert base2 == "http://vllm:8000/v1"
    assert served2 == "gemma4:e4b"


def test_resolve_vllm_target_empty_uses_configured_served_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://vllm:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
    monkeypatch.setenv("VLLM_SERVED_NAME", "qwen3:8b")

    base, served = resolve_vllm_target("")
    assert base == "http://vllm:8000/v1"
    assert served == "qwen3:8b"


def test_resolve_vllm_target_unknown_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://default:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "google/gemma-4-E4B-it")
    monkeypatch.setenv("VLLM_SERVED_NAME", "gemma4:e4b")

    base, served = resolve_vllm_target("custom-model")
    assert base == "http://default:8000/v1"
    assert served == "custom-model"
