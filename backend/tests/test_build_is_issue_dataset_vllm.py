"""Tests for vLLM wiring in ``training/scripts/build_is_issue_dataset.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "build_is_issue_dataset",
    _ROOT / "training" / "scripts" / "build_is_issue_dataset.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)


def test_resolve_provider_accepts_vllm() -> None:
    assert _mod.resolve_provider("vllm") == "vllm"


def test_resolve_model_vllm_requires_env_or_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="VLLM_MODEL"):
        _mod.resolve_model("vllm", None)


def test_resolve_model_vllm_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", "org/test-model")
    assert _mod.resolve_model("vllm", None) == "org/test-model"


def test_resolve_model_vllm_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", "org/env-model")
    assert _mod.resolve_model("vllm", "cli-model") == "cli-model"


def test_resolve_base_url_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://custom:9999/v1/")
    assert _mod.resolve_base_url("vllm") == "http://custom:9999/v1"
