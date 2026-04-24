"""Pytest configuration."""

from __future__ import annotations

import os

import pytest

# Ensure classifier env exists for app import in tests that don't mock it early.
os.environ.setdefault("CLASSIFIER_BENTOML_URL", "http://127.0.0.1:9/classify")


@pytest.fixture
def session_issue_mocks(monkeypatch: pytest.MonkeyPatch) -> dict:
    """
    In-memory session issue state for tests that mock Postgres message APIs but still
    exercise update_session_active_issue / mark_session_resolved / get_session_issue_state.
    """
    buckets: dict[str, dict] = {}

    def _ensure(sid: str) -> dict:
        buckets.setdefault(
            sid,
            {
                "intent": None,
                "user_request": None,
                "issue_category": None,
                "issue_confidence": None,
                "resolved_at": None,
                "escalated": False,
            },
        )
        return buckets[sid]

    def get_s(sid: str) -> dict:
        b = _ensure(sid)
        return {"id": sid, **b}

    def upd(
        sid: str,
        *,
        intent: str,
        user_request: str,
        problem_to_solve: str,
        issue_category: str,
        issue_confidence: float,
    ) -> None:
        b = _ensure(sid)
        b.update(
            intent=intent,
            user_request=user_request,
            problem_to_solve=problem_to_solve,
            issue_category=issue_category,
            issue_confidence=issue_confidence,
            resolved_at=None,
        )

    def mrk(sid: str) -> None:
        _ensure(sid)["resolved_at"] = "2025-01-01T00:00:00+00:00"

    monkeypatch.setattr("backend.api.routes.classify.get_session_issue_state", get_s)
    monkeypatch.setattr("backend.api.routes.classify.update_session_active_issue", upd)
    monkeypatch.setattr("backend.api.routes.classify.mark_session_resolved", mrk)
    return buckets
