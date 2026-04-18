from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.rag.query_classifier import ClassificationResult


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_classify_simple_bento_only(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    qc = MagicMock()
    qc.classify.return_value = ClassificationResult(category="ORDER", confidence=0.91)
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)

    r = client.post("/classify", json={"text": "My order is late", "full_flow": False})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "ORDER"
    assert data["confidence"] == pytest.approx(0.91)
    assert data["session_id"] is None
    assert data["messages"] == []
    assert data["intent"] == ""
    assert data["procedure_id"] == ""


def test_classify_full_flow_no_issue(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    messages_store: list[dict] = []

    def fake_postgres_ok() -> bool:
        return True

    def create_session() -> str:
        return "00000000-0000-0000-0000-000000000001"

    def get_session(sid: str):
        return {"id": sid}

    def append_message(sid: str, role: str, content: str, metadata=None):
        messages_store.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )

    def list_messages(sid: str):
        _ = sid
        return list(messages_store)

    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", fake_postgres_ok)
    monkeypatch.setattr("backend.api.routes.classify.create_session", create_session)
    monkeypatch.setattr("backend.api.routes.classify.get_session", get_session)
    monkeypatch.setattr("backend.api.routes.classify.append_message", append_message)
    monkeypatch.setattr("backend.api.routes.classify.list_messages", list_messages)

    qc = MagicMock()
    qc.classify.return_value = ClassificationResult(category="no_issue", confidence=0.99)
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)
    monkeypatch.setattr("backend.agent.issue_graph.get_query_classifier", lambda: qc)

    monkeypatch.setattr(
        "backend.agent.issue_graph.chat_completion",
        lambda **kwargs: "Hello! How can I help?",
    )

    r = client.post("/classify", json={"text": "Just saying hi", "full_flow": True})
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "00000000-0000-0000-0000-000000000001"
    assert data["category"] == "no_issue"
    assert data["intent"] == "no_issue_chat"
    assert data["procedure_id"] == "no_issue_chat"
    assert data["assistant_reply"] == "Hello! How can I help?"
    assert len(data["messages"]) >= 2


def test_classify_full_flow_validation_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    messages_store: list[dict] = []

    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr(
        "backend.api.routes.classify.create_session",
        lambda: "00000000-0000-0000-0000-000000000002",
    )
    monkeypatch.setattr("backend.api.routes.classify.get_session", lambda sid: {"id": sid})

    def append_message(sid: str, role: str, content: str, metadata=None):
        messages_store.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )

    def list_messages(sid: str):
        _ = sid
        return list(messages_store)

    monkeypatch.setattr("backend.api.routes.classify.append_message", append_message)
    monkeypatch.setattr("backend.api.routes.classify.list_messages", list_messages)

    qc = MagicMock()
    qc.classify.return_value = ClassificationResult(category="order", confidence=0.88)
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)
    monkeypatch.setattr("backend.agent.issue_graph.get_query_classifier", lambda: qc)

    monkeypatch.setattr(
        "backend.agent.issue_graph.chat_completion",
        lambda **kwargs: '{"valid": false, "missing_field_names": ["order_id", "email"], "notes": "need ids"}',
    )

    r = client.post("/classify", json={"text": "I need help", "full_flow": True})
    assert r.status_code == 200
    data = r.json()
    assert data["validation_ok"] is False
    assert data["intent"] in ("cancel_order", "order_status")
    assert "order_id" in (data.get("validation_missing") or [])
    assert data.get("assistant_reply")


def test_classify_full_flow_interrupt_sets_pending_action(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_store: list[dict] = []

    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr(
        "backend.api.routes.classify.create_session",
        lambda: "00000000-0000-0000-0000-000000000003",
    )
    monkeypatch.setattr("backend.api.routes.classify.get_session", lambda sid: {"id": sid})
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_order_status",
        lambda order_id: {"order_number": order_id, "status": "shipped", "total_amount": 120.0},
    )
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_refund_context",
        lambda order_id: {"refund_order_status": "shipped", "refund_order_total_amount": 120.0},
    )
    monkeypatch.setattr("backend.agent.issue_graph.search_policy_docs", lambda _q: [])

    def append_message(sid: str, role: str, content: str, metadata=None):
        messages_store.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )

    monkeypatch.setattr("backend.api.routes.classify.append_message", append_message)
    monkeypatch.setattr("backend.api.routes.classify.list_messages", lambda _sid: list(messages_store))

    qc = MagicMock()
    qc.classify.return_value = ClassificationResult(category="refund", confidence=0.91)
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)
    monkeypatch.setattr("backend.agent.issue_graph.get_query_classifier", lambda: qc)

    monkeypatch.setattr(
        "backend.agent.issue_graph.chat_completion",
        lambda **kwargs: '{"valid": true, "missing_field_names": [], "notes": "ok"}',
    )

    r = client.post(
        "/classify",
        json={"text": "I want a refund for ORD-12345", "full_flow": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["assistant_reply"]
    assert data["assistant_metadata"].get("pending_human_action") is True
    assert data["assistant_metadata"].get("action_type") == "refund_escalation"


def test_escalation_decision_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.api.routes.escalations.get_session", lambda sid: {"id": sid})
    monkeypatch.setattr(
        "backend.api.routes.escalations.list_messages",
        lambda _sid: [
            {
                "role": "assistant",
                "content": "Escalate?",
                "metadata": {
                    "pending_human_action": True,
                    "action_id": "act-1",
                },
            }
        ],
    )
    inserted: list[dict] = []

    def append_message(sid: str, role: str, content: str, metadata=None):
        inserted.append({"sid": sid, "role": role, "content": content, "metadata": metadata or {}})

    monkeypatch.setattr("backend.api.routes.escalations.append_message", append_message)

    class _DummyCursor:
        def execute(self, *_args, **_kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _DummyConn:
        def cursor(self):
            return _DummyCursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("backend.api.routes.escalations.get_connection", lambda: _DummyConn())

    r = client.post(
        "/escalations/decision",
        json={
            "session_id": "00000000-0000-0000-0000-000000000003",
            "action_id": "act-1",
            "decision": "accept",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["decision"] == "accept"
    assert inserted and inserted[0]["metadata"]["pending_human_action"] is False
