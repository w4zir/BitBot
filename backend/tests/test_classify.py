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
    assert data["session_issue"]["is_resolved"] is False


def test_classify_full_flow_no_issue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
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
    assert data["session_issue"]["intent"] == "no_issue_chat"
    assert data["session_issue"]["user_request"] == "Just saying hi"
    assert data["session_issue"]["is_resolved"] is True
    sid = data["session_id"]
    assert session_issue_mocks[sid]["resolved_at"] is not None


def test_classify_full_flow_validation_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
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
        "backend.agent.issue_graph.get_intents_for_category",
        lambda _category: ["order_status", "cancel_order"],
    )

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent": "order_status", "problem_to_solve": "Check order status"}'
        return '{"valid": false, "missing_field_names": ["order_id", "email"], "notes": "need ids"}'

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)

    r = client.post("/classify", json={"text": "I need help", "full_flow": True})
    assert r.status_code == 200
    data = r.json()
    assert data["validation_ok"] is False
    assert data["intent"] in ("cancel_order", "order_status")
    assert "order_id" in (data.get("validation_missing") or [])
    assert data.get("assistant_reply")
    assert data["session_issue"]["is_resolved"] is False


def test_classify_full_flow_interrupt_sets_pending_action(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
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
        lambda order_id: {"order_id": order_id, "status": "shipped", "total_amount": 120.0},
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
        "backend.agent.issue_graph.get_intents_for_category",
        lambda _category: ["get_refund"],
    )

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent": "get_refund", "problem_to_solve": "Request refund"}'
        return '{"valid": true, "missing_field_names": [], "notes": "ok"}'

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)

    r = client.post(
        "/classify",
        json={"text": "I want a refund for ORD-12345", "full_flow": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["assistant_reply"]
    assert data["assistant_metadata"].get("pending_human_action") is True
    assert data["assistant_metadata"].get("action_type") == "refund_escalation"
    assert data["session_issue"]["is_resolved"] is False


def test_classify_intent_stays_locked_second_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
    """After order_status is stored, a later turn must not reclassify intent from the latest text."""
    messages_store: list[dict] = []

    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr(
        "backend.api.routes.classify.create_session",
        lambda: "00000000-0000-0000-0000-000000000004",
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

    monkeypatch.setattr(
        "backend.agent.issue_graph.get_order_status",
        lambda order_id: {
            "order_id": order_id,
            "status": "shipped",
            "total_amount": 99.0,
        },
    )

    qc = MagicMock()
    qc.classify.side_effect = [
        ClassificationResult(category="order", confidence=0.9),
        ClassificationResult(category="refund", confidence=0.99),
    ]
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)
    monkeypatch.setattr("backend.agent.issue_graph.get_query_classifier", lambda: qc)
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_intents_for_category",
        lambda _category: ["order_status", "cancel_order"],
    )

    chat_calls: list[int] = []

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent": "order_status", "problem_to_solve": "Track order"}'
        chat_calls.append(1)
        if len(chat_calls) == 1:
            return '{"valid": false, "missing_field_names": ["order_id"], "notes": "need id"}'
        if len(chat_calls) == 2:
            return '{"valid": true, "missing_field_names": [], "notes": "ok"}'
        return "Your order ORD-12345 is shipped."

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)

    r1 = client.post(
        "/classify",
        json={"text": "What is the status of my order", "full_flow": True},
    )
    assert r1.status_code == 200
    assert r1.json()["intent"] == "order_status"
    assert r1.json()["session_issue"]["intent"] == "order_status"
    assert r1.json()["session_issue"]["is_resolved"] is False

    r2 = client.post(
        "/classify",
        json={
            "text": "ORD-12345",
            "full_flow": True,
            "session_id": "00000000-0000-0000-0000-000000000004",
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["intent"] == "order_status"
    assert d2["category"] == "order"
    assert d2["assistant_metadata"].get("intent_classifier") == "session_locked"
    assert d2["session_issue"]["user_request"] == "What is the status of my order"
    assert d2["session_issue"]["is_resolved"] is True


def test_classify_new_issue_after_resolution(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
    messages_store: list[dict] = []

    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr(
        "backend.api.routes.classify.create_session",
        lambda: "00000000-0000-0000-0000-000000000005",
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
    qc.classify.side_effect = [
        ClassificationResult(category="no_issue", confidence=0.99),
        ClassificationResult(category="order", confidence=0.91),
    ]
    monkeypatch.setattr("backend.api.routes.classify.get_query_classifier", lambda: qc)
    monkeypatch.setattr("backend.agent.issue_graph.get_query_classifier", lambda: qc)
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_intents_for_category",
        lambda _category: ["order_status", "cancel_order"],
    )

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent": "order_status", "problem_to_solve": "Track order"}'
        if "validate whether the user provided" in system:
            return '{"valid": false, "missing_field_names": ["order_id"], "notes": "need id"}'
        return "Hello!"

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)

    sid = "00000000-0000-0000-0000-000000000005"
    r1 = client.post("/classify", json={"text": "Just hi", "full_flow": True})
    assert r1.status_code == 200
    assert r1.json()["session_issue"]["is_resolved"] is True

    r2 = client.post(
        "/classify",
        json={"text": "My order is late", "full_flow": True, "session_id": sid},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["category"] == "order"
    assert d2["intent"] in ("cancel_order", "order_status")
    assert d2["session_issue"]["user_request"] == "My order is late"
    assert d2["session_issue"]["is_resolved"] is False


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


def test_classify_exposes_agent_state_and_policy_variable_maps(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
    messages_store: list[dict] = []
    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr(
        "backend.api.routes.classify.create_session",
        lambda: "00000000-0000-0000-0000-000000000101",
    )
    monkeypatch.setattr("backend.api.routes.classify.get_session", lambda sid: {"id": sid})
    monkeypatch.setattr(
        "backend.api.routes.classify.append_message",
        lambda sid, role, content, metadata=None: messages_store.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        ),
    )
    monkeypatch.setattr("backend.api.routes.classify.list_messages", lambda _sid: list(messages_store))
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_query_classifier",
        lambda: MagicMock(classify=lambda _text: ClassificationResult(category="order", confidence=0.95)),
    )
    monkeypatch.setattr(
        "backend.api.routes.classify.get_query_classifier",
        lambda: MagicMock(classify=lambda _text: ClassificationResult(category="order", confidence=0.95)),
    )
    monkeypatch.setattr("backend.agent.issue_graph.search_policy_docs", lambda _q: [{"content": "policy rule"}])
    monkeypatch.setattr("backend.agent.issue_graph.get_intents_for_category", lambda _category: ["order_status"])

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent":"order_status","problem_to_solve":"Track order status"}'
        return '{"valid": false, "missing_field_names": ["order_id"], "notes": "need id"}'

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)
    r = client.post("/classify", json={"text": "Track ORD-1004", "full_flow": True})
    assert r.status_code == 200
    data = r.json()
    md = data["assistant_metadata"]
    assert isinstance(md.get("agent_state"), dict)
    assert isinstance(md.get("stage_metadata"), dict)
    policy_constraints = md.get("policy_constraints") or {}
    assert isinstance(policy_constraints.get("variables"), dict)
    assert isinstance(policy_constraints.get("validation_results"), dict)


def test_classify_validation_wait_limit_escalates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_issue_mocks: dict
) -> None:
    messages_store: list[dict] = []
    sid = "00000000-0000-0000-0000-000000000102"
    monkeypatch.setenv("AGENT_VALIDATION_MAX_USER_WAITS", "2")
    monkeypatch.setattr("backend.api.routes.classify.postgres_configured", lambda: True)
    monkeypatch.setattr("backend.api.routes.classify.create_session", lambda: sid)
    monkeypatch.setattr("backend.api.routes.classify.get_session", lambda _sid: {"id": _sid})
    monkeypatch.setattr(
        "backend.api.routes.classify.append_message",
        lambda _sid, role, content, metadata=None: messages_store.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        ),
    )
    monkeypatch.setattr("backend.api.routes.classify.list_messages", lambda _sid: list(messages_store))
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_query_classifier",
        lambda: MagicMock(classify=lambda _text: ClassificationResult(category="order", confidence=0.9)),
    )
    monkeypatch.setattr(
        "backend.api.routes.classify.get_query_classifier",
        lambda: MagicMock(classify=lambda _text: ClassificationResult(category="order", confidence=0.9)),
    )
    monkeypatch.setattr(
        "backend.agent.issue_graph.get_intents_for_category",
        lambda _category: ["order_status"],
    )
    monkeypatch.setattr("backend.agent.issue_graph.search_policy_docs", lambda _q: [])

    def chat_completion(**kwargs):
        msgs = kwargs.get("messages") or []
        system = str(msgs[0].get("content") if msgs else "")
        if "classify a customer support session" in system:
            return '{"intent":"order_status","problem_to_solve":"Track order status"}'
        return '{"valid": false, "missing_field_names": ["order_id"], "notes": "need order id"}'

    monkeypatch.setattr("backend.agent.issue_graph.chat_completion", chat_completion)

    r1 = client.post("/classify", json={"text": "Need help", "full_flow": True})
    assert r1.status_code == 200
    assert r1.json()["assistant_metadata"]["validation_wait_count"] == 1
    assert r1.json()["assistant_metadata"]["outcome_status"] == "needs_more_data"

    r2 = client.post("/classify", json={"text": "still help", "full_flow": True, "session_id": sid})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["assistant_metadata"]["validation_wait_count"] == 2
    assert d2["assistant_metadata"]["outcome_status"] == "pending_escalation"
