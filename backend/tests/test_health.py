from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_ready_postgres_ok(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.delenv("ES_HOST", raising=False)
    monkeypatch.delenv("CLASSIFIER_BENTOML_URL", raising=False)

    class _DummyCursor:
        def execute(self, *_args, **_kwargs):
            return None

        def fetchone(self):
            return (1,)

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

    @contextmanager
    def _dummy_connection():
        yield _DummyConn()

    monkeypatch.setattr("backend.api.routes.health.get_connection", _dummy_connection)
    r = client.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["checks"]["postgres"] == "ok"


def test_ready_postgres_unreachable(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.delenv("ES_HOST", raising=False)
    monkeypatch.delenv("CLASSIFIER_BENTOML_URL", raising=False)

    @contextmanager
    def _failing_connection():
        raise RuntimeError("dial timeout")
        yield

    monkeypatch.setattr("backend.api.routes.health.get_connection", _failing_connection)
    r = client.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "degraded"
    assert "unreachable:" in data["checks"]["postgres"]
