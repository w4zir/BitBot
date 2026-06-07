from __future__ import annotations

from backend.db.refunds_repo import get_refund_tracking


def test_get_refund_tracking_requires_refund_request_row(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.db.refunds_repo.get_order_status",
        lambda _oid: {"order_id": "ORD-1009", "status": "shipped", "total_amount": 50.0},
    )

    class FakeCursor:
        def execute(self, *_args, **_kwargs) -> None:
            return None

        def fetchone(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "backend.db.refunds_repo.get_connection",
        lambda: FakeConn(),
    )

    out = get_refund_tracking("ORD-1009")
    assert out["found"] is False
    assert out["reason"] == "refund_request_not_found"
    assert out["order_id"] == "ORD-1009"
    assert out["order_status"] == "shipped"


def test_get_refund_tracking_returns_refund_row(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.db.refunds_repo.get_order_status",
        lambda _oid: {"order_id": "ORD-1001", "status": "delivered", "total_amount": 80.0},
    )

    class FakeCursor:
        def execute(self, *_args, **_kwargs) -> None:
            return None

        def fetchone(self):
            return (2, "pending", None, None)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "backend.db.refunds_repo.get_connection",
        lambda: FakeConn(),
    )

    out = get_refund_tracking("ORD-1001")
    assert out["found"] is True
    assert out["refund_id"] == 2
    assert out["refund_decision"] == "pending"
