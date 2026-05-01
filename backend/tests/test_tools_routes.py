from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_tool_order_status_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.tools.get_order_status",
        lambda _oid: {"order_id": "ORD-1", "status": "shipped"},
    )
    r = client.post("/tools/order-status", json={"order_id": "ORD-1"})
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["found"] is True


def test_tool_product_lookup_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.tools.lookup_product",
        lambda _name: {"sku": "SKU-1", "name": "Widget", "price": 9.5, "is_available": True},
    )
    r = client.post("/tools/product-lookup", json={"product_name": "widg"})
    assert r.status_code == 200
    assert r.json()["found"] is True


def test_tool_invoice_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.tools.get_invoice",
        lambda _id: {"invoice_id": "INV-1", "amount": 42.5, "status": "paid"},
    )
    r = client.post("/tools/invoice", json={"invoice_id": "INV-1"})
    assert r.status_code == 200
    assert r.json()["found"] is True


def test_tool_subscription_unsubscribe_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.tools.unsubscribe_subscription",
        lambda _email, **_kwargs: {
            "ok": True,
            "account_email": _email,
            "subscription_status": "unsubscribed",
        },
    )
    r = client.post("/tools/subscription-unsubscribe", json={"account_email": "user@example.com"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_tool_delivery_period_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.tools.get_delivery_period",
        lambda _ref: {"tracking_id": "TRK-1", "promised_delivery_at": "2026-05-10T10:00:00Z"},
    )
    r = client.post("/tools/delivery-period", json={"order_or_tracking": "TRK-1"})
    assert r.status_code == 200
    assert r.json()["found"] is True
