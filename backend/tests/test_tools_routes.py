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
        lambda _oid: {"order_number": "ORD-1", "status": "shipped"},
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
