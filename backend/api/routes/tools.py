from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.db.orders_repo import get_order_status
from backend.db.products_repo import lookup_product
from backend.db.refunds_repo import get_refund_context

router = APIRouter(prefix="/tools", tags=["tools"])


class OrderStatusRequest(BaseModel):
    order_id: str = Field(min_length=1)


class ProductLookupRequest(BaseModel):
    product_name: str = Field(min_length=1)


class RefundContextRequest(BaseModel):
    order_id: str = Field(min_length=1)


@router.post("/order-status")
async def tool_order_status(req: OrderStatusRequest) -> dict[str, Any]:
    data = get_order_status(req.order_id)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/product-lookup")
async def tool_product_lookup(req: ProductLookupRequest) -> dict[str, Any]:
    data = lookup_product(req.product_name)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/refund-context")
async def tool_refund_context(req: RefundContextRequest) -> dict[str, Any]:
    data = get_refund_context(req.order_id)
    return {"ok": True, "data": data, "found": bool(data)}

