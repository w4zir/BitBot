from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.db.orders_repo import cancel_order, get_order_status, update_shipping_address
from backend.db.products_repo import lookup_product
from backend.db.refunds_repo import create_refund_request, get_refund_context

router = APIRouter(prefix="/tools", tags=["tools"])


class OrderStatusRequest(BaseModel):
    order_id: str = Field(min_length=1)


class ProductLookupRequest(BaseModel):
    product_name: str = Field(min_length=1)


class RefundContextRequest(BaseModel):
    order_id: str = Field(min_length=1)


class CancelOrderRequest(BaseModel):
    order_id: str = Field(min_length=1)


class RefundCreateRequest(BaseModel):
    order_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class UpdateShippingAddressRequest(BaseModel):
    order_id: str = Field(min_length=1)
    new_address: str = Field(min_length=1)


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


@router.post("/cancel-order")
async def tool_cancel_order(req: CancelOrderRequest) -> dict[str, Any]:
    data = cancel_order(req.order_id)
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/create-refund-request")
async def tool_create_refund_request(req: RefundCreateRequest) -> dict[str, Any]:
    data = create_refund_request(req.order_id, req.reason)
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/update-shipping-address")
async def tool_update_shipping_address(req: UpdateShippingAddressRequest) -> dict[str, Any]:
    data = update_shipping_address(req.order_id, req.new_address)
    return {"ok": bool(data.get("ok")), "data": data}

