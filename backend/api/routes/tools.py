from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.db.delivery_repo import get_delivery_period
from backend.db.invoices_repo import get_invoice
from backend.db.orders_repo import cancel_order, get_order_status, update_shipping_address
from backend.db.payments_repo import get_payment, get_refund_tracking, list_payment_methods
from backend.db.products_repo import (
    get_product_availability,
    get_product_info,
    get_product_price,
    lookup_product,
)
from backend.db.refunds_repo import create_refund_request, get_refund_context
from backend.db.subscriptions_repo import get_subscription, unsubscribe_subscription
from backend.db.support_repo import create_support_ticket

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


class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1)


class InvoiceRequest(BaseModel):
    invoice_id: str = Field(min_length=1)


class AccountEmailRequest(BaseModel):
    account_email: str = Field(min_length=1)


class ComplaintRequest(BaseModel):
    complaint: str = Field(min_length=1)


class ContactRequest(BaseModel):
    summary: str = Field(min_length=1)


class DeliveryPeriodRequest(BaseModel):
    order_or_tracking: str = Field(min_length=1)


@router.post("/order-status")
async def tool_order_status(req: OrderStatusRequest) -> dict[str, Any]:
    data = get_order_status(req.order_id)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/product-lookup")
async def tool_product_lookup(req: ProductLookupRequest) -> dict[str, Any]:
    data = lookup_product(req.product_name)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/product-info")
async def tool_product_info(req: ProductLookupRequest) -> dict[str, Any]:
    data = get_product_info(req.product_name)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/product-price")
async def tool_product_price(req: ProductLookupRequest) -> dict[str, Any]:
    data = get_product_price(req.product_name)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/product-availability")
async def tool_product_availability(req: ProductLookupRequest) -> dict[str, Any]:
    data = get_product_availability(req.product_name)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/refund-context")
async def tool_refund_context(req: RefundContextRequest) -> dict[str, Any]:
    data = get_refund_context(req.order_id)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/cancel-order")
async def tool_cancel_order(req: CancelOrderRequest) -> dict[str, Any]:
    data = cancel_order(req.order_id, update_source="human")
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/create-refund-request")
async def tool_create_refund_request(req: RefundCreateRequest) -> dict[str, Any]:
    data = create_refund_request(req.order_id, req.reason, update_source="human")
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/update-shipping-address")
async def tool_update_shipping_address(req: UpdateShippingAddressRequest) -> dict[str, Any]:
    data = update_shipping_address(req.order_id, req.new_address, update_source="human")
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/payment")
async def tool_payment_lookup(req: TransactionRequest) -> dict[str, Any]:
    data = get_payment(req.transaction_id)
    return {"ok": True, "data": data, "found": bool(data)}


@router.get("/payment-methods")
async def tool_payment_methods() -> dict[str, Any]:
    return {"ok": True, "data": {"methods": list_payment_methods()}}


@router.post("/payment-track-refund")
async def tool_payment_track_refund(req: TransactionRequest) -> dict[str, Any]:
    data = get_refund_tracking(req.transaction_id)
    return {"ok": bool(data.get("found")), "data": data}


@router.post("/invoice")
async def tool_invoice(req: InvoiceRequest) -> dict[str, Any]:
    data = get_invoice(req.invoice_id)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/subscription-status")
async def tool_subscription_status(req: AccountEmailRequest) -> dict[str, Any]:
    data = get_subscription(req.account_email)
    return {"ok": True, "data": data, "found": bool(data)}


@router.post("/subscription-unsubscribe")
async def tool_subscription_unsubscribe(req: AccountEmailRequest) -> dict[str, Any]:
    data = unsubscribe_subscription(req.account_email, update_source="human")
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/contact-handoff")
async def tool_contact_handoff(req: ContactRequest) -> dict[str, Any]:
    data = create_support_ticket(
        issue_type="contact",
        payload={"summary": req.summary},
        routing_result="human_agent_queue",
        update_source="human",
    )
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/complaint")
async def tool_complaint(req: ComplaintRequest) -> dict[str, Any]:
    data = create_support_ticket(
        issue_type="feedback",
        payload={"complaint": req.complaint},
        routing_result="complaint_queue",
        update_source="human",
    )
    return {"ok": bool(data.get("ok")), "data": data}


@router.post("/delivery-period")
async def tool_delivery_period(req: DeliveryPeriodRequest) -> dict[str, Any]:
    data = get_delivery_period(req.order_or_tracking)
    return {"ok": True, "data": data, "found": bool(data)}

