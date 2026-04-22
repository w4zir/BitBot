from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter

from backend.rag.policy_retriever import ping_elasticsearch

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "service": "bitbot-backend"}


@router.get("/health/ready")
async def ready() -> dict[str, Any]:
    """Readiness: optional checks for Postgres, Elasticsearch, classifier."""
    out: dict[str, Any] = {"status": "ok", "checks": {}}

    pg_host = os.getenv("POSTGRES_HOST", "").strip()
    if pg_host:
        out["checks"]["postgres"] = "configured"

    es_host = os.getenv("ES_HOST", "").strip()
    if es_host:
        ok, detail = ping_elasticsearch()
        if ok:
            out["checks"]["elasticsearch"] = "ok"
        else:
            out["checks"]["elasticsearch"] = f"unreachable: {detail}"
            out["status"] = "degraded"

    clf = os.getenv("CLASSIFIER_BENTOML_URL", "").strip()
    if clf:
        try:
            base = clf.rsplit("/classify", 1)[0]
            health_url = f"{base}/health"
            with httpx.Client(timeout=3.0) as client:
                r = client.post(health_url, json={})
                r.raise_for_status()
            out["checks"]["classifier"] = "ok"
        except Exception as e:  # noqa: BLE001
            out["checks"]["classifier"] = f"unreachable: {e!s}"
            out["status"] = "degraded"

    return out
