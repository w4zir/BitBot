from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _parse_hits(data: dict[str, Any]) -> list[dict[str, Any]]:
    hits = ((data.get("hits") or {}).get("hits") or [])
    out: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.get("_source") or {}
        out.append(
            {
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "title": src.get("title"),
                "content": src.get("content"),
            }
        )
    return out


def search_policy_docs(query: str, *, size: int = 3) -> list[dict[str, Any]]:
    host = os.getenv("ES_HOST", "").strip()
    if not host:
        return []
    port = os.getenv("ES_PORT", "9200").strip()
    scheme = os.getenv("ES_SCHEME", "http").strip() or "http"
    index = os.getenv("ES_POLICY_INDEX", "policy_docs").strip() or "policy_docs"
    url = f"{scheme}://{host}:{port}/{index}/_search"
    payload = {
        "size": int(size),
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content", "tags"],
            }
        },
    }
    timeout = float(os.getenv("ES_TIMEOUT_SECONDS", "5"))
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json() or {}
    except httpx.HTTPStatusError as e:
        logger.warning("Policy search failed with HTTP status error: %s", e)
        return []
    except httpx.RequestError as e:
        logger.warning("Policy search request failed: %s", e)
        return []

    out = _parse_hits(data)
    if out:
        return out

    fallback_tags = ["policy", "foodpanda"]
    query_lc = query.lower()
    if "refund" in query_lc:
        fallback_tags.append("refund")
    if "cancel" in query_lc:
        fallback_tags.append("cancel")
    if "order" in query_lc:
        fallback_tags.append("order")

    fallback_payload = {
        "size": int(size),
        "query": {
            "bool": {
                "should": [
                    {"terms": {"tags": fallback_tags}},
                    {"match": {"tags": "policy"}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=fallback_payload)
            r.raise_for_status()
            fallback_data = r.json() or {}
    except httpx.HTTPStatusError as e:
        logger.warning("Policy fallback search failed with HTTP status error: %s", e)
        return []
    except httpx.RequestError as e:
        logger.warning("Policy fallback search request failed: %s", e)
        return []

    return _parse_hits(fallback_data)
