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
                "tags": src.get("tags"),
            }
        )
    return out


def _es_base_url() -> str:
    host = os.getenv("ES_HOST", "").strip()
    port = os.getenv("ES_PORT", "9200").strip()
    scheme = os.getenv("ES_SCHEME", "http").strip() or "http"
    return f"{scheme}://{host}:{port}"


def ping_elasticsearch() -> tuple[bool, str]:
    """Return whether Elasticsearch is reachable with a short reason."""
    host = os.getenv("ES_HOST", "").strip()
    if not host:
        return False, "ES_HOST not set"

    timeout = float(os.getenv("ES_TIMEOUT_SECONDS", "5"))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(_es_base_url())
            response.raise_for_status()
        return True, "ok"
    except httpx.HTTPStatusError as e:
        return False, f"HTTP error: {e}"
    except httpx.RequestError as e:
        return False, f"Request failed: {e}"


def search_policy_docs(query: str, *, size: int = 3) -> list[dict[str, Any]]:
    host = os.getenv("ES_HOST", "").strip()
    if not host:
        return []
    index = os.getenv("ES_POLICY_INDEX", "policy_docs").strip() or "policy_docs"
    url = f"{_es_base_url()}/{index}/_search"
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
    logger.info("Primary policy search returned zero hits for query=%r index=%r", query, index)

    fallback_tags = ["policy", "foodpanda"]
    query_lc = query.lower()
    for token, tag in (
        ("refund", "refund"),
        ("return", "returns"),
        ("cancel", "cancel"),
        ("order", "order"),
        ("damaged", "damaged"),
        ("damage", "damaged"),
        ("item", "items"),
        ("product", "items"),
        ("ship", "shipping"),
        ("deliver", "shipping"),
        ("fraud", "fraud"),
        ("subscription", "subscription"),
        ("loyalty", "loyalty"),
    ):
        if token in query_lc:
            fallback_tags.append(tag)
    fallback_tags = list(dict.fromkeys(fallback_tags))

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

    out = _parse_hits(fallback_data)
    if not out:
        logger.warning(
            "Policy search returned zero hits after fallback for query=%r index=%r tags=%s",
            query,
            index,
            fallback_tags,
        )
    return out
