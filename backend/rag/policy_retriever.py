from __future__ import annotations

import os
from typing import Any

import httpx


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
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json() or {}
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
