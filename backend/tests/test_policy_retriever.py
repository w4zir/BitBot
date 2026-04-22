from __future__ import annotations

import httpx

from backend.rag import policy_retriever


class _MockResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://example.local")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status={self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


class _SingleResponseClient:
    def __init__(self, response: _MockResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def __enter__(self) -> _SingleResponseClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict) -> _MockResponse:
        self.calls.append({"url": url, "json": json})
        return self._response

    def get(self, url: str) -> _MockResponse:
        self.calls.append({"url": url})
        return self._response


def test_no_es_host_returns_empty(monkeypatch):
    monkeypatch.delenv("ES_HOST", raising=False)
    assert policy_retriever.search_policy_docs("refund damaged item") == []


def test_primary_hit_parsing_includes_tags(monkeypatch):
    monkeypatch.setenv("ES_HOST", "localhost")
    monkeypatch.setenv("ES_PORT", "9200")
    monkeypatch.setenv("ES_SCHEME", "http")
    monkeypatch.setenv("ES_POLICY_INDEX", "policy_docs")

    payload = {
        "hits": {
            "hits": [
                {
                    "_id": "doc-1",
                    "_score": 2.5,
                    "_source": {
                        "title": "Global Returns & Refund Policy",
                        "content": "Refund details...",
                        "tags": ["foodpanda", "policy", "refund"],
                    },
                }
            ]
        }
    }
    client = _SingleResponseClient(_MockResponse(payload, status_code=200))
    monkeypatch.setattr(policy_retriever.httpx, "Client", lambda timeout: client)

    docs = policy_retriever.search_policy_docs("refund")
    assert len(docs) == 1
    assert docs[0]["title"] == "Global Returns & Refund Policy"
    assert docs[0]["tags"] == ["foodpanda", "policy", "refund"]


def test_damaged_refund_query_fallback_tags(monkeypatch):
    monkeypatch.setenv("ES_HOST", "localhost")
    monkeypatch.setenv("ES_PORT", "9200")
    monkeypatch.setenv("ES_SCHEME", "http")
    monkeypatch.setenv("ES_POLICY_INDEX", "policy_docs")

    primary_empty = _MockResponse({"hits": {"hits": []}}, status_code=200)
    fallback_hit = _MockResponse(
        {
            "hits": {
                "hits": [
                    {
                        "_id": "05-damaged-items-policy",
                        "_score": 1.1,
                        "_source": {
                            "title": "Damaged or Incorrect Item Claims",
                            "content": "Major damage...",
                            "tags": ["foodpanda", "policy", "damaged", "items"],
                        },
                    }
                ]
            }
        },
        status_code=200,
    )

    calls: list[dict] = []

    class _TwoCallClient:
        def __enter__(self) -> _TwoCallClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, json: dict) -> _MockResponse:
            calls.append({"url": url, "json": json})
            if len(calls) == 1:
                return primary_empty
            return fallback_hit

    monkeypatch.setattr(policy_retriever.httpx, "Client", lambda timeout: _TwoCallClient())

    docs = policy_retriever.search_policy_docs("user wants a refund because product is damaged")

    assert len(docs) == 1
    assert docs[0]["id"] == "05-damaged-items-policy"
    assert len(calls) == 2
    fallback_terms = calls[1]["json"]["query"]["bool"]["should"][0]["terms"]["tags"]
    assert "policy" in fallback_terms
    assert "foodpanda" in fallback_terms
    assert "refund" in fallback_terms
    assert "damaged" in fallback_terms
    assert "items" in fallback_terms


def test_ping_ok(monkeypatch):
    monkeypatch.setenv("ES_HOST", "localhost")
    monkeypatch.setenv("ES_PORT", "9200")
    monkeypatch.setenv("ES_SCHEME", "http")

    client = _SingleResponseClient(_MockResponse({}, status_code=200))
    monkeypatch.setattr(policy_retriever.httpx, "Client", lambda timeout: client)

    ok, detail = policy_retriever.ping_elasticsearch()
    assert ok is True
    assert detail == "ok"


def test_ping_unreachable(monkeypatch):
    monkeypatch.setenv("ES_HOST", "localhost")
    monkeypatch.setenv("ES_PORT", "9200")
    monkeypatch.setenv("ES_SCHEME", "http")

    class _BrokenClient:
        def __enter__(self) -> _BrokenClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str):
            raise httpx.RequestError("boom", request=httpx.Request("GET", url))

    monkeypatch.setattr(policy_retriever.httpx, "Client", lambda timeout: _BrokenClient())

    ok, detail = policy_retriever.ping_elasticsearch()
    assert ok is False
    assert "Request failed" in detail
