from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float


class QueryClassifier:
    """ModernBERT classifier via BentoML HTTP endpoint."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self._endpoint = (endpoint or os.getenv("CLASSIFIER_BENTOML_URL", "")).strip()
        if not self._endpoint:
            raise ValueError("CLASSIFIER_BENTOML_URL must be set for classifier")
        self._timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("CLASSIFIER_BENTOML_TIMEOUT_SECONDS", "5")
        )

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(category="unknown", confidence=0.0)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._endpoint, json={"text": text})
            resp.raise_for_status()
            payload = resp.json() or {}

        return ClassificationResult(
            category=str(payload.get("category", "unknown")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
        )


_QUERY_CLASSIFIER: Optional[Any] = None


def get_query_classifier() -> Any:
    global _QUERY_CLASSIFIER
    if _QUERY_CLASSIFIER is None:
        _QUERY_CLASSIFIER = QueryClassifier()
    return _QUERY_CLASSIFIER


__all__ = ["ClassificationResult", "QueryClassifier", "get_query_classifier"]
