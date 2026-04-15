"""
Minimal LangGraph flow: ingest text -> classify category via ModernBERT (Bento).

Single linear graph: classify -> END.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.rag.query_classifier import ClassificationResult, get_query_classifier


class IssueGraphState(TypedDict):
    """State for category classification."""

    text: str
    category: str
    confidence: float


def _classify_node(state: IssueGraphState) -> IssueGraphState:
    qc = get_query_classifier()
    result: ClassificationResult = qc.classify(state.get("text") or "")
    return {
        "text": state.get("text") or "",
        "category": result.category,
        "confidence": result.confidence,
    }


def build_issue_classification_graph():
    """Compile a small StateGraph for category classification."""
    g: StateGraph[IssueGraphState] = StateGraph(IssueGraphState)
    g.add_node("classify", _classify_node)
    g.set_entry_point("classify")
    g.add_edge("classify", END)
    return g.compile()


# Singleton compiled graph (thread-safe for invoke)
_COMPILED = None


def get_issue_classification_graph():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_issue_classification_graph()
    return _COMPILED


def run_issue_classification(text: str) -> dict:
    """Invoke the graph and return a plain dict for API responses."""
    graph = get_issue_classification_graph()
    out = graph.invoke(
        {
            "text": text or "",
            "category": "unknown",
            "confidence": 0.0,
        }
    )
    return {
        "text": out.get("text", ""),
        "category": str(out.get("category", "unknown")),
        "confidence": float(out.get("confidence", 0.0)),
    }
