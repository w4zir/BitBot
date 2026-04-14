"""
Minimal LangGraph flow: ingest text -> classify issue/no_issue via ModernBERT (Bento).

Single linear graph: classify -> END.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from backend.rag.query_classifier import ClassificationResult, get_query_classifier


class IssueGraphState(TypedDict):
    """State for binary issue classification."""

    text: str
    is_issue: bool
    confidence: float
    label: str


def _classify_node(state: IssueGraphState) -> IssueGraphState:
    qc = get_query_classifier()
    result: ClassificationResult = qc.classify(state.get("text") or "")
    return {
        "text": state.get("text") or "",
        "is_issue": result.is_issue,
        "confidence": result.confidence,
        "label": result.label,
    }


def build_issue_classification_graph():
    """Compile a small StateGraph for issue / no_issue routing."""
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
            "is_issue": False,
            "confidence": 0.0,
            "label": "no_issue",
        }
    )
    return {
        "text": out.get("text", ""),
        "is_issue": bool(out.get("is_issue", False)),
        "confidence": float(out.get("confidence", 0.0)),
        "label": str(out.get("label", "no_issue")),
    }
