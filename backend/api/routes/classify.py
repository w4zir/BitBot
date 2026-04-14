from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter

from backend.agent.issue_graph import run_issue_classification

router = APIRouter(tags=["classification"])


class ClassifyRequest(BaseModel):
    text: str = Field(default="", description="User utterance to classify.")


class ClassifyResponse(BaseModel):
    text: str
    is_issue: bool
    confidence: float
    label: str


@router.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Run LangGraph issue/no_issue flow (ModernBERT via Bento)."""
    result = run_issue_classification(req.text)
    return ClassifyResponse(**result)
