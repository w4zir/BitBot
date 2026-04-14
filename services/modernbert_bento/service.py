from __future__ import annotations

import os
from typing import Any, Dict

import bentoml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODERNBERT_MODEL_DIR", "/models/modernbert_finetuned")
DEFAULT_THRESHOLD = 0.7


def _get_threshold() -> float:
    raw = os.getenv("CLASSIFIER_THRESHOLD", str(DEFAULT_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_THRESHOLD
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _load_model_artifacts() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = _load_model_artifacts()
THRESHOLD = _get_threshold()


@bentoml.service
class ModernBertClassifier:
    @bentoml.api(route="/classify")
    def classify(self, text: str = "") -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {
                "is_issue": False,
                "confidence": 0.0,
                "label": "no_issue",
            }

        encoded = TOKENIZER(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = MODEL(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        issue_idx = int(MODEL.config.label2id.get("issue", 1))
        no_issue_idx = int(MODEL.config.label2id.get("no_issue", 0))
        issue_score = float(probs[issue_idx].item())
        no_issue_score = float(probs[no_issue_idx].item())
        is_issue = issue_score >= THRESHOLD
        label = "issue" if is_issue else "no_issue"
        confidence = issue_score if is_issue else no_issue_score

        return {
            "is_issue": is_issue,
            "confidence": confidence,
            "label": label,
        }

    @bentoml.api(route="/health")
    def health(self, payload: Dict[str, Any]) -> Dict[str, str]:
        _ = payload
        return {"status": "ok"}
