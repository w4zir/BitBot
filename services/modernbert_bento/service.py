from __future__ import annotations

import os
from typing import Any, Dict

import bentoml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODERNBERT_MODEL_DIR", "/models/modernbert_winner")


def _load_model_artifacts() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = _load_model_artifacts()


def _resolve_label(idx: int) -> str:
    id2label = getattr(MODEL.config, "id2label", {}) or {}
    raw = id2label.get(idx, str(idx))
    return str(raw)


@bentoml.service
class ModernBertClassifier:
    @bentoml.api(route="/classify")
    def classify(self, text: str = "") -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {
                "category": "unknown",
                "confidence": 0.0,
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

        predicted_idx = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_idx].item())
        category = _resolve_label(predicted_idx)

        return {
            "category": category,
            "confidence": confidence,
        }

    @bentoml.api(route="/health")
    def health(self, payload: Dict[str, Any]) -> Dict[str, str]:
        _ = payload
        return {"status": "ok"}
