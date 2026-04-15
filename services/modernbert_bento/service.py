from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import bentoml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

MODEL_DIR = os.getenv("MODERNBERT_MODEL_DIR", "/models/modernbert_winner")


def _resolve_model_dir() -> str:
    primary = Path(MODEL_DIR)
    fallback_dirs = sorted(
        Path("/training/models").glob("*/winner"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    candidates = [primary, *fallback_dirs, Path("/models/modernbert_winner")]

    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors").exists():
            return str(candidate)
    raise FileNotFoundError(
        "No valid ModernBERT model directory found. "
        f"Checked primary path '{MODEL_DIR}' and fallback paths under /training/models."
    )


def _load_tokenizer(model_dir: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        tokenizer_json = Path(model_dir) / "tokenizer.json"
        tokenizer_config = Path(model_dir) / "tokenizer_config.json"
        if not tokenizer_json.exists():
            raise

        tokenizer_kwargs: Dict[str, Any] = {}
        if tokenizer_config.exists():
            with tokenizer_config.open("r", encoding="utf-8") as fp:
                cfg = json.load(fp)
            for key in ("unk_token", "pad_token", "cls_token", "sep_token", "mask_token"):
                value = cfg.get(key)
                if value:
                    tokenizer_kwargs[key] = value

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            **tokenizer_kwargs,
        )
        return tokenizer


def _load_model_artifacts() -> tuple[Any, Any]:
    model_dir = _resolve_model_dir()
    tokenizer = _load_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
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
