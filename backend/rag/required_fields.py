from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


def _default_config_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "config" / "issue_required_fields.json"


def config_path() -> Path:
    raw = os.getenv("ISSUE_REQUIRED_FIELDS_PATH", "").strip()
    if raw:
        return Path(raw)
    return _default_config_path()


@lru_cache(maxsize=1)
def load_issue_categories() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {"issue_categories": {}}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"issue_categories": {}}


def normalize_category_key(category: str) -> str:
    return (category or "").strip().lower()


def get_category_spec(category: str) -> Optional[dict[str, Any]]:
    """Return issue category block (e.g. order, payment) or None."""
    key = normalize_category_key(category)
    cats = load_issue_categories().get("issue_categories") or {}
    if not isinstance(cats, dict):
        return None
    # Exact lower key
    if key in cats:
        spec = cats.get(key)
        return spec if isinstance(spec, dict) else None
    # Case-insensitive match
    for k, v in cats.items():
        if str(k).lower() == key and isinstance(v, dict):
            return v
    return None


def build_missing_prompts(spec: dict[str, Any], missing_names: list[str]) -> str:
    """Compose user-facing prompts for missing required fields."""
    required = spec.get("required_fields") or []
    if not isinstance(required, list):
        return ""
    name_to_prompt: dict[str, str] = {}
    for item in required:
        if not isinstance(item, dict):
            continue
        n = item.get("name")
        pr = item.get("prompt")
        if isinstance(n, str) and isinstance(pr, str):
            name_to_prompt[n.lower()] = pr
    lines: list[str] = []
    for m in missing_names:
        p = name_to_prompt.get(m.lower())
        if p:
            lines.append(p)
    if not lines and missing_names:
        return "Please provide the missing details so we can help."
    return "\n".join(lines)
