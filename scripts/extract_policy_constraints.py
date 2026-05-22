#!/usr/bin/env python3
"""
Extract policy constraints from retrieved policy documents and write validated
per-intent YAML artifacts under `backend/policy_constraints`.

Usage:
  python scripts/extract_policy_constraints.py
  python scripts/extract_policy_constraints.py --dry-run
  python scripts/extract_policy_constraints.py --intent order_cancel
  python scripts/extract_policy_constraints.py --log-level DEBUG

CLI options:
  --dry-run           Do not write output YAML files.
  --intent <intent>   Extract constraints for one intent only.
  --log-level <lvl>   Python logging level (default: INFO).

Environment variables are loaded from the repository ``.env`` then optional
``.env.local`` (host scripts only; see ``backend/repo_dotenv.py``). Values
already set in the process environment are preserved for ``.env``; ``.env.local``
overrides those defaults.

LLM provider and model for extraction (after the above load order):

- ``POLICY_CONSTRAINTS_MODEL_PROVIDER`` (optional), else ``LLM_PROVIDER``, else
  ``ollama``.
- ``POLICY_CONSTRAINTS_MODEL`` (optional); if unset, ``VLLM_MODEL`` when
  provider is ``vllm``, ``CEREBRAS_MODEL`` when ``cerebras``, else ``OLLAMA_MODEL``,
  else ``llama3.2``.

If the configured LLM endpoint is unreachable (for example Ollama not running),
the script still exits successfully and writes default constraints, with
``metadata.llm_error`` recording the failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.repo_dotenv import load_repo_dotenv

load_repo_dotenv(REPO_ROOT)

from backend.agent.policy_constraints import PolicyConstraints, policy_constraints_dir
from backend.agent.procedures import load_blueprints
from backend.llm.providers import chat_completion, extract_json_object
from backend.rag.policy_retriever import ping_elasticsearch, search_policy_docs

logger = logging.getLogger("extract_policy_constraints")


def _env_nonempty(key: str) -> str | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _policy_extractor_llm() -> tuple[str, str]:
    """Provider and model from env (.env then .env.local via ``load_repo_dotenv``)."""
    provider = (
        (_env_nonempty("POLICY_CONSTRAINTS_MODEL_PROVIDER") or "").lower()
        or (_env_nonempty("LLM_PROVIDER") or "").lower()
        or "ollama"
    )
    model = _env_nonempty("POLICY_CONSTRAINTS_MODEL")
    if model:
        return provider, model
    if provider == "vllm":
        return provider, _env_nonempty("VLLM_MODEL") or "llama3.2"
    if provider == "cerebras":
        return provider, _env_nonempty("CEREBRAS_MODEL") or "llama3.2"
    return provider, _env_nonempty("OLLAMA_MODEL") or "llama3.2"


def _default_constraints(category: str, intent: str, docs: list[dict[str, Any]], query: str) -> PolicyConstraints:
    return PolicyConstraints(
        category=category,
        intent=intent,
        policy_doc_names=[str(d.get("title") or d.get("id") or "").strip() for d in docs if d],
        source_query=query,
        auto_resolvable=True,
        requires_evidence=False,
        default_ineligible_reason="The request does not satisfy policy constraints.",
        time_limits={},
        eligibility_rules=[],
        required_conditions=[],
        escalation_conditions=[],
        metadata={"extractor": "offline_policy_pipeline"},
    )


def _llm_extract(category: str, intent: str, docs: list[dict[str, Any]], query: str) -> dict[str, Any]:
    model_provider, model = _policy_extractor_llm()
    policy_text = "\n\n---\n\n".join(str(doc.get("content") or "") for doc in docs[:3])
    system = (
        "You extract deterministic policy constraints for a support automation system.\n"
        "Return strict JSON only.\n"
        "Schema keys:\n"
        "schema_version, category, intent, policy_doc_names, source_query, auto_resolvable, "
        "requires_evidence, default_ineligible_reason, time_limits, eligibility_rules, required_conditions, "
        "escalation_conditions, response_guidance, metadata.\n"
        "Rules in eligibility_rules/required_conditions/escalation_conditions must use keys:\n"
        "id, description, field, op, value(optional), value_from(optional), failure_reason, applies_to.\n"
        "Use only these op strings (lowercase): eq, neq, gt, gte, lt, lte, in, nin, exists, contains.\n"
        "time_limits must be an object (use {} if none). response_guidance must be a list of strings.\n"
    )
    user = (
        f"Category: {category}\n"
        f"Intent: {intent}\n"
        f"Search query: {query}\n"
        f"Docs JSON: {json.dumps(docs, ensure_ascii=False)}\n"
        f"Policy text excerpt:\n{policy_text[:8000]}"
    )
    raw = chat_completion(
        provider=model_provider,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return extract_json_object(raw)


def _write_yaml(path: Path, model: PolicyConstraints, dry_run: bool) -> None:
    payload = model.model_dump()
    if dry_run:
        logger.info("dry-run write: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract policy constraints from Elasticsearch docs into YAML artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    parser.add_argument("--intent", default="", help="Optional single intent to extract.")
    parser.add_argument("--log-level", default="INFO", help="Python log level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    es_ok, es_reason = ping_elasticsearch()
    if not es_ok:
        msg = f"Elasticsearch is not reachable: {es_reason}"
        logger.error(msg)
        print(f"Error: {msg}", file=sys.stderr)
        return 1

    base_dir = policy_constraints_dir()
    blueprints = sorted(load_blueprints().values(), key=lambda x: (x.category, x.intent))
    for bp in blueprints:
        if args.intent and bp.intent != args.intent:
            continue
        category = bp.category.strip().lower()
        intent = bp.intent.strip().lower()
        query = f"{category} {intent} policy"
        docs = search_policy_docs(query)
        logger.info("extract start category=%s intent=%s docs=%s", category, intent, len(docs))
        if not docs:
            warn_msg = (
                f"No matching policy documents in Elasticsearch for category={category!r} "
                f"intent={intent!r} query={query!r}"
            )
            logger.warning(warn_msg)
            print(f"Warning: {warn_msg}", file=sys.stderr)

        llm_error_msg: str | None = None
        parsed: dict[str, Any] = {}
        if docs:
            try:
                parsed = _llm_extract(category, intent, docs, query)
            except httpx.HTTPError as exc:
                llm_error_msg = str(exc)
                logger.warning(
                    "llm_extract_failed category=%s intent=%s error=%s",
                    category,
                    intent,
                    exc,
                )
                print(
                    f"Warning: LLM request failed for intent={intent!r}: {exc}. Using default constraints.",
                    file=sys.stderr,
                )

        model: PolicyConstraints
        try:
            merged = {**_default_constraints(category, intent, docs, query).model_dump(), **parsed}
            merged["category"] = category
            merged["intent"] = intent
            merged["source_query"] = query
            if not merged.get("policy_doc_names"):
                merged["policy_doc_names"] = [str(d.get("title") or d.get("id") or "").strip() for d in docs if d]
            model = PolicyConstraints.model_validate(merged)
        except Exception as exc:  # noqa: BLE001
            logger.exception("validation_failed category=%s intent=%s error=%s", category, intent, exc)
            model = _default_constraints(category, intent, docs, query)
            model.auto_resolvable = False
            model.metadata = {**model.metadata, "validation_error": str(exc)}

        if llm_error_msg is not None:
            model.auto_resolvable = False
            model.metadata = {**model.metadata, "llm_error": llm_error_msg}

        output = base_dir / category / f"{intent}.yaml"
        _write_yaml(output, model, args.dry_run)
        logger.info(
            "extract done category=%s intent=%s output=%s rules=%s",
            category,
            intent,
            output,
            len(model.eligibility_rules),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
