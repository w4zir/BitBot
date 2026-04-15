from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AbstractSet

from dotenv import load_dotenv
from openai import APIError, OpenAI

# Write full JSON checkpoint every N successful batches (final write is always performed).
CHECKPOINT_EVERY_N_SUCCESSFUL_BATCHES = 5

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "issue_classification" / "generated_is_issue_dataset.json"

CATEGORIES = [
    "Greetings",
    "Goodbyes",
    "Negative reactions (no new info)",
    "Acknowledgments",
    "Gratitude",
    "Positive reactions",
    "Connection checks",
    "Pleasantries",
    "Small talk",
    "Meta-commentary",
    "Bot identity questions",
    "Fillers & expressions",
    "Testing",
    "Gibberish",
    "Thinking",
    "Customer withdraws request",
    "Indicating a question without stating it",
]

CATEGORY_TARGET_SET: frozenset[str] = frozenset(CATEGORIES)

FEW_SHOT_HEADING = "Few-shot non-issue examples"


@dataclass
class ParseResult:
    accepted: list[dict[str, Any]]
    dropped_counts: dict[str, int]


FEW_SHOT_NON_ISSUE = [
    {
        "user_message": "lol what a weird weather today",
        "is_issue": False,
        "non_issue_category": "Small talk",
        "notes": "Casual chat unrelated to support need",
    },
    {
        "user_message": "asldkjf 9912 zzz",
        "is_issue": False,
        "non_issue_category": "Gibberish",
        "notes": "Nonsensical text with no actionable intent",
    },
    {
        "user_message": "just checking if you're still there",
        "is_issue": False,
        "non_issue_category": "Connection checks",
        "notes": "Presence check only",
    },
    {
        "user_message": "hmmm let me think",
        "is_issue": False,
        "non_issue_category": "Thinking",
        "notes": "User pausing, no concrete request",
    },
]

FEW_SHOT_ISSUE = [
    {
        "user_message": "I can't log in after resetting my password.",
        "is_issue": True,
        "non_issue_category": None,
        "notes": "Account access issue after reset flow",
    },
    {
        "user_message": "My package says delivered, but I never got it.",
        "is_issue": True,
        "non_issue_category": None,
        "notes": "Delivery discrepancy requiring tracking/investigation",
    },
]


def load_env() -> None:
    candidates = [Path.cwd() / ".env", ROOT / ".env", Path(__file__).resolve().parent / ".env"]
    for env_file in candidates:
        if env_file.exists():
            load_dotenv(env_file, override=False)
            return


def resolve_provider(args_provider: str | None) -> str:
    provider = (args_provider or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if provider not in {"ollama", "cerebras"}:
        raise ValueError("provider must be one of: ollama, cerebras")
    return provider


def resolve_model(provider: str, args_model: str | None) -> str:
    if args_model:
        return args_model
    if provider == "cerebras":
        return os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
    return os.getenv("OLLAMA_MODEL", "llama3.2")


def resolve_base_url(provider: str) -> str:
    if provider == "cerebras":
        return os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"


def build_client(provider: str, base_url: str | None = None) -> tuple[OpenAI, str]:
    resolved_base_url = base_url or resolve_base_url(provider)
    if provider == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY is required for provider=cerebras")
        return OpenAI(base_url=resolved_base_url, api_key=api_key), resolved_base_url

    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    return OpenAI(base_url=resolved_base_url, api_key=api_key), resolved_base_url


def preflight_provider_connection(provider: str, base_url: str, timeout_seconds: float) -> str:
    if provider != "ollama":
        return base_url
    if timeout_seconds <= 0:
        return base_url
    base_no_v1 = base_url[:-3] if base_url.endswith("/v1") else base_url
    health_url = base_no_v1.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"Ollama preflight failed at {health_url} with status={response.status}."
                )
            return base_url
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fallback_base_no_v1 = base_no_v1.replace("://ollama:", "://localhost:", 1)
        fallback_health_url = fallback_base_no_v1.rstrip("/") + "/api/tags"
        if fallback_health_url != health_url:
            try:
                with urllib.request.urlopen(fallback_health_url, timeout=timeout_seconds) as response:
                    if response.status < 400:
                        fallback_base_url = fallback_base_no_v1.rstrip("/") + "/v1"
                        print(
                            "[warn] OLLAMA_BASE_URL host 'ollama' was unreachable; "
                            f"auto-falling back to {fallback_base_url}"
                        )
                        return fallback_base_url
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
        raise RuntimeError(
            "Cannot reach Ollama. Ensure Ollama is running and OLLAMA_BASE_URL is correct. "
            f"Tried: {health_url}. Error: {exc}"
        ) from exc


def normalize_message(text: str) -> str:
    lower = text.casefold().strip()
    lower = re.sub(r"\s+", " ", lower)
    lower = re.sub(r"[^\w\s]", "", lower)
    return lower.strip()


def strip_markdown_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_first_balanced_json_object(text: str) -> str:
    """Slice the first top-level `{...}` using brace depth, respecting JSON string rules."""
    start = text.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object")
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    raise ValueError("Unbalanced braces in model JSON")


def extract_json_object(raw: str) -> dict[str, Any]:
    text = strip_markdown_json_fence(raw)

    candidates: list[str] = []
    if text:
        candidates.append(text)
    try:
        snippet = extract_first_balanced_json_object(text)
        if snippet not in candidates:
            candidates.append(snippet)
    except ValueError:
        pass

    last_err: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise ValueError("Parsed JSON is not an object")


def desired_issue_count(total_needed: int, issue_percent: float) -> int:
    """Return target count of is_issue=true rows; clamped to [0, total_needed]."""
    if total_needed <= 0:
        return 0
    n = int(round(total_needed * issue_percent / 100.0))
    return max(0, min(total_needed, n))


def target_non_issue_counts(total_needed: int, issue_percent: float) -> dict[str, int]:
    false_target = total_needed - desired_issue_count(total_needed, issue_percent)
    if false_target <= 0:
        return {c: 0 for c in CATEGORIES}
    counts = {c: false_target // len(CATEGORIES) for c in CATEGORIES}
    remainder = false_target % len(CATEGORIES)
    for i in range(remainder):
        counts[CATEGORIES[i]] += 1
    return counts


def next_category_subset(
    category_counts: dict[str, int],
    category_target: dict[str, int],
    batch_size: int,
    rng: random.Random,
) -> list[str]:
    remaining = []
    for c in CATEGORIES:
        need = category_target[c] - category_counts.get(c, 0)
        if need > 0:
            remaining.extend([c] * need)
    if not remaining:
        remaining = CATEGORIES[:]
    rng.shuffle(remaining)
    return remaining[:batch_size]


def build_prompt(
    batch_size: int,
    next_id: int,
    is_issue_batch: bool,
    categories_for_batch: list[str],
) -> str:
    schema = (
        '{ "samples": ['
        '{"id": int, "user_message": str, "is_issue": bool, '
        '"non_issue_category": str|null, "notes": str}'
        "] }"
    )
    category_hint = ", ".join(categories_for_batch)
    non_issue_examples = json.dumps(FEW_SHOT_NON_ISSUE, ensure_ascii=False, indent=2)
    issue_examples = json.dumps(FEW_SHOT_ISSUE, ensure_ascii=False, indent=2)
    mode = "ISSUE messages (is_issue=true)" if is_issue_batch else "NON-ISSUE messages (is_issue=false)"
    extra_constraints = (
        "For issue rows: non_issue_category must be null.\n"
        "For non-issue rows: non_issue_category must be one of the 17 categories exactly.\n"
        "Use natural customer-chat style, short and varied phrasing.\n"
        "Avoid duplicates and near-duplicates in this batch."
    )

    if is_issue_batch:
        batch_instruction = (
            f"Generate {batch_size} unique ISSUE samples only. "
            "Each row must describe an actionable support problem."
        )
    else:
        batch_instruction = (
            f"Generate {batch_size} unique NON-ISSUE samples only. "
            f"Distribute across these target categories for this batch: {category_hint}."
        )

    prompt = f"""
You are generating synthetic training data for is_issue classification.

Output must be strict JSON only with this top-level schema:
{schema}

Current mode: {mode}
{batch_instruction}
Start IDs at {next_id} and increment by 1.

17 non-issue categories:
{", ".join(CATEGORIES)}

{FEW_SHOT_HEADING}:
{non_issue_examples}

Few-shot issue examples:
{issue_examples}

Rules:
{extra_constraints}

Return only JSON object with key "samples".
""".strip()
    heading_count = prompt.count(FEW_SHOT_HEADING)
    if heading_count != 1:
        raise RuntimeError(
            f"Prompt must contain {FEW_SHOT_HEADING!r} exactly once; found {heading_count} occurrence(s). "
            "Check for duplicated or missing few-shot sections."
        )
    return prompt


def parse_and_validate_samples(
    payload: dict[str, Any],
    is_issue_expected: bool,
    category_target_set: AbstractSet[str],
) -> ParseResult:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("JSON must contain a list under 'samples'")

    dropped_counts = {"is_issue_mismatch": 0, "invalid_non_issue_category": 0}
    accepted: list[dict[str, Any]] = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        user_message = str(row.get("user_message") or "").strip()
        notes = str(row.get("notes") or "").strip()
        is_issue = bool(row.get("is_issue"))
        category = row.get("non_issue_category")

        if not user_message:
            continue
        if is_issue != is_issue_expected:
            dropped_counts["is_issue_mismatch"] += 1
            continue

        if is_issue_expected:
            category = None
        else:
            category = str(category or "").strip()
            if category not in category_target_set:
                dropped_counts["invalid_non_issue_category"] += 1
                continue

        accepted.append(
            {
                "id": int(row.get("id") or 0),
                "user_message": user_message,
                "is_issue": is_issue,
                "non_issue_category": category,
                "notes": notes or ("Actionable support issue" if is_issue else "Non-issue interaction"),
            }
        )
    return ParseResult(accepted=accepted, dropped_counts=dropped_counts)


def dedupe_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for row in samples:
        key = normalize_message(row["user_message"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def dedupe_new_against_seen(
    rows: list[dict[str, Any]], seen: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Keep rows whose normalized message is not already in seen; update seen in place."""
    new_unique: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        key = normalize_message(row["user_message"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        new_unique.append(row)
    return new_unique, removed


def seed_state_from_samples(samples: list[dict[str, Any]]) -> tuple[set[str], int, int, dict[str, int]]:
    """One-pass seed: normalized keys, class counts, and non-issue category counts."""
    seen: set[str] = set()
    for row in samples:
        seen.add(normalize_message(row["user_message"]))
    true_count = sum(1 for s in samples if s.get("is_issue") is True)
    false_count = sum(1 for s in samples if s.get("is_issue") is False)
    category_counts = {c: 0 for c in CATEGORIES}
    for s in samples:
        if s.get("is_issue") is False and s.get("non_issue_category") in category_counts:
            category_counts[s["non_issue_category"]] += 1
    return seen, true_count, false_count, category_counts


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        temp_name = tmp.name
    Path(temp_name).replace(path)


def load_existing(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        samples = raw.get("samples")
        if isinstance(samples, list):
            return samples, dict(raw.get("metadata") or {})
    raise ValueError(f"Existing output at {path} is not in expected format")


def resolve_output_path(output_arg: str) -> tuple[Path, bool]:
    requested = Path(output_arg).resolve()
    if requested.exists() and requested.is_dir():
        return requested / DEFAULT_OUTPUT.name, True
    if not requested.suffix:
        requested.mkdir(parents=True, exist_ok=True)
        return requested / DEFAULT_OUTPUT.name, True
    return requested, False


def final_reindex(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(samples, start=1):
        row_copy = dict(row)
        row_copy["id"] = idx
        out.append(row_copy)
    return out


def split_counts(samples: list[dict[str, Any]]) -> tuple[int, int]:
    true_count = sum(1 for s in samples if s.get("is_issue") is True)
    false_count = sum(1 for s in samples if s.get("is_issue") is False)
    return true_count, false_count


def maybe_trim_to_balance(
    samples: list[dict[str, Any]],
    total_needed: int,
    issue_percent: float,
) -> list[dict[str, Any]]:
    desired_true = desired_issue_count(total_needed, issue_percent)
    desired_false = total_needed - desired_true
    trues = [s for s in samples if s["is_issue"] is True]
    falses = [s for s in samples if s["is_issue"] is False]
    trues = trues[:desired_true]
    falses = falses[:desired_false]
    combined = trues + falses
    return final_reindex(combined[:total_needed])


CHAT_COMPLETION_SYSTEM = (
    "Return one JSON object only. No markdown fences, no commentary before or after. "
    "Use double quotes for all keys and string values. Escape any \" inside string "
    'values as \\". Do not use single-quote strings. No trailing commas. '
    "Use true/false/null, not True/False/None."
)


def build_chat_completion_kwargs(
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    provider: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": request_timeout,
        "messages": [
            {"role": "system", "content": CHAT_COMPLETION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    if provider == "ollama":
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"format": "json"}
    elif provider == "cerebras":
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def provider_call(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    provider: str,
) -> dict[str, Any]:
    kwargs = build_chat_completion_kwargs(
        model=model,
        prompt=prompt,
        temperature=temperature,
        request_timeout=request_timeout,
        provider=provider,
    )

    try:
        response = client.chat.completions.create(**kwargs)
    except APIError as orig_exc:
        print(
            f"[warn] chat.completions.create failed with structured output params: {orig_exc!r}"
        )
        if provider == "ollama":
            kwargs.pop("response_format", None)
            kwargs.pop("extra_body", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as new_exc:
                raise new_exc from orig_exc
        elif provider == "cerebras":
            kwargs.pop("response_format", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as new_exc:
                raise new_exc from orig_exc
        else:
            raise
    content = response.choices[0].message.content or "{}"
    return extract_json_object(content)


def should_stop(
    true_count: int, false_count: int, desired_true: int, desired_false: int
) -> bool:
    return true_count >= desired_true and false_count >= desired_false


def update_state(
    batch_samples: list[dict[str, Any]],
    seen_keys: set[str],
    all_samples: list[dict[str, Any]],
    true_count: int,
    false_count: int,
    category_counts: dict[str, int],
) -> tuple[int, int, int]:
    """Dedupe batch rows against ``seen_keys``, append to ``all_samples``, update counters.

    Returns ``(true_count, false_count, duplicates_removed_this_batch)``.
    """
    new_rows, removed = dedupe_new_against_seen(batch_samples, seen_keys)
    for row in new_rows:
        all_samples.append(row)
        if row.get("is_issue") is True:
            true_count += 1
        else:
            false_count += 1
            cat = row.get("non_issue_category")
            if cat in category_counts:
                category_counts[cat] += 1
    return true_count, false_count, removed


def run_batch(
    client: OpenAI | None,
    model: str,
    prompt: str,
    args: argparse.Namespace,
    is_issue_batch: bool,
    categories_for_batch: list[str],
    all_samples: list[dict[str, Any]],
    provider: str,
    batch_idx: int,
) -> tuple[list[dict[str, Any]], int]:
    """Run one batch with retries. Returns ``(accepted rows, retries_increment)`` matching the
    original loop: successful completion does not increment ``retries_total`` for the winning try.
    """
    requested_batch_size = len(categories_for_batch)
    batch_samples: list[dict[str, Any]] = []
    retries_increment = 0
    for attempt in range(1, args.max_retries + 1):
        try:
            if args.dry_run:
                batch_samples = []
                for idx in range(requested_batch_size):
                    msg = (
                        f"test issue sample {len(all_samples) + idx + 1}"
                        if is_issue_batch
                        else f"test non issue sample {len(all_samples) + idx + 1}"
                    )
                    cat = (
                        None
                        if is_issue_batch
                        else categories_for_batch[idx % len(categories_for_batch)]
                    )
                    batch_samples.append(
                        {
                            "id": len(all_samples) + idx + 1,
                            "user_message": msg,
                            "is_issue": is_issue_batch,
                            "non_issue_category": cat,
                            "notes": "dry run sample",
                        }
                    )
            else:
                assert client is not None
                payload = provider_call(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    request_timeout=args.request_timeout,
                    provider=provider,
                )
                parsed = parse_and_validate_samples(
                    payload=payload,
                    is_issue_expected=is_issue_batch,
                    category_target_set=CATEGORY_TARGET_SET,
                )
                batch_samples = parsed.accepted
                total_dropped = sum(parsed.dropped_counts.values())
                if total_dropped > 0:
                    print(
                        f"[warn] batch {batch_idx + 1}: dropped {parsed.dropped_counts} "
                        "during parse/validate"
                    )
            if batch_samples:
                return batch_samples, retries_increment
        except Exception as exc:
            print(f"[warn] batch {batch_idx + 1} attempt {attempt} failed: {exc}")
        retries_increment += 1
        time.sleep(args.sleep_seconds)
    return batch_samples, retries_increment


def print_dry_run_sample_chat_request(
    args: argparse.Namespace,
    provider: str,
    model: str,
    rng: random.Random,
    category_target: dict[str, int],
    true_count: int,
    false_count: int,
    category_counts: dict[str, int],
    next_id: int,
) -> None:
    """Print kwargs for one representative batch (mirrors first live batch; no API call)."""
    desired_true = desired_issue_count(args.total_needed, args.issue_percent)
    desired_false = args.total_needed - desired_true
    remaining_true = max(desired_true - true_count, 0)
    remaining_false = max(desired_false - false_count, 0)
    is_issue_batch = remaining_true >= remaining_false
    remaining_for_type = remaining_true if is_issue_batch else remaining_false

    if remaining_for_type > 0:
        requested_batch_size = min(args.batch_size, remaining_for_type)
    else:
        requested_batch_size = min(args.batch_size, max(1, args.total_needed))
        is_issue_batch = True

    categories_for_batch = next_category_subset(
        category_counts, category_target, requested_batch_size, rng
    )
    prompt = build_prompt(
        batch_size=requested_batch_size,
        next_id=next_id,
        is_issue_batch=is_issue_batch,
        categories_for_batch=categories_for_batch,
    )
    kwargs = build_chat_completion_kwargs(
        model=model,
        prompt=prompt,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        provider=provider,
    )
    endpoint = resolve_base_url(provider).rstrip("/") + "/chat/completions"
    print("[dry-run] Sample chat.completions request (first batch; same body as live run)")
    print(f"[dry-run] POST {endpoint}")
    print(json.dumps(kwargs, ensure_ascii=False, indent=2))


def generate_dataset(args: argparse.Namespace) -> None:
    load_env()
    provider = resolve_provider(args.provider)
    model = resolve_model(provider, args.model)
    output_path, output_was_directory = resolve_output_path(args.output)
    if output_was_directory:
        print(f"[info] output resolved to file: {output_path}")
    rng = random.Random(args.seed)

    existing_samples, existing_metadata = load_existing(output_path) if args.resume else ([], {})
    existing_samples, removed_existing_dupes = dedupe_samples(existing_samples)
    # Reindex only once after final trim (checkpoints may have non-contiguous ids).

    client: OpenAI | None = None
    provider_base_url = ""
    if not args.dry_run:
        provider_base_url = resolve_base_url(provider)
        provider_base_url = preflight_provider_connection(provider, provider_base_url, args.preflight_timeout)
        client, provider_base_url = build_client(provider, base_url=provider_base_url)
    print(
        f"[info] generation: provider={provider}, model={model}"
        + (" (dry-run, no LLM)" if args.dry_run else "")
    )
    category_target = target_non_issue_counts(args.total_needed, args.issue_percent)

    all_samples = list(existing_samples)
    seen_keys, true_count, false_count, category_counts = seed_state_from_samples(all_samples)
    if args.dry_run:
        print_dry_run_sample_chat_request(
            args=args,
            provider=provider,
            model=model,
            rng=rng,
            category_target=category_target,
            true_count=true_count,
            false_count=false_count,
            category_counts=category_counts,
            next_id=len(all_samples) + 1,
        )
    total_duplicates_removed = removed_existing_dupes
    failed_batches = 0
    retries_total = 0
    batch_idx = 0
    successful_batches = 0

    def progress_metadata() -> dict[str, Any]:
        return {
            "source_research": "https://fin.ai/research/david-vs-goliath-are-small-llms-any-good/",
            "description": (
                "Synthetic user messages for is_issue binary classification. "
                "Batches prioritize the class with greater remaining need (issue vs non-issue)."
            ),
            "generator": {
                "provider": provider,
                "model": model,
                "temperature": args.temperature,
                "batch_size": args.batch_size,
                "max_retries": args.max_retries,
                "request_timeout": args.request_timeout,
                "issue_percent": args.issue_percent,
            },
            "total_samples": len(all_samples),
            "is_issue_true": true_count,
            "is_issue_false": false_count,
            "non_issue_categories_covered": CATEGORIES,
            "deduplicates_removed": total_duplicates_removed,
            "failed_batches": failed_batches,
            "retries_total": retries_total,
            "resumed_from_existing": args.resume,
            "checkpoint_every_n_batches": CHECKPOINT_EVERY_N_SUCCESSFUL_BATCHES,
        }

    if not args.dry_run:
        assert client is not None, "Client not initialized"

    while len(all_samples) < args.total_needed:
        batch_idx += 1

        desired_true = desired_issue_count(args.total_needed, args.issue_percent)
        desired_false = args.total_needed - desired_true
        if should_stop(true_count, false_count, desired_true, desired_false):
            break

        remaining_true = max(desired_true - true_count, 0)
        remaining_false = max(desired_false - false_count, 0)
        is_issue_batch = remaining_true >= remaining_false
        remaining_for_type = remaining_true if is_issue_batch else remaining_false
        if remaining_for_type == 0:
            break
        requested_batch_size = min(args.batch_size, remaining_for_type)

        categories_for_batch = next_category_subset(category_counts, category_target, requested_batch_size, rng)

        prompt = build_prompt(
            batch_size=requested_batch_size,
            next_id=len(all_samples) + 1,
            is_issue_batch=is_issue_batch,
            categories_for_batch=categories_for_batch,
        )

        batch_samples, retries_inc = run_batch(
            client=client,
            model=model,
            prompt=prompt,
            args=args,
            is_issue_batch=is_issue_batch,
            categories_for_batch=categories_for_batch,
            all_samples=all_samples,
            provider=provider,
            batch_idx=batch_idx - 1,
        )
        retries_total += retries_inc

        if not batch_samples:
            failed_batches += 1
            print(f"[warn] batch {batch_idx} failed after retries; continuing")
            continue

        n_before = len(all_samples)
        true_count, false_count, removed = update_state(
            batch_samples,
            seen_keys,
            all_samples,
            true_count,
            false_count,
            category_counts,
        )
        total_duplicates_removed += removed
        new_rows_count = len(all_samples) - n_before

        successful_batches += 1
        wrote_checkpoint = successful_batches % CHECKPOINT_EVERY_N_SUCCESSFUL_BATCHES == 0
        if wrote_checkpoint:
            atomic_write(output_path, {"metadata": progress_metadata(), "samples": all_samples})
        print(
            f"[info] batch {batch_idx}: +{new_rows_count} unique rows "
            f"(total={len(all_samples)}, true={true_count}, false={false_count}, "
            f"deduped_total={total_duplicates_removed}"
            f"{'; checkpoint saved' if wrote_checkpoint else ''})"
        )
        time.sleep(args.sleep_seconds)

    all_samples = maybe_trim_to_balance(all_samples, args.total_needed, args.issue_percent)
    all_samples = final_reindex(all_samples)
    true_count, false_count = split_counts(all_samples)

    per_category = {c: 0 for c in CATEGORIES}
    for row in all_samples:
        if row.get("is_issue") is False and row.get("non_issue_category") in per_category:
            per_category[row["non_issue_category"]] += 1
    missing_categories = [c for c, n in per_category.items() if n == 0]

    final_metadata = {
        "source_research": "https://fin.ai/research/david-vs-goliath-are-small-llms-any-good/",
        "description": (
            "Synthetic user messages for is_issue binary classification using "
            "17 non-issue categories and actionable issue samples."
        ),
        "generator": {
            "provider": provider,
            "model": model,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "max_retries": args.max_retries,
            "seed": args.seed,
            "request_timeout": args.request_timeout,
            "issue_percent": args.issue_percent,
            "checkpoint_every_n_batches": CHECKPOINT_EVERY_N_SUCCESSFUL_BATCHES,
        },
        "total_samples": len(all_samples),
        "is_issue_true": true_count,
        "is_issue_false": false_count,
        "non_issue_categories_covered": CATEGORIES,
        "per_non_issue_category_counts": per_category,
        "missing_non_issue_categories": missing_categories,
        "deduplicates_removed": total_duplicates_removed,
        "failed_batches": failed_batches,
        "retries_total": retries_total,
        "resumed_from_existing": args.resume,
        "previous_metadata": existing_metadata,
    }
    atomic_write(output_path, {"metadata": final_metadata, "samples": all_samples})

    print(f"[done] wrote dataset: {output_path}")
    print(f"[done] totals: total={len(all_samples)}, issue={true_count}, non_issue={false_count}")
    print(f"[done] missing categories: {missing_categories if missing_categories else 'none'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate is_issue dataset with configurable class balance (Ollama/Cerebras).",
        allow_abbrev=True,
    )
    parser.add_argument("--total-needed", type=int, default=10000, help="Total number of samples to generate.")
    parser.add_argument(
        "--issue-percent",
        type=float,
        default=50.0,
        metavar="PCT",
        help="Target percentage of samples with is_issue=true (0-100). Default: 50.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size per generation request.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON file path.")
    parser.add_argument(
        "--provider",
        "--provide",
        dest="provider",
        type=str,
        default=None,
        help="Provider override: ollama or cerebras.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model override.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per batch.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds for provider calls.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=3.0,
        help="Connectivity check timeout in seconds before generation.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between retries/batches.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for category ordering.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic synthetic rows without API calls; print one sample chat request at start.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.total_needed <= 0:
        raise ValueError("--total-needed must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not 0.0 <= args.issue_percent <= 100.0:
        raise ValueError("--issue-percent must be between 0 and 100 inclusive")
    generate_dataset(args)


if __name__ == "__main__":
    main()
