from __future__ import annotations

"""Generate JSONL of opening customer messages with gold category/intent labels.

Uses simulator seeds, a suite YAML, and the persona LLM to synthesize realistic
first-turn user queries. Each output row has ``text``, ``category``, ``intent``,
``seed_id``, and ``persona``.

Writes timestamped files to ``data/raw/simulated/category_intent_YYYYMMDD_HHMMSS.jsonl``
(by default under the repository root).

Prerequisites (from repo ``.env`` or environment):

- ``OLLAMA_BASE_URL`` — Ollama API base (when ``SIMULATOR_USER_LLM_PROVIDER=ollama``)
- ``SIMULATOR_USER_LLM_PROVIDER`` / ``SIMULATOR_USER_LLM_MODEL`` — persona LLM backend
- ``SIMULATOR_USER_LLM_TIMEOUT_SECONDS``, ``SIMULATOR_USER_LLM_TEMPERATURE``, etc. — optional tuning

Before generating, the script preflights the persona LLM (``GET /v1/models`` for vLLM/Cerebras,
``GET /api/tags`` for Ollama) and fails fast if the server is unreachable or the model is missing.

CLI options:

- ``--max-limit N`` — number of examples to generate (required)
- ``--output-dir PATH`` — output directory (default: ``data/raw/simulated``)
- ``--suite PATH`` — suite YAML under ``testing/simulator`` (default: ``suites/regression.yaml``)
- ``--seed SEED_ID`` — restrict to one seed
- ``--persona ID [ID ...]`` — restrict persona selection
- ``--randomize`` — pick a random scenario each attempt instead of cycling
- ``--ollama-url URL`` — override ``OLLAMA_BASE_URL`` for this run
- ``--temperature T`` — LLM sampling temperature (0.0–2.0) for this run
- ``--save-every N`` — checkpoint append every N rows (resume-friendly)
- ``--output-file PATH`` — write/append to a specific JSONL file; existing rows count
  toward ``--max-limit`` so you can resume an interrupted run

Usage (from repository root)::

    python testing/scripts/generate_category_intent_dataset.py --max-limit 50

    python testing/scripts/generate_category_intent_dataset.py \\
        --max-limit 100 \\
        --randomize \\
        --save-every 10

    python testing/scripts/generate_category_intent_dataset.py \\
        --max-limit 20 \\
        --seed order_status_cooperative \\
        --persona polite_first_timer impatient_escalator

    python testing/scripts/generate_category_intent_dataset.py \\
        --max-limit 30 \\
        --suite suites/regression.yaml \\
        --output-dir data/raw/simulated \\
        --ollama-url http://127.0.0.1:11434 \\
        --temperature 0.7

    python testing/scripts/generate_category_intent_dataset.py \\
        --max-limit 500 \\
        --output-file data/raw/simulated/category_intent_20260531_075702.jsonl \\
        --save-every 10

    python testing/scripts/generate_category_intent_dataset.py --help
"""

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.llm.vllm_routing import resolve_vllm_target
from backend.repo_dotenv import load_repo_dotenv

from testing.simulator.config import SeedConfig, SuiteConfig
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.persona import PersonaEngine, PersonaGenerationError
from testing.simulator.runner import (
    _load_all_seeds,
    _load_personas,
    _load_suite,
    _persona_candidates,
    _pick_persona_for_scenario,
    _resolve_path,
    _select_scenarios,
)


DEFAULT_SUITE_REL = "suites/regression.yaml"
DEFAULT_OUTPUT_SUBDIR = Path("data") / "raw" / "simulated"
OUTPUT_PREFIX = "category_intent_"
_SAMPLE_SEPARATOR = "=" * 72


def _find_simulator_root(suite_path: Path) -> Path:
    """Directory that contains seeds/ and personas/ (parent of suites/*.yaml)."""
    cur = suite_path.resolve().parent
    while True:
        if (cur / "seeds").is_dir() and (cur / "personas").is_dir():
            return cur
        if cur == cur.parent:
            raise ValueError(
                f"Could not locate simulator root (seeds/ + personas/) near suite: {suite_path}"
            )
        cur = cur.parent


def _four_digit_suffix(seed_id: str, sequence: int) -> str:
    digest = hashlib.sha256(f"{seed_id}:{sequence}".encode()).digest()
    n = int.from_bytes(digest[:2], "big") % 10000
    return f"{n:04d}"


def _first_or(values: list[str], default: str) -> str:
    return values[0] if values else default


def build_fake_scenario_instance(seed: SeedConfig, sequence: int) -> ScenarioInstance:
    """Synthetic scenario data for dataset generation (no database)."""
    db = seed.db_filter
    suffix = _four_digit_suffix(seed.seed_id, sequence)

    if db.entity_type == "order":
        status = _first_or(db.order_status, "processing")
        entity: dict[str, Any] = {
            "entity_type": "order",
            "order_id": f"ORD-{suffix}",
            "user_id": f"USR-{suffix}",
            "order_date": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "total_amount": 99.99,
            "shipping_address_line": "123 Fake St",
            "shipping_city": "Testville",
            "shipping_postal_code": "12345",
            "shipping_country": "US",
        }
    elif db.entity_type == "user":
        status = _first_or(db.user_status, "active")
        email = f"customer{suffix}@example.test"
        entity = {
            "entity_type": "user",
            "user_id": f"USR-{suffix}",
            "email": email,
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    elif db.entity_type == "subscription":
        sub_status = _first_or(db.subscription_status, "active")
        plan = _first_or(db.subscription_plan, "basic")
        account_email = f"subscriber{suffix}@example.test"
        entity = {
            "entity_type": "subscription",
            "account_email": account_email,
            "plan": plan,
            "next_renewal_at": datetime.now(timezone.utc).isoformat(),
            "last_charge_at": datetime.now(timezone.utc).isoformat(),
            "subscription_status": sub_status,
        }
    else:
        raise ValueError(f"Unsupported entity_type for fake scenario: {db.entity_type!r}")

    return ScenarioInstance(
        seed_id=seed.seed_id,
        category=seed.category,
        intent=seed.intent,
        description=seed.description,
        expected_outcome=seed.expected_outcome,
        expected_procedure_id=seed.expected_procedure_id,
        persona_id="",
        cooperation_level="cooperative",
        entity=entity,
        seed_snapshot=seed.model_dump(mode="json"),
        persona_snapshot={},
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _ollama_base_url(value: str) -> str:
    s = value.strip()
    if not s:
        raise argparse.ArgumentTypeError("must be a non-empty base URL")
    return s.rstrip("/")


def _llm_temperature(value: str) -> float:
    """Ollama sampling temperature; bounds match SuiteDefaults.user_llm_temperature."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0.0 or parsed > 2.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 2.0 (inclusive)")
    return parsed


def _resolve_output_path(raw: Path) -> Path:
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (REPO_ROOT / expanded).resolve()


def _count_jsonl_rows(output_path: Path) -> int:
    """Count non-empty, valid JSON lines in an existing JSONL file."""
    count = 0
    with output_path.open(encoding="utf-8") as in_f:
        for line_no, line in enumerate(in_f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {output_path}: {exc.msg}"
                ) from exc
            count += 1
    return count


def _append_rows_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        os.fsync(out_f.fileno())


def _apply_user_llm_env_overrides(suite: SuiteConfig) -> None:
    simulator_user_llm_provider = os.getenv("SIMULATOR_USER_LLM_PROVIDER", "").strip().lower()
    if simulator_user_llm_provider:
        suite.defaults.user_llm_provider = simulator_user_llm_provider  # type: ignore[assignment]
    simulator_user_llm_model = os.getenv("SIMULATOR_USER_LLM_MODEL", "").strip()
    if simulator_user_llm_model:
        suite.defaults.user_llm_model = simulator_user_llm_model
    simulator_user_llm_timeout = os.getenv("SIMULATOR_USER_LLM_TIMEOUT_SECONDS", "").strip()
    if simulator_user_llm_timeout:
        suite.defaults.user_llm_timeout_seconds = float(simulator_user_llm_timeout)
    simulator_user_llm_temperature = os.getenv("SIMULATOR_USER_LLM_TEMPERATURE", "").strip()
    if simulator_user_llm_temperature:
        suite.defaults.user_llm_temperature = float(simulator_user_llm_temperature)
    simulator_user_llm_top_p = os.getenv("SIMULATOR_USER_LLM_TOP_P", "").strip()
    if simulator_user_llm_top_p:
        suite.defaults.user_llm_top_p = float(simulator_user_llm_top_p)
    simulator_user_llm_repeat_penalty = os.getenv("SIMULATOR_USER_LLM_REPEAT_PENALTY", "").strip()
    if simulator_user_llm_repeat_penalty:
        suite.defaults.user_llm_repeat_penalty = float(simulator_user_llm_repeat_penalty)


def _normalize_model_id(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "")


def _model_id_listed(requested: str, available: set[str]) -> bool:
    if not requested:
        return bool(available)
    norm_req = _normalize_model_id(requested)
    for name in available:
        if not name:
            continue
        norm_name = _normalize_model_id(name)
        if norm_req == norm_name or norm_name.startswith(f"{norm_req}:"):
            return True
    return False


def _preflight_persona_llm(
    *,
    provider: str,
    model: str,
    timeout_seconds: float = 15.0,
) -> None:
    """Verify the persona LLM server is reachable and exposes the configured model."""
    p = (provider or "").strip().lower()
    m = (model or "").strip()
    headers: dict[str, str] = {}

    if p == "vllm":
        base, served = resolve_vllm_target(m)
        url = f"{base}/models"
        key = os.getenv("VLLM_API_KEY", "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        logical_model = served or m
    elif p == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        url = f"{base}/api/tags"
        logical_model = m
    elif p == "cerebras":
        base = os.getenv("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1").rstrip("/")
        url = f"{base}/models"
        key = os.getenv("CEREBRAS_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "CEREBRAS_API_KEY must be set for cerebras provider (persona LLM preflight)."
            )
        headers["Authorization"] = f"Bearer {key}"
        logical_model = m
    else:
        raise RuntimeError(f"Unsupported persona LLM provider: {provider!r}")

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url, headers=headers or None)
            response.raise_for_status()
            payload = response.json() or {}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Persona LLM preflight failed: cannot reach {p} server "
            f"(provider={p}, model={m!r}, url={url}): {exc}"
        ) from exc

    if p == "ollama":
        models = payload.get("models") or []
        available = {str(item.get("name") or "").strip() for item in models}
    else:
        models = payload.get("data") or []
        available = {str(item.get("id") or "").strip() for item in models}

    available = {name for name in available if name}
    if not available:
        raise RuntimeError(
            f"Persona LLM preflight failed: {p} server returned no models "
            f"(provider={p}, model={m!r}, url={url})."
        )

    check_name = logical_model if p == "vllm" else m
    if check_name and not _model_id_listed(check_name, available):
        sample = ", ".join(sorted(available)[:8])
        more = "" if len(available) <= 8 else f" (+{len(available) - 8} more)"
        raise RuntimeError(
            f"Persona LLM preflight failed: configured model not served "
            f"(provider={p}, model={check_name!r}, url={url}, "
            f"available=[{sample}{more}])."
        )

    print(
        f"Persona LLM preflight OK (provider={p}, model={check_name or m!r}, "
        f"url={url}, served_count={len(available)})",
        flush=True,
    )


def _default_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{OUTPUT_PREFIX}{ts}.jsonl"


def _resolve_output_dir(raw: Path) -> Path:
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (REPO_ROOT / expanded).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate JSONL of opening customer messages with gold category/intent "
            "using simulator seeds, suite scenarios, and persona LLM."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / DEFAULT_OUTPUT_SUBDIR,
        help=f"Directory for output file (default: {DEFAULT_OUTPUT_SUBDIR} under repo root).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write to this JSONL path instead of a new timestamped file. "
            "When the file already exists, its rows count toward --max-limit and "
            "new rows are appended (resume)."
        ),
    )
    parser.add_argument(
        "--max-limit",
        type=_positive_int,
        required=True,
        help="Number of examples (rows) to generate.",
    )
    parser.add_argument(
        "--seed",
        help="Restrict to this seed_id (must appear in the chosen suite).",
    )
    parser.add_argument(
        "--persona",
        nargs="*",
        default=[],
        help="Restrict persona selection to these persona_id values.",
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE_REL,
        help=f"Suite YAML path (default: {DEFAULT_SUITE_REL} under testing/simulator).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Pick a random scenario each attempt (same idea as simulator --randomize).",
    )
    parser.add_argument(
        "--ollama-url",
        type=_ollama_base_url,
        default=None,
        metavar="URL",
        help=(
            "Ollama API base URL for this run (sets OLLAMA_BASE_URL, overriding .env). "
            "Example: http://127.0.0.1:11434"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=_llm_temperature,
        default=None,
        metavar="T",
        help=(
            "LLM sampling temperature for this run (0.0–2.0). "
            "Overrides suite default and SIMULATOR_USER_LLM_TEMPERATURE when set."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Checkpoint interval for incremental JSONL persistence. "
            "When set, append every N generated rows so interruptions keep prior progress."
        ),
    )
    return parser


@dataclass(frozen=True)
class GenerationResult:
    output_path: Path
    rows_written: int
    attempts: int
    skipped: int
    existing_rows: int = 0

    @property
    def total_rows(self) -> int:
        return self.existing_rows + self.rows_written


def run_generation(
    *,
    output_path: Path,
    max_limit: int,
    suite_path: Path,
    seed_override: str | None,
    persona_filters: list[str],
    randomize: bool,
    temperature_override: float | None = None,
    save_every: int | None = None,
    existing_rows: int = 0,
    stderr_print: Callable[[str], None] | None = None,
    max_attempt_multiplier: int = 50,
    skip_preflight: bool = False,
) -> GenerationResult:
    """Generate JSONL rows until the file reaches max_limit total rows."""
    err = stderr_print or (lambda m: print(m, file=sys.stderr))

    if existing_rows < 0:
        raise ValueError("existing_rows must be non-negative.")
    if existing_rows >= max_limit:
        return GenerationResult(
            output_path=output_path,
            rows_written=0,
            attempts=0,
            skipped=0,
            existing_rows=existing_rows,
        )

    remaining = max_limit - existing_rows

    suite = _load_suite(suite_path)
    _apply_user_llm_env_overrides(suite)
    if temperature_override is not None:
        suite.defaults.user_llm_temperature = temperature_override

    simulator_root = _find_simulator_root(suite_path)
    all_seeds = _load_all_seeds(simulator_root / "seeds")
    seeds_by_id = {seed.seed_id: seed for seed in all_seeds}
    personas = _load_personas(simulator_root / "personas" / "personas.yaml")

    selected_scenarios = _select_scenarios(
        suite=suite,
        seeds_by_id=seeds_by_id,
        seed_override=seed_override,
        category_filters=[],
        persona_filters=persona_filters,
        intent_filters=[],
    )
    if not selected_scenarios:
        raise ValueError("No scenarios matched the requested filters (suite/seed/persona).")

    persona_candidates = _persona_candidates(personas, persona_filters)
    if not persona_candidates:
        raise ValueError("No personas matched the requested --persona filter.")

    if not skip_preflight:
        _preflight_persona_llm(
            provider=suite.defaults.user_llm_provider,
            model=suite.defaults.user_llm_model,
            timeout_seconds=min(float(suite.defaults.user_llm_timeout_seconds), 30.0),
        )

    if randomize:

        def pick_scenario() -> tuple[Any, SeedConfig]:
            return random.choice(selected_scenarios)

    else:
        scenario_cycle = cycle(selected_scenarios)

        def pick_scenario() -> tuple[Any, SeedConfig]:
            return next(scenario_cycle)

    successful_rows: list[dict[str, Any]] = []
    attempts = 0
    skipped = 0
    max_attempts = remaining * max_attempt_multiplier
    append_mode = existing_rows > 0 or save_every is not None

    if existing_rows > 0:
        print(
            f"Resuming {output_path}: {existing_rows}/{max_limit} row(s) already present; "
            f"generating up to {remaining} more",
            flush=True,
        )
    if save_every is None and not append_mode:
        print(
            f"Generating up to {max_limit} sample(s) → {output_path} (writes after all succeed)",
            flush=True,
        )
    elif save_every is not None:
        print(
            f"Generating up to {max_limit} total sample(s) → {output_path} "
            f"(checkpoint append every {save_every} new row(s))",
            flush=True,
        )
    elif append_mode:
        print(
            f"Generating up to {max_limit} total sample(s) → {output_path} "
            f"(append {remaining} new row(s) when complete)",
            flush=True,
        )
    persisted_rows = 0
    while len(successful_rows) < remaining and attempts < max_attempts:
        attempts += 1
        try:
            run_cfg, seed = pick_scenario()

            try:
                scenario = build_fake_scenario_instance(seed, existing_rows + attempts)
            except ValueError as exc:
                err(f"Fake scenario failed for seed {seed.seed_id!r}: {exc}")
                skipped += 1
                continue

            try:
                persona_cfg = _pick_persona_for_scenario(
                    run_cfg=run_cfg,
                    persona_candidates=persona_candidates,
                )
            except RuntimeError as exc:
                err(str(exc))
                skipped += 1
                continue

            scenario.persona_id = persona_cfg.persona_id
            scenario.persona_snapshot = persona_cfg.model_dump(mode="json")
            scenario.cooperation_level = run_cfg.cooperation_level or suite.defaults.cooperation_level

            engine = PersonaEngine(
                persona=persona_cfg,
                scenario=scenario,
                llm_provider=suite.defaults.user_llm_provider,
                llm_model=suite.defaults.user_llm_model,
                llm_timeout_seconds=suite.defaults.user_llm_timeout_seconds,
                llm_temperature=suite.defaults.user_llm_temperature,
                llm_top_p=suite.defaults.user_llm_top_p,
                llm_repeat_penalty=suite.defaults.user_llm_repeat_penalty,
                event_sink=None,
            )
            try:
                text = engine.generate_opening()
            except PersonaGenerationError as exc:
                err(f"Skipped {seed.seed_id} / {persona_cfg.persona_id}: {exc}")
                skipped += 1
                continue

            text = str(text or "").strip()
            if not text:
                skipped += 1
                continue

            row = {
                "text": text,
                "category": seed.category,
                "intent": seed.intent,
                "seed_id": seed.seed_id,
                "persona": persona_cfg.persona_id,
            }
            successful_rows.append(row)
        except Exception as exc:
            err(f"Skipped attempt {attempts}: {type(exc).__name__}: {exc}")
            skipped += 1
            continue

        new_rows = len(successful_rows)
        total_rows = existing_rows + new_rows
        if new_rows > 1:
            print(_SAMPLE_SEPARATOR, flush=True)
        print(
            f"Progress: {total_rows}/{max_limit} rows "
            f"({new_rows} new this run, attempt {attempts}, skipped so far: {skipped})",
            flush=True,
        )
        print(f"Query:\n{text}", flush=True)
        print(f"Category: {seed.category}", flush=True)
        print(f"Intent: {seed.intent}", flush=True)
        print(f"Seed ID: {seed.seed_id}", flush=True)
        print(f"Persona: {persona_cfg.persona_id}", flush=True)
        if save_every is not None and new_rows % save_every == 0:
            _append_rows_jsonl(output_path, successful_rows[persisted_rows:new_rows])
            persisted_rows = new_rows
            print(
                f"Checkpoint persisted: {existing_rows + persisted_rows}/{max_limit} total row(s)",
                flush=True,
            )

    new_rows = len(successful_rows)
    total_rows = existing_rows + new_rows

    if new_rows < remaining and attempts >= max_attempts:
        err(
            f"Stopped after {attempts} attempts ({skipped} skipped); "
            f"collected {new_rows} new row(s) ({total_rows}/{max_limit} total)."
        )

    if new_rows == 0 and existing_rows == 0:
        raise RuntimeError("No rows were written; generation failed.")

    if not append_mode:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as out_f:
            for row in successful_rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif persisted_rows < new_rows:
        _append_rows_jsonl(output_path, successful_rows[persisted_rows:new_rows])
        persisted_rows = new_rows
        if save_every is not None:
            print(
                f"Final checkpoint persisted: {existing_rows + persisted_rows}/{max_limit} total row(s)",
                flush=True,
            )

    return GenerationResult(
        output_path=output_path,
        rows_written=new_rows,
        attempts=attempts,
        skipped=skipped,
        existing_rows=existing_rows,
    )


def main(argv: list[str] | None = None) -> int:
    load_repo_dotenv(REPO_ROOT)

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.ollama_url is not None:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url

    simulator_root = REPO_ROOT / "testing" / "simulator"
    suite_path = _resolve_path(simulator_root, args.suite)

    existing_rows = 0
    if args.output_file is not None:
        output_path = _resolve_output_path(args.output_file)
        if output_path.exists():
            try:
                existing_rows = _count_jsonl_rows(output_path)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if existing_rows >= args.max_limit:
                print(
                    f"{output_path} already has {existing_rows} row(s) "
                    f"(target {args.max_limit}); nothing to do."
                )
                return 0
    else:
        output_dir = _resolve_output_dir(args.output_dir)
        output_path = _default_output_path(output_dir)

    try:
        result = run_generation(
            output_path=output_path,
            max_limit=args.max_limit,
            suite_path=suite_path,
            seed_override=args.seed,
            persona_filters=list(args.persona or []),
            randomize=bool(args.randomize),
            temperature_override=args.temperature,
            save_every=args.save_every,
            existing_rows=existing_rows,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if result.rows_written == 0 and result.existing_rows >= args.max_limit:
        print(
            f"{result.output_path} already at target "
            f"({result.existing_rows}/{args.max_limit} row(s))."
        )
    else:
        print(
            f"Wrote {result.rows_written} new row(s) to {result.output_path} "
            f"(total={result.total_rows}, attempts={result.attempts}, skipped={result.skipped})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
