from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv

from testing.simulator.config import (
    PersonaConfig,
    PersonasFileConfig,
    ScenarioRunConfig,
    SeedConfig,
    SeedFileConfig,
    SuiteConfig,
)
from testing.simulator.coverage import build_coverage_report, render_coverage_table
from testing.simulator.driver import ConversationDriver
from testing.simulator.evaluators.llm_judge import LlmJudgeResult, evaluate_llm_judge
from testing.simulator.evaluators.policy import PolicyResult, evaluate_policy
from testing.simulator.evaluators.structural import StructuralResult, evaluate_structural
from testing.simulator.hydrator import HydrationError, ScenarioHydrator
from testing.simulator.persona import PersonaEngine
from testing.simulator.persistence import SimulatorPersistence
from testing.simulator.reporter import render_console_summary, write_run_artifact
from testing.simulator.trace import ConversationTrace


def main() -> int:
    # Load repo-level .env for local simulator runs, while preserving explicit shell env.
    load_dotenv(override=False)
    simulator_pg_host = os.getenv("POSTGRES_HOST_SIMULATOR", "").strip()
    if simulator_pg_host:
        # Simulator can run against a different postgres host than app runtime.
        os.environ["POSTGRES_HOST"] = simulator_pg_host

    parser = _build_arg_parser()
    args = parser.parse_args()

    simulator_root = Path(__file__).resolve().parent
    suite_path = _resolve_path(simulator_root, args.suite)
    suite = _load_suite(suite_path)
    if args.db_snapshot:
        suite.db_snapshot = args.db_snapshot
    simulator_agent_url = os.getenv("SIMULATOR_AGENT_URL", "").strip()
    if simulator_agent_url:
        suite.agent_url = simulator_agent_url
    simulator_user_llm_provider = os.getenv("SIMULATOR_USER_LLM_PROVIDER", "").strip().lower()
    if simulator_user_llm_provider:
        suite.defaults.user_llm_provider = simulator_user_llm_provider
    simulator_user_llm_model = os.getenv("SIMULATOR_USER_LLM_MODEL", "").strip()
    if simulator_user_llm_model:
        suite.defaults.user_llm_model = simulator_user_llm_model
    simulator_user_llm_timeout = os.getenv("SIMULATOR_USER_LLM_TIMEOUT_SECONDS", "").strip()
    if simulator_user_llm_timeout:
        suite.defaults.user_llm_timeout_seconds = float(simulator_user_llm_timeout)

    all_seeds = _load_all_seeds(simulator_root / "seeds")
    seeds_by_id = {seed.seed_id: seed for seed in all_seeds}
    personas = _load_personas(simulator_root / "personas" / "personas.yaml")
    selected_scenarios = _select_scenarios(
        suite=suite,
        seeds_by_id=seeds_by_id,
        seed_override=args.seed,
        category_filters=args.category,
        difficulty_filters=args.difficulty,
        persona_filters=args.persona,
        intent_filters=args.intent,
    )
    if not selected_scenarios:
        print("No scenarios matched the requested filters.")
        return 1

    coverage = build_coverage_report(
        seeds=all_seeds,
        gaps_file=simulator_root / "seeds" / "gaps.yaml",
    )
    print(render_coverage_table(coverage))
    if args.coverage_only:
        return _coverage_exit_code(coverage, suite)

    hydrator = ScenarioHydrator()
    driver = ConversationDriver(
        agent_url=suite.agent_url,
        max_turns=suite.defaults.max_turns,
    )
    started_at = datetime.now(timezone.utc)
    traces: list[ConversationTrace] = []
    structural_results: dict[str, StructuralResult] = {}
    policy_results: dict[str, PolicyResult] = {}
    llm_judge_results: dict[str, LlmJudgeResult | None] = {}

    randomize = bool(args.randomize or suite.defaults.randomize)
    persistence_enabled = suite.defaults.persist_db if args.persist_db is None else args.persist_db
    persistence = SimulatorPersistence(enabled=persistence_enabled)
    run_metadata = {
        "randomize": randomize,
        "iterations": args.iterations,
        "forever": args.forever,
        "filters": {
            "seed": args.seed,
            "category": args.category,
            "difficulty": args.difficulty,
            "persona": args.persona,
            "intent": args.intent,
        },
    }
    persistence.start_run(
        run_id=suite.run_id,
        suite_name=suite_path.name,
        db_snapshot=suite.db_snapshot,
        baseline_ref=suite.baseline,
        run_metadata=run_metadata,
    )
    persistence.record_coverage(coverage.to_dict())

    loop_status = "completed"
    try:
        for index, (run_cfg, seed) in enumerate(
            _iter_execution_plan(
                selected_scenarios=selected_scenarios,
                randomize=randomize,
                iterations=args.iterations,
                forever=args.forever,
            ),
            start=1,
        ):
            persona_cfg = personas.get(seed.persona_id)
            if persona_cfg is None:
                raise RuntimeError(f"Persona '{seed.persona_id}' not found for seed '{seed.seed_id}'")
            try:
                scenario = hydrator.hydrate(seed)
            except HydrationError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HydrationError(f"Failed to hydrate seed '{seed.seed_id}': {exc}") from exc

            scenario.cooperation_level = (
                run_cfg.cooperation_level
                or seed.cooperation_level
                or suite.defaults.cooperation_level
            )
            persona = PersonaEngine(
                persona=persona_cfg,
                scenario=scenario,
                llm_provider=suite.defaults.user_llm_provider,
                llm_model=suite.defaults.user_llm_model,
                llm_timeout_seconds=suite.defaults.user_llm_timeout_seconds,
            )
            trace = driver.run(scenario, persona)
            scenario_key = f"{seed.seed_id}#{index}"
            trace.scenario["run_scenario_id"] = scenario_key
            eval_targets = {
                item.strip().lower()
                for item in (run_cfg.eval_targets or suite.defaults.eval_targets or [])
            }
            run_structural = not eval_targets or "structural" in eval_targets
            run_policy = not eval_targets or "policy" in eval_targets
            run_llm_judge = "llm_judge" in eval_targets
            structural = (
                evaluate_structural(trace, scenario, max_turns=suite.defaults.max_turns)
                if run_structural
                else StructuralResult(passed=True, checks={}, failures=[])
            )
            policy = (
                evaluate_policy(trace, scenario)
                if run_policy
                else PolicyResult(passed=True, checks={}, failures=[])
            )
            llm_judge = (
                evaluate_llm_judge(
                    trace=trace,
                    scenario=scenario,
                    provider=suite.defaults.llm_judge_provider,
                    model=suite.defaults.llm_judge_model,
                    thresholds=suite.defaults.llm_judge_thresholds,
                )
                if run_llm_judge
                else None
            )
            traces.append(trace)
            structural_results[scenario_key] = structural
            policy_results[scenario_key] = policy
            llm_judge_results[scenario_key] = llm_judge
            persistence.record_scenario(
                trace=trace,
                structural=structural,
                policy=policy,
                llm_judge=llm_judge,
            )
    except KeyboardInterrupt:
        loop_status = "interrupted"
        print("Simulator interrupted by user.")

    artifact_path = write_run_artifact(
        run_id=suite.run_id,
        suite_path=str(suite_path),
        db_snapshot=suite.db_snapshot,
        agent_url=suite.agent_url,
        coverage=coverage,
        traces=traces,
        structural_results=structural_results,
        policy_results=policy_results,
        llm_judge_results=llm_judge_results,
        output_dir=simulator_root / "results",
        started_at=started_at,
    )
    print("")
    print(render_console_summary(traces, structural_results, policy_results, llm_judge_results))
    print("")
    print(f"Artifact: {artifact_path}")
    exit_code = _run_exit_code(
        traces,
        structural_results,
        policy_results,
        llm_judge_results,
        coverage,
        suite,
    )
    persistence.complete_run(
        summary={
            "exit_code": exit_code,
            "status": loop_status,
            "artifact_path": str(artifact_path),
            "scenarios_executed": len(traces),
        },
        status=loop_status if loop_status != "completed" else ("completed" if exit_code == 0 else "failed"),
    )
    return exit_code


def _run_exit_code(
    traces: list[ConversationTrace],
    structural_results: dict[str, StructuralResult],
    policy_results: dict[str, PolicyResult],
    llm_judge_results: dict[str, LlmJudgeResult | None],
    coverage_report,
    suite: SuiteConfig,
) -> int:
    _ = traces
    if any(not item.passed for item in structural_results.values()):
        return 1
    if any(not item.passed for item in policy_results.values()):
        return 1
    if any((item is not None) and (not item.passed) for item in llm_judge_results.values()):
        return 1
    if suite.defaults.fail_on_coverage_gap and coverage_report.unexpected_gaps > 0:
        return 3
    return 0


def _coverage_exit_code(coverage_report, suite: SuiteConfig) -> int:
    if suite.defaults.fail_on_coverage_gap and coverage_report.unexpected_gaps > 0:
        return 3
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BitBot simulator runner")
    parser.add_argument("--suite", required=True, help="Suite YAML path, e.g. suites/smoke.yaml")
    parser.add_argument("--seed", help="Run only one seed_id from the suite")
    parser.add_argument("--db-snapshot", help="Optional override for db_snapshot field")
    parser.add_argument("--write-baseline", action="store_true", help="Reserved for future use")
    parser.add_argument("--coverage-only", action="store_true", help="Only evaluate coverage")
    parser.add_argument("--category", nargs="*", default=[], help="Filter to categories")
    parser.add_argument("--difficulty", nargs="*", default=[], help="Filter to difficulties")
    parser.add_argument("--persona", nargs="*", default=[], help="Filter to personas")
    parser.add_argument("--intent", nargs="*", default=[], help="Filter to intents")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Repeat run N times (for random mode this means N sampled scenarios)",
    )
    parser.add_argument("--forever", action="store_true", help="Run indefinitely until interrupted")
    parser.add_argument("--randomize", action="store_true", help="Randomize scenario selection order")
    parser.add_argument(
        "--persist-db",
        dest="persist_db",
        action="store_true",
        help="Persist run artifacts to Postgres",
    )
    parser.add_argument(
        "--no-persist-db",
        dest="persist_db",
        action="store_false",
        help="Disable Postgres persistence for this run",
    )
    parser.set_defaults(persist_db=None)
    return parser


def _resolve_path(simulator_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (simulator_root / raw_path).resolve()


def _load_suite(path: Path) -> SuiteConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    suite = SuiteConfig.model_validate(data)
    return suite


def _load_all_seeds(seeds_dir: Path) -> list[SeedConfig]:
    all_seeds: list[SeedConfig] = []
    for path in sorted(seeds_dir.glob("*.yaml")):
        if path.name == "gaps.yaml":
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        parsed = SeedFileConfig.model_validate(raw)
        all_seeds.extend(parsed.seeds)
    return all_seeds


def _load_personas(path: Path) -> dict[str, PersonaConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parsed = PersonasFileConfig.model_validate(raw)
    return {item.persona_id: item for item in parsed.personas}


def _select_scenarios(
    *,
    suite: SuiteConfig,
    seeds_by_id: dict[str, SeedConfig],
    seed_override: str | None,
    category_filters: list[str],
    difficulty_filters: list[str],
    persona_filters: list[str],
    intent_filters: list[str],
) -> list[tuple[ScenarioRunConfig, SeedConfig]]:
    selected: list[tuple[ScenarioRunConfig, SeedConfig]] = []
    allowed_categories = {c.strip().lower() for c in category_filters}
    allowed_difficulties = {d.strip().lower() for d in difficulty_filters}
    allowed_personas = {p.strip().lower() for p in persona_filters}
    allowed_intents = {i.strip().lower() for i in intent_filters}
    for run_cfg in suite.scenarios:
        if seed_override and run_cfg.seed_id != seed_override:
            continue
        seed = seeds_by_id.get(run_cfg.seed_id)
        if seed is None:
            raise RuntimeError(f"Seed '{run_cfg.seed_id}' referenced by suite is missing.")
        if allowed_categories and seed.category.strip().lower() not in allowed_categories:
            continue
        if allowed_difficulties and seed.difficulty.strip().lower() not in allowed_difficulties:
            continue
        if allowed_personas and seed.persona_id.strip().lower() not in allowed_personas:
            continue
        if allowed_intents and seed.intent.strip().lower() not in allowed_intents:
            continue
        selected.append((run_cfg, seed))
    return selected


def _iter_execution_plan(
    *,
    selected_scenarios: list[tuple[ScenarioRunConfig, SeedConfig]],
    randomize: bool,
    iterations: int,
    forever: bool,
) -> Iterable[tuple[ScenarioRunConfig, SeedConfig]]:
    if not selected_scenarios:
        return []
    if forever:
        while True:
            if randomize:
                yield random.choice(selected_scenarios)
            else:
                for item in selected_scenarios:
                    yield item
        return
    total_iterations = max(1, int(iterations))
    if randomize:
        for _ in range(total_iterations):
            yield random.choice(selected_scenarios)
        return
    for _ in range(total_iterations):
        for item in selected_scenarios:
            yield item


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HydrationError as exc:
        print(f"Hydration error: {exc}", file=sys.stderr)
        raise SystemExit(4)
