from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from testing.simulator.evaluators.policy import PolicyResult, evaluate_policy
from testing.simulator.evaluators.structural import StructuralResult, evaluate_structural
from testing.simulator.hydrator import HydrationError, ScenarioHydrator
from testing.simulator.persona import PersonaEngine
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
    simulator_agent_url = os.getenv("SIMULATOR_AGENT_URL", "").strip()
    if simulator_agent_url:
        suite.agent_url = simulator_agent_url

    all_seeds = _load_all_seeds(simulator_root / "seeds")
    personas = _load_personas(simulator_root / "personas" / "personas.yaml")

    selected_seed_ids = _selected_seed_ids(suite.scenarios, args.seed)
    selected_seeds = [seed for seed in all_seeds if seed.seed_id in selected_seed_ids]

    if args.category:
        allowed = {c.strip().lower() for c in args.category}
        selected_seeds = [seed for seed in selected_seeds if seed.category.strip().lower() in allowed]

    if args.difficulty:
        allowed = {d.strip().lower() for d in args.difficulty}
        selected_seeds = [seed for seed in selected_seeds if seed.difficulty.strip().lower() in allowed]

    if not selected_seeds:
        print("No seeds matched the requested filters.")
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
    eval_targets = {item.strip().lower() for item in suite.defaults.eval_targets}
    run_structural = not eval_targets or "structural" in eval_targets
    run_policy = not eval_targets or "policy" in eval_targets
    started_at = datetime.now(timezone.utc)
    traces: list[ConversationTrace] = []
    structural_results: dict[str, StructuralResult] = {}
    policy_results: dict[str, PolicyResult] = {}

    for seed in selected_seeds:
        persona_cfg = personas.get(seed.persona_id)
        if persona_cfg is None:
            raise RuntimeError(f"Persona '{seed.persona_id}' not found for seed '{seed.seed_id}'")
        try:
            scenario = hydrator.hydrate(seed)
        except HydrationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HydrationError(f"Failed to hydrate seed '{seed.seed_id}': {exc}") from exc

        scenario.cooperation_level = seed.cooperation_level or suite.defaults.cooperation_level
        persona = PersonaEngine(persona=persona_cfg, scenario=scenario)
        trace = driver.run(scenario, persona)

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

        traces.append(trace)
        structural_results[seed.seed_id] = structural
        policy_results[seed.seed_id] = policy

    artifact_path = write_run_artifact(
        run_id=suite.run_id,
        suite_path=str(suite_path),
        db_snapshot=suite.db_snapshot,
        agent_url=suite.agent_url,
        coverage=coverage,
        traces=traces,
        structural_results=structural_results,
        policy_results=policy_results,
        output_dir=simulator_root / "results",
        started_at=started_at,
    )
    print("")
    print(render_console_summary(traces, structural_results, policy_results))
    print("")
    print(f"Artifact: {artifact_path}")
    return _run_exit_code(
        traces,
        structural_results,
        policy_results,
        coverage,
        suite,
        run_structural=run_structural,
        run_policy=run_policy,
    )


def _run_exit_code(
    traces: list[ConversationTrace],
    structural_results: dict[str, StructuralResult],
    policy_results: dict[str, PolicyResult],
    coverage_report,
    suite: SuiteConfig,
    *,
    run_structural: bool,
    run_policy: bool,
) -> int:
    _ = traces
    if run_structural and any(not item.passed for item in structural_results.values()):
        return 1
    if run_policy and any(not item.passed for item in policy_results.values()):
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


def _selected_seed_ids(scenarios: list[ScenarioRunConfig], seed_override: str | None) -> set[str]:
    if seed_override:
        return {seed_override}
    return {item.seed_id for item in scenarios}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HydrationError as exc:
        print(f"Hydration error: {exc}", file=sys.stderr)
        raise SystemExit(4)
