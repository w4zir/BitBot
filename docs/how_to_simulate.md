# How to run the BitBot simulator

This guide is aligned with the current simulator spec and implementation in `testing/simulator/`.

## What the simulator does

The simulator:

- loads seed definitions from `testing/simulator/seeds/*.yaml`
- hydrates each seed with live Postgres entities (`order`, `user`, `subscription`)
- runs multi-turn conversations against `POST /classify` with `full_flow=true`
- evaluates traces with structural and policy checks by default
- optionally runs an LLM judge (`eval_targets: [llm_judge]`)
- supports deterministic loops, randomized selection, and continuous mode
- writes JSON artifacts to `testing/simulator/results/`
- can persist run/scenario/turn/message/evaluation/training data to Postgres

Recent intent coverage seeds are also available in `testing/simulator/seeds/intent_expansion.yaml`
for payment, invoice, subscription, contact, delivery, feedback, shipping address, and product flows.

## Prerequisites

1. Start the BitBot API server so `POST /classify` is reachable.
2. Ensure Postgres is configured and populated enough for seed filters.
3. Ensure procedure blueprints are available (`backend/procedures/`) for coverage checks.
4. (Recommended) refresh local DB fixtures:

```bash
psql -f db/postgres/01_schema.sql
psql -f db/postgres/02_seed.sql
psql -f db/postgres/03_smoke_checks.sql
```

Required environment variables (minimum):

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecom_support
POSTGRES_USER=admin
POSTGRES_PASSWORD=...
```

Common simulator variables:

```bash
# Override classify endpoint if needed
SIMULATOR_AGENT_URL=http://localhost:8000/classify

# Optional dedicated DB host for simulator process
POSTGRES_HOST_SIMULATOR=localhost

# User-message generation LLM (required)
SIMULATOR_USER_LLM_PROVIDER=ollama
SIMULATOR_USER_LLM_MODEL=llama3.2
SIMULATOR_USER_LLM_TIMEOUT_SECONDS=120

# Optional simulator-only creativity controls (affect persona message generation only)
SIMULATOR_USER_LLM_TEMPERATURE=0.7
SIMULATOR_USER_LLM_TOP_P=0.9
SIMULATOR_USER_LLM_REPEAT_PENALTY=1.1

# LLM judge provider/timeout when llm_judge is enabled
SIMULATOR_LLM_PROVIDER=ollama
SIMULATOR_LLM_TIMEOUT_SECONDS=120
```

Creativity notes:

- Opening messages now use a randomized style profile per conversation.
- Anti-template guardrails reduce repeated openers like "Hi there! I was hoping you could help...".
- If an opening still starts with a banned template pattern, the simulator retries once with a stronger rewrite directive.
- These knobs are simulator-scoped and do not change normal agent runtime behavior unless explicitly passed by simulator code.

## Runtime modes and CLI

From repo root:

```bash
# Deterministic suite run (default: one full pass over selected scenarios)
python -m testing.simulator.runner --suite testing/simulator/suites/smoke.yaml
```

```bash
# Repeat deterministic passes N times
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --iterations 3
```

```bash
# Randomized selection mode (N sampled scenarios)
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --randomize --iterations 20
```

```bash
# Continuous mode until interrupted
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --forever --randomize
```

```bash
# Single seed
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --seed order_cancel_processing_easy
```

```bash
# Filter by category/intent/persona/difficulty
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --category order --intent cancel_order --persona policy_prober --difficulty hard
```

```bash
# Coverage only (no conversations)
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --coverage-only
```

```bash
# Persist run data to Postgres (enabled by default; explicit flag shown for clarity)
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --persist-db

# Disable Postgres persistence for one run
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --no-persist-db
```

## Evaluators and current status

- `structural`: enabled and implemented
- `policy`: enabled and implemented
- `llm_judge`: optional and implemented
- `regression`: config keyword exists, but runtime evaluator is not yet wired (artifact field is currently `null` and `regressions=0`)

## Exit codes

- `0`: all executed scenarios passed evaluator checks
- `1`: one or more executed scenarios failed structural/policy/llm_judge checks
- `3`: unexpected coverage gaps while `fail_on_coverage_gap: true`
- `4`: hydration failure (no matching DB entity for a required filter)

## Artifacts and persistence

Every run writes:

- `testing/simulator/results/run_<timestamp>.json`

Console output includes:

- category/intent coverage table
- per-scenario PASS/FAIL lines
- artifact path

When DB persistence is enabled (default behavior, or `--persist-db`), the simulator writes:

- `simulation_runs`, `coverage_snapshots`, `simulation_scenarios`
- `simulation_turns`, `simulation_messages`
- `simulation_evaluations`, `simulation_llm_judgements`
- `simulation_training_examples`

Token and latency metrics are captured when available:

- per turn: `input_tokens`, `output_tokens`, `cache_tokens`, `total_tokens`, `latency_ms`
- per LLM judge call: provider/model + token usage + latency

## Docker Compose usage

The `simulator` service in `docker-compose.yml` is an idle container when brought up with `docker compose up`.
Run the CLI manually inside it:

```bash
docker compose exec simulator python -m testing.simulator.runner --suite testing/simulator/suites/smoke.yaml --iterations 1
```

One-off Compose execution is also available:

```bash
docker compose run --rm simulator
```

## Reading failures quickly

Use this sequence:

1. Check artifact `summary` (`structural_failures`, `policy_failures`, `llm_judge_failures`).
2. Open failing `scenarios[]` entries (`seed_id`, `expected_outcome`, `final_outcome_status`).
3. Inspect `trace[]` turn-by-turn for:
   - `outcome_status`, `procedure_id`, `validation_missing`
   - `eligibility_ok`, `policy_constraints`, `context_data`
   - `agent_state`, `stage_metadata`, `output_validation`, `context_summary`
4. Validate hydration assumptions (`entity_id` and selected entity fields) against seed `db_filter`.

## Common failure patterns

- Hydration errors (`exit 4`): DB filters too narrow for current data.
- Unexpected `policy_ineligible`: outcome drifted due to policy/procedure changes.
- Validation loops: missing fields stay unresolved across turns.
- Coverage gaps: add seeds or document intentional holes in `testing/simulator/seeds/gaps.yaml`.
- Intent-name mismatch: keep DB `category_intents` names aligned with procedure YAML `intent` names.

## Suggested workflow

1. Run `smoke.yaml` first.
2. Fix structural/policy failures before enabling LLM judge.
3. Re-run failing seed with `--seed`.
4. Use randomized `--iterations`/`--forever` mode for broader scenario sampling.
