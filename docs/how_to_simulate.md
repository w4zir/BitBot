# How to run the BitBot simulator

This guide is aligned with the current simulator spec and implementation in `testing/simulator/`.

## What the simulator does

The simulator:

- loads seed definitions from `testing/simulator/seeds/*.yaml` (committed files: `order.yaml`, `refund.yaml`, `intent_expansion.yaml`, plus `gaps.yaml` for known coverage gaps)
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
4. (Recommended) refresh local DB fixtures.

   **Host `psql`** (if installed and connected to the same DB the simulator uses):

   ```bash
   psql -f db/postgres/01_schema.sql
   psql -f db/postgres/02_seed.sql
   psql -f db/postgres/03_smoke_checks.sql
   ```

   **Docker Compose** (stack running; same pattern as [README.md](../README.md) `db/postgres/`):

   ```bash
   docker compose exec -T postgres psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-ecom_support}" -f - < db/postgres/01_schema.sql
   docker compose exec -T postgres psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-ecom_support}" -f - < db/postgres/02_seed.sql
   docker compose exec -T postgres psql -U "${POSTGRES_USER:-admin}" -d "${POSTGRES_DB:-ecom_support}" -f - < db/postgres/03_smoke_checks.sql
   ```

Required environment variables (minimum for running the simulator **on the host** against a Compose database):

Copy [`.env.local.example`](../.env.local.example) to `.env.local` next to your `.env` so repo scripts override Docker service hostnames with `localhost` (see [`backend/repo_dotenv.py`](../backend/repo_dotenv.py)). Alternatively set:

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

# User-message generation LLM (required). Provider: ollama | cerebras | vllm
SIMULATOR_USER_LLM_PROVIDER=ollama
SIMULATOR_USER_LLM_MODEL=llama3.2
SIMULATOR_USER_LLM_TIMEOUT_SECONDS=120

# Optional simulator-only creativity controls (affect persona message generation only)
SIMULATOR_USER_LLM_TEMPERATURE=0.7
SIMULATOR_USER_LLM_TOP_P=0.9
SIMULATOR_USER_LLM_REPEAT_PENALTY=1.1

# vLLM (when SIMULATOR_*_PROVIDER or suite defaults use vllm): OpenAI-compatible base URL + optional API key
# Compose: docker compose --profile vllm up --build  (see README)
# VLLM_API_BASE=http://localhost:8001/v1
# VLLM_SERVED_NAME=gemma4:e4b
# VLLM_API_KEY=

# LLM judge provider/timeout when llm_judge is enabled (ollama | cerebras | vllm)
SIMULATOR_LLM_PROVIDER=ollama
SIMULATOR_LLM_MODEL=llama3.2
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
# Filter by category/intent/persona (persona narrows runtime persona selection)
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --category order --intent cancel_order --persona policy_prober
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

Every run writes a curated debug artifact:

- `testing/simulator/results/run_<timestamp>.json`
- schema reference: `testing/simulator/metadata_template.json`

Top-level fields include `schema_version`, `artifact_type`, `environment` (agent URL, DB snapshot, run config), `coverage`, `summary`, `per_category`, `scenarios`, and `skipped_scenarios`.

Each scenario includes `session_id`, `terminated_by`, `outcome_status_reason`, `entity_snapshot`, optional `procedure_snapshot` / `policy_snapshot`, nested `evaluation`, `metrics`, and a per-turn `trace[]` with only debugging-useful fields (no full LLM prompts or raw model responses).

Console output includes:

- category/intent coverage table
- per-scenario PASS/FAIL lines
- per-turn curated agent trace JSON (same shape as artifact `trace[]`)
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

1. Check artifact `summary` (`structural_failures`, `policy_failures`, `llm_judge_failures`, `skipped`).
2. Open failing `scenarios[]` entries (`seed_id`, `expected_outcome`, `final_outcome_status`, `outcome_status_reason`, `evaluation`).
3. Inspect `trace[]` turn-by-turn for:
   - `outcome_status`, `procedure_id`, `latency_ms`, `assistant_metadata.validation_missing`
   - `assistant_metadata.eligibility_ok`, `policy_constraints`, `context_data`
   - `policy_check_results` and scenario-level `policy_snapshot`
   - `agent_state`, `output_validation`, `context_summary`, pruned `nodes` (step details + parsed LLM output, no prompt dumps)
4. Validate hydration assumptions (`entity_id`, `entity_snapshot`) against seed `db_filter`.

## Common failure patterns

- Hydration errors (`exit 4`): DB filters too narrow for current data.
- Unexpected `policy_ineligible`: outcome drifted due to policy/procedure changes.
- Validation loops: missing fields stay unresolved across turns.
- Coverage gaps: add seeds or document intentional holes in `testing/simulator/seeds/gaps.yaml`.
- Intent-name mismatch: keep DB `category_intents` names aligned with procedure YAML `intent` names.
- Category misroutes: evaluate the ModernBERT checkpoint on simulated holdout — see [testing/README.md](../testing/README.md) and [docs/finetuning-modernbert.md](finetuning-modernbert.md).

## Suggested workflow

1. Run `smoke.yaml` first.
2. Fix structural/policy failures before enabling LLM judge.
3. Re-run failing seed with `--seed`.
4. Use randomized `--iterations`/`--forever` mode for broader scenario sampling.
5. Generate training data from seeds: `python testing/scripts/generate_category_intent_dataset.py --max-limit 1000` — see [training/README.md](../training/README.md#step-21--generate-simulated-categoryintent-jsonl) for the full phase-2 dataset pipeline.
