# How to run the BitBot simulator

This guide explains how to run the simulator in `testing/simulator/` and how to interpret output artifacts.

## What the simulator does

The simulator:

- loads scenario seeds from `testing/simulator/seeds/*.yaml`
- hydrates each seed with real DB entities (orders/users/subscriptions)
- runs multi-turn conversations against `POST /classify`
- evaluates each trace with structural + policy checks
- writes a JSON artifact under `testing/simulator/results/`

## Prerequisites

1. Start the BitBot API server so `POST /classify` is reachable.
2. Ensure Postgres is configured (the simulator hydrator reads live DB rows).
3. Make sure procedure YAMLs are available under `backend/procedures/`.
4. (Recommended) Refresh local test data with:

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

Optional simulator variables:

```bash
# Override if you do not use localhost:8000/classify
SIMULATOR_AGENT_URL=http://localhost:8000/classify

# Confidence check used by structural evaluator
CATEGORY_CONFIDENCE_THRESHOLD=0.5
```

## Run commands

From repo root:

```bash
# Smoke suite (fastest way to validate end-to-end flow)
python -m testing.simulator.runner --suite testing/simulator/suites/smoke.yaml
```

```bash
# Core regression suite
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml
```

```bash
# Run one seed from the selected suite
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --seed order_cancel_processing_easy
```

```bash
# Run only selected categories
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --category order refund
```

```bash
# Run only selected difficulty levels
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --difficulty hard adversarial
```

```bash
# Coverage check only (no conversations)
python -m testing.simulator.runner --suite testing/simulator/suites/regression.yaml --coverage-only
```

## Exit codes

- `0`: all executed scenarios passed evaluator checks
- `1`: one or more scenario evaluator failures (structural and/or policy)
- `3`: coverage gaps found while `fail_on_coverage_gap: true`
- `4`: hydration failure (no matching DB entity for at least one seed)

## Where results are written

Each run writes an artifact:

- `testing/simulator/results/run_<timestamp>.json`

Console output also includes:

- coverage table (`covered`, `known_gap`, `gap`)
- per-seed PASS/FAIL summary
- artifact file path

The simulator remains artifact-first (JSON files under `testing/simulator/results/`), but the local Postgres test schema now also includes spec-aligned observability tables and reporting views for querying runs in SQL.

## Optional: inspect run telemetry in Postgres

If you seed `db/postgres/02_seed.sql`, you can quickly inspect SQL-backed run/telemetry data with:

```bash
psql -c "SELECT * FROM v_simulation_run_summary;"
psql -c "SELECT * FROM v_simulation_outcome_breakdown;"
psql -c "SELECT * FROM v_llm_performance_summary;"
psql -c "SELECT * FROM v_handoff_queue_status;"
```

Useful tables for deeper inspection:

- `simulation_runs`, `simulation_scenarios`, `coverage_snapshots`
- `tool_invocations`, `llm_metrics`, `audit_log`
- `escalation_handoffs`, `session_entities`, `procedure_blueprints`

## How to interpret results

Use this order when triaging a failed run:

1. **Start with `summary`**
   - `structural_failures` means the graph behavior/outcome mismatched expectations.
   - `policy_failures` means policy evidence or eligibility assertions failed.

2. **Inspect failing entries in `scenarios[]`**
   - check `seed_id`, `final_outcome_status`, `expected_outcome`
   - inspect `structural.failures` and `policy.failures` (human-readable reasons)

3. **Read the `trace` for the failing scenario**
   - each turn includes:
     - `user_message`
     - `agent_response`
     - `outcome_status`
     - `procedure_id`
     - `validation_missing`
     - `eligibility_ok`
     - `context_data` / `policy_constraints`
   - this is the fastest way to see whether failure came from classification, validation, policy gating, tool execution, or escalation routing.

4. **Verify hydration assumptions**
   - confirm `entity_id` and scenario entity fields match the seed filter expectations.
   - if entity selection drifts due to live DB changes, update seed filters or run against a controlled fixture DB.

## Common failure patterns

- **Hydration error (`exit 4`)**
  - seed `db_filter` no longer matches current DB contents.
  - fix by widening status/age filters or refreshing seed data.

- **Unexpected `policy_ineligible`**
  - check final turn `policy_constraints` and `eligibility_ok`.
  - verify seed expected outcome aligns with current procedure/policy behavior.

- **Validation loop never resolves**
  - inspect `validation_missing` across turns.
  - verify persona behavior and required fields in relevant procedure YAML.

- **Coverage gaps**
  - add seeds for uncovered `(category, intent)` pairs, or record intentional gaps in `testing/simulator/seeds/gaps.yaml`.

## Suggested workflow

1. Run smoke suite first.
2. Fix structural failures before policy tuning.
3. Re-run a single failing seed with `--seed`.
4. Run regression suite only after smoke is clean.
