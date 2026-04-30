-- infra/postgres/init.sql — core tables (Phase 0/1)
-- Policy text retrieval uses Elasticsearch only (see README).

CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(100),
    company_id  VARCHAR(100),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_number    VARCHAR(50) UNIQUE NOT NULL,
    user_id         VARCHAR(100),
    company_id      VARCHAR(100),
    status          VARCHAR(50),
    items           JSONB,
    total_amount    DECIMAL(10,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    estimated_delivery TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS products (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku             VARCHAR(80) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    price           DECIMAL(10,2) NOT NULL,
    is_available    BOOLEAN NOT NULL DEFAULT TRUE,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_products_name ON products (name);
CREATE INDEX IF NOT EXISTS idx_products_lower_name ON products ((lower(name)));

CREATE TABLE tickets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    issue_type  VARCHAR(100),
    summary     TEXT,
    status      VARCHAR(50) DEFAULT 'open',
    priority    VARCHAR(20) DEFAULT 'normal',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Phases 5–6: observability warehouse + analytics (must match backend/db/models)
-- Kept inline so docker-entrypoint-initdb.d applies them on first DB init.
-- For an existing volume, run infra/postgres/migrations/all_migrations.sql
-- (or scripts/apply_postgres_warehouse.ps1).
-- ---------------------------------------------------------------------------

-- From migrations/003_observability_warehouse.sql
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'intent'
  ) THEN
    ALTER TABLE sessions ADD COLUMN intent VARCHAR(50);
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'escalated'
  ) THEN
    ALTER TABLE sessions ADD COLUMN escalated BOOLEAN DEFAULT FALSE;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'resolved_at'
  ) THEN
    ALTER TABLE sessions ADD COLUMN resolved_at TIMESTAMPTZ;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'user_request'
  ) THEN
    ALTER TABLE sessions ADD COLUMN user_request TEXT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'problem_to_solve'
  ) THEN
    ALTER TABLE sessions ADD COLUMN problem_to_solve TEXT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'issue_category'
  ) THEN
    ALTER TABLE sessions ADD COLUMN issue_category VARCHAR(100);
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'issue_confidence'
  ) THEN
    ALTER TABLE sessions ADD COLUMN issue_confidence DOUBLE PRECISION;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS agent_spans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
  trace_id VARCHAR(100),
  span_name VARCHAR(100) NOT NULL,
  attributes JSONB,
  latency_ms NUMERIC(12, 3),
  "timestamp" TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS outcomes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  task VARCHAR(100) NOT NULL,
  completed BOOLEAN NOT NULL,
  escalated BOOLEAN NOT NULL DEFAULT FALSE,
  verified BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_scores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
  groundedness DOUBLE PRECISION,
  hallucination BOOLEAN,
  helpfulness DOUBLE PRECISION,
  metadata JSONB,
  evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_spans_session_id ON agent_spans (session_id);
CREATE INDEX IF NOT EXISTS idx_agent_spans_timestamp ON agent_spans ("timestamp");
CREATE INDEX IF NOT EXISTS idx_outcomes_session_id ON outcomes (session_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_created_at ON outcomes (created_at);
CREATE INDEX IF NOT EXISTS idx_eval_scores_session_id ON evaluation_scores (session_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_evaluated_at ON evaluation_scores (evaluated_at);

-- From migrations/004_analytics_views.sql
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'csat_score'
  ) THEN
    ALTER TABLE sessions ADD COLUMN csat_score SMALLINT;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'sessions' AND column_name = 'nps_score'
  ) THEN
    ALTER TABLE sessions ADD COLUMN nps_score SMALLINT;
  END IF;
END $$;

CREATE OR REPLACE VIEW v_automation_rate AS
SELECT
  COALESCE(
    COUNT(*) FILTER (WHERE completed = TRUE AND escalated = FALSE)::DOUBLE PRECISION
    / NULLIF(COUNT(*), 0),
    0.0
  ) AS automation_rate
FROM outcomes;

CREATE OR REPLACE VIEW v_escalation_rate AS
SELECT
  COALESCE(
    COUNT(*) FILTER (WHERE escalated = TRUE)::DOUBLE PRECISION
    / NULLIF(COUNT(*), 0),
    0.0
  ) AS escalation_rate
FROM outcomes;

CREATE OR REPLACE VIEW v_tool_success_rate AS
SELECT
  COALESCE(
    AVG(
      CASE
        WHEN span_name = 'execute_tool' THEN
          CASE
            WHEN COALESCE((attributes ->> 'success')::BOOLEAN, FALSE) THEN 1.0
            ELSE 0.0
          END
        ELSE NULL
      END
    ),
    0.0
  ) AS tool_success_rate
FROM agent_spans;

CREATE OR REPLACE VIEW v_hallucination_rate AS
SELECT
  COALESCE(AVG(CASE WHEN hallucination THEN 1.0 ELSE 0.0 END), 0.0) AS hallucination_rate
FROM evaluation_scores;

-- ---------------------------------------------------------------------------
-- Intent taxonomy: Bitext categories/intents + custom no_issue / product
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS intent_categories (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    source       VARCHAR(50)  NOT NULL DEFAULT 'bitext',
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS category_intents (
    id              SERIAL PRIMARY KEY,
    category_name   VARCHAR(100) NOT NULL REFERENCES intent_categories (name) ON DELETE CASCADE,
    intent_name     VARCHAR(200) NOT NULL,
    display_name    VARCHAR(200) NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (category_name, intent_name)
);

CREATE INDEX IF NOT EXISTS idx_category_intents_category ON category_intents (category_name);

-- ---------------------------------------------------------------------------
-- Simulator persistence tables
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS simulation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(120) UNIQUE NOT NULL,
    suite_name VARCHAR(120) NOT NULL,
    source VARCHAR(40) NOT NULL DEFAULT 'simulator',
    db_snapshot VARCHAR(255),
    baseline_ref VARCHAR(255),
    git_sha VARCHAR(80),
    status VARCHAR(30) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    run_metadata_json JSONB,
    summary_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    seed_id VARCHAR(150) NOT NULL,
    persona_id VARCHAR(120),
    category VARCHAR(100) NOT NULL,
    intent VARCHAR(200) NOT NULL,
    linked_order_id VARCHAR(120),
    linked_user_id VARCHAR(120),
    linked_subscription_email VARCHAR(255),
    expected_outcome VARCHAR(80),
    actual_outcome VARCHAR(80),
    passed BOOLEAN,
    assertions_json JSONB,
    trace_json JSONB,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS coverage_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    total_pairs INTEGER NOT NULL,
    covered_pairs INTEGER NOT NULL,
    known_gaps INTEGER NOT NULL DEFAULT 0,
    unexpected_gaps INTEGER NOT NULL DEFAULT 0,
    coverage_ratio NUMERIC(6, 4),
    gap_details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    request_started_at TIMESTAMPTZ,
    response_received_at TIMESTAMPTZ,
    latency_ms NUMERIC(12, 3),
    outcome_status VARCHAR(80),
    procedure_id VARCHAR(200),
    category VARCHAR(100),
    intent VARCHAR(200),
    validation_missing JSONB,
    policy_constraints JSONB,
    context_data JSONB,
    agent_state JSONB,
    stage_metadata JSONB,
    output_validation JSONB,
    context_summary JSONB,
    request_payload JSONB,
    response_payload JSONB,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_tokens INTEGER,
    total_tokens INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (scenario_id, turn_number)
);

CREATE TABLE IF NOT EXISTS simulation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    turn_id UUID REFERENCES simulation_turns(id) ON DELETE SET NULL,
    message_index INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata_json JSONB,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms NUMERIC(12, 3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    evaluator_name VARCHAR(40) NOT NULL,
    passed BOOLEAN NOT NULL,
    checks_json JSONB,
    failures_json JSONB,
    details_json JSONB,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (scenario_id, evaluator_name)
);

CREATE TABLE IF NOT EXISTS simulation_llm_judgements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id UUID NOT NULL UNIQUE REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    provider VARCHAR(40) NOT NULL,
    model_name VARCHAR(120) NOT NULL,
    passed BOOLEAN NOT NULL,
    scores_json JSONB,
    rationales_json JSONB,
    thresholds_json JSONB,
    failures_json JSONB,
    raw_response_json JSONB,
    latency_ms NUMERIC(12, 3),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_tokens INTEGER,
    total_tokens INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_training_examples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    scenario_id UUID NOT NULL UNIQUE REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    example_format VARCHAR(40) NOT NULL DEFAULT 'chatml',
    messages_json JSONB NOT NULL,
    prompt_text TEXT,
    response_text TEXT,
    labels_json JSONB,
    modernbert_category VARCHAR(100),
    modernbert_intent VARCHAR(200),
    outcome_status VARCHAR(80),
    metadata_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_status ON simulation_runs (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_simulation_scenarios_run ON simulation_scenarios (run_id, category, intent);
CREATE INDEX IF NOT EXISTS idx_coverage_snapshots_run ON coverage_snapshots (run_id);
CREATE INDEX IF NOT EXISTS idx_simulation_turns_scenario ON simulation_turns (scenario_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_simulation_messages_scenario ON simulation_messages (scenario_id, message_index);
CREATE INDEX IF NOT EXISTS idx_simulation_evaluations_scenario ON simulation_evaluations (scenario_id, evaluator_name);
CREATE INDEX IF NOT EXISTS idx_simulation_llm_judgements_scenario ON simulation_llm_judgements (scenario_id);
CREATE INDEX IF NOT EXISTS idx_simulation_training_examples_run ON simulation_training_examples (run_id, modernbert_category, modernbert_intent);
