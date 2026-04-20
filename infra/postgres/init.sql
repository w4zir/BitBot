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
