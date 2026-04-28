-- BitBot: Postgres schema for local/testing dummy data
-- Rerunnable: drops existing objects in FK-safe order, then recreates.
--
-- Includes chat/session + observability structures aligned with infra/postgres/init.sql
-- (dummy ecommerce tables below use VARCHAR order_id; infra UUID orders are not used here).

BEGIN;

-- Category / intent taxonomy (Bitext + custom); drop before dependent nothing else references these
DROP TABLE IF EXISTS category_intents CASCADE;
DROP TABLE IF EXISTS intent_categories CASCADE;

-- Session / observability (must drop before sessions)
DROP VIEW IF EXISTS v_hallucination_rate CASCADE;
DROP VIEW IF EXISTS v_tool_success_rate CASCADE;
DROP VIEW IF EXISTS v_escalation_rate CASCADE;
DROP VIEW IF EXISTS v_automation_rate CASCADE;
DROP VIEW IF EXISTS v_handoff_queue_status CASCADE;
DROP VIEW IF EXISTS v_llm_performance_summary CASCADE;
DROP VIEW IF EXISTS v_simulation_run_summary CASCADE;
DROP VIEW IF EXISTS v_simulation_outcome_breakdown CASCADE;
DROP TABLE IF EXISTS evaluation_scores CASCADE;
DROP TABLE IF EXISTS outcomes CASCADE;
DROP TABLE IF EXISTS agent_spans CASCADE;
DROP TABLE IF EXISTS coverage_snapshots CASCADE;
DROP TABLE IF EXISTS simulation_scenarios CASCADE;
DROP TABLE IF EXISTS simulation_runs CASCADE;
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS llm_metrics CASCADE;
DROP TABLE IF EXISTS tool_invocations CASCADE;
DROP TABLE IF EXISTS session_entities CASCADE;
DROP TABLE IF EXISTS escalation_handoffs CASCADE;
DROP TABLE IF EXISTS procedure_blueprints CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS tickets CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;

-- Children first (reverse dependency order)
DROP TABLE IF EXISTS security_incidents CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS refund_requests CASCADE;
DROP TABLE IF EXISTS invoices CASCADE;
DROP TABLE IF EXISTS shipments CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS loyalty_accounts CASCADE;
DROP TABLE IF EXISTS subscription_accounts CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Core User and Loyalty Tables
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'active', -- active, suspended, deleted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE loyalty_accounts (
    user_id INT PRIMARY KEY REFERENCES users(user_id),
    annual_spend DECIMAL(10, 2),
    tier VARCHAR(50), -- Silver, Gold
    benefits_json JSONB
);

-- Order and Product Tables
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY, -- Format: ORD-XXXX
    user_id INT REFERENCES users(user_id),
    order_date TIMESTAMP,
    status VARCHAR(50), -- processing, shipped, delivered, cancelled
    total_amount DECIMAL(10, 2),
    shipping_address_line TEXT,
    shipping_city VARCHAR(100),
    shipping_postal_code VARCHAR(30),
    shipping_country VARCHAR(80)
);

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    item_name VARCHAR(255),
    category VARCHAR(100), -- e.g., electronics
    is_opened BOOLEAN DEFAULT FALSE,
    qty INT,
    price DECIMAL(10, 2)
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    sku VARCHAR(80) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_products_name ON products (name);
CREATE INDEX idx_products_lower_name ON products ((lower(name)));

-- Fulfillment and Payment Tables
CREATE TABLE payments (
    transaction_id VARCHAR(50) PRIMARY KEY, -- Format: TXN-XXXX
    order_id VARCHAR(50) REFERENCES orders(order_id),
    amount DECIMAL(10, 2),
    method VARCHAR(50),
    payment_status VARCHAR(50),
    charged_at TIMESTAMP
);

CREATE TABLE shipments (
    tracking_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    shipping_tier VARCHAR(50), -- standard, priority
    promised_delivery_at TIMESTAMP,
    actual_delivery_at TIMESTAMP,
    delay_reason VARCHAR(255) -- weather, carrier_error, etc.
);

-- Subscription and Billing
CREATE TABLE subscription_accounts (
    account_email VARCHAR(255) PRIMARY KEY,
    plan VARCHAR(50),
    next_renewal_at TIMESTAMP,
    last_charge_at TIMESTAMP,
    subscription_status VARCHAR(50)
);

CREATE TABLE invoices (
    invoice_id VARCHAR(50) PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    order_id VARCHAR(50) NULL REFERENCES orders(order_id),
    account_email VARCHAR(255) NULL REFERENCES subscription_accounts(account_email),
    amount DECIMAL(10, 2),
    issued_at TIMESTAMP,
    status VARCHAR(50)
);

-- Post-Purchase and Support
CREATE TABLE refund_requests (
    refund_id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id),
    reason TEXT,
    requested_at TIMESTAMP,
    decision VARCHAR(50), -- approved, denied, pending
    decision_reason TEXT
);

CREATE TABLE support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    issue_type VARCHAR(50),
    user_id INT REFERENCES users(user_id),
    payload_json JSONB,
    validation_passed BOOLEAN,
    routing_result VARCHAR(100)
);

CREATE TABLE security_incidents (
    incident_id SERIAL PRIMARY KEY,
    ticket_id INT REFERENCES support_tickets(ticket_id),
    pii_type VARCHAR(50), -- PAN, CVV, SSN
    redacted BOOLEAN DEFAULT FALSE,
    escalated_to VARCHAR(100),
    status VARCHAR(50)
);

-- ---------------------------------------------------------------------------
-- Intent taxonomy: Bitext categories/intents + custom no_issue / product
-- ---------------------------------------------------------------------------

CREATE TABLE intent_categories (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    source       VARCHAR(50)  NOT NULL DEFAULT 'bitext',
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE category_intents (
    id              SERIAL PRIMARY KEY,
    category_name   VARCHAR(100) NOT NULL REFERENCES intent_categories (name) ON DELETE CASCADE,
    intent_name     VARCHAR(200) NOT NULL,
    display_name    VARCHAR(200) NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (category_name, intent_name)
);

CREATE INDEX idx_category_intents_category ON category_intents (category_name);

-- ---------------------------------------------------------------------------
-- Chat sessions + messages (aligns with infra/postgres/init.sql + issue state)
-- ---------------------------------------------------------------------------

CREATE TABLE sessions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            VARCHAR(100),
    company_id         VARCHAR(100),
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW(),
    intent             VARCHAR(50),
    escalated          BOOLEAN DEFAULT FALSE,
    resolved_at        TIMESTAMPTZ,
    user_request       TEXT,
    problem_to_solve   TEXT,
    issue_category     VARCHAR(100),
    issue_confidence   DOUBLE PRECISION,
    csat_score         SMALLINT,
    nps_score          SMALLINT
);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    role        VARCHAR(20) NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Support ticket records linked to a chat session (distinct from support_tickets above)
CREATE TABLE tickets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    issue_type  VARCHAR(100),
    summary     TEXT,
    status      VARCHAR(50) DEFAULT 'open',
    priority    VARCHAR(20) DEFAULT 'normal',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE agent_spans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    trace_id VARCHAR(100),
    span_name VARCHAR(100) NOT NULL,
    attributes JSONB,
    latency_ms NUMERIC(12, 3),
    "timestamp" TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    task VARCHAR(100) NOT NULL,
    completed BOOLEAN NOT NULL,
    escalated BOOLEAN NOT NULL DEFAULT FALSE,
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    agent_state_json JSONB,
    stage_metadata_json JSONB,
    output_validation_json JSONB,
    context_summary_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE evaluation_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    groundedness DOUBLE PRECISION,
    hallucination BOOLEAN,
    helpfulness DOUBLE PRECISION,
    metadata JSONB,
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Spec-aligned operational + simulator testing tables
-- ---------------------------------------------------------------------------

CREATE TABLE procedure_blueprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id VARCHAR(150) NOT NULL,
    category VARCHAR(100) NOT NULL,
    intent VARCHAR(200) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    blueprint_json JSONB NOT NULL,
    metadata JSONB,
    created_by VARCHAR(120),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (procedure_id, version)
);

CREATE TABLE escalation_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    ticket_id UUID REFERENCES tickets(id) ON DELETE SET NULL,
    procedure_id VARCHAR(150),
    outcome_status VARCHAR(80) NOT NULL,
    queue_status VARCHAR(40) NOT NULL DEFAULT 'queued',
    reason TEXT,
    escalation_bundle JSONB NOT NULL,
    queued_at TIMESTAMPTZ DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    assigned_to VARCHAR(120),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE session_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    order_id VARCHAR(50) REFERENCES orders(order_id) ON DELETE SET NULL,
    subscription_email VARCHAR(255) REFERENCES subscription_accounts(account_email) ON DELETE SET NULL,
    relation VARCHAR(50) NOT NULL DEFAULT 'primary',
    confidence DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tool_invocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    span_id UUID REFERENCES agent_spans(id) ON DELETE SET NULL,
    trace_id VARCHAR(100),
    run_id VARCHAR(120),
    tool_name VARCHAR(120) NOT NULL,
    step_id VARCHAR(120),
    procedure_id VARCHAR(150),
    status VARCHAR(30) NOT NULL DEFAULT 'success',
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_code VARCHAR(80),
    error_message TEXT,
    request_payload JSONB,
    response_payload JSONB,
    duration_ms NUMERIC(12, 3),
    invoked_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE llm_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    span_id UUID REFERENCES agent_spans(id) ON DELETE SET NULL,
    run_id VARCHAR(120),
    model_name VARCHAR(120) NOT NULL,
    stage_name VARCHAR(120),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    finish_reason VARCHAR(80),
    estimated_cost_usd NUMERIC(12, 6),
    latency_ms NUMERIC(12, 3),
    metadata JSONB,
    measured_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    actor_type VARCHAR(40) NOT NULL,
    actor_id VARCHAR(120),
    action VARCHAR(120) NOT NULL,
    entity_type VARCHAR(80) NOT NULL,
    entity_id VARCHAR(120) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    reason TEXT,
    before_json JSONB,
    after_json JSONB,
    metadata JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE simulation_runs (
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
    summary_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE simulation_scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    seed_id VARCHAR(150) NOT NULL,
    persona_id VARCHAR(120),
    category VARCHAR(100) NOT NULL,
    intent VARCHAR(200) NOT NULL,
    linked_order_id VARCHAR(50) REFERENCES orders(order_id) ON DELETE SET NULL,
    linked_user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    linked_subscription_email VARCHAR(255) REFERENCES subscription_accounts(account_email) ON DELETE SET NULL,
    expected_outcome VARCHAR(80),
    actual_outcome VARCHAR(80),
    passed BOOLEAN,
    assertions_json JSONB,
    trace_json JSONB,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE coverage_snapshots (
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

CREATE INDEX idx_agent_spans_session_id ON agent_spans (session_id);
CREATE INDEX idx_agent_spans_timestamp ON agent_spans ("timestamp");
CREATE INDEX idx_outcomes_session_id ON outcomes (session_id);
CREATE INDEX idx_outcomes_created_at ON outcomes (created_at);
CREATE INDEX idx_eval_scores_session_id ON evaluation_scores (session_id);
CREATE INDEX idx_eval_scores_evaluated_at ON evaluation_scores (evaluated_at);
CREATE INDEX idx_procedure_blueprints_lookup ON procedure_blueprints (category, intent, is_active, version DESC);
CREATE INDEX idx_escalation_handoffs_status ON escalation_handoffs (queue_status, queued_at DESC);
CREATE INDEX idx_escalation_handoffs_session ON escalation_handoffs (session_id);
CREATE INDEX idx_session_entities_session ON session_entities (session_id, relation);
CREATE INDEX idx_tool_invocations_session ON tool_invocations (session_id, invoked_at DESC);
CREATE INDEX idx_tool_invocations_run ON tool_invocations (run_id, invoked_at DESC);
CREATE INDEX idx_tool_invocations_tool ON tool_invocations (tool_name, success);
CREATE INDEX idx_llm_metrics_session ON llm_metrics (session_id, measured_at DESC);
CREATE INDEX idx_llm_metrics_run ON llm_metrics (run_id, measured_at DESC);
CREATE INDEX idx_audit_log_session ON audit_log (session_id, occurred_at DESC);
CREATE INDEX idx_audit_log_entity ON audit_log (entity_type, entity_id, occurred_at DESC);
CREATE INDEX idx_simulation_runs_status ON simulation_runs (status, started_at DESC);
CREATE INDEX idx_simulation_scenarios_run ON simulation_scenarios (run_id, category, intent);
CREATE INDEX idx_coverage_snapshots_run ON coverage_snapshots (run_id);

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

CREATE OR REPLACE VIEW v_handoff_queue_status AS
SELECT
  queue_status,
  COUNT(*)::INT AS handoff_count
FROM escalation_handoffs
GROUP BY queue_status
ORDER BY queue_status;

CREATE OR REPLACE VIEW v_llm_performance_summary AS
SELECT
  model_name,
  stage_name,
  COUNT(*)::INT AS call_count,
  ROUND(COALESCE(AVG(latency_ms), 0)::NUMERIC, 3) AS avg_latency_ms,
  ROUND(COALESCE(SUM(total_tokens), 0)::NUMERIC, 0) AS total_tokens,
  ROUND(COALESCE(SUM(estimated_cost_usd), 0)::NUMERIC, 6) AS total_estimated_cost_usd
FROM llm_metrics
GROUP BY model_name, stage_name
ORDER BY model_name, stage_name;

CREATE OR REPLACE VIEW v_simulation_run_summary AS
SELECT
  sr.run_id,
  sr.suite_name,
  sr.status,
  COUNT(ss.id)::INT AS scenario_count,
  COALESCE(COUNT(ss.id) FILTER (WHERE ss.passed = TRUE), 0)::INT AS passed_count,
  COALESCE(COUNT(ss.id) FILTER (WHERE ss.actual_outcome = 'escalated'), 0)::INT AS escalated_count,
  ROUND(
    COALESCE(
      COUNT(ss.id) FILTER (WHERE ss.passed = TRUE)::NUMERIC
      / NULLIF(COUNT(ss.id), 0),
      0
    ),
    4
  ) AS pass_rate
FROM simulation_runs sr
LEFT JOIN simulation_scenarios ss ON ss.run_id = sr.id
GROUP BY sr.id, sr.run_id, sr.suite_name, sr.status
ORDER BY sr.started_at DESC;

CREATE OR REPLACE VIEW v_simulation_outcome_breakdown AS
SELECT
  sr.run_id,
  ss.category,
  ss.intent,
  ss.actual_outcome,
  COUNT(*)::INT AS scenario_count
FROM simulation_scenarios ss
JOIN simulation_runs sr ON sr.id = ss.run_id
GROUP BY sr.run_id, ss.category, ss.intent, ss.actual_outcome
ORDER BY sr.run_id, ss.category, ss.intent, ss.actual_outcome;

COMMIT;
