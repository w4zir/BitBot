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
DROP TABLE IF EXISTS evaluation_scores CASCADE;
DROP TABLE IF EXISTS outcomes CASCADE;
DROP TABLE IF EXISTS agent_spans CASCADE;
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
    total_amount DECIMAL(10, 2)
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

CREATE INDEX idx_agent_spans_session_id ON agent_spans (session_id);
CREATE INDEX idx_agent_spans_timestamp ON agent_spans ("timestamp");
CREATE INDEX idx_outcomes_session_id ON outcomes (session_id);
CREATE INDEX idx_outcomes_created_at ON outcomes (created_at);
CREATE INDEX idx_eval_scores_session_id ON evaluation_scores (session_id);
CREATE INDEX idx_eval_scores_evaluated_at ON evaluation_scores (evaluated_at);

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

COMMIT;
