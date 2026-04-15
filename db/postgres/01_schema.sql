-- BitBot: Postgres schema for local/testing dummy data
-- Rerunnable: drops existing objects in FK-safe order, then recreates.

BEGIN;

-- Children first (reverse dependency order)
DROP TABLE IF EXISTS security_incidents CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS refund_requests CASCADE;
DROP TABLE IF EXISTS invoices CASCADE;
DROP TABLE IF EXISTS shipments CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
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

COMMIT;
