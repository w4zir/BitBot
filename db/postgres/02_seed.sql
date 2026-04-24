-- BitBot: dummy data for Postgres (aligns with backend/config/issue_required_fields.json + policy/RAG test scenarios)
-- Run after: 01_schema.sql
-- Idempotent: safe to run multiple times (UPSERTs on primary / natural keys).

BEGIN;

-- Users (procedure/tool scenarios: tiers, suspended, multi-order personas)
INSERT INTO users (user_id, email, status, created_at) VALUES
(1, 'silver_user@example.com', 'active', '2025-01-01'),
(2, 'gold_user@example.com', 'active', '2024-06-12'),
(3, 'suspended_user@example.com', 'suspended', '2025-11-20'),
(4, 'electronics_buyer@example.com', 'active', '2026-01-05'),
(5, 'subscription_holder@example.com', 'active', '2026-02-10'),
(6, 'qa_flows@example.com', 'active', '2026-03-01'),
(7, 'refund_scenarios@example.com', 'active', '2026-03-15'),
(8, 'product_hunter@example.com', 'active', '2026-04-01')
ON CONFLICT (user_id) DO UPDATE SET
  email = EXCLUDED.email,
  status = EXCLUDED.status,
  created_at = EXCLUDED.created_at;

-- Loyalty (Silver vs Gold tier scenarios + extra tier)
INSERT INTO loyalty_accounts (user_id, annual_spend, tier, benefits_json) VALUES
(1, 1200.00, 'Silver', '{"discount": 0.05, "free_shipping_tier": "standard"}'::jsonb),
(2, 2100.00, 'Gold', '{"discount": 0.10, "priority_support": true, "free_shipping_tier": "expedited"}'::jsonb),
(6, 800.00, 'Bronze', '{"discount": 0.02, "free_shipping_tier": "standard"}'::jsonb)
ON CONFLICT (user_id) DO UPDATE SET
  annual_spend = EXCLUDED.annual_spend,
  tier = EXCLUDED.tier,
  benefits_json = EXCLUDED.benefits_json;

-- Subscriptions (48h refund window tests: <48h vs >48h since last charge)
INSERT INTO subscription_accounts (account_email, plan, next_renewal_at, last_charge_at, subscription_status) VALUES
('sub_24h@example.com', 'Premium', '2026-05-14 09:00:00', '2026-04-14 09:00:00', 'active'),
('sub_72h@example.com', 'Basic', '2026-05-12 09:00:00', '2026-04-12 09:00:00', 'active'),
('subscription_holder@example.com', 'Plus', '2026-06-01 09:00:00', '2026-04-10 09:00:00', 'active')
ON CONFLICT (account_email) DO UPDATE SET
  plan = EXCLUDED.plan,
  next_renewal_at = EXCLUDED.next_renewal_at,
  last_charge_at = EXCLUDED.last_charge_at,
  subscription_status = EXCLUDED.subscription_status;

-- Orders: statuses for order_status / cancel / change_address / refund_context / get_refund
-- ORD-1001 delivered; ORD-1002 delivered electronics edge; ORD-1003 shipped; ORD-1004/1005 processing;
-- ORD-1006 cancelled; ORD-1007 delivered (address change blocked); ORD-1008 processing (address ok, refund approved);
-- ORD-1009 shipped; ORD-1010 processing low amount; ORD-1011 delivered no refund_requests row
INSERT INTO orders (
    order_id, user_id, order_date, status, total_amount,
    shipping_address_line, shipping_city, shipping_postal_code, shipping_country
) VALUES
('ORD-1001', 1, '2026-04-10 10:00:00', 'delivered', 150.00, '10 Market St', 'San Francisco', '94105', 'US'),
('ORD-1002', 4, '2026-03-26 09:00:00', 'delivered', 899.99, '220 Harbor Ave', 'Seattle', '98101', 'US'),
('ORD-1003', 2, '2026-04-14 14:00:00', 'shipped', 45.00, '8 Rose Lane', 'Austin', '73301', 'US'),
('ORD-1004', 1, '2026-04-15 08:00:00', 'processing', 25.00, '16 Pine Rd', 'Denver', '80014', 'US'),
('ORD-1005', 1, '2026-04-15 09:00:00', 'processing', 100.00, '99 Elm Street', 'Boston', '02108', 'US'),
('ORD-1006', 6, '2026-04-01 11:00:00', 'cancelled', 75.00, '3 Sunset Blvd', 'Phoenix', '85001', 'US'),
('ORD-1007', 6, '2026-04-08 12:00:00', 'delivered', 200.00, '41 River Way', 'Chicago', '60601', 'US'),
('ORD-1008', 7, '2026-04-16 09:00:00', 'processing', 49.99, '12 Birch Ave', 'Portland', '97201', 'US'),
('ORD-1009', 7, '2026-04-15 10:00:00', 'shipped', 30.00, '77 Lake Drive', 'Miami', '33101', 'US'),
('ORD-1010', 8, '2026-04-16 14:00:00', 'processing', 15.00, '5 Hill St', 'Dallas', '75001', 'US'),
('ORD-1011', 8, '2026-04-05 10:00:00', 'delivered', 120.00, '1 Cedar Court', 'New York', '10001', 'US')
ON CONFLICT (order_id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  order_date = EXCLUDED.order_date,
  status = EXCLUDED.status,
  total_amount = EXCLUDED.total_amount,
  shipping_address_line = EXCLUDED.shipping_address_line,
  shipping_city = EXCLUDED.shipping_city,
  shipping_postal_code = EXCLUDED.shipping_postal_code,
  shipping_country = EXCLUDED.shipping_country;

-- Order line items (names align with products catalog where applicable for get_product_info)
INSERT INTO order_items (item_id, order_id, item_name, category, is_opened, qty, price) VALUES
(1, 'ORD-1001', 'Smart Toaster', 'appliances', false, 1, 150.00),
(2, 'ORD-1002', 'High-End Laptop', 'electronics', true, 1, 899.99),
(3, 'ORD-1003', 'Coffee Beans', 'food', false, 2, 22.50),
(4, 'ORD-1005', 'USB Cable', 'electronics', false, 1, 100.00),
(5, 'ORD-1006', 'Cancelled Item Mix', 'general', false, 1, 75.00),
(6, 'ORD-1007', 'Desk Lamp', 'home', false, 1, 200.00),
(7, 'ORD-1008', 'Bluetooth Speaker', 'electronics', false, 1, 49.99),
(8, 'ORD-1009', 'Phone Case', 'accessories', false, 1, 30.00),
(9, 'ORD-1010', 'Sticker Pack', 'accessories', false, 1, 15.00),
(10, 'ORD-1011', 'Notebook Set', 'stationery', false, 1, 120.00)
ON CONFLICT (item_id) DO UPDATE SET
  order_id = EXCLUDED.order_id,
  item_name = EXCLUDED.item_name,
  category = EXCLUDED.category,
  is_opened = EXCLUDED.is_opened,
  qty = EXCLUDED.qty,
  price = EXCLUDED.price;

-- Product catalog (product_catalog_lookup / get_product_info: exact vs partial ILIKE, in stock vs OOS)
INSERT INTO products (product_id, sku, name, price, is_available, metadata) VALUES
(1, 'SKU-TOASTER', 'Smart Toaster', 149.99, true, '{"category": "appliances"}'::jsonb),
(2, 'SKU-LAPTOP', 'High-End Laptop', 899.99, true, '{"category": "electronics"}'::jsonb),
(3, 'SKU-BEANS', 'Coffee Beans', 11.25, true, '{"category": "food"}'::jsonb),
(4, 'SKU-WIDGET', 'Widget Pro', 19.99, true, '{"category": "general"}'::jsonb),
(5, 'SKU-OOS-CLOCK', 'Vintage Clock', 45.00, false, '{"category": "home", "note": "out_of_stock_demo"}'::jsonb),
(6, 'SKU-SPEAKER', 'Bluetooth Speaker', 49.99, true, '{"category": "electronics"}'::jsonb)
ON CONFLICT (product_id) DO UPDATE SET
  sku = EXCLUDED.sku,
  name = EXCLUDED.name,
  price = EXCLUDED.price,
  is_available = EXCLUDED.is_available,
  metadata = EXCLUDED.metadata;

-- Payments (TXN-* format for validation)
INSERT INTO payments (transaction_id, order_id, amount, method, payment_status, charged_at) VALUES
('TXN-9001', 'ORD-1001', 150.00, 'credit_card', 'captured', '2026-04-10 10:05:00'),
('TXN-9002', 'ORD-1003', 45.00, 'credit_card', 'captured', '2026-04-14 14:10:00'),
('TXN-9003', 'ORD-1005', 100.00, 'credit_card', 'captured', '2026-04-15 09:05:00'),
('TXN-9004', 'ORD-1004', 25.00, 'paypal', 'captured', '2026-04-15 08:05:00'),
('TXN-9005', 'ORD-1006', 75.00, 'credit_card', 'refunded', '2026-04-01 11:10:00'),
('TXN-9006', 'ORD-1007', 200.00, 'credit_card', 'captured', '2026-04-08 12:05:00'),
('TXN-9007', 'ORD-1008', 49.99, 'credit_card', 'captured', '2026-04-16 09:05:00'),
('TXN-9008', 'ORD-1009', 30.00, 'credit_card', 'captured', '2026-04-15 10:05:00'),
('TXN-9009', 'ORD-1010', 15.00, 'paypal', 'captured', '2026-04-16 14:05:00'),
('TXN-9010', 'ORD-1011', 120.00, 'credit_card', 'captured', '2026-04-05 10:05:00')
ON CONFLICT (transaction_id) DO UPDATE SET
  order_id = EXCLUDED.order_id,
  amount = EXCLUDED.amount,
  method = EXCLUDED.method,
  payment_status = EXCLUDED.payment_status,
  charged_at = EXCLUDED.charged_at;

-- Shipments (weather force majeure vs carrier error; extra delivered/shipped rows)
INSERT INTO shipments (tracking_id, order_id, shipping_tier, promised_delivery_at, actual_delivery_at, delay_reason) VALUES
('TRK-WEATHER', 'ORD-1001', 'priority', '2026-04-12 18:00:00', '2026-04-14 10:00:00', 'blizzard'),
('TRK-ERROR', 'ORD-1003', 'standard', '2026-04-14 18:00:00', NULL, 'carrier_error'),
('TRK-DEL-1007', 'ORD-1007', 'standard', '2026-04-10 18:00:00', '2026-04-08 14:00:00', NULL),
('TRK-SHIP-1009', 'ORD-1009', 'standard', '2026-04-18 18:00:00', NULL, NULL),
('TRK-DEL-1011', 'ORD-1011', 'standard', '2026-04-07 12:00:00', '2026-04-06 09:00:00', NULL)
ON CONFLICT (tracking_id) DO UPDATE SET
  order_id = EXCLUDED.order_id,
  shipping_tier = EXCLUDED.shipping_tier,
  promised_delivery_at = EXCLUDED.promised_delivery_at,
  actual_delivery_at = EXCLUDED.actual_delivery_at,
  delay_reason = EXCLUDED.delay_reason;

-- Invoices (order-linked vs subscription-linked)
INSERT INTO invoices (invoice_id, user_id, order_id, account_email, amount, issued_at, status) VALUES
('INV-ORD-1001', 1, 'ORD-1001', NULL, 150.00, '2026-04-10 10:06:00', 'paid'),
('INV-SUB-24H', 5, NULL, 'sub_24h@example.com', 50.00, '2026-04-14 09:00:00', 'paid')
ON CONFLICT (invoice_id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  order_id = EXCLUDED.order_id,
  account_email = EXCLUDED.account_email,
  amount = EXCLUDED.amount,
  issued_at = EXCLUDED.issued_at,
  status = EXCLUDED.status;

-- Refund requests: denied, pending, approved; ORD-1009/1011 have no row (latest decision null in tool context)
INSERT INTO refund_requests (refund_id, order_id, reason, requested_at, decision, decision_reason) VALUES
(1, 'ORD-1002', 'Do not want item anymore (electronics, opened)', '2026-04-15 10:00:00', 'denied', 'Outside electronics return window and restocking policy applies'),
(2, 'ORD-1001', 'Changed mind', '2026-04-11 12:00:00', 'pending', NULL),
(3, 'ORD-1008', 'Defective speaker', '2026-04-16 11:00:00', 'approved', 'Refund approved per return policy'),
(4, 'ORD-1010', 'Changed mind — low value', '2026-04-16 15:00:00', 'pending', NULL)
ON CONFLICT (refund_id) DO UPDATE SET
  order_id = EXCLUDED.order_id,
  reason = EXCLUDED.reason,
  requested_at = EXCLUDED.requested_at,
  decision = EXCLUDED.decision,
  decision_reason = EXCLUDED.decision_reason;

-- Support tickets: one per issue_type in issue_required_fields (+ security); explicit IDs for stable smoke tests
INSERT INTO support_tickets (ticket_id, issue_type, user_id, payload_json, validation_passed, routing_result) VALUES
(101, 'order', 1, '{"order_id": "ORD-1001", "email": "silver_user@example.com", "item_name": "Smart Toaster"}'::jsonb, true, 'order_queue'),
(102, 'payment', 1, '{"transaction_id": "TXN-9003", "amount": "99.99"}'::jsonb, false, 'payment_mismatch_review'),
(103, 'account', 3, '{"email": "suspended_user@example.com"}'::jsonb, true, 'account_recovery'),
(104, 'delivery', 1, '{"order_or_tracking": "TRK-WEATHER", "issue_summary": "Late due to weather"}'::jsonb, true, 'delivery_policy'),
(105, 'shipping', 2, '{"order_or_tracking": "ORD-1003"}'::jsonb, true, 'shipping_inquiry'),
(106, 'security', 1, '{"raw_message": "User provided CC: 4111222233334444", "cvv": "123"}'::jsonb, true, 'fraud_escalation'),
(107, 'refund', 4, '{"order_id": "ORD-1002", "reason": "Return laptop after 20 days"}'::jsonb, true, 'refund_policy'),
(108, 'cancel', 2, '{"order_id": "ORD-1003"}'::jsonb, false, 'cannot_cancel_shipped'),
(109, 'contact', 1, '{"reason": "General partnership inquiry"}'::jsonb, true, 'contact_routing'),
(110, 'feedback', 2, '{"feedback": "Love the coffee beans packaging."}'::jsonb, true, 'feedback_bucket'),
(111, 'invoice', 5, '{"invoice_id": "INV-SUB-24H"}'::jsonb, true, 'billing'),
(112, 'subscription', 5, '{"account_email": "subscription_holder@example.com", "subscription_issue": "Billing question after renewal"}'::jsonb, true, 'subscription_support')
ON CONFLICT (ticket_id) DO UPDATE SET
  issue_type = EXCLUDED.issue_type,
  user_id = EXCLUDED.user_id,
  payload_json = EXCLUDED.payload_json,
  validation_passed = EXCLUDED.validation_passed,
  routing_result = EXCLUDED.routing_result;

INSERT INTO security_incidents (incident_id, ticket_id, pii_type, redacted, escalated_to, status) VALUES
(1, 106, 'PAN/CVV', true, 'Fraud_Dept_Tier_2', 'open')
ON CONFLICT (incident_id) DO UPDATE SET
  ticket_id = EXCLUDED.ticket_id,
  pii_type = EXCLUDED.pii_type,
  redacted = EXCLUDED.redacted,
  escalated_to = EXCLUDED.escalated_to,
  status = EXCLUDED.status;

-- ---------------------------------------------------------------------------
-- Chat sessions + messages (session-aware issue state; aligns with app / infra)
-- ---------------------------------------------------------------------------

-- Unresolved: locked order_status, awaiting details / tool completion
INSERT INTO sessions (
    id, user_id, company_id,
    created_at, updated_at,
    intent, escalated, resolved_at,
    user_request, issue_category, issue_confidence,
    csat_score, nps_score
) VALUES (
    '11111111-1111-4111-8111-111111111101',
    'silver_user@example.com',
    'demo',
    '2026-04-17 09:00:00+00',
    '2026-04-17 09:01:00+00',
    'order_status',
    false,
    NULL,
    'What is the status of my order ORD-1001?',
    'order',
    0.88,
    NULL,
    NULL
)
ON CONFLICT (id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  company_id = EXCLUDED.company_id,
  updated_at = EXCLUDED.updated_at,
  intent = EXCLUDED.intent,
  escalated = EXCLUDED.escalated,
  resolved_at = EXCLUDED.resolved_at,
  user_request = EXCLUDED.user_request,
  issue_category = EXCLUDED.issue_category,
  issue_confidence = EXCLUDED.issue_confidence,
  csat_score = EXCLUDED.csat_score,
  nps_score = EXCLUDED.nps_score;

INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES
(
    '21111111-1111-4111-8111-111111111101',
    '11111111-1111-4111-8111-111111111101',
    'user',
    'What is the status of my order ORD-1001?',
    '{"source": "user"}'::jsonb,
    '2026-04-17 09:00:00+00'
),
(
    '21111111-1111-4111-8111-111111111102',
    '11111111-1111-4111-8111-111111111101',
    'assistant',
    'I can look that up. If anything is missing, share your order number again.',
    '{"category": "order", "intent": "order_status", "procedure_id": "order_status", "confidence": 0.88, "validation_ok": false, "validation_missing": ["email"]}'::jsonb,
    '2026-04-17 09:01:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  role = EXCLUDED.role,
  content = EXCLUDED.content,
  metadata = EXCLUDED.metadata,
  created_at = EXCLUDED.created_at;

-- Resolved: small talk completed; resolved_at set
INSERT INTO sessions (
    id, user_id, company_id,
    created_at, updated_at,
    intent, escalated, resolved_at,
    user_request, issue_category, issue_confidence,
    csat_score, nps_score
) VALUES (
    '22222222-2222-4222-8222-222222222202',
    'gold_user@example.com',
    'demo',
    '2026-04-16 14:00:00+00',
    '2026-04-16 14:02:00+00',
    'no_issue_chat',
    false,
    '2026-04-16 14:02:00+00',
    'Thanks, that is all I needed!',
    'no_issue',
    0.99,
    5,
    9
)
ON CONFLICT (id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  company_id = EXCLUDED.company_id,
  updated_at = EXCLUDED.updated_at,
  intent = EXCLUDED.intent,
  escalated = EXCLUDED.escalated,
  resolved_at = EXCLUDED.resolved_at,
  user_request = EXCLUDED.user_request,
  issue_category = EXCLUDED.issue_category,
  issue_confidence = EXCLUDED.issue_confidence,
  csat_score = EXCLUDED.csat_score,
  nps_score = EXCLUDED.nps_score;

INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES
(
    '22221111-2222-4222-8222-222222222201',
    '22222222-2222-4222-8222-222222222202',
    'user',
    'Thanks, that is all I needed!',
    '{"source": "user"}'::jsonb,
    '2026-04-16 14:01:00+00'
),
(
    '22221111-2222-4222-8222-222222222202',
    '22222222-2222-4222-8222-222222222202',
    'assistant',
    'Happy to help anytime!',
    '{"category": "no_issue", "intent": "no_issue_chat", "procedure_id": "no_issue_chat", "confidence": 0.99, "validation_ok": true, "validation_missing": []}'::jsonb,
    '2026-04-16 14:02:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  role = EXCLUDED.role,
  content = EXCLUDED.content,
  metadata = EXCLUDED.metadata,
  created_at = EXCLUDED.created_at;

-- Escalated: refund path pending human approval (unresolved issue)
INSERT INTO sessions (
    id, user_id, company_id,
    created_at, updated_at,
    intent, escalated, resolved_at,
    user_request, issue_category, issue_confidence,
    csat_score, nps_score
) VALUES (
    '33333333-3333-4333-8333-333333333303',
    'refund_scenarios@example.com',
    'demo',
    '2026-04-15 11:00:00+00',
    '2026-04-15 11:05:00+00',
    'get_refund',
    true,
    NULL,
    'I want a refund for ORD-1008 — defective speaker.',
    'refund',
    0.91,
    NULL,
    NULL
)
ON CONFLICT (id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  company_id = EXCLUDED.company_id,
  updated_at = EXCLUDED.updated_at,
  intent = EXCLUDED.intent,
  escalated = EXCLUDED.escalated,
  resolved_at = EXCLUDED.resolved_at,
  user_request = EXCLUDED.user_request,
  issue_category = EXCLUDED.issue_category,
  issue_confidence = EXCLUDED.issue_confidence,
  csat_score = EXCLUDED.csat_score,
  nps_score = EXCLUDED.nps_score;

INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES
(
    '33331111-3333-4333-8333-333333333301',
    '33333333-3333-4333-8333-333333333303',
    'user',
    'I want a refund for ORD-1008 — defective speaker.',
    '{"source": "user"}'::jsonb,
    '2026-04-15 11:00:00+00'
),
(
    '33331111-3333-4333-8333-333333333302',
    '33333333-3333-4333-8333-333333333303',
    'assistant',
    'A specialist needs to review this refund. Please confirm if you want escalation.',
    '{"category": "refund", "intent": "get_refund", "procedure_id": "get_refund", "confidence": 0.91, "validation_ok": true, "pending_human_action": true, "action_type": "refund_escalation", "action_id": "act-seed-1008"}'::jsonb,
    '2026-04-15 11:05:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  role = EXCLUDED.role,
  content = EXCLUDED.content,
  metadata = EXCLUDED.metadata,
  created_at = EXCLUDED.created_at;

-- Session-linked support ticket (UUID tickets table)
INSERT INTO tickets (id, session_id, issue_type, summary, status, priority, created_at) VALUES
(
    '44444444-4444-4444-8444-444444444404',
    '33333333-3333-4333-8333-333333333303',
    'refund',
    'Refund escalation for ORD-1008 (defective speaker)',
    'open',
    'high',
    '2026-04-15 11:05:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  issue_type = EXCLUDED.issue_type,
  summary = EXCLUDED.summary,
  status = EXCLUDED.status,
  priority = EXCLUDED.priority,
  created_at = EXCLUDED.created_at;

-- Observability samples (populate analytics views)
INSERT INTO agent_spans (id, session_id, trace_id, span_name, attributes, latency_ms, "timestamp", created_at) VALUES
(
    '55555555-5555-4555-8555-555555555501',
    '11111111-1111-4111-8111-111111111101',
    'trace-seed-1',
    'execute_tool',
    '{"tool": "check_order_status", "success": true}'::jsonb,
    42.5,
    '2026-04-17 09:01:00+00',
    '2026-04-17 09:01:00+00'
),
(
    '55555555-5555-4555-8555-555555555502',
    '22222222-2222-4222-8222-222222222202',
    'trace-seed-2',
    'classify_category',
    '{"model": "bento"}'::jsonb,
    12.0,
    '2026-04-16 14:01:00+00',
    '2026-04-16 14:01:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  trace_id = EXCLUDED.trace_id,
  span_name = EXCLUDED.span_name,
  attributes = EXCLUDED.attributes,
  latency_ms = EXCLUDED.latency_ms,
  "timestamp" = EXCLUDED."timestamp",
  created_at = EXCLUDED.created_at;

INSERT INTO outcomes (id, session_id, task, completed, escalated, verified, created_at) VALUES
(
    '66666666-6666-4666-8666-666666666601',
    '11111111-1111-4111-8111-111111111101',
    'order_status',
    false,
    false,
    false,
    '2026-04-17 09:01:00+00'
),
(
    '66666666-6666-4666-8666-666666666602',
    '22222222-2222-4222-8222-222222222202',
    'no_issue_chat',
    true,
    false,
    true,
    '2026-04-16 14:02:00+00'
),
(
    '66666666-6666-4666-8666-666666666603',
    '33333333-3333-4333-8333-333333333303',
    'get_refund',
    false,
    true,
    false,
    '2026-04-15 11:05:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  task = EXCLUDED.task,
  completed = EXCLUDED.completed,
  escalated = EXCLUDED.escalated,
  verified = EXCLUDED.verified,
  created_at = EXCLUDED.created_at;

INSERT INTO evaluation_scores (id, session_id, groundedness, hallucination, helpfulness, metadata, evaluated_at) VALUES
(
    '77777777-7777-4777-8777-777777777701',
    '22222222-2222-4222-8222-222222222202',
    0.92,
    false,
    0.88,
    '{"evaluator": "seed"}'::jsonb,
    '2026-04-16 14:02:00+00'
),
(
    '77777777-7777-4777-8777-777777777702',
    '33333333-3333-4333-8333-333333333303',
    0.75,
    false,
    0.70,
    '{"evaluator": "seed"}'::jsonb,
    '2026-04-15 11:05:00+00'
)
ON CONFLICT (id) DO UPDATE SET
  session_id = EXCLUDED.session_id,
  groundedness = EXCLUDED.groundedness,
  hallucination = EXCLUDED.hallucination,
  helpfulness = EXCLUDED.helpfulness,
  metadata = EXCLUDED.metadata,
  evaluated_at = EXCLUDED.evaluated_at;

-- Intent taxonomy: Bitext categories + custom no_issue / product (aligns with training/data/bitext_category + plan)
INSERT INTO intent_categories (name, display_name, source, is_active) VALUES
('account', 'Account', 'bitext', true),
('cancel', 'Cancel', 'bitext', true),
('contact', 'Contact', 'bitext', true),
('delivery', 'Delivery', 'bitext', true),
('feedback', 'Feedback', 'bitext', true),
('invoice', 'Invoice', 'bitext', true),
('order', 'Order', 'bitext', true),
('payment', 'Payment', 'bitext', true),
('refund', 'Refund', 'bitext', true),
('shipping', 'Shipping', 'bitext', true),
('subscription', 'Subscription', 'bitext', true),
('no_issue', 'No issue', 'custom', true),
('product', 'Product', 'custom', true)
ON CONFLICT (name) DO UPDATE SET
  display_name = EXCLUDED.display_name,
  source = EXCLUDED.source,
  is_active = EXCLUDED.is_active;

INSERT INTO category_intents (category_name, intent_name, display_name, is_active) VALUES
-- account
('account', 'create_account', 'Create account', true),
('account', 'delete_account', 'Delete account', true),
('account', 'edit_account', 'Edit account', true),
('account', 'recover_password', 'Recover password', true),
('account', 'registration_problems', 'Registration problems', true),
('account', 'switch_account', 'Switch account', true),
('account', 'verify_contact_info', 'Verify contact info', true),
-- cancel
('cancel', 'cancel_order', 'Cancel order', true),
('cancel', 'change_order', 'Change order', true),
('cancel', 'check_cancellation_fee', 'Check cancellation fee', true),
-- contact
('contact', 'contact_customer_service', 'Contact customer service', true),
('contact', 'contact_human_agent', 'Contact human agent', true),
-- delivery
('delivery', 'delivery_options', 'Delivery options', true),
('delivery', 'delivery_period', 'Delivery period', true),
('delivery', 'lost_or_stolen_package', 'Lost or stolen package', true),
('delivery', 'wrong_address_entered', 'Wrong address entered', true),
-- feedback
('feedback', 'complaint', 'Complaint', true),
('feedback', 'review', 'Review', true),
-- invoice
('invoice', 'check_invoice', 'Check invoice', true),
('invoice', 'get_invoice', 'Get invoice', true),
-- order
('order', 'track_order', 'Track order', true),
('order', 'place_order', 'Place order', true),
('order', 'change_order', 'Change order', true),
('order', 'cancel_order', 'Cancel order', true),
-- payment
('payment', 'check_payment_methods', 'Check payment methods', true),
('payment', 'payment_issue', 'Payment issue', true),
('payment', 'track_refund', 'Track refund', true),
-- refund
('refund', 'get_refund', 'Get refund', true),
('refund', 'check_refund_policy', 'Check refund policy', true),
('refund', 'track_refund', 'Track refund', true),
-- shipping
('shipping', 'change_shipping_address', 'Change shipping address', true),
('shipping', 'delivery_options', 'Delivery options', true),
('shipping', 'delivery_period', 'Delivery period', true),
('shipping', 'set_up_shipping_address', 'Set up shipping address', true),
-- subscription
('subscription', 'newsletter_subscription', 'Newsletter subscription', true),
('subscription', 'subscription_status', 'Subscription status', true),
('subscription', 'unsubscribe', 'Unsubscribe', true),
-- no_issue (custom)
('no_issue', 'no_issue', 'No issue', true),
-- product (custom)
('product', 'product_info', 'Product info', true),
('product', 'product_price', 'Product price', true),
('product', 'product_availability', 'Product availability', true)
ON CONFLICT (category_name, intent_name) DO UPDATE SET
  display_name = EXCLUDED.display_name,
  is_active = EXCLUDED.is_active;

COMMIT;

-- Align SERIAL sequences after explicit IDs (safe for subsequent INSERTs without specifying ids)
SELECT setval(pg_get_serial_sequence('users', 'user_id'), COALESCE((SELECT MAX(user_id) FROM users), 1), true);
SELECT setval(pg_get_serial_sequence('order_items', 'item_id'), COALESCE((SELECT MAX(item_id) FROM order_items), 1), true);
SELECT setval(pg_get_serial_sequence('products', 'product_id'), COALESCE((SELECT MAX(product_id) FROM products), 1), true);
SELECT setval(pg_get_serial_sequence('refund_requests', 'refund_id'), COALESCE((SELECT MAX(refund_id) FROM refund_requests), 1), true);
SELECT setval(pg_get_serial_sequence('support_tickets', 'ticket_id'), COALESCE((SELECT MAX(ticket_id) FROM support_tickets), 1), true);
SELECT setval(pg_get_serial_sequence('security_incidents', 'incident_id'), COALESCE((SELECT MAX(incident_id) FROM security_incidents), 1), true);
SELECT setval(pg_get_serial_sequence('intent_categories', 'id'), COALESCE((SELECT MAX(id) FROM intent_categories), 1), true);
SELECT setval(pg_get_serial_sequence('category_intents', 'id'), COALESCE((SELECT MAX(id) FROM category_intents), 1), true);
