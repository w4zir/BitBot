-- BitBot: dummy data for Postgres (aligns with backend/config/issue_required_fields.json + policy/RAG test scenarios)
-- Run after: 01_schema.sql

BEGIN;

-- Users
INSERT INTO users (user_id, email, status, created_at) VALUES
(1, 'silver_user@example.com', 'active', '2025-01-01'),
(2, 'gold_user@example.com', 'active', '2024-06-12'),
(3, 'suspended_user@example.com', 'suspended', '2025-11-20'),
(4, 'electronics_buyer@example.com', 'active', '2026-01-05'),
(5, 'subscription_holder@example.com', 'active', '2026-02-10');

-- Loyalty (Silver vs Gold tier scenarios)
INSERT INTO loyalty_accounts (user_id, annual_spend, tier, benefits_json) VALUES
(1, 1200.00, 'Silver', '{"discount": 0.05, "free_shipping_tier": "standard"}'::jsonb),
(2, 2100.00, 'Gold', '{"discount": 0.10, "priority_support": true, "free_shipping_tier": "expedited"}'::jsonb);

-- Subscriptions (48h refund window tests: <48h vs >48h since last charge)
INSERT INTO subscription_accounts (account_email, plan, next_renewal_at, last_charge_at, subscription_status) VALUES
('sub_24h@example.com', 'Premium', '2026-05-14 09:00:00', '2026-04-14 09:00:00', 'active'),
('sub_72h@example.com', 'Basic', '2026-05-12 09:00:00', '2026-04-12 09:00:00', 'active'),
('subscription_holder@example.com', 'Plus', '2026-06-01 09:00:00', '2026-04-10 09:00:00', 'active');

-- Orders
INSERT INTO orders (order_id, user_id, order_date, status, total_amount) VALUES
('ORD-1001', 1, '2026-04-10 10:00:00', 'delivered', 150.00),   -- Valid standard order + force majeure shipment
('ORD-1002', 4, '2026-03-26 09:00:00', 'delivered', 899.99),  -- ~20 days ago: electronics return conflict
('ORD-1003', 2, '2026-04-14 14:00:00', 'shipped', 45.00),     -- Non-cancellable / carrier_error shipment
('ORD-1004', 1, '2026-04-15 08:00:00', 'processing', 25.00),  -- Cancellable
('ORD-1005', 1, '2026-04-15 09:00:00', 'processing', 100.00); -- Payment amount mismatch test

-- Order line items
INSERT INTO order_items (item_id, order_id, item_name, category, is_opened, qty, price) VALUES
(1, 'ORD-1001', 'Smart Toaster', 'appliances', false, 1, 150.00),
(2, 'ORD-1002', 'High-End Laptop', 'electronics', true, 1, 899.99),
(3, 'ORD-1003', 'Coffee Beans', 'food', false, 2, 22.50),
(4, 'ORD-1005', 'USB Cable', 'electronics', false, 1, 100.00);

-- Payments (TXN-* format for validation)
INSERT INTO payments (transaction_id, order_id, amount, method, payment_status, charged_at) VALUES
('TXN-9001', 'ORD-1001', 150.00, 'credit_card', 'captured', '2026-04-10 10:05:00'),
('TXN-9002', 'ORD-1003', 45.00, 'credit_card', 'captured', '2026-04-14 14:10:00'),
('TXN-9003', 'ORD-1005', 100.00, 'credit_card', 'captured', '2026-04-15 09:05:00'),
('TXN-9004', 'ORD-1004', 25.00, 'paypal', 'captured', '2026-04-15 08:05:00');

-- Shipments (weather force majeure vs carrier error)
INSERT INTO shipments (tracking_id, order_id, shipping_tier, promised_delivery_at, actual_delivery_at, delay_reason) VALUES
('TRK-WEATHER', 'ORD-1001', 'priority', '2026-04-12 18:00:00', '2026-04-14 10:00:00', 'blizzard'),
('TRK-ERROR', 'ORD-1003', 'standard', '2026-04-14 18:00:00', NULL, 'carrier_error');

-- Invoices (order-linked vs subscription-linked)
INSERT INTO invoices (invoice_id, user_id, order_id, account_email, amount, issued_at, status) VALUES
('INV-ORD-1001', 1, 'ORD-1001', NULL, 150.00, '2026-04-10 10:06:00', 'paid'),
('INV-SUB-24H', 5, NULL, 'sub_24h@example.com', 50.00, '2026-04-14 09:00:00', 'paid');

-- Refund requests (policy / electronics edge)
INSERT INTO refund_requests (refund_id, order_id, reason, requested_at, decision, decision_reason) VALUES
(1, 'ORD-1002', 'Do not want item anymore (electronics, opened)', '2026-04-15 10:00:00', 'denied', 'Outside electronics return window and restocking policy applies'),
(2, 'ORD-1001', 'Changed mind', '2026-04-11 12:00:00', 'pending', NULL);

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
(112, 'subscription', 5, '{"account_email": "subscription_holder@example.com", "subscription_issue": "Billing question after renewal"}'::jsonb, true, 'subscription_support');

INSERT INTO security_incidents (incident_id, ticket_id, pii_type, redacted, escalated_to, status) VALUES
(1, 106, 'PAN/CVV', true, 'Fraud_Dept_Tier_2', 'open');

COMMIT;

-- Align SERIAL sequences after explicit IDs (safe for subsequent INSERTs without specifying ids)
SELECT setval(pg_get_serial_sequence('users', 'user_id'), COALESCE((SELECT MAX(user_id) FROM users), 1), true);
SELECT setval(pg_get_serial_sequence('order_items', 'item_id'), COALESCE((SELECT MAX(item_id) FROM order_items), 1), true);
SELECT setval(pg_get_serial_sequence('refund_requests', 'refund_id'), COALESCE((SELECT MAX(refund_id) FROM refund_requests), 1), true);
SELECT setval(pg_get_serial_sequence('support_tickets', 'ticket_id'), COALESCE((SELECT MAX(ticket_id) FROM support_tickets), 1), true);
SELECT setval(pg_get_serial_sequence('security_incidents', 'incident_id'), COALESCE((SELECT MAX(incident_id) FROM security_incidents), 1), true);
