-- BitBot: smoke checks after 01_schema.sql + 02_seed.sql
-- Expect non-empty result sets / counts as noted in comments.

-- Row counts (sanity)
SELECT 'users' AS tbl, COUNT(*)::int AS n FROM users
UNION ALL SELECT 'orders', COUNT(*)::int FROM orders
UNION ALL SELECT 'order_items', COUNT(*)::int FROM order_items
UNION ALL SELECT 'payments', COUNT(*)::int FROM payments
UNION ALL SELECT 'shipments', COUNT(*)::int FROM shipments
UNION ALL SELECT 'subscription_accounts', COUNT(*)::int FROM subscription_accounts
UNION ALL SELECT 'invoices', COUNT(*)::int FROM invoices
UNION ALL SELECT 'refund_requests', COUNT(*)::int FROM refund_requests
UNION ALL SELECT 'products', COUNT(*)::int FROM products
UNION ALL SELECT 'support_tickets', COUNT(*)::int FROM support_tickets
UNION ALL SELECT 'security_incidents', COUNT(*)::int FROM security_incidents
UNION ALL SELECT 'sessions', COUNT(*)::int FROM sessions
UNION ALL SELECT 'messages', COUNT(*)::int FROM messages
UNION ALL SELECT 'tickets', COUNT(*)::int FROM tickets
UNION ALL SELECT 'agent_spans', COUNT(*)::int FROM agent_spans
UNION ALL SELECT 'outcomes', COUNT(*)::int FROM outcomes
UNION ALL SELECT 'evaluation_scores', COUNT(*)::int FROM evaluation_scores;

-- Issue types: one ticket per category from issue_required_fields.json (+ security)
SELECT issue_type, COUNT(*)::int AS tickets
FROM support_tickets
GROUP BY issue_type
ORDER BY issue_type;

-- RAG / policy scenarios
-- Electronics return: laptop order ~20 days ago, opened
SELECT o.order_id, o.order_date, oi.category, oi.is_opened
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_id = 'ORD-1002';

-- Force majeure: priority late, blizzard
SELECT tracking_id, shipping_tier, delay_reason, actual_delivery_at > promised_delivery_at AS late
FROM shipments
WHERE tracking_id = 'TRK-WEATHER';

-- Subscription refund window: <48h vs >48h since last_charge_at (adjust "now" in app tests; data is static timestamps)
SELECT account_email, last_charge_at,
       EXTRACT(EPOCH FROM (TIMESTAMP '2026-04-15 12:00:00' - last_charge_at)) / 3600 AS hours_since_charge_approx
FROM subscription_accounts
WHERE account_email IN ('sub_24h@example.com', 'sub_72h@example.com')
ORDER BY account_email;

-- Loyalty tiers: Silver ($1200) vs Gold ($2100)
SELECT u.email, la.tier, la.annual_spend
FROM loyalty_accounts la
JOIN users u ON u.user_id = la.user_id
ORDER BY la.annual_spend;

-- Security: incident linked to ticket 106
SELECT st.ticket_id, st.issue_type, si.pii_type, si.redacted, si.escalated_to
FROM support_tickets st
JOIN security_incidents si ON si.ticket_id = st.ticket_id
WHERE st.ticket_id = 106;

-- Payment mismatch ticket (validation_passed = false)
SELECT ticket_id, issue_type, validation_passed, payload_json->>'amount' AS claimed_amount
FROM support_tickets
WHERE ticket_id = 102;

-- Invoices: order vs subscription-linked
SELECT invoice_id, order_id, account_email, status FROM invoices ORDER BY invoice_id;

-- Product catalog (get_product_info / product_catalog_lookup)
SELECT sku, name, price, is_available FROM products ORDER BY product_id;

-- Procedure-oriented order samples (order_status / cancel / change address / refund context)
SELECT order_id, status, total_amount FROM orders WHERE order_id IN (
  'ORD-1004', 'ORD-1006', 'ORD-1007', 'ORD-1008', 'ORD-1011'
) ORDER BY order_id;

-- Session-aware issue state (seeded demos)
SELECT id, intent, issue_category, user_request,
       resolved_at IS NOT NULL AS is_resolved,
       escalated
FROM sessions
ORDER BY created_at;

-- Messages per seeded session
SELECT s.id AS session_id, COUNT(m.id)::int AS message_count
FROM sessions s
LEFT JOIN messages m ON m.session_id = s.id
GROUP BY s.id
ORDER BY s.id;

-- Analytics views (non-empty after seed)
SELECT * FROM v_automation_rate;
SELECT * FROM v_escalation_rate;
SELECT * FROM v_tool_success_rate;
SELECT * FROM v_hallucination_rate;

-- Escalated session has ticket + pending_human_action in latest assistant metadata
SELECT s.id, t.summary,
  (
    SELECT m.metadata->>'pending_human_action'
    FROM messages m
    WHERE m.session_id = s.id AND m.role = 'assistant'
    ORDER BY m.created_at DESC NULLS LAST
    LIMIT 1
  ) AS pending_human_action
FROM sessions s
JOIN tickets t ON t.session_id = s.id
WHERE s.escalated = true;
