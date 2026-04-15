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
UNION ALL SELECT 'support_tickets', COUNT(*)::int FROM support_tickets
UNION ALL SELECT 'security_incidents', COUNT(*)::int FROM security_incidents;

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
