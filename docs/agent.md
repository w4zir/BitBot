# BitBot agent architecture

BitBot runs a staged, procedure-driven LangGraph in [`backend/agent/issue_graph.py`](../backend/agent/issue_graph.py). The graph remains deterministic: YAML procedures in [`backend/procedures/`](../backend/procedures/) define execution steps; LLM usage is limited to explicit stages (chitchat, intent extraction, required-data validation, and `llm_response` steps).

This document tracks the implemented architecture and its contracts.

## Pipeline topology

```mermaid
flowchart TD
  startNode([START]) --> classify_category
  classify_category -->|no_issue_or_low_confidence| no_issue_direct
  classify_category -->|issue_category| classify_intent
  no_issue_direct --> endNode([END])
  classify_intent --> specialist_router
  specialist_router --> fetch_procedure
  fetch_procedure --> policy_load
  policy_load --> validate_required
  validate_required -->|needs_more_data_under_limit_non_persistent| endNode
  validate_required -->|needs_more_data_under_limit_persistent| await_user_input
  await_user_input --> validate_required
  validate_required -->|needs_more_data_limit_exceeded| outcome_validator
  validate_required -->|policy_ineligible| outcome_validator
  validate_required -->|gates_pass| structured_executor
  structured_executor -->|steps_remain| structured_executor
  structured_executor -->|done| outcome_validator
  outcome_validator -->|resolved| endNode
  outcome_validator -->|escalate| human_escalation
  human_escalation --> endNode
```

## State model

`IssueGraphState` keeps spec-aligned orchestration fields plus concise runtime snapshots and per-stage debug metadata.

- Session: `text`, `session_id`, `messages`, `issue_locked`
- Classification + intent: `category`, `confidence`, `intent`, `problem_to_solve`
- Routing + procedure: `specialist_agent_id`, `tool_registry_scope`, `procedure_namespace`, `procedure_id`, `todo_list`, `current_step_index`
- Policy + gates: `policy_constraints`, `validation_ok`, `validation_missing`, `eligibility_ok`, `validation_wait_count`, `validation_wait_limit`
- Retry + loop controls: `classify_intent_attempts`, `policy_load_attempts`, `executor_turn_count`, `enable_persistent_wait_interrupt`
- Outcome + handoff: `outcome_status`, `output_validation`, `context_summary`, `escalation_bundle`
- Debug + UI JSON: `agent_state`, `stage_metadata`, `assistant_metadata`, concise `context_data`

## Stage contracts (implemented)

### 1. `classify_category`
- Uses ModernBERT via [`backend/rag/query_classifier.py`](../backend/rag/query_classifier.py) → BentoML `POST CLASSIFIER_BENTOML_URL` with `{"text":"..."}`.
- Returns `category` and `confidence` from the fine-tuned multiclass checkpoint (13 labels including `no_issue`; see [`training/data/label2id.json`](../training/data/label2id.json)).
- Routes to `no_issue_direct` when category is `no_issue` **or** confidence is below `CATEGORY_CONFIDENCE_THRESHOLD` (default `0.5` in `.env.example`).
- On classifier timeout/unavailability: deterministic fallback with `category=no_issue`, `confidence=0.0`, and `assistant_metadata.classifier_error`.
- Honors session lock (reuses locked issue values).

### 2. `no_issue_direct`
- Runs when category is `no_issue` **or** confidence is below `CATEGORY_CONFIDENCE_THRESHOLD`.
- Produces direct assistant reply without procedure execution.

### 3. `classify_intent`
- Produces strict JSON intent/problem summary using category and transcript.
- Optionally constrains intent with Postgres allowlist (`get_intents_for_category`).

### 4. `specialist_router`
- Deterministic routing node that sets specialist/tool namespace metadata.
- No autonomous tool selection occurs here.

### 5. `fetch_procedure`
- Loads procedure blueprint using fallback chain in [`backend/agent/procedures.py`](../backend/agent/procedures.py):
  1) `(category, intent)`  
  2) `(category, *_general)`  
  3) `(unknown, *_general)`

### 6. `policy_load`
- Loads typed YAML artifacts by `(category, intent)` from [`backend/policy_constraints/`](../backend/policy_constraints/).
- Runtime does no policy retrieval and no policy LLM extraction.
- Writes fail-closed constraints when artifact is missing or invalid.
- Emits structured `policy_check_results` and policy artifact metadata (`policy_constraints_path`, `policy_schema_version`).

### 7. `validate_required`
- Runs required-data validation (`validation_ok`, `validation_missing`).
- Applies eligibility gate (`eligibility_ok`) from `policy_constraints`.
- Retries validation model failures up to `AGENT_MAX_NODE_TURNS`.
- Keeps waiting for user data across turns with a bounded counter (`AGENT_VALIDATION_MAX_USER_WAITS`, defaults to `AGENT_MAX_NODE_TURNS`).
- In persistent mode (`backend/agent/persistent_agent.py`), pauses using `await_user_input` interrupt and resumes in the same validation stage on the next turn.
- Routes to `END`, `await_user_input`, `outcome_validator`, or `structured_executor`.

### 8. `structured_executor`
- Deterministic procedure execution loop across step types:
  - `tool_call`
  - `logic_gate`
  - `interrupt`
  - `llm_response`
- Updates `context_data`, `current_step_index`, and optionally `final_response`.
- Evaluates logic gates against both procedure context and loaded policy fields (`value_from` references).
- Records pass/fail reasons for each gate in `policy_check_results` and stage metadata.
- Applies `executor_turn_count` safety cap (`AGENT_MAX_NODE_TURNS`) on self-loop progression.

### 9. `outcome_validator`
- Assigns final `outcome_status` (`resolved`, `needs_more_data`, `policy_ineligible`, `tool_error`, `step_error`, `pending_escalation`, `unresolvable`).
- Verifies execution outcomes against source-of-truth data where needed (example: cancellation is confirmed with DB order status).
- Generates `output_validation` and `context_summary` for follow-up turns and debugging.
- Decides terminal vs escalation routing.

### 9a. `human_escalation`
- Builds `escalation_bundle` from state and marks escalation metadata.
- Ends graph with escalation-ready payload.

## HTTP integration compatibility

[`POST /classify`](../backend/api/routes/classify.py) remains the stable external contract.

- `full_flow=false`: Bento classifier only (no graph invoke).
- `full_flow=true`: session-aware graph invoke with lock semantics.
- Default runtime uses persistent graph execution (`AGENT_PERSISTENT_MODE=1`) via [`backend/agent/persistent_agent.py`](../backend/agent/persistent_agent.py); non-persistent execution remains available.
- Existing response shape is preserved; richer internal outcomes are surfaced through `assistant_metadata` (`agent_state`, `stage_metadata`, `output_validation`, `context_summary`).
- Session resolution behavior remains unchanged (`user_confirms_resolution` short-circuit + `graph_suggests_session_resolved`).

## Procedure compatibility

Procedure schema remains in [`backend/agent/procedures.py`](../backend/agent/procedures.py). Validation enforces duplicate step-id detection, deterministic fallback-chain resolution, policy artifact presence, and expected-outcome assertion schema.

Current procedure-backed intents in `backend/procedures/`:

- `order/order_status`
- `order/cancel_order`
- `refund/get_refund`
- `shipping/change_shipping_address`
- `product/product_info`
- `product/product_price`
- `product/product_availability`
- `payment/payment_issue`
- `payment/check_payment_methods`
- `refund/track_refund`
- `invoice/check_invoice`
- `subscription/subscription_status`
- `subscription/unsubscribe`
- `contact/contact_human_agent`
- `delivery/delivery_period`
- `feedback/complaint`

## Node retry behavior

- `classify_intent`: retries LLM parse/extract loop up to `AGENT_MAX_NODE_TURNS` before conservative fallback.
- `policy_load`: single artifact load attempt per run with fail-closed fallback.
- `validate_required`: retries validation-model exceptions up to `AGENT_MAX_NODE_TURNS`.
- `validate_required` (persistent mode): pauses at `await_user_input` and resumes in-node on next user turn until wait limit is reached.
- `structured_executor`: self-loop exits when `executor_turn_count >= AGENT_MAX_NODE_TURNS` even if blueprint index has not advanced.

## Environment variables

- `AGENT_MAX_NODE_TURNS`: global per-node retry and loop cap (default `20`).
- `AGENT_VALIDATION_MAX_USER_WAITS`: max user wait turns before escalation (defaults to `AGENT_MAX_NODE_TURNS`).
- `AGENT_CHECKPOINT_DB`: sqlite path for LangGraph checkpoint persistence (default `agent_checkpoints.db`).
- `AGENT_PERSISTENT_MODE`: `1/true` enables persistent runner for `/classify full_flow`; `0/false` uses non-persistent graph.
- `POLICY_CONSTRAINTS_DIR`: optional override for policy constraints artifact directory.
- `CLASSIFIER_BENTOML_URL`: ModernBERT Bento classify endpoint (Compose default: `http://modernbert:3000/classify`).
- `CLASSIFIER_BENTOML_TIMEOUT_SECONDS`: HTTP timeout for classifier calls (default `5`).
- `CATEGORY_CONFIDENCE_THRESHOLD`: below this confidence, route to `no_issue_direct` (default `0.5`).
- `MODERNBERT_MODEL_DIR`: active `winner/` checkpoint path inside the Bento container.

## Related routes

- [`backend/api/routes/escalations.py`](../backend/api/routes/escalations.py): `POST /escalations/decision` — accept/reject a pending escalation action for a session.

### Tools API (`/tools`)

All routes are served from [`backend/api/routes/tools.py`](../backend/api/routes/tools.py) under the `/tools` prefix. Request/response shapes use the Pydantic models in that file.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/tools/order-status` | Order status by `order_id` |
| POST | `/tools/product-lookup` | Product search by `product_name` |
| POST | `/tools/product-info` | Product details by `product_name` |
| POST | `/tools/product-price` | Product price by `product_name` |
| POST | `/tools/product-availability` | Availability by `product_name` |
| POST | `/tools/refund-context` | Refund context by `order_id` |
| POST | `/tools/cancel-order` | Cancel order (`order_id`) |
| POST | `/tools/create-refund-request` | Create refund request (`order_id`, `reason`) |
| POST | `/tools/update-shipping-address` | Update shipping address (`order_id`, `new_address`) |
| POST | `/tools/payment` | Payment lookup by `transaction_id` |
| GET | `/tools/payment-methods` | List payment methods (no body) |
| POST | `/tools/payment-track-refund` | Refund tracking by `order_id` |
| POST | `/tools/invoice` | Invoice by `invoice_id` |
| POST | `/tools/subscription-status` | Subscription by `account_email` |
| POST | `/tools/subscription-unsubscribe` | Unsubscribe by `account_email` |
| POST | `/tools/contact-handoff` | Human handoff ticket (`summary`) |
| POST | `/tools/complaint` | Complaint ticket (`complaint`) |
| POST | `/tools/delivery-period` | Delivery window by `order_or_tracking` |

## Related documentation

- [docs/finetuning-modernbert.md](finetuning-modernbert.md) — dataset prep, fine-tuning, Bento serving, evaluation
- [training/README.md](../training/README.md) — phase 1 + 2 training walkthrough
- [testing/README.md](../testing/README.md) — category eval and metric interpretation
- [specs/agent.md](../specs/agent.md) — normative stage contracts
