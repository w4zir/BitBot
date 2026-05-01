# BitBot Agent Testing Simulator — Technical Specification

> **Purpose:** This document is the authoritative spec for building the BitBot agentic RAG testing simulator. It is intended for use directly in Cursor as a development reference. All module paths, interface contracts, and config schemas described here should be implemented as written unless noted otherwise.

---

## 1. Overview

The simulator is an end-to-end testing framework that generates realistic customer support scenarios, converses with the BitBot agent over its existing HTTP API, and evaluates whether issues were correctly handled. It is designed to surface regressions across the full LangGraph pipeline — from `classify_category` through `outcome_validator` — not just the final response text.

Current implementation note: live procedure/intent naming follows runtime YAML in `backend/procedures/`
(for example `order/order_status`, `shipping/change_shipping_address`, `product/product_info`), which
supersedes older examples in this document that use legacy naming.

### Design principles

- **DB-grounded scenarios**: every test scenario is hydrated with real entities (orders, users, subscriptions) from the database so that tool calls, eligibility gates, and policy constraints are exercised against actual data, not fabricated IDs.
- **Deterministic seeds, variable instances**: a seed defines the shape of a test (category, intent, difficulty, persona); an instance is the seed hydrated with DB data. Seeds are versioned in git; instances are generated at runtime.
- **Graph-aware evaluation**: the agent response includes `outcome_status`, `procedure_id`, `policy_constraints`, `agent_state`, `stage_metadata`, and related metadata through `assistant_metadata`. Evaluators use these structured signals as primary pass/fail criteria. LLM judges are supplementary.
- **Config-driven runs**: every run is fully specified by a YAML config. No hardcoded scenarios. Configs are diffable, reproducible, and commit-trackable.
- **Coverage-complete by design**: the framework enforces that every supported `(category, intent)` pair from active procedures plus DB intent taxonomy has at least one seed, or is explicitly marked as a known gap.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Runner (CLI)                        │
│              testing/simulator/runner.py                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ reads
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Test Suite Config (YAML)                    │
│              testing/simulator/suites/*.yaml                    │
└──────┬──────────────┬──────────────────┬────────────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌────────────┐ ┌────────────┐ ┌──────────────────┐
│  Scenario  │ │   DB       │ │  Coverage        │
│  Registry  │ │  Hydrator  │ │  Checker         │
│            │ │            │ │                  │
│ seeds/*.yaml│ │ Postgres   │ │ procedures+intents│
└─────┬──────┘ └─────┬──────┘ └──────────────────┘
      │               │
      └───────┬────────┘
              │ produces ScenarioInstance
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Persona Engine                            │
│              testing/simulator/persona.py                       │
│   wraps SimulatorLLM call with persona system prompt + state    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ generates user turn
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Driver                          │
│              testing/simulator/driver.py                        │
│   manages turn loop, POST /classify, session_id, issue_locked   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ raw API response per turn
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Trace Collector                             │
│              testing/simulator/trace.py                         │
│   captures full IssueGraphState fields per turn                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ConversationTrace
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluator Suite                             │
│              testing/simulator/evaluators/                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ structural.py   │ policy.py   │ llm_judge.py │ regression │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ EvaluationResult per scenario
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Reporter                                 │
│              testing/simulator/reporter.py                      │
│   JSON artifact + console summary + regression delta            │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.1 Runtime and Deployment Modes (Updated)

- The simulator must support:
  - deterministic suite runs (`--iterations N`, no randomization),
  - randomized selection runs (`--randomize` with persona/category/intent filters),
  - continuous mode (`--forever`) until interrupted.
- A Docker Compose `simulator` service should run the same CLI entrypoint used locally.
- Simulator output remains artifact-first (`testing/simulator/results/*.json`) and additionally persists run/scenario/turn/evaluation/training rows to Postgres when persistence is enabled.
- Postgres persistence should include token usage (`input`, `output`, `cache`, `total`) and latency fields at least at turn and judge-call level. If provider metadata is unavailable, fields remain nullable.

---

## 3. Directory Structure

```
testing/
└── simulator/
    ├── runner.py                    # CLI entry point
    ├── config.py                    # Pydantic models for all config schemas
    ├── hydrator.py                  # DB-grounded scenario hydration
    ├── persona.py                   # Persona engine
    ├── driver.py                    # Conversation turn loop + HTTP client
    ├── trace.py                     # IssueGraphState trace capture
    ├── reporter.py                  # Run artifact + console output
    ├── coverage.py                  # procedures + intent taxonomy coverage enforcement
    ├── evaluators/
    │   ├── __init__.py
    │   ├── structural.py            # outcome_status, procedure_id assertions
    │   ├── policy.py                # eligibility gate + policy_constraints checks
    │   ├── llm_judge.py             # LLM-as-judge for tone, completeness, hallucination
    │   └── regression.py            # Diff against pinned baseline artifact
    ├── suites/                      # Test suite YAML configs (committed to git)
    │   ├── smoke.yaml
    │   ├── regression.yaml
    │   └── adversarial.yaml
    ├── seeds/                       # Scenario seed definitions
    │   ├── order_cancellation.yaml
    │   ├── refund.yaml
    │   ├── shipping.yaml
    │   └── ...                      # one file per category
    ├── personas/
    │   └── personas.yaml            # Persona definitions
    ├── baselines/                   # Pinned baseline run artifacts (committed)
    │   └── baseline_<run_id>.json
    └── results/                     # Generated run artifacts (gitignored)
        └── run_<timestamp>.json
```

---

## 4. Config Schemas

All configs are validated by Pydantic models in `testing/simulator/config.py`.

### 4.1 Test Suite Config

**File:** `testing/simulator/suites/<suite_name>.yaml`

```yaml
# testing/simulator/suites/regression.yaml

run_id: regression_sprint_42        # used in artifact filenames and reporting
agent_url: http://localhost:8000/classify
db_snapshot: live                   # "live" or path to a SQL fixture file
baseline: baselines/baseline_v1.json  # omit to skip regression diff

defaults:
  max_turns: 6
  cooperation_level: cooperative    # cooperative | passive | resistant
  persist_db: true                  # enabled by default; CLI can override per run
  user_llm_provider: ollama         # dedicated user-message generation LLM
  user_llm_model: llama3.2
  user_llm_timeout_seconds: 120
  eval_targets:                     # which evaluators to run
    - structural
    - policy
    - llm_judge
  llm_judge_provider: ollama
  llm_judge_model: claude-sonnet-4-20250514
  fail_on_regression: false         # config field exists; regression evaluator is not wired
  fail_on_coverage_gap: false

scenarios:
  - seed_id: cancel_order_easy
  - seed_id: cancel_order_hard
  - seed_id: refund_damaged_easy
  - seed_id: refund_damaged_hard
    cooperation_level: resistant    # overrides default for this scenario
  - seed_id: multi_issue_refund_shipping
    eval_targets:                   # override evaluators for this scenario
      - structural
      - llm_judge
```

### 4.2 Scenario Seed

**File:** `testing/simulator/seeds/<category>.yaml`

Each seed is a named, reusable template. A seed is hydrated into a `ScenarioInstance` at runtime.

```yaml
# testing/simulator/seeds/order_cancellation.yaml

seeds:
  - seed_id: cancel_order_easy
    category: order_cancellation
    intent: cancel_order_before_delivery
    difficulty: easy
    persona_id: polite_first_timer
    description: "Recent order well within cancellation window, cooperative user"
    expected_outcome: resolved
    expected_procedure_id: cancel_order_before_delivery   # from YAML procedures
    db_filter:
      entity_type: order              # what to pull from DB: order | user | subscription
      order_status: [pending, confirmed]
      order_age_minutes: [5, 60]      # freshness constraint
    multi_issue: false

  - seed_id: cancel_order_hard
    category: order_cancellation
    intent: cancel_order_before_delivery
    difficulty: hard
    persona_id: impatient_escalator
    cooperation_level: resistant
    description: "Order at policy boundary — restaurant already preparing"
    expected_outcome: policy_ineligible   # or escalate — depends on your procedure logic
    expected_procedure_id: cancel_order_before_delivery
    db_filter:
      entity_type: order
      order_status: [preparing]
      order_age_minutes: [25, 40]
    adversarial_flags:
      - eligibility_boundary           # signals evaluator to check eligibility gate specifically
    multi_issue: false

  - seed_id: cancel_order_boundary_probe
    category: order_cancellation
    intent: cancel_order_before_delivery
    difficulty: hard
    persona_id: policy_prober
    description: "User explicitly argues about policy cutoff to probe eligibility gate"
    expected_outcome: policy_ineligible
    db_filter:
      entity_type: order
      order_status: [preparing]
      order_age_minutes: [28, 32]      # narrow band — right at the edge
    adversarial_flags:
      - eligibility_boundary
      - policy_argument                # persona will verbally contest the policy decision
    multi_issue: false

  - seed_id: multi_issue_cancel_and_refund
    category: order_cancellation
    intent: cancel_order_before_delivery
    difficulty: medium
    persona_id: verbose_multi_tasker
    description: "User starts with cancellation then pivots to a refund on a previous order"
    expected_outcome: resolved
    multi_issue: true
    secondary_issue:
      category: refund
      intent: refund_wrong_item
      db_filter:
        entity_type: order
        order_status: [delivered]
        order_age_days: [1, 7]
    db_filter:
      entity_type: order
      order_status: [pending, confirmed]
      order_age_minutes: [5, 60]
```

### 4.3 Persona Definitions

**File:** `testing/simulator/personas/personas.yaml`

```yaml
personas:
  - persona_id: polite_first_timer
    display_name: "Polite First-Timer"
    vocabulary: simple               # simple | technical | informal
    patience: high                   # high | medium | low
    cooperation_level: cooperative
    escalation_tendency: low
    typical_message_length: medium
    traits:
      - provides requested info promptly
      - accepts policy decisions without pushback
      - thanks the agent

  - persona_id: impatient_escalator
    display_name: "Impatient Escalator"
    vocabulary: informal
    patience: low
    cooperation_level: resistant
    escalation_tendency: high
    typical_message_length: short
    traits:
      - expresses frustration early
      - frequently asks to speak to a human
      - provides partial info and waits for prompting

  - persona_id: verbose_multi_tasker
    display_name: "Verbose Multi-Tasker"
    vocabulary: informal
    patience: medium
    cooperation_level: cooperative
    escalation_tendency: low
    typical_message_length: long
    traits:
      - introduces multiple issues in one message
      - goes off-topic with background context
      - cooperative once the agent asks direct questions

  - persona_id: policy_prober
    display_name: "Policy Prober"
    vocabulary: technical
    patience: medium
    cooperation_level: passive
    escalation_tendency: medium
    typical_message_length: medium
    traits:
      - challenges policy decisions with specific arguments
      - asks about exceptions and edge cases
      - does not accept a policy_ineligible outcome without pushing back at least once

  - persona_id: data_withholder
    display_name: "Data Withholder"
    vocabulary: simple
    patience: medium
    cooperation_level: resistant
    escalation_tendency: low
    typical_message_length: short
    traits:
      - deliberately omits required fields (order_id, account details)
      - provides info only when asked two or more times
      - stresses the validate_required multi-turn retry loop
```

---

## 5. Module Specifications

### 5.1 `hydrator.py` — DB-Grounded Scenario Hydration

**Purpose:** Resolve a seed's `db_filter` against the live database (or a SQL fixture) and return a `ScenarioInstance` containing actual entity data. This is what makes tool calls, eligibility checks, and policy retrieval exercise real paths.

**Interface:**

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ScenarioInstance:
    seed_id: str
    category: str
    intent: str
    difficulty: str                  # easy | medium | hard
    persona_id: str
    cooperation_level: str           # cooperative | passive | resistant
    expected_outcome: str
    expected_procedure_id: str | None
    adversarial_flags: list[str]
    entity: dict[str, Any]           # hydrated DB record, e.g. {order_id, status, user_id, ...}
    secondary_entity: dict[str, Any] | None   # for multi_issue seeds
    multi_issue: bool
    secondary_category: str | None
    secondary_intent: str | None

class ScenarioHydrator:
    def __init__(self, db_url: str): ...

    def hydrate(self, seed: SeedConfig) -> ScenarioInstance:
        """
        Query DB using seed.db_filter constraints.
        Raise HydrationError if no matching entity found.
        Return a ScenarioInstance with entity populated.
        """
        ...
```

**DB query logic:**
- Entity type `order`: query `orders` table with `status IN [...]` and age constraint computed from `created_at`.
- Entity type `user`: query `users` with account standing filter.
- Entity type `subscription`: query `subscriptions` with status and plan filter.
- Select **randomly** from matching rows so each run exercises different real data.
- If `multi_issue: true`, run a second independent query for `secondary_issue.db_filter` and populate `secondary_entity`.
- Log the selected entity IDs to the run artifact so failures are reproducible.

### 5.2 `persona.py` — Persona Engine

**Purpose:** Produce the simulated user's messages turn-by-turn, governed by the persona definition and the current conversation state.

**Interface:**

```python
class PersonaEngine:
    def __init__(
        self,
        persona: PersonaConfig,
        scenario: ScenarioInstance,
        llm_provider: str,
        llm_model: str,
        llm_timeout_seconds: float,
    ): ...

    def generate_opening(self) -> str:
        """
        Produce the first user message that introduces the issue.
        Uses scenario.entity to ground the message (real order_id, dates, etc.)
        """
        ...

    def generate_response(
        self,
        agent_message: str,
        turn_number: int,
        conversation_history: list[dict],
        agent_metadata: dict,
    ) -> str | None:
        """
        Produce the next user message given the agent's last response.
        Returns None to signal the user has accepted resolution (end conversation).
        Uses LLM-generated response with directive controls derived from persona/scenario state.
        Terminal outcomes return None immediately.
        Low-patience personas can request human escalation on later turns.
        Multi-issue personas can introduce secondary_issue after turn 2 (once).
        """
        ...
```

**LLM invocation shape** (implemented inline in `persona.py`):

```
system: _PERSONA_SYSTEM_PROMPT
user: JSON payload containing:
- mode: "opening" | "response"
- persona: PersonaConfig dump
- scenario: ScenarioInstance dump (including hydrated entities)
- turn_number, agent_message, conversation_history, agent_metadata
- internal state counters/flags
- directives (force behaviors for resistant/missing-field, human escalation, multi-issue)
- required_output_schema
```

**Required LLM output contract:**

```json
{
  "message": "customer utterance",
  "stop": false
}
```

Parsing uses `backend.llm.providers.extract_json_object`. Empty/invalid payloads and request failures raise runtime errors (fail-fast; no template fallback).

**Grounding and safety constraints (enforced in code):**

- For `entity_type=order`, opening messages must include hydrated `order_id` unless scenario flags indicate intentional missing-data behavior.
- When `validation_missing` includes `order_id` (and not in resistant first-pushback mode), generated response must include hydrated `order_id`.
- Generated text must stay grounded to provided scenario/entity content (system prompt rule).

**Cooperation-level behavior (implemented with directives/state):**

| Level | Behaviour |
|---|---|
| `cooperative` | Tends to provide requested data promptly. |
| `passive` | Generally cooperative with softer/indirect phrasing patterns. |
| `resistant` | First missing-field prompt can challenge the assistant before providing data. |

### 5.3 `driver.py` — Conversation Turn Loop

**Purpose:** Orchestrate the turn-by-turn conversation between the persona engine and the agent's `POST /classify` endpoint. Manage session state.

**Interface:**

```python
@dataclass
class TurnRecord:
    turn_number: int
    user_message: str
    agent_response: str
    outcome_status: str
    procedure_id: str | None
    validation_missing: list[str]
    eligibility_ok: bool | None
    escalation_bundle: dict | None
    policy_constraints: dict | None
    context_data: dict | None
    latency_ms: float

@dataclass
class ConversationTrace:
    scenario: ScenarioInstance
    session_id: str
    turns: list[TurnRecord]
    final_outcome_status: str
    terminated_by: str     # "resolved" | "max_turns" | "persona_accepted" | "escalated"
    total_latency_ms: float
    total_tokens_used: int | None

class ConversationDriver:
    def __init__(self, agent_url: str, max_turns: int): ...

    def run(
        self,
        scenario: ScenarioInstance,
        persona: PersonaEngine,
    ) -> ConversationTrace:
        """
        1. Generate opening message from persona.
        2. POST to /classify with full_flow=true and session_id.
        3. Capture full assistant_metadata from response.
        4. Pass agent response to persona.generate_response().
        5. Repeat until: persona returns None (accepted), max_turns reached,
           or outcome_status is terminal (resolved/escalated/policy_ineligible/etc).
        6. Return ConversationTrace.
        """
        ...
```

**Request shape** (existing API contract):

```python
{
    "text": user_message,
    "session_id": session_id,
    "full_flow": True
}
```

**Response fields to capture from `assistant_metadata`:**

```python
{
    "outcome_status": str,
    "procedure_id": str | None,
    "validation_missing": list[str],
    "eligibility_ok": bool | None,
    "escalation_bundle": dict | None,  # optional; may be absent in API response
    "policy_constraints": dict | None,
    "context_data": dict | None,
    "specialist_agent_id": str | None,
    "agent_state": dict | None,
    "stage_metadata": dict | None,
    "output_validation": dict | None,
    "context_summary": dict | None,
    "validation_wait_count": int | None,
    "validation_wait_limit": int | None,
}
```

**Termination conditions** (check in order each turn):

1. `outcome_status` in `{resolved, policy_ineligible, tool_error, step_error, unresolvable}` → terminal.
2. `outcome_status == pending_escalation` → terminal (escalated).
3. Persona returned `None` → persona accepted.
4. `turn_number >= max_turns` → max turns reached.

### 5.4 `evaluators/structural.py` — Structural Evaluator

**Purpose:** Assert on `outcome_status`, `procedure_id`, and graph routing using the `ConversationTrace`. This is the primary pass/fail signal.

**Checks:**

| Check | Pass condition |
|---|---|
| `outcome_status_match` | `trace.final_outcome_status == seed.expected_outcome` |
| `procedure_id_match` | If `expected_procedure_id` set: correct procedure was loaded. |
| `no_unexpected_escalation` | If `expected_outcome != escalate`: no escalation bundle present. |
| `validation_resolved` | If outcome is `resolved`: `validation_missing` is empty on final turn. |
| `classification_confidence` | `confidence` field never below `CATEGORY_CONFIDENCE_THRESHOLD` on turns 2+. |
| `issue_lock_respected` | If `issue_locked=true` in turn 2+: `category` and `intent` remain stable. |
| `max_turns_not_breached` | Conversation ended before `max_turns` (a grace failure, not a hard fail). |

**Output:**

```python
@dataclass
class StructuralResult:
    passed: bool
    checks: dict[str, bool]     # check_name -> pass/fail
    failures: list[str]         # human-readable failure descriptions
```

### 5.5 `evaluators/policy.py` — Policy Fidelity Evaluator

**Purpose:** Verify that the policy retrieval and eligibility gate behaved correctly given the hydrated entity.

**Checks:**

| Check | Pass condition |
|---|---|
| `eligibility_correct` | `eligibility_ok` matches what DB entity data implies for the policy |
| `policy_docs_retrieved` | `context_data` contains at least one policy doc relevant to category/intent |
| `boundary_handling` | For `adversarial_flags: [eligibility_boundary]`: eligibility decision is deterministic across re-runs with same entity |
| `ineligible_explanation_present` | For `policy_ineligible` outcomes: agent response contains a reason, not just a refusal |

**Implementation note:** `eligibility_correct` requires a small truth function per category that reads the entity fields and the policy constraints to determine what the correct eligibility decision should have been. Implement these as `_check_eligibility_<category>(entity, policy_constraints) -> bool` functions.

### 5.6 `evaluators/llm_judge.py` — LLM-as-Judge Evaluator

**Purpose:** Score response quality on dimensions that structural checks cannot capture: tone, completeness, hallucination, and escalation appropriateness.

**Scoring dimensions (1–5 scale):**

| Dimension | What it measures |
|---|---|
| `tone` | Appropriate warmth and professionalism given the persona and outcome |
| `completeness` | All user questions were addressed; no dangling issues |
| `groundedness` | Response references only data that was actually in `context_data`; no fabricated order details |
| `escalation_appropriateness` | If escalated: was it warranted? If not escalated: should it have been? |
| `resolution_clarity` | If resolved: is the resolution clearly stated and actionable? |

**Prompt structure** (rationale-before-score for auditability):

```
You are evaluating a customer support conversation.

## Conversation
{{ conversation_turns }}

## Agent's final response
{{ final_response }}

## Context available to agent
Policy docs retrieved: {{ policy_docs_summary }}
Entity data: {{ entity_summary }}
Outcome status: {{ outcome_status }}

## Scoring instructions
For each dimension below, first write 1-2 sentences of rationale, then assign a score 1–5.
Do not assign a score before writing the rationale.

Dimensions: tone, completeness, groundedness, escalation_appropriateness, resolution_clarity

Respond in JSON only:
{
  "tone": {"rationale": "...", "score": N},
  "completeness": {"rationale": "...", "score": N},
  "groundedness": {"rationale": "...", "score": N},
  "escalation_appropriateness": {"rationale": "...", "score": N},
  "resolution_clarity": {"rationale": "...", "score": N}
}
```

**Thresholds** (configurable per suite):

```yaml
llm_judge_thresholds:
  tone: 3
  completeness: 3
  groundedness: 4          # higher threshold — hallucination is a hard failure
  escalation_appropriateness: 3
  resolution_clarity: 3
```

A scenario fails if any dimension score falls below its threshold.

### 5.7 `evaluators/regression.py` — Regression Evaluator

**Purpose:** Diff the current run against a pinned baseline artifact and flag any category/intent pairs whose resolution rate, escalation rate, or average LLM judge scores changed beyond thresholds.

**Baseline artifact structure:**

```json
{
  "run_id": "baseline_v1",
  "generated_at": "2026-04-25T10:00:00Z",
  "per_seed": {
    "cancel_order_easy": {
      "outcome_status": "resolved",
      "structural_passed": true,
      "llm_judge_scores": {"tone": 4.0, "completeness": 4.5, "groundedness": 5.0, "resolution_clarity": 4.5},
      "latency_ms": 1820
    }
  },
  "per_category": {
    "order_cancellation": {"resolution_rate": 0.85, "escalation_rate": 0.05}
  }
}
```

**Regression conditions:**

| Signal | Threshold (configurable) |
|---|---|
| `outcome_status` changed for a seed | Always a regression (no tolerance) |
| LLM judge score dropped | > 0.5 points on any dimension |
| Category resolution rate dropped | > 5 percentage points |
| Category escalation rate increased | > 5 percentage points |
| p95 latency increased | > 500ms per node |

### 5.8 `coverage.py` — Coverage Checker

**Purpose:** Enforce that every supported `(category, intent)` pair from active procedure blueprints and DB intent taxonomy is either covered by at least one seed or explicitly listed as a known gap.

**Usage:** Runs automatically before each test suite execution. Prints a coverage table and writes it to the run artifact.

**Coverage table output format:**

```
Category Coverage Report
========================
category              intent                          seeds   status
--------------------  ------------------------------  ------  -------
order_cancellation    cancel_order_before_delivery    3       ✓
order_cancellation    cancel_order_after_dispatch     0       ⚠ GAP
refund                refund_damaged_item             2       ✓
refund                refund_wrong_item               1       ✓
shipping              track_order                     0       KNOWN GAP (see gaps.yaml)
...

Coverage: 14/18 intents covered (78%)
Known gaps: 2
Unexpected gaps: 2  ← build will warn (or fail if fail_on_coverage_gap: true)
```

**Known gaps file** (`testing/simulator/seeds/gaps.yaml`):

```yaml
known_gaps:
  - category: shipping
    intent: track_order
    reason: "Tracking tool not yet implemented in procedure YAML"
    ticket: BITBOT-142
```

---

## 6. Difficulty Level Definitions

Difficulty is not just query complexity. It maps to specific graph nodes and edge cases.

| Difficulty | What it stresses | Example |
|---|---|---|
| `easy` | Happy path: correct category, cooperative user, entity well within policy bounds. `structured_executor` runs cleanly. | Order 30 min old, user provides order_id immediately. |
| `medium` | Moderate friction: one missing required field, or an entity near (but inside) policy bounds. `validate_required` fires once. | User forgets order_id; provides it on second turn. |
| `hard` | Policy edge case, boundary entity, resistant persona, or multi-issue session. `eligibility_ok` gate, `logic_gate` steps, and `issue_locked` mechanics are all exercised. | Order is at the 30-min boundary; restaurant is "preparing"; user contests the ineligibility decision. |
| `adversarial` | Intentional adversarial input: policy probing, category confusion injection, jailbreak-adjacent phrasing, deliberate data spoofing. | User claims order_id belongs to them but DB shows it's another user's order. |

---

## 7. Adversarial Test Categories

Adversarial scenarios are explicitly defined, not emergent. The following flags are available in seed configs:

| Flag | What it tests |
|---|---|
| `eligibility_boundary` | Entity is at the exact policy cutoff (age, status, value). Tests `eligibility_ok` determinism. |
| `policy_argument` | Persona verbally contests a `policy_ineligible` decision. Tests that the agent maintains position without hallucinating exceptions. |
| `category_confusion` | Query is phrased to look like the wrong category. Tests ModernBERT confidence + `no_issue_direct` fallback. |
| `data_spoofing` | User provides plausible but incorrect entity identifiers. Tests tool-call validation. |
| `multi_issue_injection` | User introduces a second, unrelated issue mid-session. Tests `issue_locked` handling. |
| `escalation_fishing` | User repeatedly requests a human agent without a legitimate escalation reason. Tests `human_escalation` guard. |
| `missing_data_stubborn` | `cooperation_level: resistant` combined with the persona refusing required data for 3+ turns. Tests `validate_required` retry exhaustion. |

---

## 8. Runner CLI

**File:** `testing/simulator/runner.py`

```bash
# Run a full suite
python -m testing.simulator.runner --suite suites/regression.yaml

# Run a single seed
python -m testing.simulator.runner --suite suites/regression.yaml --seed cancel_order_hard

# Run with a DB fixture instead of live DB
python -m testing.simulator.runner --suite suites/smoke.yaml --db-snapshot fixtures/snapshot_20260425.sql

# Pin current results as a new baseline
python -m testing.simulator.runner --suite suites/regression.yaml --write-baseline

# Check coverage only (no conversations)
python -m testing.simulator.runner --suite suites/regression.yaml --coverage-only

# Run specific categories only
python -m testing.simulator.runner --suite suites/regression.yaml --category order_cancellation refund

# Run specific difficulty levels only
python -m testing.simulator.runner --suite suites/regression.yaml --difficulty hard adversarial

# Continuous mode until interrupted
python -m testing.simulator.runner --suite suites/regression.yaml --forever --randomize

# Disable DB persistence for a single run (default is enabled)
python -m testing.simulator.runner --suite suites/regression.yaml --no-persist-db
```

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | All scenarios passed |
| `1` | One or more structural/policy/llm_judge failures |
| `3` | Coverage gaps found (if `fail_on_coverage_gap: true`) |
| `4` | Hydration error (no matching DB entity for a seed) |

---

## 9. Run Artifact Schema

Every run writes a JSON artifact to `testing/simulator/results/run_<timestamp>.json`.

```json
{
  "run_id": "regression_sprint_42",
  "suite": "suites/regression.yaml",
  "started_at": "2026-04-25T10:00:00Z",
  "completed_at": "2026-04-25T10:12:34Z",
  "db_snapshot": "live",
  "agent_url": "http://localhost:8000/classify",
  "coverage": {
    "total_intents": 18,
    "covered": 14,
    "known_gaps": 2,
    "unexpected_gaps": 2
  },
  "summary": {
    "total_scenarios": 12,
    "passed": 10,
    "failed": 2,
    "structural_failures": 1,
    "llm_judge_failures": 1,
    "regressions": 0
  },
  "per_category": {
    "order_cancellation": {
      "resolution_rate": 0.80,
      "escalation_rate": 0.10,
      "avg_turns": 3.2,
      "avg_latency_ms": 1950
    }
  },
  "scenarios": [
    {
      "seed_id": "cancel_order_easy",
      "entity_id": "order_9821",
      "persona_id": "polite_first_timer",
      "turns": 2,
      "final_outcome_status": "resolved",
      "expected_outcome": "resolved",
      "structural": {"passed": true, "checks": {...}},
      "policy": {"passed": true, "checks": {...}},
      "llm_judge": {
        "passed": true,
        "scores": {"tone": 5, "completeness": 4, "groundedness": 5, "resolution_clarity": 5}
      },
      "regression": {"passed": true, "deltas": {}},
      "trace": [
        {
          "turn": 1,
          "user": "Hi I need to cancel my order #9821",
          "agent": "...",
          "outcome_status": "needs_more_data",
          "procedure_id": "cancel_order_before_delivery",
          "validation_missing": [],
          "eligibility_ok": true,
          "latency_ms": 1820
        }
      ]
    }
  ]
}
```

---

## 10. Implementation Notes

The current implementation is already functional with:

1. **`config.py`** — Pydantic schemas for suite/seeds/personas/artifacts.
2. **`hydrator.py`** — DB-grounded entity hydration from Postgres.
3. **`persona.py`** — LLM-backed persona engine with structured output + grounding checks.
4. **`driver.py`** — HTTP turn loop to `POST /classify` with `full_flow=true`.
5. **`evaluators/structural.py`**, **`evaluators/policy.py`**, **`evaluators/llm_judge.py`** — active evaluators.
6. **`runner.py`** + **`reporter.py`** + **`persistence.py`** — CLI, artifact writing, and optional DB persistence (enabled by default).
7. **`coverage.py`** — pre-run coverage report/check.

`regression` remains a config keyword and artifact field but is not currently executed as a runtime evaluator.

---

## 11. Integration Notes

- **API contract**: the simulator exclusively uses `POST /classify` with `full_flow=true`. Do not bypass the HTTP layer — the test must exercise the same path as production traffic.
- **Session management**: each scenario gets a fresh `session_id` (UUID4). The driver tracks this and sends it on every turn to exercise `issue_locked` semantics.
- **Persona LLM config**: simulator user-message generation uses dedicated settings (`defaults.user_llm_*` and `SIMULATOR_USER_LLM_*`) and currently supports `ollama` or `cerebras` via `backend.llm.providers.chat_completion`.
- **Judge LLM config**: LLM judge keeps separate provider/model settings (`llm_judge_*`).
- **Postgres fixture**: `testing/simulator/fixtures/` should contain a minimal anonymised snapshot generated with `pg_dump --data-only --table=orders --table=users --table=subscriptions`. This allows deterministic re-runs in CI without a live DB.
- **CI integration**: run `suites/smoke.yaml` on every PR (fast, ~5 scenarios, no LLM judge). Run `suites/regression.yaml` nightly (full suite with LLM judge and baseline diff).
- **Environment variables** required by the simulator:

```bash
SIMULATOR_AGENT_URL=http://localhost:8000/classify
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecom_support
POSTGRES_USER=admin
POSTGRES_PASSWORD=...
POSTGRES_HOST_SIMULATOR=localhost

# Persona (user-message generation) LLM
SIMULATOR_USER_LLM_PROVIDER=ollama
SIMULATOR_USER_LLM_MODEL=llama3.2
SIMULATOR_USER_LLM_TIMEOUT_SECONDS=120

# Optional LLM judge settings (if llm_judge enabled)
SIMULATOR_LLM_PROVIDER=ollama
SIMULATOR_LLM_MODEL=llama3.2
SIMULATOR_LLM_TIMEOUT_SECONDS=120
```

---

## 12. What This Framework Does Not Cover

The following are explicitly out of scope and should not be added without deliberate architectural review:

- **Load/stress testing**: this is a correctness framework. Use Locust or k6 for load testing separately.
- **UI/chat widget testing**: the simulator targets the HTTP API only. Frontend testing is a separate concern.
- **Live production traffic replay**: replaying production sessions introduces PII handling complexity. Use anonymised fixtures instead.
- **A/B evaluation between agent versions**: this framework tests one agent version against a baseline. Multi-version comparison requires a separate harness.
