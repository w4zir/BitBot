# Procedure Spec: Fin-Inspired Structured Automation Architecture

> **Purpose:** Implementation reference for building a deterministic, blueprint-driven agentic system in LangGraph, modeled on Intercom's Fin Procedures & Workflows architecture.

---

## 1. Core Philosophy

This system separates **Intent Classification** from **Plan Generation**, replacing a stochastic ReAct planner loop with a deterministic **Stateful Plan Executor**. The LLM is constrained to executing predefined steps — it never decides *what* to do, only *how* to do the current step.

| Concern | Owner |
|---|---|
| What to do (routing) | Intent Classifier (LLM with structured output) |
| Which steps to run | Blueprint Library (YAML / DB) |
| How to run a step | Structured Executor (LLM-assisted, constrained) |
| When to pause | LangGraph Interrupt (HITL) |

---

## 2. Blueprint Library

### 2.1 Schema

Each intent maps 1:1 to a **Blueprint** — a YAML file that fully defines the procedure.

```yaml
# blueprints/refund_request.yaml
intent: refund_request
steps:
  - id: retrieve_policy
    type: retrieval
    tool: get_refund_policy
    required_data: [user_id, order_id]

  - id: check_eligibility
    type: logic_gate
    condition: "days_since_purchase < 30"
    on_true: process_refund
    on_false: escalate_to_human

  - id: process_refund
    type: tool_call
    tool: stripe_refund_api
    required_data: [user_id, order_id, refund_amount]

  - id: escalate_to_human
    type: interrupt
    message: "Customer ineligible for automated refund. Escalating."
```

### 2.2 Step Types

| Type | Description |
|---|---|
| `retrieval` | RAG lookup; result stored in `context_data` |
| `tool_call` | External API call; requires all `required_data` present in `context_data` |
| `logic_gate` | Conditional branch; evaluates expression against `context_data` |
| `interrupt` | Pauses execution for human-in-the-loop approval |
| `llm_response` | LLM drafts a message to the user using `context_data` |

### 2.3 Storage Strategy

| Environment | Storage | Rationale |
|---|---|---|
| Development | `.yaml` files in `/blueprints/` directory | Version-controlled, testable, diff-friendly |
| Production | PostgreSQL JSONB or MongoDB document store | Allows ops/product teams to update procedures without code deploys |

### 2.4 Validation

All blueprints must be validated at load time using **Pydantic**.

```python
from pydantic import BaseModel
from typing import Literal, Optional

class BlueprintStep(BaseModel):
    id: str
    type: Literal["retrieval", "tool_call", "logic_gate", "interrupt", "llm_response"]
    tool: Optional[str] = None
    required_data: Optional[list[str]] = []
    condition: Optional[str] = None
    on_true: Optional[str] = None
    on_false: Optional[str] = None
    message: Optional[str] = None

class Blueprint(BaseModel):
    intent: str
    steps: list[BlueprintStep]
```

---

## 3. LangGraph State

The graph state is the single source of truth across all nodes.

```python
from typing import TypedDict, Any

class AgentState(TypedDict):
    messages: list[dict]          # Full conversation history
    intent: str                   # Classified intent key (e.g., "refund_request")
    todo_list: list[dict]         # Ordered steps from the matched blueprint
    current_step_index: int       # Pointer to the active step
    context_data: dict[str, Any]  # Accumulated results from retrieval and tool calls
    final_response: str           # Assembled response to send to user
```

---

## 4. Graph Nodes

### 4.1 Intent Classifier Node

- **Input:** `messages`
- **Output:** `intent`
- **Logic:** LLM with structured output (Pydantic) maps the user's query to a registered `intent_id`. If no match, defaults to `"general_research"`.

```python
def intent_classifier(state: AgentState) -> dict:
    # Use structured output to constrain LLM to valid intent keys
    intent = model.predict(state["messages"])  # returns validated intent string
    return {"intent": intent}
```

### 4.2 Procedure Loader Node

- **Input:** `intent`
- **Output:** `todo_list`, `current_step_index`
- **Logic:** Fetches the blueprint from the library. Falls back to `general_research` blueprint if `intent` is not found.

```python
def fetch_blueprint(state: AgentState) -> dict:
    blueprint = blueprint_library.get(state["intent"])
    if not blueprint:
        blueprint = blueprint_library.get("general_research")
    return {
        "todo_list": blueprint["steps"],
        "current_step_index": 0
    }
```

### 4.3 Structured Executor Node

- **Input:** `todo_list`, `current_step_index`, `context_data`
- **Output:** updated `context_data`, incremented `current_step_index`
- **Logic:** Dispatches the current step based on its `type`. The LLM is only invoked to extract tool arguments or draft user-facing text — never to decide the next step.

```python
def structured_executor(state: AgentState) -> dict:
    step = state["todo_list"][state["current_step_index"]]

    if step["type"] == "retrieval":
        result = run_rag(step["tool"], state["context_data"])
        state["context_data"].update(result)

    elif step["type"] == "tool_call":
        assert_required_data(step["required_data"], state["context_data"])
        result = call_tool(step["tool"], state["context_data"])
        state["context_data"].update(result)

    elif step["type"] == "logic_gate":
        branch = evaluate_condition(step["condition"], state["context_data"])
        next_step_id = step["on_true"] if branch else step["on_false"]
        return jump_to_step(state, next_step_id)

    elif step["type"] == "llm_response":
        response = draft_response(step, state["context_data"], state["messages"])
        state["final_response"] = response

    return {"context_data": state["context_data"],
            "current_step_index": state["current_step_index"] + 1}
```

### 4.4 Termination Condition

```python
def should_continue(state: AgentState) -> str:
    if state["current_step_index"] >= len(state["todo_list"]):
        return "end"
    next_step = state["todo_list"][state["current_step_index"]]
    if next_step["type"] == "interrupt":
        return "interrupt"
    return "continue"
```

---

## 5. Graph Topology

```
[START]
   │
   ▼
[intent_classifier]
   │
   ▼
[fetch_blueprint]
   │
   ▼
[structured_executor] ◄──────────────┐
   │                                  │
   ├─ continue ───────────────────────┘
   │
   ├─ interrupt ──► [HITL: human approval] ──► [structured_executor]
   │
   └─ end ──► [END]
```

---

## 6. Human-in-the-Loop (HITL)

Use LangGraph `interrupt()` on high-stakes steps (payments, deletions, escalations). The graph pauses and waits for an external `resume` signal.

```python
from langgraph.types import interrupt

def structured_executor(state: AgentState) -> dict:
    step = state["todo_list"][state["current_step_index"]]

    if step["type"] == "interrupt":
        approval = interrupt({
            "message": step["message"],
            "context": state["context_data"]
        })
        if not approval["approved"]:
            return {"final_response": "Action cancelled by reviewer."}
    # ... continue execution
```

**Recommended interrupt points:**
- Before any `tool_call` that mutates financial data (payments, refunds)
- Before any delete or deactivation action
- When `logic_gate` routes to `escalate_to_human`

---

## 7. RAG Integration

**BitBot implementation:** `retrieval` steps call Elasticsearch via `multi_match` on indexed policy documents (`title`, `content`, `tags`). There is no Postgres vector store or in-process reranker in the current codebase; the pseudocode below is the target pattern.

After each `retrieval` step, apply a **reranker** before storing results in `context_data`. This ensures that only genuinely relevant chunks are passed to subsequent tool calls or LLM steps.

```python
def run_rag(tool_name: str, context: dict) -> dict:
    raw_results = retriever.invoke(context)
    reranked = reranker.compress_documents(raw_results, query=context["user_query"])
    return {"retrieved_docs": reranked}
```

---

## 8. Fallback Blueprint

Every unrecognized intent must route to a `general_research` blueprint that performs standard RAG without any tool calls or business logic.

```yaml
# blueprints/general_research.yaml
intent: general_research
steps:
  - id: retrieve_context
    type: retrieval
    tool: knowledge_base_search
    required_data: [user_query]

  - id: draft_answer
    type: llm_response
```

---

## 9. Design Principles

1. **Deterministic over Autonomous** — The planner follows the path; it never finds it.
2. **Tool Grounding** — Every `tool_call` step must be preceded by a step that ensures `required_data` is present in `context_data`. Fail loudly if preconditions are not met.
3. **Graceful Degradation** — Unknown intents fall back to `general_research`. Never expose a raw planner to unclassified input.
4. **Observability** — Each step has an explicit `id`. Logging `current_step_index` + step `id` gives a full execution trace. Failures are attributable to a specific named step, not a black-box LLM chain.
5. **Human-in-the-Loop** — Insert `interrupt` steps before any irreversible or high-stakes action. HITL is a first-class citizen in the blueprint schema, not an afterthought.
6. **Ops-Friendly Updates** — In production, blueprints live in a database. Support/product teams can modify procedures without a code deploy or engineer involvement.

---

## 10. Directory Structure

```
project/
├── blueprints/                  # YAML blueprint definitions (dev)
│   ├── refund_request.yaml
│   ├── cancel_subscription.yaml
│   └── general_research.yaml
├── graph/
│   ├── state.py                 # AgentState TypedDict
│   ├── nodes.py                 # All node functions
│   ├── edges.py                 # Conditional edge logic
│   └── graph.py                 # StateGraph assembly
├── blueprints_loader.py         # YAML → Pydantic validation + library dict
├── tools/                       # External API wrappers
└── tests/
    └── test_blueprints.py       # Validate all blueprints at CI time
```

---

## 11. Recommended Libraries

| Concern | Library |
|---|---|
| Graph execution | `langgraph` (`StateGraph`) |
| Blueprint validation | `pydantic` v2 |
| Blueprint storage (dev) | `pyyaml` |
| Blueprint storage (prod) | `asyncpg` + PostgreSQL JSONB |
| RAG retrieval | `langchain` retrievers |
| Reranking | `FlashrankRerank` or `CohereRerank` |
| Observability | LangSmith or custom step-level logging |
