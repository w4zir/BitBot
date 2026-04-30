from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import Json

from backend.db.postgres import get_connection, postgres_configured
from testing.simulator.evaluators.llm_judge import LlmJudgeResult
from testing.simulator.evaluators.policy import PolicyResult
from testing.simulator.evaluators.structural import StructuralResult
from testing.simulator.trace import ConversationTrace


class SimulatorPersistence:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and postgres_configured()
        self._run_db_id: str | None = None

    @property
    def run_db_id(self) -> str | None:
        return self._run_db_id

    def start_run(
        self,
        *,
        run_id: str,
        suite_name: str,
        db_snapshot: str,
        baseline_ref: str | None,
        run_metadata: dict[str, Any],
    ) -> str | None:
        if not self.enabled:
            return None
        sql = """
            INSERT INTO simulation_runs (
                run_id, suite_name, db_snapshot, baseline_ref, status, started_at, run_metadata_json
            )
            VALUES (%s, %s, %s, %s, 'running', NOW(), %s)
            ON CONFLICT (run_id) DO UPDATE SET
              suite_name = EXCLUDED.suite_name,
              db_snapshot = EXCLUDED.db_snapshot,
              baseline_ref = EXCLUDED.baseline_ref,
              status = 'running',
              started_at = NOW(),
              completed_at = NULL,
              run_metadata_json = EXCLUDED.run_metadata_json,
              summary_json = NULL
            RETURNING id
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        run_id,
                        suite_name,
                        db_snapshot,
                        baseline_ref,
                        Json(run_metadata or {}),
                    ),
                )
                row = cur.fetchone()
        self._run_db_id = str(row[0]) if row else None
        return self._run_db_id

    def record_coverage(self, coverage: dict[str, Any]) -> None:
        if not self.enabled or not self._run_db_id:
            return
        total = int(coverage.get("total_intents") or 0)
        covered = int(coverage.get("covered") or 0)
        known = int(coverage.get("known_gaps") or 0)
        unexpected = int(coverage.get("unexpected_gaps") or 0)
        ratio = (float(covered) / float(total)) if total > 0 else 0.0
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO coverage_snapshots (
                        run_id, total_pairs, covered_pairs, known_gaps, unexpected_gaps, coverage_ratio, gap_details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self._run_db_id,
                        total,
                        covered,
                        known,
                        unexpected,
                        ratio,
                        Json(coverage),
                    ),
                )

    def record_scenario(
        self,
        *,
        trace: ConversationTrace,
        structural: StructuralResult,
        policy: PolicyResult,
        llm_judge: LlmJudgeResult | None,
    ) -> None:
        if not self.enabled or not self._run_db_id:
            return
        scenario = trace.scenario
        entity = scenario.get("entity") or {}
        session_id = _parse_uuid_or_none(trace.session_id)
        scenario_id = self._insert_scenario(
            session_id=session_id,
            scenario=scenario,
            trace=trace,
            structural=structural,
            policy=policy,
            llm_judge=llm_judge,
            entity=entity,
        )
        self._insert_turns_and_messages(scenario_id=scenario_id, trace=trace)
        self._insert_evaluations(
            scenario_id=scenario_id,
            structural=structural,
            policy=policy,
            llm_judge=llm_judge,
        )
        self._insert_training_example(scenario_id=scenario_id, trace=trace)

    def complete_run(self, *, summary: dict[str, Any], status: str = "completed") -> None:
        if not self.enabled or not self._run_db_id:
            return
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE simulation_runs
                    SET status = %s,
                        completed_at = NOW(),
                        summary_json = %s
                    WHERE id = %s
                    """,
                    (status, Json(summary), self._run_db_id),
                )

    def _insert_scenario(
        self,
        *,
        session_id: str | None,
        scenario: dict[str, Any],
        trace: ConversationTrace,
        structural: StructuralResult,
        policy: PolicyResult,
        llm_judge: LlmJudgeResult | None,
        entity: dict[str, Any],
    ) -> str:
        passed = structural.passed and policy.passed and (llm_judge.passed if llm_judge else True)
        assertions = {
            "structural": asdict(structural),
            "policy": asdict(policy),
            "llm_judge": asdict(llm_judge) if llm_judge else None,
        }
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO simulation_scenarios (
                        run_id, session_id, seed_id, persona_id, category, intent,
                        linked_order_id, linked_user_id, linked_subscription_email,
                        expected_outcome, actual_outcome, passed, assertions_json, trace_json, evaluated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    (
                        self._run_db_id,
                        session_id,
                        scenario.get("seed_id"),
                        scenario.get("persona_id"),
                        scenario.get("category"),
                        scenario.get("intent"),
                        entity.get("order_id"),
                        entity.get("user_id"),
                        entity.get("account_email"),
                        scenario.get("expected_outcome"),
                        trace.final_outcome_status,
                        passed,
                        Json(assertions),
                        Json(trace.to_dict()),
                    ),
                )
                row = cur.fetchone()
        return str(row[0])

    def _insert_turns_and_messages(self, *, scenario_id: str, trace: ConversationTrace) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                message_index = 0
                for turn in trace.turns:
                    cur.execute(
                        """
                        INSERT INTO simulation_turns (
                            scenario_id, turn_number, request_started_at, response_received_at,
                            latency_ms, outcome_status, procedure_id, category, intent,
                            validation_missing, policy_constraints, context_data,
                            agent_state, stage_metadata, output_validation, context_summary,
                            request_payload, response_payload,
                            input_tokens, output_tokens, cache_tokens, total_tokens
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                        """,
                        (
                            scenario_id,
                            turn.turn_number,
                            _parse_ts(turn.request_started_at),
                            _parse_ts(turn.response_received_at),
                            turn.latency_ms,
                            turn.outcome_status,
                            turn.procedure_id,
                            turn.category,
                            turn.intent,
                            Json(turn.validation_missing),
                            Json(turn.policy_constraints),
                            Json(turn.context_data),
                            Json(turn.agent_state),
                            Json(turn.stage_metadata),
                            Json(turn.output_validation),
                            Json(turn.context_summary),
                            Json(turn.request_payload),
                            Json(turn.response_payload),
                            turn.input_tokens,
                            turn.output_tokens,
                            turn.cache_tokens,
                            turn.total_tokens,
                        ),
                    )
                    turn_row = cur.fetchone()
                    turn_id = str(turn_row[0])
                    message_index += 1
                    cur.execute(
                        """
                        INSERT INTO simulation_messages (
                            scenario_id, turn_id, message_index, role, content, metadata_json
                        ) VALUES (%s, %s, %s, 'user', %s, %s)
                        """,
                        (
                            scenario_id,
                            turn_id,
                            message_index,
                            turn.user_message,
                            Json({"turn_number": turn.turn_number}),
                        ),
                    )
                    message_index += 1
                    cur.execute(
                        """
                        INSERT INTO simulation_messages (
                            scenario_id, turn_id, message_index, role, content, metadata_json,
                            input_tokens, output_tokens, cache_tokens, total_tokens, latency_ms
                        ) VALUES (%s, %s, %s, 'assistant', %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            scenario_id,
                            turn_id,
                            message_index,
                            turn.agent_response,
                            Json(
                                {
                                    "outcome_status": turn.outcome_status,
                                    "procedure_id": turn.procedure_id,
                                }
                            ),
                            turn.input_tokens,
                            turn.output_tokens,
                            turn.cache_tokens,
                            turn.total_tokens,
                            turn.latency_ms,
                        ),
                    )

    def _insert_evaluations(
        self,
        *,
        scenario_id: str,
        structural: StructuralResult,
        policy: PolicyResult,
        llm_judge: LlmJudgeResult | None,
    ) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for name, result in (
                    ("structural", structural),
                    ("policy", policy),
                ):
                    cur.execute(
                        """
                        INSERT INTO simulation_evaluations (
                            scenario_id, evaluator_name, passed, checks_json, failures_json, details_json
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (scenario_id, evaluator_name) DO UPDATE SET
                            passed = EXCLUDED.passed,
                            checks_json = EXCLUDED.checks_json,
                            failures_json = EXCLUDED.failures_json,
                            details_json = EXCLUDED.details_json,
                            evaluated_at = NOW()
                        """,
                        (
                            scenario_id,
                            name,
                            result.passed,
                            Json(result.checks),
                            Json(result.failures),
                            Json({}),
                        ),
                    )
                if llm_judge is not None:
                    cur.execute(
                        """
                        INSERT INTO simulation_evaluations (
                            scenario_id, evaluator_name, passed, checks_json, failures_json, details_json
                        ) VALUES (%s, 'llm_judge', %s, %s, %s, %s)
                        ON CONFLICT (scenario_id, evaluator_name) DO UPDATE SET
                            passed = EXCLUDED.passed,
                            checks_json = EXCLUDED.checks_json,
                            failures_json = EXCLUDED.failures_json,
                            details_json = EXCLUDED.details_json,
                            evaluated_at = NOW()
                        """,
                        (
                            scenario_id,
                            llm_judge.passed,
                            Json(
                                {
                                    key: llm_judge.scores.get(key, 0.0)
                                    >= llm_judge.thresholds.get(key, 0.0)
                                    for key in llm_judge.thresholds
                                }
                            ),
                            Json(llm_judge.failures),
                            Json(
                                {
                                    "scores": llm_judge.scores,
                                    "thresholds": llm_judge.thresholds,
                                    "rationales": llm_judge.rationales,
                                }
                            ),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO simulation_llm_judgements (
                            scenario_id, provider, model_name, passed, scores_json, rationales_json,
                            thresholds_json, failures_json, raw_response_json, latency_ms,
                            input_tokens, output_tokens, cache_tokens, total_tokens
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (scenario_id) DO UPDATE SET
                            provider = EXCLUDED.provider,
                            model_name = EXCLUDED.model_name,
                            passed = EXCLUDED.passed,
                            scores_json = EXCLUDED.scores_json,
                            rationales_json = EXCLUDED.rationales_json,
                            thresholds_json = EXCLUDED.thresholds_json,
                            failures_json = EXCLUDED.failures_json,
                            raw_response_json = EXCLUDED.raw_response_json,
                            latency_ms = EXCLUDED.latency_ms,
                            input_tokens = EXCLUDED.input_tokens,
                            output_tokens = EXCLUDED.output_tokens,
                            cache_tokens = EXCLUDED.cache_tokens,
                            total_tokens = EXCLUDED.total_tokens
                        """,
                        (
                            scenario_id,
                            llm_judge.provider,
                            llm_judge.model,
                            llm_judge.passed,
                            Json(llm_judge.scores),
                            Json(llm_judge.rationales),
                            Json(llm_judge.thresholds),
                            Json(llm_judge.failures),
                            Json(llm_judge.raw_response),
                            llm_judge.latency_ms,
                            llm_judge.input_tokens,
                            llm_judge.output_tokens,
                            llm_judge.cache_tokens,
                            llm_judge.total_tokens,
                        ),
                    )

    def _insert_training_example(self, *, scenario_id: str, trace: ConversationTrace) -> None:
        messages: list[dict[str, str]] = []
        for turn in trace.turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.agent_response})
        prompt_text = trace.turns[0].user_message if trace.turns else ""
        response_text = trace.turns[-1].agent_response if trace.turns else ""
        labels = {
            "category": trace.scenario.get("category"),
            "intent": trace.scenario.get("intent"),
            "seed_id": trace.scenario.get("seed_id"),
            "expected_outcome": trace.scenario.get("expected_outcome"),
        }
        metadata = {
            "terminated_by": trace.terminated_by,
            "total_latency_ms": trace.total_latency_ms,
            "total_tokens_used": trace.total_tokens_used,
        }
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO simulation_training_examples (
                        run_id, scenario_id, session_id, example_format, messages_json, prompt_text,
                        response_text, labels_json, modernbert_category, modernbert_intent, outcome_status, metadata_json
                    ) VALUES (%s, %s, %s, 'chatml', %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (scenario_id) DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        session_id = EXCLUDED.session_id,
                        messages_json = EXCLUDED.messages_json,
                        prompt_text = EXCLUDED.prompt_text,
                        response_text = EXCLUDED.response_text,
                        labels_json = EXCLUDED.labels_json,
                        modernbert_category = EXCLUDED.modernbert_category,
                        modernbert_intent = EXCLUDED.modernbert_intent,
                        outcome_status = EXCLUDED.outcome_status,
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    (
                        self._run_db_id,
                        scenario_id,
                        _parse_uuid_or_none(trace.session_id),
                        Json(messages),
                        prompt_text,
                        response_text,
                        Json(labels),
                        trace.scenario.get("category"),
                        trace.scenario.get("intent"),
                        trace.final_outcome_status,
                        Json(metadata),
                    ),
                )


def _parse_uuid_or_none(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)
