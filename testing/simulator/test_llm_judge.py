from __future__ import annotations

from testing.simulator.config import LlmJudgeThresholdsConfig
from testing.simulator.evaluators import llm_judge
from testing.simulator.hydrator import ScenarioInstance
from testing.simulator.trace import ConversationTrace, TurnRecord


def _trace() -> ConversationTrace:
    turn = TurnRecord(
        turn_number=1,
        user_message="I need a refund",
        agent_response="I can help with that.",
        outcome_status="resolved",
        procedure_id="refund_v1",
        context_data={"policy_doc_names": ["refund_policy"]},
    )
    return ConversationTrace(
        scenario={"seed_id": "seed1", "category": "refund", "intent": "get_refund"},
        session_id="",
        turns=[turn],
        final_outcome_status="resolved",
        terminated_by="resolved",
        total_latency_ms=120.0,
    )


def _scenario() -> ScenarioInstance:
    return ScenarioInstance(
        seed_id="seed1",
        category="refund",
        intent="get_refund",
        difficulty="easy",
        persona_id="p1",
        cooperation_level="cooperative",
        expected_outcome="resolved",
        expected_procedure_id="refund_v1",
        adversarial_flags=[],
        entity={"order_id": "ORD-1"},
        secondary_entity=None,
        multi_issue=False,
        secondary_category=None,
        secondary_intent=None,
    )


def test_llm_judge_threshold_failure(monkeypatch) -> None:
    def _fake_chat_completion_with_usage(*, provider, model, messages):
        return (
            """{
              "tone":{"rationale":"ok","score":4},
              "completeness":{"rationale":"ok","score":4},
              "groundedness":{"rationale":"weak","score":2},
              "escalation_appropriateness":{"rationale":"ok","score":4},
              "resolution_clarity":{"rationale":"ok","score":4}
            }""",
            {"input_tokens": 10, "output_tokens": 5, "cache_tokens": 0, "total_tokens": 15},
            55.0,
        )

    monkeypatch.setattr(llm_judge, "_chat_completion_with_usage", _fake_chat_completion_with_usage)
    result = llm_judge.evaluate_llm_judge(
        trace=_trace(),
        scenario=_scenario(),
        provider="ollama",
        model="m",
        thresholds=LlmJudgeThresholdsConfig(groundedness=4.0),
    )
    assert not result.passed
    assert "groundedness" in " ".join(result.failures)
    assert result.total_tokens == 15
