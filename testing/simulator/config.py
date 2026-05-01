from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


CooperationLevel = Literal["cooperative", "passive", "resistant"]
DifficultyLevel = Literal["easy", "medium", "hard", "adversarial"]
EvalTarget = Literal["structural", "policy", "llm_judge", "regression"]
EntityType = Literal["order", "user", "subscription"]


class DbFilterConfig(BaseModel):
    entity_type: EntityType
    order_status: list[str] = Field(default_factory=list)
    order_age_minutes: list[int] = Field(default_factory=list)
    order_age_days: list[int] = Field(default_factory=list)
    user_status: list[str] = Field(default_factory=list)
    subscription_status: list[str] = Field(default_factory=list)
    subscription_plan: list[str] = Field(default_factory=list)


class SecondaryIssueConfig(BaseModel):
    category: str
    intent: str
    db_filter: DbFilterConfig


class SeedConfig(BaseModel):
    seed_id: str
    category: str
    intent: str
    difficulty: DifficultyLevel = "medium"
    persona_id: str
    description: str = ""
    expected_outcome: str = "resolved"
    expected_procedure_id: str | None = None
    cooperation_level: CooperationLevel | None = None
    adversarial_flags: list[str] = Field(default_factory=list)
    db_filter: DbFilterConfig
    multi_issue: bool = False
    secondary_issue: SecondaryIssueConfig | None = None

    @model_validator(mode="after")
    def validate_secondary_issue(self) -> "SeedConfig":
        if self.multi_issue and self.secondary_issue is None:
            raise ValueError("secondary_issue is required when multi_issue is true")
        if not self.multi_issue and self.secondary_issue is not None:
            raise ValueError("secondary_issue must be omitted when multi_issue is false")
        return self


class SeedFileConfig(BaseModel):
    seeds: list[SeedConfig]


class PersonaConfig(BaseModel):
    persona_id: str
    display_name: str
    vocabulary: Literal["simple", "technical", "informal"] = "simple"
    patience: Literal["high", "medium", "low"] = "medium"
    cooperation_level: CooperationLevel = "cooperative"
    escalation_tendency: Literal["low", "medium", "high"] = "low"
    typical_message_length: Literal["short", "medium", "long"] = "medium"
    traits: list[str] = Field(default_factory=list)


class PersonasFileConfig(BaseModel):
    personas: list[PersonaConfig]


class LlmJudgeThresholdsConfig(BaseModel):
    tone: float = 3.0
    completeness: float = 3.0
    groundedness: float = 4.0
    escalation_appropriateness: float = 3.0
    resolution_clarity: float = 3.0


class DefaultsConfig(BaseModel):
    max_turns: int = Field(default=6, ge=1, le=25)
    cooperation_level: CooperationLevel = "cooperative"
    eval_targets: list[EvalTarget] = Field(default_factory=lambda: ["structural", "policy"])
    randomize: bool = False
    persist_db: bool = True
    user_llm_provider: Literal["ollama", "cerebras"] = "ollama"
    user_llm_model: str = "llama3.2"
    user_llm_timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    user_llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    user_llm_top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    user_llm_repeat_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_judge_model: str = "claude-sonnet-4-20250514"
    llm_judge_provider: Literal["ollama", "cerebras"] = "ollama"
    llm_judge_thresholds: LlmJudgeThresholdsConfig = Field(
        default_factory=LlmJudgeThresholdsConfig
    )
    fail_on_regression: bool = False
    fail_on_coverage_gap: bool = False


class ScenarioRunConfig(BaseModel):
    seed_id: str
    cooperation_level: CooperationLevel | None = None
    eval_targets: list[EvalTarget] | None = None


class SuiteConfig(BaseModel):
    run_id: str
    agent_url: str = "http://localhost:8000/classify"
    db_snapshot: str = "live"
    baseline: str | None = None
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    scenarios: list[ScenarioRunConfig]


class KnownGapConfig(BaseModel):
    category: str
    intent: str
    reason: str = ""
    ticket: str = ""


class KnownGapsFileConfig(BaseModel):
    known_gaps: list[KnownGapConfig] = Field(default_factory=list)


class EvaluatorResultConfig(BaseModel):
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ScenarioArtifactConfig(BaseModel):
    seed_id: str
    entity_id: str | None = None
    persona_id: str
    turns: int
    final_outcome_status: str
    expected_outcome: str
    structural: EvaluatorResultConfig
    policy: EvaluatorResultConfig | None = None
    llm_judge: dict[str, Any] | None = None
    regression: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


class RunArtifactConfig(BaseModel):
    run_id: str
    suite: str
    started_at: str
    completed_at: str
    db_snapshot: str
    agent_url: str
    coverage: dict[str, Any]
    summary: dict[str, Any]
    per_category: dict[str, Any]
    scenarios: list[ScenarioArtifactConfig]


def normalize_suite_path(path_value: str, simulator_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return simulator_root / path
