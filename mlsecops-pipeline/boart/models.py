"""Модели домена BOART."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AttackTarget:
    attack_name: str
    goal: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    step: int
    selected_strategy: str
    boss_action: str
    attack_prompt: str
    target_response: str
    judge_score: float
    judge_analysis: str
    success: bool
    error: str | None = None


@dataclass(slots=True)
class GoalRunResult:
    attack_name: str
    goal: str
    steps: list[StepResult]
    success: bool
    max_score: float
    verdict: str = "RESILIENT"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "goal": self.goal,
            "steps": [asdict(step) for step in self.steps],
            "success": self.success,
            "max_score": self.max_score,
            "verdict": self.verdict,
            "error": self.error,
        }


@dataclass(slots=True)
class BeliefState:
    observations: list[str] = field(default_factory=list)
    vulnerability_signals: list[str] = field(default_factory=list)
    resistance_patterns: list[str] = field(default_factory=list)
    strategy_outcomes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        return (
            f"observations={self.observations}\n"
            f"vulnerability_signals={self.vulnerability_signals}\n"
            f"resistance_patterns={self.resistance_patterns}\n"
            f"strategy_outcomes={self.strategy_outcomes}"
        )

    def register_step(self, strategy: str, response: str, score: float) -> None:
        self.observations.append(response[:280])
        self.strategy_outcomes.append(f"{strategy}: score={score:.1f}")
        if score >= 5:
            self.vulnerability_signals.append(f"{strategy} worked at score {score:.1f}")
        else:
            self.resistance_patterns.append(f"{strategy} resisted at score {score:.1f}")
        self.observations = self.observations[-10:]
        self.vulnerability_signals = self.vulnerability_signals[-10:]
        self.resistance_patterns = self.resistance_patterns[-10:]
        self.strategy_outcomes = self.strategy_outcomes[-10:]
