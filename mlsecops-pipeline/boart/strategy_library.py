"""Dynamic strategy library with effectiveness tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StrategyRecord:
    strategy: str
    definition: str
    representation: str
    interaction_pattern: str
    attempts: int = 0
    successes: int = 0
    score_sum: float = 0.0
    recency_tick: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def avg_score(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.score_sum / self.attempts


@dataclass(slots=True)
class StrategyLibrary:
    max_size: int
    strategies: list[StrategyRecord] = field(default_factory=list)
    _tick: int = 0

    @classmethod
    def from_json(cls, path: str | Path, max_size: int = 10) -> "StrategyLibrary":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        strategies = [
            StrategyRecord(
                strategy=item["strategy"],
                definition=item["definition"],
                representation=item.get("representation", "narrative"),
                interaction_pattern=item.get("interaction_pattern", "single-shot"),
            )
            for item in raw
        ]
        return cls(max_size=max_size, strategies=strategies[:max_size])

    def ranked(self) -> list[StrategyRecord]:
        return sorted(self.strategies, key=self._effectiveness_score, reverse=True)

    def _effectiveness_score(self, strategy: StrategyRecord) -> float:
        recency = strategy.recency_tick / max(1, self._tick)
        return strategy.success_rate * 0.5 + (strategy.avg_score / 10.0) * 0.3 + recency * 0.2

    def ranked_text(self) -> str:
        lines = []
        for idx, st in enumerate(self.ranked(), start=1):
            lines.append(
                f"{idx}. {st.strategy} | success_rate={st.success_rate:.2f}, "
                f"avg_score={st.avg_score:.2f}, attempts={st.attempts}"
            )
            lines.append(f"   definition: {st.definition}")
        return "\n".join(lines)

    def update_metrics(self, strategy_name: str, score: float) -> None:
        self._tick += 1
        strategy = self._find_or_none(strategy_name)
        if strategy is None:
            strategy = StrategyRecord(
                strategy=strategy_name,
                definition="Auto-added strategy from summarizer.",
                representation="mixed",
                interaction_pattern="adaptive",
            )
            self.strategies.append(strategy)
        strategy.attempts += 1
        strategy.score_sum += score
        strategy.recency_tick = self._tick
        if score >= 5.0:
            strategy.successes += 1
        self._trim()

    def add_generated_strategy(self, payload: dict[str, Any]) -> None:
        strategy_name = str(payload.get("strategy", "")).strip()
        if not strategy_name:
            return
        existing = self._find_or_none(strategy_name)
        if existing:
            if payload.get("definition"):
                existing.definition = str(payload["definition"])
            if payload.get("representation"):
                existing.representation = str(payload["representation"])
            if payload.get("interaction_pattern"):
                existing.interaction_pattern = str(payload["interaction_pattern"])
            return
        self.strategies.append(
            StrategyRecord(
                strategy=strategy_name,
                definition=str(payload.get("definition", "Generated from successful attack.")),
                representation=str(payload.get("representation", "mixed")),
                interaction_pattern=str(payload.get("interaction_pattern", "adaptive")),
            )
        )
        self._trim()

    def _trim(self) -> None:
        if len(self.strategies) <= self.max_size:
            return
        self.strategies = self.ranked()[: self.max_size]

    def _find_or_none(self, strategy_name: str) -> StrategyRecord | None:
        normalized = strategy_name.strip().lower()
        for item in self.strategies:
            if item.strategy.lower() == normalized:
                return item
        return None

