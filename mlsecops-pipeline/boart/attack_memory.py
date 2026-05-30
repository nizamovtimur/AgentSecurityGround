"""Память успешных атак между целями."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemoryItem:
    goal: str
    strategy: str
    attack_prompt: str
    target_response: str
    score: float


@dataclass(slots=True)
class AttackMemory:
    max_items: int = 10
    items: list[MemoryItem] = field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self.items = self.items[-self.max_items :]

    def to_text(self) -> str:
        if not self.items:
            return "No successful attacks yet."
        chunks = []
        for idx, item in enumerate(self.items[-self.max_items :], start=1):
            chunks.append(
                f"{idx}) goal={item.goal[:120]} | strategy={item.strategy} | "
                f"score={item.score:.1f} | prompt={item.attack_prompt[:160]}"
            )
        return "\n".join(chunks)
