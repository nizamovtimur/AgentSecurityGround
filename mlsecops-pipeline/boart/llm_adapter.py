"""Адаптер LLMClient под интерфейс BOART (system_prompt / user_prompt)."""

from __future__ import annotations

from typing import Protocol

from llm import LLMClient


class BoartLLM(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class LLMClientAdapter:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._client.complete(system=system_prompt, user=user_prompt)
