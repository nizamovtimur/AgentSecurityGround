"""Отчёт MAESTRO по графу Langflow (один вызов LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from llm.openai_client import OpenAIClient
from models.security_graph import SecurityGraph
from services.synopsis_builder import build_security_synopsis


def _read_text(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        return handle.read()


class ThreatModelingService:
    """MAESTRO: системный промпт из шаблонов + краткий user-запрос → Markdown."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        threat_model_path: str | Path,
        system_prompt_path: str | Path,
    ) -> None:
        self.openai_client = openai_client
        self.threat_model_context_text = _read_text(threat_model_path)
        self.system_prompt_template = _read_text(system_prompt_path)

    def _build_system_prompt(self, synopsis_json: str) -> str:
        prompt = self.system_prompt_template.replace(
            "<THREAT_MODEL_CONTEXT>", self.threat_model_context_text.strip()
        )
        return prompt.replace("<JSON>", synopsis_json)

    def generate_report(self, graph: SecurityGraph) -> str:
        synopsis = build_security_synopsis(graph)
        synopsis_json = json.dumps(synopsis, ensure_ascii=False, indent=2)
        system_prompt = self._build_system_prompt(synopsis_json)
        return self.openai_client.complete(
            system_prompt=system_prompt,
            user_prompt=(
                "Используя фреймворк MAESTRO и JSON графа workflow из системного сообщения, "
                "сформируй полный отчёт в Markdown на русском языке со всеми обязательными "
                "разделами (0–7). Не выдумывай узлы, отсутствующие в JSON."
            ),
        )

