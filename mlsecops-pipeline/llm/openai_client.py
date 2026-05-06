"""OpenAI wrapper for pipeline services."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from logging_utils import get_logger
from openai import OpenAI

logger = get_logger(__name__)


@dataclass(slots=True)
class OpenAIConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    """HTTP read timeout in seconds (local/slow backends often need 180–600+)."""
    timeout: float = 300.0
    """Retries for transient errors (OpenAI SDK); see OPENAI_MAX_RETRIES."""
    max_retries: int = 5
    base_url: str = "https://api.openai.com/v1"


class OpenAIClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment.")
        self.config = config or OpenAIConfig()
        timeout = float(os.getenv("OPENAI_TIMEOUT", str(self.config.timeout)))
        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", str(self.config.max_retries)))
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or self.config.base_url
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.exception("OpenAI completion call failed.")
            raise RuntimeError("OpenAI completion call failed.") from exc
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content

