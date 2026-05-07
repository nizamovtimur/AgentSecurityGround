"""Target client protocol for BOART: Langflow via ``services.langflow_client``, else generic JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from services.langflow_client import (
    LangflowClient,
    extract_langflow_run_message,
    langflow_run_timeout_from_env,
)


def http_target_timeout_from_env() -> float:
    """Backward-compatible alias for :func:`langflow_run_timeout_from_env`."""
    return langflow_run_timeout_from_env()


class TargetClient(Protocol):
    def send(self, message: str, history: list[dict[str, str]]) -> str:
        """Send attack message to target and return textual response."""


def _is_langflow_run_url(url: str) -> bool:
    return "/api/v1/run/" in url


@dataclass(slots=True)
class _GenericHttpJsonTarget:
    """POST ``{message, history}`` for non-Langflow integration tests or custom targets."""

    endpoint: str
    timeout_seconds: float = field(default_factory=langflow_run_timeout_from_env)
    verify_ssl: bool = True

    def send(self, message: str, history: list[dict[str, str]]) -> str:
        payload = {"message": message, "history": history}
        timeout = httpx.Timeout(self.timeout_seconds)
        with httpx.Client(verify=self.verify_ssl, timeout=timeout) as client:
            response = client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            return str(data)
        for key in ("response", "answer", "output", "message"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return str(data)


@dataclass(slots=True)
class HttpTargetClient:
    """BOART entry: Langflow run URL → :class:`LangflowClient`; otherwise generic JSON body.

    ``timeout_seconds`` applies to the HTTP read (Langflow run uses
    :func:`langflow_run_timeout_from_env` by default).
    """

    endpoint: str
    timeout_seconds: float = field(default_factory=langflow_run_timeout_from_env)
    api_key: str | None = None
    verify_ssl: bool = True
    _delegate: TargetClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if _is_langflow_run_url(self.endpoint):
            self._delegate = LangflowClient.from_run_endpoint(
                self.endpoint,
                api_key=self.api_key,
                run_timeout_seconds=self.timeout_seconds,
                verify=self.verify_ssl,
            )
        else:
            self._delegate = _GenericHttpJsonTarget(
                endpoint=self.endpoint,
                timeout_seconds=self.timeout_seconds,
                verify_ssl=self.verify_ssl,
            )

    def send(self, message: str, history: list[dict[str, str]]) -> str:
        return self._delegate.send(message, history)


__all__ = [
    "TargetClient",
    "HttpTargetClient",
    "extract_langflow_run_message",
    "http_target_timeout_from_env",
]
