"""Клиент цели BOART: Langflow run URL или generic JSON POST."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from boart.errors import TargetCallError
from langflow_run import LangflowRunClient, extract_langflow_run_message, run_timeout_from_env


class TargetClient(Protocol):
    def send(self, message: str, history: list[dict[str, str]]) -> str: ...


def _is_langflow_run_url(url: str) -> bool:
    return "/api/v1/run/" in url


@dataclass(slots=True)
class _GenericHttpJsonTarget:
    endpoint: str
    timeout_seconds: float = field(default_factory=run_timeout_from_env)
    verify_ssl: bool = True

    def send(self, message: str, history: list[dict[str, str]]) -> str:
        payload = {"message": message, "history": history}
        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            with httpx.Client(verify=self.verify_ssl, timeout=timeout) as client:
                response = client.post(self.endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise TargetCallError(
                f"Таймаут цели ({self.timeout_seconds:.0f} с)",
                url=self.endpoint,
            ) from exc
        except httpx.HTTPError as exc:
            raise TargetCallError(f"Сеть: {exc}", url=self.endpoint) from exc
        if response.status_code >= 400:
            raise TargetCallError(
                f"HTTP {response.status_code}",
                status_code=response.status_code,
                url=self.endpoint,
                body=response.text,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise TargetCallError(
                "Ответ цели не JSON",
                status_code=response.status_code,
                url=self.endpoint,
                body=response.text,
            ) from exc
        if not isinstance(data, dict):
            return str(data)
        for key in ("response", "answer", "output", "message"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return str(data)


@dataclass(slots=True)
class HttpTargetClient:
    endpoint: str
    timeout_seconds: float = field(default_factory=run_timeout_from_env)
    api_key: str | None = None
    verify_ssl: bool = True
    _delegate: TargetClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if _is_langflow_run_url(self.endpoint):
            self._delegate = LangflowRunClient.from_run_endpoint(
                self.endpoint,
                api_key=self.api_key,
                run_timeout_seconds=self.timeout_seconds,
                verify_ssl=self.verify_ssl,
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
]
