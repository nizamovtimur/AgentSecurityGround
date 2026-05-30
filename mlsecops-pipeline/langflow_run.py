"""POST /api/v1/run/{flow_id} — цель для BOART (black-box)."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from boart.errors import TargetCallError
from config import ssl_verify
from logging_utils import get_logger

log = get_logger("langflow_run")

load_dotenv()


def run_timeout_from_env() -> float:
    for key in ("VALIDATOR_TARGET_TIMEOUT", "LANGFLOW_RUN_TIMEOUT", "MLSECOPS_TARGET_TIMEOUT"):
        raw = (os.getenv(key) or "").strip()
        if raw:
            return max(1.0, float(raw))
    ot = (os.getenv("OPENAI_TIMEOUT") or "").strip()
    if ot:
        return max(1.0, float(ot))
    return 300.0


def extract_langflow_run_message(data: dict[str, Any]) -> str:
    try:
        msg = data["outputs"][0]["outputs"][0]["messages"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Неожиданный формат ответа Langflow run.") from exc
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        if isinstance(msg.get("text"), str):
            return msg["text"]
        return str(msg)
    return str(msg)


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def split_run_url(url: str) -> tuple[str, str]:
    marker = "/api/v1/run/"
    if marker not in url:
        raise ValueError(f"Ожидался URL вида .../api/v1/run/<FLOW_ID>: {url!r}")
    base, _, rest = url.partition(marker)
    flow_id = rest.strip("/").split("/", 1)[0]
    if not flow_id:
        raise ValueError(f"Нет flow id в URL: {url!r}")
    return _normalize_url(base), flow_id


def run_endpoint_url(base_url: str, flow_id: str) -> str:
    return f"{_normalize_url(base_url)}/api/v1/run/{flow_id.strip('/')}"


def resolve_target_endpoint(
    *,
    explicit: str | None = None,
    langflow_url: str | None = None,
    flow_id: str | None = None,
) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    base = (langflow_url or os.getenv("LANGFLOW_URL") or "").strip().rstrip("/")
    fid = (flow_id or os.getenv("FLOW_ID") or "").strip()
    if base and fid:
        return run_endpoint_url(base, fid)
    raise ValueError(
        "Укажите TARGET_ENDPOINT или пару LANGFLOW_URL + FLOW_ID."
    )


@dataclass(slots=True)
class LangflowRunConfig:
    url: str
    flow_id: str
    api_key: str = ""
    run_timeout_seconds: float = field(default_factory=run_timeout_from_env)
    verify_ssl: bool = True


class LangflowRunClient:
    """Один POST на каждый шаг BOART (новый session_id)."""

    def __init__(self, config: LangflowRunConfig) -> None:
        self.config = config

    @classmethod
    def from_run_endpoint(
        cls,
        endpoint: str,
        *,
        api_key: str | None = None,
        run_timeout_seconds: float | None = None,
        verify_ssl: bool | None = None,
    ) -> LangflowRunClient:
        base, flow_id = split_run_url(endpoint.strip())
        key = (api_key if api_key is not None else os.getenv("LANGFLOW_API_KEY") or "").strip()
        rt = run_timeout_seconds if run_timeout_seconds is not None else run_timeout_from_env()
        verify = ssl_verify("langflow") if verify_ssl is None else verify_ssl
        return cls(
            LangflowRunConfig(
                url=base,
                flow_id=flow_id,
                api_key=key,
                run_timeout_seconds=rt,
                verify_ssl=verify,
            )
        )

    def send(self, message: str, history: list[dict[str, str]]) -> str:
        _ = history
        url = run_endpoint_url(self.config.url, self.config.flow_id)
        payload: dict[str, Any] = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": message,
            "session_id": str(uuid.uuid4()),
        }
        headers = {"x-api-key": self.config.api_key} if self.config.api_key else {}
        timeout = httpx.Timeout(self.config.run_timeout_seconds)
        log.debug("POST %s (%s симв.)", url, len(message))
        try:
            with httpx.Client(verify=self.config.verify_ssl, timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TargetCallError(
                f"Таймаут Langflow ({self.config.run_timeout_seconds:.0f} с)",
                url=url,
            ) from exc
        except httpx.HTTPError as exc:
            raise TargetCallError(f"Сеть Langflow: {exc}", url=url) from exc

        if response.status_code >= 400:
            raise TargetCallError(
                f"Langflow run: HTTP {response.status_code}",
                status_code=response.status_code,
                url=url,
                body=response.text,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise TargetCallError(
                "Langflow: ответ не JSON",
                status_code=response.status_code,
                url=url,
                body=response.text,
            ) from exc
        if not isinstance(data, dict):
            return str(data)
        try:
            return extract_langflow_run_message(data)
        except ValueError as exc:
            raise TargetCallError(
                str(exc),
                status_code=response.status_code,
                url=url,
                body=str(data)[:800],
            ) from exc
