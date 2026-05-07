"""Single HTTP client for Langflow: flow export (GET) and chat run (POST /api/v1/run/...)."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from logging_utils import get_logger

logger = get_logger(__name__)


def langflow_run_timeout_from_env() -> float:
    """Read timeout for POST /api/v1/run (BOART, slow models).

    Precedence: ``MLSECOPS_TARGET_TIMEOUT`` → ``LANGFLOW_RUN_TIMEOUT`` → ``OPENAI_TIMEOUT`` → ``300``.
    """
    for key in ("MLSECOPS_TARGET_TIMEOUT", "LANGFLOW_RUN_TIMEOUT"):
        raw = (os.getenv(key) or "").strip()
        if raw:
            return max(1.0, float(raw))
    ot = (os.getenv("OPENAI_TIMEOUT") or "").strip()
    if ot:
        return max(1.0, float(ot))
    return 300.0


def extract_langflow_run_message(data: dict[str, Any]) -> str:
    """Parse assistant text from Langflow POST /api/v1/run/{flow_id} JSON (chat mode)."""
    try:
        msg = data["outputs"][0]["outputs"][0]["messages"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Langflow run response shape (no outputs/.../messages/message).") from exc
    if isinstance(msg, str):
        return msg
    if isinstance(msg, dict):
        if isinstance(msg.get("text"), str):
            return msg["text"]
        return str(msg)
    return str(msg)


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def split_langflow_run_url(url: str) -> tuple[str, str]:
    """Split ``http://host/api/v1/run/<flow_id>`` → (``http://host``, ``flow_id``)."""
    marker = "/api/v1/run/"
    if marker not in url:
        raise ValueError(f"Not a Langflow run URL (expected '.../api/v1/run/<FLOW_ID>'): {url!r}")
    base, _, rest = url.partition(marker)
    flow_id = rest.strip("/").split("/", 1)[0]
    if not flow_id:
        raise ValueError(f"Missing flow id in run URL: {url!r}")
    return _normalize_url(base), flow_id


def run_endpoint_url(base_url: str, flow_id: str) -> str:
    return f"{_normalize_url(base_url)}/api/v1/run/{flow_id.strip('/')}"


def _extract_flow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return flow JSON shape expected by parse_langflow_flow()."""
    if isinstance(payload.get("data"), dict) and {"nodes", "edges"} <= set(payload["data"].keys()):
        return payload
    if isinstance(payload.get("flow"), dict):
        flow = payload["flow"]
        if isinstance(flow.get("data"), dict):
            return flow
    if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("data"), dict):
        nested = payload["data"]["data"]
        if {"nodes", "edges"} <= set(nested.keys()):
            return {"data": nested}
    raise ValueError("Unsupported Langflow flow response format.")


@dataclass(slots=True)
class LangflowConfig:
    """Connection parameters for one Langflow deployment + flow id."""

    url: str
    flow_id: str
    api_key: str
    timeout_seconds: float = 120.0
    run_timeout_seconds: float = field(default_factory=langflow_run_timeout_from_env)
    verify: bool = True


class LangflowClient:
    """Langflow HTTP API: GET flow JSON (S1) and POST chat run (BOART / black-box)."""

    def __init__(self, config: LangflowConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        *,
        url: str | None = None,
        flow_id: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        run_timeout_seconds: float | None = None,
        verify: bool = True,
    ) -> LangflowClient:
        load_dotenv()
        resolved_url = url or os.getenv("LANGFLOW_URL", "")
        resolved_flow_id = flow_id or os.getenv("FLOW_ID", "")
        resolved_api_key = api_key or os.getenv("LANGFLOW_API_KEY", "")
        if not resolved_url:
            raise ValueError("LANGFLOW_URL is not set.")
        if not resolved_flow_id:
            raise ValueError("FLOW_ID is not set.")
        if not resolved_api_key:
            raise ValueError("LANGFLOW_API_KEY is not set.")
        rt = run_timeout_seconds if run_timeout_seconds is not None else langflow_run_timeout_from_env()
        return cls(
            LangflowConfig(
                url=_normalize_url(resolved_url),
                flow_id=resolved_flow_id,
                api_key=resolved_api_key,
                timeout_seconds=timeout_seconds,
                run_timeout_seconds=rt,
                verify=verify,
            )
        )

    @classmethod
    def from_run_endpoint(
        cls,
        endpoint: str,
        *,
        api_key: str | None = None,
        run_timeout_seconds: float | None = None,
        verify: bool = True,
    ) -> LangflowClient:
        """Build client from full run URL (same as BOART ``--target-endpoint``)."""
        base, flow_id = split_langflow_run_url(endpoint.strip())
        resolved_key = (api_key if api_key is not None else os.getenv("LANGFLOW_API_KEY") or "").strip()
        rt = run_timeout_seconds if run_timeout_seconds is not None else langflow_run_timeout_from_env()
        return cls(
            LangflowConfig(
                url=base,
                flow_id=flow_id,
                api_key=resolved_key,
                timeout_seconds=120.0,
                run_timeout_seconds=rt,
                verify=verify,
            )
        )

    def fetch_raw(self) -> dict[str, Any]:
        endpoint = f"{self.config.url}/api/v1/flows/{self.config.flow_id}"
        headers = {
            "accept": "application/json",
            "x-api-key": self.config.api_key,
        }
        try:
            with httpx.Client(verify=self.config.verify, timeout=self.config.timeout_seconds) as client:
                response = client.get(endpoint, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            raise RuntimeError(f"Langflow API returned HTTP {status} for flow '{self.config.flow_id}'.") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch flow from Langflow endpoint: {endpoint}") from exc

        logger.info("Fetched flow %s from Langflow.", self.config.flow_id)
        return payload

    def fetch_flow(self) -> dict[str, Any]:
        payload = self.fetch_raw()
        try:
            return _extract_flow_payload(payload)
        except ValueError as exc:
            raise RuntimeError("Langflow response format is unsupported for flow parsing.") from exc

    def send(self, message: str, history: list[dict[str, str]]) -> str:
        """BOART target: POST ``/api/v1/run/{flow_id}`` (chat contract, new ``session_id`` per call)."""
        _ = history
        session_id = str(uuid.uuid4())
        url = run_endpoint_url(self.config.url, self.config.flow_id)
        payload: dict[str, Any] = {
            "output_type": "chat",
            "input_type": "chat",
            "input_value": message,
            "session_id": session_id,
        }
        key = self.config.api_key.strip()
        headers = {"x-api-key": key} if key else {}

        timeout = httpx.Timeout(self.config.run_timeout_seconds)
        with httpx.Client(verify=self.config.verify, timeout=timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            return str(data)
        return extract_langflow_run_message(data)


# Backward-compatible name used in notebooks and early imports
LangflowFlowClient = LangflowClient
