"""Langflow flow fetcher for analyzing deployed (preprod) configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class LangflowConfig:
    url: str
    flow_id: str
    api_key: str
    timeout_seconds: float = 30.0


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _extract_flow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return flow JSON shape expected by parse_langflow_flow().

    Supports both:
    - direct flow object: {"data": {"nodes": ..., "edges": ...}}
    - wrapped API payloads: {"flow": {...}} or {"data": {"data": {...}}}
    """
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


class LangflowFlowClient:
    """Fetches flow JSON directly from Langflow API."""

    def __init__(self, config: LangflowConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        *,
        url: str | None = None,
        flow_id: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> "LangflowFlowClient":
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
        return cls(
            LangflowConfig(
                url=_normalize_url(resolved_url),
                flow_id=resolved_flow_id,
                api_key=resolved_api_key,
                timeout_seconds=timeout_seconds,
            )
        )

    def fetch_raw(self) -> dict[str, Any]:
        endpoint = f"{self.config.url}/api/v1/flows/{self.config.flow_id}"
        headers = {
            "accept": "application/json",
            "x-api-key": self.config.api_key,
        }
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
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

