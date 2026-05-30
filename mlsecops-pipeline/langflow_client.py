"""HTTP-клиент Langflow (stdlib urllib, без httpx) + загрузка из файла."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import urllib_ssl_context
from flow_meta import extract_flow_metadata
from logging_utils import get_logger

log = get_logger("langflow")


@dataclass(slots=True)
class FlowFetchResult:
    """Граф для анализа + метаданные сценария для отчёта."""

    graph: dict[str, Any]
    metadata: dict[str, Any]


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def _extract_flow(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("data"), dict) and {"nodes", "edges"} <= set(payload["data"].keys()):
        return payload
    if isinstance(payload.get("flow"), dict):
        flow = payload["flow"]
        if isinstance(flow.get("data"), dict):
            return flow
    nested = payload.get("data", {}).get("data") if isinstance(payload.get("data"), dict) else None
    if isinstance(nested, dict) and {"nodes", "edges"} <= set(nested.keys()):
        return {"data": nested}
    raise ValueError("Неподдерживаемый формат ответа Langflow.")


def fetch_flow(
    langflow_url: str,
    flow_id: str,
    api_key: str | None = None,
    *,
    ssl_verify: bool | None = None,
    timeout: float = 120.0,
) -> FlowFetchResult:
    base = _normalize_url(langflow_url)
    key = (api_key or os.getenv("LANGFLOW_API_KEY") or "").strip()
    if not key:
        raise ValueError("Не задан LANGFLOW_API_KEY (аргумент или переменная окружения).")
    endpoint = f"{base}/api/v1/flows/{flow_id.strip()}"
    from config import ssl_verify as _ssl_verify

    verify = _ssl_verify("langflow", override=ssl_verify)
    log.debug("GET %s (TLS verify=%s)", endpoint, verify)
    req = urllib.request.Request(
        endpoint,
        headers={"accept": "application/json", "x-api-key": key},
        method="GET",
    )
    ctx = urllib_ssl_context(verify)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Langflow HTTP {exc.code} для flow {flow_id}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось получить flow: {endpoint}") from exc
    graph = _extract_flow(payload)
    meta = extract_flow_metadata(payload, flow_id=flow_id.strip())
    log.debug(
        "  метаданные: name=%s, endpoint_name=%s",
        meta.get("name", "—"),
        meta.get("endpoint_name", "—"),
    )
    log.debug("Ответ Langflow, ключи верхнего уровня: %s", list(payload.keys())[:12])
    return FlowFetchResult(graph=graph, metadata=meta)


def load_flow_from_file(file_path: str | Path) -> FlowFetchResult:
    """
    Загрузить flow из локального JSON-файла.

    Поддерживаемые форматы:
    - Langflow API response (с data.nodes/data.edges)
    - Langflow GraphQL response (data.flow.data.nodes/data.flow.data.edges)
    - Прямой граф (data.nodes/data.edges)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() not in (".json", ""):
        log.warning("Файл %s имеет расширение %s, ожидался .json", path, path.suffix)

    log.debug("Загрузка flow из файла: %s", path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    graph = _extract_flow(payload)
    flow_id = path.stem
    meta = extract_flow_metadata(payload, flow_id=flow_id)
    log.debug("  загружено: %s", path)
    return FlowFetchResult(graph=graph, metadata=meta)
