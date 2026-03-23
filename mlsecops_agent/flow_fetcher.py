"""
Загрузка флоу из Langflow API или локального файла.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

log = logging.getLogger("mlsecops_agent.flow_fetcher")


def fetch_flow_from_file(path):
    """
    Загружает флоу из локального JSON-файла (экспорт Langflow).

    Поддерживает формат с ключом "data" (nodes, edges) или сырой граф.
    Используется для анализа без доступа к Langflow API.

    Args:
        path: Путь к JSON-файлу (например, langflow/flows/Windchaser.json)

    Returns:
        dict с ключами data, name, description (или data: {...} для сырого графа)

    Raises:
        FileNotFoundError: если файл не найден
        json.JSONDecodeError: при ошибке парсинга JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Flow file not found: {}".format(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    log.debug("Loaded flow from file: %s", path)
    if "data" in data:
        return data
    return {"data": data, "name": "Local", "description": ""}


def fetch_flow(
    flow_id: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Загружает флоу по FLOW_ID через Langflow API.

    Выполняет GET /api/v1/flows/{flow_id}. Ответ содержит метаданные
    флоу и граф (nodes, edges) в поле data.

    Args:
        flow_id: UUID флоу в Langflow
        base_url: URL Langflow (по умолчанию LANGFLOW_URL или http://localhost:7860)
        api_key: API ключ. Берётся из LANGFLOW_API_KEY

    Returns:
        dict с полями: id, name, description, data (nodes, edges), ...

    Raises:
        ValueError: если api_key не задан
        httpx.HTTPStatusError: при ошибке HTTP (404, 401, etc.)
    """
    url = (base_url or os.environ.get("LANGFLOW_URL", "http://localhost:7860")).rstrip("/")
    key = api_key or os.environ.get("LANGFLOW_API_KEY")

    if not key:
        raise ValueError("LANGFLOW_API_KEY must be set")
    flow_id = str(flow_id).strip()
    if not flow_id:
        raise ValueError("flow_id cannot be empty")

    resp = httpx.get(
        "{}/api/v1/flows/{}".format(url, flow_id),
        headers={"accept": "application/json", "x-api-key": key},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    log.info("Fetched flow %s: %s", flow_id, data.get("name", "Unknown"))
    return data
