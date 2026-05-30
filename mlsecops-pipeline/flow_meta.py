"""Метаданные сценария Langflow (корень экспорта / API), без графа."""

from __future__ import annotations

import json
from typing import Any, Mapping

META_KEYS: tuple[str, ...] = (
    "name",
    "id",
    "description",
    "endpoint_name",
    "tags",
    "last_tested_version",
    "is_component",
    "locked",
)


def extract_flow_metadata(raw: Mapping[str, Any] | None, *, flow_id: str = "") -> dict[str, Any]:
    """Поля ``name``, ``id``, ``endpoint_name``, … из корня ответа Langflow."""
    if raw is None or not isinstance(raw, dict):
        return {"id": flow_id} if flow_id else {}

    sources: list[Mapping[str, Any]] = [raw]
    flow = raw.get("flow")
    if isinstance(flow, dict):
        sources.append(flow)

    out: dict[str, Any] = {}
    for key in META_KEYS:
        for src in sources:
            if key in src:
                out[key] = src[key]
                break
    if flow_id and not out.get("id"):
        out["id"] = flow_id
    return out


def format_agent_outputs_markdown(meta: Mapping[str, Any]) -> str:
    """Таблица «Выходные данные агента» для итогового отчёта."""
    if not meta:
        return (
            "## Выходные данные агента\n\n"
            "*Метаданные сценария в ответе Langflow не найдены "
            "(`name`, `id`, `endpoint_name`, …).*\n"
        )

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, str) and not value.strip():
            return "—"
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False).replace("|", "\\|")
        return str(value).replace("|", "\\|")

    rows = ["## Выходные данные агента", "", "| Поле | Значение |", "|------|-----------|"]
    for key in META_KEYS:
        if key not in meta:
            continue
        rows.append(f"| `{key}` | {_fmt(meta[key])} |")
    rows.append("")
    return "\n".join(rows)
