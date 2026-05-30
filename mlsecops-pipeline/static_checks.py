"""Статические проверки (только литеральные секреты и метаданные доступа)."""

from __future__ import annotations

import re
from typing import Any

_SECRET_RX = [
    re.compile(p, re.I) for p in [
        r"(?:api[_\-]?key|secret|token|password)\s*[:=]\s*\S{6,}",
        r"\b(sk-[a-zA-Z0-9\-_.]{10,})\b",
        r"\b(ghp_[a-zA-Z0-9]{20,})\b",
        r"[a-z][a-z0-9+\-.]{1,15}://[^\s\"'<>]{6,}@[^\s\"'<>]+",
    ]
]
_ACCESS_RX = [
    re.compile(p, re.I) for p in [
        r"\b(?:role|permission|privilege|access[_\-]?level)\s*[:=]\s*\S",
        r"\b(?:limit|quota|rate[_\-]?limit)\s*[:=]\s*\d",
        r"\b(?:is[_\-]?admin|superuser)\s*[:=]\s*(?:true|1|yes)\b",
    ]
]


def _scan_text(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    return [p.pattern[:40] for p in patterns if p.search(text)]


def static_data_min_findings(synopsis: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for sp in synopsis.get("system_prompts", []):
        text = sp.get("text") or ""
        if _scan_text(text, _SECRET_RX):
            out.append({
                "node": sp.get("node_name", ""),
                "issue": "literal_secret_pattern",
                "field": sp.get("field", ""),
            })
    return out


def static_meta_findings(synopsis: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for sp in synopsis.get("system_prompts", []):
        text = sp.get("text") or ""
        if _scan_text(text, _ACCESS_RX):
            out.append({
                "node": sp.get("node_name", ""),
                "issue": "access_metadata_in_prompt",
                "field": sp.get("field", ""),
            })
    return out


def static_least_privilege_hints(synopsis: dict[str, Any]) -> list[str]:
    tools = synopsis.get("tool_edges", [])
    types = {t.get("source_type") for t in tools if t.get("source_type")}
    hints: list[str] = []
    if len(tools) >= 3 and len(types) >= 2:
        hints.append(f"агент подключён к {len(tools)} инструментам разных типов ({', '.join(sorted(types))})")
    mcp = [t for t in tools if t.get("source_type") and "mcp" in str(t["source_type"]).lower()]
    if len(mcp) >= 2:
        names = [t.get("source_name") for t in mcp]
        hints.append(f"несколько MCP без разделения: {', '.join(names)}")
    return hints
