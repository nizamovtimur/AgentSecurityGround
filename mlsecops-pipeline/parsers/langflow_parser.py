"""Deterministic parser for Langflow flow JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.security_graph import SecurityEdge, SecurityGraph, SecurityNode

ENTRYPOINT_TYPES = {"ChatInput", "URL", "File", "APIRequest", "MCPTools"}
CONTROL_TYPES = {"GuardrailValidator", "Regex", "FilterData", "ParserComponent"}
SENSITIVE_FIELD_MARKERS = ("api_key", "token", "secret", "password")
SYSTEM_PROMPT_FIELD_MARKERS = ("system_prompt", "system_message", "instructions", "prompt")
MAX_TEMPLATE_STRING_LEN = 400


def _safe_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_TEMPLATE_STRING_LEN:
        return value[:MAX_TEMPLATE_STRING_LEN] + "...[truncated]"
    return value


def _is_tool_mode_enabled(template: dict[str, Any]) -> bool:
    for item in template.values():
        if not isinstance(item, dict):
            continue
        key_name = _safe_str(item.get("name")).lower()
        input_type = _safe_str(item.get("_input_type")).lower()
        if key_name == "tool_mode" or input_type == "boolinput":
            if bool(item.get("value")) and key_name == "tool_mode":
                return True
    return False


def _has_system_prompt_fields(template: dict[str, Any]) -> bool:
    for key, value in template.items():
        lowered = key.lower()
        if not any(marker in lowered for marker in SYSTEM_PROMPT_FIELD_MARKERS):
            continue
        if isinstance(value, dict) and value.get("value"):
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _get_node_role(node_type: str) -> str:
    lowered = node_type.lower()
    if "agent" in lowered:
        return "agent"
    if node_type in ENTRYPOINT_TYPES or "input" in lowered or "output" in lowered:
        return "io"
    if "model" in lowered or "embedding" in lowered:
        return "model"
    if "guardrail" in lowered or node_type in CONTROL_TYPES:
        return "guardrail"
    if "tool" in lowered or "search" in lowered or "mcp" in lowered:
        return "tool"
    if "data" in lowered or "chroma" in lowered or "vector" in lowered:
        return "data"
    return "component"


def _normalize_template_fields(template: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized: dict[str, Any] = {}
    risk_flags: list[str] = []
    for key, value in template.items():
        lowered = key.lower()
        if key == "code":
            continue
        if isinstance(value, dict):
            is_sensitive = any(marker in lowered for marker in SENSITIVE_FIELD_MARKERS)
            normalized[key] = {
                "type": _safe_str(value.get("_input_type")),
                "required": bool(value.get("required", False)),
                "password": bool(value.get("password", False)),
                "has_value": bool(value.get("value")),
                "value": (
                    "***REDACTED***"
                    if is_sensitive or value.get("password")
                    else _truncate_value(value.get("value"))
                ),
            }
            if is_sensitive or value.get("password"):
                risk_flags.append(f"sensitive_field:{key}")
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = _truncate_value(value)
    return normalized, sorted(set(risk_flags))


def parse_langflow_flow(flow_data: dict[str, Any]) -> SecurityGraph:
    """Parse Langflow JSON into normalized graph used by threat modeling."""
    data = flow_data.get("data", {})
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])

    nodes: list[SecurityNode] = []
    node_types: dict[str, str] = {}
    entrypoints: set[str] = set()
    controls: set[str] = set()

    for raw_node in raw_nodes:
        node_data = raw_node.get("data", {})
        node_meta = node_data.get("node", {})
        node_id = _safe_str(node_data.get("id") or raw_node.get("id"))
        display_name = _safe_str(node_meta.get("display_name") or node_meta.get("name") or node_id)
        node_type = _safe_str(node_data.get("type") or node_meta.get("name") or "Unknown")
        template = node_data.get("node", {}).get("template", {}) if isinstance(node_meta, dict) else {}
        role = _get_node_role(node_type)
        template_dict = template if isinstance(template, dict) else {}
        tool_mode_enabled = _is_tool_mode_enabled(template_dict)

        template_fields, template_risks = _normalize_template_fields(template_dict)
        risk_flags = list(template_risks)
        if tool_mode_enabled:
            risk_flags.append("tool_mode_enabled")
        if _has_system_prompt_fields(template_dict):
            risk_flags.append("system_prompt_surface")
        if role == "agent":
            risk_flags.append("agent_node")
        if node_type in ENTRYPOINT_TYPES:
            entrypoints.add(node_id)
        if node_type in CONTROL_TYPES or "guardrail" in node_type.lower():
            controls.add(node_id)

        nodes.append(
            SecurityNode(
                node_id=node_id,
                display_name=display_name,
                node_type=node_type,
                role=role,
                template_fields=template_fields,
                risk_flags=sorted(set(risk_flags)),
            )
        )
        node_types[node_id] = node_type

    edges: list[SecurityEdge] = []
    tool_component_sources: set[str] = set()
    system_prompt_targets: set[str] = set()
    for raw_edge in raw_edges:
        edge_data = raw_edge.get("data", {})
        src = _safe_str(raw_edge.get("source"))
        dst = _safe_str(raw_edge.get("target"))
        src_handle = edge_data.get("sourceHandle", {}) if isinstance(edge_data, dict) else {}
        dst_handle = edge_data.get("targetHandle", {}) if isinstance(edge_data, dict) else {}
        source_data_type = _safe_str(src_handle.get("dataType")) or node_types.get(src)
        source_handle_name = _safe_str(src_handle.get("name"))
        target_field_name = _safe_str(dst_handle.get("fieldName"))
        target_input_types = dst_handle.get("inputTypes", []) if isinstance(dst_handle.get("inputTypes", []), list) else []
        target_field_name_lower = target_field_name.lower()
        source_handle_name_lower = source_handle_name.lower()

        if source_handle_name_lower == "component_as_tool" and target_field_name_lower == "tools":
            tool_component_sources.add(src)
        if target_field_name_lower in {"system_prompt", "system_message"}:
            system_prompt_targets.add(dst)

        edges.append(
            SecurityEdge(
                source=src,
                target=dst,
                source_data_type=source_data_type or None,
                source_handle_name=source_handle_name or None,
                target_field_name=target_field_name or None,
                target_input_types=target_input_types,
            )
        )

    assets: set[str] = set()
    for node in nodes:
        if node.node_id in tool_component_sources:
            node.role = "tool"
            node.risk_flags = sorted(set(node.risk_flags + ["tool_attached_to_agent"]))
        if node.node_id in system_prompt_targets:
            node.risk_flags = sorted(set(node.risk_flags + ["system_prompt_surface"]))
        if node.role in {"agent", "tool", "data", "model"}:
            assets.add(f"{node.display_name}::{node.role}")

    return SecurityGraph(
        nodes=nodes,
        edges=edges,
        entrypoints=sorted(entrypoints),
        controls=sorted(controls),
        assets=sorted(assets),
    )


def parse_langflow_file(path: str | Path) -> SecurityGraph:
    flow_path = Path(path)
    with flow_path.open("r", encoding="utf-8") as handle:
        flow_data = json.load(handle)
    return parse_langflow_flow(flow_data)

