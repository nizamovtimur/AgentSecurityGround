"""Build compact security synopsis for LLM context."""

from __future__ import annotations

from typing import Any

from models.security_graph import SecurityGraph

CORE_META_PARAMETER_KEYS = {
    "system_prompt",
    "system_message",
    "instructions",
    "template",
    "prompt",
    "tool_mode",
    "model",
    "model_name",
    "temperature",
    "max_tokens",
}


def _node_brief(node: dict[str, Any]) -> dict[str, Any]:
    template_fields = node.get("template_fields", {})
    author_parameters = {
        key: value
        for key, value in template_fields.items()
        if key in CORE_META_PARAMETER_KEYS
    }
    return {
        "id": node["node_id"],
        "name": node["display_name"],
        "type": node["node_type"],
        "role": node["role"],
        "risk_flags": node["risk_flags"],
        "author_parameters": author_parameters,
    }


def _meta_text(meta: dict[str, Any], key: str) -> str | None:
    value = meta.get(key)
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, str) and raw.strip():
            return raw
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _node_meta_text(node: dict[str, Any], key: str) -> str | None:
    return _meta_text(node.get("author_parameters", {}), key)


def build_security_synopsis(graph: SecurityGraph) -> dict[str, Any]:
    """Return compact, deterministic *factual* synopsis from Langflow graph."""
    graph_dict = graph.to_dict()
    nodes = [_node_brief(node) for node in graph_dict["nodes"]]
    edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "source_data_type": edge["source_data_type"],
            "source_handle_name": edge["source_handle_name"],
            "target_field": edge["target_field_name"],
            "target_input_types": edge["target_input_types"],
        }
        for edge in graph_dict["edges"]
    ]

    node_by_id = {node["id"]: node for node in nodes}
    system_prompts: list[dict[str, Any]] = []
    tool_edges: list[dict[str, Any]] = []
    prompt_edges: list[dict[str, Any]] = []

    for node in nodes:
        meta = node.get("author_parameters", {})
        for field in ("system_prompt", "system_message", "instructions"):
            text = _meta_text(meta, field)
            if text:
                system_prompts.append(
                    {
                        "node_id": node["id"],
                        "node_name": node["name"],
                        "node_type": node["type"],
                        "field": field,
                        "text": text,
                    }
                )
    for edge in edges:
        source_handle = str(edge.get("source_handle_name") or "").lower()
        target_field = str(edge.get("target_field") or "").lower()
        source_node = node_by_id.get(edge["source"])
        target_node = node_by_id.get(edge["target"])

        if source_handle == "component_as_tool" and target_field == "tools":
            tool_edges.append(
                {
                    "source_id": edge["source"],
                    "source_name": source_node["name"] if source_node else edge["source"],
                    "source_type": source_node["type"] if source_node else None,
                    "target_id": edge["target"],
                    "target_name": target_node["name"] if target_node else edge["target"],
                    "target_type": target_node["type"] if target_node else None,
                    "source_handle": edge["source_handle_name"],
                    "target_field": edge["target_field"],
                    "target_input_types": edge["target_input_types"],
                }
            )

        if target_field in {"system_prompt", "system_message"}:
            prompt_edges.append(
                {
                    "source_id": edge["source"],
                    "source_name": source_node["name"] if source_node else edge["source"],
                    "source_type": source_node["type"] if source_node else None,
                    "target_id": edge["target"],
                    "target_name": target_node["name"] if target_node else edge["target"],
                    "target_type": target_node["type"] if target_node else None,
                    "source_handle": edge["source_handle_name"],
                    "target_field": edge["target_field"],
                    "target_input_types": edge["target_input_types"],
                }
            )
            if source_node:
                dynamic_text = (
                    _node_meta_text(source_node, "template")
                    or _node_meta_text(source_node, "prompt")
                    or _node_meta_text(source_node, "instructions")
                )
                if dynamic_text:
                    system_prompts.append(
                        {
                            "node_id": source_node["id"],
                            "node_name": source_node["name"],
                            "node_type": source_node["type"],
                            "field": f"dynamic->{target_field}",
                            "text": dynamic_text,
                        }
                    )

    base: dict[str, Any] = {
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entrypoint_count": len(graph.entrypoints),
            "control_count": len(graph.controls),
        },
        "entrypoints": graph.entrypoints,
        "controls": graph.controls,
        "assets": graph.assets,
        "nodes": nodes,
        "edges": edges,
        "system_prompts": sorted(system_prompts, key=lambda item: (item["node_name"], item["field"])),
        "prompt_edges": sorted(prompt_edges, key=lambda item: (item["target_name"], item["source_name"])),
        "tool_edges": sorted(tool_edges, key=lambda item: (item["target_name"], item["source_name"])),
        "author_parameters_by_node": [
            {
                "node_id": node["id"],
                "node_name": node["name"],
                "node_type": node["type"],
                "parameters": node["author_parameters"],
            }
            for node in nodes
            if node["author_parameters"]
        ],
    }
    return base


_MISSION_HEADERS = (
    "## 1. Миссия",
    "## 1. Mission",
    "# 1. Миссия",
)


_MAX_MISSION_SENTENCES = 2
_MAX_SYSPROMPT_CHARS = 120


def _first_sentences(text: str, n: int) -> str:
    """Return first n sentences from text."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s.strip() for s in sentences[:n] if s.strip())


def _extract_mission_from_threat_model(markdown: str) -> str:
    """First paragraph of '## 1. Миссия' section, trimmed to 2 sentences."""
    lines = markdown.splitlines()
    in_section = False
    section_lines: list[str] = []
    for line in lines:
        if any(line.strip().startswith(h) for h in _MISSION_HEADERS):
            in_section = True
            continue
        if in_section:
            if line.startswith("## ") or line.startswith("# "):
                break
            section_lines.append(line)
    text = "\n".join(section_lines).strip()
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            return _first_sentences(para, _MAX_MISSION_SENTENCES)
    return ""


def build_target_description(synopsis: dict[str, Any], threat_model_markdown: str) -> str:
    """One compact paragraph for BoartConfig.target_description.

    Boss and Attacker get: what the system does (mission, ≤2 sentences),
    what tools it has, what it's called — nothing more.
    """
    lines: list[str] = []

    mission = _extract_mission_from_threat_model(threat_model_markdown)
    if mission:
        lines.append(mission)

    # First system prompt: role hint only (first sentence, hard cap)
    for sp in synopsis.get("system_prompts", []):
        text = (sp.get("text") or "").strip()
        if text:
            hint = _first_sentences(text, 1)[:_MAX_SYSPROMPT_CHARS]
            lines.append(f"Инструкция агента: {hint}{'…' if len(text) > _MAX_SYSPROMPT_CHARS else ''}")
            break  # one prompt is enough

    tool_names = sorted({te["source_name"] for te in synopsis.get("tool_edges", []) if te.get("source_name")})
    if tool_names:
        lines.append(f"Инструменты: {', '.join(tool_names)}.")

    return " ".join(lines) if lines else "Агентная диалоговая система."

