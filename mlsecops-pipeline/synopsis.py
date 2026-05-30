"""Компактный security-synopsis для LLM-агентов."""

from __future__ import annotations

from typing import Any

from flow_parser import FlowGraph

_META_KEYS = frozenset({
    "system_prompt", "system_message", "instructions", "template", "prompt",
    "tool_mode", "model", "model_name", "temperature", "max_tokens",
})


def _meta_text(meta: dict[str, Any], key: str) -> str | None:
    v = meta.get(key)
    if isinstance(v, dict):
        raw = v.get("value")
        return raw.strip() if isinstance(raw, str) and raw.strip() else None
    return v.strip() if isinstance(v, str) and v.strip() else None


def build_synopsis(graph: FlowGraph) -> dict[str, Any]:
    g = graph.to_dict()
    nodes = []
    for n in g["nodes"]:
        ap = {k: v for k, v in n["template_fields"].items() if k in _META_KEYS}
        nodes.append({
            "id": n["node_id"],
            "name": n["display_name"],
            "type": n["node_type"],
            "role": n["role"],
            "risk_flags": n["risk_flags"],
            "author_parameters": ap,
        })
    edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "source_handle": e["source_handle_name"],
            "target_field": e["target_field_name"],
            "source_type": next((x["type"] for x in nodes if x["id"] == e["source"]), None),
            "target_type": next((x["type"] for x in nodes if x["id"] == e["target"]), None),
        }
        for e in g["edges"]
    ]
    by_id = {n["id"]: n for n in nodes}
    system_prompts: list[dict[str, Any]] = []
    tool_edges: list[dict[str, Any]] = []

    for n in nodes:
        for field in ("system_prompt", "system_message", "instructions", "prompt"):
            text = _meta_text(n["author_parameters"], field)
            if text:
                system_prompts.append({
                    "node_id": n["id"], "node_name": n["name"],
                    "node_type": n["type"], "field": field, "text": text,
                })

    for e in edges:
        sh = str(e.get("source_handle") or "").lower()
        tf = str(e.get("target_field") or "").lower()
        sn, tn = by_id.get(e["source"]), by_id.get(e["target"])
        if sh == "component_as_tool" and tf == "tools":
            tool_edges.append({
                "source_id": e["source"],
                "source_name": sn["name"] if sn else e["source"],
                "source_type": sn["type"] if sn else None,
                "target_id": e["target"],
                "target_name": tn["name"] if tn else e["target"],
            })
        if tf in {"system_prompt", "system_message"} and sn:
            dyn = _meta_text(sn["author_parameters"], "template") or _meta_text(sn["author_parameters"], "prompt")
            if dyn:
                system_prompts.append({
                    "node_id": sn["id"], "node_name": sn["name"],
                    "node_type": sn["type"], "field": f"dynamic->{tf}", "text": dyn,
                })

    return {
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "entrypoints": len(g["entrypoints"]),
            "controls": len(g["controls"]),
        },
        "entrypoints": g["entrypoints"],
        "controls": g["controls"],
        "nodes": nodes,
        "edges": edges,
        "system_prompts": system_prompts,
        "tool_edges": tool_edges,
    }


def build_llm_context(synopsis: dict[str, Any]) -> dict[str, Any]:
    """Контекст для LLM: полные system_prompts, без code/лишних template-полей."""
    return {
        "summary": synopsis.get("summary"),
        "entrypoints": synopsis.get("entrypoints"),
        "controls": synopsis.get("controls"),
        "system_prompts": synopsis.get("system_prompts"),
        "tool_edges": synopsis.get("tool_edges"),
        "nodes": [
            {k: n[k] for k in ("id", "name", "type", "role", "risk_flags")}
            for n in synopsis.get("nodes", [])
        ],
        "edges": synopsis.get("edges"),
    }


def build_compliance_context(synopsis: dict[str, Any], static: dict[str, Any]) -> dict[str, Any]:
    """Контекст для ComplianceAgent: полные промпты + топология для архитектурного аудита."""
    return {
        "summary": synopsis.get("summary"),
        "entrypoints": synopsis.get("entrypoints"),
        "controls": synopsis.get("controls"),
        "system_prompts": synopsis.get("system_prompts"),
        "tool_edges": synopsis.get("tool_edges"),
        "nodes": [
            {k: n[k] for k in ("id", "name", "type", "role", "risk_flags")}
            for n in synopsis.get("nodes", [])
        ],
        "edges": synopsis.get("edges"),
        "static_findings": static,
    }
