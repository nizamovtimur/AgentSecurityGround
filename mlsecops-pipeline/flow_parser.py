"""Детерминированная нормализация JSON Langflow (без LLM, без маскирования и обрезки)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ENTRYPOINT_TYPES = {"ChatInput", "URL", "File", "APIRequest", "MCPTools"}
CONTROL_TYPES = {"GuardrailValidator", "Regex", "FilterData", "ParserComponent"}
SENSITIVE_MARKERS = ("api_key", "token", "secret", "password")
PROMPT_MARKERS = ("system_prompt", "system_message", "instructions", "prompt", "template")


def _s(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _role(node_type: str) -> str:
    low = node_type.lower()
    if "agent" in low:
        return "agent"
    if node_type in ENTRYPOINT_TYPES or "input" in low or "output" in low:
        return "io"
    if "model" in low or "embedding" in low:
        return "model"
    if "guardrail" in low or node_type in CONTROL_TYPES:
        return "guardrail"
    if "tool" in low or "search" in low or "mcp" in low:
        return "tool"
    if "data" in low or "chroma" in low or "vector" in low:
        return "data"
    return "component"


@dataclass(slots=True)
class FlowNode:
    node_id: str
    display_name: str
    node_type: str
    role: str
    template_fields: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FlowEdge:
    source: str
    target: str
    source_data_type: str | None = None
    source_handle_name: str | None = None
    target_field_name: str | None = None
    target_input_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FlowGraph:
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    entrypoints: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "entrypoints": self.entrypoints,
            "controls": self.controls,
        }


def _norm_template(template: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Все значения шаблона как в Langflow — без REDACTED и без trunc."""
    out: dict[str, Any] = {}
    risks: list[str] = []
    for key, val in template.items():
        low = key.lower()
        if isinstance(val, dict):
            sens = any(m in low for m in SENSITIVE_MARKERS) or val.get("password")
            out[key] = {
                "type": _s(val.get("_input_type")),
                "required": bool(val.get("required")),
                "has_value": bool(val.get("value")),
                "value": val.get("value"),
            }
            if sens:
                risks.append(f"sensitive_field:{key}")
        elif isinstance(val, (str, int, float, bool)):
            out[key] = val
    return out, sorted(set(risks))


def parse_flow(flow_data: dict[str, Any]) -> FlowGraph:
    data = flow_data.get("data", flow_data)
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    nodes: list[FlowNode] = []
    types: dict[str, str] = {}
    entrypoints: set[str] = set()
    controls: set[str] = set()
    tool_src: set[str] = set()
    sys_targets: set[str] = set()

    for rn in raw_nodes:
        nd = rn.get("data", {})
        meta = nd.get("node", {}) if isinstance(nd.get("node"), dict) else {}
        nid = _s(nd.get("id") or rn.get("id"))
        name = _s(meta.get("display_name") or meta.get("name") or nid)
        ntype = _s(nd.get("type") or meta.get("name") or "Unknown")
        tmpl = meta.get("template", {}) if isinstance(meta.get("template"), dict) else {}
        role = _role(ntype)
        fields, tr = _norm_template(tmpl)
        flags = list(tr)
        if any(m in k.lower() for k in tmpl for m in PROMPT_MARKERS):
            flags.append("system_prompt_surface")
        if role == "agent":
            flags.append("agent_node")
        if ntype in ENTRYPOINT_TYPES:
            entrypoints.add(nid)
        if ntype in CONTROL_TYPES or "guardrail" in ntype.lower():
            controls.add(nid)
        nodes.append(FlowNode(nid, name, ntype, role, fields, sorted(set(flags))))
        types[nid] = ntype

    edges: list[FlowEdge] = []
    for re_ in raw_edges:
        ed = re_.get("data", {})
        src = _s(re_.get("source"))
        dst = _s(re_.get("target"))
        sh = ed.get("sourceHandle", {}) if isinstance(ed, dict) else {}
        th = ed.get("targetHandle", {}) if isinstance(ed, dict) else {}
        shn = _s(sh.get("name"))
        tfn = _s(th.get("fieldName"))
        tit = th.get("inputTypes", []) if isinstance(th.get("inputTypes"), list) else []
        if shn.lower() == "component_as_tool" and tfn.lower() == "tools":
            tool_src.add(src)
        if tfn.lower() in {"system_prompt", "system_message"}:
            sys_targets.add(dst)
        edges.append(
            FlowEdge(src, dst, _s(sh.get("dataType")) or types.get(src), shn or None, tfn or None, tit)
        )

    for node in nodes:
        if node.node_id in tool_src:
            node.role = "tool"
            node.risk_flags = sorted(set(node.risk_flags + ["tool_attached_to_agent"]))
        if node.node_id in sys_targets:
            node.risk_flags = sorted(set(node.risk_flags + ["system_prompt_surface"]))

    return FlowGraph(nodes, edges, sorted(entrypoints), sorted(controls))
