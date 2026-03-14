"""
Анализ флоу Langflow: узлы, связи, промпты, активы (tools, memory, RAG).
"""

import logging
from dataclasses import dataclass, field

log = logging.getLogger("mlsecops_agent.flow_analyzer")
from typing import Any, Dict, List


@dataclass
class FlowAsset:
    """Актив флоу (tool, memory, RAG, API)."""
    type: str
    name: str
    component_id: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowNode:
    """Узел флоу."""
    id: str
    data_type: str
    display_name: str
    description: str = ""
    template: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowAnalysis:
    """Результат анализа флоу."""
    flow_id: str
    name: str
    description: str
    nodes: List = field(default_factory=list)
    edges: List = field(default_factory=list)
    assets: List = field(default_factory=list)
    agents: List = field(default_factory=list)
    tools: List = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    entrypoints: List[str] = field(default_factory=list)
    graph_summary: str = ""


# Типы компонентов по MAESTRO / Agentic Radar
TOOL_TYPES = {
    "FAISS", "SearchComponent", "URL", "CalculatorComponent", "MCPTools",
    "File", "SQLDatabase", "WebSearch", "APIRequest", "CodeExecutor",
}
AGENT_TYPES = {"Agent"}
INPUT_TYPES = {"ChatInput"}
MEMORY_TYPES = {"Memory", "InMemoryStore", "Redis"}
RAG_INDICATORS = {"FAISS", "VectorStore", "SplitText", "Embeddings"}


def _extract_value(obj: Any, *keys: str, default: str = "") -> str:
    """
    Извлекает значение из вложенной структуры Langflow.

    Поддерживает template-поля с вложенным value (например, {"value": "..."}).
    """
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            val = obj[key]
            if isinstance(val, dict) and "value" in val:
                return str(val.get("value", default))
            return str(val) if val is not None else default
    return default


def _get_node_data_type(node: dict) -> str:
    """Определяет тип узла по data.node.display_name или id компонента."""
    data = node.get("data", {})
    node_obj = data.get("node", {})
    return node_obj.get("display_name", data.get("id", "").split("-")[0])


def analyze_flow(flow_response, flow_id=""):
    """
    Анализирует ответ Langflow API и извлекает граф, активы, промпты.

    Парсит nodes и edges, классифицирует компоненты (Agent, Tool, RAG, Memory),
    строит граф взаимодействия (ChatInput -> Agent -> Tool -> ...) и извлекает
    системные промпты из template-полей.

    Args:
        flow_response: JSON от GET /api/v1/flows/{id} или fetch_flow_from_file
        flow_id: ID флоу (если не передан в response)

    Returns:
        FlowAnalysis с полями:
        - nodes, edges: сырые данные графа
        - agents, tools, assets: классифицированные компоненты
        - prompts: извлечённые системные промпты
        - entrypoints: точки входа (ChatInput)
        - graph_summary: текстовое представление графа
    """
    if not flow_response or not isinstance(flow_response, dict):
        raise ValueError("flow_response must be a non-empty dict")
    flow_id = str(flow_id or flow_response.get("id", "") or "")
    data = flow_response.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", [])
    if not isinstance(nodes_raw, list):
        nodes_raw = []
    if not isinstance(edges_raw, list):
        edges_raw = []

    nodes = []
    assets = []
    agents = []
    tools = []
    prompts = []
    entrypoints = []

    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        d = n.get("data") or {}
        node_obj = d.get("node", {})
        comp_id = d.get("id", n.get("id", ""))
        display_name = node_obj.get("display_name", comp_id.split("-")[0])
        data_type = display_name or comp_id.split("-")[0]

        template = d.get("template", node_obj.get("template", {}))
        if isinstance(template, dict):
            template = template
        else:
            template = {}

        fn = FlowNode(
            id=comp_id,
            data_type=data_type,
            display_name=display_name,
            description=node_obj.get("description", ""),
            template=template,
        )
        nodes.append(fn)

        # Промпты (Prompt Template, system_prompt в Agent)
        prompt_val = _extract_value(template, "template", "system_prompt", "prompt", "value")
        if prompt_val and len(prompt_val) > 20:
            prompts.append(prompt_val[:2000])

        # Entrypoints
        if data_type in INPUT_TYPES or "ChatInput" in comp_id:
            entrypoints.append(comp_id)

        # Agents
        if data_type in AGENT_TYPES or "Agent" in comp_id:
            agents.append(fn)

        # Tools
        if data_type in TOOL_TYPES or "Tool" in str(node_obj.get("output_types", [])):
            fa = FlowAsset(
                type="tool",
                name=display_name,
                component_id=comp_id,
                description=node_obj.get("description", ""),
                metadata={"data_type": data_type},
            )
            tools.append(fa)
            assets.append(fa)

        # Memory / RAG
        if data_type in MEMORY_TYPES or data_type in RAG_INDICATORS:
            assets.append(FlowAsset(
                type="memory" if data_type in MEMORY_TYPES else "rag",
                name=display_name,
                component_id=comp_id,
                description=node_obj.get("description", ""),
            ))

    # Граф: User -> Agent1 -> Tool -> Agent2 -> ...
    graph_parts = []
    source_to_target = {}
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        src = str(e.get("source", "") or "")
        tgt = str(e.get("target", "") or "")
        if src and tgt:
            source_to_target[src] = tgt

    id_to_name = {n.id: n.display_name or n.data_type for n in nodes}
    visited = set()
    for ep in entrypoints:
        cur = ep
        path = [id_to_name.get(cur, cur)]
        while cur and cur not in visited:
            visited.add(cur)
            cur = source_to_target.get(cur)
            if cur:
                path.append(id_to_name.get(cur, cur))
        if len(path) > 1:
            graph_parts.append(" -> ".join(path))

    graph_summary = "\n".join(graph_parts) if graph_parts else "; ".join(
        f"{n.display_name}({n.data_type})" for n in nodes[:10]
    )

    log.debug("Analyzed flow: %d nodes, %d edges, %d agents, %d tools", len(nodes), len(edges_raw), len(agents), len(tools))
    return FlowAnalysis(
        flow_id=flow_id,
        name=str(flow_response.get("name") or "Unnamed"),
        description=str(flow_response.get("description") or ""),
        nodes=nodes,
        edges=edges_raw,
        assets=assets,
        agents=agents,
        tools=tools,
        prompts=prompts,
        entrypoints=entrypoints,
        graph_summary=graph_summary,
    )
