from __future__ import annotations

from pathlib import Path

from parsers.langflow_parser import parse_langflow_file

from conftest import REPO_ROOT

TRAVEL_FLOW = REPO_ROOT / "langflow" / "flows" / "TravelPlanningAgents.json"
WINDCHASER_FLOW = REPO_ROOT / "langflow" / "flows" / "Windchaser.json"


def test_parser_extracts_nodes_edges_and_entrypoints() -> None:
    graph = parse_langflow_file(TRAVEL_FLOW)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert len(graph.entrypoints) > 0


def test_parser_extracts_controls_from_guardrails() -> None:
    graph = parse_langflow_file(WINDCHASER_FLOW)
    assert any("GuardrailValidator" in node.node_type for node in graph.nodes)
    assert len(graph.controls) > 0


def test_parser_redacts_sensitive_template_fields() -> None:
    graph = parse_langflow_file(TRAVEL_FLOW)
    redacted = any(
        isinstance(field, dict) and field.get("value") == "***REDACTED***"
        for node in graph.nodes
        for field in node.template_fields.values()
    )
    assert redacted


def test_parser_marks_component_as_tool_edges_as_tools() -> None:
    graph = parse_langflow_file(TRAVEL_FLOW)
    node_by_id = {node.node_id: node for node in graph.nodes}
    for edge in graph.edges:
        if edge.source_handle_name == "component_as_tool" and edge.target_field_name == "tools":
            source = node_by_id[edge.source]
            assert source.role == "tool"
            assert "tool_attached_to_agent" in source.risk_flags


def test_parser_marks_dynamic_system_prompt_surface() -> None:
    graph = parse_langflow_file(WINDCHASER_FLOW)
    node_by_id = {node.node_id: node for node in graph.nodes}
    for edge in graph.edges:
        if edge.target_field_name in {"system_prompt", "system_message"}:
            target = node_by_id[edge.target]
            assert "system_prompt_surface" in target.risk_flags
