from __future__ import annotations

import json
from pathlib import Path

from models.security_graph import SecurityEdge, SecurityGraph, SecurityNode
from services.synopsis_builder import build_security_synopsis
from services.threat_modeling_service import ThreatModelingService

from conftest import PIPELINE_ROOT


class StubOpenAIClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        assert "<JSON>" not in system_prompt
        assert "<THREAT_MODEL_CONTEXT>" not in system_prompt
        assert "JSON графа" in user_prompt or "workflow JSON" in user_prompt
        assert "Анализ угроз агентного workflow" in system_prompt or "Threat Analysis" in system_prompt
        return "# Анализ угроз агентного workflow\n\n## 1. Миссия\nТестовая миссия."


class LooseStubOpenAIClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        assert "<JSON>" not in system_prompt
        assert "<THREAT_MODEL_CONTEXT>" not in system_prompt
        assert "MAESTRO REF SNIPPET" in system_prompt
        assert "MAESTRO TEST" in system_prompt
        return "# Анализ угроз агентного workflow\n\n## 1. Миссия\nТестовая миссия."


def build_sample_graph() -> SecurityGraph:
    return SecurityGraph(
        nodes=[
            SecurityNode("n1", "Chat Input", "ChatInput", "io"),
            SecurityNode(
                "n2",
                "Main Agent",
                "Agent",
                "agent",
                template_fields={"system_prompt": {"value": "You are a safe assistant.", "has_value": True}},
            ),
            SecurityNode("n3", "MCP Tool", "MCPTools", "tool"),
            SecurityNode(
                "n4",
                "Prompt Template",
                "Prompt Template",
                "component",
                template_fields={"template": {"value": "System: strict policy.", "has_value": True}},
            ),
        ],
        edges=[
            SecurityEdge("n1", "n2", "ChatInput", "message", "input_value", ["Message"]),
            SecurityEdge("n3", "n2", "MCPTools", "component_as_tool", "tools", ["Tool"]),
            SecurityEdge("n4", "n2", "Prompt Template", "prompt", "system_prompt", ["Message"]),
        ],
        entrypoints=["n1", "n3", "n4"],
        controls=[],
        assets=["Main Agent::agent", "MCP Tool::tool", "Prompt Template::component"],
    )


def test_synopsis_tool_edges() -> None:
    synopsis = build_security_synopsis(build_sample_graph())
    assert synopsis["summary"]["node_count"] == 4
    assert len(synopsis["tool_edges"]) == 1
    assert synopsis["tool_edges"][0]["source_name"] == "MCP Tool"
    assert synopsis["tool_edges"][0]["target_name"] == "Main Agent"


def test_synopsis_factual_sections() -> None:
    synopsis = build_security_synopsis(build_sample_graph())
    assert "entrypoints" in synopsis
    assert "assets" in synopsis
    assert "system_prompts" in synopsis
    assert "prompt_edges" in synopsis
    assert "tool_edges" in synopsis
    assert "author_parameters_by_node" in synopsis
    assert any(item["node_name"] == "Main Agent" for item in synopsis["author_parameters_by_node"])
    assert all("meta_parameters" not in node for node in synopsis["nodes"])
    fields = {item["field"] for item in synopsis["system_prompts"]}
    names = {item["node_name"] for item in synopsis["system_prompts"]}
    # Main Agent has static system_prompt; Prompt Template feeds dynamic->system_prompt via edge
    assert any("system_prompt" in f for f in fields)
    assert "dynamic->system_prompt" in fields
    assert "Main Agent" in names
    assert "Prompt Template" in names


def test_threat_modeling_service_builds_prompt_and_returns_markdown(tmp_path: Path) -> None:
    threat_model = tmp_path / "threat_model.txt"
    system_prompt = tmp_path / "threat_model_system.txt"
    threat_model.write_text("MAESTRO TEST\nMAESTRO REF SNIPPET", encoding="utf-8")
    system_prompt.write_text(
        "Prefix\n<THREAT_MODEL_CONTEXT>\n<JSON>\n## 1. Миссия",
        encoding="utf-8",
    )
    service = ThreatModelingService(
        openai_client=LooseStubOpenAIClient(),
        threat_model_path=threat_model,
        system_prompt_path=system_prompt,
    )
    report = service.generate_report(build_sample_graph())
    assert report.startswith("# Анализ угроз агентного workflow")
    assert "MAESTRO REF SNIPPET" not in report
    json.dumps(build_security_synopsis(build_sample_graph()))


def test_service_uses_real_prompt_templates() -> None:
    service = ThreatModelingService(
        openai_client=StubOpenAIClient(),
        threat_model_path=PIPELINE_ROOT / "prompts" / "threat_model.txt",
        system_prompt_path=PIPELINE_ROOT / "prompts" / "threat_model_system_ru.txt",
    )
    report = service.generate_report(build_sample_graph())
    assert "Анализ угроз агентного workflow" in report
