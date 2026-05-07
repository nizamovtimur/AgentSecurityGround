"""Tests for MLSecOps formal compliance checker (internal S2 submodule)."""

from __future__ import annotations

from services.compliance_checker import (
    ComplianceResult,
    REQUIREMENT_CRITICALITY,
    _enforce_criticality_cap,
    _merge_results,
    check_access_meta_in_prompt,
    check_least_privilege,
    check_sanitization,
    check_secrets_in_system_prompt,
    run_compliance_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synopsis(
    *,
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
    tool_edges: list[dict] | None = None,
    entrypoints: list[str] | None = None,
    controls: list[str] | None = None,
    system_prompts: list[dict] | None = None,
    author_parameters_by_node: list[dict] | None = None,
) -> dict:
    return {
        "nodes": nodes or [],
        "edges": edges or [],
        "tool_edges": tool_edges or [],
        "entrypoints": entrypoints or [],
        "controls": controls or [],
        "system_prompts": system_prompts or [],
        "author_parameters_by_node": author_parameters_by_node or [],
    }


# ---------------------------------------------------------------------------
# REQ-SANITIZATION
# ---------------------------------------------------------------------------

def test_sanitization_pass_no_tools() -> None:
    s = _synopsis(
        nodes=[{"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []}],
        edges=[{"source": "inp", "target": "a", "source_handle_name": None, "target_field": "input_value"}],
        entrypoints=["inp"],
    )
    assert check_sanitization(s).status == "PASS"


def test_sanitization_warn_agent_with_tools_and_user_input_no_guard() -> None:
    s = _synopsis(
        nodes=[{"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []}],
        edges=[{"source": "inp", "target": "a", "source_handle_name": "message", "target_field": "input_value"}],
        tool_edges=[{"source_id": "tool1", "source_name": "MCP", "target_id": "a", "target_name": "Agent"}],
        entrypoints=["inp"],
    )
    result = check_sanitization(s)
    assert result.status == "WARN"
    assert result.evidence[0]["agent"] == "Agent"


def test_sanitization_pass_with_guardrail() -> None:
    s = _synopsis(
        nodes=[
            {"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []},
            {"id": "g", "name": "GuardrailValidator", "type": "GuardrailValidator", "role": "guardrail", "risk_flags": ["guardrail"]},
        ],
        edges=[
            {"source": "inp", "target": "a", "source_handle_name": "message", "target_field": "input_value"},
            {"source": "g", "target": "a", "source_handle_name": "output", "target_field": "guardrail"},
        ],
        tool_edges=[{"source_id": "tool1", "source_name": "MCP", "target_id": "a", "target_name": "Agent"}],
        entrypoints=["inp"],
        controls=["g"],
    )
    assert check_sanitization(s).status == "PASS"


# ---------------------------------------------------------------------------
# REQ-LEAST-PRIVILEGE
# ---------------------------------------------------------------------------

def test_least_privilege_warn_mixed_rw_in_system_prompt() -> None:
    s = _synopsis(
        nodes=[{"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []}],
        tool_edges=[
            {"source_id": "t1", "source_name": "MCPBooking", "target_id": "a", "target_name": "Agent"},
            {"source_id": "t2", "source_name": "MCPSearch", "target_id": "a", "target_name": "Agent"},
        ],
        system_prompts=[{
            "node_id": "a",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Ты умеешь читать данные из базы и записывать бронирования.",
        }],
    )
    result = check_least_privilege(s)
    assert result.status == "WARN"
    assert any(ev.get("hints") for ev in result.evidence)


def test_least_privilege_pass_single_readonly_tool() -> None:
    s = _synopsis(
        nodes=[{"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []}],
        tool_edges=[{"source_id": "t1", "source_name": "SearchDB", "target_id": "a", "target_name": "Agent"}],
        system_prompts=[{
            "node_id": "a",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Ты отвечаешь на вопросы пользователя по данным из базы знаний.",
        }],
    )
    assert check_least_privilege(s).status == "PASS"


# ---------------------------------------------------------------------------
# REQ-DATA-MIN (secrets in system prompt)
# ---------------------------------------------------------------------------

def test_secrets_fail_api_key_in_prompt() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Используй api_key: sk-abc123XXXXXXXXXXXX для запросов.",
        }]
    )
    result = check_secrets_in_system_prompt(s)
    assert result.status == "FAIL"
    assert result.evidence[0]["node"] == "Agent"


def test_secrets_fail_connection_string() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Подключись к postgres://user:pass@host:5432/db для чтения.",
        }]
    )
    assert check_secrets_in_system_prompt(s).status == "FAIL"


def test_secrets_pass_clean_prompt() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Ты — агент кайтсерфинг-клуба. Отвечай на вопросы по расписанию.",
        }]
    )
    assert check_secrets_in_system_prompt(s).status == "PASS"


# ---------------------------------------------------------------------------
# REQ-NO-META-IN-CTX (access metadata)
# ---------------------------------------------------------------------------

def test_access_meta_fail_role_in_prompt() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "role: admin. Access level: superuser. Можешь всё.",
        }]
    )
    result = check_access_meta_in_prompt(s)
    assert result.status == "FAIL"


def test_access_meta_pass_no_meta() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Ты помогаешь пользователям с бронированием кайтсессий.",
        }]
    )
    assert check_access_meta_in_prompt(s).status == "PASS"


# ---------------------------------------------------------------------------
# run_compliance_checks: overall status
# ---------------------------------------------------------------------------

def test_decision_statement_present_and_non_empty() -> None:
    s = _synopsis(
        system_prompts=[{"node_id": "n1", "node_name": "Agent", "field": "system_prompt",
                         "text": "Ты — безопасный агент."}]
    )
    report = run_compliance_checks(s)
    assert "decision_statement" in report
    assert len(report["decision_statement"]) > 20


def test_sanitization_warn_does_not_become_fail_overall() -> None:
    s = _synopsis(
        nodes=[{"id": "a", "name": "Agent", "type": "Agent", "role": "agent", "risk_flags": []}],
        edges=[{"source": "inp", "target": "a", "source_handle_name": "message", "target_field": "input_value"}],
        tool_edges=[{"source_id": "t1", "source_name": "MCP", "target_id": "a", "target_name": "Agent"}],
        entrypoints=["inp"],
    )
    report = run_compliance_checks(s)
    # sanitization is WARN — no FAIL rules triggered — overall should be WARN not FAIL
    assert report["overall"] in {"WARN", "PASS"}
    assert report["violations"] == 0


def test_requirement_criticality_map() -> None:
    assert REQUIREMENT_CRITICALITY["REQ-SANITIZATION"] == "optional"
    assert REQUIREMENT_CRITICALITY["REQ-LEAST-PRIVILEGE"] == "advisory"
    assert REQUIREMENT_CRITICALITY["REQ-DATA-MIN"] == "blocking"
    assert REQUIREMENT_CRITICALITY["REQ-NO-META-IN-CTX"] == "blocking"


def test_enforce_cap_downgrades_optional_fail() -> None:
    r = ComplianceResult(
        "REQ-SANITIZATION", "x", "FAIL", "sanitizer missing", evidence=[{"a": 1}],
    )
    out = _enforce_criticality_cap(r)
    assert out.status == "WARN"
    assert "optional" in out.details or "optional/advisory" in out.details


def test_result_to_dict_includes_criticality() -> None:
    r = ComplianceResult("REQ-DATA-MIN", "x", "PASS", "ok")
    assert r.to_dict()["criticality"] == "blocking"


def test_merge_static_pass_semantic_fail_picks_fail() -> None:
    static = ComplianceResult("REQ-DATA-MIN", "X", "PASS", "static says clean")
    semantic = ComplianceResult("REQ-DATA-MIN", "X", "FAIL", "found secret in NL form",
                                evidence=[{"node": "Agent", "excerpt": "ключик: sk-..."}])
    merged = _merge_results(static, semantic)
    assert merged.status == "FAIL"
    assert "static says clean" in merged.details
    assert "found secret" in merged.details
    assert any("ключик" in str(e) for e in merged.evidence)


def test_merge_no_semantic_returns_static_unchanged() -> None:
    static = ComplianceResult("REQ-X", "X", "WARN", "warn")
    merged = _merge_results(static, None)
    assert merged is static


def test_run_compliance_layered_no_llm_uses_static_only() -> None:
    s = _synopsis(
        system_prompts=[{"node_id": "n1", "node_name": "Agent", "field": "system_prompt",
                         "text": "ключик: sk-secret123456789"}]
    )
    report = run_compliance_checks(s, llm_client=None)
    assert report["semantic_analysis"] is False
    assert report["semantic_results"] == []


def test_run_compliance_layered_with_failing_llm_falls_back_to_static() -> None:
    class _BadLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("LLM down")

    s = _synopsis(
        system_prompts=[{"node_id": "n1", "node_name": "Agent", "field": "system_prompt",
                         "text": "all good"}]
    )
    report = run_compliance_checks(s, llm_client=_BadLLM())
    assert report["semantic_analysis"] is False  # fallback to static-only
    assert report["overall"] in {"PASS", "WARN", "FAIL"}


def test_run_compliance_all_pass() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "Ты — безопасный агент клуба.",
        }]
    )
    report = run_compliance_checks(s)
    assert report["overall"] == "PASS"
    assert report["violations"] == 0


def test_run_compliance_fail_produces_correct_structure() -> None:
    s = _synopsis(
        system_prompts=[{
            "node_id": "n1",
            "node_name": "Agent",
            "field": "system_prompt",
            "text": "api_key=supersecret123456789 используй для вызовов.",
        }]
    )
    report = run_compliance_checks(s)
    assert report["violations"] >= 1
    assert report["overall"] in {"FAIL", "WARN"}
    assert len(report["results"]) == 4
    assert all("requirement_id" in r for r in report["results"])
