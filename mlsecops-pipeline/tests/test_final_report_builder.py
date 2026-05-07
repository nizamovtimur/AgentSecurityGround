from __future__ import annotations

from services.final_report_builder import build_final_report, score_to_severity, score_to_severity_ru


def test_score_to_severity() -> None:
    assert score_to_severity(4.9) == "LOW"
    assert score_to_severity(5.0) == "MEDIUM"
    assert score_to_severity(6.5) == "HIGH"
    assert score_to_severity(8.5) == "CRITICAL"
    assert score_to_severity_ru(6.5) == "ВЫСОКИЙ"


def test_build_final_report() -> None:
    synopsis = {
        "entrypoints": ["n1"],
        "assets": ["Agent::agent"],
        "controls": ["c1"],
    }
    boart_report = {
        "summary": {"goals_total": 2, "goals_successful": 1, "asr": 0.5},
        "results": [
            {
                "attack_name": "system_prompt_leakage",
                "goal": "leak prompt",
                "success": True,
                "max_score": 7.0,
                "steps": [{"target_response": "sample response"}],
            },
            {
                "attack_name": "harmbench_text",
                "goal": "unsafe output",
                "success": False,
                "max_score": 4.0,
                "steps": [{"target_response": "refusal"}],
            },
        ],
    }
    report = build_final_report(
        flow_path="flow.json",
        synopsis=synopsis,
        threat_model_markdown="# Threat model",
        boart_report=boart_report,
        attack_plan={"attacks": ["system_prompt_leakage"], "rationale": ["test"]},
        flow_export_payload={
            "name": "Windchaser",
            "id": "1b40c9e0-35dc-4823-85b8-6e692d1473de",
            "endpoint_name": "windchaser",
            "description": "",
            "tags": [],
            "last_tested_version": "1.8.3",
            "is_component": False,
            "locked": False,
        },
    )
    assert report["system"]["flow_export_metadata"]["name"] == "Windchaser"
    assert report["system"]["flow_export_metadata"]["endpoint_name"] == "windchaser"
    assert "overall_severity" not in report["risk"]
    per_threat = {item["threat"]: item for item in report["risk"]["per_threat_severity"]}
    assert per_threat["system_prompt_leakage"]["severity"] == "HIGH"
    assert per_threat["system_prompt_leakage"]["severity_ru"] == "ВЫСОКИЙ"
    assert "Утечка" in per_threat["system_prompt_leakage"]["threat_label_ru"]
    assert per_threat["harmbench_text"]["severity"] == "LOW"
    assert report["risk"]["severity_scale_ru"][2] == "ВЫСОКИЙ"
    assert report["adversarial_testing"]["attack_plan"]["attacks"] == ["system_prompt_leakage"]
    assert report["adversarial_testing"]["summary"]["asr"] == 0.5
    assert len(report["adversarial_testing"]["goals"]) == 2


def test_build_final_report_export_meta_defaults_empty() -> None:
    report = build_final_report(
        flow_path="x.json",
        synopsis={"entrypoints": [], "assets": [], "controls": []},
        threat_model_markdown="# T",
        boart_report={"summary": {"goals_total": 0}, "results": []},
    )
    assert report["system"]["flow_export_metadata"] == {}


def test_extract_flow_export_metadata_minimal_export() -> None:
    from services.final_report_builder import extract_flow_export_metadata

    raw = {"data": {"nodes": []}, "name": "X", "id": "abc", "endpoint_name": "x", "tags": ["p"]}
    meta = extract_flow_export_metadata(raw)
    assert meta["name"] == "X" and meta["id"] == "abc" and meta["tags"] == ["p"]
    assert "data" not in meta


def test_build_security_assessment_markdown_sections() -> None:
    from services.final_report_builder import build_security_assessment_markdown

    md = build_security_assessment_markdown(
        threat_model_markdown="## Отчёт\nТекст угрозы.",
        compliance_report={
            "overall": "PASS",
            "violations": 0,
            "warnings": 0,
            "semantic_analysis": False,
            "decision_statement": "Всё ок.",
        },
        flow_export_metadata={"name": "F", "id": "i1"},
        flow_source_label="/path/to/flow.json",
        generated_at_iso="2099-01-01 00:00:00 UTC",
    )
    assert "2099-01-01" in md
    assert "`name` | F |" in md or "| `name` | F |" in md
    assert "Часть 1." in md
    assert "## Отчёт" in md
    assert "Часть 2." in md
    assert "Часть 3." in md
    assert "Всё ок." in md
    assert "security gate" in md.lower()


def test_security_assessment_markdown_when_compliance_skipped() -> None:
    from services.final_report_builder import build_security_assessment_markdown

    md = build_security_assessment_markdown(
        threat_model_markdown="# Угрозы",
        compliance_report=None,
        flow_export_metadata=None,
        flow_source_label="/f.json",
        generated_at_iso="t",
    )
    assert "Часть 2." in md
    assert "Часть 3." in md
    assert "--no-compliance" in md or "политик" in md.lower()


def test_console_formatters_compact_output() -> None:
    from services.final_report_builder import (
        format_compliance_console_brief,
        format_maestro_console_brief,
        format_security_gate_plaintext,
    )

    tm = "## Отчёт\n\nНе использовать ключ API напрямую в промпте."
    brief_tm = format_maestro_console_brief(tm)
    assert "##" not in brief_tm
    assert len(brief_tm) < len(tm)

    cr = {
        "overall": "WARN",
        "violations": 0,
        "warnings": 2,
        "semantic_analysis": True,
        "results": [
            {
                "requirement_id": "REQ-X",
                "status": "PASS",
                "requirement_short": "Кратко",
            },
        ],
    }
    brief_cr = format_compliance_console_brief(cr)
    assert "REQ-X: PASS" in brief_cr
    assert "decision_statement" in brief_cr

    gate = format_security_gate_plaintext(cr)
    assert "Gate:" in gate
    assert "Три слоя" not in gate


def test_security_package_console_digest_shape() -> None:
    from services.final_report_builder import format_scan_summary

    text = format_scan_summary(
        flow_source_label="/x/flow.json",
        synopsis={
            "summary": {"node_count": 2, "edge_count": 1, "entrypoint_count": 1, "control_count": 0},
            "entrypoints": ["ChatInput-X"],
            "assets": [],
            "controls": [],
        },
        threat_model_markdown="## Отчёт\nПара строк.",
        compliance_report={
            "overall": "PASS",
            "violations": 0,
            "warnings": 0,
            "semantic_analysis": False,
            "results": [{"requirement_id": "REQ-A", "status": "PASS", "requirement_short": "Штука"}],
        },
        raw_flow_export={"name": "N", "id": "i1"},
        compliance_was_skipped=False,
    )
    assert "Описание флоу" in text
    assert "MAESTRO" in text
    assert "REQ-A: PASS" in text
    assert "Итоговое решение" in text or "Итог для прода" in text
    assert "Выходные данные" in text or "`name`" in text


def test_security_package_console_digest_no_compliance() -> None:
    from services.final_report_builder import format_scan_summary

    t = format_scan_summary(
        flow_source_label="file.json",
        synopsis={"summary": {}, "entrypoints": []},
        threat_model_markdown="# T\nok",
        compliance_report=None,
        raw_flow_export={},
        compliance_was_skipped=True,
    )
    assert "--no-compliance" in t

