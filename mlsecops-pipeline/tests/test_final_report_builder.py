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
    )
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
