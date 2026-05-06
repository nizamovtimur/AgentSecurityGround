from __future__ import annotations

from services.attack_planner import select_attacks_from_context

from conftest import PIPELINE_ROOT


def test_attack_planner_uses_threat_markers() -> None:
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    report = """
    ## Threats
    - System prompt leakage risk
    - Jailbreak and unsafe content generation
    """
    plan = select_attacks_from_context(
        synopsis=synopsis,
        threat_model_markdown=report,
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert "system_prompt_leakage" in plan.attacks
    assert "harmbench_text" in plan.attacks
    assert plan.planner == "heuristic"


def test_attack_planner_russian_threat_markers() -> None:
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    report = "## Угрозы\nРиск утечки системного промпта. Джейлбрейк и вредоносный вывод."
    plan = select_attacks_from_context(
        synopsis=synopsis,
        threat_model_markdown=report,
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert "system_prompt_leakage" in plan.attacks
    assert "harmbench_text" in plan.attacks


def test_attack_planner_falls_back_to_static_signals() -> None:
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    plan = select_attacks_from_context(
        synopsis=synopsis,
        threat_model_markdown="No explicit threat tags.",
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert len(plan.attacks) >= 1
    assert "system_prompt_leakage" in plan.attacks
