from __future__ import annotations

import json
from typing import cast

from llm.openai_client import OpenAIClient
from services.attack_planner import (
    AttackPlan,
    plan_attacks,
    select_attacks_from_context,
)

from conftest import PIPELINE_ROOT


# ---------------------------------------------------------------------------
# Heuristic baseline (no LLM)
# ---------------------------------------------------------------------------

def test_heuristic_uses_threat_markers() -> None:
    plan = select_attacks_from_context(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="## Threats\n- System prompt leakage risk\n- Jailbreak content",
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert "system_prompt_leakage" in plan.attacks
    assert "harmbench_text" in plan.attacks
    assert plan.planner == "heuristic"


def test_heuristic_russian_markers() -> None:
    plan = select_attacks_from_context(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="Риск утечки системного промпта. Джейлбрейк и вредоносный вывод.",
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert "system_prompt_leakage" in plan.attacks
    assert "harmbench_text" in plan.attacks


def test_heuristic_falls_back_to_static_signals() -> None:
    plan = select_attacks_from_context(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="No explicit threat tags.",
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert len(plan.attacks) >= 1
    assert "system_prompt_leakage" in plan.attacks


# ---------------------------------------------------------------------------
# LLM agent + fallback
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        return self._response


def test_agent_includes_compliance_statement_in_prompt() -> None:
    captured: dict[str, str] = {}

    class _RecordingLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
            captured["user"] = user_prompt
            return json.dumps(
                {"attacks": ["system_prompt_leakage"], "rationale": ["По секретам в промпте."]},
                ensure_ascii=False,
            )

    decision = (
        "REQ-DATA-MIN: обнаружены секреты в system prompt. Приоритет: тесты на утечку инструкций."
    )
    plan_attacks(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="## Угрозы\nУтечка.",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, _RecordingLLM()),
        mode="agent",
        compliance_decision_statement=decision,
    )
    _header, payload = captured["user"].split("\n\n", 1)
    assert "входной JSON" in _header
    parsed = json.loads(payload)
    assert parsed["compliance_decision_statement"] == decision


def test_agent_uses_llm_subset() -> None:
    fake = _FakeLLM(json.dumps(
        {"attacks": ["system_prompt_leakage"], "rationale": ["Только утечка по МУ."]},
        ensure_ascii=False,
    ))
    plan = plan_attacks(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="## Угрозы\nУтечка системного промпта.",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, fake),
        mode="agent",
    )
    assert isinstance(plan, AttackPlan)
    assert plan.planner == "agent"
    assert plan.attacks == ["system_prompt_leakage"]
    assert plan.rationale[0] == "Только утечка по МУ."


def test_agent_prompt_has_null_compliance_when_omitted() -> None:
    captured: dict[str, str] = {}

    class _RecordingLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
            captured["user"] = user_prompt
            return json.dumps({"attacks": ["harmbench_text"], "rationale": ["x"]}, ensure_ascii=False)

    plan_attacks(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="jailbreak",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, _RecordingLLM()),
        mode="agent",
    )
    parsed = json.loads(captured["user"].split("\n\n", 1)[1])
    assert parsed["compliance_decision_statement"] is None
def test_agent_falls_back_on_bad_json() -> None:
    plan = plan_attacks(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="## Угрозы\nРиск утечки системного промпта.",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, _FakeLLM("not json")),
        mode="agent",
    )
    assert plan.planner == "heuristic"
    assert "system_prompt_leakage" in plan.attacks


def test_mode_heuristic_skips_llm_entirely() -> None:
    plan = plan_attacks(
        synopsis={"entrypoints": ["n1"], "assets": ["Main Agent::agent"]},
        threat_model_markdown="jailbreak unsafe content",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, _FakeLLM("{}")),
        mode="heuristic",
    )
    assert plan.planner == "heuristic"
    assert "harmbench_text" in plan.attacks
