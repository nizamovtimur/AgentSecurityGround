from __future__ import annotations

import json
from typing import cast

from llm.openai_client import OpenAIClient
from services.attack_planner import plan_attacks, select_attacks_from_context

from conftest import PIPELINE_ROOT


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        return self._response


def test_plan_attacks_agent_uses_llm_subset() -> None:
    fake = _FakeLLM(
        json.dumps(
            {"attacks": ["system_prompt_leakage"], "rationale": ["Только утечка по МУ."]},
            ensure_ascii=False,
        )
    )
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    plan = plan_attacks(
        synopsis=synopsis,
        threat_model_markdown="## Угрозы\nУтечка системного промпта.",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, fake),
        mode="agent",
        prompts_dir=PIPELINE_ROOT / "prompts",
    )
    assert plan.planner == "agent"
    assert plan.attacks == ["system_prompt_leakage"]
    assert plan.rationale[0] == "Только утечка по МУ."


def test_plan_attacks_agent_falls_back_on_bad_json() -> None:
    fake = _FakeLLM("not json")
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    plan = plan_attacks(
        synopsis=synopsis,
        threat_model_markdown="## Угрозы\nРиск утечки системного промпта.",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, fake),
        mode="agent",
        prompts_dir=PIPELINE_ROOT / "prompts",
    )
    assert plan.planner == "heuristic"
    heuristic = select_attacks_from_context(
        synopsis=synopsis,
        threat_model_markdown="## Угрозы\nРиск утечки системного промпта.",
        datasets_dir=PIPELINE_ROOT / "datasets",
    )
    assert plan.attacks == heuristic.attacks


def test_plan_attacks_heuristic_skips_llm() -> None:
    fake = _FakeLLM("{}")
    synopsis = {"entrypoints": ["n1"], "assets": ["Main Agent::agent"]}
    plan = plan_attacks(
        synopsis=synopsis,
        threat_model_markdown="jailbreak unsafe content",
        datasets_dir=PIPELINE_ROOT / "datasets",
        llm_client=cast(OpenAIClient, fake),
        mode="heuristic",
        prompts_dir=PIPELINE_ROOT / "prompts",
    )
    assert plan.planner == "heuristic"
    assert "harmbench_text" in plan.attacks
