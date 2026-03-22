"""
Генерация целей атаки (init dataset) на основе threat assessment.

Промпт загружается из prompts/goal_generator.txt.
"""

import json
from dataclasses import dataclass

from .config import APP_SEC_ATTACK_MODEL, get_openai_client
from .logging_config import get_logger
from .prompts_loader import load_prompt

log = get_logger("goal_generator")


@dataclass
class AttackGoal:
    """Цель атаки."""

    goal: str
    threat_category: str
    asset: str = ""


def generate_attack_goals(flow_analysis, threat_report, client=None, max_goals=5):
    """
    Генерирует 3–5 целей атаки на основе флоу и threat assessment.

    Промпт загружается из prompts/goal_generator.txt. Цели используются
    как init dataset для адверсарного мультиагента.

    Args:
        flow_analysis: Dict с ключами name, tools, agents, graph_summary
        threat_report: Markdown-отчёт threat modeling
        client: OpenAI client. Если None — создаётся через get_openai_client()
        max_goals: Максимальное количество целей (для Security Gates: 3–5)

    Returns:
        Список AttackGoal с полями goal, threat_category, asset

    Example:
        >>> goals = generate_attack_goals(analysis_dict, threat_report, max_goals=5)
        >>> [g.goal for g in goals]
    """
    if not flow_analysis or not isinstance(flow_analysis, dict):
        return []
    threat_report = str(threat_report) if threat_report else ""
    max_goals = max(1, min(int(max_goals) if max_goals else 5, 20))
    log.info("Generating %d attack goals", max_goals)

    client = client or get_openai_client()

    tools_list = flow_analysis.get("tools") or []
    tools_desc = ", ".join(
        str(a.get("name") or a.get("component_id", "")) for a in tools_list if isinstance(a, dict)
    ) or "none"
    agents_list = flow_analysis.get("agents") or []
    agents_desc = ", ".join(
        str(a.get("display_name") or a.get("id", "")) for a in agents_list if isinstance(a, dict)
    ) or "none"

    prompt = load_prompt(
        "goal_generator.txt",
        max_goals=str(max_goals),
        flow_name=str(flow_analysis.get("name") or "Unknown"),
        tools_desc=tools_desc,
        agents_desc=agents_desc,
        graph_summary=flow_analysis.get("graph_summary", "")[:500],
        threat_excerpt=threat_report[:3000],
    )

    if not prompt:
        return []

    resp = client.chat.completions.create(
        model=APP_SEC_ATTACK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = resp.choices[0].message.content or "[]"
    text = text.strip()
    if not text.startswith("["):
        start = text.find("[")
        if start >= 0:
            text = text[start:]
        else:
            return []

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return []

    goals = []
    for g in (raw or [])[:max_goals]:
        if isinstance(g, dict) and g.get("goal"):
            goals.append(
                AttackGoal(
                    goal=str(g["goal"]),
                    threat_category=str(g.get("threat_category", "unknown")),
                    asset=str(g.get("asset", "")),
                )
            )
    log.info("Generated %d goals", len(goals))
    return goals
