"""Markdown-отчёт и таблицы для ноутбука / артефактов."""

from __future__ import annotations

from typing import Any

from boart.verdict import format_goal_line


def _clip(text: str, n: int = 120) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def progress_table_markdown(results: list[dict[str, Any] | Any]) -> str:
    """Таблица промежуточных вердиктов BROKEN / RESILIENT / ERROR."""
    lines = [
        "## Прогресс BOART",
        "",
        "| Вердикт | Датасет | Score | Цель | Ошибка |",
        "|---------|---------|-------|------|--------|",
    ]
    for item in results:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        v = item.get("verdict", "?")
        err = _clip(str(item.get("error") or ""), 60) or "—"
        lines.append(
            f"| **{v}** | `{item.get('attack_name', '?')}` | "
            f"{item.get('max_score', 0):.1f} | {_clip(str(item.get('goal', '')))} | {err} |"
        )
    return "\n".join(lines) + "\n"


def plan_table_markdown(plan: dict[str, Any], catalog: dict[str, str]) -> str:
    lines = [
        "## План атак (выбор датасетов)",
        "",
        "| Датасет | Описание | Обоснование |",
        "|---------|----------|-------------|",
    ]
    attacks = plan.get("attacks") or []
    rationale = plan.get("rationale") or []
    for i, name in enumerate(attacks):
        why = rationale[i] if i < len(rationale) else "—"
        lines.append(f"| `{name}` | {catalog.get(name, '—')} | {why} |")
    lines.append("")
    lines.append(f"*Планировщик:* **{plan.get('planner', '?')}**")
    return "\n".join(lines)


def summary_table_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary") or {}
    asr = float(s.get("asr") or 0.0)
    return "\n".join([
        "## Сводка BOART",
        "",
        "| Метрика | Значение |",
        "|---------|----------|",
        f"| Целей всего | {s.get('goals_total', 0)} |",
        f"| **BROKEN** (уязвимость) | {s.get('broken', s.get('goals_successful', 0))} |",
        f"| **RESILIENT** (устоял) | {s.get('resilient', 0)} |",
        f"| **ERROR** (сбой цели/LLM) | {s.get('errors', 0)} |",
        f"| **ASR** | **{asr:.0%}** |",
        "",
    ])


def results_table_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## Результаты по целям",
        "",
        "| Вердикт | Датасет | Max score | Цель |",
        "|---------|---------|-----------|------|",
    ]
    for item in report.get("results") or []:
        v = item.get("verdict", "RESILIENT" if not item.get("success") else "BROKEN")
        lines.append(
            f"| **{v}** | `{item.get('attack_name', '?')}` | "
            f"{item.get('max_score', 0):.1f} | {_clip(str(item.get('goal', '')))} |"
        )
    return "\n".join(lines) + "\n"


def steps_detail_markdown(report: dict[str, Any], *, max_goals: int = 4) -> str:
    parts = ["## Детали прогонов", ""]
    for item in (report.get("results") or [])[:max_goals]:
        v = item.get("verdict", "?")
        parts.append(f"### {v} · {item.get('attack_name')} — {_clip(str(item.get('goal')), 80)}")
        if item.get("error"):
            parts.append(f"\n**Ошибка:** `{_clip(str(item['error']), 200)}`\n")
        parts.append("")
        parts.append("| Шаг | Стратегия | Score | Вердикт |")
        parts.append("|-----|-----------|-------|---------|")
        for step in item.get("steps") or []:
            sl = "BROKEN" if step.get("success") else "RESILIENT"
            if step.get("error"):
                sl = "ERROR"
            parts.append(
                f"| {step.get('step')} | {_clip(str(step.get('selected_strategy')), 40)} | "
                f"{step.get('judge_score', 0):.1f} | {sl} |"
            )
        last = (item.get("steps") or [])[-1] if item.get("steps") else None
        if last and not item.get("error"):
            parts.extend([
                "",
                "**Последняя атака:**",
                "",
                f"> {_clip(str(last.get('attack_prompt')), 300)}",
                "",
                "**Ответ цели:**",
                "",
                f"> {_clip(str(last.get('target_response')), 300)}",
                "",
            ])
    return "\n".join(parts)


def build_boart_markdown(
    plan: dict[str, Any],
    report: dict[str, Any],
    catalog: dict[str, str],
    *,
    target_endpoint: str,
    gate_verdict: str | None = None,
) -> str:
    header = [
        "# BOART — отчёт о состязательном тестировании",
        "",
        f"- **Цель (endpoint):** `{target_endpoint}`",
    ]
    if gate_verdict:
        header.append(f"- **Security Gate:** **{gate_verdict}**")
    header.append("")
    body = [
        plan_table_markdown(plan, catalog),
        progress_table_markdown(report.get("results") or []),
        summary_table_markdown(report),
        results_table_markdown(report),
        steps_detail_markdown(report),
    ]
    return "\n".join(header + body)


def format_goal_line_from_dict(item: dict[str, Any]) -> str:
    return format_goal_line(item)
