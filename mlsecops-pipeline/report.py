"""Сборка итогового Markdown security gate."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from flow_meta import format_agent_outputs_markdown


class GateVerdict(BaseModel):
    status: str = Field(description="PASS или FAIL")
    comment: str = Field(description="Согласовано / Не согласовано…")


class SecurityGateReport(BaseModel):
    flow_id: str
    langflow_url: str
    verdict: GateVerdict
    markdown: str
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    compliance: dict[str, Any] = Field(default_factory=dict)
    artifacts_dir: str | None = Field(
        default=None,
        description="Каталог сохранённых артефактов (если задан output_dir)",
    )


def _verdict_section(verdict: GateVerdict) -> str:
    return "\n".join([
        "## Итоговое заключение",
        "",
        f"**Результат:** {verdict.status}",
        "",
        f"**Комментарий:** {verdict.comment}",
        "",
    ])


def compute_verdict(compliance: dict[str, Any]) -> GateVerdict:
    data_min = compliance.get("REQ-DATA-MIN", {}).get("status", "PASS")
    least = compliance.get("REQ-LEAST-PRIVILEGE", {}).get("status", "PASS")
    human = compliance.get("REQ-HUMAN-REVIEW", {}).get("status", "PASS")

    blocking_fail = data_min == "FAIL"
    if blocking_fail:
        reason = compliance.get("REQ-DATA-MIN", {}).get(
            "rationale", "нарушение минимизации данных / секреты в контексте LLM",
        )
        return GateVerdict(
            status="FAIL",
            comment=f"Не согласовано по причине: {reason}",
        )
    notes: list[str] = []
    if least in ("WARN", "FAIL"):
        notes.append(f"замечания по разделению привилегий ({least})")
    if human == "WARN":
        notes.append("есть сигналы для ручной проверки эксперта")
    if notes:
        return GateVerdict(
            status="PASS",
            comment="Согласовано с оговорками: " + "; ".join(notes) + ". Рекомендуется ревью MLSecOps.",
        )
    return GateVerdict(status="PASS", comment="Согласовано: блокирующих нарушений не выявлено.")


def build_markdown(
    *,
    flow_id: str,
    langflow_url: str,
    threat_md: str,
    compliance_md: str,
    human_review_md: str,
    agent_outputs: dict[str, Any],
    verdict: GateVerdict,
) -> str:
    parts = [
        "# Security Gate — заключение MLSecOps",
        "",
        f"- **Langflow:** `{langflow_url}`",
        f"- **Flow ID:** `{flow_id}`",
        "",
        "---",
        "",
        threat_md.strip(),
        "",
        "---",
        "",
        compliance_md.strip(),
        "",
        "---",
        "",
        human_review_md.strip(),
        "",
        "---",
        "",
        _verdict_section(verdict),
        format_agent_outputs_markdown(agent_outputs).rstrip(),
        "",
    ]
    return "\n".join(parts)
