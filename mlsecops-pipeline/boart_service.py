"""Оркестрация BOART поверх Security Gate."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from boart.models import GoalRunResult

from attack_planner import AttackPlan, list_datasets, plan_attacks
from boart import BoartConfig, BoartRunner
from boart.llm_adapter import LLMClientAdapter
from boart.target_client import HttpTargetClient
from boart_report import build_boart_markdown
from config import ssl_verify
from langflow_run import resolve_target_endpoint, run_timeout_from_env
from llm import LLMClient
from logging_utils import get_logger

log = get_logger("boart_service")
_PACKAGE_ROOT = Path(__file__).resolve().parent


def build_target_description(
    synopsis: dict[str, Any],
    threat_md: str,
    *,
    compliance_comment: str | None = None,
) -> str:
    lines: list[str] = []
    for sp in synopsis.get("system_prompts", []):
        text = (sp.get("text") or "").strip()
        if text:
            lines.append(f"Инструкция агента: {text[:400]}{'…' if len(text) > 400 else ''}")
            break
    tool_names = sorted({te["source_name"] for te in synopsis.get("tool_edges", []) if te.get("source_name")})
    if tool_names:
        lines.append(f"Инструменты: {', '.join(tool_names)}.")
    if compliance_comment and compliance_comment.strip():
        lines.append(f"Заключение compliance: {compliance_comment.strip()[:600]}")
    if threat_md.strip():
        for para in threat_md.split("\n\n"):
            p = para.strip()
            if p and not p.startswith("|"):
                lines.append(p[:500])
                break
    return " ".join(lines) if lines else "Агентная диалоговая система Langflow."


def run_boart(
    *,
    synopsis: dict[str, Any],
    threat_md: str,
    llm_client: LLMClient,
    target_endpoint: str | None = None,
    langflow_url: str | None = None,
    flow_id: str | None = None,
    attacks: list[str] | None = None,
    planner_mode: str = "agent",
    goals_per_attack: int = 2,
    max_steps: int = 5,
    language: str = "ru",
    compliance_comment: str | None = None,
    target_description: str = "",
    show_progress: bool = True,
    langflow_ssl_verify: bool | None = None,
    on_goal_complete: Callable[[GoalRunResult], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """
    План атак + прогон BOART + Markdown-отчёт.

    :returns: (attack_plan, boart_report, markdown)
    """
    endpoint = resolve_target_endpoint(
        explicit=target_endpoint,
        langflow_url=langflow_url,
        flow_id=flow_id,
    )
    catalog = list_datasets()
    plan_obj: AttackPlan
    if attacks:
        plan_obj = AttackPlan(attacks=attacks, rationale=["Ручной список."], planner="manual")
    else:
        plan_obj = plan_attacks(
            synopsis,
            threat_md,
            llm_client=llm_client,
            mode=planner_mode,
            compliance_comment=compliance_comment,
        )
    plan = plan_obj.to_dict()
    log.info("План атак: %s (%s)", plan["attacks"], plan["planner"])

    td = target_description.strip() or build_target_description(
        synopsis, threat_md, compliance_comment=compliance_comment
    )
    verify = ssl_verify("langflow", override=langflow_ssl_verify)
    report = BoartRunner(
        config=BoartConfig(
            attacks=list(plan["attacks"]),
            goals_per_attack=goals_per_attack,
            max_steps=max_steps,
            language=language,
            target_description=td,
            show_progress=show_progress,
            on_goal_complete=on_goal_complete,
        ),
        llm_client=LLMClientAdapter(llm_client),
        target_client=HttpTargetClient(
            endpoint=endpoint,
            timeout_seconds=run_timeout_from_env(),
            api_key=os.getenv("LANGFLOW_API_KEY"),
            verify_ssl=verify,
        ),
        prompts_dir=_PACKAGE_ROOT / "prompts",
        datasets_dir=_PACKAGE_ROOT / "datasets",
    ).run()

    md = build_boart_markdown(plan, report, catalog, target_endpoint=endpoint)
    return plan, report, md


def save_boart_artifacts(
    run_dir: str | Path,
    *,
    plan: dict[str, Any],
    report: dict[str, Any],
    markdown: str,
) -> None:
    base = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "attack_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base / "boart_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base / "boart_report.md").write_text(markdown, encoding="utf-8")
    log.info("BOART артефакты: %s", base)


def save_pipeline_final_report(
    run_dir: str | Path,
    *,
    langflow_url: str,
    flow_id: str,
    synopsis: dict[str, Any],
    threat_model_markdown: str,
    validator_compliance: dict[str, Any],
    gate_verdict_status: str,
    gate_verdict_comment: str,
    flow_export_payload: dict[str, Any] | None = None,
    attack_plan: dict[str, Any] | None = None,
    boart_report: dict[str, Any] | None = None,
    boart_md: str | None = None,
) -> dict[str, Any]:
    """BOART-артефакты + ``final_report.json`` в формате legacy mlsecops-pipeline."""
    from final_report_builder import finalize_pipeline_artifacts

    if attack_plan is not None and boart_report is not None and boart_md is not None:
        save_boart_artifacts(
            run_dir, plan=attack_plan, report=boart_report, markdown=boart_md
        )
    flow_path = f"{langflow_url.rstrip('/')}/flow/{flow_id}"
    final = finalize_pipeline_artifacts(
        run_dir,
        flow_path=flow_path,
        synopsis=synopsis,
        threat_model_markdown=threat_model_markdown,
        validator_compliance=validator_compliance,
        gate_verdict_status=gate_verdict_status,
        gate_verdict_comment=gate_verdict_comment,
        flow_export_payload=flow_export_payload,
        attack_plan=attack_plan,
        boart_report=boart_report,
    )
    log.info("Итоговый отчёт: %s/final_report.json", run_dir)
    return final
