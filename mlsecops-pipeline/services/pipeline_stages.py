"""Общие шаги CLI и ноутбука: загрузка флоу, артефакты угроз/политик, план атак, BOART."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from boart.runner import BoartConfig, BoartRunner
from boart.target_client import HttpTargetClient
from llm.openai_client import OpenAIClient, OpenAIConfig
from models.security_graph import SecurityGraph
from parsers.langflow_parser import parse_langflow_flow

from .attack_planner import plan_attacks
from .compliance_checker import run_compliance_checks
from .final_report_builder import build_security_assessment_markdown
from .langflow_client import LangflowFlowClient
from .synopsis_builder import build_target_description
from .threat_modeling_service import ThreatModelingService

_THREAT_STUB = "*MAESTRO не сгенерирован (нет ключа или generate_maestro=False).*"


def ssl_verify(explicit: bool | None, *, env_key: str, default: bool = True) -> bool:
    if explicit is not None:
        return explicit
    raw = (os.getenv(env_key) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


def write_artifact(path: Path, payload: Any) -> None:
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_parsed_flow(
    *,
    flow_source: Literal["file", "langflow"],
    flow_file: Path | None,
    artifacts_dir: Path,
    flow_id: str = "",
    langflow_url: str = "",
    langflow_api_key: str = "",
    fetch_timeout_s: float | None = None,
    langflow_verify_ssl: bool | None = None,
) -> tuple[SecurityGraph, str, dict[str, Any]]:
    timeout = float(
        fetch_timeout_s if fetch_timeout_s is not None else os.getenv("MLSECOPS_LANGFLOW_API_TIMEOUT", "120")
    )
    verify = ssl_verify(
        langflow_verify_ssl,
        env_key="MLSECOPS_LANGFLOW_VERIFY_SSL",
        default=True,
    )

    if flow_source == "langflow":
        client = LangflowFlowClient.from_env(
            url=(langflow_url or None),
            flow_id=(flow_id or None),
            api_key=(langflow_api_key or None),
            timeout_seconds=timeout,
            verify=verify,
        )
        payload = client.fetch_flow()
        write_artifact(artifacts_dir / "flow_from_langflow.json", payload)
        raw = payload if isinstance(payload, dict) else {}
        return parse_langflow_flow(payload), f"langflow://{flow_id or 'FLOW_ID'}", raw

    if not flow_file or not Path(flow_file).is_file():
        raise ValueError("--flow (путь к JSON) обязателен при источнике «file»")
    raw = json.loads(Path(flow_file).read_text(encoding="utf-8"))
    return parse_langflow_flow(raw), str(flow_file), raw if isinstance(raw, dict) else {}


def load_flow_or_file(
    *,
    artifacts_dir: Path,
    langflow_url: str,
    flow_id: str,
    langflow_api_key: str,
    fallback_flow_file: Path | None = None,
    fetch_timeout_s: float | None = None,
    langflow_verify_ssl: bool | None = None,
) -> tuple[SecurityGraph, str, dict[str, Any], str]:
    """Langflow API при ключе; иначе JSON с диска, если передан ``fallback_flow_file``."""
    timeout = (
        fetch_timeout_s
        if fetch_timeout_s is not None
        else float(os.getenv("MLSECOPS_LANGFLOW_API_TIMEOUT", "120"))
    )
    last_exc: BaseException | None = None
    if langflow_api_key:
        try:
            g, label, raw = load_parsed_flow(
                flow_source="langflow",
                flow_file=None,
                artifacts_dir=artifacts_dir,
                flow_id=flow_id,
                langflow_url=langflow_url,
                langflow_api_key=langflow_api_key,
                fetch_timeout_s=timeout,
                langflow_verify_ssl=langflow_verify_ssl,
            )
            return g, label, raw, "langflow-api"
        except Exception as exc:
            last_exc = exc
    if fallback_flow_file is None:
        msg = (
            "Нужен граф: задайте fallback_flow_file=Path(...) к JSON флоу, либо "
            "почините доступ к Langflow (LANGFLOW_URL, FLOW_ID, LANGFLOW_API_KEY)."
        )
        if last_exc is not None:
            raise RuntimeError(msg) from last_exc
        raise RuntimeError(msg + " Сейчас LANGFLOW_API_KEY пуст — без файла загрузить негде.")
    fb = Path(fallback_flow_file)
    if not fb.is_file():
        raise FileNotFoundError(f"Файл флоу не найден: {fb}")
    raw_fallback = json.loads(fb.read_text(encoding="utf-8"))
    g = parse_langflow_flow(raw_fallback)
    raw = raw_fallback if isinstance(raw_fallback, dict) else {}
    return g, str(fb), raw, "local-file"


def decision_from_compliance(report: dict[str, Any] | None) -> str | None:
    if report is None:
        return None
    ds = report.get("decision_statement")
    return ds.strip() if isinstance(ds, str) and ds.strip() else None


@dataclass
class PlanInput:
    attacks_manual: str = ""
    planner_mode: Literal["agent", "heuristic"] = "agent"


def build_attack_plan(
    opts: PlanInput,
    synopsis: dict[str, Any],
    threat_md: str,
    llm_client: OpenAIClient | None,
    *,
    prompts_dir: Path,
    datasets_dir: Path,
    compliance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if opts.attacks_manual.strip():
        return {
            "attacks": [s.strip() for s in opts.attacks_manual.split(",") if s.strip()],
            "rationale": ["Ручной список атак (--attacks)."],
            "planner": "manual",
        }
    plan = plan_attacks(
        synopsis=synopsis,
        threat_model_markdown=threat_md,
        datasets_dir=str(datasets_dir),
        llm_client=llm_client,
        mode=opts.planner_mode,
        prompts_dir=str(prompts_dir),
        compliance_decision_statement=decision_from_compliance(compliance_report),
    )
    return plan.to_dict()


def run_boart(
    *,
    target_endpoint: str,
    synopsis: dict[str, Any],
    threat_md: str,
    attacks: list[str],
    llm_client: OpenAIClient,
    prompts_dir: Path,
    datasets_dir: Path,
    compliance_report: dict[str, Any] | None,
    goals_per_attack: int = 3,
    max_steps: int = 5,
    language: str = "ru",
    max_strategies: int = 10,
    target_timeout_s: float | None = None,
    target_description_override: str = "",
    target_verify_ssl: bool | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    tgt_kw: dict[str, Any] = {
        "verify_ssl": ssl_verify(
            target_verify_ssl,
            env_key="MLSECOPS_TARGET_VERIFY_SSL",
            default=True,
        ),
    }
    if target_timeout_s is not None:
        tgt_kw["timeout_seconds"] = float(target_timeout_s)

    td = (
        target_description_override.strip()
        or build_target_description(
            synopsis,
            threat_md,
            compliance_decision_statement=decision_from_compliance(compliance_report),
        )
    )
    return BoartRunner(
        config=BoartConfig(
            attacks=attacks,
            goals_per_attack=goals_per_attack,
            max_steps=max_steps,
            language=language,
            max_strategies=max_strategies,
            target_description=td,
            show_progress=show_progress,
        ),
        llm_client=llm_client,
        target_client=HttpTargetClient(endpoint=target_endpoint, **tgt_kw),
        prompts_dir=str(prompts_dir),
        datasets_dir=str(datasets_dir),
    ).run()


def write_threat_and_compliance(
    *,
    graph: SecurityGraph,
    synopsis: dict[str, Any],
    raw_flow: dict[str, Any],
    flow_source_label: str,
    artifacts_dir: Path,
    prompts_dir: Path,
    threat_template: Path,
    threat_system_prompt: Path,
    llm_client: OpenAIClient | None,
    skip_compliance: bool = False,
    generate_maestro: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    threat_md = ""
    if generate_maestro and llm_client is not None:
        svc = ThreatModelingService(
            openai_client=llm_client,
            threat_model_path=threat_template,
            system_prompt_path=threat_system_prompt,
        )
        threat_md = svc.generate_report(graph)

    text_for_disk = threat_md.strip() if threat_md.strip() else _THREAT_STUB
    write_artifact(artifacts_dir / "threat_model.md", text_for_disk + "\n")

    compliance_report: dict[str, Any] | None = None
    if not skip_compliance:
        compliance_report = run_compliance_checks(synopsis, llm_client=llm_client)
        write_artifact(artifacts_dir / "compliance_report.json", compliance_report)

    tm_for_md = threat_md.strip() or _THREAT_STUB
    assessment_md = build_security_assessment_markdown(
        threat_model_markdown=tm_for_md,
        compliance_report=compliance_report,
        flow_export_metadata=raw_flow,
        flow_source_label=flow_source_label,
    )
    write_artifact(artifacts_dir / "security_assessment.md", assessment_md)

    return tm_for_md, compliance_report
