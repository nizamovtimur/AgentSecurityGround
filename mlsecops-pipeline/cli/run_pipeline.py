"""Оркестратор: граф → угрозы/политики → (опц.) план атак → BOART → отчёт."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from llm.openai_client import OpenAIClient, OpenAIConfig
from logging_utils import get_logger, setup_logging
from services import build_final_report, build_security_synopsis
from services.final_report_builder import format_scan_summary
from services.pipeline_stages import (
    PlanInput,
    build_attack_plan,
    load_parsed_flow,
    run_boart,
    write_artifact,
    write_threat_and_compliance,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS = _PACKAGE_ROOT / "prompts"
_DATASETS = _PACKAGE_ROOT / "datasets"
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Разбор флоу Langflow, отчёт по угрозам и политикам, опционально BOART.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--flow", default="", help="JSON флоу Langflow (режим file).")
    p.add_argument("--flow-source", choices=["file", "langflow"], default="file")
    p.add_argument("--flow-id", default="", help="Иначе из FLOW_ID.")
    p.add_argument("--langflow-url", default="", help="Иначе из LANGFLOW_URL.")
    p.add_argument("--langflow-api-key", default="", help="Иначе из LANGFLOW_API_KEY.")

    p.add_argument("--target-endpoint", default="",
                   help="URL цели для BOART. Нужен, если не задан --no-boart.")
    p.add_argument("--target-timeout", type=float, default=None, metavar="SEC")
    p.add_argument("--target-description", default="",
                   help="Описание цели для BOART (по умолчанию собирается из графа и угроз).")

    p.add_argument("--no-compliance", action="store_true",
                   help="Пропустить проверки политик (нет compliance_report.json).")
    p.add_argument("--no-boart", action="store_true", help="Только синопсис и отчёты угроз/политик.")

    p.add_argument("--attacks", default="", metavar="NAMES",
                   help="Ручной список датасетов атак через запятую (stems).")
    p.add_argument("--attack-planner", choices=["agent", "heuristic"], default="agent")

    p.add_argument("--goals-per-attack", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--max-strategies", type=int, default=10)
    p.add_argument("--language", choices=["ru", "en", "any"], default="ru")

    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--prompts-dir", default=str(_PROMPTS))
    p.add_argument("--datasets-dir", default=str(_DATASETS))
    p.add_argument("--threat-model", default=str(_PROMPTS / "threat_model.txt"))
    p.add_argument("--threat-system-prompt", default=str(_PROMPTS / "threat_model_system_ru.txt"))
    p.add_argument("--artifacts-dir", default=str(_PACKAGE_ROOT.parent / "artifacts" / "pipeline"))

    p.add_argument("--verbose", action="store_true")

    p.add_argument(
        "--no-boart-progress",
        action="store_true",
        help="Не показывать tqdm во время BOART.",
    )
    return p.parse_args()


def _cli_load_graph(
    args: argparse.Namespace,
    artifacts_dir: Path,
    *,
    langflow_verify_ssl: bool | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    if args.flow_source == "langflow":
        return load_parsed_flow(
            flow_source="langflow",
            flow_file=None,
            artifacts_dir=artifacts_dir,
            flow_id=args.flow_id,
            langflow_url=args.langflow_url,
            langflow_api_key=args.langflow_api_key or None,
            langflow_verify_ssl=langflow_verify_ssl,
        )
    if not args.flow:
        raise ValueError("--flow is required when --flow-source file")
    return load_parsed_flow(
        flow_source="file",
        flow_file=Path(args.flow),
        artifacts_dir=artifacts_dir,
    )


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_boart and not args.target_endpoint:
        print("Ошибка: укажите --target-endpoint или используйте --no-boart.", file=sys.stderr)
        raise SystemExit(2)

    try:
        llm_client = OpenAIClient(OpenAIConfig(model=args.model))

        tls_explicit = False if args.insecure else None
        graph, flow_label, raw_flow = _cli_load_graph(
            args, artifacts_dir, langflow_verify_ssl=tls_explicit
        )
        synopsis = build_security_synopsis(graph)
        write_artifact(artifacts_dir / "security_synopsis.json", synopsis)

        log_stage = logger.info if args.verbose else logger.debug
        log_stage(
            "Синопсис: %d узлов, %d рёбер.",
            len(synopsis["nodes"]),
            len(synopsis["edges"]),
        )

        prompts_dir = Path(args.prompts_dir)
        datasets_dir = Path(args.datasets_dir)

        threat_md, compliance_report = write_threat_and_compliance(
            graph=graph,
            synopsis=synopsis,
            raw_flow=raw_flow,
            flow_source_label=flow_label,
            artifacts_dir=artifacts_dir,
            prompts_dir=prompts_dir,
            threat_template=Path(args.threat_model),
            threat_system_prompt=Path(args.threat_system_prompt),
            llm_client=llm_client,
            skip_compliance=args.no_compliance,
            generate_maestro=True,
        )
        log_stage("Угрозы и политики записаны.")

        attack_plan: dict[str, Any] | None = None
        boart_report: dict[str, Any] | None = None
        if not args.no_boart:
            attack_plan = build_attack_plan(
                PlanInput(
                    attacks_manual=args.attacks,
                    planner_mode=args.attack_planner,
                ),
                synopsis,
                threat_md,
                llm_client,
                prompts_dir=prompts_dir,
                datasets_dir=datasets_dir,
                compliance_report=compliance_report,
            )
            write_artifact(artifacts_dir / "attack_plan.json", attack_plan)
            log_stage("План атак: %d датасетов.", len(attack_plan["attacks"]))

            boart_report = run_boart(
                target_endpoint=args.target_endpoint,
                synopsis=synopsis,
                threat_md=threat_md,
                attacks=attack_plan["attacks"],
                llm_client=llm_client,
                prompts_dir=prompts_dir,
                datasets_dir=datasets_dir,
                compliance_report=compliance_report,
                goals_per_attack=args.goals_per_attack,
                max_steps=args.max_steps,
                language=args.language,
                max_strategies=args.max_strategies,
                target_timeout_s=args.target_timeout,
                target_description_override=args.target_description,
                target_verify_ssl=tls_explicit,
                show_progress=not args.no_boart_progress,
            )
            write_artifact(artifacts_dir / "boart_report.json", boart_report)
            log_stage("BOART завершён.")

            final_report = build_final_report(
                flow_path=flow_label,
                synopsis=synopsis,
                threat_model_markdown=threat_md,
                boart_report=boart_report,
                attack_plan=attack_plan,
                compliance_report=compliance_report,
                flow_export_payload=raw_flow,
            )
            write_artifact(artifacts_dir / "final_report.json", final_report)
            log_stage("Итоговый отчёт записан.")

        print()
        print(
            format_scan_summary(
                flow_source_label=flow_label,
                synopsis=synopsis,
                threat_model_markdown=threat_md,
                compliance_report=compliance_report,
                raw_flow_export=raw_flow,
                compliance_was_skipped=args.no_compliance,
            )
        )
        print()
        print(f"Готово. Артефакты: {artifacts_dir}")
        if not args.no_boart:
            print("Дополнительно: attack_plan.json, boart_report.json, final_report.json.")

    except Exception as exc:
        logger.exception("Сбой пайплайна.")
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
