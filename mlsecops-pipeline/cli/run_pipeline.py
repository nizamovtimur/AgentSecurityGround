"""End-to-end security validation orchestrator (S1 -> S5).

Usage (from any directory):
    python -m cli.run_pipeline \
        --flow path/to/flow.json \
        --target-endpoint http://localhost:7860/api/v1/run/<FLOW_ID>

All path defaults are resolved relative to the mlsecops-pipeline package root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boart.runner import BoartConfig, BoartRunner
from boart.target_client import HttpTargetClient
from llm.openai_client import OpenAIClient, OpenAIConfig
from logging_utils import get_logger, setup_logging
from parsers.langflow_parser import parse_langflow_file, parse_langflow_flow
from services.attack_planner import plan_attacks
from services.final_report_builder import build_final_report
from services.langflow_client import LangflowFlowClient
from services.synopsis_builder import build_security_synopsis, build_target_description
from services.threat_modeling_service import ThreatModelingService

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS = _PACKAGE_ROOT / "prompts"
_DATASETS = _PACKAGE_ROOT / "datasets"
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full security validation pipeline (S1 static analysis → S2 threat modeling → "
        "S3 attack planning → S4 BOART → S5 final report).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flow", default="", help="Path to Langflow flow JSON (required for --flow-source file).")
    parser.add_argument(
        "--flow-source",
        choices=["file", "langflow"],
        default="file",
        help="Load flow from local file or directly from Langflow API.",
    )
    parser.add_argument("--flow-id", default="", help="Langflow flow ID (optional, else FLOW_ID env).")
    parser.add_argument("--langflow-url", default="", help="Langflow base URL (optional, else LANGFLOW_URL env).")
    parser.add_argument(
        "--langflow-api-key",
        default="",
        help="Langflow API key (optional, else LANGFLOW_API_KEY env).",
    )
    parser.add_argument(
        "--target-endpoint",
        required=True,
        help="Black-box target URL. Langflow: .../api/v1/run/<FLOW_ID> (uses LANGFLOW_API_KEY). "
        "Other: POST JSON {message, history}, response fields response|answer|output|message.",
    )
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="HTTP read timeout for target POST (Langflow run). Default: MLSECOPS_TARGET_TIMEOUT / "
        "LANGFLOW_RUN_TIMEOUT / OPENAI_TIMEOUT / 300s.",
    )
    parser.add_argument(
        "--attacks",
        default="",
        metavar="NAMES",
        help="Optional comma-separated dataset names (override auto-planning). E.g. system_prompt_leakage,harmbench_text",
    )
    parser.add_argument(
        "--attack-planner",
        choices=["agent", "heuristic"],
        default="agent",
        help="S3: agent = LLM по полному списку datasets + МУ (при сбое — эвристика); heuristic = только правила.",
    )
    parser.add_argument(
        "--target-description",
        default="",
        help="Описание цели для BOART. Если не задано — строится автоматически из S1 synopsis и S2 модели угроз.",
    )
    parser.add_argument("--goals-per-attack", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--language",
        choices=["ru", "en", "any"],
        default="ru",
        help="Dataset language filter; ru also switches BOART judge/summarizer to Russian templates.",
    )
    parser.add_argument("--max-strategies", type=int, default=10)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--prompts-dir", default=str(_PROMPTS), help="Directory with prompt templates.")
    parser.add_argument("--datasets-dir", default=str(_DATASETS), help="Directory with attack datasets.")
    parser.add_argument("--threat-model", default=str(_PROMPTS / "threat_model.txt"))
    parser.add_argument(
        "--threat-system-prompt",
        default=str(_PROMPTS / "threat_model_system_ru.txt"),
        help="System prompt template for threat modeling (placeholders: <THREAT_MODEL_CONTEXT>, <JSON>).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(_PACKAGE_ROOT.parent / "artifacts" / "pipeline"),
        help="Directory where all output files are written.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    artifacts_dir = Path(args.artifacts_dir)
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        llm_client = OpenAIClient(OpenAIConfig(model=args.model))

        # S1 — static flow analysis (from file or live Langflow)
        if args.flow_source == "langflow":
            logger.info("Loading flow from Langflow API.")
            lf_client = LangflowFlowClient.from_env(
                url=args.langflow_url or None,
                flow_id=args.flow_id or None,
                api_key=args.langflow_api_key or None,
            )
            flow_payload = lf_client.fetch_flow()
            graph = parse_langflow_flow(flow_payload)
            (artifacts_dir / "flow_from_langflow.json").write_text(
                json.dumps(flow_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            flow_source_label = f"langflow://{args.flow_id or 'FLOW_ID'}"
        else:
            if not args.flow:
                raise ValueError("--flow is required when --flow-source file.")
            logger.info("Loading flow from file: %s", args.flow)
            graph = parse_langflow_file(args.flow)
            flow_source_label = args.flow
        synopsis = build_security_synopsis(graph)
        (artifacts_dir / "security_synopsis.json").write_text(
            json.dumps(synopsis, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # S2 — MAESTRO threat modeling
        logger.info("Generating threat model.")
        threat_service = ThreatModelingService(
            openai_client=llm_client,
            threat_model_path=args.threat_model,
            system_prompt_path=args.threat_system_prompt,
        )
        threat_model_markdown = threat_service.generate_report(graph)
        (artifacts_dir / "threat_model.md").write_text(threat_model_markdown, encoding="utf-8")

        # S3 — attack planning
        if args.attacks.strip():
            selected_attacks = [s.strip() for s in args.attacks.split(",") if s.strip()]
            attack_plan = {
                "attacks": selected_attacks,
                "rationale": ["Ручное переопределение списка атак (флаг --attacks)."],
                "planner": "manual",
            }
        else:
            planned = plan_attacks(
                synopsis=synopsis,
                threat_model_markdown=threat_model_markdown,
                datasets_dir=args.datasets_dir,
                llm_client=llm_client,
                mode=args.attack_planner,
                prompts_dir=args.prompts_dir,
            )
            selected_attacks = planned.attacks
            attack_plan = planned.to_dict()
        (artifacts_dir / "attack_plan.json").write_text(
            json.dumps(attack_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # S4 — BOART adversarial testing
        logger.info("Running BOART adversarial testing.")
        boart_report = BoartRunner(
            config=BoartConfig(
                attacks=selected_attacks,
                goals_per_attack=args.goals_per_attack,
                max_steps=args.max_steps,
                language=args.language,
                max_strategies=args.max_strategies,
                target_description=(
                    args.target_description.strip()
                    or build_target_description(synopsis, threat_model_markdown)
                ),
            ),
            llm_client=llm_client,
            target_client=HttpTargetClient(
                endpoint=args.target_endpoint,
                **(
                    {}
                    if args.target_timeout is None
                    else {"timeout_seconds": float(args.target_timeout)}
                ),
            ),
            prompts_dir=args.prompts_dir,
            datasets_dir=args.datasets_dir,
        ).run()
        (artifacts_dir / "boart_report.json").write_text(
            json.dumps(boart_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # S5 — final report
        final_report = build_final_report(
            flow_path=flow_source_label,
            synopsis=synopsis,
            threat_model_markdown=threat_model_markdown,
            boart_report=boart_report,
            attack_plan=attack_plan,
        )
        (artifacts_dir / "final_report.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\nPipeline complete. Flow source: {flow_source_label}")
        print(f"Artifacts in: {artifacts_dir}")
        names = ["security_synopsis.json", "threat_model.md", "attack_plan.json", "boart_report.json", "final_report.json"]
        if args.flow_source == "langflow":
            names.insert(0, "flow_from_langflow.json")
        for name in names:
            print(f"  {name}")
    except Exception as exc:
        logger.exception("Pipeline execution failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
