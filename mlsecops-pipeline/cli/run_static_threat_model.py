"""CLI for static flow analysis and MAESTRO threat modeling (standalone, S1+S2 only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm.openai_client import OpenAIClient, OpenAIConfig
from logging_utils import get_logger, setup_logging
from parsers.langflow_parser import parse_langflow_file, parse_langflow_flow
from services.langflow_client import LangflowFlowClient
from services.synopsis_builder import build_security_synopsis
from services.threat_modeling_service import ThreatModelingService

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS = _PACKAGE_ROOT / "prompts"
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run static analysis + MAESTRO threat modeling on a single Langflow flow.",
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
    parser.add_argument("--threat-model", default=str(_PROMPTS / "threat_model.txt"))
    parser.add_argument(
        "--system-prompt",
        default=str(_PROMPTS / "threat_model_system_ru.txt"),
        help="System prompt template for threat modeling (placeholders: <THREAT_MODEL_CONTEXT>, <JSON>).",
    )
    parser.add_argument(
        "--output",
        default=str(_PACKAGE_ROOT.parent / "artifacts" / "threat_model.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--synopsis-output",
        default=str(_PACKAGE_ROOT.parent / "artifacts" / "security_synopsis.json"),
        help="Output path for intermediate security synopsis JSON.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)

    try:
        if args.flow_source == "langflow":
            client = LangflowFlowClient.from_env(
                url=args.langflow_url or None,
                flow_id=args.flow_id or None,
                api_key=args.langflow_api_key or None,
            )
            logger.info("Loading flow from Langflow API.")
            flow_payload = client.fetch_flow()
            graph = parse_langflow_flow(flow_payload)
        else:
            if not args.flow:
                raise ValueError("--flow is required when --flow-source file.")
            logger.info("Loading flow from file: %s", args.flow)
            graph = parse_langflow_file(args.flow)

        synopsis = build_security_synopsis(graph)
        synopsis_path = Path(args.synopsis_output)
        synopsis_path.parent.mkdir(parents=True, exist_ok=True)
        synopsis_path.write_text(json.dumps(synopsis, ensure_ascii=False, indent=2), encoding="utf-8")

        client = OpenAIClient(OpenAIConfig(model=args.model))
        service = ThreatModelingService(
            openai_client=client,
            threat_model_path=args.threat_model,
            system_prompt_path=args.system_prompt,
        )
        logger.info("Generating threat model.")
        report = service.generate_report(graph)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        print(f"Security synopsis: {synopsis_path}")
        print(f"Threat model report: {output_path}")
    except Exception as exc:
        logger.exception("Static threat modeling failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
