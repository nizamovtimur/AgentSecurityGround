"""CLI entrypoint for BOART adversarial multi-agent testing (standalone, S4 only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boart.runner import BoartConfig, BoartRunner
from boart.target_client import HttpTargetClient
from llm.openai_client import OpenAIClient, OpenAIConfig
from logging_utils import get_logger, setup_logging

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS = _PACKAGE_ROOT / "prompts"
_DATASETS = _PACKAGE_ROOT / "datasets"
logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BOART adversarial multi-agent testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--attacks",
        required=True,
        metavar="NAMES",
        help="Comma-separated dataset names without extension, e.g. system_prompt_leakage,harmbench_text",
    )
    parser.add_argument(
        "--target-endpoint",
        required=True,
        help="Target URL. Langflow chat run: http://host/api/v1/run/<FLOW_ID> (header x-api-key from LANGFLOW_API_KEY). "
        "Otherwise: POST {message, history}.",
    )
    parser.add_argument(
        "--target-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="HTTP timeout for target (default from MLSECOPS_TARGET_TIMEOUT / LANGFLOW_RUN_TIMEOUT / OPENAI_TIMEOUT / 300).",
    )
    parser.add_argument(
        "--target-description",
        default="",
        help="Описание цели для BOART (что атакуем). Обязательно для run_boart; для run_pipeline строится автоматически из S1+S2.",
    )
    parser.add_argument("--goals-per-attack", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--language",
        choices=["ru", "en", "any"],
        default="ru",
        help="Dataset language; ru enables Russian judge/summarizer prompts.",
    )
    parser.add_argument("--max-strategies", type=int, default=10)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--datasets-dir", default=str(_DATASETS))
    parser.add_argument("--prompts-dir", default=str(_PROMPTS))
    parser.add_argument(
        "--output",
        default=str(_PACKAGE_ROOT.parent / "artifacts" / "boart_report.json"),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(verbose=args.verbose)
    try:
        logger.info("Starting BOART run against endpoint: %s", args.target_endpoint)
        report = BoartRunner(
            config=BoartConfig(
                attacks=[s.strip() for s in args.attacks.split(",") if s.strip()],
                goals_per_attack=args.goals_per_attack,
                max_steps=args.max_steps,
                language=args.language,
                max_strategies=args.max_strategies,
                target_description=args.target_description,
            ),
            llm_client=OpenAIClient(OpenAIConfig(model=args.model)),
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

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"BOART report: {output_path}")
    except Exception as exc:
        logger.exception("BOART execution failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
