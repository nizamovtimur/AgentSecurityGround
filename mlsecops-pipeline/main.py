"""Оркестратор security gate: fetch → sanitize → 3 LLM-агента → Markdown."""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv()
from contextlib import contextmanager
from typing import Any, Iterator

from agents import (
    ComplianceAgent,
    HumanReviewAgent,
    ThreatModelAgent,
    human_review_from_compliance,
)
from config import resolve_openai_base_url, ssl_verify
from flow_parser import parse_flow
from langflow_client import fetch_flow, load_flow_from_file
from llm import LLMClient
from logging_utils import get_logger, setup_logging
from artifacts import prepare_run_dir, resolve_output_dir, save_artifacts
from report import SecurityGateReport, build_markdown, compute_verdict
from synopsis import build_synopsis

log = get_logger("main")


@contextmanager
def _step(name: str) -> Iterator[None]:
    log.info("▶ %s …", name)
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        log.exception("✗ %s — ошибка (%.1f с)", name, time.monotonic() - t0)
        raise
    else:
        log.info("✓ %s — готово (%.1f с)", name, time.monotonic() - t0)


def _log_synopsis_brief(synopsis: dict[str, Any]) -> None:
    s = synopsis.get("summary", {})
    log.debug(
        "  граф: узлов=%s, рёбер=%s, entrypoints=%s, controls=%s, system_prompts=%s, tool_edges=%s",
        s.get("nodes"), s.get("edges"), s.get("entrypoints"), s.get("controls"),
        len(synopsis.get("system_prompts", [])), len(synopsis.get("tool_edges", [])),
    )


def _log_compliance(compliance: dict[str, Any]) -> None:
    for req_id, data in compliance.items():
        if isinstance(data, dict):
            log.info("  %s → %s", req_id, data.get("status", "?"))
            if req_id == "REQ-DATA-MIN" and data.get("status") == "FAIL":
                for ev in (data.get("evidence") or [])[:3]:
                    if isinstance(ev, dict):
                        log.info("    ↳ %s: %s", ev.get("node"), (ev.get("reason") or "")[:120])


def run_security_gate(
    langflow_url: str,
    flow_id: str,
    *,
    langflow_api_key: str | None = None,
    langflow_ssl_verify: bool | None = None,
    llm: LLMClient | None = None,
    openai_base_url: str | None = None,
    openai_model: str | None = None,
    openai_ssl_verify: bool | None = None,
    print_report: bool = True,
    log_level: str = "INFO",
    output_dir: str | os.PathLike[str] | None = None,
    flow_file: str | os.PathLike[str] | None = None,
    flow_fetch_result: Any = None,
) -> SecurityGateReport:
    """
    Security gate для агентного сценария Langflow.

    :param log_level: DEBUG/INFO/WARNING (по умолчанию INFO)
    :param output_dir: каталог артефактов (или VALIDATOR_OUTPUT_DIR); None — не сохранять
    :param flow_file: путь к локальному JSON-файлу (альтернатива flow_fetch_result)
    :param flow_fetch_result: предзагруженный FlowFetchResult (альтернатива flow_file)
    """
    setup_logging(log_level)
    lf_verify = ssl_verify("langflow", override=langflow_ssl_verify)
    oa_verify = ssl_verify("openai", override=openai_ssl_verify)
    oa_url = openai_base_url or resolve_openai_base_url()
    oa_model = openai_model

    out_base = resolve_output_dir(output_dir)
    log.info(
        "Security Gate: flow=%s, model=%s, artifacts=%s",
        flow_id,
        oa_model or "(default)",
        "да" if out_base else "нет",
    )

    with _step("Загрузка flow"):
        if flow_fetch_result is not None:
            fetched = flow_fetch_result
        elif flow_file is not None:
            fetched = load_flow_from_file(flow_file)
            flow_id = fetched.metadata.get("flow_id") or fetched.metadata.get("name") or str(flow_file)
            langflow_url = "file://local"
        else:
            fetched = fetch_flow(
                langflow_url,
                flow_id,
                api_key=langflow_api_key,
                ssl_verify=langflow_ssl_verify,
            )
        agent_outputs = fetched.metadata
        nodes_n = len(fetched.graph.get("data", {}).get("nodes", []))
        edges_n = len(fetched.graph.get("data", {}).get("edges", []))
        log.info("  получено: nodes=%s, edges=%s", nodes_n, edges_n)

    with _step("Парсинг и нормализация графа"):
        graph = parse_flow(fetched.graph)

    with _step("Построение security synopsis"):
        synopsis = build_synopsis(graph)
        _log_synopsis_brief(synopsis)

    with _step("Инициализация LLM-клиента"):
        client = llm or LLMClient(
            base_url=openai_base_url,
            model=openai_model,
            verify_ssl=openai_ssl_verify,
        )

    with _step("Агент 1/3 — модель угроз и митигации"):
        threat_md = ThreatModelAgent(client).run(synopsis)
        log.debug("  ThreatModelAgent ответ: %s симв.", len(threat_md))

    with _step("Агент 2/3 — проверка соответствия (REQ)"):
        try:
            compliance, compliance_md = ComplianceAgent(client).run(synopsis)
        except Exception as exc:
            log.error("ComplianceAgent: сбой LLM (%s), только статика", exc)
            compliance, compliance_md = ComplianceAgent.static_fallback(synopsis)
        _log_compliance(compliance)
        log.debug("  ComplianceAgent отчёт: %s симв.", len(compliance_md))

    with _step("Агент 3/3 — сигналы для ручной проверки"):
        try:
            human_md = HumanReviewAgent(client).run(synopsis)
            log.debug("  HumanReviewAgent ответ: %s симв.", len(human_md))
        except Exception as exc:
            log.warning("HumanReviewAgent недоступен (%s), fallback из REQ-HUMAN-REVIEW", exc)
            human_md = human_review_from_compliance(compliance)

    with _step("Итоговый вердикт и сборка отчёта"):
        verdict = compute_verdict(compliance)
        log.info("  вердикт: %s — %s", verdict.status, verdict.comment)
        md = build_markdown(
            flow_id=flow_id,
            langflow_url=langflow_url.rstrip("/"),
            threat_md=threat_md,
            compliance_md=compliance_md,
            human_review_md=human_md,
            agent_outputs=agent_outputs,
            verdict=verdict,
        )
        log.debug("  полный отчёт: %s симв.", len(md))

    artifacts_path: str | None = None
    if out_base:
        with _step("Сохранение артефактов"):
            run_dir = prepare_run_dir(out_base, flow_id)
            save_artifacts(
                run_dir,
                flow_id=flow_id,
                langflow_url=langflow_url.rstrip("/"),
                markdown=md,
                verdict=verdict,
                compliance=compliance,
                agent_outputs=agent_outputs,
                synopsis=synopsis,
                threat_md=threat_md,
                compliance_md=compliance_md,
                human_md=human_md,
            )
            artifacts_path = str(run_dir)
            log.info("  каталог прогона: %s", run_dir)

    report = SecurityGateReport(
        flow_id=flow_id,
        langflow_url=langflow_url.rstrip("/"),
        verdict=verdict,
        markdown=md,
        agent_outputs=agent_outputs,
        compliance=compliance,
        artifacts_dir=artifacts_path,
    )

    log.info("=" * 60)
    log.info("Security Gate — завершён: %s", verdict.status)
    if print_report:
        print(md)
    log.info("Готово: %s", verdict.status)
    return report


def main() -> None:
    import argparse
    from pathlib import Path as PathArg

    p = argparse.ArgumentParser(description="MLSecOps security gate для Langflow")
    p.add_argument("--url", default=os.getenv("LANGFLOW_URL", "http://localhost:7860"))
    p.add_argument("--flow-id", default=os.getenv("FLOW_ID", ""))
    p.add_argument(
        "--flow-file", "-f",
        type=PathArg,
        default=None,
        help="Путь к локальному JSON-файлу с flow (вместо --flow-id и --url)",
    )
    p.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Отключить проверку TLS для Langflow и LLM (или SSL_VERIFY=false)",
    )
    p.add_argument("--openai-base-url", default=None, help="Корпоративный OpenAI-compatible URL")
    p.add_argument("--openai-model", default=None, help="Модель LLM для всех агентов")
    p.add_argument(
        "-v", "--verbose",
        action="store_const",
        const="DEBUG",
        dest="log_level",
        help="Подробные логи (DEBUG)",
    )
    p.add_argument("--log-level", default=os.getenv("VALIDATOR_LOG_LEVEL", "INFO"))
    p.add_argument(
        "-o", "--output-dir",
        default=os.getenv("VALIDATOR_OUTPUT_DIR"),
        help="Каталог для артефактов (или VALIDATOR_OUTPUT_DIR)",
    )
    args = p.parse_args()

    verify = False if args.no_ssl_verify else None

    if not args.flow_file and not args.flow_id and "FLOW_ID" not in os.environ:
        p.error("Укажите --flow-id, FLOW_ID или используйте --flow-file")
        return

    run_security_gate(
        args.url if not args.flow_file else "file://local",
        args.flow_id or str(args.flow_file.stem),
        langflow_api_key=None,
        langflow_ssl_verify=verify,
        openai_ssl_verify=verify,
        openai_base_url=args.openai_base_url or resolve_openai_base_url(),
        openai_model=args.openai_model,
        log_level=args.log_level or "INFO",
        output_dir=args.output_dir,
        flow_file=args.flow_file,
    )


if __name__ == "__main__":
    main()
