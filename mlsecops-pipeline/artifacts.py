"""Сохранение артефактов security gate в заданную директорию."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logging_utils import get_logger
from report import GateVerdict

log = get_logger("artifacts")


def resolve_output_dir(explicit: str | Path | None = None) -> Path | None:
    """Каталог из аргумента или ``VALIDATOR_OUTPUT_DIR``; ``None`` — не сохранять."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = (os.getenv("VALIDATOR_OUTPUT_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return None


def prepare_run_dir(base: Path, flow_id: str) -> Path:
    """Подкаталог одного прогона: ``{flow_id_8}_{YYYYMMDD_HHMMSS}``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = flow_id.replace("-", "")[:8] or "flow"
    run_dir = base / f"{short}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_artifacts(
    run_dir: Path,
    *,
    flow_id: str,
    langflow_url: str,
    markdown: str,
    verdict: GateVerdict,
    compliance: dict[str, Any],
    agent_outputs: dict[str, Any],
    synopsis: dict[str, Any],
    threat_md: str,
    compliance_md: str,
    human_md: str,
) -> dict[str, Path]:
    """Записать файлы прогона; вернуть словарь имя → путь."""
    meta = {
        "flow_id": flow_id,
        "langflow_url": langflow_url,
        "verdict": verdict.model_dump(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    files: dict[str, tuple[str, str | dict]] = {
        "gate_report.md": ("text", markdown),
        "verdict.json": ("json", meta),
        "compliance.json": ("json", compliance),
        "agent_outputs.json": ("json", agent_outputs),
        "synopsis.json": ("json", synopsis),
        "system_prompts.json": (
            "json",
            synopsis.get("system_prompts", []),
        ),
        "threat_model.md": ("text", threat_md),
        "compliance_check.md": ("text", compliance_md),
        "human_review.md": ("text", human_md),
    }

    written: dict[str, Path] = {}
    for name, (kind, content) in files.items():
        path = run_dir / name
        if kind == "json":
            path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            path.write_text(content, encoding="utf-8")
        written[name] = path
        log.info("  артефакт: %s", path)

    return written
