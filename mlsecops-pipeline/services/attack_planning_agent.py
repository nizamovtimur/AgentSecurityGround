"""LLM-based attack dataset planner (S3): all datasets in → minimal relevant subset out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.openai_client import OpenAIClient
from logging_utils import get_logger

from services.attack_planner import AttackPlan

logger = get_logger(__name__)

_MAX_THREAT_MODEL_CHARS = 16_000


def _load_dataset_catalog(prompts_dir: Path) -> dict[str, str]:
    path = prompts_dir / "attack_datasets_catalog.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read attack_datasets_catalog.json: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def build_available_datasets_manifest(
    dataset_stems: set[str],
    prompts_dir: Path,
) -> list[dict[str, Any]]:
    catalog = _load_dataset_catalog(prompts_dir)
    manifest: list[dict[str, Any]] = []
    for name in sorted(dataset_stems):
        desc = catalog.get(name)
        manifest.append(
            {
                "name": name,
                "description": desc if desc else None,
            }
        )
    return manifest


def _read_system_prompt(prompts_dir: Path) -> str:
    path = prompts_dir / "attack_planning_agent_system_ru.txt"
    return path.read_text(encoding="utf-8")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_attack_plan_json(text: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    cleaned = _strip_json_fence(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object.")
    attacks_raw = data.get("attacks")
    rationale_raw = data.get("rationale")
    if not isinstance(attacks_raw, list) or not attacks_raw:
        raise ValueError("Missing or empty attacks array.")
    attacks: list[str] = []
    for item in attacks_raw:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if name in allowed and name not in attacks:
            attacks.append(name)
    if not attacks:
        raise ValueError("No valid attack names after filtering to available datasets.")
    rationale: list[str] = []
    if isinstance(rationale_raw, list):
        rationale = [str(x).strip() for x in rationale_raw if str(x).strip()]
    while len(rationale) < len(attacks):
        rationale.append("Выбрано агентом планирования по соответствию модели угроз и synopsis.")
    if len(rationale) > len(attacks):
        rationale = rationale[: len(attacks)]
    return attacks, rationale


def plan_attacks_with_llm(
    *,
    synopsis: dict[str, Any],
    threat_model_markdown: str,
    available_stems: set[str],
    llm_client: OpenAIClient,
    prompts_dir: str | Path,
) -> AttackPlan:
    """Ask LLM to pick a minimal subset of dataset stems; raises on failure (caller may fallback)."""
    pdir = Path(prompts_dir)
    system = _read_system_prompt(pdir)
    manifest = build_available_datasets_manifest(available_stems, pdir)
    if not manifest:
        raise ValueError("No datasets in catalog (empty datasets directory).")

    tm = threat_model_markdown if len(threat_model_markdown) <= _MAX_THREAT_MODEL_CHARS else (
        threat_model_markdown[:_MAX_THREAT_MODEL_CHARS]
        + "\n\n[… текст модели угроз усечён для планировщика …]"
    )
    user_payload = {
        "available_datasets": manifest,
        "security_synopsis": synopsis,
        "threat_model_markdown": tm,
    }
    user_prompt = (
        "Проанализируй входной JSON и верни только JSON с полями attacks и rationale "
        "(и опционально rejected), как в системной инструкции.\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )

    raw = llm_client.complete(system_prompt=system, user_prompt=user_prompt)
    attacks, rationale = _parse_attack_plan_json(raw, allowed=available_stems)
    return AttackPlan(attacks=attacks, rationale=rationale, planner="agent")
