"""Выбор подмножества datasets/*.parquet для BOART."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boart.llm_adapter import LLMClientAdapter
from llm import LLMClient
from logging_utils import get_logger

log = get_logger("attack_planner")
_MAX_THREAT_CHARS = 16_000
_MAX_COMPLIANCE_CHARS = 8_192

_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class AttackPlan:
    attacks: list[str]
    rationale: list[str]
    planner: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {"attacks": self.attacks, "rationale": self.rationale, "planner": self.planner}


def list_datasets(datasets_dir: str | Path | None = None) -> dict[str, str]:
    """Имена датасетов и описания из каталога."""
    base = Path(datasets_dir or _PACKAGE_ROOT / "datasets")
    catalog_path = _PACKAGE_ROOT / "prompts" / "attack_datasets_catalog.json"
    catalog: dict[str, str] = {}
    if catalog_path.is_file():
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            catalog = {str(k): str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    for p in sorted(base.glob("*.parquet")):
        out[p.stem] = catalog.get(p.stem, f"Датасет `{p.stem}`")
    return out


def _list_available(datasets_dir: Path) -> set[str]:
    return {p.stem for p in datasets_dir.glob("*.parquet")}


_LEAKAGE_MARKERS = (
    "system prompt", "prompt leakage", "instruction leakage", "sensitive prompt",
    "information disclosure", "утечк", "системн", "промпт", "инструкц", "раскрыти",
)
_JAILBREAK_MARKERS = (
    "jailbreak", "unsafe content", "harmful content", "toxic", "policy bypass",
    "джейлбрейк", "вредоносн", "опасн", "токсичн", "обход политик",
)


def select_attacks_heuristic(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    datasets_dir: str | Path,
) -> AttackPlan:
    available = _list_available(Path(datasets_dir))
    text = threat_model_markdown.lower()
    selected: list[str] = []
    rationale: list[str] = []

    if "system_prompt_leakage" in available and any(m in text for m in _LEAKAGE_MARKERS):
        selected.append("system_prompt_leakage")
        rationale.append("МУ: утечка системного промпта или инструкций.")
    if "harmbench_text" in available and any(m in text for m in _JAILBREAK_MARKERS):
        selected.append("harmbench_text")
        rationale.append("МУ: джейлбрейк или небезопасный контент.")

    has_agent = any(
        "agent" in str(n.get("type", "")).lower() or n.get("role") == "agent"
        for n in synopsis.get("nodes", [])
        if isinstance(n, dict)
    )
    if synopsis.get("entrypoints") and has_agent:
        if "system_prompt_leakage" in available and "system_prompt_leakage" not in selected:
            selected.append("system_prompt_leakage")
            rationale.append("Статика: агент и внешние точки входа.")
        if "harmbench_text" in available and "harmbench_text" not in selected:
            selected.append("harmbench_text")
            rationale.append("Статика: риск небезопасного вывода модели.")

    if not selected and available:
        fallback = sorted(available)[0]
        selected.append(fallback)
        rationale.append(f"Запасной выбор: «{fallback}».")

    return AttackPlan(attacks=selected, rationale=rationale, planner="heuristic")


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_plan(text: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    data = json.loads(_strip_json_fence(text))
    if not isinstance(data, dict):
        raise ValueError("Ответ LLM не JSON-объект.")
    raw_attacks = data.get("attacks")
    if not isinstance(raw_attacks, list) or not raw_attacks:
        raise ValueError("Пустой массив attacks.")
    attacks: list[str] = []
    for item in raw_attacks:
        if isinstance(item, str):
            name = item.strip()
            if name in allowed and name not in attacks:
                attacks.append(name)
    if not attacks:
        raise ValueError("Нет допустимых имён датасетов.")
    raw_rationale = data.get("rationale") or []
    rationale = [str(x).strip() for x in raw_rationale if str(x).strip()] if isinstance(raw_rationale, list) else []
    while len(rationale) < len(attacks):
        rationale.append("Выбрано агентом по модели угроз и synopsis.")
    return attacks, rationale[: len(attacks)]


def plan_attacks(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    *,
    llm_client: LLMClient | None = None,
    mode: str = "agent",
    datasets_dir: str | Path | None = None,
    prompts_dir: str | Path | None = None,
    compliance_comment: str | None = None,
    manual_attacks: list[str] | None = None,
) -> AttackPlan:
    if manual_attacks:
        return AttackPlan(
            attacks=list(manual_attacks),
            rationale=["Ручной выбор датасетов."],
            planner="manual",
        )

    ddir = Path(datasets_dir or _PACKAGE_ROOT / "datasets")
    available = _list_available(ddir)
    if not available:
        raise ValueError("В datasets/ нет файлов *.parquet.")

    pdir = Path(prompts_dir or _PACKAGE_ROOT / "prompts")

    if mode == "agent" and llm_client is not None:
        try:
            catalog = list_datasets(ddir)
            manifest = [{"name": n, "description": catalog.get(n)} for n in sorted(available)]
            system = (pdir / "attack_planning_agent_system_ru.txt").read_text(encoding="utf-8")
            tm = threat_model_markdown
            if len(tm) > _MAX_THREAT_CHARS:
                tm = tm[:_MAX_THREAT_CHARS] + "\n\n[… усечено …]"
            ds = compliance_comment
            if ds and len(ds) > _MAX_COMPLIANCE_CHARS:
                ds = ds[:_MAX_COMPLIANCE_CHARS].rstrip() + "…"
            payload: dict[str, Any] = {
                "available_datasets": manifest,
                "security_synopsis": synopsis,
                "threat_model_markdown": tm,
                "compliance_decision_statement": ds,
            }
            user_prompt = (
                "Проанализируй JSON и верни только JSON с полями attacks и rationale.\n\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )
            raw = LLMClientAdapter(llm_client).complete(
                system_prompt=system, user_prompt=user_prompt
            )
            attacks, rationale = _parse_llm_plan(raw, allowed=available)
            return AttackPlan(attacks=attacks, rationale=rationale, planner="agent")
        except Exception as exc:
            log.warning("LLM-планировщик: %s — эвристика.", exc)

    return select_attacks_heuristic(synopsis, threat_model_markdown, ddir)
