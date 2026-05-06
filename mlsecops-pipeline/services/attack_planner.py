"""Automatic attack dataset planner based on S1/S2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from logging_utils import get_logger

if TYPE_CHECKING:
    from llm.openai_client import OpenAIClient

logger = get_logger(__name__)


@dataclass(slots=True)
class AttackPlan:
    """План атак: `planner` — agent (LLM) или heuristic (маркеры + статика)."""

    attacks: list[str]
    rationale: list[str]
    planner: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {"attacks": self.attacks, "rationale": self.rationale, "planner": self.planner}


def _list_available_datasets(datasets_dir: str | Path) -> set[str]:
    base = Path(datasets_dir)
    return {path.stem for path in base.glob("*.parquet")}


def select_attacks_from_context(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    datasets_dir: str | Path,
) -> AttackPlan:
    available = _list_available_datasets(datasets_dir)
    text = threat_model_markdown.lower()
    selected: list[str] = []
    rationale: list[str] = []

    # Threat-driven planning: choose relevant datasets based on threat model markers
    leakage_markers = (
        "system prompt",
        "prompt leakage",
        "instruction leakage",
        "sensitive prompt",
        "information disclosure",
        "утечк",
        "системн",
        "промпт",
        "инструкц",
        "раскрыти",
        "конфиденциальн",
    )
    jailbreak_markers = (
        "jailbreak",
        "unsafe content",
        "harmful content",
        "toxic",
        "policy bypass",
        "джейлбрейк",
        "вредоносн",
        "опасн",
        "токсичн",
        "обход политик",
    )

    if "system_prompt_leakage" in available:
        if any(marker in text for marker in leakage_markers):
            selected.append("system_prompt_leakage")
            rationale.append("Подобрано по маркерам модели угроз: утечка системного промпта или инструкций.")

    if "harmbench_text" in available:
        if any(marker in text for marker in jailbreak_markers):
            selected.append("harmbench_text")
            rationale.append("Подобрано по маркерам модели угроз: джейлбрейк или генерация небезопасного контента.")

    # Architecture-driven fallback from static analysis
    entrypoints = synopsis.get("entrypoints", [])
    assets = synopsis.get("assets", [])
    has_agent_surface = any("::agent" in str(asset).lower() for asset in assets if isinstance(asset, str))
    has_entrypoints = isinstance(entrypoints, list) and len(entrypoints) > 0

    if "system_prompt_leakage" in available and "system_prompt_leakage" not in selected:
        if has_agent_surface and has_entrypoints:
            selected.append("system_prompt_leakage")
            rationale.append("Подобрано по статическому анализу: поверхность агента и внешние точки входа.")

    if "harmbench_text" in available and "harmbench_text" not in selected:
        if has_agent_surface:
            selected.append("harmbench_text")
            rationale.append("Подобрано по статическому анализу: присутствует модель/агент с риском небезопасного вывода.")

    # Conservative fallback: choose first available dataset if still empty
    if not selected and available:
        fallback = sorted(available)[0]
        selected.append(fallback)
        rationale.append(
            f"Запасной выбор: явные маркеры не найдены, выбран первый доступный набор данных «{fallback}»."
        )

    return AttackPlan(attacks=selected, rationale=rationale, planner="heuristic")


def plan_attacks(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    datasets_dir: str | Path,
    *,
    llm_client: OpenAIClient | None = None,
    mode: str = "agent",
    prompts_dir: str | Path | None = None,
) -> AttackPlan:
    """S3: по умолчанию LLM-агент по всем `datasets/*.parquet`; при сбое или `heuristic` — старые правила."""
    from services.attack_planning_agent import plan_attacks_with_llm

    available = _list_available_datasets(datasets_dir)
    if not available:
        raise ValueError("В каталоге datasets нет файлов *.parquet.")

    prompts_path = (
        Path(prompts_dir)
        if prompts_dir is not None
        else Path(__file__).resolve().parents[1] / "prompts"
    )

    if mode == "agent" and llm_client is not None:
        try:
            return plan_attacks_with_llm(
                synopsis=dict(synopsis),
                threat_model_markdown=threat_model_markdown,
                available_stems=available,
                llm_client=llm_client,
                prompts_dir=prompts_path,
            )
        except Exception as exc:
            logger.warning("Агент планирования атак (LLM) недоступен: %s — эвристика.", exc, exc_info=True)

    return select_attacks_from_context(synopsis, threat_model_markdown, datasets_dir)

