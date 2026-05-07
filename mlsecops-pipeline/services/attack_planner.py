"""S3 — Attack dataset planner: pick a minimal subset of `datasets/*.parquet` for BOART.

Two modes (both behind one entry point :func:`plan_attacks`):

  agent     — LLM picks based on synopsis + threat model + dataset catalog.
              On any LLM/parse failure falls back to ``heuristic``.
  heuristic — text markers in threat model markdown + structural signals from synopsis.
              Used as a deterministic baseline; also exposed as :func:`select_attacks_from_context`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logging_utils import get_logger

if TYPE_CHECKING:
    from llm.openai_client import OpenAIClient

logger = get_logger(__name__)
_MAX_THREAT_MODEL_CHARS = 16_000
_MAX_COMPLIANCE_DECISION_CHARS = 8_192


def _normalize_compliance_statement(text: str | None, max_chars: int) -> str | None:
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    if len(s) > max_chars:
        return s[:max_chars].rstrip() + "…"
    return s


@dataclass(slots=True)
class AttackPlan:
    """Selected datasets + per-item rationale; ``planner`` records who picked them."""
    attacks: list[str]
    rationale: list[str]
    planner: str = "heuristic"   # agent | heuristic | manual

    def to_dict(self) -> dict[str, object]:
        return {"attacks": self.attacks, "rationale": self.rationale, "planner": self.planner}


# ---------------------------------------------------------------------------
# Filesystem & catalog helpers
# ---------------------------------------------------------------------------

def _list_available_datasets(datasets_dir: str | Path) -> set[str]:
    return {p.stem for p in Path(datasets_dir).glob("*.parquet")}


def _load_dataset_catalog(prompts_dir: Path) -> dict[str, str]:
    path = prompts_dir / "attack_datasets_catalog.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read attack_datasets_catalog.json: %s", exc)
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)} if isinstance(raw, dict) else {}


# ---------------------------------------------------------------------------
# Heuristic planner (fallback, deterministic, no LLM)
# ---------------------------------------------------------------------------

_LEAKAGE_MARKERS = (
    "system prompt", "prompt leakage", "instruction leakage", "sensitive prompt",
    "information disclosure",
    "утечк", "системн", "промпт", "инструкц", "раскрыти", "конфиденциальн",
)
_JAILBREAK_MARKERS = (
    "jailbreak", "unsafe content", "harmful content", "toxic", "policy bypass",
    "джейлбрейк", "вредоносн", "опасн", "токсичн", "обход политик",
)


def select_attacks_from_context(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    datasets_dir: str | Path,
) -> AttackPlan:
    """Heuristic baseline used as fallback for ``plan_attacks(mode='agent')`` failures."""
    available = _list_available_datasets(datasets_dir)
    text = threat_model_markdown.lower()
    selected: list[str] = []
    rationale: list[str] = []

    if "system_prompt_leakage" in available and any(m in text for m in _LEAKAGE_MARKERS):
        selected.append("system_prompt_leakage")
        rationale.append("Маркеры МУ: утечка системного промпта или инструкций.")
    if "harmbench_text" in available and any(m in text for m in _JAILBREAK_MARKERS):
        selected.append("harmbench_text")
        rationale.append("Маркеры МУ: джейлбрейк или генерация небезопасного контента.")

    # Architecture fallback
    assets = synopsis.get("assets", [])
    has_agent = any("::agent" in str(a).lower() for a in assets if isinstance(a, str))
    has_entrypoints = bool(synopsis.get("entrypoints"))
    if has_agent and has_entrypoints and "system_prompt_leakage" in available \
            and "system_prompt_leakage" not in selected:
        selected.append("system_prompt_leakage")
        rationale.append("Статика: поверхность агента и внешние точки входа.")
    if has_agent and "harmbench_text" in available and "harmbench_text" not in selected:
        selected.append("harmbench_text")
        rationale.append("Статика: модель/агент с риском небезопасного вывода.")

    if not selected and available:
        fallback = sorted(available)[0]
        selected.append(fallback)
        rationale.append(f"Запасной выбор: «{fallback}» (явные маркеры не найдены).")

    return AttackPlan(attacks=selected, rationale=rationale, planner="heuristic")


# ---------------------------------------------------------------------------
# LLM agent (mode="agent")
# ---------------------------------------------------------------------------

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
        raise ValueError("LLM output is not a JSON object.")
    raw_attacks = data.get("attacks")
    if not isinstance(raw_attacks, list) or not raw_attacks:
        raise ValueError("Missing or empty 'attacks' array.")

    attacks: list[str] = []
    for item in raw_attacks:
        if isinstance(item, str):
            name = item.strip()
            if name in allowed and name not in attacks:
                attacks.append(name)
    if not attacks:
        raise ValueError("No valid attack names after filtering to available datasets.")

    raw_rationale = data.get("rationale") or []
    rationale = [str(x).strip() for x in raw_rationale if str(x).strip()] if isinstance(raw_rationale, list) else []
    while len(rationale) < len(attacks):
        rationale.append("Выбрано агентом по соответствию модели угроз и synopsis.")
    return attacks, rationale[:len(attacks)]


def _plan_with_llm(
    *,
    synopsis: dict[str, Any],
    threat_model_markdown: str,
    available: set[str],
    llm_client: "OpenAIClient",
    prompts_dir: Path,
    compliance_decision_statement: str | None,
) -> AttackPlan:
    catalog = _load_dataset_catalog(prompts_dir)
    manifest = [{"name": n, "description": catalog.get(n)} for n in sorted(available)]
    if not manifest:
        raise ValueError("В каталоге datasets нет файлов *.parquet.")

    system = (prompts_dir / "attack_planning_agent_system_ru.txt").read_text(encoding="utf-8")
    tm = threat_model_markdown
    if len(tm) > _MAX_THREAT_MODEL_CHARS:
        tm = tm[:_MAX_THREAT_MODEL_CHARS] + "\n\n[… усечено …]"

    payload: dict[str, Any] = {
        "available_datasets": manifest,
        "security_synopsis": synopsis,
        "threat_model_markdown": tm,
        "compliance_decision_statement": _normalize_compliance_statement(
            compliance_decision_statement, _MAX_COMPLIANCE_DECISION_CHARS
        ),
    }
    user_prompt = (
        "Проанализируй входной JSON и верни только JSON с полями attacks и rationale "
        "(и опционально rejected), как в системной инструкции.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    raw = llm_client.complete(system_prompt=system, user_prompt=user_prompt)
    attacks, rationale = _parse_llm_plan(raw, allowed=available)
    return AttackPlan(attacks=attacks, rationale=rationale, planner="agent")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_attacks(
    synopsis: dict[str, object],
    threat_model_markdown: str,
    datasets_dir: str | Path,
    *,
    llm_client: "OpenAIClient | None" = None,
    mode: str = "agent",
    prompts_dir: str | Path | None = None,
    compliance_decision_statement: str | None = None,
) -> AttackPlan:
    """Pick a minimal subset of `datasets/*.parquet` for BOART.

    ``mode='agent'`` (default) calls the LLM and falls back to heuristic on any failure.
    ``mode='heuristic'`` skips the LLM entirely.

    ``compliance_decision_statement`` — optional formal policy rollup (``decision_statement``); passed only
    as extra context to the LLM agent (never merged into threat modeling).
    """
    available = _list_available_datasets(datasets_dir)
    if not available:
        raise ValueError("В каталоге datasets нет файлов *.parquet.")

    pdir = Path(prompts_dir) if prompts_dir else Path(__file__).resolve().parents[1] / "prompts"

    if mode == "agent" and llm_client is not None:
        try:
            return _plan_with_llm(
                synopsis=dict(synopsis),
                threat_model_markdown=threat_model_markdown,
                available=available,
                llm_client=llm_client,
                prompts_dir=pdir,
                compliance_decision_statement=compliance_decision_statement,
            )
        except Exception as exc:
            logger.warning("LLM-планировщик атак недоступен (%s) — fallback на эвристику.", exc)

    return select_attacks_from_context(synopsis, threat_model_markdown, datasets_dir)
