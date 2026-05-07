"""Layered compliance checks for agentic flows (формальный подмодуль внутри S2).

Three-stage pipeline (no LLM is required; semantic stage is opt-in):

  static  → 4 requirement checks via regex + structural graph rules.
  semantic→ single focused LLM call returning strict per-requirement JSON (optional).
  merge   → worst-status wins per requirement; then **criticality cap**
            (optional/advisory never end as FAIL).

Per-requirement tier in JSON: ``criticality`` — ``optional`` | ``advisory`` | ``blocking``.
Only ``blocking`` (secrets + access metadata in prompts) drives ``overall: FAIL``.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

REQUIREMENT_LABELS: dict[str, str] = {
    "REQ-SANITIZATION":   "Очистка входных данных перед агентом",
    "REQ-LEAST-PRIVILEGE":"Принцип минимальных привилегий",
    "REQ-DATA-MIN":       "Отсутствие секретов в системном промпте",
    "REQ-NO-META-IN-CTX": "Метаданные доступа не передаются в контекст агента",
}

# Policy tier: drives overall FAIL (only ``blocking`` may fail the run) and semantic caps.
REQUIREMENT_CRITICALITY: dict[str, str] = {
    "REQ-SANITIZATION":    "optional",   # рекомендация, не блокирует согласование
    "REQ-LEAST-PRIVILEGE": "advisory",  # рекомендация; нарушение не FAIL-ит прогон
    "REQ-DATA-MIN":        "blocking",
    "REQ-NO-META-IN-CTX":  "blocking",
}

_STATUS_RANK = {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}


@dataclass(slots=True)
class ComplianceResult:
    requirement_id: str
    requirement_short: str
    status: str           # PASS | WARN | FAIL | SKIP
    details: str
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_short": self.requirement_short,
            "criticality": REQUIREMENT_CRITICALITY.get(self.requirement_id, "blocking"),
            "status": self.status,
            "details": self.details,
            "evidence": self.evidence,
        }


def _result(req_id: str, status: str, details: str, evidence: list | None = None) -> ComplianceResult:
    return ComplianceResult(req_id, REQUIREMENT_LABELS[req_id], status, details, evidence or [])


# ---------------------------------------------------------------------------
# Secret patterns (bundled rules + structural fallback)
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).resolve().parents[1]

_FALLBACK_SECRETS: list[str] = [
    # Generic key/token assignment (any operator)
    r"(?:api[_\-]?key|secret[_\-]?key|access[_\-]?key|auth[_\-]?token|client[_\-]?secret)\s*[:=]\s*\S{8,}",
    r"(?:password|passwd|passphrase)\s*[:=]\s*\S{4,}",
    r"(?<![a-z])bearer\s+[A-Za-z0-9\-_.]{10,}",
    # Russian natural-language credential keywords + value
    r"(?:ключик?|ключ[её]м?|токен[а-я]*|пароль[а-я]*|секрет[а-я]*)\s*[:：]\s*(\S{6,})",
    # Token-like strings after any colon (catches "use key: sk-...")
    r":\s+([a-zA-Z0-9][a-zA-Z0-9\-_.]{8,}[a-zA-Z0-9])",
    # Well-known prefixes Gitleaks misses (with hyphens)
    r"\b(sk-[a-zA-Z0-9\-_.]{10,})\b",
    r"\b(xox[bpaso]-[a-zA-Z0-9\-]{6,})\b",
    r"\b(ghp_[a-zA-Z0-9]{20,})\b",
    r"\b(glpat-[a-zA-Z0-9\-_]{10,})\b",
    r"\b(npm_[a-zA-Z0-9]{16,})\b",
    r"\b(AKIA[A-Z0-9]{16})\b",
    # Connection strings
    r"[a-z][a-z0-9+\-.]{1,15}://[^\s\"'<>]{6,}@[^\s\"'<>]+",
    r"jdbc:[a-z]+://",
    r"\bDSN\s*=",
    # Long opaque token after =
    r"=\s*['\"]?[A-Za-z0-9+/]{40,}={0,2}['\"]?",
]

_ACCESS_META: list[str] = [
    r"\b(?:role|group|permission|access[_\-]?level|privilege)\s*[:=]\s*\S",
    r"\b(?:limit|quota|rate[_\-]?limit|max[_\-]?requests?)\s*[:=]\s*\d",
    r"\b(?:scope|audience)\s*[:=]\s*\S",
    r"\b(?:is[_\-]?admin|superuser|root|elevated)\s*[:=]\s*(?:true|1|yes|on)\b",
    r"(?:auth|authz|authorization)\s*:\s*\{",
]


def _load_external_secret_patterns() -> list[re.Pattern[str]]:
    """Regex из ``secret-patterns/gitleaks.toml`` и ``secret-patterns/rules-stable.yml`` (если есть)."""
    patterns: list[re.Pattern[str]] = []
    rules_dir = _PKG_ROOT / "secret-patterns"
    gl_path = rules_dir / "gitleaks.toml"
    rs_path = rules_dir / "rules-stable.yml"

    if gl_path.is_file():
        try:
            import tomllib
        except ImportError:
            tomllib = None  # type: ignore[assignment]
        if tomllib:
            try:
                data = tomllib.loads(gl_path.read_text(encoding="utf-8"))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for rule in data.get("rules", []):
                        rx = rule.get("regex") or rule.get("pattern")
                        if rx:
                            try:
                                patterns.append(re.compile(rx))
                            except re.error:
                                pass
            except Exception:
                pass

    if rs_path.is_file():
        try:
            import yaml  # type: ignore[import-untyped]
            data = yaml.safe_load(rs_path.read_text(encoding="utf-8"))
            for entry in data.get("patterns", []):
                pat = entry.get("pattern", {})
                if pat.get("confidence") == "high" and pat.get("regex"):
                    try:
                        patterns.append(re.compile(pat["regex"]))
                    except re.error:
                        pass
        except (ImportError, Exception):
            pass

    return patterns


_SECRET_PATTERNS: list[re.Pattern[str]] = (
    [re.compile(p, re.IGNORECASE) for p in _FALLBACK_SECRETS]
    + _load_external_secret_patterns()
)
_ACCESS_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _ACCESS_META]


# ---------------------------------------------------------------------------
# Sanitizer classification
# ---------------------------------------------------------------------------

_SANITIZER_TYPES = frozenset({
    "GuardrailValidator", "ParserComponent", "FilterData", "Regex",
    "SanitizerComponent", "TextSanitizer", "Rephraser", "Summarizer",
})
_SANITIZER_KEYWORDS = ("guardrail", "sanitiz", "filter", "clean", "rephras", "summar", "parser")

_MULTI_TOOL_THRESHOLD = 2
_HETEROGENEOUS_TYPE_THRESHOLD = 2

# Heuristic: разные «системы» интеграции для LEAST-PRIVILEGE (структурный сигнал советного уровня).
_MUTATION_HINT = re.compile(
    r"(?:write|delete|update|create|post|put|patch|insert|sql|commit|booking|брон|запис|удал|обнов|созда|оплат|transfer|send|upload)",
    re.IGNORECASE,
)


def _tool_integration_key(source_type: str, source_name: str) -> str:
    st = (source_type or "").strip()
    sn = (source_name or "").strip().lower()
    # Разные MCP-серверы / инструменты — разные системы.
    if "mcp" in st.lower():
        return f"mcp::{sn}"
    return f"type::{st}"


def _tool_edges_mutating_hint(edges: list[dict], nmap: dict[str, dict]) -> bool:
    for e in edges:
        sid = e.get("source_id")
        node = nmap.get(sid, {})
        blob = f"{e.get('source_name', '')} {e.get('source_type', '')} {node.get('name', '')} {node.get('type', '')}"
        if _MUTATION_HINT.search(blob):
            return True
    return False


def _is_sanitizer(node: dict) -> bool:
    t = (node.get("type") or "").lower()
    n = (node.get("name") or "").lower()
    return (
        node.get("type") in _SANITIZER_TYPES
        or "guardrail" in (node.get("risk_flags") or [])
        or any(k in t or k in n for k in _SANITIZER_KEYWORDS)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pattern_hits(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    """De-duplicated short excerpts around each first match."""
    seen: set[str] = set()
    out: list[str] = []
    for pat in patterns:
        m = pat.search(text)
        if not m:
            continue
        a, b = max(0, m.start() - 8), min(len(text), m.end() + 16)
        excerpt = f"…{text[a:b].strip()}…"
        if excerpt not in seen:
            seen.add(excerpt)
            out.append(excerpt)
    return out


def _prompt_fields(synopsis: dict) -> Iterator[tuple[str, str, str]]:
    """Yield (node_name, field_name, text) for each system-prompt-like text."""
    for sp in synopsis.get("system_prompts", []):
        text = (sp.get("text") or "").strip()
        if text:
            yield sp.get("node_name", ""), sp.get("field", ""), text
    for np in synopsis.get("author_parameters_by_node", []):
        for key, val in (np.get("parameters") or {}).items():
            if key.lower() not in {"system_prompt", "system_message", "instructions", "template", "prompt"}:
                continue
            raw = val.get("value", "") if isinstance(val, dict) else str(val or "")
            if isinstance(raw, str) and raw.strip():
                yield np.get("node_name", ""), key, raw.strip()


# ---------------------------------------------------------------------------
# Static rule checks
# ---------------------------------------------------------------------------

def check_sanitization(synopsis: dict) -> ComplianceResult:
    """Agents with tools + user input must have a guardrail wired to them."""
    nodes = synopsis.get("nodes", [])
    nmap = {n["id"]: n for n in nodes}
    entrypoints = set(synopsis.get("entrypoints", []))
    edges = synopsis.get("edges", [])
    tool_edges = synopsis.get("tool_edges", [])

    at_risk: list[dict] = []
    for agent in (n for n in nodes if n.get("role") == "agent"):
        agent_tools = [e for e in tool_edges if e["target_id"] == agent["id"]]
        if not agent_tools:
            continue
        upstream = {e["source"] for e in edges if e["target"] == agent["id"]}
        if not (upstream & entrypoints):
            continue
        if any(_is_sanitizer(nmap[s]) for s in upstream if s in nmap):
            continue
        at_risk.append({
            "agent": agent["name"],
            "tools": [e["source_name"] for e in agent_tools],
            "entrypoints_feeding": [nmap[s]["name"] for s in upstream if s in entrypoints and s in nmap],
        })

    if not at_risk:
        return _result("REQ-SANITIZATION", "PASS",
                       "Все агенты с инструментами имеют подключённый компонент очистки/гарда на входе.")
    return _result("REQ-SANITIZATION", "WARN",
                   "Агент(ы) с внешними инструментами получают пользовательский ввод без подключённого "
                   "компонента очистки. Рекомендуется добавить GuardrailValidator на вход.",
                   evidence=at_risk)


def check_least_privilege(synopsis: dict) -> ComplianceResult:
    """Advisory tier: never FAIL; only structural hints.

    Приоритетное предупреждение — агрегация операций с изменением состояния между
    **разными** интеграционными системами (по типу/имени tool). Остальные сигналы —
    обычная рекомендация по минимальным привилегиям.
    """
    nodes = synopsis.get("nodes", [])
    nmap = {n["id"]: n for n in nodes}
    tool_edges = synopsis.get("tool_edges", [])
    elevated: list[dict] = []
    advisory: list[dict] = []

    for agent in (n for n in nodes if n.get("role") == "agent"):
        connected = [e for e in tool_edges if e["target_id"] == agent["id"]]
        if not connected:
            continue
        tool_names = [e["source_name"] for e in connected]
        tool_types = {nmap[e["source_id"]]["type"] for e in connected if e.get("source_id") in nmap}
        integ_keys = {
            _tool_integration_key(str(e.get("source_type") or ""), str(e.get("source_name") or ""))
            for e in connected
        }
        mut = _tool_edges_mutating_hint(connected, nmap)
        hints: list[str] = []
        if len(connected) >= _MULTI_TOOL_THRESHOLD:
            hints.append(f"{len(connected)} инструментов на одном агенте: {', '.join(tool_names)}")
        if len(tool_types) >= _HETEROGENEOUS_TYPE_THRESHOLD:
            hints.append(f"Разнородные типы инструментов: {', '.join(sorted(tool_types))}")
        base = {
            "agent": agent["name"],
            "hints": hints,
            "integration_surfaces": sorted(integ_keys),
        }
        if len(integ_keys) >= 2 and mut:
            elevated.append({**base, "note": "Разные системы + признаки мутации состояния."})
        elif hints:
            advisory.append(base)

    shared: dict[str, list[str]] = {}
    for e in tool_edges:
        shared.setdefault(str(e["source_name"]), []).append(str(e["target_name"]))
    for tool_name, agent_names in shared.items():
        if len(agent_names) >= 2:
            advisory.append({
                "tool": tool_name,
                "hints": [f"Инструмент «{tool_name}» подключён к нескольким агентам: {', '.join(agent_names)}"],
                "integration_surfaces": [],
            })

    if elevated:
        return _result(
            "REQ-LEAST-PRIVILEGE", "WARN",
            "Приоритетное замечание (advisory): один агент объединяет операции, затрагивающие состояние "
            "в **разных** интеграционных системах. Рекомендуется разнести по отдельным агентам/модулям с "
            "минимальными привилегиями.",
            evidence=elevated,
        )
    if advisory:
        return _result(
            "REQ-LEAST-PRIVILEGE", "WARN",
            "Структурные признаки, по которым полезно проверить разделение компонентов по границам доступа.",
            evidence=advisory,
        )
    return _result(
        "REQ-LEAST-PRIVILEGE", "PASS",
        "Структурных признаков агрегации привилегий между разными системами не выявлено.",
    )


def check_secrets_in_system_prompt(synopsis: dict) -> ComplianceResult:
    violations: list[dict] = []
    for node, field_name, text in _prompt_fields(synopsis):
        hits = _pattern_hits(text, _SECRET_PATTERNS)
        if hits:
            violations.append({"node": node, "field": field_name, "matches": hits[:5]})
    if not violations:
        return _result("REQ-DATA-MIN", "PASS",
                       f"Секреты не обнаружены ({len(_SECRET_PATTERNS)} активных паттернов).")
    return _result("REQ-DATA-MIN", "FAIL",
                   "В системных промптах обнаружены паттерны секретов или строк подключения.",
                   evidence=violations)


def check_access_meta_in_prompt(synopsis: dict) -> ComplianceResult:
    violations: list[dict] = []
    for node, field_name, text in _prompt_fields(synopsis):
        hits = _pattern_hits(text, _ACCESS_PATTERNS)
        if hits:
            violations.append({"node": node, "field": field_name, "matches": hits})
    if not violations:
        return _result("REQ-NO-META-IN-CTX", "PASS",
                       "Метаданные ролей/разрешений/лимитов в системных промптах не обнаружены.")
    return _result("REQ-NO-META-IN-CTX", "FAIL",
                   "В системных промптах обнаружены метаданные доступа (blocking).",
                   evidence=violations)


# ---------------------------------------------------------------------------
# Semantic stage (single LLM call → strict JSON per requirement)
# ---------------------------------------------------------------------------

def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:]).rsplit("```", 1)[0]
    return s.strip()


def _semantic_payload(synopsis: dict, static_results: list[ComplianceResult]) -> str:
    payload = {
        "system_prompts": synopsis.get("system_prompts", []),
        "tool_edges": [{"source": e["source_name"], "source_type": e.get("source_type"),
                        "target": e["target_name"]}
                       for e in synopsis.get("tool_edges", [])],
        "entrypoints": synopsis.get("entrypoints", []),
        "controls": synopsis.get("controls", []),
        "agents": [{"id": n["id"], "name": n["name"], "type": n["type"]}
                   for n in synopsis.get("nodes", []) if n.get("role") == "agent"],
        "static_findings": [
            {"requirement_id": r.requirement_id, "status": r.status, "details": r.details}
            for r in static_results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _run_semantic(synopsis: dict, llm_client: Any, static: list[ComplianceResult]) -> dict[str, ComplianceResult]:
    prompt_path = _PKG_ROOT / "prompts" / "compliance_semantic_system_ru.txt"
    if not prompt_path.is_file():
        return {}
    try:
        system_prompt = prompt_path.read_text(encoding="utf-8")
        raw = llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=(
                "Каждый элемент `system_prompts` содержит полный `text` инструкции — "
                "оцени смысл целиком (плейсхолдеры в скобках, перефразирование, любые имена полей).\n\n"
                "Входные данные:\n"
                + _semantic_payload(synopsis, static)
            ),
        )
        data = json.loads(_strip_json_fence(raw))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, ComplianceResult] = {}
    for req_id in REQUIREMENT_LABELS:
        entry = data.get(req_id)
        if not isinstance(entry, dict):
            continue
        status = entry.get("status", "SKIP")
        if status not in _STATUS_RANK:
            status = "SKIP"
        ev = entry.get("evidence") or []
        out[req_id] = _result(
            req_id, status,
            str(entry.get("rationale", "")).strip() or "—",
            evidence=[e for e in ev if isinstance(e, dict)],
        )
    return out


def _merge_results(static: ComplianceResult, semantic: ComplianceResult | None) -> ComplianceResult:
    """Worst-status wins; details and evidence concatenated."""
    if semantic is None:
        return static
    final_status = max(static.status, semantic.status, key=lambda s: _STATUS_RANK.get(s, 0))
    parts = []
    if static.details:
        parts.append(f"[статика] {static.details}")
    if semantic.details and semantic.details != "—":
        parts.append(f"[семантика] {semantic.details}")
    return ComplianceResult(
        requirement_id=static.requirement_id,
        requirement_short=static.requirement_short,
        status=final_status,
        details=" ".join(parts) or static.details,
        evidence=[*static.evidence, *({"source": "semantic", **e} for e in semantic.evidence)],
    )


_CRITICALITY_FAIL_CAP_NOTE = (
    "Политика уровня критичности: для этого пункта максимальный статус в отчёте — WARN "
    "(optional/advisory)."
)


def _enforce_criticality_cap(result: ComplianceResult) -> ComplianceResult:
    """Optional/advisory tiers never contribute FAIL after merge."""
    tier = REQUIREMENT_CRITICALITY.get(result.requirement_id, "blocking")
    if tier in {"optional", "advisory"} and result.status == "FAIL":
        return ComplianceResult(
            result.requirement_id,
            result.requirement_short,
            "WARN",
            f"{result.details} [{_CRITICALITY_FAIL_CAP_NOTE}]",
            result.evidence,
        )
    return result


# ---------------------------------------------------------------------------
# Decision statement
# ---------------------------------------------------------------------------

_DECISION_PREFACE = (
    "**Оркестратор MLSecOps (агрегат после отдельного MAESTRO и отдельной проверки REQ-*):** "
    "качественный отчёт MAESTRO выполняется **независимо** от этого блока; здесь — только результат "
    "формальных требований. **Вывод для прода по политике MLSecOps:** при отсутствии **FAIL** по "
    "`REQ-DATA-MIN` и `REQ-NO-META-IN-CTX` организационно допускается выпуск **с привлечением "
    "эксперта MLSecOps** (приёмка, приоритеты, учёт технического долга из опциональных/советных REQ). "
    "Любой **FAIL** по `REQ-DATA-MIN` или `REQ-NO-META-IN-CTX` **блокирует** выпуск до устранения "
    "или явного исключения сверху.\n\n"
)


def _decision_statement(results: list[ComplianceResult]) -> str:
    fails = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]
    passed_short = [r.requirement_short for r in results if r.status == "PASS"]
    advisory_warns = [
        r for r in warns
        if REQUIREMENT_CRITICALITY.get(r.requirement_id) in {"optional", "advisory"}
    ]
    blocking_warns = [
        r for r in warns
        if REQUIREMENT_CRITICALITY.get(r.requirement_id) == "blocking"
    ]
    blocking_ids_hard = frozenset({"REQ-DATA-MIN", "REQ-NO-META-IN-CTX"})
    blocking_fails = [f for f in fails if f.requirement_id in blocking_ids_hard]

    if not fails and not warns:
        body = (
            "По проверяемым требованиям **нет статусов FAIL/WARN**. "
            "В частности **нет FAIL** по **`REQ-DATA-MIN`** и **`REQ-NO-META-IN-CTX`**.\n\n"
            "**Решение оркестратора (агрегат политик, при отдельно зафиксированном MAESTRO):** "
            "**выпуск в прод организационно допустим** при **обязательном привлечении эксперта MLSecOps** "
            "в контуре приёмки и дальнейшего надзора (качественные риски из MAESTRO остаются вне этого gate).\n\n"
            f"Требования со статусом PASS: {', '.join(passed_short)}."
        )
        return _DECISION_PREFACE + body

    debts = [f"  — [{r.requirement_id}] {r.requirement_short}: {r.details}" for r in warns + fails]

    if blocking_fails:
        ids_bf = ", ".join(f.requirement_id for f in blocking_fails)
        head = (
            f"**Стоп выпуску:** статус **FAIL** по блокирующим для прома требованиям: **`{ids_bf}`** "
            "(секреты в системном промпте или метаданные доступа в контексте агента). "
            "**В эксплуатацию не выпускать** без устранения или управленческого исключения."
        )
    elif fails:
        ids_nf = ", ".join(f.requirement_id for f in fails)
        head = (
            f"**Стоп выпуску:** есть **FAIL** по требованиям `{ids_nf}`. По текущей политике критичности "
            "все блокирующие для overall FAIL относятся к секретам/метаданным; проработать до статусов "
            "до деплоя."
        )
    elif blocking_warns:
        head = (
            "**WARN по блокирующим категориям** (`REQ-DATA-MIN`, `REQ-NO-META-IN-CTX`): "
            "**FAIL по ним отсутствует**, автоматической «полной блокировке» статусами нет — но **до выпуска в прод "
            "нужен разбор эксперта MLSecOps** до приемлемого результата (gate не является «полностью зелёным» без разбора)."
        )
    else:
        head = (
            "**По `REQ-DATA-MIN` и `REQ-NO-META-IN-CTX` нет FAIL**; замечания касаются только опциональных/советных "
            "(санитизация, архитектурное разделение привилегий). "
            "**Выпуск в прод:** допускается **при участии эксперта MLSecOps**; учесть технический долг ниже под надзором."
        )

    parts: list[str] = [_DECISION_PREFACE + head.strip(), "", "Детали по статусам WARN/FAIL:", *debts]
    if passed_short:
        parts += ["", f"PASS: {', '.join(passed_short)}."]
    if blocking_fails:
        parts.append(
            "\n**Ключевое правило:** любой FAIL по **`REQ-DATA-MIN`** или **`REQ-NO-META-IN-CTX`** — "
            "**жёсткий стоп для самостоятельного прод-релиза** до правок либо явного управленческого решения.",
        )
    elif not fails and advisory_warns:
        parts.append(
            "\nПримечание оркестратора: советные REQ (`REQ-SANITIZATION`, `REQ-LEAST-PRIVILEGE`) **не равняются** "
            "блокирующим паре секретов/метаданных по политике MLSecOps, но должны находиться под вниманием эксперта.",
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_compliance_checks(synopsis: dict, llm_client: Any | None = None) -> dict[str, Any]:
    """Layered compliance: static → optional semantic → merge per requirement.

    If ``llm_client`` is provided, a single focused LLM call extends the static baseline.
    On any failure the semantic stage is silently skipped (static result is final).
    """
    static = [
        check_sanitization(synopsis),
        check_least_privilege(synopsis),
        check_secrets_in_system_prompt(synopsis),
        check_access_meta_in_prompt(synopsis),
    ]
    semantic_map = _run_semantic(synopsis, llm_client, static) if llm_client is not None else {}
    merged_raw = [_merge_results(s, semantic_map.get(s.requirement_id)) for s in static]
    merged = [_enforce_criticality_cap(r) for r in merged_raw]

    fails = [r for r in merged if r.status == "FAIL"]
    warns = [r for r in merged if r.status == "WARN"]
    return {
        "requirements_checked": len(merged),
        "violations": len(fails),
        "warnings": len(warns),
        "overall": "FAIL" if fails else ("WARN" if warns else "PASS"),
        "secret_patterns_loaded": len(_SECRET_PATTERNS),
        "semantic_analysis": bool(semantic_map),
        "decision_statement": _decision_statement(merged),
        "results": [r.to_dict() for r in merged],
        "static_only_results": [r.to_dict() for r in static],
        "semantic_results": [r.to_dict() for r in semantic_map.values()],
    }
