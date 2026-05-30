"""Итоговые отчёты и текстовые сводки для CLI и артефактов."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

_MSK = ZoneInfo("Europe/Moscow")


SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SEVERITY_RU = {
    "LOW": "НИЗКИЙ",
    "MEDIUM": "СРЕДНИЙ",
    "HIGH": "ВЫСОКИЙ",
    "CRITICAL": "КРИТИЧЕСКИЙ",
}

THREAT_LABEL_RU: dict[str, str] = {
    "system_prompt_leakage": "Утечка системного промпта / конфиденциальных инструкций",
    "harmbench_text": "Вредоносный или небезопасный текстовый вывод (jailbreak / harmful)",
}

# Поля экспорта флоу Langflow из корня JSON (препрод / UI экспорт — см. экспорт вида Windchaser.json).
FLOW_EXPORT_META_KEYS_ORDERED: tuple[str, ...] = (
    "name",
    "id",
    "description",
    "endpoint_name",
    "tags",
    "last_tested_version",
    "is_component",
    "locked",
)

_REQ_SPECS: dict[str, tuple[str, str]] = {
    "REQ-DATA-MIN": ("Отсутствие секретов в системном промпте", "blocking"),
    "REQ-NO-META-IN-CTX": ("Метаданные доступа не передаются в контекст агента", "blocking"),
    "REQ-SANITIZATION": ("Очистка входных данных перед агентом", "optional"),
    "REQ-LEAST-PRIVILEGE": ("Принцип минимальных привилегий", "advisory"),
    "REQ-HUMAN-REVIEW": ("Сигналы для ручной проверки", "optional"),
}


def enrich_synopsis(synopsis: dict[str, Any]) -> dict[str, Any]:
    """Добавить ``assets`` и поля summary для совместимости со старым pipeline."""
    out = dict(synopsis)
    assets: list[str] = []
    for n in synopsis.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        name = str(n.get("name") or n.get("id") or "?")
        role = str(n.get("role") or n.get("type") or "node")
        assets.append(f"{name}::{role}")
    out["assets"] = sorted(set(assets))
    sm = dict(out.get("summary") or {})
    sm.setdefault("node_count", sm.get("nodes", len(out.get("nodes") or [])))
    sm.setdefault("edge_count", sm.get("edges", len(out.get("edges") or [])))
    sm.setdefault("entrypoint_count", sm.get("entrypoints", len(out.get("entrypoints") or [])))
    sm.setdefault("control_count", sm.get("controls", len(out.get("controls") or [])))
    out["summary"] = sm
    return out


def _req_row(
    requirement_id: str,
    status: str,
    details: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    short, criticality = _REQ_SPECS[requirement_id]
    return {
        "requirement_id": requirement_id,
        "requirement_short": short,
        "criticality": criticality,
        "status": status,
        "details": details,
        "evidence": evidence,
    }


def _normalize_validator_evidence(evidence: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    out: list[dict[str, Any]] = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        out.append({
            "source": ev.get("source", "semantic"),
            "node": ev.get("node", "?"),
            "excerpt": ev.get("excerpt", ""),
            "reason": ev.get("reason", ""),
        })
    return out


def _infer_no_meta_status(compliance: Mapping[str, Any]) -> str:
    dm = compliance.get("REQ-DATA-MIN")
    if not isinstance(dm, dict):
        return "PASS"
    for ev in dm.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        blob = f"{ev.get('reason', '')} {ev.get('excerpt', '')}".lower()
        if "access_metadata" in blob or "метадан" in blob or "рол" in blob and "доступ" in blob:
            return "FAIL"
    return "PASS"


def _infer_sanitization_status(synopsis: Mapping[str, Any]) -> str:
    if synopsis.get("tool_edges") and synopsis.get("controls") and synopsis.get("entrypoints"):
        return "WARN"
    return "PASS"


def _sanitization_details(synopsis: Mapping[str, Any]) -> str:
    if _infer_sanitization_status(synopsis) == "WARN":
        return (
            "Агент(ы) с внешними инструментами и точками входа; в графе есть controls, "
            "но по synopsis нет гарантии подключения Guardrail на путь ChatInput → Agent. "
            "Рекомендуется проверить wiring guardrail."
        )
    return "Санитизация на пути к агенту не требует эскалации по статике synopsis."


def compliance_from_validator(
    compliance: Mapping[str, Any],
    *,
    gate_verdict_status: str,
    gate_verdict_comment: str,
    synopsis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Формат ``compliance`` как в ``final_report.json`` (старый mlsecops-pipeline)."""
    results: list[dict[str, Any]] = []
    for rid in ("REQ-DATA-MIN", "REQ-LEAST-PRIVILEGE", "REQ-HUMAN-REVIEW"):
        entry = compliance.get(rid) if isinstance(compliance.get(rid), dict) else {}
        results.append(
            _req_row(
                rid,
                str(entry.get("status", "PASS")),
                str(entry.get("rationale") or ""),
                _normalize_validator_evidence(entry.get("evidence")),
            )
        )
    syn = synopsis or {}
    results.append(
        _req_row(
            "REQ-NO-META-IN-CTX",
            _infer_no_meta_status(compliance),
            "Метаданные доступа в контексте LLM (эвристика validator).",
            [],
        )
    )
    results.append(
        _req_row(
            "REQ-SANITIZATION",
            _infer_sanitization_status(syn),
            _sanitization_details(syn),
            [],
        )
    )

    violations = sum(1 for r in results if r.get("status") == "FAIL")
    warnings = sum(1 for r in results if r.get("status") == "WARN")
    overall = "FAIL" if violations else ("WARN" if warnings else "PASS")

    pipeline: dict[str, Any] = {
        "requirements_checked": len(results),
        "violations": violations,
        "warnings": warnings,
        "overall": overall,
        "secret_patterns_loaded": 0,
        "semantic_analysis": True,
        "results": results,
        "static_only_results": [dict(r) for r in results],
        "semantic_results": results,
    }
    _, _, rec_md, _ = _resolve_security_gate_messages(pipeline)
    pipeline["decision_statement"] = _build_decision_statement(
        pipeline, gate_verdict_status, gate_verdict_comment, rec_md
    )
    return pipeline


def _build_decision_statement(
    pipeline: Mapping[str, Any],
    gate_status: str,
    gate_comment: str,
    orchestrator_rec_md: str,
) -> str:
    lines = [
        "**Оркестратор MLSecOps:** MAESTRO и REQ-* — отдельные слои; "
        "ниже — агрегат политик в формате legacy pipeline.",
        "",
        f"**Security Gate (validator):** {gate_status} — {gate_comment}",
        "",
        orchestrator_rec_md,
        "",
        "Детали по статусам WARN/FAIL:",
    ]
    for r in pipeline.get("results") or []:
        if r.get("status") in ("WARN", "FAIL"):
            lines.append(
                f"  — [{r.get('requirement_id')}] {r.get('requirement_short')}: "
                f"[{r.get('status')}] {(r.get('details') or '')[:300]}"
            )
    lines.append("")
    lines.append(
        "PASS: см. полный список в ``compliance.results``; "
        "BOART — в ``adversarial_testing``."
    )
    return "\n".join(lines)


def extract_flow_export_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Подмножество корневых полей экспорта флоу (имя, id, endpoint, теги, …).

    Если ``raw`` не dict или полей нет — возвращается пустой dict.
    """
    if raw is None or not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in FLOW_EXPORT_META_KEYS_ORDERED:
        if key not in raw:
            continue
        val = raw[key]
        out[key] = val
    return out


def format_flow_export_metadata_markdown(meta: Mapping[str, Any]) -> str:
    """Таблица параметров экспортируемого флоу для вставки в отчёт."""
    if not meta:
        return (
            "*В корне входного JSON не найдены поля экспорта флоу* (`name`, `id`, "
            "`endpoint_name`, …). *Для режима file используйте полный экспорт Langflow из препрода "
            "(корень JSON с блоком `data` и метаданными `name`, `id`, …).*\n"
        )
    rows = ["| Поле | Значение |", "|------|-----------|"]

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
            return text.replace("|", "\\|")
        text = str(value).replace("|", "\\|")
        return text or "—"

    for key in FLOW_EXPORT_META_KEYS_ORDERED:
        if key not in meta:
            continue
        rows.append(f"| `{key}` | {_fmt(meta[key])} |")
    return "\n".join(rows) + "\n"


def assessment_timestamp_msk(when: datetime | None = None) -> str:
    """Отметка времени для документов: Москва (= UTC+3, без переходов на летнее)."""
    if when is None:
        dt = datetime.now(_MSK)
    elif when.tzinfo is None:
        dt = when.replace(tzinfo=_MSK)
    else:
        dt = when.astimezone(_MSK)
    return dt.strftime("%Y-%m-%d %H:%M:%S MSK (UTC+3)")


def _blocking_pair_status(
    compliance_report: Mapping[str, Any],
) -> tuple[dict[str, str | None], bool, bool]:
    """STATUSES REQ-DATA-MIN / REQ-NO-META-IN-CTX; has FAIL; WARN без FAIL ни по одной."""
    statuses: dict[str, str | None] = {"REQ-DATA-MIN": None, "REQ-NO-META-IN-CTX": None}
    for r in compliance_report.get("results") or []:
        rid = r.get("requirement_id")
        if rid in statuses:
            statuses[str(rid)] = str(r.get("status") or "") or None
    bf = statuses["REQ-DATA-MIN"] == "FAIL" or statuses["REQ-NO-META-IN-CTX"] == "FAIL"
    bw = (
        statuses["REQ-DATA-MIN"] == "WARN" or statuses["REQ-NO-META-IN-CTX"] == "WARN"
    ) and not bf
    return statuses, bf, bw


def _resolve_security_gate_messages(
    compliance_report: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """(строка статусов пары для MD, метка gate, рекомендация MD, та же строка без ** и т.п.)."""
    states, bf, bw = _blocking_pair_status(compliance_report)
    st_line_md = ", ".join(
        f"`{rid}`→`{states[rid] or '—'}`" for rid in ("REQ-DATA-MIN", "REQ-NO-META-IN-CTX")
    )
    if bf:
        gate_plain = "БЛОКИРОВАТЬ ДО ФИКСА"
        rec_plain = (
            "Есть FAIL по блокирующей паре (секреты в промпте / метаданные доступа в контексте). "
            "Самостоятельный прод без устранения или управленческого решения недопустим."
        )
        rec_md = (
            "Есть **FAIL** по блокирующей паре (секреты в промпте / метаданные доступа в контексте). "
            "Самостоятельный прод-выпуск **запрещён** без устранения или управленческого решения поверх этого отчёта."
        )
    elif bw:
        gate_plain = "УСЛОВНЫЙ GATE / ЭСКАЛАЦИЯ"
        rec_plain = (
            "FAIL по паре нет, но есть WARN по REQ-DATA-MIN или REQ-NO-META-IN-CTX. Нужен разбор эксперта MLSecOps."
        )
        rec_md = (
            "**FAIL по паре нет**, но есть **WARN** по `REQ-DATA-MIN` и/или `REQ-NO-META-IN-CTX`. "
            "До прома нужен разбор эксперта MLSecOps; gate **не является полностью зелёным** без этого шага."
        )
    elif str(compliance_report.get("overall", "")) == "WARN":
        gate_plain = "ДОПУСТИМ К ПРОДУ ПРИ УЧАСТИИ ЭКСПЕРТА MLSecOps"
        rec_plain = (
            "Пара REQ без блокирующих FAIL; есть только советные WARN. В прод только с участием эксперта MLSecOps."
        )
        rec_md = (
            "**Пара REQ-DATA-MIN / REQ-NO-META-IN-CTX** без FAIL. Предупреждения только по советным "
            "**REQ-SANITIZATION**, **REQ-LEAST-PRIVILEGE**. В прод **с официальной зоной внимания эксперта MLSecOps**: "
            "учесть технический долг из части 2 и качественные выводы MAESTRO из части 1."
        )
    else:
        gate_plain = "ДОПУСТИМ К ПРОДУ ПРИ УЧАСТИИ ЭКСПЕРТА MLSecOps"
        rec_plain = (
            "overall PASS формально не заменяет инженерную приёмку: выпуск только с экспертом MLSecOps по регламенту."
        )
        rec_md = (
            "**Пара REQ без FAIL.** Формальный overall PASS **не заменяет** инженерную приёмку: выпуск возможен только с "
            "**привлечением эксперта MLSecOps** под регламент организации и с учётом MAESTRO (часть 1)."
        )
    return st_line_md, gate_plain, rec_md, rec_plain


def format_maestro_console_brief(threat_md: str, *, max_chars: int = 900) -> str:
    """Сжатый текст из отчёта MAESTRO для консоли (ноутбук): без заголовков и таблиц."""
    if not threat_md or not threat_md.strip():
        return "— модель угроз не построена (нет ключа OpenAI или блок не выполнялся)."
    chunks: list[str] = []
    for line in threat_md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        if s.startswith("---"):
            continue
        if s.startswith("|"):
            continue
        if re.match(r"^[*_-]{3,}\s*$", s):
            continue
        chunks.append(s)
    blob = " ".join(w for w in chunks if w)
    blob = re.sub(r"\s+", " ", blob).strip()
    if len(blob) > max_chars:
        blob = blob[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return blob


def format_compliance_console_brief(compliance_report: Mapping[str, Any] | None) -> str:
    """Краткая сводка REQ-* для консоли (без полного decision_statement)."""
    if compliance_report is None:
        return "Compliance не выполнялся."
    sts, _, _ = _blocking_pair_status(compliance_report)
    lines = [
        f"Итог: {compliance_report.get('overall')} · блокирующих FAIL: "
        f"{compliance_report.get('violations')} · предупреждений (агрегат): "
        f"{compliance_report.get('warnings')} · семантика политик: "
        f"{'да' if compliance_report.get('semantic_analysis') else 'нет'}",
        "",
        (
            "Блокирующая пара (REQ-DATA-MIN / REQ-NO-META-IN-CTX): "
            f"{sts.get('REQ-DATA-MIN') or '—'} · {sts.get('REQ-NO-META-IN-CTX') or '—'}"
        ),
        "",
        "Статусы REQ-*:",
    ]
    for r in compliance_report.get("results") or []:
        rid = r.get("requirement_id")
        status = r.get("status")
        title = str(r.get("requirement_short", "")).strip()
        suffix = f" — {title}" if title else ""
        lines.append(f"  • {rid}: {status}{suffix}")
    lines.append("")
    lines.append(
        "(Полный текст оркестратора см. поле decision_statement в compliance_report.json и в security_assessment.md.)"
    )
    return "\n".join(lines)


def format_security_gate_plaintext(compliance_report: Mapping[str, Any] | None) -> str:
    """Компактный security gate для stdout: без раздела «три слоя»."""
    if compliance_report is None:
        return (
            "Итог для прода (security gate): не выполнялось (нет данных compliance).\n"
            "В прод считать согласование не замкнутым стандартным контуром MLSecOps без отдельного решения.\n"
        )
    _, gate_plain, _, rec_plain = _resolve_security_gate_messages(compliance_report)
    statuses, _, _ = _blocking_pair_status(compliance_report)
    pair_plain = "; ".join(
        f"{rid}→{statuses.get(rid) or '—'}"
        for rid in ("REQ-DATA-MIN", "REQ-NO-META-IN-CTX")
    )
    overall = compliance_report.get("overall", "?")
    viol = compliance_report.get("violations", "?")
    warns_agg = compliance_report.get("warnings", "?")
    return (
        "Итог для прода (security gate):\n"
        f"  {pair_plain}\n"
        f"  overall / FAIL / WARN (агрегат): {overall} / {viol} / {warns_agg}\n"
        f"  Gate: {gate_plain}\n"
        f"  {rec_plain}\n"
    )


def format_scan_summary(
    *,
    flow_source_label: str,
    synopsis: Mapping[str, Any],
    threat_model_markdown: str,
    compliance_report: dict[str, Any] | None,
    raw_flow_export: Mapping[str, Any] | None,
    compliance_was_skipped: bool = False,
) -> str:
    """Краткий текст для терминала: граф, MAESTRO, compliance, gate, метаданные флоу."""
    lines = [
        "Краткая сводка для тикета. Подробно: threat_model.md, compliance_report.json, security_assessment.md.",
        f"Шапка security_assessment (MSK): {assessment_timestamp_msk()}",
        "",
        "--- Описание флоу (топология и поверхность) ---",
        f"Источник графа: {flow_source_label}",
    ]
    sm = synopsis.get("summary") or {}
    lines.append(
        "Сводка: "
        f"узлов {sm.get('node_count', '—')} · рёбер {sm.get('edge_count', '—')} · "
        f"входов {sm.get('entrypoint_count', '—')} · контуров контроля {sm.get('control_count', '—')}"
    )
    ep = synopsis.get("entrypoints") or []
    if isinstance(ep, list) and ep:
        lines.append(f"Точки входа: {', '.join(str(x) for x in ep)}")
    ctr = synopsis.get("controls") or []
    if isinstance(ctr, list) and ctr:
        lines.append(
            f"Контроли (кратко): {', '.join(str(x) for x in ctr[:8])}"
            + (" …" if len(ctr) > 8 else "")
        )
    assets = synopsis.get("assets") or []
    if isinstance(assets, list) and assets:
        lines.append(
            f"Активы (кратко): {', '.join(str(x) for x in assets[:8])}"
            + (" …" if len(assets) > 8 else "")
        )

    lines.extend(
        [
            "",
            "--- Модель угроз (MAESTRO), выжимка ---",
            format_maestro_console_brief(threat_model_markdown),
            "",
            "--- Проверка требований MLSecOps (REQ-*) ---",
        ]
    )
    if compliance_report is not None:
        lines.append(format_compliance_console_brief(compliance_report))
    elif compliance_was_skipped:
        lines.append(
            "Не выполнялась (--no-compliance). Полный gate — в security_assessment.md; "
            "стандартный контур MLSecOps по JSON политик не замкнут."
        )
    else:
        lines.append(format_compliance_console_brief(None))

    lines.extend(
        [
            "",
            "--- Итоговое решение для прода (security gate) ---",
            format_security_gate_plaintext(compliance_report),
            "",
            "--- Выходные данные экспорта флоу (метаданные агента) ---",
            format_flow_export_metadata_markdown(
                extract_flow_export_metadata(raw_flow_export if isinstance(raw_flow_export, dict) else None),
            ).rstrip(),
        ]
    )
    return "\n".join(lines)


def format_security_gate_section_markdown(compliance_report: Mapping[str, Any] | None) -> str:
    """Часть 3 сводного отчёта: оркестратор и вердикт для прода."""
    intro = (
        "### Три слоя и агрегатор\n\n"
        "| Слой | Содержание |\n"
        "|------|------------|\n"
        "| **MAESTRO** | Отдельный качественный отчёт (часть 1 документа). |\n"
        "| **Проверка REQ-*** | Отдельные детерминированные/семантические проверки по каждому требованию (часть 2 и JSON отчёт). |\n"
        "| **Оркестратор** | Ниже + поле **`decision_statement`** в части 2: агрегирует результат политик **наряду** с выходными метаданными экспорта флоу сверху. |\n"
        "\n"
        "**Правило прома MLSecOps:** нет **FAIL** по **`REQ-DATA-MIN`** или **`REQ-NO-META-IN-CTX`** → организационно **можно выпускать в прод**, но **обязательно с привлечением эксперта MLSecOps** для приёмки и учёта рисков из MAESTRO и советных REQ. "
        "**Любой FAIL по этим двум REQ** → **станов до устранения** (или управленческого исключения).\n"
    )
    if compliance_report is None:
        gate_block = (
            "### Вердикт security gate для прода\n\n"
            "- **Gate:** `НЕ УСТАНОВЛЕН` — формальная проверка политик не выполнялась "
            "(`--no-compliance` или эквивалент).\n"
            "- **Рекомендация:** выпуск считать **не подтверждённым** стандартным контуром MLSecOps, пока проверки нет или нет исключения.\n"
        )
        return intro + "\n" + gate_block

    st_line_md, gate_plain, rec_md, _ = _resolve_security_gate_messages(compliance_report)
    overall = str(compliance_report.get("overall", "?"))
    viol = compliance_report.get("violations", "?")
    warns_agg = compliance_report.get("warnings", "?")
    gate_inline = f"`{gate_plain}`"

    gate_block = (
        "### Вердикт security gate для прода\n\n"
        f"- **Статусы блокирующей пары:** {st_line_md}\n"
        f"- **overall / violations / warnings:** `{overall}` / {viol} / {warns_agg}\n"
        f"- **Gate (по паре секретов/метаданных + политики MLSecOps):** {gate_inline}\n"
        f"- **Рекомендация оркестратора:** {rec_md}\n"
    )
    return intro + "\n" + gate_block


def build_security_assessment_markdown(
    *,
    threat_model_markdown: str,
    compliance_report: dict[str, Any] | None,
    flow_export_metadata: Mapping[str, Any] | None,
    flow_source_label: str,
    generated_at_iso: str | None = None,
) -> str:
    """Сводка для тикета: экспорт флоу, MAESTRO, REQ-*, decision_statement, security gate."""
    ts = generated_at_iso if generated_at_iso is not None else assessment_timestamp_msk()
    meta_dict = extract_flow_export_metadata(flow_export_metadata) if flow_export_metadata else {}

    conclusion_lines: list[str] = []
    if compliance_report is None:
        conclusion_lines.extend([
            "### Детализация по требованиям (REQ-*)",
            "Проверки политик MLSecOps отключены (`--no-compliance` или эквивалент); детальные статусы недоступны.",
            "",
            "### `decision_statement` (оркестратор)",
            "Не сформирован: нет JSON compliance.",
            "",
        ])
    else:
        overall = compliance_report.get("overall", "?")
        viol = compliance_report.get("violations", "?")
        warns = compliance_report.get("warnings", "?")
        sem = compliance_report.get("semantic_analysis")
        conclusion_lines.extend([
            "### Детализация по требованиям (REQ-*, отдельный слой машинных проверок)",
            f"- **Итог (overall):** `{overall}`",
            f"- **Блокирующих нарушений (violations / FAIL):** {viol}",
            f"- **Предупреждений (warnings):** {warns}",
            f"- **Семантический слой политик использован:** {'да' if sem else 'нет'}",
            "",
            "### `decision_statement` — текст агрегатора-оркестратора (после REQ-*)",
            "",
            str(compliance_report.get("decision_statement", "(отсутствует)")),
            "",
        ])

    conclusion_block = "\n".join(conclusion_lines)
    part3 = format_security_gate_section_markdown(compliance_report)

    rek_lines = [f"- **Дата и время формирования документа:** {ts}"]
    if generated_at_iso is None:
        rek_lines.append("- **Часовой пояс отчёта:** MSK (UTC+3)")
    rek_lines.append(f"- **Источник топологии (флоу):** `{flow_source_label}`")

    parts = [
        "# Сводный документ безопасности флоу",
        "",
        "**Структура:** (1) экземпляр флоу и выходные метаданные экспорта → (2) **MAESTRO** отдельно → "
        "(3) **проверки REQ-*** отдельно + агрегированный **`decision_statement`** оркестратора → "
        "(4) ниже — **security gate** с правилом группы `REQ-DATA-MIN` / `REQ-NO-META-IN-CTX` и участием эксперта MLSecOps.",
        "",
        "## Служебные реквизиты",
        "",
        *rek_lines,
        "",
        "### Выходные данные флоу (экспорт / препрод)",
        "",
        format_flow_export_metadata_markdown(meta_dict),
        "---",
        "",
        "## Часть 1. MAESTRO — отдельный качественный слой (модель угроз)",
        "",
        threat_model_markdown.strip(),
        "",
        "---",
        "",
        "## Часть 2. Проверки требований REQ-* и `decision_statement` оркестратора",
        "",
        conclusion_block.strip(),
        "",
        "---",
        "",
        "## Часть 3. Агрегированное решение для прода (оркестратор / security gate)",
        "",
        part3.strip(),
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def score_to_severity(score: float) -> str:
    if score >= 8.5:
        return "CRITICAL"
    if score >= 6.5:
        return "HIGH"
    if score >= 5.0:
        return "MEDIUM"
    return "LOW"


def score_to_severity_ru(score: float) -> str:
    return SEVERITY_RU[score_to_severity(score)]


def _normalize_attack_summary(boart_report: dict[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in boart_report.get("results", []):
        steps = item.get("steps", [])
        verdict = item.get("verdict") or ("BROKEN" if item.get("success") else "RESILIENT")
        summary.append(
            {
                "attack_name": item.get("attack_name"),
                "goal": item.get("goal"),
                "verdict": verdict,
                "success": verdict == "BROKEN" or bool(item.get("success")),
                "max_score": item.get("max_score", 0.0),
                "steps_count": len(steps),
                "last_response_preview": (steps[-1].get("target_response", "")[:280] if steps else ""),
                "error": item.get("error"),
            }
        )
    return summary


def _per_threat_severity(boart_report: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for item in boart_report.get("results", []):
        attack_name = str(item.get("attack_name", "unknown"))
        score = float(item.get("max_score", 0.0))
        grouped.setdefault(attack_name, []).append(score)

    results: list[dict[str, Any]] = []
    for attack_name, scores in sorted(grouped.items()):
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        sev = score_to_severity(max_score)
        results.append(
            {
                "threat": attack_name,
                "threat_label_ru": THREAT_LABEL_RU.get(attack_name, attack_name),
                "goals_count": len(scores),
                "max_score": max_score,
                "avg_score": round(avg_score, 2),
                "severity": sev,
                "severity_ru": SEVERITY_RU[sev],
            }
        )
    return results


def build_final_report(
    flow_path: str,
    synopsis: dict[str, Any],
    threat_model_markdown: str,
    boart_report: dict[str, Any],
    attack_plan: dict[str, Any] | None = None,
    compliance_report: dict[str, Any] | None = None,
    flow_export_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    syn = enrich_synopsis(synopsis)
    return {
        "system": {
            "flow_path": flow_path,
            "flow_export_metadata": extract_flow_export_metadata(
                flow_export_payload if isinstance(flow_export_payload, dict) else None,
            ),
            "entrypoints": syn.get("entrypoints", []),
            "assets": syn.get("assets", []),
            "controls": syn.get("controls", []),
        },
        "threat_model": {
            "report_markdown": threat_model_markdown,
        },
        "adversarial_testing": {
            "attack_plan": attack_plan or {},
            "summary": boart_report.get("summary", {}),
            "goals": _normalize_attack_summary(boart_report),
            "raw_report": boart_report,
        },
        "compliance": compliance_report or {"overall": "SKIP", "details": "Compliance check was not run."},
        "risk": {
            "per_threat_severity": _per_threat_severity(boart_report),
            "severity_scale": SEVERITY_ORDER,
            "severity_scale_ru": [SEVERITY_RU[s] for s in SEVERITY_ORDER],
            "описание_ru": "Оценка по максимальному баллу судьи (1–10) в разрезе набора атак; шкала severity для приоритизации.",
        },
    }


def finalize_pipeline_artifacts(
    run_dir: str | Path,
    *,
    flow_path: str,
    synopsis: dict[str, Any],
    threat_model_markdown: str,
    validator_compliance: Mapping[str, Any],
    gate_verdict_status: str,
    gate_verdict_comment: str,
    flow_export_payload: Mapping[str, Any] | None = None,
    attack_plan: dict[str, Any] | None = None,
    boart_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать и записать ``final_report.json`` + ``security_assessment.md`` (legacy pipeline)."""
    syn = enrich_synopsis(synopsis)
    comp = compliance_from_validator(
        validator_compliance,
        gate_verdict_status=gate_verdict_status,
        gate_verdict_comment=gate_verdict_comment,
        synopsis=syn,
    )
    br = boart_report if boart_report is not None else {"summary": {}, "results": []}
    final = build_final_report(
        flow_path,
        syn,
        threat_model_markdown,
        br,
        attack_plan=attack_plan,
        compliance_report=comp,
        flow_export_payload=flow_export_payload,
    )
    assessment = build_security_assessment_markdown(
        threat_model_markdown=threat_model_markdown,
        compliance_report=comp,
        flow_export_metadata=flow_export_payload,
        flow_source_label=flow_path,
    )
    base = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "final_report.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base / "security_assessment.md").write_text(assessment, encoding="utf-8")
    (base / "compliance_report.json").write_text(
        json.dumps(comp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final

