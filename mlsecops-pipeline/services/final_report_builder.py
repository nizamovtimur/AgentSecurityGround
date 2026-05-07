"""Итоговые отчёты и текстовые сводки для CLI и артефактов."""

from __future__ import annotations

import json
import re
from datetime import datetime
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
        summary.append(
            {
                "attack_name": item.get("attack_name"),
                "goal": item.get("goal"),
                "success": item.get("success", False),
                "max_score": item.get("max_score", 0.0),
                "steps_count": len(steps),
                "last_response_preview": (steps[-1].get("target_response", "")[:280] if steps else ""),
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
    return {
        "system": {
            "flow_path": flow_path,
            "flow_export_metadata": extract_flow_export_metadata(
                flow_export_payload if isinstance(flow_export_payload, dict) else None,
            ),
            "entrypoints": synopsis.get("entrypoints", []),
            "assets": synopsis.get("assets", []),
            "controls": synopsis.get("controls", []),
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

