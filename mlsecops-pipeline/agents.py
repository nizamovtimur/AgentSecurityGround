"""Три LLM-агента security gate: угрозы, соответствие, эскалация человеку."""

from __future__ import annotations

import json
import re
from typing import Any

from llm import LLMClient
from logging_utils import get_logger
from resources import CORPORATE_THREAT_MODEL, SENSITIVE_DATA_CATEGORIES
from static_checks import (
    static_data_min_findings,
    static_least_privilege_hints,
    static_meta_findings,
)
from synopsis import build_compliance_context, build_llm_context

log = get_logger("agents")

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _strip_fence(raw: str) -> str:
    return _JSON_FENCE.sub("", raw.strip())


def _extract_json(raw: str) -> dict[str, Any]:
    """Вытащить JSON из ответа (чистый или внутри markdown/мусора)."""
    cleaned = _strip_fence(raw)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    log.warning("ComplianceAgent: не удалось разобрать JSON (%s симв.), превью: %s", len(raw), raw[:200])
    return {}


class ThreatModelAgent:
    """Конкретная МУ и меры митигации по корпоративной таксономии."""

    _SYSTEM = """Ты — специалист по безопасности AI-агентов.
Источник архитектуры — JSON synopsis Langflow-flow в сообщении пользователя.
Корпоративная эталонная модель угроз (таксономия поверхностей, классов, мер):
---
{threat_model}
---
Сформируй Markdown **только** с двумя разделами (на русском):
## Конкретная модель угроз для сценария
Таблица: Поверхность атаки | Класс угроз | Узел/связь из JSON | Краткий kill chain | Вероятность (низк/сред/выс)
Только угрозы, достижимые по топологии JSON. Не выдумывай узлы.

## Меры митигации (из корпоративной МУ)
Таблица: Угроза (строка выше) | Типовые меры из эталона | Что уже есть в flow (controls) | Gap
Бери меры из эталонной таблицы, не придумывай новые классы.
Не повторяй фразы. Не выходи за эти два раздела."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system = self._SYSTEM.format(threat_model=CORPORATE_THREAT_MODEL.strip())

    def run(self, synopsis: dict[str, Any]) -> str:
        ctx = build_llm_context(synopsis)
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        log.debug("ThreatModelAgent: payload %s байт", len(payload))
        return self._llm.complete(
            self._system,
            f"Synopsis flow:\n```json\n{payload}\n```\nСформируй оба раздела.",
            max_tokens=6000,
        )


class ComplianceAgent:
    """Проверка трёх корпоративных требований; ответ — JSON + Markdown."""

    _SYSTEM = """Ты — аудитор безопасности AI-агентных систем (MLSecOps). Ответ — ОДИН валидный JSON, без markdown.

Категории чувствительных данных организации (эталон для REQ-DATA-MIN):
---
{sensitive}
---

Главная задача REQ-DATA-MIN — семантический аудит АРХИТЕКТУРЫ и ЛОГИКИ обработки по полным text в system_prompts и топологии флоу (entrypoints, controls, tool_edges, nodes). Не ограничивайся поиском литеральных api_key= в промпте.

REQ-DATA-MIN — FAIL, если по смыслу инструкций и связей флоу:
1. Сырой пользовательский ввод или полная история диалога подаётся в контекст LLM для анализа (input_value, user_request, {{text}}, CUSTOMER:/OPERATOR:, «разбери последнее сообщение», «полная история переписки» и т.п.) — категории ПДн/банковские данные проходят через модель.
2. Модели поручено санитизировать/маскировать/фильтровать/«не выдавать» ПДн в ответе — значит чувствительные данные УЖЕ в контексте; допустима только предобработка ДО LLM отдельным компонентом в графе (controls), не инструкцией агенту.
3. Агент по промпту обрабатывает операции/счета/суммы/идентификаторы клиента/историю банковских операций из переписки внутри LLM (даже если в ответе маскировать PPPPP...XXXX).
4. Инструкция подставлять/использовать ключи, токены, пароли, bearer, credential из пользовательского сообщения (любые имена полей и плейсхолдеры).
5. В system prompt вшиты роли/разрешения/лимиты доступа или «права из чата» вместо изоляции на сервере.

REQ-DATA-MIN — PASS только если чувствительные категории не проектируются в контекст LLM по задуманной логике, либо явно обоснован бизнес-процессом (укажи в rationale).

static_findings (только литеральные секреты/метаданные) — вспомогательный сигнал; архитектурное нарушение оценивай сам по полному тексту промптов.

REQ-LEAST-PRIVILEGE — разделение привилегий (read/write, разные MCP/домены); FAIL при явном совмещении несовместимых доменов, иначе WARN.

REQ-HUMAN-REVIEW — сомнительные места для эксперта; WARN или PASS.

В evidence обязательно цитата (excerpt) из system_prompts и причина на русском.

Формат:
{{
  "REQ-DATA-MIN": {{"status":"PASS|WARN|FAIL","rationale":"...","evidence":[{{"node":"...","excerpt":"...","reason":"..."}}]}},
  "REQ-LEAST-PRIVILEGE": {{"status":"PASS|WARN|FAIL","rationale":"...","evidence":[]}},
  "REQ-HUMAN-REVIEW": {{"status":"PASS|WARN","rationale":"...","evidence":[]}}
}}"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._system = self._SYSTEM.format(sensitive=SENSITIVE_DATA_CATEGORIES.strip())

    @staticmethod
    def static_fallback(synopsis: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Только статические проверки, если LLM недоступен."""
        static = ComplianceAgent._collect_static(synopsis)
        agent = ComplianceAgent.__new__(ComplianceAgent)
        merged = agent._merge_static({}, static)
        for req in merged:
            if not merged[req].get("rationale"):
                merged[req]["rationale"] = "LLM-анализ недоступен; применена только статическая проверка."
        return merged, agent._to_markdown(merged)

    @staticmethod
    def _collect_static(synopsis: dict[str, Any]) -> dict[str, Any]:
        return {
            "data_min": static_data_min_findings(synopsis),
            "access_meta": static_meta_findings(synopsis),
            "least_privilege_hints": static_least_privilege_hints(synopsis),
        }

    def run(self, synopsis: dict[str, Any]) -> tuple[dict[str, Any], str]:
        static = self._collect_static(synopsis)
        log.debug(
            "ComplianceAgent static: secrets=%s, access_meta=%s, lp_hints=%s",
            len(static["data_min"]),
            len(static["access_meta"]),
            len(static["least_privilege_hints"]),
        )
        ctx = build_compliance_context(synopsis, static)
        user = (
            "Оцени в первую очередь АРХИТЕКТУРУ: проходят ли чувствительные категории через контекст LLM "
            "по логике system_prompts (не только литеральные секреты). "
            "Прочитай каждый system_prompts[].text целиком.\n\n"
            "Вход:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
        )
        raw = self._llm.complete(self._system, user, json_mode=True, max_tokens=4096)
        result = self._parse(raw, static)
        log.debug("ComplianceAgent: LLM-анализ завершён")
        return result

    def _parse(self, raw: str, static: dict) -> tuple[dict[str, Any], str]:
        data = _extract_json(raw)
        merged = self._merge_static(data, static)
        md = self._to_markdown(merged)
        return merged, md

    @staticmethod
    def _worst(*statuses: str) -> str:
        rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
        return max(statuses, key=lambda s: rank.get(s, 0))

    def _merge_static(self, llm: dict, static: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for req in ("REQ-DATA-MIN", "REQ-LEAST-PRIVILEGE", "REQ-HUMAN-REVIEW"):
            entry = llm.get(req) if isinstance(llm.get(req), dict) else {}
            status = entry.get("status", "PASS")
            if req == "REQ-DATA-MIN":
                if static.get("data_min") or static.get("access_meta"):
                    status = self._worst(status, "FAIL")
            if req == "REQ-LEAST-PRIVILEGE" and static.get("least_privilege_hints"):
                status = self._worst(status, "WARN")
            out[req] = {
                "status": status if status in ("PASS", "WARN", "FAIL") else "PASS",
                "rationale": entry.get("rationale", ""),
                "evidence": entry.get("evidence", []),
            }
        return out

    def _to_markdown(self, results: dict[str, Any]) -> str:
        labels = {
            "REQ-DATA-MIN": "Минимизация данных и отсутствие секретов в контексте LLM",
            "REQ-LEAST-PRIVILEGE": "Разделение привилегий (компоненты / MCP)",
            "REQ-HUMAN-REVIEW": "Сигналы для ручной проверки",
        }
        lines = ["## Проверка соответствия требованиям", ""]
        for rid, title in labels.items():
            r = results.get(rid, {})
            st = r.get("status", "SKIP")
            lines += [f"### {rid} — {title}", f"**Статус:** {st}", "", r.get("rationale") or "—", ""]
            for ev in r.get("evidence") or []:
                if isinstance(ev, dict):
                    lines.append(f"- `{ev.get('node','?')}`: {ev.get('reason', ev.get('excerpt', ''))}")
            lines.append("")
        return "\n".join(lines)


def human_review_from_compliance(compliance: dict[str, Any]) -> str:
    """Fallback без отдельного LLM-вызова."""
    hr = compliance.get("REQ-HUMAN-REVIEW", {})
    lines = ["## Сигналы для эксперта MLSecOps", ""]
    if hr.get("rationale"):
        lines.append(str(hr["rationale"]))
        lines.append("")
    for ev in hr.get("evidence") or []:
        if isinstance(ev, dict):
            lines.append(f"- `{ev.get('node', '?')}`: {ev.get('reason', ev.get('excerpt', ''))}")
    if len(lines) <= 2:
        lines.append("*Дополнительных сигналов нет (см. блок соответствия REQ-HUMAN-REVIEW).*")
    lines.append("")
    return "\n".join(lines)


class HumanReviewAgent:
    """Дополнительный обзор подозрительных мест (не блокирует gate)."""

    _SYSTEM = """Ты — аналитик безопасности. По JSON найди риски для ручной проверки.
Верни ТОЛЬКО Markdown одного раздела:
## Сигналы для эксперта MLSecOps
До 10 коротких пунктов списка (- ), каждый с именем узла из JSON.
Без повторов. Без лишних разделов. Без JSON."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, synopsis: dict[str, Any]) -> str:
        ctx = {
            "summary": synopsis.get("summary"),
            "system_prompts": synopsis.get("system_prompts"),
            "tool_edges": synopsis.get("tool_edges"),
            "controls": synopsis.get("controls"),
            "entrypoints": synopsis.get("entrypoints"),
        }
        payload = json.dumps(ctx, ensure_ascii=False, indent=2)
        return self._llm.complete(
            self._SYSTEM,
            f"Контекст:\n```json\n{payload}\n```",
            max_tokens=2048,
        )
