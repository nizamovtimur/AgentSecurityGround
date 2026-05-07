# Пайплайн

Один вход: `python -m cli.run_pipeline`.

## Шаги

1. **Граф** — из файла или Langflow API → `security_synopsis.json` (+ `flow_from_langflow.json` при API).
2. **Угрозы и политики** — MAESTRO → `threat_model.md`; проверки → `compliance_report.json`; сводка → `security_assessment.md`. Флаг `--no-compliance` отключает только политики.
3. **План атак** — агент или эвристика → `attack_plan.json` (только если не `--no-boart`).
4. **BOART** → `boart_report.json`.
5. **Итог** → `final_report.json` (если был BOART).

`--no-boart` — шаги 1–2; цель HTTP не нужна.

## Политики (compliance)

Сначала статические правила по синопсису, при наличии LLM — один дополнительный вызов для уточнения; результат объединяется. Итог в `decision_statement` в JSON; при планировании атак и BOART передаётся отдельным полем, не смешивается с текстом MAESTRO.

## BOART

Описание цели для агентов по умолчанию строится из синопсиса, MAESTRO и при необходимости краткой выдержки из `decision_statement`. Переопределение: `--target-description`.

## Суровость по баллам судьи BOART

Ниже 5 — LOW; 5.0–6.4 — MEDIUM; 6.5–8.4 — HIGH; от 8.5 — CRITICAL.
