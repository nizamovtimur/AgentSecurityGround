# MLSecOps Pipeline

Разбор графа Langflow → синопсис → отчёт по угрозам (MAESTRO) и проверка политик → по желанию план атак и BOART → итоговый JSON.

## Запуск

```bash
cd mlsecops-pipeline
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

Демо: `pipeline_demo.ipynb` (тот же порядок шагов, что `python -m cli.run_pipeline`).

## CLI

**Без живой цели (только разбор и отчёты):**

```bash
python -m cli.run_pipeline \
  --flow ../langflow/flows/Windchaser.json \
  --no-boart \
  --artifacts-dir ../artifacts/sast
```

**С BOART** — нужны `--target-endpoint` (для Langflow обычно `{LANGFLOW_URL}/api/v1/run/{FLOW_ID}`) и ключи в окружении.

Полезное:

- `--no-compliance` — не писать `compliance_report.json`
- `--no-boart` — не запускать атаки
- `--attacks a,b` — фиксированный список датасетов
- `--insecure` — `verify=False` для HTTPS Langflow и цели BOART

## Переменные

- `OPENAI_API_KEY`; опционально `OPENAI_BASE_URL`, `OPENAI_TIMEOUT`, `PIPELINE_OPENAI_MODEL`
- Langflow: `LANGFLOW_URL`, `FLOW_ID`, `LANGFLOW_API_KEY`
- Таймаут цели BOART: `MLSECOPS_TARGET_TIMEOUT` / `LANGFLOW_RUN_TIMEOUT` / `OPENAI_TIMEOUT` (иначе 300 с)
- TLS: `MLSECOPS_LANGFLOW_VERIFY_SSL`, `MLSECOPS_TARGET_VERIFY_SSL` (`0` / `false` — без проверки сертификата)

## Файлы в `--artifacts-dir`

| Файл | Назначение |
|------|------------|
| `security_synopsis.json` | Узлы, рёбра, промпты |
| `threat_model.md` | MAESTRO |
| `compliance_report.json` | Политики REQ-* |
| `security_assessment.md` | Сводка |
| `attack_plan.json` | Выбранные атаки |
| `boart_report.json` | Результаты BOART |
| `final_report.json` | Агрегат |
| `flow_from_langflow.json` | Сырой ответ API (если грузили с Langflow) |

## Документы

- `docs/static-analyzer.md` — парсер и синопсис  
- `docs/boart.md` — BOART  
- `docs/pipeline-orchestrator.md` — общая схема  
