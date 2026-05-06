# MLSecOps Pipeline

Статический и динамический контур безопасности для агентных диалоговых систем (Langflow): парсинг графа → synopsis → модель угроз (MAESTRO) → план атак → BOART → итоговый отчёт.

## Быстрый старт

```bash
cd mlsecops-pipeline
python -m pip install -r requirements.txt
python -m pip install pytest   # или: pip install -e ".[dev]"
pytest -q
```

Пошаговый сценарий (препрод / артефакты): `pipeline_demo.ipynb`.

## CLI

**S1+S2 только** (один флоу):

```bash
python -m cli.run_static_threat_model \
  --flow ../langflow/flows/TravelPlanningAgents.json \
  --synopsis-output ../artifacts/travel-security-synopsis.json \
  --output ../artifacts/travel-threat-model.md
```

С загрузкой из Langflow:

```bash
export LANGFLOW_URL="https://your-langflow.example.com"
export FLOW_ID="your_flow_id"
export LANGFLOW_API_KEY="your_api_key"

python -m cli.run_static_threat_model \
  --flow-source langflow \
  --synopsis-output ../artifacts/preprod-security-synopsis.json \
  --output ../artifacts/preprod-threat-model.md
```

**BOART** (S4 отдельно): см. `docs/boart.md`.

```bash
export LANGFLOW_URL="http://localhost:7860"
export FLOW_ID="your_flow_id"
export LANGFLOW_API_KEY="your_api_key"

python -m cli.run_boart \
  --attacks system_prompt_leakage,harmbench_text \
  --target-endpoint "${LANGFLOW_URL}/api/v1/run/${FLOW_ID}" \
  --goals-per-attack 3 \
  --max-steps 5 \
  --language ru \
  --output ../artifacts/boart_report.json
```

**Полный оркестратор** S1–S5: `docs/pipeline-orchestrator.md`.

```bash
export LANGFLOW_URL="http://localhost:7860"
export FLOW_ID="same-as-in-langflow-ui"
export LANGFLOW_API_KEY="your_api_key"

python -m cli.run_pipeline \
  --flow ../langflow/flows/Windchaser.json \
  --target-endpoint "${LANGFLOW_URL}/api/v1/run/${FLOW_ID}" \
  --goals-per-attack 3 \
  --max-steps 5 \
  --language ru \
  --artifacts-dir ../artifacts/pipeline
```

Препрод Langflow (без локального JSON):

```bash
export LANGFLOW_URL="http://localhost:7860"
export FLOW_ID="your_flow_id"
export LANGFLOW_API_KEY="your_api_key"

python -m cli.run_pipeline \
  --flow-source langflow \
  --langflow-url "$LANGFLOW_URL" \
  --flow-id "$FLOW_ID" \
  --target-endpoint "${LANGFLOW_URL}/api/v1/run/${FLOW_ID}" \
  --artifacts-dir ../artifacts/pipeline-preprod
```

Флаг `--attacks` опционален: иначе S3 вызывает LLM-планировщик по полному списку `datasets/*.parquet` и МУ (`--attack-planner agent`, по умолчанию); при сбое или `--attack-planner heuristic` — эвристика по маркерам и synopsis.

## Окружение

- `.env` в корне репозитория или в `mlsecops-pipeline`: `OPENAI_API_KEY`.
- Для Langflow: `LANGFLOW_API_KEY` (и при необходимости `LANGFLOW_URL`, `FLOW_ID`) — для загрузки графа **и** для `HttpTargetClient`, если `--target-endpoint` указывает на `/api/v1/run/...` (заголовок `x-api-key`).
- Таймаут HTTP к цели (BOART → Langflow run): `MLSECOPS_TARGET_TIMEOUT` или `LANGFLOW_RUN_TIMEOUT`, иначе подхватывается `OPENAI_TIMEOUT`, иначе **300** сек; при необходимости `--target-timeout` в CLI.
- Опционально: `OPENAI_BASE_URL`, `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES`.
- S2: `prompts/threat_model_system_ru.txt` (плейсхолдеры `<THREAT_MODEL_CONTEXT>`, `<JSON>`) и эталон из `prompts/threat_model.txt` (поверхности, классы угроз, процесс оценки).

## Артефакты `run_pipeline`

| Файл | Смысл |
|------|--------|
| `security_synopsis.json` | Компактная карта графа |
| `threat_model.md` | Отчёт MAESTRO |
| `attack_plan.json` | Выбранные датасеты, rationale, поле `planner` (`agent` / `heuristic` / `manual`) |
| `boart_report.json` | Трейсы BOART, ASR |
| `final_report.json` | Сводка для тикета |
| `flow_from_langflow.json` | Только при `--flow-source langflow` |

## Документация

- `docs/static-analyzer.md` — парсер и S1–S2
- `docs/boart.md` — BOART
- `docs/pipeline-orchestrator.md` — полный пайплайн
