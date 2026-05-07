# Static Security Analyzer (S1 + S2)

Static analysis of Langflow flows: parse → synopsis → MAESTRO threat model.

## Module Layout

- `parsers/langflow_parser.py` — tolerant deterministic parser.
- `models/security_graph.py` — normalized graph dataclasses.
- `services/synopsis_builder.py` — compact synopsis + `build_target_description`.
- `services/threat_modeling_service.py` — MAESTRO prompt assembly + LLM call.
- `llm/openai_client.py` — OpenAI wrapper with `python-dotenv`.

## Parsing Rules

- Source of truth: `edge.data.sourceHandle` / `edge.data.targetHandle` structured fields.
  Do not parse semantics from `edge.id` strings.
- Skip optional/missing fields gracefully.
- Drop `template.code` (token-heavy).
- Redact sensitive template values (`api_key`, `token`, `secret`, `password`).

## Synopsis Contract

`build_security_synopsis()` returns:

- `summary` — counters (nodes, edges, entrypoints, controls).
- `entrypoints` — externally reachable nodes (`ChatInput`, `URL`, `File`, `MCPTools`, `APIRequest`).
- `controls` — security / validation nodes (`GuardrailValidator`, parser/filter-like controls).
- `assets` — `<display_name>::<role>`.
- `nodes` — `id`, `name`, `type`, `role`, `risk_flags`, `author_parameters`.
- `edges` — compact edge list with dataflow handle semantics.
- `system_prompts` — system prompt texts (static + dynamic via `prompt → system_prompt` edges).
- `prompt_edges`, `tool_edges` — semantic edge subsets.
- `author_parameters_by_node` — key params set by the flow author.

## Threat Modeling

`ThreatModelingService.generate_report(graph)` composes prompt from:

- `prompts/threat_model.txt` → `<THREAT_MODEL_CONTEXT>` (поверхности, классы угроз, процесс).
- `prompts/threat_model_system_ru.txt` (плейсхолдеры `<THREAT_MODEL_CONTEXT>`, `<JSON>`).
- Synopsis JSON.

Структура отчёта MAESTRO: секции **0–7**. Раздел **6** объединяет риски, эксплуатацию и наблюдаемость вокруг **сквозных kill chain**’ов по JSON-топологии и **узловых точках (choke points)**; раздел **7** связывает рекомендации с именованными цепочками и этими точками.

Единый вывод **`security_assessment.md`** формирует CLI в конце **S2**: отметка MSK (UTC+3), метаданные корня экспорта флоу, полный текст MAESTRO, формальный блок политик и **вердикт для security gate в прод** — с объяснением, почему MAESTRO может давать качественные замечания даже при `PASS` по политикам. При `--no-compliance` gate помечен как не установлен.

```bash
python -m cli.run_pipeline --flow path/to/flow.json --no-boart --no-compliance \
  --artifacts-dir ../artifacts/sast-only
```

Пошаговый режим только S1 и S2 (без выполнения ячеек S3–S5): `pipeline_demo.ipynb`.


```bash
cd mlsecops-pipeline && pytest -q
```

Covers parser robustness, sensitive redaction, synopsis extraction, prompt assembly.
