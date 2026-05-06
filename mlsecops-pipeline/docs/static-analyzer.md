# Static Security Analyzer (S1 -> S2)

## Purpose

This module implements static security analysis for Langflow-based agentic systems:

1. Parse large Langflow flow JSON deterministically.
2. Build compact security synopsis for LLM context.
3. Generate MAESTRO-style threat model markdown report through OpenAI.

## Module Layout

- `parsers/langflow_parser.py`: tolerant deterministic parser.
- `models/security_graph.py`: normalized graph dataclasses.
- `services/synopsis_builder.py`: compact synopsis builder.
- `llm/openai_client.py`: OpenAI wrapper with `python-dotenv`.
- `services/threat_modeling_service.py`: prompt assembly + LLM call.
- `cli/run_static_threat_model.py`: CLI entrypoint for one flow.

## Deterministic Parsing Rules

- Use structured fields from `edge.data.sourceHandle` and `edge.data.targetHandle` as source of truth.
- Do not parse semantics from `edge.id` and stringified handles.
- Gracefully skip optional/missing fields.
- Remove token-heavy and non-essential fields such as `template.code`.
- Redact sensitive template values (`api_key`, `token`, `secret`, `password`).

## Synopsis Contract

`build_security_synopsis()` returns:

- `summary`: node/edge/entrypoint/control counters.
- `entrypoints`: externally reachable nodes (`ChatInput`, `URL`, `File`, `MCPTools`, `APIRequest`).
- `controls`: security/validation nodes (`GuardrailValidator`, parser/filter-like controls).
- `assets`: normalized core assets (`<display_name>::<role>`).
- `nodes`: node list with `id`, `name`, `type`, `role`, `risk_flags`, `author_parameters` (key meta-params from template).
- `edges`: compact edge list with dataflow handle semantics.
- `system_prompts`: system prompt texts (static fields + dynamic via `prompt → system_prompt` edges).
- `prompt_edges`, `tool_edges`: semantic edge subsets (system prompt sources; component-as-tool wiring).
- `author_parameters_by_node`: key parameters set by the flow author (system prompt, model, temperature, tool_mode, …).

## LLM Integration

`ThreatModelingService` composes prompt from:

- `prompts/threat_model.txt` (текст для `<THREAT_MODEL_CONTEXT>`: поверхности, классы угроз, процесс),
- `prompts/threat_model_system_ru.txt` (инструкции и структура отчёта; плейсхолдеры `<THREAT_MODEL_CONTEXT>`, `<JSON>`),
- generated synopsis JSON.

`OpenAIClient` reads API key via `.env` (`OPENAI_API_KEY`) and uses `openai` SDK.

## Usage

```bash
cd mlsecops-pipeline
python -m pip install -r requirements.txt
python -m cli.run_static_threat_model \
  --flow ../langflow/flows/Windchaser.json \
  --synopsis-output ../artifacts/windchaser-security-synopsis.json \
  --output ../artifacts/windchaser-threat-model.md
```

CLI outputs two artifacts:

- intermediate `security_synopsis.json` (for auditability and debugging);
- final MAESTRO markdown threat model report.

## Tests

Run:

```bash
cd mlsecops-pipeline
pytest -q
```

Coverage focus:

- Parser robustness on real flow examples.
- Sensitive field redaction.
- Synopsis extraction and high-risk marker logic.
- Prompt assembly and threat modeling service contract.

