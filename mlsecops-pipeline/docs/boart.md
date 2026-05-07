# BOART Adversarial Multi-Agent Module

## Overview

This module implements Boss-Orchestrated Agentic Red-Teaming (BOART) for black-box security testing of agentic dialogue systems.

Components:

- Boss agent: selects strategy and gives adaptation guidance.
- Attacker agent: generates attack prompts.
- Judge agent: scores target responses from 1 to 10.
- Summarizer: extracts new strategy from successful attacks.
- Strategy library: tracks effectiveness and keeps top-N strategies.
- Belief state + attack memory: stores per-goal and cross-goal context.

## Inputs

- attacks: list of attack dataset names (`.parquet` in `datasets/`).
- goals per attack: default `3`.
- max steps per goal: default `5`.
- language: `ru`, `en`, or `any` (default `ru` — русские `judge_ru.txt` / `summarizer_ru.txt`).
- max strategies in library: default `10`.

## CLI Usage

```bash
cd mlsecops-pipeline
python -m cli.run_boart \
  --attacks system_prompt_leakage,harmbench_text \
  --target-endpoint http://localhost:7860/api/v1/run/<FLOW_ID> \
  --target-description "Кратко: миссия системы и какие инструменты доступны агенту." \
  --goals-per-attack 3 \
  --max-steps 5 \
  --language ru \
  --max-strategies 10 \
  --output ../artifacts/boart_report.json
```

## Target API Contract

### Langflow (`/api/v1/run/<FLOW_ID>`)

If `--target-endpoint` contains `/api/v1/run/`, `HttpTargetClient` delegates to **`LangflowClient`** in `services/langflow_client` (the same module that fetches flow JSON for S1 via `GET /api/v1/flows/...`). Payload matches `ClientLangFlow` in `llamator/llamator-borat.ipynb`:

- **POST** JSON: `output_type`, `input_type` (`chat`), `input_value` (current user/attacker line), `session_id` (UUID, generated for every request/send; each Langflow call starts a fresh session).
- **Header:** `x-api-key` from env `LANGFLOW_API_KEY`.
- **Response:** assistant text from `outputs[0].outputs[0].messages[0].message` (string or `{ "text": "..." }`).
- **Timeouts:** each `POST` uses an httpx read timeout. Configure with `MLSECOPS_TARGET_TIMEOUT` or `LANGFLOW_RUN_TIMEOUT` (seconds), or reuse `OPENAI_TIMEOUT`; default if unset is **300** (slow local models often exceed 60s). CLI: `--target-timeout SEC` on `run_boart` / `run_pipeline`.

### Generic HTTP target

Otherwise the client sends:

```json
{
  "message": "<attack prompt>",
  "history": [{"role": "user|assistant", "content": "..."}]
}
```

Expected response JSON should contain one of fields: `response`, `answer`, `output`, `message` (string).

## Output

`boart_report.json` includes:

- run config;
- summary (`goals_total`, `goals_successful`, `asr`);
- per-goal step-by-step traces (selected strategy, prompts, responses, scores).

