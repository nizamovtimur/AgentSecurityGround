# End-to-End Pipeline Orchestrator

## Goal

Single entrypoint that executes:

- S1: static flow analysis (parse graph, build synopsis);
- S2: MAESTRO threat modeling;
- S3: attack planning (LLM agent or heuristic);
- S4: BOART adversarial testing;
- S5: consolidated final report.

CLI: `python -m cli.run_pipeline`

## Input Parameters

- `--flow`: Langflow JSON file.
- `--target-endpoint`: black-box target. Langflow: `http://<host>/api/v1/run/<FLOW_ID>` (requires `LANGFLOW_API_KEY` for `x-api-key`). Other backends: POST `{message, history}`.
  Langflow mode uses a new `session_id` on every request (`send`) by design.
- `--target-timeout` (optional): override HTTP timeout (seconds) for each target call; else `MLSECOPS_TARGET_TIMEOUT` / `LANGFLOW_RUN_TIMEOUT` / `OPENAI_TIMEOUT` / 300s.
- `--attacks`: optional comma-separated dataset names from `datasets/*.parquet` (manual override).
- `--attack-planner` (`agent` default | `heuristic`): S3 uses an LLM to pick a minimal subset of all `*.parquet` stems from synopsis + threat model (`prompts/attack_datasets_catalog.json` adds descriptions); on failure or `heuristic`, use marker-based rules (`select_attacks_from_context`).
- `--goals-per-attack` (default `3`).
- `--max-steps` (default `5`).
- `--language` (`ru|en|any`, default `ru`; включает русские шаблоны судьи/суммаризатора BOART).
- `--max-strategies` (default `10`).
- `--artifacts-dir`: where all outputs are written.

## Output Artifacts

- `security_synopsis.json` — factual graph parse: assets, entrypoints, system prompts, tool edges, author parameters.
- `threat_model.md` — structured MAESTRO threat report.
- `attack_plan.json` — selected attacks with rationale and `planner` (`agent` | `heuristic` | `manual` if `--attacks`).
- `boart_report.json` — detailed adversarial test traces and ASR.
- `final_report.json` — consolidated report with per-threat severity.

## Automatic Attack Planning

If `--attacks` is not provided, the default **agent** planner sends the full manifest of `datasets/*.parquet` stems (plus optional catalog descriptions), `security_synopsis`, and `threat_model.md` to an LLM and expects a minimal JSON `attacks` list. **`--attack-planner heuristic`** skips the LLM and uses marker + static rules.

## BOART Target Description

`target_description` for Boss/Attacker agents is built automatically from S1 synopsis and S2 threat model: system mission (≤2 sentences), first line of system prompt, tool names. Override with `--target-description`.

## Severity Mapping

Final severity is based on maximum BOART judge score:

- `LOW`: score < 5.0
- `MEDIUM`: 5.0–6.4
- `HIGH`: 6.5–8.4
- `CRITICAL`: >= 8.5
