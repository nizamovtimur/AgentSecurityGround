"""
Моделирование угроз для агентных флоу.

Использует модель угроз из файла prompts/threat_model.txt.
Модель LLM — из APP_SEC_ATTACK_MODEL.
"""

import json

from .config import APP_SEC_ATTACK_MODEL, get_openai_client
from .logging_config import get_logger

log = get_logger("threat_modeling")
from .prompts_loader import load_prompt, load_threat_model


def run_threat_modeling(flow_analysis, client=None):
    """
    Выполняет threat modeling для агентного флоу.

    Использует модель угроз из файла prompts/threat_model.txt. Системный промпт — prompts/threat_model_system.txt.
    Модель LLM — APP_SEC_ATTACK_MODEL, API — OPENAI_API_BASE.

    Args:
        flow_analysis: Результат FlowAnalysis в виде dict (nodes, edges, agents, tools, etc.)
        client: OpenAI client. Если None — создаётся через get_openai_client()

    Returns:
        Markdown-отчёт с разделами: Mission, Assets, Entrypoints, Threats, Risks, etc.
    """
    if not flow_analysis or not isinstance(flow_analysis, dict):
        return ""
    log.info("Running threat modeling (MAESTRO)")
    client = client or get_openai_client()
    threat_model = load_threat_model()
    sys_prompt = load_prompt("threat_model_system.txt")

    if not sys_prompt:
        sys_prompt = (
            "You are an expert in cybersecurity threat modeling. "
            "Generate a detailed threat analysis report in Markdown format."
        )

    # Подставляем модель угроз
    sys_prompt = sys_prompt.replace("<THREAT_MODEL>", threat_model)

    # JSON графа для анализа
    graph_json = json.dumps(flow_analysis, ensure_ascii=False, indent=2)
    sys_prompt = sys_prompt.replace("<JSON>", graph_json[:12000])

    resp = client.chat.completions.create(
        model=APP_SEC_ATTACK_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Perform the threat analysis. Output only the report in Markdown."},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    log.info("Threat modeling complete, report length: %d", len(content))
    return content
