"""
Конфигурация модуля из переменных окружения.

- APP_SEC_ATTACK_MODEL — модель для Boss и Attacker
- APP_SEC_JUDGE_MODEL — модель для Judge
- OPENAI_API_BASE — OpenAI-compatible API base URL (например https://api.openai.com/v1)
"""

import os

# Модели LLM
APP_SEC_ATTACK_MODEL = os.environ.get("APP_SEC_ATTACK_MODEL", "gpt-4o-mini")
APP_SEC_JUDGE_MODEL = os.environ.get("APP_SEC_JUDGE_MODEL", "gpt-4o-mini")

# OpenAI-compatible API base URL (если пусто — используется стандартный OpenAI)
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")


def get_openai_client():
    """Создаёт OpenAI client с base_url из OPENAI_API_BASE (если задан)."""
    from openai import OpenAI
    if OPENAI_API_BASE:
        return OpenAI(base_url=OPENAI_API_BASE.rstrip("/"))
    return OpenAI()
