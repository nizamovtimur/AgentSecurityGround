"""Общие настройки из окружения: SSL, URL и модели LLM."""

from __future__ import annotations

import os
import ssl
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

Target = Literal["langflow", "openai"]


def env_bool(name: str, default: bool = True) -> bool:
    """Истина для 1/true/yes/on; ложь для 0/false/no/off; иначе default."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def ssl_verify(target: Target = "langflow", *, override: bool | None = None) -> bool:
    """
    Проверка TLS-сертификатов.

    ``override`` — явное значение из кода/CLI.
    Иначе ``LANGFLOW_SSL_VERIFY`` / ``OPENAI_SSL_VERIFY``, затем общий
    ``SSL_VERIFY`` или ``VALIDATOR_SSL_VERIFY`` (по умолчанию True).
    """
    if override is not None:
        return override
    global_default = env_bool("VALIDATOR_SSL_VERIFY", env_bool("SSL_VERIFY", True))
    if target == "langflow" and os.getenv("LANGFLOW_SSL_VERIFY") is not None:
        return env_bool("LANGFLOW_SSL_VERIFY", True)
    if target == "openai" and os.getenv("OPENAI_SSL_VERIFY") is not None:
        return env_bool("OPENAI_SSL_VERIFY", True)
    return global_default


def urllib_ssl_context(verify: bool | None = None) -> ssl.SSLContext:
    if verify is None:
        verify = ssl_verify("langflow")
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def openai_http_client(verify: bool | None = None) -> Any:
    """httpx.Client для OpenAI SDK (verify=False для корпоративных CA)."""
    import httpx

    if verify is None:
        verify = ssl_verify("openai")
    return httpx.Client(verify=verify)


def resolve_openai_base_url(explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    for key in ("OPENAI_BASE_URL", "AITUNNEL_BASE_URL", "LANGFLOW_OPENAI_BASE_URL"):
        val = (os.getenv(key) or "").strip()
        if val:
            return val.rstrip("/")
    return None
