"""Тонкая обёртка OpenAI (pydantic-конфиг, корпоративный base URL, отключение SSL)."""

from __future__ import annotations

import json
import os
import time

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field

from config import (
    openai_http_client,
    resolve_openai_base_url,
    ssl_verify as resolve_ssl_verify,
)
from logging_utils import get_logger

log = get_logger("llm")


def _max_tokens_default() -> int:
    return int(os.getenv("VALIDATOR_MAX_TOKENS", "8192"))


def _thinking_disabled_by_default(model: str) -> bool:
    """DeepSeek V4 в thinking mode часто отдаёт пустой content — для gate отключаем."""
    if os.getenv("VALIDATOR_THINKING", "").strip().lower() in ("1", "true", "yes", "on", "enabled"):
        return False
    return "deepseek" in model.lower()


def _extra_body_for_model(model: str) -> dict[str, object] | None:
    if not _thinking_disabled_by_default(model):
        return {"thinking": {"type": "enabled"}}
    if "deepseek" in model.lower():
        return {"thinking": {"type": "disabled"}}
    return None


def _content_parts_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
                elif part.get("text"):
                    chunks.append(str(part["text"]))
            elif hasattr(part, "text") and part.text:
                chunks.append(str(part.text))
        return "\n".join(chunks).strip()
    return ""


def _message_text(message: object, *, choice: object | None = None) -> str:
    """Текст из OpenAI message: content, parts, reasoning_content (DeepSeek V4)."""
    content = getattr(message, "content", None)
    text = _content_parts_to_text(content)
    if text:
        return text

    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val and str(val).strip():
            log.warning(
                "LLM: поле content пусто, используем %s (%s симв.)",
                attr,
                len(str(val)),
            )
            return str(val).strip()

    if hasattr(message, "model_dump"):
        data = message.model_dump()
        text = _content_parts_to_text(data.get("content"))
        if text:
            return text
        for key in ("reasoning_content", "reasoning"):
            val = data.get(key)
            if val and str(val).strip():
                log.warning("LLM: content пусто, взято из model_dump[%s]", key)
                return str(val).strip()

    if choice is not None:
        fr = getattr(choice, "finish_reason", None)
        if fr:
            log.error("LLM: пустой ответ, finish_reason=%s", fr)
    return ""


class LLMConfig(BaseModel):
    model: str = Field(default="Qwen/Qwen3-Coder-Next")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout: float = Field(default=300.0, gt=0)
    base_url: str | None = None
    ssl_verify: bool = True

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            model=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-Coder-Next"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "300")),
            base_url=resolve_openai_base_url(),
            ssl_verify=resolve_ssl_verify("openai"),
        )


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("Не задан OPENAI_API_KEY.")
        cfg = config or LLMConfig.from_env()
        if base_url is not None:
            cfg = cfg.model_copy(update={"base_url": resolve_openai_base_url(base_url)})
        if model is not None:
            cfg = cfg.model_copy(update={"model": model})
        if verify_ssl is not None:
            cfg = cfg.model_copy(update={"ssl_verify": verify_ssl})

        verify = cfg.ssl_verify
        http_client = openai_http_client(verify=verify)
        kwargs: dict = {
            "api_key": key,
            "timeout": cfg.timeout,
            "http_client": http_client,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self.config = cfg
        self._client = OpenAI(**kwargs)
        log.debug(
            "LLM клиент: model=%s, base_url=%s, tls_verify=%s, timeout=%ss",
            cfg.model,
            cfg.base_url or "(default)",
            cfg.ssl_verify,
            cfg.timeout,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        mt = max_tokens if max_tokens is not None else _max_tokens_default()
        log.debug(
            "Запрос chat.completions: system=%s симв., user=%s симв., json_mode=%s, max_tokens=%s",
            len(system),
            len(user),
            json_mode,
            mt,
        )
        t0 = time.monotonic()
        use_json = json_mode
        last_err: Exception | None = None

        extra = _extra_body_for_model(self.config.model)
        disable_thinking_retry = False

        for attempt in range(3):
            try:
                req: dict = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": mt,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if use_json:
                    req["response_format"] = {"type": "json_object"}
                body = extra
                if disable_thinking_retry and "deepseek" in self.config.model.lower():
                    body = {"thinking": {"type": "disabled"}}
                if body:
                    req["extra_body"] = body
                log.debug("Request body: %s", json.dumps(req, ensure_ascii=False)[:2000])
                log.debug("Request URL: %s/model=%s", self.config.base_url, self.config.model)
                resp = self._client.chat.completions.create(**req)
                choice = resp.choices[0]
                text = _message_text(choice.message, choice=choice)
                if not text:
                    if not disable_thinking_retry and "deepseek" in self.config.model.lower():
                        log.warning(
                            "LLM: пустой ответ, повтор с thinking=disabled (попытка %s)",
                            attempt + 1,
                        )
                        disable_thinking_retry = True
                        continue
                    fr = getattr(choice, "finish_reason", "?")
                    raise RuntimeError(
                        f"Пустой ответ LLM (finish_reason={fr}). "
                        "Попробуйте VALIDATOR_MAX_TOKENS или другую модель."
                    )
                elapsed = time.monotonic() - t0
                log.debug("Ответ LLM: %s симв. за %.1f с", len(text), elapsed)
                return text
            except BadRequestError as exc:
                last_err = exc
                if use_json and attempt == 0:
                    log.warning("json_mode отклонён бэкендом, повтор без response_format: %s", exc)
                    use_json = False
                    continue
                log.error("BadRequestError: status=%s, body=%s, request_id=%s",
                          exc.status_code, exc.body, getattr(exc, 'request_id', None))
                raise
            except Exception as exc:
                last_err = exc
                raise

        if last_err:
            raise last_err
        raise RuntimeError("LLM: исчерпаны попытки запроса.")
