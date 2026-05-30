"""Ошибки вызова цели BOART (Langflow / HTTP)."""

from __future__ import annotations


class TargetCallError(Exception):
    """Сбой POST к цели: HTTP 4xx/5xx, таймаут, сеть."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str = "",
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = (body or "")[:800]

    def short(self) -> str:
        parts = [str(self)]
        if self.status_code is not None:
            parts.append(f"HTTP {self.status_code}")
        if self.body:
            parts.append(self.body[:200])
        return " — ".join(parts)
