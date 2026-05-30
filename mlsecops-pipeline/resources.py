"""Раннее связывание корпоративных артефактов (загрузка при импорте модуля)."""

from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

CORPORATE_THREAT_MODEL: str = (_PKG_DIR / "assets" / "threat_model.txt").read_text(encoding="utf-8")
SENSITIVE_DATA_CATEGORIES: str = (_PKG_DIR / "assets" / "sensetive-data.txt").read_text(encoding="utf-8")
