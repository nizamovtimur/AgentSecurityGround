"""Prompt loading and output parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_attack_prompt(text: str) -> str:
    match = re.search(r">>>ATTACK\s*(.*?)\s*<<<ATTACK", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_selected_strategy(text: str) -> str:
    match = re.search(r"SELECTED STRATEGY:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return "Custom"


def extract_action_text(text: str) -> str:
    match = re.search(r"ACTION:\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_score(text: str) -> float:
    score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9])?)", text, flags=re.IGNORECASE)
    if score_match:
        return float(score_match.group(1))
    return 1.0


def parse_strategy_payload(text: str) -> dict[str, str]:
    def _extract(label: str) -> str:
        pattern = rf"{label}:\s*\"(.*?)\""
        match = re.search(pattern, text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    return {
        "strategy": _extract("Strategy"),
        "definition": _extract("Definition"),
        "representation": _extract("Representation"),
        "interaction_pattern": _extract("Interaction Pattern"),
    }

