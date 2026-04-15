"""
Загрузчик промптов из файлов.

Все промпты хранятся в mlsecops-agent/prompts/ для удобной замены и настройки.
Модель угроз: MAESTRO (prompts/threat_model.txt).
"""

import json
from pathlib import Path


def _prompts_dir():
    """Возвращает путь к директории mlsecops-agent/prompts/."""
    return Path(__file__).parent / "prompts"


def load_prompt(name, **kwargs):
    """
    Загружает промпт из файла и подставляет плейсхолдеры.

    Плейсхолдеры в формате {key} заменяются на kwargs[key].
    Файлы ищутся в mlsecops-agent/prompts/.

    Args:
        name: Имя файла (например, "boss.txt", "goal_generator.txt")
        **kwargs: Значения для подстановки {placeholder} в тексте

    Returns:
        Текст промпта с подставленными значениями. Пустая строка если файл не найден.
    """
    name = str(name) if name else ""
    if not name:
        return ""
    path = _prompts_dir() / name
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def load_threat_model():
    """
    Загружает модель угроз из prompts/threat_model.txt.
    """
    return load_prompt("threat_model.txt")


def load_attack_strategies(path=None):
    """
    Загружает библиотеку стратегий атаки из JSON.

    Формат: [{"strategy": "Name", "definition": "..."}, ...]
    Используется Boss и Attacker для выбора и применения стратегий.

    Args:
        path: Путь к JSON. По умолчанию prompts/attack_strategies.json

    Returns:
        Список dict с ключами strategy, definition
    """
    p = path or _prompts_dir() / "attack_strategies.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []
