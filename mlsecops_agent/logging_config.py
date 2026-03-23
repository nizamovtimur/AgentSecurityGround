"""
Настройка логирования: консоль + файл в директории артефактов при каждом запуске.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(artifacts_path=None, debug_level=1):
    """
    Настраивает логирование: консоль (INFO) + файл (DEBUG) в run-директории.

    Создаёт ./artifacts/MLSECOPS_run_YYYY-MM-DD_HH-MM-SS/ и log.txt.

    Args:
        artifacts_path: Базовая директория (по умолчанию ./artifacts)
        debug_level: 0=WARNING, 1=INFO, 2=DEBUG

    Returns:
        Path к run-директории (artifacts_path/MLSECOPS_run_...)
    """
    base = Path(artifacts_path or os.environ.get("MLSECOPS_ARTIFACTS_PATH", "./artifacts"))
    run_name = "MLSECOPS_run_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "log.txt"
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(debug_level, 2)]

    root = logging.getLogger("mlsecops_agent")
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return run_dir


def get_run_dir():
    """Возвращает текущую run-директорию (если настроена) или None."""
    root = logging.getLogger("mlsecops_agent")
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and hasattr(h, "baseFilename"):
            return str(Path(h.baseFilename).parent)
    return None


def get_logger(name):
    """Возвращает логгер для модуля."""
    return logging.getLogger("mlsecops_agent.{}".format(name) if name else "mlsecops_agent")
