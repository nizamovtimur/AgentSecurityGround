"""Загрузка целей атак из datasets/*.parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from boart.models import AttackTarget


def load_attack_targets(
    datasets_dir: str | Path,
    attacks: list[str],
    goals_per_attack: int = 3,
    language: str = "ru",
) -> list[AttackTarget]:
    base = Path(datasets_dir)
    selected: list[AttackTarget] = []
    for attack_name in attacks:
        dataset_path = base / f"{attack_name}.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
        frame = pd.read_parquet(dataset_path)
        if "goal" not in frame.columns:
            raise ValueError(f"В {dataset_path} нужна колонка 'goal'.")
        if language in {"ru", "en"} and "language" in frame.columns:
            filtered = frame[frame["language"].fillna("").str.lower() == language]
            if not filtered.empty:
                frame = filtered
        frame = frame.head(goals_per_attack)
        for _, row in frame.iterrows():
            selected.append(
                AttackTarget(
                    attack_name=attack_name,
                    goal=str(row["goal"]),
                    metadata={col: str(row[col]) for col in frame.columns if col != "goal"},
                )
            )
    return selected
