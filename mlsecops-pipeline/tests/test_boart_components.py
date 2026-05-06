from __future__ import annotations

import pandas as pd

from boart.goal_loader import load_attack_targets
from boart.prompt_utils import extract_attack_prompt, extract_score, parse_strategy_payload
from boart.strategy_library import StrategyLibrary

from conftest import PIPELINE_ROOT


def test_prompt_extractors() -> None:
    text = "THOUGHT: test\nACTION:\n>>>ATTACK\nhello\n<<<ATTACK"
    assert extract_attack_prompt(text) == "hello"
    assert extract_score("Score: 6.5") == 6.5


def test_parse_strategy_payload() -> None:
    payload = parse_strategy_payload(
        'Strategy: "New Strategy"\n'
        'Definition: "desc"\n'
        'Representation: "narrative"\n'
        'Interaction Pattern: "single-shot"\n'
    )
    assert payload["strategy"] == "New Strategy"
    assert payload["representation"] == "narrative"


def test_strategy_library_update_and_trim() -> None:
    library = StrategyLibrary.from_json(
        PIPELINE_ROOT / "prompts" / "attack_strategies.json", max_size=3
    )
    assert len(library.strategies) == 3
    library.update_metrics("Context Expansion and Camouflage", 7.0)
    assert library.ranked()[0].attempts >= 1
    library.add_generated_strategy(
        {
            "strategy": "Generated Strategy",
            "definition": "generated",
            "representation": "mixed",
            "interaction_pattern": "adaptive",
        }
    )
    assert len(library.strategies) == 3


def test_goal_loader_filters_language(tmp_path: str) -> None:
    from pathlib import Path

    df = pd.DataFrame(
        [
            {"goal": "ru goal", "language": "ru"},
            {"goal": "en goal", "language": "en"},
        ]
    )
    dataset_path = Path(tmp_path) / "sample.parquet"
    df.to_parquet(dataset_path, index=False)
    targets = load_attack_targets(Path(tmp_path), ["sample"], goals_per_attack=2, language="ru")
    assert len(targets) == 1
    assert targets[0].goal == "ru goal"
