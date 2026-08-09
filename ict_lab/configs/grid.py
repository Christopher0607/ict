"""Phase 4 item 4: loads configs/grid.json's named configs. These replace
Phase 3's single CONSENSUS_CONFIG placeholder as the reference set going
forward -- CONSENSUS_CONFIG stays defined (Phase 3's own verification and
tests still pin its exact values) but new work should use as_taught_5m,
as_taught_1m, or as_traded instead.

as_traded is a default placeholder (as_taught_5m with MSS required) until
the user replaces it with their own discretionary parameters -- see
PLACEHOLDER_NOTES and grid.json's "placeholders" section. Every output
derived from it must carry that label so it's never mistaken for a verified
taught definition.
"""
from __future__ import annotations

import json
from pathlib import Path

from ict_lab.configs.strategy_config import StrategyConfig

GRID_PATH = Path(__file__).parent / "grid.json"

_TUPLE_FIELDS = ("windows", "sweep_level_types")


def load_grid(path: Path = GRID_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _to_strategy_config(name: str, raw: dict) -> StrategyConfig:
    kwargs = {k: (tuple(v) if k in _TUPLE_FIELDS else v) for k, v in raw.items()}
    return StrategyConfig(name=name, **kwargs)


def load_named_configs(path: Path = GRID_PATH) -> dict[str, StrategyConfig]:
    grid = load_grid(path)
    return {name: _to_strategy_config(name, raw) for name, raw in grid["named_configs"].items()}


def placeholder_note(name: str, path: Path = GRID_PATH) -> str | None:
    """None for a fully user-specified config; a disclaimer string for a
    stand-in like as_traded that must be flagged wherever it's used."""
    return load_grid(path).get("placeholders", {}).get(name)
