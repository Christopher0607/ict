"""Phase 5 Part 0 (items 1-2): grid enumeration, canonicalization, and
dedup. configs/grid.json's "sweep_axes" is the combinatorial grid (window
subsets, bias_method excluding perfect, sweep_required/universe,
mss_required, displacement_required, fvg_timeframe, entry_level, stop_type,
target_type, max_trades_per_window); "sweep_fixed" holds the dial
parameters we chose not to sweep (sweep_k, penetration ticks, displacement
ATR multiple, R multiples, swing_n/swing_15m_n, stop buffer, bias
sub-parameters, FVG min-size) to keep the grid tractable -- the spec names
only "sweep universe preset, window list, max_trades_per_window" explicitly
as axes; the rest of our choices are documented here rather than silently
assumed.

Scope decisions, since the spec doesn't spell out the full axis list:
- bias=perfect is excluded from sweep_axes entirely per explicit
  instruction ("Exclude bias=perfect from the sweep entirely; it exists
  only in the ablation ladder, labeled as lookahead").
- stop_type is limited to swing/gap_distal (excludes fixed_points) and
  target_type to fixed_r/next_liquidity (excludes time): both excluded
  options need their own numeric sub-parameter (stop_fixed_points,
  target_time_bars) which would add a whole further combinatorial
  dimension the spec doesn't call for. They remain usable manually
  (ablation ladder, named configs) but aren't part of the automated sweep.
- max_trades_per_window is swept over {1, 10}, not every integer 1-10,
  matching how Phase 4 itself frames the field: "1 (current behavior) or
  unlimited with a hard safety cap of 10" -- a binary choice, not a
  continuous one.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

from ict_lab.configs.grid import GRID_PATH, load_grid
from ict_lab.configs.strategy_config import StrategyConfig

_AXIS_ORDER = (
    "windows", "bias_method", "sweep_required", "sweep_universe", "mss_required",
    "displacement_required", "fvg_timeframe", "entry_level", "stop_type", "target_type",
    "max_trades_per_window",
)


def load_sweep_axes(path: Path = GRID_PATH) -> dict:
    return load_grid(path)["sweep_axes"]


def load_sweep_fixed(path: Path = GRID_PATH) -> dict:
    return load_grid(path)["sweep_fixed"]


def _raw_kwargs(axes: dict) -> list[dict]:
    keys = [k for k in _AXIS_ORDER if k in axes]
    value_lists = [axes[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        kwargs = dict(zip(keys, values))
        if "windows" in kwargs:
            kwargs["windows"] = tuple(kwargs["windows"])
        combos.append(kwargs)
    return combos


def enumerate_raw_configs(axes: dict | None = None, fixed: dict | None = None) -> tuple[list[StrategyConfig], int]:
    """Returns (valid configs, invalid_count). Invalid combos (e.g.
    stop_type='swing' with sweep_required=False) are exactly the ones
    StrategyConfig's own __post_init__ already rejects -- we don't
    duplicate that validity logic, we just catch it."""
    axes = axes if axes is not None else load_sweep_axes()
    fixed = fixed if fixed is not None else load_sweep_fixed()

    valid: list[StrategyConfig] = []
    invalid_count = 0
    for i, kwargs in enumerate(_raw_kwargs(axes)):
        kwargs.update(fixed)
        try:
            valid.append(StrategyConfig(name=f"sweep_raw_{i:06d}", **kwargs))
        except ValueError:
            invalid_count += 1
    return valid, invalid_count


def canonicalize(config: StrategyConfig) -> StrategyConfig:
    """Nulls parameters that have no effect given the rest of a config's
    state, so two configs differing only in a dead parameter collapse to
    the same canonical form. Written against field values generically
    (not just the specific axis choices above) so it stays correct if the
    swept axes ever change."""
    updates = {}
    if not config.sweep_required:
        updates.update(sweep_universe=None, sweep_level_types=(), sweep_k=0, sweep_min_penetration_ticks=0.0)
    if config.target_type != "next_liquidity":
        updates["target_fallback_r_multiple"] = 0.0
    if config.target_type != "time":
        updates["target_time_bars"] = 0
    if config.stop_type != "fixed_points":
        updates["stop_fixed_points"] = 0.0
    if config.bias_method != "swing_structure":
        updates["bias_timeframe"] = ""
        updates["bias_swing_n"] = 0
    if config.bias_method != "daily_ma_slope":
        updates["bias_ma_period"] = 0
    return replace(config, **updates) if updates else config


def canonical_hash(config: StrategyConfig) -> str:
    """Stable content hash excluding `name` -- two configs with identical
    parameters but different labels must dedup to the same entry."""
    d = config.to_dict()
    d.pop("name", None)
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_canonical_population(axes: dict | None = None, fixed: dict | None = None) -> tuple[list[StrategyConfig], dict]:
    """Returns (canonical configs, report). report has raw_valid,
    raw_invalid, and canonical counts, per Part 0 item 2's "report raw vs
    canonical counts"."""
    raw_valid, raw_invalid = enumerate_raw_configs(axes, fixed)

    seen: dict[str, StrategyConfig] = {}
    for config in raw_valid:
        canonical = canonicalize(config)
        h = canonical_hash(canonical)
        if h not in seen:
            seen[h] = replace(canonical, name=f"sweep_{h}")

    report = {"raw_valid": len(raw_valid), "raw_invalid": raw_invalid, "canonical": len(seen)}
    return list(seen.values()), report
