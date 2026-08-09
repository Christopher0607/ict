"""Named sweep-universe presets (Phase 4 item 3): which liquidity level
types a sweep (or a next_liquidity target) considers. Resolved per-window
since pre-window level types are window-specific (e.g. pre_killzone_ny_am_high
vs pre_killzone_london_high).
"""
from __future__ import annotations

PRESETS = ("session_refs", "session_refs_plus_swings", "bsl_ssl_15m", "swings_only")

_SESSION_REF_TYPES = ("prior_session_high", "prior_session_low")
_SWING_1M_TYPES = ("swing_high", "swing_low")
_SWING_15M_TYPES = ("swing_high_15m", "swing_low_15m")


def resolve_sweep_universe(preset: str, window: str) -> tuple[str, ...]:
    if preset == "swings_only":
        return _SWING_1M_TYPES

    pre_types = (f"pre_{window}_high", f"pre_{window}_low")
    if preset == "session_refs":
        return _SESSION_REF_TYPES + pre_types
    if preset == "session_refs_plus_swings":
        return _SESSION_REF_TYPES + pre_types + _SWING_1M_TYPES
    if preset == "bsl_ssl_15m":
        return _SESSION_REF_TYPES + pre_types + _SWING_15M_TYPES

    raise ValueError(f"unknown sweep universe preset {preset!r}. Choose from {PRESETS}.")


def resolve_sweep_universe_for_windows(preset: str, windows: tuple[str, ...]) -> tuple[str, ...]:
    """Union of resolve_sweep_universe(preset, w) across every w in windows,
    for callers that compute one levels/sweeps frame up front to cover a
    multi-window config's full window set (e.g. a verification script),
    rather than resolving per-window the way the signal/execution engine
    does internally."""
    return tuple(sorted({t for w in windows for t in resolve_sweep_universe(preset, w)}))
