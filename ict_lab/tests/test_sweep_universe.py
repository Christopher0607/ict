from __future__ import annotations

import pytest

from ict_lab.configs.sweep_universe import resolve_sweep_universe, resolve_sweep_universe_for_windows


def test_session_refs():
    assert resolve_sweep_universe("session_refs", "killzone_ny_am") == (
        "prior_session_high", "prior_session_low", "pre_killzone_ny_am_high", "pre_killzone_ny_am_low",
    )


def test_session_refs_plus_swings_adds_1m_swings():
    result = resolve_sweep_universe("session_refs_plus_swings", "killzone_london")
    assert set(result) == {
        "prior_session_high", "prior_session_low",
        "pre_killzone_london_high", "pre_killzone_london_low",
        "swing_high", "swing_low",
    }


def test_bsl_ssl_15m_uses_15m_swings_not_1m():
    result = resolve_sweep_universe("bsl_ssl_15m", "killzone_ny_pm")
    assert "swing_high_15m" in result and "swing_low_15m" in result
    assert "swing_high" not in result and "swing_low" not in result
    assert "pre_killzone_ny_pm_high" in result


def test_swings_only_excludes_session_refs():
    result = resolve_sweep_universe("swings_only", "killzone_ny_am")
    assert set(result) == {"swing_high", "swing_low"}


def test_presets_differ_by_window():
    a = resolve_sweep_universe("session_refs", "killzone_ny_am")
    b = resolve_sweep_universe("session_refs", "killzone_london")
    assert a != b


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown sweep universe preset"):
        resolve_sweep_universe("bogus", "killzone_ny_am")


def test_resolve_for_windows_unions_across_all_windows():
    result = resolve_sweep_universe_for_windows("session_refs", ("killzone_ny_am", "killzone_london"))
    assert set(result) == {
        "prior_session_high", "prior_session_low",
        "pre_killzone_ny_am_high", "pre_killzone_ny_am_low",
        "pre_killzone_london_high", "pre_killzone_london_low",
    }


def test_resolve_for_windows_single_window_matches_resolve_sweep_universe():
    assert set(resolve_sweep_universe_for_windows("bsl_ssl_15m", ("killzone_ny_pm",))) == set(
        resolve_sweep_universe("bsl_ssl_15m", "killzone_ny_pm")
    )
