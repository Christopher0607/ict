from __future__ import annotations

import pandas as pd

from ict_lab.engine.feature_store import FeatureStore
from ict_lab.features.fvg import detect_fvg
from ict_lab.features.swings import swing_points


def test_repeated_call_returns_cached_object_not_recomputed(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    store = FeatureStore(df)

    a = store.fvgs("5m")
    b = store.fvgs("5m")
    assert a is b  # identity, not just equality -- proves it wasn't recomputed


def test_different_params_are_not_conflated(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    store = FeatureStore(df)

    a = store.fvgs("5m")
    b = store.fvgs("15m")
    assert a is not b


def test_fvgs_matches_direct_call(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    store = FeatureStore(df)
    pd.testing.assert_frame_equal(store.fvgs("5m", min_size_points=1.0), detect_fvg(df, "5m", 1.0, 0.0))


def test_swings_matches_direct_call(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    store = FeatureStore(df)
    pd.testing.assert_frame_equal(store.swings(5), swing_points(df, 5))


def test_sweeps_reuses_cached_liquidity_levels(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-05 09:00:00")
    store = FeatureStore(df)

    levels_first = store.liquidity_levels(5)
    store.sweeps(5, ("prior_session_high", "prior_session_low"), 3, 1.0, 0.25)
    levels_after = store.liquidity_levels(5)
    assert levels_first is levels_after  # sweeps() didn't trigger a second liquidity computation


def test_sweeps_level_types_order_does_not_create_separate_cache_entries(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-05 09:00:00")
    store = FeatureStore(df)

    a = store.sweeps(5, ("prior_session_high", "prior_session_low"), 3, 1.0, 0.25)
    b = store.sweeps(5, ("prior_session_low", "prior_session_high"), 3, 1.0, 0.25)
    assert a is b


def test_liquidity_levels_swing_15m_n_is_not_conflated(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-10 09:00:00")
    store = FeatureStore(df)

    a = store.liquidity_levels(5, swing_15m_n=3)
    b = store.liquidity_levels(5, swing_15m_n=7)
    assert a is not b
    a_15m = set(a[a["level_type"].isin(["swing_high_15m", "swing_low_15m"])]["price"])
    b_15m = set(b[b["level_type"].isin(["swing_high_15m", "swing_low_15m"])]["price"])
    assert a_15m != b_15m  # different swing_15m_n must actually reach all_liquidity_levels, not be ignored


def test_sweeps_swing_15m_n_is_not_conflated(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-10 09:00:00")
    store = FeatureStore(df)

    a = store.sweeps(5, ("swing_high_15m", "swing_low_15m"), 3, 1.0, 0.25, swing_15m_n=3)
    b = store.sweeps(5, ("swing_high_15m", "swing_low_15m"), 3, 1.0, 0.25, swing_15m_n=7)
    assert a is not b


def test_bias_updates_none_and_prior_day(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    store = FeatureStore(df)
    assert store.bias_updates("none").empty
    assert not store.bias_updates("prior_day").empty


def test_bias_updates_perfect_requires_window(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    store = FeatureStore(df)
    try:
        store.bias_updates("perfect")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert not store.bias_updates("perfect", window="killzone_ny_am").empty
