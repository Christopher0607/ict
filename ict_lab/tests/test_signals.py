from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.signals import generate_signals

WINDOW = "killzone_ny_am"  # 10:00-11:00 ET == 14:00-14:59 UTC on a non-DST-boundary date
SESSION_DATE = pd.Timestamp("2024-06-03")
WINDOW_START = pd.Timestamp("2024-06-03 14:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2024-06-03 14:59:00", tz="UTC")


def _flat_bars(start="2024-06-03 13:50:00", end="2024-06-03 15:10:00"):
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10, "contract": "X"},
        index=idx,
    )


def _store_with_stubs(df_1m, config, tick_size, **stubs):
    """Seeds a real FeatureStore's cache with fully-controlled fake detector
    outputs, using the exact cache-key shapes each method builds internally,
    so signals.py's orchestration is tested independent of whether the real
    detectors would naturally produce these exact events."""
    store = FeatureStore(df_1m)
    if "fvgs" in stubs:
        key = ("fvg", config.fvg_timeframe, config.fvg_min_size_points, config.fvg_min_size_atr_mult)
        store._cache[key] = stubs["fvgs"]
    if "sweeps" in stubs:
        key = (
            "sweeps",
            config.swing_n,
            tuple(sorted(config.sweep_level_types)),
            config.sweep_k,
            config.sweep_min_penetration_ticks,
            tick_size,
        )
        store._cache[key] = stubs["sweeps"]
    if "swings" in stubs:
        store._cache[("swings", config.swing_n)] = stubs["swings"]
    if "displacement" in stubs:
        store._cache[("displacement_atr", config.displacement_atr_mult)] = stubs["displacement"]
    if "bias" in stubs:
        key = ("bias", config.bias_method, config.bias_timeframe, config.bias_swing_n, config.bias_ma_period, config.window)
        store._cache[key] = stubs["bias"]
    return store


def _empty_fvgs():
    return pd.DataFrame(
        columns=["direction", "timeframe", "bar_index", "timestamp", "knowable_at", "gap_top", "gap_bottom", "midpoint", "size_points", "size_atr_mult", "mitigated_at", "filled_at"]
    )


def _fvg_row(direction, knowable_at, top=101.0, bottom=100.0):
    return {
        "direction": direction, "timeframe": "5m", "bar_index": 0, "timestamp": knowable_at,
        "knowable_at": knowable_at, "gap_top": top, "gap_bottom": bottom, "midpoint": (top + bottom) / 2,
        "size_points": top - bottom, "size_atr_mult": 1.0, "mitigated_at": pd.NaT, "filled_at": pd.NaT,
    }


def _empty_sweeps():
    return pd.DataFrame(columns=["level_type", "session_date", "level_price", "swept_direction", "penetration_at", "penetration_price", "confirmed_at"])


def _sweep_row(swept_direction, confirmed_at, level_type="prior_session_low", level_price=99.0):
    return {
        "level_type": level_type, "session_date": SESSION_DATE, "level_price": level_price,
        "swept_direction": swept_direction, "penetration_at": confirmed_at - pd.Timedelta(minutes=1),
        "penetration_price": level_price, "confirmed_at": confirmed_at,
    }


def _no_gates_config(**overrides):
    kwargs = dict(
        name="t", window=WINDOW, bias_method="none", sweep_required=False,
        mss_required=False, displacement_required=False, stop_type="gap_distal",
    )
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


def test_all_gates_off_takes_first_fvg_in_either_direction():
    config = _no_gates_config()
    fvgs = pd.DataFrame(
        [
            _fvg_row("bearish", WINDOW_START + pd.Timedelta(minutes=10)),
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=5)),  # earlier -> should win
        ]
    )
    store = _store_with_stubs(_flat_bars(), config, 0.25, fvgs=fvgs)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert len(no_signals) == 0
    assert len(signals) == 1
    row = signals.iloc[0]
    assert row["direction"] == "bullish"
    assert row["setup_at"] == WINDOW_START + pd.Timedelta(minutes=5)


def test_no_fvg_at_all_produces_no_fvg_reason():
    config = _no_gates_config()
    store = _store_with_stubs(_flat_bars(), config, 0.25, fvgs=_empty_fvgs())

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "no_fvg"


def test_sweep_gate_narrows_direction_and_only_fvg_after_sweep_counts():
    config = _no_gates_config(
        sweep_required=True, sweep_level_types=("prior_session_low", "prior_session_high")
    )
    sweep_ts = WINDOW_START + pd.Timedelta(minutes=10)
    sweeps = pd.DataFrame([_sweep_row("low", sweep_ts)])  # implies bullish
    fvgs = pd.DataFrame(
        [
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=2)),  # before sweep -> ignored
            _fvg_row("bearish", WINDOW_START + pd.Timedelta(minutes=15)),  # wrong direction -> ignored
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=20)),  # the one that should win
        ]
    )
    store = _store_with_stubs(_flat_bars(), config, 0.25, sweeps=sweeps, fvgs=fvgs)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert no_signals.empty
    row = signals.iloc[0]
    assert row["direction"] == "bullish"
    assert row["setup_at"] == WINDOW_START + pd.Timedelta(minutes=20)
    assert row["sweep_confirmed_at"] == sweep_ts
    assert row["sweep_level_type"] == "prior_session_low"


def test_no_sweep_produces_no_sweep_reason():
    config = _no_gates_config(sweep_required=True, sweep_level_types=("prior_session_low",))
    store = _store_with_stubs(_flat_bars(), config, 0.25, sweeps=_empty_sweeps())

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "no_sweep"


def test_sweep_outside_window_is_ignored():
    config = _no_gates_config(sweep_required=True, sweep_level_types=("prior_session_low",))
    sweeps = pd.DataFrame([_sweep_row("low", WINDOW_START - pd.Timedelta(minutes=5))])  # before window
    store = _store_with_stubs(_flat_bars(), config, 0.25, sweeps=sweeps)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "no_sweep"


def test_bias_gate_restricts_to_bullish_and_skips_earlier_bearish_fvg():
    config = _no_gates_config(bias_method="prior_day")
    bias_updates = pd.DataFrame({"as_of": [WINDOW_START - pd.Timedelta(hours=1)], "bias": ["bullish"]})
    fvgs = pd.DataFrame(
        [
            _fvg_row("bearish", WINDOW_START + pd.Timedelta(minutes=1)),
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=8)),
        ]
    )
    store = _store_with_stubs(_flat_bars(), config, 0.25, bias=bias_updates, fvgs=fvgs)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert no_signals.empty
    row = signals.iloc[0]
    assert row["direction"] == "bullish"
    assert row["bias_value"] == "bullish"


def test_bias_gate_none_value_fails_with_bias_gate_reason():
    config = _no_gates_config(bias_method="prior_day")
    bias_updates = pd.DataFrame(columns=["as_of", "bias"])  # no updates -> projects to "none"
    store = _store_with_stubs(_flat_bars(), config, 0.25, bias=bias_updates, fvgs=_empty_fvgs())

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "bias_gate"


def test_displacement_gate_ignores_fvg_before_displacement_bar():
    config = _no_gates_config(displacement_required=True)
    disp_at = WINDOW_START + pd.Timedelta(minutes=12)
    disp_index = pd.date_range(WINDOW_START, WINDOW_END, freq="1min", tz="UTC")
    displacement = pd.Series(False, index=disp_index)
    displacement.loc[disp_at] = True
    fvgs = pd.DataFrame(
        [
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=3)),  # before displacement -> ignored
            _fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=20)),  # after -> wins
        ]
    )
    store = _store_with_stubs(_flat_bars(), config, 0.25, displacement=displacement, fvgs=fvgs)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert no_signals.empty
    row = signals.iloc[0]
    assert row["setup_at"] == WINDOW_START + pd.Timedelta(minutes=20)
    assert row["displacement_at"] == disp_at


def test_no_displacement_produces_no_displacement_reason():
    config = _no_gates_config(displacement_required=True)
    disp_index = pd.date_range(WINDOW_START, WINDOW_END, freq="1min", tz="UTC")
    displacement = pd.Series(False, index=disp_index)
    store = _store_with_stubs(_flat_bars(), config, 0.25, displacement=displacement, fvgs=_empty_fvgs())

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "no_displacement"


def test_mss_gate_uses_real_price_break_and_narrows_direction():
    # Real bars this time: MSS uses store.df_1m + swings directly, not a stub.
    idx = pd.date_range("2024-06-03 13:50:00", "2024-06-03 15:10:00", freq="1min", tz="UTC", inclusive="left")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10, "contract": "X"}, index=idx
    )
    break_at = WINDOW_START + pd.Timedelta(minutes=15)
    df.loc[break_at, ["high", "close"]] = [102.0, 101.8]  # breaks above the swing high of 101

    config = _no_gates_config(mss_required=True)
    sweep_ts = WINDOW_START + pd.Timedelta(minutes=10)
    swings = pd.DataFrame(
        [
            {
                "kind": "high", "bar_index": 0, "timestamp": WINDOW_START - pd.Timedelta(minutes=30),
                "price": 101.0, "confirmed_at_index": 0, "confirmed_at": WINDOW_START - pd.Timedelta(minutes=25),
            }
        ]
    )
    fvgs = pd.DataFrame([_fvg_row("bullish", break_at + pd.Timedelta(minutes=2))])
    store = _store_with_stubs(df, config, 0.25, swings=swings, fvgs=fvgs)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert no_signals.empty
    row = signals.iloc[0]
    assert row["mss_broken_at"] == break_at
    assert row["direction"] == "bullish"


def test_no_mss_produces_no_mss_reason():
    df = _flat_bars()  # never breaks 101 -- flat at 100
    config = _no_gates_config(mss_required=True)
    swings = pd.DataFrame(
        [
            {
                "kind": "high", "bar_index": 0, "timestamp": WINDOW_START - pd.Timedelta(minutes=30),
                "price": 101.0, "confirmed_at_index": 0, "confirmed_at": WINDOW_START - pd.Timedelta(minutes=25),
            }
        ]
    )
    store = _store_with_stubs(df, config, 0.25, swings=swings)

    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty
    assert no_signals.iloc[0]["reason"] == "no_mss"


def test_at_most_one_signal_row_per_session():
    config = _no_gates_config()
    fvgs = pd.DataFrame(
        [_fvg_row("bullish", WINDOW_START + pd.Timedelta(minutes=t)) for t in (2, 5, 8, 12)]
    )
    store = _store_with_stubs(_flat_bars(), config, 0.25, fvgs=fvgs)

    signals, _ = generate_signals(store, config, tick_size=0.25)
    assert len(signals[signals["session_date"] == SESSION_DATE]) == 1


def test_sessions_without_window_bars_produce_no_row_at_all():
    config = _no_gates_config()
    # Data that never touches killzone_ny_am (10:00-11:00 ET) at all.
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10, "contract": "X"},
        index=pd.date_range("2024-06-03 20:00:00", "2024-06-03 21:00:00", freq="1min", tz="UTC"),
    )
    store = FeatureStore(df)
    signals, no_signals = generate_signals(store, config, tick_size=0.25)
    assert signals.empty and no_signals.empty
