from __future__ import annotations

import math

import pandas as pd
import pytest

from ict_lab.configs.strategy_config import COST_MODELS, StrategyConfig
from ict_lab.engine.execution import (
    NO_TRADE_COLUMNS,
    TRADE_COLUMNS,
    _entry_price,
    _mae_mfe,
    _simulate_entry,
    _simulate_exit,
    _snap_to_tick,
    _stop_price,
    _target_price,
    simulate_trades,
)


def _bars(rows, start="2024-06-03 14:00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    if "volume" not in df:
        df["volume"] = 10
    if "contract" not in df:
        df["contract"] = "X"
    return df


# ---------- tick snapping / entry price ----------


def test_snap_to_tick_rounds_to_nearest():
    assert _snap_to_tick(100.13, 0.25) == 100.25
    assert _snap_to_tick(100.10, 0.25) == 100.0
    assert _snap_to_tick(100.125, 0.25) in (100.0, 100.25)  # exact tie, either is defensible


@pytest.mark.parametrize(
    "direction,entry_level,expected",
    [
        ("bullish", "proximal", 102.0),  # top
        ("bullish", "distal", 100.0),  # bottom
        ("bullish", "50%", 101.0),
        ("bearish", "proximal", 100.0),  # bottom
        ("bearish", "distal", 102.0),  # top
        ("bearish", "50%", 101.0),
    ],
)
def test_entry_price_by_direction_and_level(direction, entry_level, expected):
    assert _entry_price(direction, entry_level, fvg_top=102.0, fvg_bottom=100.0, tick_size=0.25) == expected


# ---------- entry fill simulation ----------


def test_bullish_entry_fills_on_strict_through_not_touch():
    rows = [
        {"open": 101, "high": 101.2, "low": 101.0, "close": 101.1},  # touches 101 exactly -> no fill
        {"open": 100.9, "high": 101.0, "low": 100.8, "close": 100.9},  # trades through -> fills
    ]
    df = _bars(rows)
    setup_at = df.index[0] - pd.Timedelta(minutes=1)
    filled = _simulate_entry("bullish", 101.0, setup_at, df.index[-1], df)
    assert filled == df.index[1]


def test_bearish_entry_fills_on_strict_through():
    rows = [
        {"open": 101, "high": 101.0, "low": 100.9, "close": 100.95},  # touches -> no fill
        {"open": 101.1, "high": 101.2, "low": 101.0, "close": 101.1},  # trades through -> fills
    ]
    df = _bars(rows)
    setup_at = df.index[0] - pd.Timedelta(minutes=1)
    filled = _simulate_entry("bearish", 101.0, setup_at, df.index[-1], df)
    assert filled == df.index[1]


def test_entry_unfilled_by_window_end_returns_none():
    rows = [{"open": 105, "high": 105.2, "low": 104.9, "close": 105.1} for _ in range(3)]
    df = _bars(rows)
    setup_at = df.index[0] - pd.Timedelta(minutes=1)
    assert _simulate_entry("bullish", 100.0, setup_at, df.index[-1], df) is None


def test_entry_scan_starts_strictly_after_setup_bar():
    # A bar AT setup_at that would fill must be ignored -- entry starts the
    # bar *after* setup confirmation.
    rows = [
        {"open": 100.9, "high": 101.0, "low": 100.5, "close": 100.9},  # would fill, but this IS setup_at
        {"open": 101.0, "high": 101.5, "low": 101.0, "close": 101.2},  # touches only, doesn't fill either
    ]
    df = _bars(rows)
    filled = _simulate_entry("bullish", 101.0, df.index[0], df.index[-1], df)
    assert filled is None


# ---------- stop price ----------


def test_stop_swing_uses_swept_level_with_buffer():
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        stop_buffer_ticks=2,
    )
    stop = _stop_price(
        "bullish", entry_price=101.0, config=config, tick_size=0.25, fvg_top=102, fvg_bottom=100,
        swept_level_price=99.0,
    )
    assert stop == 99.0 - 2 * 0.25


def test_stop_swing_without_swept_level_raises():
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",)
    )
    with pytest.raises(ValueError, match="swept level"):
        _stop_price("bullish", 101.0, config, 0.25, 102, 100, swept_level_price=None)


def test_stop_gap_distal_bearish():
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, stop_buffer_ticks=1)
    stop = _stop_price("bearish", entry_price=101.0, config=config, tick_size=0.25, fvg_top=102, fvg_bottom=100, swept_level_price=None)
    assert stop == 102 + 1 * 0.25


def test_stop_fixed_points_bullish():
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="fixed_points", stop_fixed_points=5, sweep_required=False)
    stop = _stop_price("bullish", entry_price=100.0, config=config, tick_size=0.25, fvg_top=102, fvg_bottom=100, swept_level_price=None)
    assert stop == 95.0


# ---------- target price ----------


def test_target_fixed_r_bullish_and_bearish():
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, target_type="fixed_r", target_r_multiple=2)
    price, time_bars = _target_price("bullish", entry_price=100.0, stop_price=98.0, config=config, tick_size=0.25, levels=pd.DataFrame(), sweeps=pd.DataFrame(), entry_at=pd.Timestamp("2024-06-03", tz="UTC"))
    assert price == 104.0 and time_bars is None

    price2, _ = _target_price("bearish", entry_price=100.0, stop_price=102.0, config=config, tick_size=0.25, levels=pd.DataFrame(), sweeps=pd.DataFrame(), entry_at=pd.Timestamp("2024-06-03", tz="UTC"))
    assert price2 == 96.0


def test_target_time_returns_bars_not_price():
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, target_type="time", target_time_bars=5)
    price, time_bars = _target_price("bullish", 100.0, 98.0, config, 0.25, pd.DataFrame(), pd.DataFrame(), pd.Timestamp("2024-06-03", tz="UTC"))
    assert price is None and time_bars == 5


def test_target_next_liquidity_picks_nearest_unswept_opposing_level():
    entry_at = pd.Timestamp("2024-06-03 14:10:00", tz="UTC")
    levels = pd.DataFrame(
        [
            {"level_type": "prior_session_high", "session_date": pd.Timestamp("2024-06-03"), "price": 103.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
            {"level_type": "swing_high", "session_date": pd.Timestamp("2024-06-03"), "price": 105.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
            {"level_type": "swing_low", "session_date": pd.Timestamp("2024-06-03"), "price": 90.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
        ]
    )
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, target_type="next_liquidity")
    price, time_bars = _target_price("bullish", 100.0, 98.0, config, 0.25, levels, pd.DataFrame(), entry_at)
    assert price == 103.0 and time_bars is None  # nearest high-type level above entry, not the farther 105


def test_target_next_liquidity_skips_already_swept_level():
    entry_at = pd.Timestamp("2024-06-03 14:10:00", tz="UTC")
    levels = pd.DataFrame(
        [
            {"level_type": "prior_session_high", "session_date": pd.Timestamp("2024-06-03"), "price": 103.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
            {"level_type": "swing_high", "session_date": pd.Timestamp("2024-06-03"), "price": 105.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
        ]
    )
    sweeps = pd.DataFrame(
        [{"level_type": "prior_session_high", "level_price": 103.0, "confirmed_at": entry_at - pd.Timedelta(minutes=1)}]
    )
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, target_type="next_liquidity")
    price, _ = _target_price("bullish", 100.0, 98.0, config, 0.25, levels, sweeps, entry_at)
    assert price == 105.0


def test_target_next_liquidity_respects_target_level_types_narrowing():
    # Phase 4 item 3: "next opposing liquidity must draw from the same
    # preset the sweep uses" -- target_level_types narrows the candidate
    # pool even though the raw levels frame has other opposing-type levels.
    entry_at = pd.Timestamp("2024-06-03 14:10:00", tz="UTC")
    levels = pd.DataFrame(
        [
            {"level_type": "prior_session_high", "session_date": pd.Timestamp("2024-06-03"), "price": 103.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
            {"level_type": "swing_high", "session_date": pd.Timestamp("2024-06-03"), "price": 105.0, "knowable_at": entry_at - pd.Timedelta(minutes=5)},
        ]
    )
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False, target_type="next_liquidity")

    price, _ = _target_price("bullish", 100.0, 98.0, config, 0.25, levels, pd.DataFrame(), entry_at)
    assert price == 103.0  # no narrowing -> nearest of either type wins

    price2, _ = _target_price(
        "bullish", 100.0, 98.0, config, 0.25, levels, pd.DataFrame(), entry_at,
        target_level_types=("swing_high", "swing_low"),
    )
    assert price2 == 105.0  # prior_session_high excluded -> falls through to swing_high

    price3, _ = _target_price(
        "bullish", 100.0, 98.0, config, 0.25, levels, pd.DataFrame(), entry_at,
        target_level_types=("swing_low",),
    )
    assert price3 == 100.0 + config.target_fallback_r_multiple * 2.0  # no member qualifies -> R fallback


def test_target_next_liquidity_falls_back_to_r_multiple_when_none_qualify():
    entry_at = pd.Timestamp("2024-06-03 14:10:00", tz="UTC")
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), stop_type="gap_distal", sweep_required=False,
        target_type="next_liquidity", target_fallback_r_multiple=3,
    )
    price, _ = _target_price("bullish", 100.0, 98.0, config, 0.25, pd.DataFrame(columns=["level_type", "price", "knowable_at"]), pd.DataFrame(), entry_at)
    assert price == 106.0  # entry + 3 * (100-98)


# ---------- exit simulation ----------


def test_exit_stop_only():
    rows = [{"open": 100, "high": 100.2, "low": 99.4, "close": 99.8}]  # low breaches stop 99.5
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=99.5, target_price=105.0, target_time_bars=None, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "stop"
    assert result["ambiguous_bar"] is False
    assert result["exit_price"] == 99.25  # stop (on-grid) minus 1 tick of slippage


def test_exit_target_only_no_slippage():
    rows = [{"open": 100, "high": 105.2, "low": 99.9, "close": 105.0}]
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=95.0, target_price=105.0, target_time_bars=None, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "target"
    assert result["exit_price"] == 105.0  # exact, no slippage


def test_exit_ambiguous_bar_stop_always_wins():
    rows = [{"open": 100, "high": 105.2, "low": 94.9, "close": 100}]  # crosses both stop and target
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=95.0, target_price=105.0, target_time_bars=None, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "stop"
    assert result["ambiguous_bar"] is True


def test_exit_hard_exit_when_nothing_triggers():
    rows = [{"open": 100, "high": 100.5, "low": 99.7, "close": 100.2} for _ in range(3)]
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=90.0, target_price=110.0, target_time_bars=None, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "hard_exit"
    assert result["exit_at"] == df.index[-1]
    assert result["exit_price"] == df["close"].iloc[-1]
    assert result["ambiguous_bar"] is False


def test_exit_target_time_exits_at_nth_bar_close():
    rows = [{"open": 100, "high": 100.5, "low": 99.7, "close": 100 + i * 0.1} for i in range(5)]
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=90.0, target_price=110.0, target_time_bars=3, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "target_time"
    assert result["exit_at"] == df.index[2]  # 3rd bar after entry
    assert result["exit_price"] == df["close"].iloc[2]


def test_exit_price_target_beats_time_target_at_same_bar():
    rows = [
        {"open": 100, "high": 100.5, "low": 99.7, "close": 100.1},
        {"open": 100.1, "high": 105.2, "low": 100.0, "close": 105.0},  # target hit AND is the 2nd bar
    ]
    df = _bars(rows)
    entry_at = df.index[0] - pd.Timedelta(minutes=1)
    result = _simulate_exit("bullish", entry_at, 100.0, stop_price=90.0, target_price=105.0, target_time_bars=2, hard_exit_at=df.index[-1], session_bars=df, tick_size=0.25, stop_slippage_ticks=1)
    assert result["exit_reason"] == "target"


# ---------- MAE / MFE ----------


def test_mae_mfe_bullish():
    rows = [
        {"open": 100, "high": 100.5, "low": 98.5, "close": 99},  # adverse dip to 98.5
        {"open": 99, "high": 103.0, "low": 99, "close": 102},  # favorable run to 103
    ]
    df = _bars(rows)
    mae, mfe = _mae_mfe("bullish", 100.0, df.index[0], df.index[-1], df)
    assert mae == pytest.approx(1.5)
    assert mfe == pytest.approx(3.0)


def test_mae_mfe_bearish():
    rows = [
        {"open": 100, "high": 101.5, "low": 99.5, "close": 101},  # adverse rise to 101.5
        {"open": 101, "high": 101, "low": 97.0, "close": 98},  # favorable drop to 97
    ]
    df = _bars(rows)
    mae, mfe = _mae_mfe("bearish", 100.0, df.index[0], df.index[-1], df)
    assert mae == pytest.approx(1.5)
    assert mfe == pytest.approx(3.0)


# ---------- full simulate_trades integration ----------


def _signal_row(**overrides):
    row = dict(
        session_date=pd.Timestamp("2024-06-03"), window="killzone_ny_am", direction="bullish",
        setup_at=pd.Timestamp("2024-06-03 14:05:00", tz="UTC"),
        fvg_timeframe="1m", fvg_top=101.0, fvg_bottom=100.0, fvg_midpoint=100.5,
        sweep_level_type="prior_session_low", sweep_level_price=99.0,
        sweep_confirmed_at=pd.Timestamp("2024-06-03 14:03:00", tz="UTC"),
        mss_broken_at=None, displacement_at=None, bias_value=None,
    )
    row.update(overrides)
    return row


def _session_df(fill_row, extra_rows_after=None):
    idx = pd.date_range("2024-06-03 14:00:00", "2024-06-03 14:59:00", freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.5, "high": 100.7, "low": 100.3, "close": 100.5, "volume": 10, "contract": "X", "is_roll_day": False},
        index=idx,
    )
    for ts, values in fill_row.items():
        for col, v in values.items():
            df.loc[ts, col] = v
    return df


def test_simulate_trades_end_to_end_target_hit():
    signal = _signal_row()
    entry_bar = pd.Timestamp("2024-06-03 14:06:00", tz="UTC")  # bar after setup_at
    target_bar = pd.Timestamp("2024-06-03 14:20:00", tz="UTC")
    df = _session_df(
        {
            entry_bar: {"low": 100.4},  # trades through 50% entry (100.5) -> fills
            target_bar: {"high": 105.0},
        }
    )
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2,
    )
    signals = pd.DataFrame([signal])
    no_signals = pd.DataFrame(columns=NO_TRADE_COLUMNS)

    trades, no_trades = simulate_trades(
        signals, no_signals, config, "NQ", df, levels=pd.DataFrame(), sweeps=pd.DataFrame()
    )

    assert no_trades.empty
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_price"] == 100.5  # midpoint of [100,101]
    assert row["stop_price"] == 99.0  # swept level, buffer=0
    assert row["target_price"] == 100.5 + 2 * (100.5 - 99.0)  # entry + 2R
    assert row["exit_reason"] == "target"
    assert row["gross_pnl"] == pytest.approx(((row["target_price"] - row["entry_price"]) / 0.25) * COST_MODELS["NQ"].tick_value)
    assert row["net_pnl"] == pytest.approx(row["gross_pnl"] - COST_MODELS["NQ"].commission_round_turn)
    assert row["r_multiple"] == pytest.approx(2.0)
    assert row["symbol"] == "NQ"
    assert row["config"]["name"] == "t"
    assert bool(row["is_roll_day"]) is False
    assert set(trades.columns) == set(TRADE_COLUMNS)


def test_simulate_trades_limit_unfilled_becomes_no_trade():
    signal = _signal_row()
    # Flat baseline bars are [100.3, 100.7] and never trade down to the
    # entry_level="distal" price (fvg_bottom=100.0) -- stays unfilled.
    df = _session_df({})
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="distal", stop_type="swing", stop_buffer_ticks=0,
    )
    signals = pd.DataFrame([signal])
    no_signals = pd.DataFrame(columns=NO_TRADE_COLUMNS)

    trades, no_trades = simulate_trades(signals, no_signals, config, "NQ", df, pd.DataFrame(), pd.DataFrame())
    assert trades.empty
    assert no_trades.iloc[0]["reason"] == "limit_unfilled"


# ---------- Phase 4: multiple trades per window ----------


def test_simulate_trades_multiple_nonoverlapping_signals_both_become_trades():
    df = _session_df(
        {
            pd.Timestamp("2024-06-03 14:06:00", tz="UTC"): {"low": 100.4},  # signal A fills
            pd.Timestamp("2024-06-03 14:08:00", tz="UTC"): {"high": 105.0},  # signal A hits target
            pd.Timestamp("2024-06-03 14:16:00", tz="UTC"): {"low": 100.4},  # signal B fills
            pd.Timestamp("2024-06-03 14:18:00", tz="UTC"): {"high": 105.0},  # signal B hits target
        }
    )
    signals = pd.DataFrame(
        [_signal_row(), _signal_row(setup_at=pd.Timestamp("2024-06-03 14:15:00", tz="UTC"))]
    )
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2, max_trades_per_window=2,
    )
    trades, no_trades = simulate_trades(
        signals, pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df, pd.DataFrame(), pd.DataFrame()
    )
    assert no_trades.empty
    assert list(trades["entry_at"]) == [
        pd.Timestamp("2024-06-03 14:06:00", tz="UTC"), pd.Timestamp("2024-06-03 14:16:00", tz="UTC"),
    ]
    assert list(trades["exit_at"]) == [
        pd.Timestamp("2024-06-03 14:08:00", tz="UTC"), pd.Timestamp("2024-06-03 14:18:00", tz="UTC"),
    ]


def test_simulate_trades_skips_signal_while_position_still_open():
    df = _session_df(
        {
            pd.Timestamp("2024-06-03 14:06:00", tz="UTC"): {"low": 100.4},  # A fills
            pd.Timestamp("2024-06-03 14:20:00", tz="UTC"): {"high": 105.0},  # A's target, well after B's setup
        }
    )
    signal_a = _signal_row()  # setup_at 14:05
    signal_b = _signal_row(setup_at=pd.Timestamp("2024-06-03 14:10:00", tz="UTC"))  # while A is still open
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2, max_trades_per_window=5,
    )
    trades, no_trades = simulate_trades(
        pd.DataFrame([signal_a, signal_b]), pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df,
        pd.DataFrame(), pd.DataFrame(),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["setup_at"] == signal_a["setup_at"]
    assert no_trades.iloc[0]["reason"] == "position_open"


def test_simulate_trades_signal_at_exact_exit_bar_is_still_skipped():
    # setup_at <= position_open_until is a strict-inclusive skip: a signal
    # landing on the exact bar the prior trade exited does not get to fill.
    exit_bar = pd.Timestamp("2024-06-03 14:10:00", tz="UTC")
    df = _session_df(
        {pd.Timestamp("2024-06-03 14:06:00", tz="UTC"): {"low": 100.4}, exit_bar: {"high": 105.0}}
    )
    signal_a = _signal_row()
    signal_b = _signal_row(setup_at=exit_bar)
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2, max_trades_per_window=5,
    )
    trades, no_trades = simulate_trades(
        pd.DataFrame([signal_a, signal_b]), pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df,
        pd.DataFrame(), pd.DataFrame(),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["exit_at"] == exit_bar
    assert no_trades.iloc[0]["reason"] == "position_open"


def test_simulate_trades_caps_at_max_trades_per_window():
    df = _session_df(
        {
            pd.Timestamp("2024-06-03 14:06:00", tz="UTC"): {"low": 100.4},
            pd.Timestamp("2024-06-03 14:08:00", tz="UTC"): {"high": 105.0},
            pd.Timestamp("2024-06-03 14:16:00", tz="UTC"): {"low": 100.4},
            pd.Timestamp("2024-06-03 14:18:00", tz="UTC"): {"high": 105.0},
            pd.Timestamp("2024-06-03 14:26:00", tz="UTC"): {"low": 100.4},
            pd.Timestamp("2024-06-03 14:28:00", tz="UTC"): {"high": 105.0},
        }
    )
    signals = pd.DataFrame(
        [
            _signal_row(setup_at=pd.Timestamp("2024-06-03 14:05:00", tz="UTC")),
            _signal_row(setup_at=pd.Timestamp("2024-06-03 14:15:00", tz="UTC")),
            _signal_row(setup_at=pd.Timestamp("2024-06-03 14:25:00", tz="UTC")),
        ]
    )
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2, max_trades_per_window=2,
    )
    trades, no_trades = simulate_trades(
        signals, pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df, pd.DataFrame(), pd.DataFrame()
    )
    assert len(trades) == 2
    assert no_trades.iloc[0]["reason"] == "max_trades_reached"


def test_simulate_trades_unfilled_entry_does_not_block_next_signal():
    # Signal A's FVG (49-50) is unreachable by any bar in the session, so its
    # limit never fills; signal B (later setup_at, normal FVG) must still be
    # evaluated fully rather than being treated as blocked by A.
    df = _session_df(
        {
            pd.Timestamp("2024-06-03 14:16:00", tz="UTC"): {"low": 100.4},
            pd.Timestamp("2024-06-03 14:18:00", tz="UTC"): {"high": 105.0},
        }
    )
    signal_a = _signal_row(
        setup_at=pd.Timestamp("2024-06-03 14:05:00", tz="UTC"), fvg_top=50.0, fvg_bottom=49.0, fvg_midpoint=49.5,
    )
    signal_b = _signal_row(setup_at=pd.Timestamp("2024-06-03 14:15:00", tz="UTC"))
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_level_types=("prior_session_low",),
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="fixed_r", target_r_multiple=2, max_trades_per_window=5,
    )
    trades, no_trades = simulate_trades(
        pd.DataFrame([signal_a, signal_b]), pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df,
        pd.DataFrame(), pd.DataFrame(),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["entry_at"] == pd.Timestamp("2024-06-03 14:16:00", tz="UTC")
    assert set(no_trades["reason"]) == {"limit_unfilled"}


def test_simulate_trades_next_liquidity_target_uses_resolved_sweep_universe():
    # sweep_universe="swings_only" resolves to swing_high/swing_low only --
    # the target must ignore the nearer prior_session_high even though it's
    # present in the passed-in levels frame, matching what the sweep gate
    # itself would have been scoped to.
    df = _session_df({})
    signal = _signal_row()
    levels = pd.DataFrame(
        [
            {
                "level_type": "prior_session_high", "session_date": pd.Timestamp("2024-06-03"),
                "price": 101.5, "knowable_at": pd.Timestamp("2024-06-03 14:00:00", tz="UTC"),
            },
            {
                "level_type": "swing_high", "session_date": pd.Timestamp("2024-06-03"),
                "price": 110.0, "knowable_at": pd.Timestamp("2024-06-03 14:00:00", tz="UTC"),
            },
        ]
    )
    config = StrategyConfig(
        name="t", windows=("killzone_ny_am",), sweep_required=True, sweep_universe="swings_only",
        mss_required=False, displacement_required=False, entry_level="50%", stop_type="swing",
        stop_buffer_ticks=0, target_type="next_liquidity",
    )
    trades, _ = simulate_trades(
        pd.DataFrame([signal]), pd.DataFrame(columns=NO_TRADE_COLUMNS), config, "NQ", df, levels, pd.DataFrame()
    )
    assert len(trades) == 1
    assert trades.iloc[0]["target_price"] == 110.0


def test_simulate_trades_passes_through_no_signal_reasons():
    no_signals = pd.DataFrame(
        [{"session_date": pd.Timestamp("2024-06-03"), "window": "killzone_ny_am", "reason": "no_sweep"}]
    )
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    trades, no_trades = simulate_trades(
        pd.DataFrame(columns=["session_date", "window", "direction", "setup_at", "fvg_top", "fvg_bottom", "sweep_level_price"]),
        no_signals, config, "NQ", _session_df({}), pd.DataFrame(), pd.DataFrame(),
    )
    assert trades.empty
    assert no_trades.iloc[0]["reason"] == "no_sweep"
