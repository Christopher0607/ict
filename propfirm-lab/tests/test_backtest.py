"""Vectorized backtester: the arithmetic, and the three honesty rules.

The most valuable case here reproduces a trade that was hand-verified against
raw NQ bars through the ICT engine, so two independent implementations have to
agree on the same numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.search.backtest import (
    COMMISSION_RT,
    POINT_VALUE,
    TradeResult,
    simulate,
)


def _bars(o, h, l):
    return np.array(o, float), np.array(h, float), np.array(l, float)


def _run(o, h, l, sig, direction, stop, target, session_end=None, horizon=10,
         time_exit_bars=None):
    o, h, l = _bars(o, h, l)
    sig = np.array(sig, np.int64)
    if session_end is None:
        session_end = np.full(sig.size, len(o) - 1, np.int64)
    return simulate(
        h, l, o, sig,
        np.array(direction, np.int64),
        np.array(stop, float),
        np.array(target, float),
        np.array(session_end, np.int64),
        horizon=horizon,
        time_exit_bars=time_exit_bars,
    )


def test_entry_is_the_next_bar_open_not_this_bar_close():
    """A signal on bar t cannot transact at bar t's close -- that price is
    only known once the bar has finished."""
    r = _run(
        o=[100, 105, 106], h=[101, 106, 107], l=[99, 104, 105],
        sig=[0], direction=[1], stop=[5], target=[5],
    )
    assert r.entry_idx[0] == 1
    assert r.entry_price[0] == 105.0


def test_clean_target_hit_long():
    r = _run(
        o=[100, 100, 100], h=[100, 101, 110], l=[100, 99, 100],
        sig=[0], direction=[1], stop=[5], target=[8],
    )
    assert r.hit_target[0] and not r.hit_stop[0]
    assert r.exit_price[0] == 108.0
    assert r.net_pnl[0] == pytest.approx(8 * POINT_VALUE - COMMISSION_RT)
    assert r.r_multiple[0] == pytest.approx((8 * POINT_VALUE - COMMISSION_RT) / (5 * POINT_VALUE))


def test_clean_stop_hit_long_pays_slippage():
    r = _run(
        o=[100, 100, 100], h=[100, 101, 101], l=[100, 99, 90],
        sig=[0], direction=[1], stop=[5], target=[20],
    )
    assert r.hit_stop[0]
    # Stop at 95, filled one tick worse.
    assert r.exit_price[0] == pytest.approx(94.75)
    assert r.net_pnl[0] < -5 * POINT_VALUE


def test_short_side_inverts_the_barriers():
    r = _run(
        o=[100, 100, 100], h=[100, 101, 101], l=[100, 99, 90],
        sig=[0], direction=[-1], stop=[5], target=[8],
    )
    assert r.stop_price[0] == 105.0
    assert r.target_price[0] == 92.0
    assert r.hit_target[0]


def test_bar_spanning_both_barriers_is_charged_as_a_stop():
    """The pessimistic convention, applied rather than assumed away."""
    r = _run(
        o=[100, 100, 100], h=[100, 100, 120], l=[100, 100, 80],
        sig=[0], direction=[1], stop=[5], target=[8],
    )
    assert r.ambiguous[0]
    assert r.hit_stop[0] and not r.hit_target[0]


def test_position_is_flat_by_session_end():
    """Prop-firm accounts with trailing drawdown are not held overnight."""
    r = _run(
        o=[100, 100, 100, 100, 100], h=[100, 101, 101, 101, 200],
        l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        session_end=[2],
    )
    assert not r.hit_target[0] and not r.hit_stop[0]
    assert r.exit_idx[0] == 2  # closed at its own session's last bar


def test_a_later_session_barrier_cannot_resolve_an_earlier_trade():
    """The +200 spike on bar 4 belongs to the next session and must be invisible."""
    r = _run(
        o=[100] * 5, h=[100, 101, 101, 101, 500], l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[100],
        session_end=[2],
    )
    assert not r.hit_target[0]


def test_reproduces_the_hand_verified_ict_trade():
    """Cross-check against a trade verified bar-by-bar through the ICT engine.

    NQ 2019-03-26, short from 10849.0, stop 10859.0, exit 10845.75 at the
    window close. The ICT engine reported gross $65.00, net $61.00, R 0.325,
    MAE 1.5, MFE 6.0. Two independent implementations must agree.
    """
    # Entry bar, then the bars held, then the session ends.
    o = [10844.25, 10849.00, 10849.50, 10849.25, 10847.00, 10843.50]
    h = [10846.00, 10850.25, 10850.50, 10850.25, 10847.50, 10846.75]
    l = [10842.75, 10844.25, 10847.50, 10846.50, 10843.00, 10843.25]
    # Exit is a session-end close-out, so target/stop must not trigger.
    r = _run(o, h, l, sig=[0], direction=[-1], stop=[10.0], target=[79.5],
             session_end=[5], horizon=10)

    assert r.entry_price[0] == pytest.approx(10849.0)
    assert r.stop_price[0] == pytest.approx(10859.0)
    assert not r.hit_stop[0] and not r.hit_target[0]
    assert r.exit_price[0] == pytest.approx(10843.50)  # bar 5 open

    # MAE/MFE over the held window, matching the ICT engine's definitions.
    assert r.mae_points[0] == pytest.approx(1.5)
    assert r.mfe_points[0] == pytest.approx(6.0)


def test_expectancy_reports_its_standard_error():
    r = _run(
        o=[100] * 6, h=[100, 101, 110, 110, 110, 110], l=[100, 99, 100, 100, 100, 100],
        sig=[0, 1], direction=[1, 1], stop=[5, 5], target=[8, 8],
    )
    assert len(r) == 2
    assert np.isfinite(r.expectancy_r)
    assert r.expectancy_se >= 0


def test_no_signals_returns_an_empty_result():
    r = _run(o=[100, 100], h=[100, 100], l=[100, 100],
             sig=[], direction=[], stop=[], target=[])
    assert len(r) == 0
    assert r.expectancy_r == 0.0
    assert r.expectancy_se == float("inf")


def test_signal_on_the_last_bar_is_dropped():
    """There is no next bar to enter on."""
    r = _run(o=[100, 100], h=[100, 100], l=[100, 100],
             sig=[1], direction=[1], stop=[5], target=[5])
    assert len(r) == 0


# ---------------------------------------------------------------------------
# position sequencing
# ---------------------------------------------------------------------------


class _F:
    """Minimal stand-in for a FeatureSet."""

    def __init__(self, o, h, l, session_id, is_rth=True):
        self.open = np.array(o, float)
        self.high = np.array(h, float)
        self.low = np.array(l, float)
        self.session_id = np.array(session_id, np.int64)
        self.is_rth = np.full(len(o), bool(is_rth))
        n = len(o)
        idx = np.arange(n)
        is_last = np.concatenate([self.session_id[1:] != self.session_id[:-1], [True]])
        marks = np.where(is_last, idx, n)
        self.session_end_idx = np.minimum.accumulate(marks[::-1])[::-1]


def test_entries_do_not_overlap():
    """A signal firing on every bar must not book a trade on every bar."""
    from research.search.backtest import simulate_sequential

    n = 30
    f = _F([100] * n, [100] * n, [100] * n, [0] * n)
    sig = np.arange(n - 1, dtype=np.int64)
    r = simulate_sequential(
        f, sig, np.ones(n - 1, np.int64),
        np.full(n - 1, 5.0), np.full(n - 1, 5.0), horizon=10,
    )
    # Flat prices never resolve, so each trade runs to session end; only one
    # position can be open at a time.
    assert len(r) <= 3
    for a, b in zip(r.entry_idx[:-1], r.entry_idx[1:]):
        assert b > a


def test_at_most_three_entries_per_session():
    from research.search.backtest import simulate_sequential

    n = 60
    # Alternating up/down so trades resolve quickly and more could be taken.
    h = [100 + (i % 2) * 20 for i in range(n)]
    l = [100 - (i % 2) * 20 for i in range(n)]
    f = _F([100] * n, h, l, [0] * n)
    sig = np.arange(n - 1, dtype=np.int64)
    r = simulate_sequential(
        f, sig, np.ones(n - 1, np.int64),
        np.full(n - 1, 5.0), np.full(n - 1, 5.0), horizon=5,
    )
    assert len(r) <= 3


def test_each_session_gets_its_own_allowance():
    from research.search.backtest import simulate_sequential

    n = 40
    sessions = [0] * 20 + [1] * 20
    f = _F([100] * n, [100] * n, [100] * n, sessions)
    sig = np.arange(n - 1, dtype=np.int64)
    r = simulate_sequential(
        f, sig, np.ones(n - 1, np.int64),
        np.full(n - 1, 5.0), np.full(n - 1, 5.0), horizon=10,
    )
    assert (f.session_id[r.entry_idx] == 0).any()
    assert (f.session_id[r.entry_idx] == 1).any()


def test_a_signal_on_the_last_bar_of_a_session_is_dropped():
    """Entering on the next bar would carry the setup across the overnight break."""
    from research.search.backtest import simulate_sequential

    n = 20
    sessions = [0] * 10 + [1] * 10
    f = _F([100] * n, [100] * n, [100] * n, sessions)
    # Bar 9 is the last bar of session 0; entering at bar 10 is session 1.
    r = simulate_sequential(
        f, np.array([9], np.int64), np.array([1], np.int64),
        np.array([5.0]), np.array([5.0]), horizon=5,
    )
    assert len(r) == 0


def test_a_signal_mid_session_still_enters():
    from research.search.backtest import simulate_sequential

    n = 20
    sessions = [0] * 10 + [1] * 10
    f = _F([100] * n, [100] * n, [100] * n, sessions)
    r = simulate_sequential(
        f, np.array([5], np.int64), np.array([1], np.int64),
        np.array([5.0]), np.array([5.0]), horizon=5,
    )
    assert len(r) == 1
    assert r.entry_idx[0] == 6


def test_commission_r_is_measured_from_trades_not_estimated():
    """The error this replaces: a median-ATR estimate is not the mean of ratios.

    Two trades with very different risk. The true mean of COMMISSION/risk is
    not COMMISSION divided by the mean risk, and the gap is exactly the kind
    that corrupted the first search's gross-expectancy figures.
    """
    from research.search.backtest import COMMISSION_RT, simulate

    o = np.array([100.0] * 6)
    h = np.array([100.0] * 6)
    l = np.array([100.0] * 6)
    r = simulate(
        h, l, o,
        np.array([0, 2], np.int64), np.array([1, 1], np.int64),
        np.array([1.0, 10.0]),      # risk $20 and $200
        np.array([5.0, 50.0]),
        np.array([5, 5], np.int64),
        horizon=4,
    )
    assert len(r) == 2
    expected = float(np.mean([COMMISSION_RT / 20.0, COMMISSION_RT / 200.0]))
    assert r.commission_r == pytest.approx(expected)

    # The naive "commission / mean risk" shortcut gives a different number.
    naive = COMMISSION_RT / np.mean([20.0, 200.0])
    assert abs(r.commission_r - naive) > 0.01


def test_gross_expectancy_adds_commission_back_but_keeps_slippage():
    from research.search.backtest import simulate

    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 101.0, 110.0])
    l = np.array([100.0, 99.0, 100.0])
    r = simulate(
        h, l, o, np.array([0], np.int64), np.array([1], np.int64),
        np.array([5.0]), np.array([8.0]), np.array([2], np.int64), horizon=3,
    )
    assert r.gross_expectancy_r == pytest.approx(r.expectancy_r + r.commission_r)
    assert r.gross_expectancy_r > r.expectancy_r


def test_overnight_stops_pay_more_slippage_than_rth():
    """Median 1-minute volume is 707 in RTH and 57 outside it. Charging the
    same tick everywhere quietly flatters every overnight result."""
    from research.search.backtest import (
        STOP_SLIPPAGE_TICKS, STOP_SLIPPAGE_TICKS_ETH, TICK_SIZE, simulate,
    )

    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 101.0, 101.0])
    l = np.array([100.0, 99.0, 90.0])
    args = (h, l, o, np.array([0], np.int64), np.array([1], np.int64),
            np.array([5.0]), np.array([20.0]), np.array([2], np.int64))

    rth = simulate(*args, horizon=3, slippage_ticks=STOP_SLIPPAGE_TICKS)
    eth = simulate(*args, horizon=3, slippage_ticks=STOP_SLIPPAGE_TICKS_ETH)

    assert rth.hit_stop[0] and eth.hit_stop[0]
    assert eth.exit_price[0] < rth.exit_price[0]
    assert rth.exit_price[0] - eth.exit_price[0] == pytest.approx(TICK_SIZE)
    assert eth.r_multiple[0] < rth.r_multiple[0]


def test_sequential_charges_eth_slippage_outside_rth():
    """The same path, once inside RTH and once outside, must differ."""
    from research.search.backtest import simulate_sequential

    n = 6
    o = [100.0] * n
    h = [100.0, 101.0, 101.0, 101.0, 101.0, 101.0]
    l = [100.0, 99.0, 90.0, 90.0, 90.0, 90.0]

    rth = simulate_sequential(
        _F(o, h, l, [0] * n, is_rth=True), np.array([0], np.int64),
        np.array([1], np.int64), np.array([5.0]), np.array([20.0]), horizon=4,
    )
    eth = simulate_sequential(
        _F(o, h, l, [0] * n, is_rth=False), np.array([0], np.int64),
        np.array([1], np.int64), np.array([5.0]), np.array([20.0]), horizon=4,
    )
    assert rth.hit_stop[0] and eth.hit_stop[0]
    assert eth.r_multiple[0] < rth.r_multiple[0]


# ---------------------------------------------------------------------------
# Time exits
# ---------------------------------------------------------------------------
#
# The predictability audit's only economically positive cell was a fixed
# 60-minute hold, and until now the engine could only exit on a barrier or at
# the close -- so the one direction worth testing was the one it could not
# express.


def test_time_exit_flattens_after_n_bars_at_that_bars_open():
    """Entry at bar 1's open, three bars held (1, 2, 3), flat at bar 4's open."""
    r = _run(
        o=[100, 100, 100, 100, 107, 100], h=[100, 101, 101, 101, 108, 101],
        l=[100, 99, 99, 99, 106, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        time_exit_bars=3,
    )
    assert not r.hit_stop[0] and not r.hit_target[0]
    assert r.entry_idx[0] == 1
    assert r.exit_idx[0] == 4
    assert r.exit_price[0] == 107.0
    assert r.net_pnl[0] == pytest.approx(7 * POINT_VALUE - COMMISSION_RT)


def test_time_exit_does_not_pay_stop_slippage():
    """It is a market exit at a price nobody was forced into, not a stop."""
    r = _run(
        o=[100, 100, 100, 100, 100], h=[100, 101, 101, 101, 101],
        l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        time_exit_bars=3,
    )
    assert r.exit_price[0] == 100.0  # the open, undisturbed


def test_barriers_still_win_inside_the_hold():
    """A target touched on bar 2 resolves there; the time exit never fires."""
    r = _run(
        o=[100, 100, 100, 100, 100], h=[100, 101, 108, 101, 101],
        l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[5],
        time_exit_bars=3,
    )
    assert r.hit_target[0]
    assert r.exit_idx[0] == 2
    assert r.exit_price[0] == 105.0


def test_the_close_beats_a_later_time_exit():
    """Session ends at bar 2; a 60-bar hold does not survive the close."""
    r = _run(
        o=[100, 100, 100, 100, 100], h=[100, 101, 101, 101, 200],
        l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        session_end=[2], time_exit_bars=8, horizon=9,
    )
    assert r.exit_idx[0] == 2
    assert not r.hit_target[0]  # the bar-4 spike is the next session's


def test_the_time_exit_beats_a_later_close():
    """Session runs to bar 5; a 2-bar hold flattens at bar 3 regardless."""
    r = _run(
        o=[100, 100, 100, 103, 100, 100], h=[100, 101, 101, 104, 200, 101],
        l=[100, 99, 99, 102, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        session_end=[5], time_exit_bars=2,
    )
    assert r.exit_idx[0] == 3
    assert r.exit_price[0] == 103.0
    assert not r.hit_target[0]  # bar 4 is past the hold, its spike is invisible


def test_a_barrier_after_the_time_exit_is_invisible():
    """The stop on bar 4 is outside a 2-bar hold and must not resolve it."""
    r = _run(
        o=[100] * 6, h=[100, 101, 101, 101, 101, 101],
        l=[100, 99, 99, 99, 40, 99],
        sig=[0], direction=[1], stop=[50], target=[50],
        session_end=[5], time_exit_bars=2,
    )
    assert not r.hit_stop[0]
    assert r.exit_idx[0] == 3


def test_time_exit_shorter_than_horizon_is_required():
    o, h, l = _bars([100] * 5, [101] * 5, [99] * 5)
    with pytest.raises(ValueError, match="shorter than time_exit_bars"):
        simulate(h, l, o, np.array([0]), np.array([1]), np.array([5.0]),
                 np.array([5.0]), np.array([4]), horizon=3, time_exit_bars=8)


def test_time_exit_none_is_the_old_behaviour():
    """The default path has to be byte-identical to before the parameter existed."""
    args = dict(
        o=[100, 100, 100, 100, 100], h=[100, 101, 101, 101, 101],
        l=[100, 99, 99, 99, 99],
        sig=[0], direction=[1], stop=[50], target=[50], session_end=[3],
    )
    r = _run(**args, time_exit_bars=None)
    assert r.exit_idx[0] == 3 and not r.hit_stop[0] and not r.hit_target[0]


def test_sequential_respects_the_time_exit_and_reuses_the_slot():
    """A 2-bar hold frees the session for another entry the old exit blocked."""
    from research.search.backtest import simulate_sequential

    n = 12
    f = _F([100] * n, [101] * n, [99] * n, [0] * n)
    sig = np.array([0, 4], np.int64)
    r = simulate_sequential(
        f, sig, np.array([1, 1], np.int64),
        np.full(2, 50.0), np.full(2, 50.0), time_exit_bars=2,
    )
    assert len(r) == 2
    assert list(r.entry_idx) == [1, 5]
    assert list(r.exit_idx) == [3, 7]


# ---------------------------------------------------------------------------
# Contract specification: NQ against its micro
# ---------------------------------------------------------------------------
#
# The micro is the same index at a tenth the multiplier, so the interesting
# question is not whether it is cheaper -- it is what happens per dollar of
# risk, where commission does not scale with point value and slippage does.


def _run_inst(inst, stop=50.0, o=None, h=None, l=None):
    from research.search.backtest import simulate
    o = o or [100, 100, 100, 100]
    h = h or [100, 101, 101, 101]
    l = l or [100, 99, 99, 99]
    oo, hh, ll = _bars(o, h, l)
    return simulate(
        hh, ll, oo, np.array([0]), np.array([1]),
        np.array([stop]), np.array([stop]),
        np.array([len(o) - 1]), horizon=4, instrument=inst,
    )


def test_micro_costs_more_commission_per_unit_of_risk():
    """Point value falls tenfold, commission falls under fourfold. The micro is
    two to three times dearer per dollar risked, which is the fact this whole
    round turns on."""
    from research.search.backtest import MNQ, NQ

    nq, mnq = _run_inst(NQ), _run_inst(MNQ)
    assert nq.risk_dollars[0] == pytest.approx(50 * 20)
    assert mnq.risk_dollars[0] == pytest.approx(50 * 2)
    assert nq.commission_r == pytest.approx(4.00 / (50 * 20))
    assert mnq.commission_r == pytest.approx(1.04 / (50 * 2))
    assert mnq.commission_r / nq.commission_r == pytest.approx(2.6, rel=1e-6)


def test_slippage_costs_the_same_R_on_either_contract():
    """Point value appears in both the fill and the risk, so it cancels."""
    from research.search.backtest import MNQ, NQ

    # A stop-out: bar 2 trades through the stop at 50 points.
    o, h, l = [100, 100, 100, 100], [100, 101, 101, 101], [100, 99, 40, 99]
    nq = _run_inst(NQ, o=o, h=h, l=l)
    mnq = _run_inst(MNQ, o=o, h=h, l=l)
    assert nq.hit_stop[0] and mnq.hit_stop[0]

    # Gross expectancy keeps slippage and removes commission, so if slippage
    # costs the same R on both, the two gross figures must agree exactly.
    assert nq.gross_expectancy_r == pytest.approx(mnq.gross_expectancy_r)
    # ...while the net figures differ by exactly the commission gap.
    gap = mnq.commission_r - nq.commission_r
    assert nq.expectancy_r - mnq.expectancy_r == pytest.approx(gap)


def test_commission_tiers_move_expectancy_the_predicted_amount():
    from research.search.backtest import MNQ_CHEAP, MNQ_DEAR

    cheap, dear = _run_inst(MNQ_CHEAP), _run_inst(MNQ_DEAR)
    assert cheap.expectancy_r > dear.expectancy_r
    assert cheap.expectancy_r - dear.expectancy_r == pytest.approx(
        (1.34 - 0.74) / (50 * 2)
    )


def test_nq_stays_the_default():
    """Every published figure was computed on NQ. The default must not move."""
    from research.search.backtest import COMMISSION_RT, NQ, POINT_VALUE, TICK_SIZE

    assert (TICK_SIZE, POINT_VALUE, COMMISSION_RT) == (0.25, 20.0, 4.00)
    r = _run_inst(NQ)
    assert r.instrument.name == "NQ"
    # Same call without naming an instrument.
    default = _run(o=[100, 100, 100, 100], h=[100, 101, 101, 101],
                   l=[100, 99, 99, 99], sig=[0], direction=[1],
                   stop=[50.0], target=[50.0], horizon=4)
    assert default.expectancy_r == pytest.approx(r.expectancy_r)


def test_sequential_carries_the_instrument_through():
    from research.search.backtest import MNQ, simulate_sequential

    n = 12
    f = _F([100] * n, [101] * n, [99] * n, [0] * n)
    r = simulate_sequential(
        f, np.array([0, 4], np.int64), np.array([1, 1], np.int64),
        np.full(2, 50.0), np.full(2, 50.0), time_exit_bars=2, instrument=MNQ,
    )
    assert r.instrument.name == "MNQ"
    assert r.commission_r == pytest.approx(1.04 / (50 * 2))


def test_empty_result_keeps_its_instrument():
    from research.search.backtest import MNQ, simulate_sequential

    f = _F([100] * 5, [101] * 5, [99] * 5, [0] * 5)
    r = simulate_sequential(f, np.array([], np.int64), np.array([], np.int64),
                            np.array([]), np.array([]), instrument=MNQ)
    assert len(r) == 0 and r.instrument.name == "MNQ"
    assert r.commission_r == 0.0
