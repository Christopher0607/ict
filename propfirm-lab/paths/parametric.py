"""Generate equity paths from a trader-legible description of an edge.

The knobs are the ones a trader actually knows about themselves -- dollars
risked per trade, reward-to-risk, win rate -- rather than a drift and a
diffusion coefficient. The single most important knob is ``expectancy_r``:
expected R per trade after costs. That is the x-axis of the edge-requirement
curve, and the number every strategy must eventually be reduced to.

Each trade emits four marks rather than one, because a trailing floor cares
about the excursion, not just the settlement:

    entry -> first excursion -> second excursion -> exit

A winning trade normally dips against you before it works (MAE first); a
losing trade normally shows some profit before it fails (MFE first). Both
orderings are supported because with bar data you often cannot tell which
happened, and the difference is worth real money under an intraday trailing
threshold. See ``excursion_order``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MARKS_PER_TRADE = 4


@dataclass(frozen=True)
class TradeModel:
    """A strategy, described only by the statistics that affect survival."""

    r_dollars: float = 250.0
    """Dollars risked per trade -- one R."""

    expectancy_r: float = 0.0
    """Expected R per trade, net of commission and slippage. The edge axis."""

    payoff_r: float = 2.0
    """Reward-to-risk of a winner. Win rate is derived from this + expectancy."""

    trades_per_day: int = 3

    mae_frac: float = 0.5
    """Winners first go this far against you, as a fraction of R (uniform 0..x)."""

    mfe_frac: float = 0.5
    """Losers first go this far in your favour, as a fraction of R (uniform 0..x)."""

    excursion_order: str = "realistic"
    """``realistic`` | ``worst`` | ``best``.

    ``worst`` puts every trade's favourable excursion first, which maximizes
    how far an intraday trailing floor ratchets up before the trade resolves.
    ``best`` does the opposite. Running all three gives the ambiguity band that
    bar data genuinely cannot resolve.
    """

    @property
    def win_rate(self) -> float:
        """p such that p*payoff - (1-p) == expectancy."""
        p = (self.expectancy_r + 1.0) / (self.payoff_r + 1.0)
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"expectancy_r={self.expectancy_r} is unreachable with "
                f"payoff_r={self.payoff_r} (implies win rate {p:.3f})"
            )
        return p

    def __post_init__(self) -> None:
        if self.r_dollars <= 0:
            raise ValueError("r_dollars must be positive")
        if self.payoff_r <= 0:
            raise ValueError("payoff_r must be positive")
        if self.trades_per_day < 1:
            raise ValueError("trades_per_day must be >= 1")
        if self.excursion_order not in ("realistic", "worst", "best"):
            raise ValueError("excursion_order must be realistic|worst|best")
        self.win_rate  # validate reachability eagerly


def generate_path(
    model: TradeModel,
    n_days: int,
    start_equity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One equity path. Returns ``(equity, days)`` of equal length."""
    n_trades = n_days * model.trades_per_day
    wins = rng.random(n_trades) < model.win_rate

    outcome_r = np.where(wins, model.payoff_r, -1.0)

    # Excursion against the eventual direction, in R.
    adverse = rng.random(n_trades) * model.mae_frac
    favourable = rng.random(n_trades) * model.mfe_frac
    first_r = np.where(wins, -adverse, favourable)
    second_r = np.where(wins, favourable * model.payoff_r * 0.5, -adverse)

    if model.excursion_order == "worst":
        # Favourable excursion always first: ratchets a trailing floor up
        # before the trade has actually paid.
        lo = np.minimum(first_r, second_r)
        hi = np.maximum(first_r, second_r)
        first_r, second_r = hi, lo
    elif model.excursion_order == "best":
        lo = np.minimum(first_r, second_r)
        hi = np.maximum(first_r, second_r)
        first_r, second_r = lo, hi

    # Per-trade marks, in dollars, relative to the equity before the trade.
    legs = np.stack([first_r, second_r, outcome_r], axis=1) * model.r_dollars

    # Equity after each trade's settlement.
    settled = start_equity + np.cumsum(outcome_r * model.r_dollars)
    before = np.concatenate(([start_equity], settled[:-1]))

    equity = np.empty(n_trades * MARKS_PER_TRADE, dtype=float)
    equity[0::MARKS_PER_TRADE] = before
    equity[1::MARKS_PER_TRADE] = before + legs[:, 0]
    equity[2::MARKS_PER_TRADE] = before + legs[:, 1]
    equity[3::MARKS_PER_TRADE] = settled

    days = np.repeat(
        np.repeat(np.arange(n_days, dtype=np.int64), model.trades_per_day),
        MARKS_PER_TRADE,
    )
    return equity, days


def generate_paths(
    model: TradeModel,
    n_paths: int,
    n_days: int,
    start_equity: float,
    rng: np.random.Generator,
):
    """Lazily yield ``n_paths`` independent paths."""
    for _ in range(n_paths):
        yield generate_path(model, n_days, start_equity, rng)
