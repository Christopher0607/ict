"""Monte Carlo drivers: turn a TradeModel + Ruleset into survival statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from paths import TradeModel, generate_path
from rules import Ruleset
from sim.account import Outcome, simulate_lifecycle, simulate_stage


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves at p near 0 or 1, unlike normal-approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class EvalStudy:
    """Eval-stage-only statistics: the cheap half of the question."""

    n_paths: int
    passes: int
    outcomes: Counter = field(default_factory=Counter)
    days_to_pass: list[int] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n_paths if self.n_paths else 0.0

    @property
    def pass_rate_ci(self) -> tuple[float, float]:
        return _wilson(self.passes, self.n_paths)

    @property
    def median_days_to_pass(self) -> float | None:
        return float(np.median(self.days_to_pass)) if self.days_to_pass else None


def study_eval(
    model: TradeModel,
    rs: Ruleset,
    *,
    n_paths: int = 5_000,
    n_days: int = 400,
    seed: int = 0,
) -> EvalStudy:
    """How often does this trader clear this eval, and what kills them?"""
    rng = np.random.default_rng(seed)
    out = EvalStudy(n_paths=n_paths, passes=0)
    for _ in range(n_paths):
        equity, days = generate_path(model, n_days, rs.starting_balance, rng)
        res = simulate_stage(equity, days, rs, target_equity=rs.target_equity)
        out.outcomes[res.outcome.value] += 1
        if res.outcome is Outcome.PASSED:
            out.passes += 1
            out.days_to_pass.append(res.stop_day + 1)
    return out


@dataclass
class LifecycleStudy:
    """The whole question: dollars in, dollars out, per account bought."""

    n_accounts: int
    net: np.ndarray
    passed: int
    payout_count: np.ndarray
    death_reasons: Counter = field(default_factory=Counter)
    payout_blocks: Counter = field(default_factory=Counter)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n_accounts if self.n_accounts else 0.0

    @property
    def mean_net(self) -> float:
        return float(np.mean(self.net))

    @property
    def mean_net_ci(self) -> tuple[float, float]:
        """95% CI on the mean, from the standard error. Always report this."""
        n = self.net.size
        if n < 2:
            return (self.mean_net, self.mean_net)
        se = float(np.std(self.net, ddof=1)) / np.sqrt(n)
        return (self.mean_net - 1.96 * se, self.mean_net + 1.96 * se)

    @property
    def median_net(self) -> float:
        return float(np.median(self.net))

    @property
    def prob_profitable(self) -> float:
        """Share of accounts that ended up ahead.

        Report this next to the mean, always. These distributions are savagely
        right-skewed -- a positive mean routinely sits on top of a >90% chance
        that this particular account loses its fee.
        """
        return float(np.mean(self.net > 0))


def study_lifecycle(
    model: TradeModel,
    rs: Ruleset,
    *,
    n_accounts: int = 3_000,
    eval_days: int = 250,
    funded_days: int = 250,
    seed: int = 0,
    withdraw_at_profit: float = 0.0,
) -> LifecycleStudy:
    """Buy ``n_accounts`` accounts, trade each one, and total up the damage."""
    rng = np.random.default_rng(seed)
    nets = np.empty(n_accounts, dtype=float)
    counts = np.empty(n_accounts, dtype=np.int64)
    passed = 0
    deaths: Counter = Counter()
    blocks: Counter = Counter()

    for i in range(n_accounts):
        ee, ed = generate_path(model, eval_days, rs.starting_balance, rng)
        fe, fd = generate_path(model, funded_days, rs.starting_balance, rng)
        res = simulate_lifecycle(
            ee, ed, fe, fd, rs, withdraw_at_profit=withdraw_at_profit
        )
        nets[i] = res.net
        counts[i] = len(res.payouts)
        passed += int(res.passed_eval)
        if res.death_reason is not None:
            deaths[res.death_reason.value] += 1
        if res.payout_blocked_by:
            blocks[res.payout_blocked_by] += 1

    return LifecycleStudy(
        n_accounts=n_accounts,
        net=nets,
        passed=passed,
        payout_count=counts,
        death_reasons=deaths,
        payout_blocks=blocks,
    )
