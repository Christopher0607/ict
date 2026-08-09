from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostModel:
    tick_size: float
    tick_value: float
    commission_round_turn: float
    stop_slippage_ticks: float


COST_MODELS: dict[str, CostModel] = {
    "NQ": CostModel(tick_size=0.25, tick_value=5.00, commission_round_turn=4.00, stop_slippage_ticks=1.0),
    "ES": CostModel(tick_size=0.25, tick_value=12.50, commission_round_turn=4.00, stop_slippage_ticks=1.0),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Every Phase 3 knob for the signal pipeline and execution layer. One
    window only, one trade per session per window -- Phase 4 is what
    introduces multi-window configs and multiple trades per window; adding
    those fields here now would be building ahead of the phase that defines
    them.
    """

    name: str
    window: str

    bias_method: str = "none"  # none | prior_day | swing_structure | daily_ma_slope | perfect
    bias_timeframe: str = "15m"  # swing_structure only
    bias_swing_n: int = 5  # swing_structure only
    bias_ma_period: int = 20  # daily_ma_slope only

    sweep_required: bool = True
    sweep_level_types: tuple[str, ...] = ()
    sweep_k: int = 3
    sweep_min_penetration_ticks: float = 1.0

    mss_required: bool = False
    mss_break_style: str = "close"  # close | wick

    displacement_required: bool = True
    displacement_atr_mult: float = 1.5

    fvg_timeframe: str = "5m"
    fvg_min_size_points: float = 0.0
    fvg_min_size_atr_mult: float = 0.0

    swing_n: int = 5  # liquidity levels + MSS reference swings

    entry_level: str = "50%"  # proximal | 50% | distal
    stop_type: str = "swing"  # swing | gap_distal | fixed_points
    stop_fixed_points: float = 0.0
    stop_buffer_ticks: float = 1.0

    target_type: str = "fixed_r"  # fixed_r | next_liquidity | time
    target_r_multiple: float = 2.0
    target_fallback_r_multiple: float = 2.0  # next_liquidity, when no level qualifies
    target_time_bars: int = 0  # time only

    hard_exit: str = "window_end"  # window_end | rth_end

    max_trades_per_window: int = 1

    def __post_init__(self) -> None:
        if self.entry_level not in ("proximal", "50%", "distal"):
            raise ValueError(f"entry_level must be proximal/50%/distal, got {self.entry_level!r}")
        if self.stop_type not in ("swing", "gap_distal", "fixed_points"):
            raise ValueError(f"stop_type must be swing/gap_distal/fixed_points, got {self.stop_type!r}")
        if self.stop_type == "swing" and not self.sweep_required:
            raise ValueError("stop_type='swing' needs sweep_required=True -- no swept level to reference")
        if self.stop_type == "fixed_points" and self.stop_fixed_points <= 0:
            raise ValueError("stop_type='fixed_points' needs stop_fixed_points > 0")
        if self.target_type not in ("fixed_r", "next_liquidity", "time"):
            raise ValueError(f"target_type must be fixed_r/next_liquidity/time, got {self.target_type!r}")
        if self.target_type == "time" and self.target_time_bars <= 0:
            raise ValueError("target_type='time' needs target_time_bars > 0")
        if self.hard_exit not in ("window_end", "rth_end"):
            raise ValueError(f"hard_exit must be window_end/rth_end, got {self.hard_exit!r}")
        if self.bias_method not in ("none", "prior_day", "swing_structure", "daily_ma_slope", "perfect"):
            raise ValueError(f"unknown bias_method {self.bias_method!r}")
        if self.mss_break_style not in ("close", "wick"):
            raise ValueError(f"mss_break_style must be close/wick, got {self.mss_break_style!r}")
        if self.sweep_required and not self.sweep_level_types:
            raise ValueError("sweep_required=True needs a non-empty sweep_level_types")
        if self.max_trades_per_window != 1:
            raise ValueError("Phase 3 scope is exactly 1 trade per session per window")

    def to_dict(self) -> dict:
        return asdict(self)


# Phase 3's VERIFICATION section: "use the consensus parameters from
# configs/grid.json; if not yet defined there, take: 5m FVG, no min gap
# size, sweep required on prior-session or pre-window H/L with K=3 and 1
# tick penetration, displacement 1.5x ATR(14), MSS not required, entry at
# 50% of the gap, stop beyond the displacement swing, target 2R, NY AM
# window, no bias." grid.json doesn't exist yet, so this is that baseline.
CONSENSUS_CONFIG = StrategyConfig(
    name="consensus",
    window="killzone_ny_am",
    bias_method="none",
    sweep_required=True,
    sweep_level_types=(
        "prior_session_high",
        "prior_session_low",
        "pre_killzone_ny_am_high",
        "pre_killzone_ny_am_low",
    ),
    sweep_k=3,
    sweep_min_penetration_ticks=1.0,
    mss_required=False,
    displacement_required=True,
    displacement_atr_mult=1.5,
    fvg_timeframe="5m",
    fvg_min_size_points=0.0,
    entry_level="50%",
    stop_type="swing",
    target_type="fixed_r",
    target_r_multiple=2.0,
)
