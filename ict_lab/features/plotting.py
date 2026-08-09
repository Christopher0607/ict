"""Phase 2's final piece: a candlestick chart for a session/killzone window,
using the UNADJUSTED series (so prices match what a human sees on a live
chart), with every detected feature overlaid -- FVGs as shaded boxes,
liquidity levels as horizontal lines, sweeps as markers, displacement bars
highlighted, MSS as a vertical line. Meant for visual sanity-checking the
detectors against real data; generate_review_charts() only produces
something worth eyeballing once real ES/NQ bars are available -- there's
nothing a human would recognize in synthetic random-walk data.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from ict_lab.data.sessions import add_session_columns

CHART_DIR = Path(__file__).resolve().parents[1] / "analysis" / "charts"


def plot_killzone_chart(
    df_unadjusted: pd.DataFrame,
    session_date: pd.Timestamp,
    window: str,
    fvgs: pd.DataFrame | None = None,
    levels: pd.DataFrame | None = None,
    sweeps: pd.DataFrame | None = None,
    displacement: pd.Series | None = None,
    mss_events: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> Path:
    """All feature arguments are optional, so this also works to eyeball raw
    price action before every detector has something to show for a window.
    """
    with_sessions = add_session_columns(df_unadjusted)
    scoped = with_sessions[(with_sessions["session_date"] == session_date) & with_sessions[window]]
    if scoped.empty:
        raise ValueError(f"No bars for session_date={session_date}, window={window!r}.")

    ohlc = scoped[["open", "high", "low", "close", "volume"]].rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    start, end = ohlc.index.min(), ohlc.index.max()

    # Liquidity levels accumulate over the whole history (every swing point
    # ever confirmed, etc.) -- without pruning, the chart drowns in dozens of
    # lines that are either priced far away, already swept and no longer a
    # live reference, or just too dense to read (a noisy series can put many
    # swings in any given price band). Keep only what's actually relevant to
    # eyeball here: nearby, still-active, and capped to a readable count.
    price_lo, price_hi = ohlc["Low"].min(), ohlc["High"].max()
    margin = max(price_hi - price_lo, 1e-9) * 3
    max_levels_shown = 20
    hlines, hline_colors = [], []
    if levels is not None and not levels.empty:
        nearby = levels[
            (levels["knowable_at"] <= end)
            & (levels["price"] >= price_lo - margin)
            & (levels["price"] <= price_hi + margin)
        ]
        if sweeps is not None and not sweeps.empty:
            already_swept = sweeps.loc[sweeps["confirmed_at"] < start, ["level_type", "level_price"]]
            if not already_swept.empty:
                swept_keys = set(zip(already_swept["level_type"], already_swept["level_price"]))
                nearby = nearby[
                    ~nearby.apply(lambda r: (r["level_type"], r["price"]) in swept_keys, axis=1)
                ]
        if len(nearby) > max_levels_shown:
            last_close = ohlc["Close"].iloc[-1]
            nearby = nearby.assign(_dist=(nearby["price"] - last_close).abs()).nsmallest(
                max_levels_shown, "_dist"
            )
        for _, lvl in nearby.iterrows():
            hlines.append(lvl["price"])
            hline_colors.append("darkorange" if lvl["level_type"].endswith("_high") else "teal")

    vlines = []
    if mss_events is not None and not mss_events.empty:
        vlines = [t for t in mss_events["broken_at"] if start <= t <= end]

    addplots = []
    if sweeps is not None and not sweeps.empty:
        in_range = sweeps[(sweeps["confirmed_at"] >= start) & (sweeps["confirmed_at"] <= end)]
        if not in_range.empty:
            marker_series = pd.Series(float("nan"), index=ohlc.index)
            for _, sw in in_range.iterrows():
                if sw["confirmed_at"] in marker_series.index:
                    marker_series.loc[sw["confirmed_at"]] = sw["penetration_price"]
            if marker_series.notna().any():
                addplots.append(
                    mpf.make_addplot(marker_series, type="scatter", markersize=100, marker="x", color="red")
                )

    # mplfinance rejects these kwargs being explicitly passed as None -- they
    # must be omitted entirely when there's nothing to draw.
    plot_kwargs = dict(
        type="candle",
        style="charles",
        returnfig=True,
        figsize=(14, 7),
        title=f"{session_date.date()} {window}",
        volume=False,
    )
    if hlines:
        plot_kwargs["hlines"] = dict(hlines=hlines, colors=hline_colors, linestyle="--", linewidths=0.8)
    if vlines:
        plot_kwargs["vlines"] = dict(vlines=vlines, colors="purple", linewidths=1.2)
    if addplots:
        plot_kwargs["addplot"] = addplots

    fig, axes = mpf.plot(ohlc, **plot_kwargs)
    ax = axes[0]
    n_bars = len(ohlc)

    if fvgs is not None and not fvgs.empty:
        in_range = fvgs[(fvgs["knowable_at"] >= start) & (fvgs["knowable_at"] <= end)]
        for _, gap in in_range.iterrows():
            gap_end = gap["filled_at"] if pd.notna(gap["filled_at"]) else end
            gap_end = min(gap_end, end)
            x0 = ohlc.index.get_indexer([gap["knowable_at"]], method="nearest")[0]
            x1 = max(ohlc.index.get_indexer([gap_end], method="nearest")[0], x0 + 1)
            color = "lightgreen" if gap["direction"] == "bullish" else "lightcoral"
            ax.axhspan(
                gap["gap_bottom"], gap["gap_top"], xmin=x0 / n_bars, xmax=x1 / n_bars, color=color, alpha=0.3
            )

    if displacement is not None:
        flagged = displacement.reindex(ohlc.index).fillna(False)
        for i, is_flagged in enumerate(flagged.to_numpy()):
            if is_flagged:
                ax.axvspan(i - 0.4, i + 0.4, color="gold", alpha=0.25)

    save_path = save_path or (CHART_DIR / f"{pd.Timestamp(session_date).date()}_{window}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def generate_review_charts(
    df_unadjusted: pd.DataFrame,
    fvgs: pd.DataFrame,
    levels: pd.DataFrame,
    sweeps: pd.DataFrame,
    displacement: pd.Series,
    mss_events: pd.DataFrame,
    window: str = "killzone_ny_am",
    n: int = 15,
    seed: int = 0,
) -> list[Path]:
    """Phase 2's final step: charts for n random days so the detected
    features can be checked by eye against what a human would mark. Run
    this once real data lands in ict_lab/data/raw/ -- it will happily run
    against synthetic fixtures too, but there is nothing recognizable to
    confirm in random-walk bars, so doing so proves nothing.
    """
    with_sessions = add_session_columns(df_unadjusted)
    candidates = sorted(with_sessions.loc[with_sessions[window], "session_date"].unique())
    if not candidates:
        return []
    chosen = random.Random(seed).sample(candidates, k=min(n, len(candidates)))

    return [
        plot_killzone_chart(df_unadjusted, session_date, window, fvgs, levels, sweeps, displacement, mss_events)
        for session_date in chosen
    ]
