from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.features.displacement import displacement_atr
from ict_lab.features.fvg import detect_fvg
from ict_lab.features.liquidity import all_liquidity_levels
from ict_lab.features.mss import detect_mss_after_sweeps
from ict_lab.features.plotting import generate_review_charts, plot_killzone_chart
from ict_lab.features.sweep import detect_sweeps
from ict_lab.features.swings import swing_points
from ict_lab.data.sessions import add_session_columns


@pytest.fixture
def full_pipeline(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    fvgs = detect_fvg(df, timeframe="5m")
    levels = all_liquidity_levels(df)
    swings = swing_points(df, n=5)
    sweeps = detect_sweeps(df, levels, k=3, min_penetration_ticks=1, tick_size=0.25)
    mss = detect_mss_after_sweeps(df, swings, sweeps, break_style="close")
    disp = displacement_atr(df, atr_mult=1.5)
    return df, fvgs, levels, sweeps, disp, mss


def _a_session(df, window="killzone_ny_am"):
    with_sessions = add_session_columns(df)
    return sorted(with_sessions.loc[with_sessions[window], "session_date"].unique())[0]


def test_plot_killzone_chart_with_all_features(full_pipeline, tmp_path):
    df, fvgs, levels, sweeps, disp, mss = full_pipeline
    session = _a_session(df)
    out = tmp_path / "chart.png"

    result = plot_killzone_chart(
        df, session, "killzone_ny_am", fvgs=fvgs, levels=levels, sweeps=sweeps,
        displacement=disp, mss_events=mss, save_path=out,
    )

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 1000  # a real image, not an empty/broken file


def test_plot_killzone_chart_with_no_features(full_pipeline, tmp_path):
    df, *_ = full_pipeline
    session = _a_session(df)
    out = tmp_path / "bare.png"

    result = plot_killzone_chart(df, session, "killzone_ny_am", save_path=out)

    assert result.exists()
    assert result.stat().st_size > 1000


def test_plot_killzone_chart_raises_on_empty_window(full_pipeline):
    df, *_ = full_pipeline
    with pytest.raises(ValueError, match="No bars"):
        plot_killzone_chart(df, pd.Timestamp("1999-01-01"), "killzone_ny_am")


def test_generate_review_charts_produces_requested_count(full_pipeline, tmp_path, monkeypatch):
    df, fvgs, levels, sweeps, disp, mss = full_pipeline
    monkeypatch.chdir(tmp_path)

    import ict_lab.features.plotting as plotting_mod

    monkeypatch.setattr(plotting_mod, "CHART_DIR", tmp_path / "charts")

    paths = generate_review_charts(
        df, fvgs, levels, sweeps, disp, mss, window="killzone_ny_am", n=2, seed=0
    )

    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_generate_review_charts_deterministic_with_same_seed(full_pipeline, tmp_path, monkeypatch):
    df, fvgs, levels, sweeps, disp, mss = full_pipeline
    import ict_lab.features.plotting as plotting_mod

    monkeypatch.setattr(plotting_mod, "CHART_DIR", tmp_path / "run_a")
    paths_a = generate_review_charts(df, fvgs, levels, sweeps, disp, mss, n=2, seed=7)

    monkeypatch.setattr(plotting_mod, "CHART_DIR", tmp_path / "run_b")
    paths_b = generate_review_charts(df, fvgs, levels, sweeps, disp, mss, n=2, seed=7)

    assert [p.name for p in paths_a] == [p.name for p in paths_b]


def test_generate_review_charts_caps_at_available_sessions(full_pipeline, tmp_path, monkeypatch):
    df, fvgs, levels, sweeps, disp, mss = full_pipeline
    import ict_lab.features.plotting as plotting_mod

    monkeypatch.setattr(plotting_mod, "CHART_DIR", tmp_path / "charts")
    paths = generate_review_charts(df, fvgs, levels, sweeps, disp, mss, n=1000, seed=0)

    with_sessions = add_session_columns(df)
    available = with_sessions.loc[with_sessions["killzone_ny_am"], "session_date"].nunique()
    assert len(paths) == available
