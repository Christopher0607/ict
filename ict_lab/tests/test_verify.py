from __future__ import annotations

from ict_lab.data import loader as loader_mod
from ict_lab.data.verify import show_window
from ict_lab.tests.test_loader import _write_symbol_files


def test_show_window_returns_requested_slice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(loader_mod, "RAW_DIR", tmp_path)
    index = _write_symbol_files(tmp_path, "NQ", n=200)

    window = show_window(
        "NQ",
        str(index[10]),
        str(index[20]),
        price_series="unadjusted",
        include_holdout=True,
    )

    assert len(window) == 11  # inclusive of both endpoints
    assert window.index.min() == index[10]
    assert window.index.max() == index[20]
    captured = capsys.readouterr()
    assert "NQ unadjusted" in captured.out
