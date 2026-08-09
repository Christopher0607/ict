from __future__ import annotations

import ict_lab.engine.run_verification as run_verification_mod


def test_run_verification_smoke(synthetic_bars, monkeypatch, capsys):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")

    def fake_load_symbol(symbol, price_series="backadjusted"):
        assert symbol == "NQ"
        assert price_series == "backadjusted"
        return df

    monkeypatch.setattr(run_verification_mod, "load_symbol", fake_load_symbol)

    run_verification_mod.run("NQ", "2024-06-03", "2024-06-10")

    out = capsys.readouterr().out
    assert "signals:" in out
    assert "trades:" in out
