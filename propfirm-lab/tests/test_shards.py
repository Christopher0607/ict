"""Checkpointed search results: never recompute, never let shard boundaries show.

The point of this file is the second property. Resumability that changes the
output depending on where the run was interrupted is worse than no resumability,
because the difference is invisible until someone compares two files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.search import registry, shards


def _cfg(family="orb", n=0, **params):
    return registry.Config(family, {"side": "long", "seq": n, **params}, 1.5, 2.0)


def _row(cfg, trades=100, exp=0.01):
    return {
        "name": cfg.name, "family": cfg.family, "stop_atr": cfg.stop_atr,
        "target_r": cfg.target_r, "time_exit_bars": cfg.time_exit_bars,
        **cfg.params, "trades": trades, "expectancy_r": exp,
    }


def test_a_finished_config_is_not_recomputed(tmp_path):
    cfgs = [_cfg(n=i) for i in range(5)]
    shards.flush_shard(tmp_path, [_row(c) for c in cfgs[:3]])
    done = shards.done_names(tmp_path)
    assert [c.name for c in cfgs if c.name not in done] == [c.name for c in cfgs[3:]]


def test_merge_is_independent_of_how_the_run_was_chopped_up(tmp_path):
    """One shard of 20 and four shards of 5 have to merge to the same frame."""
    cfgs = [_cfg(n=i) for i in range(20)]
    rows = [_row(c, trades=100 + i, exp=i / 1000) for i, c in enumerate(cfgs)]

    one = tmp_path / "one"
    shards.flush_shard(one, rows)

    many = tmp_path / "many"
    for start in (10, 0, 15, 5):          # written out of order, on purpose
        shards.flush_shard(many, rows[start:start + 5])

    a, b = shards.merge(one, cfgs), shards.merge(many, cfgs)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["name"]) == [c.name for c in cfgs]


def test_a_duplicated_config_keeps_one_row(tmp_path):
    """A crash between compute and flush recomputes a config; the merge dedupes."""
    cfgs = [_cfg(n=i) for i in range(3)]
    shards.flush_shard(tmp_path, [_row(c) for c in cfgs])
    shards.flush_shard(tmp_path, [_row(cfgs[1], trades=999)])
    out = shards.merge(tmp_path, cfgs)
    assert len(out) == 3
    assert out.loc[out["name"] == cfgs[1].name, "trades"].item() == 999


def test_a_truncated_shard_is_ignored(tmp_path):
    """A killed writer leaves a .part file; it must not be read as done."""
    cfgs = [_cfg(n=i) for i in range(3)]
    shards.flush_shard(tmp_path, [_row(cfgs[0])])
    (tmp_path / "shard_deadbeef.parquet.part").write_bytes(b"not a parquet file")
    assert shards.done_names(tmp_path) == {cfgs[0].name}


def test_unrun_configs_show_up_as_missing_not_dropped(tmp_path):
    cfgs = [_cfg(n=i) for i in range(4)]
    shards.flush_shard(tmp_path, [_row(c) for c in cfgs[:2]])
    out = shards.merge(tmp_path, cfgs)
    assert len(out) == 4
    assert out["trades"].isna().sum() == 2


def test_column_order_does_not_depend_on_which_family_ran_first(tmp_path):
    """Two families with different parameters, flushed in either order."""
    a = registry.Config("orb", {"or_minutes": 15, "side": "long"}, 1.5, 2.0)
    b = registry.Config("momentum", {"lookback": 5, "side": "short"}, 1.5, 2.0)
    cfgs = [a, b]

    first = tmp_path / "ab"
    shards.flush_shard(first, [_row(a)])
    shards.flush_shard(first, [_row(b)])

    second = tmp_path / "ba"
    shards.flush_shard(second, [_row(b)])
    shards.flush_shard(second, [_row(a)])

    pd.testing.assert_frame_equal(shards.merge(first, cfgs), shards.merge(second, cfgs))


def test_empty_shard_dir_is_empty_not_an_error(tmp_path):
    cfgs = [_cfg(n=0)]
    out = shards.merge(tmp_path / "nothing_here", cfgs)
    assert len(out) == 1 and out["trades"].isna().all()


def test_time_exit_stays_out_of_the_name_when_unset():
    """Round-1 names are already on disk and in findings; they must not move."""
    plain = registry.Config("orb", {"or_minutes": 15}, 1.5, 2.0)
    timed = registry.Config("orb", {"or_minutes": 15}, 1.5, 2.0, time_exit_bars=60)
    assert plain.name == "orb[or_minutes=15]stop=1.5R=2.0"
    assert timed.name == "orb[or_minutes=15]stop=1.5R=2.0t=60"
