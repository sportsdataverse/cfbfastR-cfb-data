"""Unit tests for the CFB punt end-yardline distribution builder (nfl4th punt_df analog)."""
from __future__ import annotations

import polars as pl

from model_training import punt_distribution as pd_mod


def _punts() -> pl.DataFrame:
    """Synthetic punts: net = yds_punted - yds_punt_return; ytg_end = ytg - net."""
    return pl.DataFrame({
        "punt": [True, True, True, True, False],
        "yds_punted": [40.0, 45.0, 38.0, 50.0, None],   # last row not a punt
        "yds_punt_return": [5.0, 0.0, 10.0, 0.0, None],
        "start.yardsToEndzone": [70, 60, 80, 95, 50],
        "punt_tb": [False, False, False, True, False],
    })


def test_punt_outcomes_net_and_touchback():
    out = pd_mod._punt_outcomes(_punts()).sort("yards_to_goal")
    rows = {r["yards_to_goal"]: r["yards_to_goal_end"] for r in out.iter_rows(named=True)}
    # ytg 60, net 45 -> end 15 ; ytg 70, net 35 -> end 35 ; ytg 80, net 28 -> end 52
    assert rows[60] == 15
    assert rows[70] == 35
    assert rows[80] == 52
    # ytg 95 is a touchback -> receiving team at own 25 -> yards_to_goal_end == 25
    assert rows[95] == 25
    # the non-punt row is excluded
    assert out.height == 4


def test_outcomes_exclude_out_of_range_start():
    df = _punts().with_columns(pl.Series("start.yardsToEndzone", [10, 25, 80, 95, 50]))
    out = pd_mod._punt_outcomes(df)
    # ytg 10 and 25 are below PUNT_YTG_MIN(31) -> dropped; only 80 + 95(tb) remain
    assert set(out["yards_to_goal"].to_list()) == {80, 95}


def test_build_distribution_normalizes_per_yardline():
    # many punts from one yardline so the KDE has support; pct must sum to 1 per ytg
    n = 200
    df = pl.DataFrame({
        "punt": [True] * n,
        "yds_punted": [40.0 + (i % 11) for i in range(n)],
        "yds_punt_return": [float(i % 7) for i in range(n)],
        "start.yardsToEndzone": [50 + (i % 20) for i in range(n)],
        "punt_tb": [False] * n,
    })
    dist = pd_mod.build_punt_distribution(df, bw_method=0.4)
    assert dist.columns == ["yards_to_goal", "yards_to_goal_end", "pct"]
    sums = dist.group_by("yards_to_goal").agg(s=pl.col("pct").sum())
    assert ((sums["s"] - 1.0).abs() < 1e-9).all()
    assert dist["yards_to_goal"].min() >= pd_mod.PUNT_YTG_MIN
