"""Sample-size (``_n``) contract: every rate carries the count it is computed over."""

from __future__ import annotations

import polars as pl

from cfb_data_build.team_summaries import (
    _TEAM_MEAN_SOURCES,
    _TEAM_RATIO_DENOMINATORS,
    _clean_rank_columns,
    _summarize_team,
)


def _plays() -> pl.DataFrame:
    # team "1": 3 plays over 2 games / 2 drives; one red-zone play, one 3rd down,
    # one explosive play (so 2 non-explosive), one rush carrying line yards
    return pl.DataFrame(
        {
            "pos_team_id": ["1", "1", "1", "2"],
            "game_id": ["g1", "g1", "g2", "g1"],
            "drive_id": ["d1", "d1", "d2", "d3"],
            "pass": [1, 0, 1, 1],
            "rush": [0, 1, 0, 0],
            "havoc": [False, True, False, False],
            "explosive": [False, False, True, False],
            "EPA": [0.5, -0.2, 2.6, 0.1],
            "yards_gained": [8, -1, 30, 4],
            "play_stuffed": [False, True, False, False],
            "epa_success": [1.0, 0.0, 1.0, 1.0],
            "red_zone_success": [None, None, 1.0, None],
            "third_down_success": [1.0, None, None, None],
            "third_down_distance": [4.0, None, None, None],
            "late_down_success": [1.0, None, None, None],
            "early_down_EPA": [None, -0.2, 2.6, 0.1],
            "drive_start_yards_to_goal": [75, 75, 40, 60],
            "nonExplosiveEpa": [0.5, -0.2, None, 0.1],
            "line_yards": [None, -1.2, None, None],
            "opportunity_run": [False, False, False, False],
        }
    )


def test_every_team_rate_gets_the_count_it_was_computed_over():
    g = _summarize_team(_plays(), "pos_team_id", ascending=False).sort("pos_team_id")
    one = g.row(0, named=True)
    assert one["EPAplay_n"] == 3 and one["red_zone_success_n"] == 1
    assert one["third_down_success_n"] == 1 and one["third_down_distance_n"] == 1
    assert one["nonExplosiveEpaPerPlay_n"] == 2 and one["line_yards_n"] == 1
    assert one["EPAgame_n"] == 2 and one["EPAdrive_n"] == 2
    assert {c[:-2] for c in g.columns if c.endswith("_n")} == set(
        _TEAM_MEAN_SOURCES
    ) | set(_TEAM_RATIO_DENOMINATORS)
    assert all(g.schema[c] == pl.Int64 for c in g.columns if c.endswith("_n"))


def test_removed_metrics_take_their_sample_size_with_them():
    rc = ("start_position", "start_position_rank", "start_position_n")
    g = _summarize_team(_plays(), "pos_team_id", ascending=False, remove_cols=rc)
    assert not [c for c in g.columns if c.startswith("start_position")]


def test_n_suffix_moves_to_the_end_like_rank():
    df = pl.DataFrame(
        {"EPAplay_n_off_pass": [3], "TEPA_rank_off": [1.0], "plays_off": [3]}
    )
    assert _clean_rank_columns(df).columns == [
        "EPAplay_off_pass_n",
        "TEPA_off_rank",
        "plays_off",
    ]
