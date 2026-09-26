"""League baselines contract (``league_averages``).

Constructed frames on purpose: what is under test is which rows and columns count
(ids, ranks, percentiles and sample sizes never do; a NaN or null never does; a
non-qualifier never does) and how levels split -- not the season aggregation,
which the summaries parity suite owns.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_data_build.league_averages import SCHEMA, build_league_averages, summarize

LEVELS = {
    "fbs": None,
    "p4": pl.col("fbs_class").is_in(["P4", "P5"]),
    "g5": pl.col("fbs_class").is_in(["G5", "G6"]),
}
QUALIFIERS = {"passing": (pl.col("dropbacks") >= 14.0 * pl.col("team_games"), 14.0)}


def test_summarize_counts_only_finite_metric_values():
    df = pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "team_id": [1, 2, 3, 4],
            "EPAplay": [0.1, 0.2, 0.3, None],
            "yardsplay": [1.0, float("nan"), 3.0, 5.0],
            "EPAplay_rank": [3.0, 2.0, 1.0, 4.0],
            "EPAplay_pct": [25.0, 50.0, 75.0, None],
            "EPAplay_n": [10, 10, 10, 0],
            "conference": ["A", "A", "B", "B"],
        }
    )
    out = summarize(
        df, season=2025, level="fbs", entity="team", category="team_summaries"
    )
    assert list(out.schema.items()) == list(SCHEMA.items())
    assert out["metric"].to_list() == ["EPAplay", "yardsplay"]
    epa, ypp = out.row(0, named=True), out.row(1, named=True)
    assert epa["mean"] == pytest.approx(0.2) and epa["median"] == pytest.approx(0.2)
    assert (
        epa["sd"] == pytest.approx(0.1)
        and epa["n"] == 3
        and epa["qualifier_min"] is None
    )
    assert ypp["n"] == 3 and ypp["mean"] == pytest.approx(3.0)


def test_summarize_empty_frame_keeps_the_schema():
    empty = pl.DataFrame({"team_id": pl.Series([], dtype=pl.Int64)})
    out = summarize(
        empty, season=2025, level="fbs", entity="team", category="team_summaries"
    )
    assert out.height == 0 and list(out.schema.items()) == list(SCHEMA.items())


def test_players_are_gated_and_levels_split_on_fbs_class():
    passing = pl.DataFrame(
        {
            "team_id": [1, 2, 3, 4],
            "player_id": [11, 12, 13, 14],
            "team_games": [10, 10, 10, 10],
            "dropbacks": [200.0, 100.0, 300.0, 150.0],  # player 12 misses 14/team-game
            "EPAplay": [0.2, 0.9, 0.1, 0.3],
            "EPAplay_rank": [2.0, None, 3.0, 1.0],
            "EPAplay_n": [200, 100, 300, 150],
            "fbs_class": ["P4", "G6", "G6", None],  # a null class counts in fbs only
        }
    )
    out = build_league_averages(
        {"passing": passing}, 2025, levels=LEVELS, qualifiers=QUALIFIERS
    )
    ep = {
        r["level"]: r
        for r in out.filter(pl.col("metric") == "EPAplay").iter_rows(named=True)
    }
    assert {lv: r["n"] for lv, r in ep.items()} == {"fbs": 3, "p4": 1, "g5": 1}
    assert (
        ep["fbs"]["mean"] == pytest.approx(0.2) and ep["fbs"]["qualifier_min"] == 14.0
    )
    assert ep["p4"]["sd"] is None  # one qualifier: no spread
    assert set(out["entity"].unique()) == {"player"}
    assert sorted(out["metric"].unique().to_list()) == ["EPAplay", "dropbacks"]


def test_a_frame_without_the_level_column_only_reports_the_unfiltered_level():
    per_game = pl.DataFrame(
        {"game_id": ["g1", "g1"], "pos_team": ["A", "B"], "EPAplay": [0.1, 0.3]}
    )
    out = build_league_averages(
        {"team_game": per_game}, 2025, levels=LEVELS, qualifiers={}
    )
    assert out["level"].to_list() == ["fbs"] and out["category"].to_list() == [
        "team_game"
    ]


def test_league_averages_is_a_published_summaries_table():
    from cfb_data_build.config import PKG_FUNCTION, SUMMARIES_REGISTRY

    spec = SUMMARIES_REGISTRY["league_averages"]
    assert (spec.dataset, spec.stem, spec.tag) == (
        "league_averages",
        "cfb_league_averages",
        "cfb_league_averages",
    )
    assert (
        PKG_FUNCTION["cfb_league_averages"]
        == "python/cfb_data_build/league_averages.py"
    )


def test_player_qualifiers_are_the_leaderboard_gates():
    from cfb_data_build.team_summaries import PLAYER_QUALIFIERS

    assert {k: v[1] for k, v in PLAYER_QUALIFIERS.items()} == {
        "passing": 14.0,
        "rushing": 6.25,
        "receiving": 1.875,
    }


def test_team_game_median_is_the_percentile_ladders_50th():
    """The published percentiles table is the oracle for the team_game medians."""
    from cfb_data_build.team_summaries import LEVELS as TS_LEVELS
    from cfb_data_build.team_summaries import PERCENTILE_METRICS, _quantiles

    n = 9
    per_game = pl.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "pos_team": [f"t{i}" for i in range(n)],
            **{
                m: [float((i * (j + 3)) % 11) for i in range(n)]
                for j, m in enumerate(PERCENTILE_METRICS)
            },
        }
    ).with_columns(
        pl.when(pl.col("game_id") == "g0")
        .then(None)
        .otherwise(pl.col("pass_success"))
        .alias("pass_success")
    )
    la = build_league_averages(
        {"team_game": per_game}, 2025, levels=TS_LEVELS, qualifiers={}
    )
    mid = (
        _quantiles(per_game)
        .filter((pl.col("pctile") - 0.5).abs() < 1e-9)
        .row(0, named=True)
    )
    assert la["level"].unique().to_list() == ["fbs"]
    assert sorted(la["metric"].to_list()) == sorted(PERCENTILE_METRICS)
    for r in la.iter_rows(named=True):
        assert r["median"] == pytest.approx(mid[r["metric"]], abs=1e-9), r["metric"]
    assert la.filter(pl.col("metric") == "pass_success")["n"].item() == 8
