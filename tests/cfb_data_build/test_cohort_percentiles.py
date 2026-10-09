"""Cohort percentiles: ``_conf_pct`` (teams by conference), ``_pos_pct`` (players by position group).

Constructed frames on purpose, like ``test_leader_percentiles.py``: what is under
test is an arithmetic contract -- the rank WITHIN the cohort, the Weibull scale,
the cohort floors and the nulls. The league-wide ``_rank`` the cohort rank is cut
from is built with the producer's own ``_rank`` / ``_attach_leader_ranks``, and the
two conferences interleave, so a league rank passed through unchanged would fail.
"""

from __future__ import annotations

import polars as pl
import pytest
import sportsdataverse.cfb

from cfb_data_build import summaries_build
from cfb_data_build.team_summaries import (
    MIN_COHORT_PLAYERS,
    MIN_COHORT_TEAMS,
    _attach_cohort_percentiles,
    _attach_conference_percentiles,
    _attach_leader_ranks,
    _attach_position_cohorts,
    _rank,
)

# conference A: 6 teams; B: 4 teams (below the floor), interleaved with A's values
A_EPA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
B_EPA = [0.15, 0.35, 0.55, 0.65]
A_HAVOC = [0.20, 0.10, 0.25, 0.15, 0.30, 0.05]
B_HAVOC = [0.12, 0.22, 0.08, 0.18]


def _teams(
    a_epa: list[float | None] = A_EPA,
    conferences: list[str | None] | None = None,
) -> pl.DataFrame:
    n = len(a_epa) + len(B_EPA)
    return pl.DataFrame(
        {
            "team_id": list(range(1, n + 1)),
            "conference": conferences or ["A"] * len(a_epa) + ["B"] * len(B_EPA),
            "EPAplay_off": a_epa + B_EPA,
            "havoc_off": A_HAVOC + B_HAVOC,
        }
    ).with_columns(
        # league-wide, as _summarize_team builds them: high EPA, low havoc is best
        EPAplay_off_rank=_rank("EPAplay_off", descending=True),
        havoc_off_rank=_rank("havoc_off", descending=False),
    )


def _conf(df: pl.DataFrame) -> pl.DataFrame:
    return _attach_cohort_percentiles(
        df,
        cohort="conference",
        rank_cols=["EPAplay_off_rank", "havoc_off_rank"],
        suffix="_conf_pct",
        min_cohort=MIN_COHORT_TEAMS,
    )


def test_conference_percentile_is_weibull_within_the_conference():
    out = _conf(_teams()).filter(pl.col("conference") == "A").sort("EPAplay_off")
    pct = out["EPAplay_off_conf_pct"].to_list()

    assert pct[-1] == pytest.approx(100 * 6 / 7)  # the 0.6 team
    assert pct[0] == pytest.approx(100 * 1 / 7)  # the 0.1 team
    assert pct == pytest.approx([100 * k / 7 for k in range(1, 7)])


def test_a_conference_below_the_floor_gets_nulls():
    assert MIN_COHORT_TEAMS > len(B_EPA)
    out = _conf(_teams()).filter(pl.col("conference") == "B")

    assert out["EPAplay_off_conf_pct"].null_count() == out.height
    assert out["havoc_off_conf_pct"].null_count() == out.height


def test_a_null_metric_is_null_and_drops_out_of_n():
    a = [0.1, 0.2, None, 0.4, 0.5, 0.6]
    out = _conf(_teams(a_epa=a)).filter(pl.col("conference") == "A").sort("team_id")
    pct = out["EPAplay_off_conf_pct"].to_list()

    # _rank leaves the null unranked (C4), and the cohort percentile follows
    assert out["EPAplay_off_rank"][2] is None
    assert pct[2] is None
    assert [p for p in pct if p is not None] == pytest.approx(
        [100 * k / 6 for k in range(1, 6)]
    )


def test_a_low_is_good_rank_needs_no_direction_of_its_own():
    out = _conf(_teams()).filter(pl.col("conference") == "A")
    best = out.sort("havoc_off").row(0, named=True)

    assert best["havoc_off"] == min(A_HAVOC)
    assert best["havoc_off_conf_pct"] == pytest.approx(100 * 6 / 7)
    assert best["havoc_off_conf_pct"] == out["havoc_off_conf_pct"].max()


def test_a_null_cohort_key_is_null_and_not_a_cohort_of_its_own():
    # five conference-less teams would clear the floor if null were a group
    conferences = ["A"] * 5 + [None] * 5
    out = _conf(_teams(conferences=conferences))
    no_conf = out.filter(pl.col("conference").is_null())

    assert no_conf.height == 5 >= MIN_COHORT_TEAMS
    assert no_conf["EPAplay_off_conf_pct"].null_count() == 5
    assert (
        out.filter(pl.col("conference") == "A")["EPAplay_off_conf_pct"].null_count()
        == 0
    )


def test_fbs_independents_are_no_conference_cohort():
    df = pl.DataFrame(
        {
            "team_id": list(range(1, 13)),
            "conference": ["A"] * 6 + ["FBS Independents"] * 6,
            # interleaved, so independents counted into A's n would move A's values
            "EPAplay_off": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        }
    ).with_columns(EPAplay_off_rank=_rank("EPAplay_off", descending=True))
    out = _attach_conference_percentiles(df)
    ind = out.filter(pl.col("conference") == "FBS Independents")
    a = out.filter(pl.col("conference") == "A").sort("EPAplay_off")

    assert ind["EPAplay_off"].null_count() == 0  # the metric is there...
    assert ind["EPAplay_off_conf_pct"].null_count() == 6  # ...the cohort is not
    assert a["EPAplay_off_conf_pct"].to_list() == pytest.approx(
        [100 * k / 7 for k in range(1, 7)]
    )
    # the published conference column is untouched and the temp key is gone
    assert out["conference"].to_list() == df["conference"].to_list()
    assert out.columns == [*df.columns, "EPAplay_off_conf_pct"]


# ---- players ---------------------------------------------------------------

QB_IDS = list(range(1, 12))  # 11 qualified QBs
RB_IDS = [12, 13]  # 2 qualified RBs
NONQUAL_QB = 14  # a QB under the rushing gate
NO_ROSTER = 15  # a qualifier the roster does not list
QB_EPA = [0.31, -0.05, 0.12, 0.44, 0.02, 0.27, -0.11, 0.19, 0.08, 0.36, 0.23]


def _rushers() -> pl.DataFrame:
    ids = QB_IDS + RB_IDS + [NONQUAL_QB, NO_ROSTER]
    df = pl.DataFrame(
        {
            "pos_team_id": ids,
            "player_id": ids,
            "plays": [50] * 13 + [1, 50],
            "EPAplay": QB_EPA + [0.5, 0.4, 0.9, 0.6],
        }
    )
    return _attach_leader_ranks(
        df,
        keys=["pos_team_id", "player_id"],
        min_expr=pl.col("plays") >= 5,
        rank_cols=["EPAplay"],
    )


ROSTER = pl.DataFrame(
    {
        "athlete_id": QB_IDS + RB_IDS + [NONQUAL_QB],
        "position_abbreviation": ["QB"] * 11 + ["RB", "RB"] + ["QB"],
    }
)


def test_position_percentile_among_qualified_qbs_only():
    assert MIN_COHORT_PLAYERS > len(RB_IDS)
    out = _attach_position_cohorts(_rushers(), ROSTER)
    qbs = out.filter(pl.col("player_id").is_in(QB_IDS)).sort("EPAplay")

    assert qbs["position_group"].to_list() == ["QB"] * 11
    # n = 11 qualified QBs: the non-qualifier QB is not in n
    assert qbs["EPAplay_pos_pct"].to_list() == pytest.approx(
        [100 * k / 12 for k in range(1, 12)]
    )

    by_id = {r["player_id"]: r for r in out.iter_rows(named=True)}
    assert all(by_id[i]["EPAplay_pos_pct"] is None for i in RB_IDS)  # 2 < floor
    assert by_id[NONQUAL_QB]["position_group"] == "QB"
    assert by_id[NONQUAL_QB]["EPAplay_pos_pct"] is None
    assert by_id[NO_ROSTER]["position_group"] is None
    assert by_id[NO_ROSTER]["EPAplay_pos_pct"] is None


def test_position_groups_fold_fb_into_rb_and_the_rest_into_other():
    roster = pl.DataFrame(
        {
            "athlete_id": [1, 2, 3, 4, 5, 6, 7],
            "position_abbreviation": ["QB", "RB", "FB", "WR", "TE", "OL", "-"],
        }
    )
    df = pl.DataFrame({"player_id": [1, 2, 3, 4, 5, 6, 7]})
    out = _attach_position_cohorts(df, roster).sort("player_id")

    # ESPN's "-" is an unknown position, not "other"
    assert out["position_group"].to_list() == [
        "QB",
        "RB",
        "RB",
        "WR",
        "TE",
        "other",
        None,
    ]


def test_an_athlete_listed_twice_never_fans_out_a_row():
    roster = pl.DataFrame(
        {
            "athlete_id": [1, 1, 2, 2, 3, 3],
            "position_abbreviation": ["QB", "QB", "-", "WR", "QB", "WR"],
        }
    )
    df = pl.DataFrame({"player_id": [1, 2, 3]})
    out = _attach_position_cohorts(df, roster).sort("player_id")

    assert out.height == 3
    # same group twice (a transfer) counts once; "-" defers to the known one;
    # two different groups are ambiguous and stay null
    assert out["position_group"].to_list() == ["QB", "WR", None]


def test_a_string_roster_id_joins_through_an_integer_parse():
    roster = ROSTER.with_columns(pl.col("athlete_id").cast(pl.Utf8))
    out = _attach_position_cohorts(_rushers(), roster)

    assert out.schema["player_id"] == pl.Int64
    assert (
        out.filter(pl.col("player_id").is_in(QB_IDS))["position_group"].null_count()
        == 0
    )


@pytest.mark.parametrize(
    "roster_dtype,board_dtype",
    [(pl.Float64, pl.Int64), (pl.Int64, pl.Float64), (pl.Float64, pl.Utf8)],
)
def test_a_float_id_raises_before_the_join(roster_dtype, board_dtype):
    roster = ROSTER.with_columns(pl.col("athlete_id").cast(roster_dtype))
    board = _rushers().with_columns(pl.col("player_id").cast(board_dtype))

    with pytest.raises(TypeError, match="never a float"):
        _attach_position_cohorts(board, roster)


def test_no_roster_leaves_position_group_and_pos_pct_null_with_the_schema():
    out = _attach_position_cohorts(_rushers(), pl.DataFrame())

    assert out.schema["position_group"] == pl.Utf8
    assert out.schema["EPAplay_pos_pct"] == pl.Float64
    assert out["EPAplay_pos_pct"].null_count() == out.height


def test_season_roster_reads_local_build_output_before_the_release(
    tmp_path, monkeypatch
):
    local = tmp_path / "cfb_rosters" / "parquet"
    local.mkdir(parents=True)
    pl.DataFrame({"athlete_id": [1], "position_abbreviation": ["QB"]}).write_parquet(
        local / "cfb_rosters_2025.parquet"
    )
    calls: list = []

    def fake_release(seasons):
        calls.append(seasons)
        return pl.DataFrame({"athlete_id": [2], "position_abbreviation": ["WR"]})

    monkeypatch.setattr(sportsdataverse.cfb, "load_cfb_rosters", fake_release)

    assert summaries_build._load_season_rosters(2025, base=str(tmp_path))[
        "athlete_id"
    ].to_list() == [1]
    assert not calls
    assert summaries_build._load_season_rosters(2024, base=str(tmp_path))[
        "athlete_id"
    ].to_list() == [2]
    assert calls == [[2024]]
