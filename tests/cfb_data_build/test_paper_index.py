"""Paper Index per game (stage 67) and the luck columns on team_summaries / _weekly.

Real rows only: tests/fixtures/paper_index/README.md has the provenance and the
hand numbers behind every literal here. Nothing in this module can publish or
reach the network; both are stubbed to fail for every test.
"""

from __future__ import annotations

import math
import socket
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sportsdataverse import paper_index as t1

import cfb_data_build.summaries_build as sb
from cfb_data_build import league_averages
from cfb_data_build.cli import DERIVED
from cfb_data_build.config import PKG_FUNCTION, SUMMARIES_REGISTRY
from cfb_data_build.derived import BUILDERS, SPECS, build_derived
from cfb_data_build.paper_index import (
    LUCK_COLUMNS,
    OUTPUT_SCHEMA,
    attach_luck,
    build_paper_index_games,
    paper_index_games_table,
    paper_index_span,
)
from cfb_data_build.publish import RELEASE_NOTES
from cfb_model_build.cfb_higher_models import data as hm_data
from cfb_model_build.cfb_higher_models.train_game import (
    LEAN_FAMILIES,
    family_of,
    lean_features,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "paper_index"
ASU, OKST, UCF, TEXAS = 9, 197, 2116, 251
TOLEDO, NORFOLK = 2649, 2450
#: Arizona State 2024: at Oklahoma State (week 10), UCF (week 11), Peach Bowl vs Texas
WK10, WK11, BOWL = 401636915, 401636917, 401677182
#: Toledo 2021: Norfolk State (11 and 18 snaps in the feed), at Notre Dame
UNDER_FLOOR, AT_ND = 401309543, 401282705
#: Arizona State's share in each of its three games (README, "hand numbers")
P_WK10, P_WK11, P_BOWL = 0.70925875798001, 0.12240098401069241, 0.11377000178908554
#: Game on Paper's own module on the same rows, home (Oklahoma State) share of WK10
GOP_WK10_HOME_SHARE = 0.2907412420199904
#: Toledo's share at Notre Dame, a 29-32 loss
P_AT_ND = 0.5748213286851482


@pytest.fixture(autouse=True)
def _no_publish_no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a paper-index test reached a publish path or the network")

    monkeypatch.setattr("cfb_data_build.publish.publish_dataset", refuse)
    monkeypatch.setattr(sb, "publish_dataset", refuse)
    monkeypatch.setattr("cfb_data_build.publish._gh_runner", refuse)
    monkeypatch.setattr(socket, "socket", refuse)


def _pbp(season: int | None = None) -> pl.DataFrame:
    pbp = pl.read_parquet(FIX / "play_by_play_slice.parquet")
    return pbp if season is None else pbp.filter(pl.col("season") == season)


def _schedule() -> pl.DataFrame:
    return pl.read_parquet(FIX / "cfb_schedules_2024_slice.parquet")


@pytest.fixture(scope="module")
def games() -> pl.DataFrame:
    return paper_index_games_table(_pbp())


def _teams(*ids: int) -> pl.DataFrame:
    """A ``team_summaries``-shaped frame: text ``team_id``, one real-looking metric."""
    n = len(ids)
    return pl.DataFrame(
        {
            "team_id": [str(i) for i in ids],
            "season": [2024] * n,
            "EPAplay_off": [0.1 * (i + 1) for i in range(n)],
            "EPAplay_off_rank": [float(n - i) for i in range(n)],
        }
    )


def _row(df: pl.DataFrame, team: int) -> dict:
    rows = df.filter(pl.col("team_id") == str(team)).to_dicts()
    assert len(rows) == 1, rows
    return rows[0]


# --- the per-game table -------------------------------------------------------


def test_rows_are_sdv_pys_unchanged_plus_the_span(games):
    assert games.schema == pl.Schema(OUTPUT_SCHEMA)
    assert list(OUTPUT_SCHEMA) == [
        "game_id",
        "team_id",
        *t1.GAMES_SCHEMA,
        "paper_index_span",
    ]
    direct = t1.paper_index_games(_pbp().select(t1.PBP_COLUMNS), "cfb")
    assert_frame_equal(games.drop("paper_index_span"), direct)
    assert games.schema["game_id"] == games.schema["team_id"] == pl.Int64
    assert games.select("game_id", "team_id").is_duplicated().sum() == 0
    spans = dict(games.select("season", "paper_index_span").unique().iter_rows())
    assert spans == {2021: "train", 2024: "holdout"}


def test_each_games_shares_sum_to_one_and_deserved_wins_sum_to_games(games):
    per_game = games.group_by("game_id").agg(
        sides=pl.len(), share=pl.col("paper_share").sum(), winners=pl.col("won").sum()
    )
    assert per_game["sides"].to_list() == [2] * 4
    assert per_game["winners"].to_list() == [1] * 4
    assert per_game["share"].to_list() == pytest.approx([1.0] * 4, abs=1e-12)
    rollup = t1.deserved_wins(games)
    assert rollup["deserved_wins"].sum() == pytest.approx(4.0, abs=1e-12)
    assert rollup["deserved_wins"].sum() == pytest.approx(games["game_id"].n_unique())


def test_the_reused_oracle_game_matches_game_on_paper(games):
    asu = games.filter((pl.col("game_id") == WK10) & (pl.col("team_id") == ASU))
    assert asu["paper_share"].item() == pytest.approx(
        1 - GOP_WK10_HOME_SHARE, abs=1e-12
    )
    assert asu["paper_share"].item() == pytest.approx(P_WK10, abs=1e-12)
    assert asu["won"].item() is True


def test_a_game_under_the_snap_floor_is_in_neither_table(games, monkeypatch):
    pbp = _pbp()
    is_snap = pl.col("scrimmage_play") == True  # noqa: E712
    snaps = dict(
        pbp.filter((pl.col("game_id") == UNDER_FLOOR) & is_snap)
        .group_by("pos_team_id")
        .len()
        .iter_rows()
    )
    assert snaps == {TOLEDO: 11, NORFOLK: 18}
    assert max(snaps.values()) < t1.MIN_PLAYS_PER_TEAM
    assert UNDER_FLOOR not in games["game_id"].to_list()
    assert NORFOLK not in games["team_id"].to_list()

    toledo = _row(
        attach_luck(_teams(TOLEDO), games.filter(pl.col("season") == 2021), 2021),
        TOLEDO,
    )
    # one game counted (the loss at Notre Dame); the 49-10 win is not a win here
    assert toledo["paper_index_games_n"] == 1
    assert toledo["deserved_wins"] == pytest.approx(P_AT_ND, abs=1e-12)
    assert toledo["luck_wins"] == pytest.approx(0 - P_AT_ND, abs=1e-12)

    # the floor is what drops it: lower the floor and the same rows are scored
    monkeypatch.setattr(t1, "MIN_PLAYS_PER_TEAM", 11)
    assert UNDER_FLOOR in paper_index_games_table(pbp)["game_id"].to_list()


# --- the season roll-up on team_summaries -------------------------------------


def test_luck_wins_is_wins_minus_deserved_on_a_three_game_team(games):
    # the record from the schedule, not from the code under test: W 42-21, W 35-31, L 31-39
    sched = _schedule()
    asu_pts = (
        pl.when(pl.col("home_id") == ASU)
        .then(pl.col("home_points"))
        .otherwise(pl.col("away_points"))
    )
    opp_pts = (
        pl.when(pl.col("home_id") == ASU)
        .then(pl.col("away_points"))
        .otherwise(pl.col("home_points"))
    )
    wins = sched.select((asu_pts > opp_pts).sum()).item()
    assert wins == 2 and sched.height == 3

    team = _teams(ASU, OKST, UCF, TEXAS, 9999)
    out = attach_luck(team, games.filter(pl.col("season") == 2024), 2024)
    assert out.columns == [*team.columns, *LUCK_COLUMNS]
    assert out.schema["team_id"] == pl.Utf8
    assert out.schema["paper_index_games_n"] == pl.Int64
    assert_frame_equal(out.select(team.columns), team)

    deserved = P_WK10 + P_WK11 + P_BOWL
    variance = sum(p * (1 - p) for p in (P_WK10, P_WK11, P_BOWL))
    asu = _row(out, ASU)
    assert asu["paper_index_games_n"] == 3
    assert asu["deserved_wins"] == pytest.approx(deserved, abs=1e-12)
    assert asu["deserved_wins"] == pytest.approx(0.945429743779788, abs=1e-12)
    assert asu["luck_wins"] == pytest.approx(wins - deserved, abs=1e-12)
    assert asu["luck_z"] == pytest.approx(
        (wins - deserved) / math.sqrt(variance), abs=1e-12
    )
    assert asu["paper_index_span"] == "holdout"

    # ranks: 1 = the luckiest, over the frame's teams that have a value
    by_team = {int(r["team_id"]): r for r in out.to_dicts()}
    assert [by_team[t]["luck_wins_rank"] for t in (ASU, TEXAS, OKST, UCF)] == [
        1,
        2,
        3,
        4,
    ]
    assert [by_team[t]["luck_z_rank"] for t in (ASU, TEXAS, OKST, UCF)] == [1, 2, 3, 4]

    # a team with no scored game: a zero count, null luck, no rank, the same label
    none = by_team[9999]
    assert none["paper_index_games_n"] == 0
    assert [none[c] for c in LUCK_COLUMNS[:5]] == [None] * 5
    assert none["paper_index_span"] == "holdout"


def test_another_seasons_games_are_refused(games):
    with pytest.raises(ValueError, match="other seasons"):
        attach_luck(_teams(ASU), games, 2024)


def test_span_comes_from_the_sdv_py_fit_constants(monkeypatch):
    assert {s: paper_index_span(s) for s in (2015, 2020, 2024, 2026)} == {
        2015: "out_of_span",
        2020: "train",
        2024: "holdout",
        2026: "out_of_span",
    }
    assert [paper_index_span(s) for s in (2016, 2023, 2025)] == [
        "train",
        "train",
        "holdout",
    ]
    # imported, not copied: move the sdv-py train span and the label moves with it
    monkeypatch.setitem(t1.TRAIN_SEASONS, "cfb", (2016, 2024))
    assert paper_index_span(2024) == "train"


def test_ids_stay_int64_and_a_float_or_text_id_raises(games):
    pbp = _pbp(2024)
    for col in ("game_id", "pos_team_id", "homeTeamId", "awayTeamId"):
        for bad in (pl.Float64, pl.Utf8):
            with pytest.raises(TypeError, match=col):
                paper_index_games_table(pbp.with_columns(pl.col(col).cast(bad)))
    g24 = games.filter(pl.col("season") == 2024)
    with pytest.raises(TypeError, match="never a float"):
        attach_luck(
            _teams(ASU).with_columns(pl.col("team_id").cast(pl.Float64)), g24, 2024
        )
    with pytest.raises(TypeError, match="never a float"):
        attach_luck(
            _teams(ASU), g24.with_columns(pl.col("team_id").cast(pl.Float64)), 2024
        )
    # an integer team table joins too, with no cast through text
    as_int = attach_luck(
        _teams(ASU).with_columns(pl.col("team_id").cast(pl.Int64)), g24, 2024
    )
    assert as_int["paper_index_games_n"].item() == 3


# --- the weekly snapshot contract ---------------------------------------------


def _summaries(monkeypatch, tmp_path, through_week: int | None) -> pl.DataFrame:
    """``team_summaries`` as ``build_summaries_season`` writes it for one snapshot.

    The play-side builder is faked (it needs ~60 pbp columns and a full slate);
    the per-game shares, the snapshot filter and the luck roll-up are the real ones.
    """
    seen: dict = {}

    def fake_build(plays, _season, *, through_week=None, rosters=None):
        seen["play_games"] = set(plays["game_id"].to_list())
        return {k: None for k in SUMMARIES_REGISTRY} | {
            "team_summaries": _teams(ASU, OKST, UCF, TEXAS)
        }

    def spy_attach(team, shares, season):
        seen["share_games"] = set(shares["game_id"].cast(pl.Utf8).to_list())
        return attach_luck(team, shares, season)

    monkeypatch.setattr(
        sb,
        "prepare_plays_input",
        lambda p, _s, _season: p.select(pl.col("game_id").cast(pl.Utf8)),
    )
    monkeypatch.setattr(sb, "build_team_summaries", fake_build)
    monkeypatch.setattr(sb, "attach_luck", spy_attach)
    monkeypatch.setattr(
        sb, "write_dataset", lambda df, dataset, *a, **k: seen.__setitem__(dataset, df)
    )
    sb.build_summaries_season(
        2024,
        through_week=through_week,
        base=str(tmp_path),
        pbp=_pbp(2024),
        schedule=_schedule(),
        rosters=pl.DataFrame(),
    )
    # the plays and the shares were cut by the same games
    assert seen["share_games"] == seen["play_games"]
    return seen["team_summaries"]


def test_weekly_snapshot_is_inclusive_and_never_holds_the_bowl(monkeypatch, tmp_path):
    def snap(week, team=ASU):
        return _row(_summaries(monkeypatch, tmp_path, week), team)

    # through_week = W holds the week-W game: INCLUSIVE, unlike cfb_ratings_weekly
    w11 = snap(11)
    assert w11["paper_index_games_n"] == 2
    assert w11["deserved_wins"] == pytest.approx(P_WK10 + P_WK11, abs=1e-12)
    assert w11["luck_wins"] == pytest.approx(2 - (P_WK10 + P_WK11), abs=1e-12)
    # ...and W - 1 does not
    w10 = snap(10)
    assert w10["paper_index_games_n"] == 1
    assert w10["deserved_wins"] == pytest.approx(P_WK10, abs=1e-12)
    assert snap(11, UCF)["paper_index_games_n"] == 1
    assert snap(10, UCF)["paper_index_games_n"] == 0
    w9 = snap(9)
    assert w9["paper_index_games_n"] == 0 and w9["deserved_wins"] is None

    # the Peach Bowl is postseason week 1: in the season table, in no snapshot
    season = snap(None)
    assert season["paper_index_games_n"] == 3
    assert season["deserved_wins"] == pytest.approx(P_WK10 + P_WK11 + P_BOWL, abs=1e-12)
    assert snap(None, TEXAS)["paper_index_games_n"] == 1
    for week in range(1, 17):
        assert snap(week)["paper_index_games_n"] == (week >= 10) + (week >= 11), week
        assert snap(week, TEXAS)["paper_index_games_n"] == 0, week


# --- leakage: no pattern-built list may take the outcome columns ---------------


def _weekly_like(games: pl.DataFrame) -> pl.DataFrame:
    """A ``team_summaries_weekly``-shaped frame carrying the real luck columns."""
    team = _teams(ASU, OKST, UCF, TEXAS)
    out = attach_luck(team, games.filter(pl.col("season") == 2024), 2024)
    # the producer's list is everything attach_luck adds: a new column must join it
    assert [c for c in out.columns if c not in team.columns] == list(LUCK_COLUMNS)
    return out.with_columns(through_week=pl.lit(11, dtype=pl.Int32))


@pytest.mark.parametrize("keep_ranks", [False, True])
def test_no_luck_column_can_be_a_model_feature(games, monkeypatch, keep_ranks):
    weekly = _weekly_like(games)
    feats = hm_data.feature_columns(weekly, keep_ranks=keep_ranks)
    assert "EPAplay_off" in feats  # not vacuous: a real feature still passes
    assert ("EPAplay_off_rank" in feats) is keep_ranks
    assert not set(feats) & set(LUCK_COLUMNS)

    # the named exclusion is what keeps them out: without it they walk in
    monkeypatch.setattr(hm_data, "_OUTCOME_DERIVED", frozenset())
    leaked = set(hm_data.feature_columns(weekly, keep_ranks=keep_ranks)) & set(
        LUCK_COLUMNS
    )
    expected = {"deserved_wins", "luck_wins", "luck_z", "paper_index_games_n"}
    if keep_ranks:
        expected |= {"luck_wins_rank", "luck_z_rank"}
    assert leaked == expected


def test_luck_columns_are_never_in_the_lean_model_set():
    diffs = [f"{c}_diff" for c in LUCK_COLUMNS]
    assert {family_of(d) for d in diffs} == {"outcome"}
    assert "outcome" not in LEAN_FAMILIES
    assert lean_features(diffs) == []


def test_league_averages_never_average_the_luck_columns(games):
    # league_averages reads the season table, which has no through_week
    weekly = _weekly_like(games).drop("through_week")
    metrics = league_averages.metric_columns(weekly)
    assert "EPAplay_off" in metrics
    assert not set(metrics) & set(LUCK_COLUMNS)
    out = league_averages.summarize(
        weekly, season=2024, level="all", entity="team", category="team_summaries"
    )
    assert out["metric"].to_list() == ["EPAplay_off"]
    # the count sits with the counts, by name and not only by its _n suffix
    assert "paper_index_games_n" in league_averages._NOT_METRICS


# --- stage 67 wiring ----------------------------------------------------------


def test_stage_67_builds_from_the_tree_and_is_registered(games, tmp_path):
    spec = SPECS["paper_index_games"]
    assert (spec.dataset, spec.stem, spec.tag) == (
        "paper_index_games",
        "cfb_paper_index_games",
        "cfb_paper_index_games",
    )
    assert "paper_index_games" in DERIVED and "paper_index_games" in BUILDERS
    assert spec.tag in PKG_FUNCTION
    # the M-LUCK honesty rule is in the release note, with the sdv-py years
    note = RELEASE_NOTES[spec.tag]
    assert "`train` for 2016-2023" in note and "in-sample" in note
    assert "`holdout` for 2024-2025" in note

    pbp_dir = tmp_path / "pbp" / "parquet"
    pbp_dir.mkdir(parents=True)
    _pbp(2024).write_parquet(pbp_dir / "play_by_play_2024.parquet")
    built = build_paper_index_games(2024, base=str(tmp_path))
    assert_frame_equal(built, games.filter(pl.col("season") == 2024))

    missing = build_paper_index_games(2023, base=str(tmp_path))
    assert missing.height == 0 and missing.schema == pl.Schema(OUTPUT_SCHEMA)

    # the stage driver writes it under the dataset's own directory, and does not publish
    assert build_derived("paper_index_games", 2024, 2024, base=str(tmp_path)) == []
    written = pl.read_parquet(
        tmp_path
        / "paper_index_games"
        / "parquet"
        / "cfb_paper_index_games_2024.parquet"
    )
    assert_frame_equal(written, built)
