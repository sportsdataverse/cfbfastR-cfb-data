"""Five Factors columns on ``team_summaries`` (IF-2): finishing, turnovers, explosive margin.

Synthetic two-team, two-game frame. Team "1" (A) and team "2" (B) meet twice:

* A has 3 drives: a TD after a snap at the 35, a FG whose only snap inside
  the opponent 40 is exactly at the 40, and an interception that never gets
  inside it. Two scoring opportunities worth 10 points.
* A's second giveaway is on special teams: it muffs a B punt in game 2. The
  builder never sees that play (it keeps runs and passes only), but it is a
  turnover, so it counts.
* B has 2 drives: a TD after a snap at the 25, and a pick-six thrown from the
  38 -- an opportunity worth 0 points to B, and A's one takeaway.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_data_build.summaries_input import game_giveaways, prepare_plays_input
from cfb_data_build.team_summaries import (
    _clean_rank_columns,
    _drive_owners,
    _mutate_summary_margins,
    _suffix_nonkey,
    _summarize_drives,
    _summarize_team,
)

# run/pass snaps: (game, offense, defense, drive, drive.result, yards_to_goal,
# offense lost it, explosive)
_PLAYS = [
    ("g1", "1", "2", "dA1", "TD", 75, False, False),
    ("g1", "1", "2", "dA1", "TD", 50, False, True),
    ("g1", "1", "2", "dA1", "TD", 35, False, False),
    ("g1", "1", "2", "dA2", "FG", 60, False, False),
    ("g1", "1", "2", "dA2", "FG", 40, False, False),  # the 40 itself counts
    ("g1", "2", "1", "dB1", "TD", 70, False, True),
    ("g1", "2", "1", "dB1", "TD", 25, False, False),
    ("g2", "1", "2", "dA3", "INT", 80, False, False),
    ("g2", "1", "2", "dA3", "INT", 45, True, False),
    ("g2", "2", "1", "dB2", "INT TD", 38, True, False),
]
# special-teams plays: (game, offense, defense, offense lost it, defense lost it)
_SPECIAL_TEAMS = [
    ("g2", "2", "1", False, True),  # B punts, A's returner muffs, B recovers
]
_FLAGS = ["game_id", "pos_team_id", "def_pos_team_id"]
_LOST = ["is_pos_team_turnover", "is_def_pos_team_turnover"]


def _plays(rows=_PLAYS, special=_SPECIAL_TEAMS) -> pl.DataFrame:
    """The builder's run/pass frame, carrying giveaways counted over every play."""
    snaps = pl.DataFrame(
        rows,
        schema=[
            *_FLAGS,
            "drive_id",
            "drive.result",
            "yards_to_goal",
            "is_pos_team_turnover",
            "explosive",
        ],
        orient="row",
    ).with_columns(is_def_pos_team_turnover=pl.lit(False))
    every_play = pl.concat(
        [
            snaps.select(*_FLAGS, *_LOST),
            pl.DataFrame(special, schema=[*_FLAGS, *_LOST], orient="row"),
        ]
    ).with_columns(home_id=pl.lit("1"), away_id=pl.lit("2"))
    n = snaps.height
    return snaps.join(
        game_giveaways(every_play), on=["game_id", "pos_team_id"], how="left"
    ).with_columns(
        drive_start_yards_to_goal=pl.col("yards_to_goal").first().over("drive_id"),
        drive_yards=pl.lit(10),
        **{
            c: pl.Series([v] * n)
            for c, v in {
                "pass": 1.0,
                "rush": 0.0,
                "havoc": False,
                "EPA": 0.1,
                "yards_gained": 5,
                "play_stuffed": False,
                "epa_success": 1.0,
                "red_zone_success": 1.0,
                "third_down_success": 1.0,
                "third_down_distance": 5.0,
                "late_down_success": 1.0,
                "early_down_EPA": 0.1,
                "nonExplosiveEpa": 0.1,
                "line_yards": 1.0,
                "opportunity_run": False,
            }.items()
        },
    )


def _overall(plays: pl.DataFrame) -> pl.DataFrame:
    """The whole-team off/def table as ``build_team_summaries`` assembles it."""
    off = _suffix_nonkey(
        _summarize_team(plays, "pos_team_id", ascending=False), "pos_team_id", "_off"
    )
    def_ = _suffix_nonkey(
        _summarize_team(plays, "def_pos_team_id", ascending=True),
        "def_pos_team_id",
        "_def",
    )
    joined = off.join(
        def_, left_on="pos_team_id", right_on="def_pos_team_id", how="left"
    )
    return _clean_rank_columns(_mutate_summary_margins(joined)).sort("pos_team_id")


def _drives(plays: pl.DataFrame) -> pl.DataFrame:
    return _clean_rank_columns(_summarize_drives(plays)).sort("pos_team_id")


def _team(df: pl.DataFrame, team: str) -> dict:
    return df.filter(pl.col("pos_team_id") == team).row(0, named=True)


def test_points_per_scoring_opportunity():
    d = _drives(_plays())
    a, b = _team(d, "1"), _team(d, "2")
    # A: (7 + 3) / 2 -- the INT drive never got inside the 40
    assert a["pts_per_opp_off"] == 5.0 and a["pts_per_opp_off_n"] == 2
    # B's defense faced exactly A's offense
    assert b["pts_per_opp_def"] == a["pts_per_opp_off"]
    assert b["pts_per_opp_def_n"] == 2
    # B: a TD and a pick-six -- the INT TD is A's points, not B's
    assert b["pts_per_opp_off"] == 3.5 and a["pts_per_opp_def"] == 3.5
    assert a["pts_per_opp_margin"] == 1.5 and b["pts_per_opp_margin"] == -1.5
    # more points per trip is better on offense, fewer allowed on defense
    assert (a["pts_per_opp_off_rank"], b["pts_per_opp_off_rank"]) == (1.0, 2.0)
    assert (a["pts_per_opp_def_rank"], b["pts_per_opp_def_rank"]) == (1.0, 2.0)
    assert (a["pts_per_opp_margin_rank"], b["pts_per_opp_margin_rank"]) == (1.0, 2.0)
    for c in ("pts_per_opp_off", "pts_per_opp_def", "pts_per_opp_margin"):
        assert d.schema[c] == pl.Float64 and d.schema[f"{c}_rank"] == pl.Float64
    assert d.schema["pts_per_opp_off_n"] == d.schema["pts_per_opp_def_n"] == pl.Int64


@pytest.mark.parametrize(
    ("td", "fg"),
    [
        ("PASSING TD", "FG GOOD"),
        ("RUSHING TD", "MADE FG"),
        ("PASSING TD TD", "FG GOOD"),
        ("RUSHING TD TD", "MADE FG"),
    ],
)
def test_pre_2014_drive_results_still_score(td: str, fg: str):
    # 2004-2013 releases spell the outcomes out; matching only "TD"/"FG"
    # scored every one of those seasons at zero points per opportunity
    old = [(*r[:4], {"TD": td, "FG": fg}.get(r[4], r[4]), *r[5:]) for r in _PLAYS]
    assert _team(_drives(_plays(old)), "1")["pts_per_opp_off"] == 5.0


def test_a_stray_snap_in_the_other_teams_drive_is_nobodys_opportunity():
    # ESPN drive 40175280229 (2025): MSU's drive ends in a pick-six that ESPN
    # labels "TD", and WMU's two-point try after it is filed under MSU's
    # drive. Grouped by (team, drive), the try at the 3 hands WMU a phantom
    # 7-point opportunity and charges it to MSU's defense.
    rows = [
        ("g1", "1", "2", "d1", "FG", 30, False, False),  # team 1: a real FG
        ("g1", "2", "1", "d2", "TD", 77, False, False),  # team 2 drives...
        ("g1", "2", "1", "d2", "TD", 71, True, False),  # ...and throws a pick-six
        ("g1", "1", "2", "d2", "TD", 3, False, False),  # team 1's try, filed in d2
    ]
    d = _drives(_plays(rows, special=[]))
    one, two = _team(d, "1"), _team(d, "2")
    assert (one["pts_per_opp_off"], one["pts_per_opp_off_n"]) == (3.0, 1)
    assert (two["pts_per_opp_def"], two["pts_per_opp_def_n"]) == (3.0, 1)
    assert two["pts_per_opp_off_n"] == 0 and one["pts_per_opp_def_n"] == 0
    # the yardage totals keep R's per-(team, drive) grouping: the stray
    # (team 1, d2) row still adds d2's start, the 77
    assert one["total_available_yards_off"] == 30 + 77


def test_drive_owner_is_espns_drive_team_when_it_has_a_snap():
    rows = [
        # x: ESPN names team 1, which ran 1 of the 3 snaps -- ESPN decides
        ("x", "1", "ONE"),
        ("x", "2", "ONE"),
        ("x", "2", "ONE"),
        # y: ESPN names team 1, which ran none of them -- the snaps decide
        ("y", "2", "ONE"),
        ("y", "2", "ONE"),
        # z: no ESPN label, one snap each -- the first snap decides
        ("z", "2", None),
        ("z", "1", None),
    ]
    plays = pl.DataFrame(
        rows,
        schema=["drive_id", "pos_team_id", "drive.team.abbreviation"],
        orient="row",
    ).with_columns(
        home_id=pl.lit("1"),
        away_id=pl.lit("2"),
        homeTeamAbbrev=pl.lit("ONE"),
        awayTeamAbbrev=pl.lit("TWO"),
    )
    got = dict(_drive_owners(plays).iter_rows())
    assert got == {"x": "1", "y": "2", "z": "2"}
    # without ESPN's drive team, the most snaps decide
    assert dict(_drive_owners(plays.drop("drive.team.abbreviation")).iter_rows()) == {
        "x": "2",
        "y": "2",
        "z": "2",
    }


def test_no_scoring_opportunity_is_null_and_unranked():
    # team 1 scores from the 30; team 2 never gets inside the 40
    rows = [
        ("g1", "1", "2", "d1", "TD", 30, False, False),
        ("g1", "2", "1", "d2", "PUNT", 70, False, False),
    ]
    d = _drives(_plays(rows, special=[]))
    one, two = _team(d, "1"), _team(d, "2")
    assert two["pts_per_opp_off_n"] == 0
    # null, not 0/0 = NaN, and unranked rather than ranked last
    assert two["pts_per_opp_off"] is None and two["pts_per_opp_off_rank"] is None
    assert one["pts_per_opp_off_rank"] == 1.0
    assert one["pts_per_opp_def"] is None and one["pts_per_opp_def_rank"] is None
    assert d["pts_per_opp_margin"].is_null().all()
    assert d["pts_per_opp_margin_rank"].is_null().all()


def test_turnovers_per_game_and_margin():
    o = _overall(_plays())
    a, b = _team(o, "1"), _team(o, "2")
    # A: 2 giveaways (one on special teams), 1 takeaway, 2 games
    assert a["turnovers_off"] == 1.0
    assert a["turnovers_def"] == 0.5
    assert a["turnover_margin"] == -0.5
    assert (b["turnovers_off"], b["turnovers_def"], b["turnover_margin"]) == (
        0.5,
        1.0,
        0.5,
    )
    for c in ("turnovers_off", "turnovers_def", "turnover_margin"):
        assert o.schema[c] == pl.Float64 and o.schema[f"{c}_rank"] == pl.Float64


def test_turnover_ranks_reward_fewer_giveaways_and_more_takeaways():
    o = _overall(_plays())
    a, b = _team(o, "1"), _team(o, "2")
    # B gave it away less, took it away more, and won the margin
    assert (b["turnovers_off_rank"], a["turnovers_off_rank"]) == (1.0, 2.0)
    assert (b["turnovers_def_rank"], a["turnovers_def_rank"]) == (1.0, 2.0)
    assert (b["turnover_margin_rank"], a["turnover_margin_rank"]) == (1.0, 2.0)


def test_explosive_margin_is_offense_minus_defense():
    o = _overall(_plays())
    diff = o["explosive_off"] - o["explosive_def"]
    assert (o["explosive_margin"] - diff).abs().max() < 1e-12
    a, b = _team(o, "1"), _team(o, "2")
    # A: 1 of 7 snaps explosive, allowed 1 of 3
    assert a["explosive_margin"] == pytest.approx(1 / 7 - 1 / 3)
    assert (b["explosive_margin_rank"], a["explosive_margin_rank"]) == (1.0, 2.0)
    assert o.schema["explosive_margin"] == pl.Float64


def test_game_giveaways_count_each_side_of_every_play():
    plays = pl.DataFrame(
        [
            ("g1", "1", True, False),  # 1 throws an interception
            ("g1", "2", False, True),  # 2 punts, 1's returner muffs it
            ("g1", "2", True, True),  # 2 is picked, 1 fumbles the return back
            ("g1", "1", None, None),  # unflagged: no turnover
        ],
        schema=["game_id", "pos_team_id", *_LOST],
        orient="row",
    ).with_columns(home_id=pl.lit("1"), away_id=pl.lit("2"))
    got = dict(
        game_giveaways(plays)
        .select("pos_team_id", "pos_team_game_giveaways")
        .iter_rows()
    )
    assert got == {"1": 3, "2": 1}


def test_special_teams_turnovers_survive_the_scrimmage_filter():
    # game 1: an interception by team 1, a punt team 1 muffs, a run by team 2;
    # game 2: a run by team 1, an interception by team 2
    base = {
        "home_id": "1",
        "away_id": "2",
        "home": "One",
        "away": "Two",
        "text": "play",
        "start.adj_TimeSecsRem": 900.0,
        "success": 0.0,
        "EPA": 0.0,
        "receiver_player_name": None,
        "receiver_player_id": None,
        "target": False,
        "completion": False,
        "passer_player_name": None,
        "passer_player_id": None,
        "pass_attempt": False,
        "int": False,
        "sack_vec": False,
        "drive_start_yards_to_goal": 75.0,
    }
    rows = [
        ("g1", 1, "1", True, False, True, False),
        ("g1", 2, "2", False, False, False, True),
        ("g1", 3, "2", False, True, False, False),
        ("g2", 1, "1", False, True, False, False),
        ("g2", 2, "2", True, False, True, False),
    ]
    keys = ["game_id", "game_play_number", "pos_team_id", "pass", "rush", *_LOST]
    pbp = pl.DataFrame(
        [{**base, **dict(zip(keys, r)), "int": r[5]} for r in rows],
        infer_schema_length=None,
    )
    sched = pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "home_division": ["fbs", "fbs"],
            "away_division": ["fbs", "fbs"],
        }
    )
    out = prepare_plays_input(pbp, sched, 2025)
    assert out.select("game_id", "game_play_number").rows() == [
        ("g1", 1),
        ("g1", 3),  # the punt is gone...
        ("g2", 1),
        ("g2", 2),
    ]
    # ...but its muff is still one of team 1's two game-1 giveaways, and each
    # team's count is per GAME: team 1 had none in game 2
    assert out["pos_team_game_giveaways"].to_list() == [2, 0, 0, 1]
