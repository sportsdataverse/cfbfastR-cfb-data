"""Summaries metric definitions that match what they are ranked against (percentile audit 2026-10-08).

One synthetic game between two FBS teams, run through the real
``build_team_summaries``. Team "1" has, in order:

* three runs: a "Rush" for -2 (a stuff), a "Rush" for 5 (an opportunity) and a
  "Fumble Recovery (Own)" run for -1 (sdv-py's ``stuffed_run`` is play type
  "Rush" only, so not a stuff);
* four dropbacks by QB "1Q": a successful completion for 8, an incompletion, a
  sack for -7, and an interception whose statYardage carries a 30-yard RETURN
  (so the rule-based ``success`` scores it a success).

Team "2" runs for 3 and for 0 (a stuff) and throws a completion and an
incompletion. No snap is inside the 20, so ``red_zone_success`` is all-null.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_data_build import team_summaries
from cfb_data_build.team_summaries import _rank, _summarize_drives, build_team_summaries

TEAMS = {"1": "Home U", "2": "Away St"}
_IDS = (
    "passer_player_name",
    "completion_player",
    "completion_player_id",
    "incompletion_player",
    "incompletion_player_id",
    "sack_taken_player_id",
    "interception_thrown_player_id",
    "rush_player_id",
    "rusher_player_name",
    "reception_player",
    "reception_player_id",
    "target_player",
    "target_player_id",
    "pass_breakup_player_name",
    "play_type",
)


def _snap(off: str, **kw) -> dict:
    row = {
        "game_id": "401",
        "season": 2025,
        "home": TEAMS["1"],
        "away": TEAMS["2"],
        "home_id": "1",
        "away_id": "2",
        "home_team_id": "1",
        "home_team_division": "fbs",
        "away_team_division": "fbs",
        "home_team_conference": "A",
        "away_team_conference": "B",
        "pos_team": TEAMS[off],
        "down": 1,
        "distance": 10,
        "yards_to_goal": 60,
        "EPA": 0.0,
        "epa_success": 0.0,
        "success": 0.0,
        "wpa": 0.01,
        "drive_id": f"d{off}",
        "drive.result": "PUNT",
        "drive_start_yards_to_goal": 75.0,
        "drive_yards": 10,
        "pos_team_game_giveaways": 0,
        "pos_score_diff_start": 0,
        "yards_gained": 0,
        "yds_rushed": None,
        "yds_receiving": None,
        "yds_sacked": None,
        **{c: 0.0 for c in ("pass", "rush", "pass_attempt", "completion", "int")},
        **{c: 0.0 for c in ("sack_vec", "fumble_vec", "pass_td", "rush_td")},
        **{c: None for c in _IDS},
    }
    return row | kw


def _run(off: str, yds: int, play_type: str = "Rush", **kw) -> dict:
    return _snap(
        off,
        rush=1.0,
        yds_rushed=yds,
        yards_gained=yds,
        play_type=play_type,
        rush_player_id=f"{off}R",
        rusher_player_name=f"RB {off}",
        **kw,
    )


def _dropback(off: str, **kw) -> dict:
    return _snap(off, **{"pass": 1.0, "passer_player_name": f"QB {off}", **kw})


def _complete(off: str, yds: int) -> dict:
    qb, wr = f"{off}Q", f"{off}W"
    return _dropback(
        off,
        pass_attempt=1.0,
        completion=1.0,
        yards_gained=yds,
        yds_receiving=yds,
        play_type="Pass Reception",
        EPA=0.5,
        epa_success=1.0,
        success=1.0,
        completion_player=f"QB {off}",
        completion_player_id=qb,
        reception_player=f"WR {off}",
        reception_player_id=wr,
        target_player=f"WR {off}",
        target_player_id=wr,
    )


def _incomplete(off: str) -> dict:
    return _dropback(
        off,
        pass_attempt=1.0,
        play_type="Pass Incompletion",
        EPA=-0.4,
        incompletion_player=f"QB {off}",
        incompletion_player_id=f"{off}Q",
    )


PLAYS = [
    _run("1", -2, EPA=-0.5),
    _run("1", 5, EPA=0.3, epa_success=1.0, success=1.0),
    _run("1", -1, play_type="Fumble Recovery (Own)", EPA=-0.6),
    _complete("1", 8),
    _incomplete("1"),
    _dropback(
        "1",
        sack_vec=1.0,
        yards_gained=-7,
        yds_sacked=-7,
        play_type="Sack",
        EPA=-1.5,
        sack_taken_player_id="1Q",
    ),
    _dropback(
        "1",
        pass_attempt=1.0,
        int=1.0,
        yards_gained=30,
        play_type="Interception Return",
        EPA=-3.0,
        success=1.0,
        interception_thrown_player_id="1Q",
    ),
    _run("2", 3, EPA=0.1),
    _run("2", 0, EPA=-0.4),
    _complete("2", 12),
    _incomplete("2"),
]


def _plays() -> pl.DataFrame:
    df = pl.DataFrame(
        PLAYS,
        schema_overrides={c: pl.Utf8 for c in _IDS},
        infer_schema_length=None,
    ).with_row_index("game_play_number", offset=1)
    home = pl.col("pos_team") == pl.col("home")
    return df.with_columns(
        **{
            f"{side}_EPA_{kind}": pl.when(is_side & (pl.col(kind) == 1)).then(
                pl.col("EPA")
            )
            for side, is_side in (("home", home), ("away", ~home))
            for kind in ("pass", "rush")
        }
    )


_GROUPS = pl.DataFrame(
    {
        "season": [2025, 2025],
        "team_id_source": ["espn", "espn"],
        "team_id": ["1", "2"],
        "subdivision_id": ["cfb:fbs", "cfb:fbs"],
        "conference_id": ["cfb:sec", "cfb:mac"],
    }
)


@pytest.fixture(scope="module")
def tables() -> dict[str, pl.DataFrame]:
    # the ridge needs a season; one game is not one, and it is not under test here
    adjusted = pl.DataFrame(
        {
            "team_id": ["1", "2"],
            "pos_team": [TEAMS["1"], TEAMS["2"]],
            **{
                c: [0.1, -0.1]
                for c in (
                    "adj_off_epa",
                    "adj_def_epa",
                    "net_adj_epa",
                    "off_strength_faced",
                    "def_strength_faced",
                )
            },
        }
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(team_summaries, "cfb_adjusted_epa", lambda _plays: adjusted)
        # through_week exempts the one-game "season" from the adjustment gate
        return build_team_summaries(_plays(), 2025, through_week=1, groups=_GROUPS)


def _team(tables: dict[str, pl.DataFrame], team: str) -> dict:
    return tables["team_summaries"].filter(pl.col("team_id") == team).row(0, named=True)


def _ladder_median(tables: dict[str, pl.DataFrame], col: str) -> float:
    p = tables["percentiles"]
    return p.filter((pl.col("pctile") - 0.5).abs() < 1e-9)[col].item()


def test_play_stuffed_is_a_run_stuff_rate_over_rushes(tables):
    """F1: incompletions and sacks are not stuffs; a stuff is a "Rush" for <= 0."""
    one, two = _team(tables, "1"), _team(tables, "2")
    assert one["play_stuffed_off"] == pytest.approx(1 / 3)  # was 4/7
    assert one["play_stuffed_off_n"] == 3  # the denominator is rushes
    assert two["play_stuffed_off"] == pytest.approx(1 / 2)
    assert one["play_stuffed_def"] == pytest.approx(1 / 2)
    # the pass split has no runs: no stuff rate, not 0.5
    assert one["play_stuffed_off_pass"] is None
    # the team-game ladder is cut over the same definition
    assert _ladder_median(tables, "play_stuffed") == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_opportunity_rate_is_over_rushes(tables):
    """C1: the share of CARRIES reaching 4 yards, not of all plays."""
    one = _team(tables, "1")
    assert one["opportunity_rate_off"] == pytest.approx(1 / 3)  # was 1/7
    assert one["opportunity_rate_off_n"] == 3
    assert _team(tables, "2")["opportunity_rate_off"] == 0.0
    assert _ladder_median(tables, "opportunity_run") == pytest.approx(1 / 6)


def test_an_interception_return_is_not_offensive_yardage(tables):
    """A3: the pick's 30-yard return is the defense's, so it counts as 0 yards."""
    one = _team(tables, "1")
    # -2 + 5 - 1 + 8 + 0 - 7 + 0 over 7 plays (was 33 yards with the return)
    assert one["yards_off"] == 3
    assert one["yardsplay_off"] == pytest.approx(3 / 7)
    per_game_2 = (12 + 3) / 4
    assert _ladder_median(tables, "yardsplay") == pytest.approx(
        (3 / 7 + per_game_2) / 2
    )


def test_passer_success_rate_is_over_dropbacks(tables):
    """D4: sacks and picks count against success the way they count in EPAplay."""
    qb = tables["passing"].filter(pl.col("player_id") == "1Q").row(0, named=True)
    assert (qb["att"], qb["sacked"], qb["pass_int"], qb["dropbacks"]) == (2, 1, 1, 4)
    # one successful completion over four dropbacks; the pick is not a success
    # even though its return makes the rule-based `success` say so (was 1/2)
    assert qb["success"] == pytest.approx(1 / 4)
    assert qb["success_n"] == 4
    assert "_sack_success" not in tables["passing"].columns


def test_ranks_are_null_where_they_say_nothing(tables):
    """C4: no rank for a null metric, an all-null column or a constant one."""
    ts = tables["team_summaries"]
    for col in (
        "play_stuffed_off_pass_rank",  # all null: no runs on pass plays
        "red_zone_success_off_rank",  # all null: no red-zone snaps
        "passrate_off_pass_rank",  # constant: every pass play is a pass
        "rushrate_off_rush_rank",
    ):
        assert ts[col].null_count() == ts.height, col
    # a single null among real values is unranked; the rest rank among themselves
    ranked = pl.DataFrame({"x": [3.0, None, 1.0, 2.0]}).select(
        r=_rank("x", descending=True)
    )
    assert ranked["r"].to_list() == [1.0, None, 3.0, 2.0]


def test_available_yards_share_counts_owned_drives_and_never_exceeds_one():
    """E4: a stray snap does not take the other team's drive yards; a 0 start is no share."""
    plays = pl.DataFrame(
        {
            "drive_id": ["dA", "dA", "dA", "dB", "dB", "dB"],
            "pos_team_id": ["1", "1", "2", "2", "2", "2"],
            "def_pos_team_id": ["2", "2", "1", "1", "1", "1"],
            "yards_to_goal": [75, 40, 25, 70, 65, 60],
            "drive.result": ["TD", "TD", "TD", "PUNT", "PUNT", "PUNT"],
            # team 2's lone snap in team 1's drive carries ITS start, 25; team 2's
            # own drive carries ESPN's 0 placeholder
            "drive_start_yards_to_goal": [75.0, 75.0, 25.0, 0.0, 0.0, 0.0],
            # ESPN's drive.yards over-runs the 75 available on dA
            "drive_yards": [80, 80, 80, 12, 12, 12],
        }
    )
    d = _summarize_drives(plays).sort("pos_team_id")
    one, two = d.row(0, named=True), d.row(1, named=True)
    # was 80 / 75: a drive gains at most what it had available
    assert one["available_yards_pct_off"] == 1.0
    # was (80 + 12) / 25 from the stray snap; with no available yards: null, not Infinity
    assert two["total_available_yards_off"] == 0
    assert two["available_yards_pct_off"] is None
    assert one["available_yards_pct_def"] is None


def test_drive_start_follows_the_first_snap_orientation_and_skips_placeholders():
    """E4: the header's orientation is the one the first snap agrees with; a 0 header falls back."""
    from cfb_data_build.summaries_input import drive_start_yards_to_goal

    df = pl.DataFrame(
        {
            "drive_id": ["a", "a", "b", "c", "d"],
            "pos_team_id": ["1", "1", "1", "2", "2"],
            "home_id": ["1"] * 5,
            "game_play_number": [2, 1, 3, 4, 5],
            "yards_to_goal": [70, 75, 75, 68, 70],
            # a: home team, measured from its own goal (25 -> 75 to go)
            # b: home team, header measured from the OTHER goal (75 -> 25 read naively)
            # c: ESPN's 0 placeholder; d: no header at all
            "drive.start.yardLine": [25, 25, 75, 0, None],
        }
    )
    out = df.select(s=drive_start_yards_to_goal())["s"].to_list()
    assert out == [75.0, 75.0, 75.0, 68.0, 70.0]  # was [75, 75, 25, 0, None]
