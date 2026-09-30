"""Team-factor extensions on ``team_summaries``: starting EP, havoc EPA, expected turnovers.

Chosen from the 2026-09-29 exploration (ClaudeCowork
notes/2026-09-29-gop-factors-epa-exploration.md): the EPA versions of Connelly's five
factors that earned a column. Synthetic frames; every expected value is worked by hand.
"""

from __future__ import annotations

import polars as pl
import pytest
from cfb_data_build.team_summaries import (
    _MARGIN_BASES,
    _add_turnover_luck,
    _clean_rank_columns,
    _field_position_ep,
    _havoc_and_expected_turnovers,
    _mutate_summary_margins,
    _summarize_drives,
)


def _ep(own_yardline: int) -> float:
    return (
        _field_position_ep().filter(pl.col("yardline_own") == own_yardline)["ep"].item()
    )


def test_field_position_ep_is_the_yardline_only_table():
    t = _field_position_ep()
    assert t.height == 99 and t["yardline_own"].to_list() == list(range(1, 100))
    # points rise toward the opponent goal; nothing else (score, clock) is an input
    assert t["ep"].is_sorted()
    assert _ep(1) < 1.0 < _ep(50) < 3.0 < _ep(99)


# run/pass snaps: (offense, defense, drive, drive.result, yards_to_goal); the drive's
# first snap is its start. Team 1's first drive has THREE snaps and its second ONE, so a
# per-play mean would weight the 25-yard-line start three times.
_DRIVE_SNAPS = [
    ("1", "2", "d1", "PUNT", 75),
    ("1", "2", "d1", "PUNT", 60),
    ("1", "2", "d1", "PUNT", 45),
    ("1", "2", "d2", "FG", 40),
    ("2", "1", "d3", "PUNT", 80),
]


def _drive_plays() -> pl.DataFrame:
    return pl.DataFrame(
        _DRIVE_SNAPS,
        schema=[
            "pos_team_id",
            "def_pos_team_id",
            "drive_id",
            "drive.result",
            "yards_to_goal",
        ],
        orient="row",
    ).with_columns(
        game_id=pl.lit("g1"),
        drive_start_yards_to_goal=pl.col("yards_to_goal")
        .first()
        .over("drive_id")
        .cast(pl.Float64),
        drive_yards=pl.lit(10),
    )


def test_field_position_averages_drives_not_plays():
    # the build relocates ranks (drive_start_ep_rank_off -> drive_start_ep_off_rank) right after
    out = _clean_rank_columns(_summarize_drives(_drive_plays())).sort("pos_team_id")
    one, two = out.row(0, named=True), out.row(1, named=True)
    per_drive = (
        _ep(25) + _ep(60)
    ) / 2  # starts at own 25 (75 to go) and own 60 (40 to go)
    per_play = (3 * _ep(25) + _ep(60)) / 4
    assert one["drive_start_ep_off"] == pytest.approx(per_drive)
    assert one["drive_start_ep_off"] != pytest.approx(per_play)
    # a defense's start EP is its opponents' drive starts
    assert one["drive_start_ep_def"] == pytest.approx(_ep(20))
    assert two["drive_start_ep_off"] == pytest.approx(_ep(20))
    assert two["drive_start_ep_def"] == pytest.approx(per_drive)
    assert one["drive_start_ep_margin"] == pytest.approx(per_drive - _ep(20))
    # the better start ranks first on offense, the worse start allowed first on defense
    assert (one["drive_start_ep_off_rank"], two["drive_start_ep_off_rank"]) == (1.0, 2.0)
    assert (one["drive_start_ep_def_rank"], two["drive_start_ep_def_rank"]) == (1.0, 2.0)
    assert one["drive_start_ep_margin_rank"] == 1.0
    # the same drives in yards: (75 + 40) / 2, not the per-play (3 * 75 + 40) / 4
    assert one["start_position_off"] == pytest.approx(57.5)
    assert one["start_position_off"] != pytest.approx(66.25)
    assert one["start_position_off_n"] == 2 and two["start_position_off_n"] == 1
    assert one["start_position_def"] == pytest.approx(80.0)
    assert one["start_position_margin"] == pytest.approx(80.0 - 57.5)
    # fewer yards to go ranks first on offense; more allowed ranks first on defense
    assert (one["start_position_off_rank"], two["start_position_off_rank"]) == (1.0, 2.0)
    assert (one["start_position_def_rank"], two["start_position_def_rank"]) == (1.0, 2.0)
    assert one["start_position_margin_rank"] == 1.0


# scrimmage snaps: (game, offense, defense, EPA, havoc, int, pass, pass breakup, fumble)
_SNAPS = [
    ("g1", "A", "B", -3.0, True, 1.0, 1.0, None, 0.0),  # A intercepted
    ("g1", "A", "B", -0.5, True, 0.0, 1.0, "B DB", 0.0),  # A's pass broken up
    ("g1", "A", "B", -2.0, True, 0.0, 0.0, None, 1.0),  # A fumbles a run
    ("g2", "A", "B", 0.5, False, 0.0, 0.0, None, 0.0),
    ("g1", "B", "A", 1.0, False, 0.0, 1.0, None, 0.0),
    ("g2", "B", "A", -0.4, True, 0.0, 1.0, "A LB", 0.0),  # B's pass broken up
    ("g2", "B", "A", -1.5, True, 0.0, 0.0, None, 1.0),  # B fumbles a run
]


def _team_off() -> pl.DataFrame:
    return pl.DataFrame(
        _SNAPS,
        schema=[
            "game_id",
            "pos_team_id",
            "def_pos_team_id",
            "EPA",
            "havoc",
            "int",
            "pass",
            "pass_breakup_player_name",
            "fumble_vec",
        ],
        orient="row",
    )


def test_havoc_epa_per_game_and_expected_turnover_margin():
    out = _havoc_and_expected_turnovers(_team_off()).sort("pos_team_id")
    a, b = out.row(0, named=True), out.row(1, named=True)
    # A's own havoc snaps cost (-3.0 - 0.5 - 2.0) over 2 games; B's cost (-0.4 - 1.5) / 2
    assert a["havoc_EPAgame_off"] == pytest.approx(-2.75)
    assert b["havoc_EPAgame_off"] == pytest.approx(-0.95)
    assert a["havoc_EPAgame_def"] == pytest.approx(-0.95)
    assert b["havoc_EPAgame_def"] == pytest.approx(-2.75)
    assert a["havoc_EPAgame_margin"] == pytest.approx(-1.8)
    assert b["havoc_EPAgame_margin"] == pytest.approx(1.8)
    # INT share = 1 INT / 3 passes defensed. Fumbles cancel (one each); A's passes were
    # defended twice in 2 games, B's once: A = 1/3 * (1/2 - 2/2) = -1/6. The sign must
    # survive: in the exploration, subtracting raw UInt32 counts wrapped it to ~4.3e9.
    assert a["expected_turnover_margin"] == pytest.approx(-1 / 6)
    assert b["expected_turnover_margin"] == pytest.approx(1 / 6)
    # the two sides: A expects 0.5 * 1/2 fumbles + 1/3 * 2/2 defended passes = 7/12
    # giveaways a game, and 0.5 * 1/2 + 1/3 * 1/2 = 5/12 takeaways; margin 5/12 - 7/12
    assert a["expected_turnovers_off"] == pytest.approx(7 / 12)
    assert a["expected_turnovers_def"] == pytest.approx(5 / 12)
    assert b["expected_turnovers_off"] == pytest.approx(5 / 12)
    assert (b["expected_turnovers_off_rank"], a["expected_turnovers_off_rank"]) == (1.0, 2.0)
    assert (b["expected_turnovers_def_rank"], a["expected_turnovers_def_rank"]) == (1.0, 2.0)
    # less EPA lost to havoc ranks first on offense; more inflicted ranks first on defense
    assert (b["havoc_EPAgame_off_rank"], a["havoc_EPAgame_off_rank"]) == (1.0, 2.0)
    assert (b["havoc_EPAgame_def_rank"], a["havoc_EPAgame_def_rank"]) == (1.0, 2.0)
    assert (b["expected_turnover_margin_rank"], a["expected_turnover_margin_rank"]) == (
        1.0,
        2.0,
    )


def test_havoc_margin_is_created_minus_allowed_whole_team_only():
    """def - off: havoc is bad for an offense. Whole-team table only, like explosive_margin."""
    base = {f"{b}_{side}": [0.5, 0.5] for b, _ in _MARGIN_BASES for side in ("off", "def")}
    whole = pl.DataFrame(
        base
        | {"pos_team_id": ["1", "2"], "explosive_off": [0.1, 0.1], "explosive_def": [0.1, 0.1],
           "turnovers_off": [1.0, 1.0], "turnovers_def": [1.0, 1.0],
           "havoc_off": [0.10, 0.18], "havoc_def": [0.16, 0.12]}
    )
    out = _mutate_summary_margins(whole).sort("pos_team_id")
    assert out["havoc_margin"].to_list() == pytest.approx([0.06, -0.06])
    assert out["havoc_margin_rank"].to_list() == [1.0, 2.0]
    # a pass/rush split carries no turnovers, so it gets no whole-team margins
    split = whole.drop("turnovers_off", "turnovers_def")
    assert "havoc_margin" not in _mutate_summary_margins(split).columns


def test_turnover_luck_per_side_sums_to_the_margin():
    """Positive = lucky on both sides; off + def = 5 x (turnover_margin - expected margin)."""
    t = pl.DataFrame(
        {
            "pos_team_id": ["X", "Y"],
            "turnovers_off": [0.5, 1.5],
            "turnovers_def": [2.0, 0.5],
            "expected_turnovers_off": [1.0, 1.0],
            "expected_turnovers_def": [1.2, 1.0],
        }
    ).with_columns(
        turnover_margin=pl.col("turnovers_def") - pl.col("turnovers_off"),
        expected_turnover_margin=pl.col("expected_turnovers_def")
        - pl.col("expected_turnovers_off"),
    )
    out = _add_turnover_luck(t).sort("pos_team_id")
    x, y = out.row(0, named=True), out.row(1, named=True)
    # X: 0.5 fewer giveaways and 0.8 more takeaways than expected
    assert x["turnover_luck_off"] == pytest.approx(2.5)
    assert x["turnover_luck_def"] == pytest.approx(4.0)
    assert x["turnover_luck"] == pytest.approx(6.5)
    assert y["turnover_luck"] == pytest.approx(y["turnover_luck_off"] + y["turnover_luck_def"])
    assert (x["turnover_luck_rank"], y["turnover_luck_rank"]) == (1.0, 2.0)
    assert (x["turnover_luck_off_rank"], x["turnover_luck_def_rank"]) == (1.0, 1.0)
