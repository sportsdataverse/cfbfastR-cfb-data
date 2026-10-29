"""G-CFB-1 (``cfb_data_build.gate``): a clean season reads 0 on every check, and each
planted defect is counted -- a gate never seen to fail is decoration (see test_checks.py).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from cfb_data_build.gate import _adv_team_checks, _checks, _finals_with_plays

#: Two games; a kickoff touchback, a pre-2014 XP row and two carries (a gain and a loss).
CLEAN = {
    "game_id": [1, 1, 1, 2, 2],
    "firstHalfKickoffTeamId": ["10", "10", "10", "20", "20"],
    "period": [1, 2, 3, 1, 1],
    "start.pos_team_receives_2H_kickoff": [True, False, False, False, False],
    "kickoff_play": [True, False, False, False, False],
    "start.yardsToEndzone.touchback": [80, None, None, None, None],
    "kickoff_tb": [True, False, False, False, False],
    "yds_kickoff_return": [None, None, None, None, None],
    "type.text": ["Kickoff", "Extra Point Good", "Rush", "Rush", "Pass"],
    "text": ["kickoff", "PAT good", "12 yd run", "2 yd loss", "pass"],
    "xp_attempt": [False, True, False, False, False],
    "rush": [False, False, True, True, False],
    "yds_rushed": [None, None, 12.0, -2.0, None],
    "line_yards": [None, None, 7.0, -2.4, None],
    "second_level_yards": [None, None, 3.0, 0.0, None],
    "open_field_yards": [None, None, 2.0, 0.0, None],
}


def _pbp(**overrides: tuple[int, object]) -> pl.DataFrame:
    data = {k: list(v) for k, v in CLEAN.items()}
    for col, (row, value) in overrides.items():
        data[col.replace("__", ".")][row] = value
    return pl.DataFrame(data, strict=False)


def test_clean_season_passes():
    assert set(_checks(_pbp(), 2011).values()) == {0}


@pytest.mark.parametrize(
    ("check", "season", "override"),
    [
        ("first_half_kickoff_team_varies", 2011, {"firstHalfKickoffTeamId": (2, "20")}),
        (
            "receives_2h_kickoff_after_half",
            2011,
            {"start__pos_team_receives_2H_kickoff": (2, True)},
        ),
        ("kickoff_touchback_spot", 2011, {"start__yardsToEndzone__touchback": (0, 75)}),
        ("kickoff_touchback_spot", 2012, {}),  # 80 is the pre-2012 spot
        ("pre2014_xp_not_flagged", 2011, {"xp_attempt": (1, False)}),
        ("line_yards_parts", 2011, {"open_field_yards": (2, 1.0)}),
        (
            "line_yards_parts",
            2011,
            {"line_yards": (2, None)},
        ),  # a null part is a defect
    ],
)
def test_each_defect_is_counted(check, season, override):
    assert _checks(_pbp(**override), season)[check] == 1


def test_pre2014_failed_two_point_try_is_not_an_xp():
    # the pre-2014 feed types a failed 2-pt try "Extra Point Missed"; it is not an xp_attempt
    failed_2pt = _pbp(
        type__text=(1, "Extra Point Missed"),
        text=(1, "Aaron Murray pass attempt failed ."),
        xp_attempt=(1, False),
    )
    assert _checks(failed_2pt, 2011)["pre2014_xp_not_flagged"] == 0
    missed_kick = _pbp(type__text=(1, "Extra Point Missed"), xp_attempt=(1, False))
    assert _checks(missed_kick, 2011)["pre2014_xp_not_flagged"] == 1


def test_2012_touchback_returns_to_the_25():
    pbp = _pbp(start__yardsToEndzone__touchback=(0, 75))
    assert _checks(pbp, 2012)["touchback_return_not_25"] == 1
    assert _checks(
        _pbp(start__yardsToEndzone__touchback=(0, 75), yds_kickoff_return=(0, 25)), 2012
    ) == {k: 0 for k in _checks(pbp, 2012)}


def test_adv_team_yards_per_dropback():
    assert _adv_team_checks(None) == {"adv_team_missing": 1}
    assert _adv_team_checks(pl.DataFrame({"team_id": ["1"]})) == {
        "yards_per_dropback_missing": 1
    }
    frame = pl.DataFrame({"yards_per_dropback": [6.1, None]})
    assert _adv_team_checks(frame) == {"yards_per_dropback_missing": 1}
    # zero dropbacks has no rate: exempt; a null passes_rate still counts
    no_dropbacks = pl.DataFrame({"yards_per_dropback": [6.1, None], "passes_rate": [0.4, 0.0]})
    assert _adv_team_checks(no_dropbacks) == {"yards_per_dropback_missing": 0}
    unknown = pl.DataFrame({"yards_per_dropback": [None, None], "passes_rate": [0.0, None]})
    assert _adv_team_checks(unknown) == {"yards_per_dropback_missing": 1}
    empty = pl.DataFrame({"yards_per_dropback": []}, schema={"yards_per_dropback": pl.Float64})
    assert _adv_team_checks(empty) == {"adv_team_missing": 1}


def test_finals_count(tmp_path):
    finals = tmp_path / "cfb" / "json" / "final"
    finals.mkdir(parents=True)
    assert _finals_with_plays(finals, 2025) is None  # no master: the caller fails the season
    pl.DataFrame(
        {"season": [2025] * 4, "id": [1, 2, 3, 4], "status_type_completed": [True, True, True, False]}
    ).write_parquet(tmp_path / "cfb" / "cfb_schedule_master.parquet")
    (finals / "1.json").write_text(json.dumps({"plays": [{"id": 1}]}))
    (finals / "2.json").write_text(json.dumps({"plays": []}))  # current but empty
    # game 3 completed with no final; game 4 not played yet
    assert _finals_with_plays(finals, 2025) == (1, 1)
