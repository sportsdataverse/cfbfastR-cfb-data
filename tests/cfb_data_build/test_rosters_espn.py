"""Unit tests for the ESPN-native ``cfb_rosters`` compiler.

The load-bearing logic is the collapse to one row per ``(season, team_id,
athlete_id)`` plus the position/division resolution, so that is what is asserted:
grain, "last appearance wins", href -> position resolution, id dtypes, and the
pinned schema. All hermetic -- no network, no raw checkout.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from cfb_data_build import rosters_espn as R

POSITIONS_DOC = {
    "captured_at": "2026-08-27T00:00:00+00:00",
    "count": 3,
    "positions": {
        "0": {
            "id": "0",
            "name": "Unknown",
            "displayName": "Unknown",
            "abbreviation": "-",
            "leaf": False,
        },
        "9": {
            "id": "9",
            "name": "Running Back",
            "displayName": "Running Back",
            "abbreviation": "RB",
            "leaf": True,
            "parent": {"$ref": "http://x/positions/70?lang=en"},
        },
        "70": {
            "id": "70",
            "name": "Offense",
            "displayName": "Offense",
            "abbreviation": "OFF",
            "leaf": False,
        },
    },
}

TEAMS_DOC = {"season": 2023, "divisions": {"80": ["333", "99"], "81": ["2000"]}}


def _pos_href(pid: int) -> str:
    return f"http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/positions/{pid}?lang=en&region=us"


def _game(game_id: int, week: int, rows: list[dict]) -> dict:
    return {"game_id": game_id, "season": 2023, "week": week, "data": rows}


@pytest.fixture()
def positions() -> pl.DataFrame:
    return R.load_positions("mem://", downloader=lambda _u: json.dumps(POSITIONS_DOC))


@pytest.fixture()
def divisions() -> pl.DataFrame:
    return R.load_divisions(2023, "mem://", downloader=lambda _u: json.dumps(TEAMS_DOC))


def test_position_id_parsed_from_href():
    assert R._position_id(_pos_href(9)) == 9
    assert R._position_id(None) is None
    assert R._position_id("http://x/positions/not-a-number?lang=en") is None


def test_positions_reference_dtypes_and_parent(positions):
    assert positions.height == 3
    assert positions.schema["position_id"] == pl.Int64
    assert positions.schema["position_parent_id"] == pl.Int64
    rb = positions.filter(pl.col("position_id") == 9).row(0, named=True)
    assert (rb["position"], rb["position_abbreviation"], rb["position_parent_id"]) == (
        "Running Back",
        "RB",
        70,
    )


def test_missing_position_reference_raises_rather_than_shipping_hrefs():
    with pytest.raises(RuntimeError, match="position reference unavailable"):
        R.load_positions("mem://", downloader=lambda _u: None)


def test_divisions_map(divisions):
    assert divisions.schema["team_id"] == pl.Int64
    assert dict(zip(divisions["team_id"], divisions["division"])) == {
        333: "fbs",
        99: "fbs",
        2000: "fcs",
    }


def test_derive_collapses_to_one_row_per_athlete_team_with_last_values(
    positions, divisions
):
    payloads = [
        _game(
            1,
            1,
            [
                {
                    "athlete_id": 5,
                    "team_id": 333,
                    "jersey": "12 ",
                    "position_href": _pos_href(0),
                    "weight": 200.0,
                },
                {
                    "athlete_id": 6,
                    "team_id": 333,
                    "jersey": "7",
                    "position_href": _pos_href(9),
                    "weight": 190.0,
                },
            ],
        ),
        # Week 2: athlete 5 gained weight and a real position; athlete 7 is new.
        _game(
            2,
            2,
            [
                {
                    "athlete_id": 5,
                    "team_id": 333,
                    "jersey": "12",
                    "position_href": _pos_href(9),
                    "weight": 210.0,
                },
                {
                    "athlete_id": 7,
                    "team_id": 2000,
                    "jersey": "3",
                    "position_href": _pos_href(9),
                    "weight": 180.0,
                },
            ],
        ),
    ]
    df = R.derive_rosters(R.roster_rows(payloads), positions, divisions)

    assert df.height == 3, "one row per (team_id, athlete_id)"
    assert list(df.columns) == list(R.OUTPUT_COLS), "pinned schema + column order"

    a5 = df.filter(pl.col("athlete_id") == 5).row(0, named=True)
    assert a5["weight"] == 210.0, "last appearance (by week) supplies attributes"
    assert a5["games_rostered"] == 2
    assert (a5["position_id"], a5["position"], a5["position_abbreviation"]) == (
        9,
        "Running Back",
        "RB",
    )
    assert a5["jersey"] == "12", "ESPN space-pads jersey"
    assert a5["division"] == "fbs"
    assert df.filter(pl.col("athlete_id") == 7).row(0, named=True)["division"] == "fcs"


def test_id_columns_are_int64_and_never_float_stringified(positions, divisions):
    df = R.derive_rosters(
        R.roster_rows(
            [
                _game(
                    1,
                    1,
                    [{"athlete_id": 5, "team_id": 333, "position_href": _pos_href(9)}],
                )
            ]
        ),
        positions,
        divisions,
    )
    for col in R.ID_COLS:
        assert df.schema[col] == pl.Int64, col


def test_season_with_no_captures_returns_the_documented_empty_schema(
    positions, divisions
):
    df = R.derive_rosters([], positions, divisions)
    assert df.height == 0
    assert list(df.columns) == list(R.OUTPUT_COLS)


def test_columns_espn_never_sent_are_null_filled_not_dropped(positions, divisions):
    """A 2004-era payload lacks hand_*/citizenship/jersey_right; schema must not shift."""
    df = R.derive_rosters(
        R.roster_rows(
            [
                _game(
                    1,
                    1,
                    [{"athlete_id": 5, "team_id": 333, "position_href": _pos_href(0)}],
                )
            ]
        ),
        positions,
        divisions,
    )
    for col in ("hand_type", "citizenship", "jersey_right", "display_name"):
        assert col in df.columns
        assert df[col].null_count() == df.height
