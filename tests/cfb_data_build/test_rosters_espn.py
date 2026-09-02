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


# ------------------------------------------------------- CFBD hometown/recruiting


def _cfbd_parquet(rows: list[dict]) -> bytes:
    """A CFBD-shaped roster parquet in memory (string ids, List(Int32) recruits)."""
    import io

    buf = io.BytesIO()
    pl.DataFrame(
        rows,
        schema={
            "athlete_id": pl.Utf8,
            "team": pl.Utf8,
            "home_city": pl.Utf8,
            "home_state": pl.Utf8,
            "home_country": pl.Utf8,
            "home_latitude": pl.Utf8,
            "home_longitude": pl.Utf8,
            "home_county_fips": pl.Utf8,
            "recruit_ids": pl.List(pl.Int32),
            "headshot_url": pl.Utf8,
        },
    ).write_parquet(buf)
    return buf.getvalue()


def _cfbd_row(aid: str, team: str, city: str, recruits: list[int]) -> dict:
    return {
        "athlete_id": aid,
        "team": team,
        "home_city": city,
        "home_state": "TX",
        "home_country": "USA",
        "home_latitude": "30.1",
        "home_longitude": "-97.7",
        "home_county_fips": "48453",
        "recruit_ids": recruits,
        "headshot_url": "http://x/head.png",
    }


def test_cfbd_loader_normalizes_ids_and_dedupes_transfers():
    body = _cfbd_parquet(
        [
            _cfbd_row("5", "A", "Austin", [111]),
            _cfbd_row("5", "B", "Austin", [111]),  # transfer: same athlete, two teams
            _cfbd_row("6", "A", "Waco", [222, 333]),
        ]
    )
    df = R.load_cfbd_rosters(2023, "mem://", fetcher=lambda _u: body)

    assert df.schema["athlete_id"] == pl.Int64  # never via float -> never "5.0"
    assert df.height == 2  # the transfer collapsed losslessly
    assert set(df.columns) == {"athlete_id", *R.CFBD_COLS.values()}
    got = dict(zip(df["athlete_id"], df["cfbd_recruit_ids"]))
    assert got == {5: "[111]", 6: "[222,333]"}  # one dtype across parquet/rds/csv


def test_cfbd_loader_raises_when_a_transfers_rows_disagree():
    body = _cfbd_parquet(
        [
            _cfbd_row("5", "A", "Austin", [111]),
            _cfbd_row("5", "B", "Dallas", [111]),  # same id, different hometown
        ]
    )
    with pytest.raises(RuntimeError, match="not unique after dedup"):
        R.load_cfbd_rosters(2023, "mem://", fetcher=lambda _u: body)


def test_missing_cfbd_season_degrades_to_null_enrichment(positions, divisions):
    cfbd = R.load_cfbd_rosters(1999, "mem://", fetcher=lambda _u: None)
    assert cfbd.height == 0 and cfbd.schema["athlete_id"] == pl.Int64

    payloads = [_game(1, 1, [{"athlete_id": 5, "team_id": 333, "position_href": _pos_href(9)}])]
    df = R.derive_rosters(R.roster_rows(payloads), positions, divisions, cfbd)
    assert df.height == 1
    assert df.row(0, named=True)["cfbd_home_city"] is None


def test_cfbd_join_is_left_and_never_changes_the_espn_row_set(positions, divisions):
    payloads = [
        _game(
            1,
            1,
            [
                {"athlete_id": 5, "team_id": 333, "position_href": _pos_href(9)},
                {"athlete_id": 6, "team_id": 333, "position_href": _pos_href(9)},
            ],
        )
    ]
    espn_only = R.derive_rosters(R.roster_rows(payloads), positions, divisions)

    # athlete 7 exists only in CFBD (the FCS-coverage case) and must NOT be unioned in.
    body = _cfbd_parquet(
        [
            _cfbd_row("5", "A", "Austin", [111]),
            _cfbd_row("5", "B", "Austin", [111]),
            _cfbd_row("7", "C", "Tyler", [999]),
        ]
    )
    cfbd = R.load_cfbd_rosters(2023, "mem://", fetcher=lambda _u: body)
    df = R.derive_rosters(R.roster_rows(payloads), positions, divisions, cfbd)

    assert df.height == espn_only.height == 2
    assert 7 not in df["athlete_id"].to_list()
    assert list(df.columns) == list(R.OUTPUT_COLS) and len(R.OUTPUT_COLS) == 85
    by_id = {r["athlete_id"]: r for r in df.iter_rows(named=True)}
    assert by_id[5]["cfbd_home_city"] == "Austin"
    assert by_id[6]["cfbd_home_city"] is None  # unmatched ESPN row keeps its place
    # hometown is NOT birthplace: ESPN's own column is untouched by the join.
    assert by_id[5]["birth_place_city"] is None
    # every one of the 78 pre-existing columns is byte-identical to the plain build
    pre = [c for c in R.OUTPUT_COLS if c not in R.CFBD_COLS.values()]
    assert len(pre) == 78
    assert df.select(pre).equals(espn_only.select(pre))


# --------------------------------------------------------------------------
# Pre-season degradation: a source that has not published yet
#
# These exercise the DOWNLOADERS, not an injected `fetcher=`. The existing
# `fetcher=lambda _u: None` tests inject past the failure point, which is why a
# pre-season 404 took the whole `cfb_rosters` build down unnoticed: `download`
# RAISES on a 404 rather than returning a non-200 response, so the "return None"
# branch those tests cover was unreachable in production.
# --------------------------------------------------------------------------


def _raising_download(exc):
    """Stand in for ``dl_utils.download``, recording that it was actually called."""
    calls = {"n": 0}

    def fake(url, *a, **k):
        calls["n"] += 1
        raise exc

    return fake, calls


def test_download_bytes_degrades_when_the_asset_is_missing(monkeypatch):
    """A season CFBD has not published yet must yield ``None``, not an exception.

    ``cfb_rosters`` LEFT-joins the CFBD columns onto the ESPN rosters, so an absent
    CFBD asset means null enrichment -- never a failed dataset.
    """
    import sportsdataverse.dl_utils as dl
    from sportsdataverse import errors

    missing = getattr(errors, "NoDataError", None) or errors.NoESPNDataError
    fake, calls = _raising_download(missing("404 for cfb_rosters_2026.parquet"))
    monkeypatch.setattr(dl, "download", fake)

    assert R._download_bytes("https://example.invalid/cfb_rosters_2026.parquet") is None
    assert calls["n"] == 1, "download was never called -- the test mocked past the code under test"


def test_download_degrades_when_the_asset_is_missing(monkeypatch):
    """Same rule for the text downloader (positions reference, per-season teams)."""
    import sportsdataverse.dl_utils as dl
    from sportsdataverse import errors

    missing = getattr(errors, "NoDataError", None) or errors.NoESPNDataError
    fake, calls = _raising_download(missing("404"))
    monkeypatch.setattr(dl, "download", fake)

    assert R._download("https://example.invalid/positions.json") is None
    assert calls["n"] == 1


def test_a_failed_fetch_still_propagates(monkeypatch):
    """Only a definitive 404 degrades.

    A 403, a rate limit, or an exhausted retry budget means the answer is UNKNOWN.
    Swallowing those would publish a season with silently-null enrichment and no
    error -- the failure mode this whole guard exists to prevent.
    """
    import sportsdataverse.dl_utils as dl

    fake, calls = _raising_download(RuntimeError("HTTP 403 rate limited"))
    monkeypatch.setattr(dl, "download", fake)

    with pytest.raises(RuntimeError, match="403"):
        R._download_bytes("https://example.invalid/cfb_rosters_2025.parquet")
    assert calls["n"] == 1


def test_missing_cfbd_season_builds_empty_typed_frame(monkeypatch):
    """End to end: the raise becomes an empty frame carrying the pinned schema."""
    import sportsdataverse.dl_utils as dl
    from sportsdataverse import errors

    missing = getattr(errors, "NoDataError", None) or errors.NoESPNDataError
    fake, _ = _raising_download(missing("404"))
    monkeypatch.setattr(dl, "download", fake)

    out = R.load_cfbd_rosters(2026)
    assert out.height == 0
    assert out.schema["athlete_id"] == pl.Int64
    assert set(R.CFBD_COLS.values()).issubset(set(out.columns))


# --------------------------------------------------------------------------
# The zero-row guard has to mean two opposite things depending on the season
# --------------------------------------------------------------------------


def _stub_empty_build(monkeypatch, completed: int):
    """Make build_season compile zero rows, with `completed` games on the schedule."""
    monkeypatch.setattr(R, "load_positions", lambda *a, **k: {})
    monkeypatch.setattr(R, "load_divisions", lambda *a, **k: {})
    monkeypatch.setattr(R, "load_cfbd_rosters", lambda *a, **k: pl.DataFrame(
        schema={"athlete_id": pl.Int64, **{v: pl.Utf8 for v in R.CFBD_COLS.values()}}))
    monkeypatch.setattr(R, "season_game_ids", lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(R, "fetch_game_rosters", lambda *a, **k: [])  # nothing came back
    monkeypatch.setattr(R, "season_completed_games", lambda *a, **k: completed)


def test_zero_rows_before_kickoff_is_skipped_not_raised(monkeypatch):
    """August: the schedule is published, no game has been played, so there are
    no per-game rosters to fetch. Zero rows is the correct answer."""
    _stub_empty_build(monkeypatch, completed=0)
    out = R.build_season(2026, write=False, publish=False)
    assert out.height == 0


def test_zero_rows_after_games_played_still_raises(monkeypatch):
    """In-season: games are complete and we still compiled nothing, which means
    every roster fetch failed. That must stay loud -- returning quietly would let
    the CLI report success with no artifact for the season."""
    _stub_empty_build(monkeypatch, completed=42)
    with pytest.raises(RuntimeError, match="compiled zero roster rows"):
        R.build_season(2025, write=False, publish=False)
