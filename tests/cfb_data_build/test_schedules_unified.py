"""Offline unit tests for the unified ``cfb_schedules`` build.

``tidy_cfbd`` and ``unify`` are pure, so the whole contract is testable without
a CFBD key or a network hop.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_data_build import schedules_unified
from cfb_data_build.schedules_unified import (
    EXCLUDED_CFBD,
    SCHEMA,
    attach_ranks,
    build_schedules,
    tidy_cfbd,
    unify,
)


def _cfbd(**over):
    game = {
        "id": 1,
        "season": 2023,
        "week": 1,
        "seasonType": "regular",
        "startDate": "2023-08-26T18:30:00.000Z",
        "startTimeTBD": False,
        "completed": True,
        "neutralSite": False,
        "conferenceGame": True,
        "attendance": 100,
        "venueId": 7,
        "venue": "Stadium",
        "homeId": 10,
        "homeTeam": "Home",
        "homeClassification": "fbs",
        "homeConference": "SEC",
        "homePoints": 21,
        "awayId": 20,
        "awayTeam": "Away",
        "awayClassification": "fbs",
        "awayConference": "SEC",
        "awayPoints": 14,
        "homePregameElo": 1700,
        "excitementIndex": 5.0,
        "playoff": None,
    }
    game.update(over)
    return game


def _espn(**over):
    row = {
        "game_id": 1,
        "season": 2023,
        "week": 1,
        "season_type": 2,
        "game_date": "2023-08-26T18:30Z",
        "neutral_site": False,
        "conference_competition": False,
        "home_id": 10,
        "away_id": 20,
        "home_team": "Home",
        "away_team": "Away",
        "home_abbreviation": "HME",
        "away_abbreviation": "AWY",
        "home_score": 21,
        "away_score": 14,
        "home_winner": True,
        "away_winner": False,
        "venue": "Stadium",
        "attendance": 100,
        "status": "STATUS_FINAL",
    }
    row.update(over)
    return row


def test_schema_is_stable_and_excludes_the_modeling_columns():
    df = unify(tidy_cfbd([_cfbd()]), pl.from_dicts([_espn()]), 2023)
    assert list(df.columns) == list(SCHEMA)
    assert not set(df.columns) & set(EXCLUDED_CFBD)
    assert df.schema["game_id"] == pl.Int64


def test_null_division_reads_false_not_null():
    """The gotcha: division is null for teams outside ESPN's group-80/81
    universe. ``pl.col(x) == 'fbs'`` yields NULL there; the derived flags must
    yield FALSE."""
    games = [
        _cfbd(id=1),
        _cfbd(id=2, awayClassification=None),
        _cfbd(id=3, homeClassification=None, awayClassification=None),
        _cfbd(id=4, homeClassification="fcs", awayClassification="fcs"),
    ]
    df = unify(tidy_cfbd(games), pl.DataFrame(), 2023).sort("game_id")
    assert df["fbs_game"].null_count() == 0
    assert df["fbs_participant"].null_count() == 0
    assert df["fbs_game"].to_list() == [True, False, False, False]
    assert df["fbs_participant"].to_list() == [True, True, False, False]


def test_cfbd_is_canonical_and_espn_only_fallbacks_fill_the_gaps():
    cfbd = tidy_cfbd(
        [_cfbd(homePoints=None, awayPoints=None, completed=None, attendance=None)]
    )
    df = unify(cfbd, pl.from_dicts([_espn()]), 2023)
    row = df.to_dicts()[0]
    assert row["home_points"] == 21 and row["away_points"] == 14  # ESPN fallback
    assert row["completed"] is True  # from STATUS_FINAL
    assert row["attendance"] == 100
    assert row["start_date"] == "2023-08-26T18:30:00.000Z"  # CFBD wins over game_date
    assert row["conference_game"] is True  # CFBD wins over conference_competition
    assert row["conference_competition"] is False  # kept: the two genuinely differ
    assert row["home_abbreviation"] == "HME"
    assert row["season_type"] == "regular" and row["season_type_id"] == 2


def test_espn_only_rows_are_unioned_in_with_their_status():
    """A COVID cancellation CFBD drops is still a schedule fact."""
    espn = pl.from_dicts(
        [
            _espn(),
            _espn(
                game_id=99,
                status="STATUS_CANCELED",
                home_score=0,
                away_score=0,
                home_winner=False,
                away_winner=False,
            ),
        ]
    )
    df = unify(tidy_cfbd([_cfbd()]), espn, 2023).sort("game_id")
    assert df.height == 2
    extra = df.filter(pl.col("game_id") == 99).to_dicts()[0]
    assert extra["status"] == "STATUS_CANCELED"
    assert extra["completed"] is False
    # `source` does not ship: `status` is what identifies an unplayed game.
    assert extra["season_type"] == "regular" and extra["season_type_id"] == 2
    assert extra["season"] == 2023


def test_winner_is_derived_when_espn_did_not_state_it():
    df = unify(tidy_cfbd([_cfbd()]), pl.DataFrame(), 2023)
    row = df.to_dicts()[0]
    assert row["home_winner"] is True and row["away_winner"] is False


def test_empty_sources_still_ship_the_documented_schema():
    df = unify(tidy_cfbd([]), pl.DataFrame(), 2023)
    assert df.height == 0
    assert list(df.columns) == list(SCHEMA)


# --- rank at kickoff (IF-0) ---------------------------------------------------


def _master(rows, game_id_dtype=pl.Int32):
    """The raw schedule master's rank slice: Int32 ``game_id``, 99 = unranked."""
    return pl.DataFrame(
        rows,
        schema={
            "game_id": game_id_dtype,
            "home_current_rank": pl.Int64,
            "away_current_rank": pl.Int64,
        },
        orient="row",
    )


def _three_games():
    return unify(tidy_cfbd([_cfbd(id=i) for i in (1, 2, 3)]), pl.DataFrame(), 2023)


def _ranks(df):
    return {r["game_id"]: (r["home_rank"], r["away_rank"]) for r in df.to_dicts()}


def test_rank_at_kickoff_joins_the_master_and_99_reads_unranked():
    out = attach_ranks(_three_games(), _master([(1, 3, 1), (2, 99, 12)]))
    assert list(out.columns) == list(SCHEMA)
    assert out.schema["home_rank"] == pl.Int64
    assert out.schema["away_rank"] == pl.Int64
    # game 3 is absent from the master (2003, FCS-only): null, never a guess
    assert _ranks(out) == {1: (3, 1), 2: (None, 12), 3: (None, None)}


def test_rank_outside_1_to_25_is_null():
    out = attach_ranks(_three_games(), _master([(1, 0, 26), (2, -1, 25), (3, 1, 99)]))
    assert _ranks(out) == {1: (None, None), 2: (None, 25), 3: (1, None)}
    shipped = pl.concat([out["home_rank"], out["away_rank"]]).drop_nulls()
    assert shipped.is_between(1, 25).all()


def test_string_master_game_id_raises_before_the_join():
    """A Utf8 id would join nothing (or cast silently); fail loudly instead."""
    with pytest.raises(AssertionError, match="game_id"):
        attach_ranks(_three_games(), _master([("1", 3, 1)], game_id_dtype=pl.Utf8))


def _offline(monkeypatch):
    monkeypatch.setattr(
        schedules_unified,
        "fetch_cfbd_games",
        lambda season, **_: [_cfbd(id=i) for i in (1, 2, 3)],
    )
    monkeypatch.setattr(
        schedules_unified, "load_espn", lambda season, **_: pl.DataFrame()
    )


def test_build_joins_an_int32_master_found_via_cfb_raw_root(tmp_path, monkeypatch):
    _offline(monkeypatch)
    (tmp_path / "cfb").mkdir()
    _master([(1, 3, 1), (2, 99, 12)]).with_columns(
        season=pl.lit(2023, pl.Int32)  # the real master carries ~80 more columns
    ).write_parquet(tmp_path / "cfb" / "cfb_schedule_master.parquet")
    monkeypatch.setenv("CFB_RAW_ROOT", str(tmp_path))
    out = build_schedules(2023)
    assert list(out.columns) == list(SCHEMA)
    assert _ranks(out) == {1: (3, 1), 2: (None, 12), 3: (None, None)}


def test_missing_master_ships_all_null_int64_ranks_and_warns_once(
    tmp_path, monkeypatch
):
    """No raw store (CI fetch failed, no sibling checkout) is an absent input:
    the columns still ship, all null, and the build says so once."""
    _offline(monkeypatch)
    monkeypatch.setenv("CFB_RAW_ROOT", str(tmp_path))  # no master under it
    with pytest.warns(UserWarning, match="schedule master") as caught:
        out = build_schedules(2023)
    assert len(caught) == 1
    assert list(out.columns) == list(SCHEMA)
    assert out.schema["home_rank"] == pl.Int64
    assert out.schema["away_rank"] == pl.Int64
    assert out.height == 3
    assert out["home_rank"].null_count() == 3
    assert out["away_rank"].null_count() == 3
