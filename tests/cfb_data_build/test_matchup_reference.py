"""The eleven non-CFBD side-input columns: derivation rules and 2025 agreement.

Oracle: ``fixtures/matchup/matchup_line_2025.csv``. The reference tables live
in ``data/`` (two derived by ``matchup_reference refresh``, two imported).

Agreement measured 2026-09-15 on the 2025 line (home / away), floored with
margin below and never lowered:

======================= ============= =====================================
column                  observed      why it is not 1.000
======================= ============= =====================================
team_talent_weighted    1.000 / 1.000 -
off_rtprod              0.994 / 0.992 a team the table predates is null
def_rtprod              0.994 / 0.992 (one: a new FBS member)
ovr_rtprod              0.955 / 0.959 recomputed as the rounded mean of the
                                      two sides, as the source recomputes
                                      it; the gap is rounding on halves
oc_cont                 0.889 / 0.890 derived from a newer coordinator
dc_cont                 0.894 / 0.903 vintage than the 2025 line used
returning_qb            0.847 / 0.833 a DEFINED rule (below), measured
                                      against a hand projection
qb_name                 0.673 / 0.651 "
athlete_id              0.693 / 0.673 "
qb_starter_years        0.678 / 0.656 "
qb_games                0.069 / 0.066 "
======================= ============= =====================================

``returning_qb`` follows a stated rule rather than the source's: it is 1 when
this season's starter was, for the same team last season, EITHER the
opening-day starter OR the starter of a plurality of games, else 0.

**The rest of the QB block measures a different quantity on purpose.** The
delivered line carries a hand-maintained PRESEASON PROJECTION of each team's
starter. This port derives the season's REALIZED starter -- the passer with
the most attempts in the ESPN play-by-play release -- which is reproducible
and backfilled to 2004, but is known only after the fact: for the season it
describes these five columns are post-hoc, and a backtest that feeds them as
pregame inputs will be optimistic. ``qb_games`` diverges furthest because the
source counted appearances through an athlete id its own name join left null
for a quarter of rows, so its counts are systematically low. Treat the QB
agreement numbers as descriptive, not as a parity bar.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cfb_data_build.matchup_features import to_display_names
from cfb_data_build.matchup_reference import (
    coach_continuity,
    load_coach_continuity,
    load_qb_starters,
    match_team_names,
    qb_game_counts,
    qb_starters,
    season_reference,
    starters_from_pbp,
)

FIX = Path(__file__).parent / "fixtures" / "matchup"

FLOORS = {
    "team_talent_weighted": 1.0,
    "off_rtprod": 0.99,
    "def_rtprod": 0.98,
    "ovr_rtprod": 0.94,
    "oc_cont": 0.87,
    "dc_cont": 0.87,
    "returning_qb": 0.80,
    "qb_name": 0.63,
    "athlete_id": 0.65,
    "qb_starter_years": 0.63,
}
#: measured 0.066-0.069 -- a definitional difference, pinned so a silent change shows
QB_GAMES_AGREEMENT = 0.05


@pytest.fixture(scope="module")
def oracle() -> pl.DataFrame:
    return pl.read_csv(
        FIX / "matchup_line_2025.csv", null_values=["NA", ""], infer_schema_length=10000
    )


@pytest.fixture(scope="module")
def reference() -> pl.DataFrame:
    teams = pl.read_parquet(FIX / "cfbd_teams_2025.parquet")
    ref = season_reference(2025, teams=teams, schools=teams["team"].to_list())
    return to_display_names(ref, "team")


def _match(a: pl.Series, b: pl.Series) -> float:
    if a.dtype.is_numeric() and b.dtype.is_numeric():
        x, y = a.cast(pl.Float64).to_numpy(), b.cast(pl.Float64).to_numpy()
        ok = np.isclose(x, y, rtol=0, atol=1e-6, equal_nan=True) | (
            np.isnan(x) & np.isnan(y)
        )
    else:
        ok = (
            (a.cast(pl.Utf8) == b.cast(pl.Utf8))
            .fill_null(a.is_null() & b.is_null())
            .to_numpy()
        )
    return float(ok.mean())


def test_reference_agrees_with_the_delivered_line(
    oracle: pl.DataFrame, reference: pl.DataFrame
) -> None:
    for prefix in ("home", "away"):
        key = f"{prefix}_team"
        j = oracle.select(key, *[f"{prefix}_{c}" for c in (*FLOORS, "qb_games")]).join(
            reference.rename({"team": key}), on=key, how="left"
        )
        assert j.height == 1557 // 2 + 1 or j.height > 700
        for col, floor in FLOORS.items():
            got = _match(j[f"{prefix}_{col}"], j[col])
            assert got >= floor, (prefix, col, got, floor)
        qbg = _match(j[f"{prefix}_qb_games"], j["qb_games"])
        assert qbg >= QB_GAMES_AGREEMENT, (prefix, qbg)


def test_only_a_new_fbs_member_is_unfilled(
    oracle: pl.DataFrame, reference: pl.DataFrame
) -> None:
    """Every team on the 2025 line resolves except ones the tables predate."""
    on_line = set(oracle["home_team"].to_list()) | set(oracle["away_team"].to_list())
    filled = set(reference.filter(pl.col("off_rtprod").is_not_null())["team"].to_list())
    assert sorted(on_line - filled) == ["Missouri State"]


def test_continuity_is_backfilled_to_the_first_derivable_season() -> None:
    """The source shipped one season; the derivation covers every season but the first."""
    cont = load_coach_continuity()
    assert cont["season"].min() == 2014 and cont["season"].max() >= 2026
    assert cont["season"].n_unique() >= 13
    assert set(cont["oc_cont"].unique()) <= {0, 1}
    assert cont.unique(subset=["season", "school_mascot"]).height == cont.height


def test_continuity_matches_any_coordinator_and_never_leaves_null() -> None:
    """Co-coordinators split on '/', punctuation drift is not a change, missing is 0."""
    src = pl.DataFrame(
        {
            "season": [2020, 2021, 2022, 2023],
            "school_mascot": ["A Aces"] * 4,
            "offensive_coordinators": [
                "D. J. Durkin",
                "D.J. Durkin / Pat Jones",  # retained, punctuation differs
                "Pat Jones",  # the co-coordinator stays
                None,  # missing name: never a match
            ],
            "defensive_coordinators": ["X One", "Y Two", "Y Two", "Y Two"],
        }
    )
    out = coach_continuity(src).sort("season")
    assert out["oc_cont"].to_list() == [1, 1, 0]
    assert out["dc_cont"].to_list() == [0, 1, 1]
    assert out["season"].to_list() == [2021, 2022, 2023]


def test_starter_is_the_passer_with_the_most_attempts() -> None:
    pbp = pl.DataFrame(
        {
            "season": [2008] * 5,
            "game_id": [1, 1, 2, 2, 2],
            "pos_team": ["A", "A", "A", "A", "B"],
            "pass": [1, 1, 1, 0, 1],
            "passer_player_name": ["Backup", "Starter", "Starter", None, "Only"],
            "passer_player_id": [2, 1, 1, None, 3],
        }
    )
    out = starters_from_pbp(pbp).sort("team")
    assert out["qb_name"].to_list() == ["Starter", "Only"]
    assert out["athlete_id"].to_list() == [1, 3]
    assert out.schema["athlete_id"] == pl.Int64
    # a run play never identifies a starter
    counts = qb_game_counts(pbp).sort("athlete_id")
    assert counts["athlete_id"].to_list() == [1, 2, 3]
    assert counts["games"].to_list() == [2, 1, 1]


def test_qb_games_counts_every_prior_season() -> None:
    """A season the QB did not appear in must not break the career count."""
    s = pl.DataFrame(
        {
            "season": [2016, 2019],
            "team": ["A", "A"],
            "qb_name": ["x", "x"],
            "athlete_id": [1, 1],
        }
    )
    appear = pl.DataFrame(
        {"season": [2015, 2017], "athlete_id": [1, 1], "games": [2, 5]}
    )
    out = qb_starters(s, appear).sort("season")
    assert out["qb_games"].to_list() == [2, 7]
    assert out["qb_starter_years"].to_list() == [0, 1]
    # same athlete, same team, but not consecutive -> not a returning starter
    assert out["returning_qb"].to_list() == [0, 0]


def test_qb_table_is_backfilled_to_2004_with_an_id_for_every_row() -> None:
    qb = load_qb_starters()
    assert qb["season"].min() == 2004 and qb["season"].n_unique() >= 22
    assert qb["athlete_id"].null_count() == 0
    assert qb.unique(subset=["season", "team"]).height == qb.height


def test_unmatched_team_names_are_reported_not_dropped_silently() -> None:
    frame = pl.DataFrame({"team": ["Ole Miss", "Not A School"], "v": [1, 2]})
    out, missing = match_team_names(frame, ["Mississippi", "Alabama"])
    assert out["team"].to_list() == ["Mississippi"]
    assert missing == ["Not A School"]


def test_accented_and_plain_spellings_are_one_key() -> None:
    """The coordinator table spells one school both ways; both must resolve."""
    from cfb_data_build.matchup_reference import _contract

    assert _contract("San José State Spartans") == _contract("San Jose State Spartans")
    out, missing = match_team_names(
        pl.DataFrame({"team": ["San Jose State"], "v": [1]}), ["San José State"]
    )
    assert out["team"].to_list() == ["San José State"] and missing == []


def test_continuity_requires_the_immediately_prior_season() -> None:
    """A gap year must not let a two-year-old staff count as continuity."""
    src = pl.DataFrame(
        {
            "season": [2018, 2020],  # no 2019 row for this school
            "school_mascot": ["A Aces"] * 2,
            "offensive_coordinators": ["Same Name", "Same Name"],
            "defensive_coordinators": ["Same D", "Same D"],
        }
    )
    out = coach_continuity(src)
    assert out["season"].to_list() == [2020]
    assert out["oc_cont"].to_list() == [0] and out["dc_cont"].to_list() == [0]


def test_conflicting_spellings_for_one_school_raise() -> None:
    """Two source names resolving to one school must not be settled by sort order."""
    clash = pl.DataFrame({"team": ["Connecticut", "UConn"], "off_rtprod": [0.5, 0.9]})
    with pytest.raises(ValueError, match="different values"):
        match_team_names(clash, ["UConn"])
    agree = pl.DataFrame({"team": ["Connecticut", "UConn"], "off_rtprod": [0.5, 0.5]})
    out, _ = match_team_names(agree, ["UConn"])
    assert out.height == 1


def test_every_imported_table_is_unique_on_its_key() -> None:
    """A duplicated team-season would raise mid-build; catch it at rest instead."""
    from cfb_data_build.matchup_reference import (
        load_coordinators,
        load_returning_production,
        load_roster_talent,
    )

    for frame, keys in (
        (load_returning_production(), ["season", "team"]),
        (load_roster_talent(), ["season", "team"]),
        (load_coordinators(), ["season", "school_mascot"]),
        (load_qb_starters(), ["season", "team"]),
    ):
        assert frame.unique(subset=keys).height == frame.height


def test_a_partial_qb_refresh_is_refused() -> None:
    """Rebuilding from a later start would drop the backfill and undercount careers."""
    from cfb_data_build.matchup_reference import main

    with pytest.raises(SystemExit, match="refusing to rebuild"):
        main(
            [
                "refresh",
                "--start-season",
                "2014",
                "--end-season",
                "2025",
                "--only",
                "qb",
            ]
        )


def test_unresolved_spellings_are_reported() -> None:
    """A name that matches no school must surface, not just left-join to null."""
    teams = pl.read_parquet(FIX / "cfbd_teams_2025.parquet")
    with pytest.warns(UserWarning, match="matched no CFBD school"):
        season_reference(2025, teams=teams, schools=["Alabama"])


def test_a_season_before_the_coverage_floors_builds_with_nulls() -> None:
    """2004 predates every table but the QB one; the line must still build.

    Regression: the collision guard once removed the empty-mapping path, so a
    pre-coverage season raised ColumnNotFoundError instead of leaving those
    columns null.
    """
    teams = pl.read_parquet(FIX / "cfbd_teams_2025.parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = season_reference(2004, teams=teams, schools=teams["team"].to_list())
    assert out.height == teams.height
    for col in ("off_rtprod", "def_rtprod", "ovr_rtprod", "team_talent_weighted"):
        assert out[col].null_count() == out.height, col
    # the QB table does reach 2004
    assert out["qb_name"].null_count() < out.height


def test_match_team_names_on_an_empty_table_returns_a_typed_frame() -> None:
    empty = pl.DataFrame(
        schema={"season": pl.Int64, "team": pl.Utf8, "off_rtprod": pl.Float64}
    )
    out, missing = match_team_names(empty, ["Alabama"])
    assert out.height == 0 and missing == []
    assert "team" in out.columns and out.schema["off_rtprod"] == pl.Float64


def test_returning_qb_is_the_opening_day_or_plurality_starter() -> None:
    """Either prior-season starter counts; a third quarterback does not.

    2023 team A: QB 1 starts the opener and is then replaced, so QB 1 is the
    opening-day starter and QB 2 started a plurality of games. Both make a
    2024 starter "returning".
    """
    from cfb_data_build.matchup_reference import game_starters, prior_season_starters

    pbp = pl.DataFrame(
        {
            "season": [2023] * 8,
            "week": [1, 1, 2, 2, 3, 3, 4, 4],
            "game_id": [10, 10, 11, 11, 12, 12, 13, 13],
            "pos_team": ["A"] * 8,
            "pass": [True] * 8,
            "passer_player_id": [1, 1, 2, 2, 2, 2, 2, 2],
        }
    )
    prior = prior_season_starters(game_starters(pbp))
    assert prior.row(0, named=True)["opening_starter"] == 1
    assert prior.row(0, named=True)["plurality_starter"] == 2

    appear = pl.DataFrame({"season": [2023], "athlete_id": [1], "games": [1]})
    for athlete, expected in ((1, 1), (2, 1), (3, 0)):
        starter = pl.DataFrame(
            {
                "season": [2024],
                "team": ["A"],
                "qb_name": ["x"],
                "athlete_id": [athlete],
            }
        )
        out = qb_starters(starter, appear, prior)
        assert out["returning_qb"].to_list() == [expected], athlete


def test_returning_qb_needs_the_same_team() -> None:
    """A quarterback who started elsewhere last season is not returning."""
    from cfb_data_build.matchup_reference import prior_season_starters

    prior = prior_season_starters(
        pl.DataFrame(
            {
                "season": [2023],
                "week": [1],
                "game_id": [1],
                "team": ["A"],
                "athlete_id": [7],
                "attempts": [20],
            }
        )
    )
    moved = pl.DataFrame(
        {"season": [2024], "team": ["B"], "qb_name": ["x"], "athlete_id": [7]}
    )
    out = qb_starters(
        moved,
        pl.DataFrame(
            schema={"season": pl.Int64, "athlete_id": pl.Int64, "games": pl.Int64}
        ),
        prior,
    )
    assert out["returning_qb"].to_list() == [0]
