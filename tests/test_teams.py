"""Tests for the raw-bundle -> cfb_teams compile.

Asserted against a VERBATIM SLICE of a real 2023 capture -- one team per rung of
the NCAA group tree (FBS, FCS, D-II, D-III, and one filed straight under group
35) plus their conferences -- not a hand-written payload. The shapes that break
this compile (the logo ``rel`` lists, the season-type-3 group ref, a groups ref
pointing AT a structural node, string ids) are exactly the ones a synthetic
fixture gets wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cfb_data_build.teams import (
    CFBD_SCHEMA,
    SCHEMA,
    bundle_url,
    compile_teams,
    enrich_cfbd,
    team_info_url,
)

#: compile_teams appends the CFBD backport, so the documented shape is both.
FULL_SCHEMA = {**SCHEMA, **CFBD_SCHEMA}

FIXTURE = (
    Path(__file__).parent
    / "cfb_data_build"
    / "fixtures"
    / "teams_bundle_2023_slice.json"
)


def _bundle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_compiles_real_slice():
    df = compile_teams(_bundle())
    assert df.height == 5
    assert df.schema == FULL_SCHEMA
    auburn = df.filter(pl.col("team_id") == 2).to_dicts()[0]
    assert auburn["display_name"] == "Auburn Tigers"
    assert auburn["division"] == "fbs"
    # The team's own groups $ref lives under season type 3 while the group-80
    # children live under type 2; the join is on the id, so this must resolve.
    # It resolves to the DIVISION within the conference (group 7, "SEC - West"),
    # so the conference columns must carry the walked-up parent, not that.
    assert auburn["team_group_id"] == 7
    assert auburn["team_group_name"] == "SEC - West"
    assert auburn["conference_id"] == 8
    assert auburn["conference_name"] == "Southeastern Conference"
    # ESPN lowercases the CONFERENCE-level abbreviation ("sec"); shortName keeps case.
    assert auburn["conference_abbreviation"] == "sec"
    assert auburn["conference_short_name"] == "SEC"
    assert auburn["conference_parent_id"] == 80
    assert auburn["team_logo"].endswith("/ncaa/500/2.png")
    assert "500-dark" in auburn["team_logo_dark"]
    assert auburn["conference_logo"].startswith("https://")
    assert auburn["venue_id"] == 3785
    assert auburn["venue_name"] == "Jordan-Hare Stadium"


def test_division_comes_from_group_membership():
    df = compile_teams(_bundle())
    assert set(df["division"].to_list()) == {"fbs", "fcs", "d2", "d3", "d2_d3"}
    assert df.filter(pl.col("team_id") == 13)["division"].item() == "fcs"


def test_ids_are_int64_never_float_derived():
    df = compile_teams(_bundle())
    for col in ("team_id", "conference_id", "venue_id"):
        assert df.schema[col] == pl.Int64, col
    # A float-origin id stringifies as "123.0"; the compile must never produce one.
    assert df["team_id"].cast(pl.Utf8).to_list() == ["2", "3", "11", "13", "95"]


def test_missing_team_payload_keeps_the_row():
    """A team the scraper could not fetch must survive as a null row, not vanish."""
    bundle = _bundle()
    bundle["teams"].pop("13")
    df = compile_teams(bundle)
    assert df.height == 5
    ghost = df.filter(pl.col("team_id") == 13).to_dicts()[0]
    assert ghost["division"] == "fcs"
    assert ghost["display_name"] is None
    assert ghost["is_exhibition"] is False  # unfetched != exhibition squad


def test_conference_walkup_stops_at_a_conference_without_divisions():
    """Cal Poly's group IS the conference (Big Sky has no divisions) -- no walk."""
    df = compile_teams(_bundle())
    row = df.filter(pl.col("team_id") == 13).to_dicts()[0]
    assert row["team_group_id"] == row["conference_id"] == 20
    assert row["conference_name"] == "Big Sky Conference"
    assert row["conference_parent_id"] == 81


def test_exhibition_squads_are_flagged():
    """ESPN files all-star squads inside group 80/81; they must be flaggable."""
    bundle = _bundle()
    bundle["divisions"]["80"].append("3144")
    bundle["teams"]["3144"] = {
        "id": "3144",
        "displayName": "SOUTH All-Stars",
        "isAllStar": True,
    }
    df = compile_teams(bundle)
    assert df.filter(pl.col("team_id") == 3144)["is_exhibition"].item() is True
    assert df.filter(pl.col("team_id") == 2)["is_exhibition"].item() is False


def test_empty_bundle_keeps_the_documented_schema():
    df = compile_teams(
        {"season": 2023, "divisions": {}, "teams": {}, "conferences": {}}
    )
    assert df.height == 0
    assert df.schema == FULL_SCHEMA


def test_bundle_url_is_http_not_a_local_path():
    assert bundle_url(2023).startswith("https://raw.githubusercontent.com/")
    assert bundle_url(2023).endswith("/cfb/teams/json/2023.json")


def test_division_covers_the_whole_ncaa_tree():
    """Every classification group resolves to its own label, leaf beating parent."""
    df = compile_teams(_bundle())
    got = dict(zip(df["team_id"].to_list(), df["division"].to_list()))
    assert got == {2: "fbs", 3: "d3", 11: "d2", 13: "fcs", 95: "d2_d3"}


def test_group_35_direct_teams_are_kept_and_get_no_fabricated_conference():
    """ESPN files 107 of 2023's 800 teams straight under group 35 -- they are real.

    Their ``groups`` ref points AT the structural node, so a naive walk would
    publish "Division II/III" as their conference name.
    """
    row = compile_teams(_bundle()).filter(pl.col("team_id") == 95).to_dicts()[0]
    assert row["display_name"] == "Pikeville Bears"
    assert row["division"] == "d2_d3"
    assert row["team_group_id"] == 35  # the structural node is still reported as-is
    assert row["team_group_name"] == "Division II/III"
    assert row["conference_id"] is None
    assert row["conference_name"] is None
    # Same for a D-II team whose own group ref is the structural D-II node.
    mesa = compile_teams(_bundle()).filter(pl.col("team_id") == 11).to_dicts()[0]
    assert mesa["team_group_id"] == 57
    assert mesa["conference_name"] is None


def test_is_fbs_is_never_null():
    """`pl.col("division") == "fbs"` is NULL for a groupless team; is_fbs is not."""
    bundle = _bundle()
    bundle["teams"]["999999"] = {"id": "999999", "displayName": "Ungrouped"}
    df = compile_teams(bundle)
    assert df.schema["is_fbs"] == pl.Boolean
    assert df["is_fbs"].null_count() == 0
    assert df.filter(pl.col("team_id") == 999999)["is_fbs"].item() is False
    assert df.filter(pl.col("team_id") == 999999)["division"].item() is None
    # The two halves must partition the frame exactly -- no row lost to a null.
    assert (
        df.filter(pl.col("is_fbs")).height + df.filter(~pl.col("is_fbs")).height
        == df.height
    )
    assert df["is_fbs"].sum() == 1


def test_exhibition_flag_does_not_leak_outside_division_i():
    """Conference-less + logo-less describes hundreds of real D-II/D-III programs."""
    df = compile_teams(_bundle())
    outside = df.filter(~pl.col("division").is_in(["fbs", "fcs"]))
    assert outside.height == 3
    assert outside["is_exhibition"].to_list() == [False, False, False]


# --- CFBD backport -------------------------------------------------------------
#
# The ESPN universe is much larger than CFBD's, so the join has to be provably
# left: a regression here deletes two thirds of the dataset, quietly.


def _team_info() -> pl.DataFrame:
    """Two of the fixture's five teams, shaped like the released team_info."""
    return pl.DataFrame(
        {
            "team_id": pl.Series([2, 13], dtype=pl.Int32),  # CFBD ships Int32
            "school": ["Auburn", "Cal Poly"],
            "mascot": ["Tigers", "Mustangs"],
            "alt_name1": ["AUB", "CP"],
            "alt_name2": [None, None],
            "alt_name3": [None, None],
            "conference": ["SEC", "Big Sky"],
            "classification": ["fbs", "fcs"],
            "city": ["Auburn", "San Luis Obispo"],
            "state": ["AL", "CA"],
            "country_code": ["US", "US"],
            "timezone": ["America/Chicago", "America/Los_Angeles"],
            "latitude": [32.6024, 35.2999],
            "longitude": [-85.4894, -120.6592],
            "elevation": ["216.7", "70.1"],
            "capacity": pl.Series([88043, 11075], dtype=pl.Int32),
            "dome": [False, False],
            "grass": [True, True],
        }
    ).with_columns(pl.col("team_id").cast(pl.Int64))


def test_cfbd_backport_is_left_never_inner():
    df = compile_teams(_bundle(), _team_info())
    assert df.height == 5  # all five ESPN rows survive
    matched = df.filter(pl.col("school").is_not_null())
    assert matched.height == 2
    # The three unmatched (D-II/D-III) rows keep nulls, not dropped rows.
    unmatched = df.filter(pl.col("school").is_null())
    assert unmatched["team_id"].to_list() == [3, 11, 95]
    assert unmatched["classification"].to_list() == [None, None, None]


def test_cfbd_conference_cannot_be_confused_with_the_espn_family():
    df = compile_teams(_bundle(), _team_info())
    row = df.filter(pl.col("team_id") == 2).to_dicts()[0]
    assert row["cfbd_conference"] == "SEC"
    assert "conference" not in df.columns  # CFBD's bare name never lands
    # ESPN's own conference columns are untouched by the backport.
    assert row["conference_name"] == "Southeastern Conference"
    assert row["conference_id"] == 8


def test_cfbd_backport_leaves_espn_semantics_alone():
    """classification is a second opinion, not a replacement for division/is_fbs."""
    plain = compile_teams(_bundle())
    joined = compile_teams(_bundle(), _team_info())
    for col in ("division", "is_fbs", "is_exhibition", "abbreviation", "color"):
        assert joined[col].to_list() == plain[col].to_list(), col
    assert joined.filter(pl.col("team_id") == 2)["classification"].item() == "fbs"


def test_join_key_dtype_mismatch_raises_rather_than_papering_over():
    df = compile_teams(_bundle())
    bad = _team_info().with_columns(pl.col("team_id").cast(pl.Utf8))
    with pytest.raises(ValueError, match="team_id dtype mismatch"):
        enrich_cfbd(df, bad)


def test_duplicate_cfbd_rows_raise_rather_than_multiplying_espn_rows():
    df = compile_teams(_bundle())
    dup = pl.concat([_team_info(), _team_info().head(1)])
    with pytest.raises(ValueError, match="duplicate team_id"):
        enrich_cfbd(df, dup)


def test_absent_team_info_still_ships_the_documented_schema():
    df = compile_teams(_bundle(), None)
    assert df.schema == FULL_SCHEMA
    assert df["classification"].null_count() == df.height


def test_team_info_url_is_the_release_not_a_local_path():
    assert team_info_url(2023) == (
        "https://github.com/sportsdataverse/sportsdataverse-data/"
        "releases/download/cfb_team_info/cfb_team_info_2023.parquet"
    )
