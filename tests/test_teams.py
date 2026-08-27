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

from cfb_data_build.teams import SCHEMA, bundle_url, compile_teams

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
    assert df.schema == SCHEMA
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
    assert df.schema == SCHEMA


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
