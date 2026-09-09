"""Contract for the vendored 2003 production table.

ESPN's player box starts in 2004, so returning production for 2004 has no S-1
side and `cfb_returning_production(2004)` raised SeasonNotFoundError -- it took
down a whole build (run 34142076600). This table is the 2003 offensive half,
parsed once from CFBD play-by-play (the only source that reaches 2003) and
vendored because 2003 is immutable history.

Rebuild with `python -m cfb_data_build.build_production_2003` (needs CFBD_API_KEY).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

TABLE = Path(__file__).resolve().parents[1] / "data" / "cfb_production_2003.parquet"

EXPECTED_SCHEMA = {
    "season": pl.Int64,
    "team_id": pl.Utf8,
    "player_id": pl.Utf8,
    "player_name": pl.Utf8,
    "unit": pl.Utf8,
    "prod_weight": pl.Float64,
    "position": pl.Utf8,
}


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    if not TABLE.exists():
        pytest.skip(f"{TABLE} not vendored")
    return pl.read_parquet(TABLE)


def test_schema_matches_production_from_box(table):
    """It is consumed as if it came from `_production_from_box`, so it must
    carry that exact shape -- a drifted dtype would break the id join."""
    assert dict(table.schema) == EXPECTED_SCHEMA


def test_every_row_is_season_2003_offense(table):
    assert table["season"].unique().to_list() == [2003]
    # 2003 play text has no tacklers at all, so there is no defensive half to
    # publish; def_returning stays null for 2004, as it is for every season
    # through 2016 anyway.
    assert table["unit"].unique().to_list() == ["offense"]


def test_ids_are_strings_never_float_stringified(table):
    """"123.0" would match no ESPN athlete id -- the recurring SDV id trap."""
    assert not table["player_id"].str.contains(r"^\d+\.0$").any()
    assert not table["team_id"].str.contains(r"^\d+\.0$").any()


def test_unresolved_players_carry_a_namespaced_id(table):
    """A 2003 producer absent from the 2004 roster still belongs in the
    DENOMINATOR, and its synthetic id must never collide with a real one."""
    synthetic = table.filter(pl.col("player_id").str.starts_with("cfbd2003:"))
    real = table.filter(~pl.col("player_id").str.starts_with("cfbd2003:"))
    assert synthetic.height > 0 and real.height > 0
    assert real["player_id"].str.contains(r"^\d+$").all()


def test_no_duplicate_player_rows(table):
    """One row per (team, player): a dupe would double-count that production."""
    assert table.select("team_id", "player_id").is_duplicated().sum() == 0


def test_thin_teams_are_excluded(table):
    """Teams CFBD barely saw produced a fraction off a near-empty denominator
    (five FCS schools at 224-522 yards came out 0.00-0.05)."""
    by_team = table.group_by("team_id").agg(pl.col("prod_weight").sum().alias("yds"))
    assert by_team["yds"].min() >= 1000.0


def test_coverage_is_plausible_for_2003_fbs(table):
    assert 90 <= table["team_id"].n_unique() <= 140
    assert table.height > 1500
    assert table["prod_weight"].min() > 0
