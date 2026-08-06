"""Tests for the team-week enrichment feeding the pregame model.

Concentrated on ONE failure mode, because it already happened: a join that
produces well-formed ALL-NULL columns. `cfb_roster_talent` returned zero rows,
the talent join yielded three null columns, and the resulting A/B recorded
12.978 -> 12.999 -- which reads as a small negative result about talent and was
in fact no result at all. Talent had never been tested.

A column of nulls exists, has a name, and tells a model nothing. Presence is
not data, so these assert on VALUES.
"""

from __future__ import annotations

import polars as pl
import pytest

from cfb_higher_models.features import ROSTER_COLS, add_roster_context


def _weekly() -> pl.DataFrame:
    """Two teams x two weeks x one season, in the team-week shape."""
    return pl.DataFrame(
        {
            "team_id": ["130", "130", "2", "2"],
            "season": [2023, 2023, 2023, 2023],
            "through_week": [5, 6, 5, 6],
            "EPAplay_off": [0.15, 0.16, 0.05, 0.06],
        }
    )


def _talent() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2023, 2023],
            "team_id": ["130", "2"],
            "talent_composite": [900.0, 500.0],
            "blue_chip_ratio": [0.6, 0.2],
        }
    )


def _retprod() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2023, 2023],
            "team_id": ["130", "2"],
            "off_returning": [0.7, 0.4],
            "overall_returning": [0.7, 0.4],
        }
    )


def test_roster_context_actually_populates(monkeypatch) -> None:
    """The join must yield VALUES, not just columns.

    This is the regression test for the null-column episode: the arms of an A/B
    differed by three columns that were entirely null, and the comparison read
    as a real (negative) result.
    """
    import sportsdataverse.cfb as sdv_cfb

    monkeypatch.setattr(
        sdv_cfb, "load_cfb_team_talent", lambda s: _talent(), raising=False
    )
    monkeypatch.setattr(
        sdv_cfb, "load_cfb_returning_production", lambda s: _retprod(), raising=False
    )

    out = add_roster_context(_weekly())
    for c in ROSTER_COLS:
        assert c in out.columns, c
        assert out[c].null_count() == 0, f"{c} is all-null -- the join did not match"
        assert out[c].n_unique() > 1, f"{c} is constant -- it can tell a model nothing"

    # the value is per (team, season), so it repeats across that team's weeks
    michigan = out.filter(pl.col("team_id") == "130")
    assert michigan["talent_composite"].unique().to_list() == [900.0]


def test_roster_context_is_season_level_not_week_level(monkeypatch) -> None:
    """Talent must NOT vary within a season -- it is a preseason quantity.

    If it did vary by week it would be carrying in-season information, which is
    where a leak would hide. Constant-within-season is the check that it isn't.
    """
    import sportsdataverse.cfb as sdv_cfb

    monkeypatch.setattr(
        sdv_cfb, "load_cfb_team_talent", lambda s: _talent(), raising=False
    )
    monkeypatch.setattr(
        sdv_cfb, "load_cfb_returning_production", lambda s: _retprod(), raising=False
    )

    out = add_roster_context(_weekly())
    per_team = out.group_by("team_id").agg(
        pl.col("talent_composite").n_unique().alias("n")
    )
    assert per_team["n"].max() == 1, per_team


def test_missing_release_degrades_to_nulls_not_a_crash(monkeypatch) -> None:
    """An unpublished season must not take the whole frame down.

    The tree heads handle nulls; a raise here would make the pregame model
    unbuildable the moment the releases lag a season.
    """
    import sportsdataverse.cfb as sdv_cfb

    empty = pl.DataFrame(schema={"season": pl.Int64, "team_id": pl.Utf8})
    monkeypatch.setattr(sdv_cfb, "load_cfb_team_talent", lambda s: empty, raising=False)
    monkeypatch.setattr(
        sdv_cfb, "load_cfb_returning_production", lambda s: empty, raising=False
    )

    out = add_roster_context(_weekly())
    assert out.height == 4  # rows survive
    assert "_rk" not in out.columns  # the join key is cleaned up


@pytest.mark.parametrize("col", ROSTER_COLS)
def test_roster_cols_are_declared(col: str) -> None:
    """ROSTER_COLS is the contract the A/B and the docs both quote."""
    assert isinstance(col, str) and col
