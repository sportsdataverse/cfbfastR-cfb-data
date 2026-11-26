"""The adjusted-EPA ledger (stage 68): ``team_summaries``' Adj EPA/Play laid out game by game.

The model is ``sportsdataverse.cfb.cfb_adjusted_epa_ledger``; this module adds no formula.
It runs it on exactly the frame ``build_team_summaries`` hands ``cfb_adjusted_epa``
(the released pbp + schedule through :func:`prepare_plays_input` and
:func:`add_derived_metrics`), so each team's ``contribution`` column sums to the
``net_adj_epa`` that season's ``cfb_team_summaries`` publishes. The build checks that sum
against a fresh ``cfb_adjusted_epa`` fit and fails the season if it misses by more than
1e-9, rather than publish a ledger that disagrees with the number it explains.

Ids are published as Int64 (the released summaries' dtype); sdv-py returns them as Utf8.
"""

from __future__ import annotations

import polars as pl
from sportsdataverse.cfb import cfb_adjusted_epa, cfb_adjusted_epa_ledger

from cfb_data_build.summaries_input import prepare_plays_input
from cfb_data_build.team_summaries import add_derived_metrics

#: ``cfb_team_adjustment_ledger``, in column order. ``season_type`` is ESPN's (2 regular, 3 post).
OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "game_id": pl.Int64,
    "week": pl.Int64,
    "season_type": pl.Int64,
    "team_id": pl.Int64,
    "pos_team": pl.Utf8,
    "opponent_id": pl.Int64,
    "site": pl.Utf8,
    "off_plays": pl.Int64,
    "def_plays": pl.Int64,
    "raw_off_epa": pl.Float64,
    "raw_def_epa": pl.Float64,
    "raw_net_epa": pl.Float64,
    "off_strength_faced": pl.Float64,
    "def_strength_faced": pl.Float64,
    "schedule_adj": pl.Float64,
    "adj_off_epa": pl.Float64,
    "adj_def_epa": pl.Float64,
    "adj_net_epa": pl.Float64,
    "off_weight": pl.Float64,
    "def_weight": pl.Float64,
    "raw_contribution": pl.Float64,
    "schedule_contribution": pl.Float64,
    "contribution": pl.Float64,
    "net_adj_epa": pl.Float64,
    "loo_net_adj_epa": pl.Float64,
    "game_influence": pl.Float64,
}

#: the largest |sum of contributions - cfb_adjusted_epa net_adj_epa| a season may publish with
SUM_TOLERANCE = 1e-9

_INT_IDS = ("game_id", "team_id", "opponent_id")


def ledger_table(plays: pl.DataFrame, season: int) -> pl.DataFrame:
    """The published ledger for one season's summaries input frame (``add_derived_metrics`` output).

    Raises:
        ValueError: a team's contributions miss its ``cfb_adjusted_epa`` value by more than
            :data:`SUM_TOLERANCE`, or the ledger rates a different set of teams.
    """
    ledger = cfb_adjusted_epa_ledger(plays)
    season_fit = cfb_adjusted_epa(plays).select("team_id", expected="net_adj_epa")
    sums = ledger.group_by("team_id").agg(total=pl.col("contribution").sum())
    assert sums.schema["team_id"] == season_fit.schema["team_id"]
    joined = season_fit.join(sums, on="team_id", how="full", coalesce=True)
    if joined["total"].null_count() or joined["expected"].null_count():
        raise ValueError(
            f"adjustment_ledger {season}: the ledger and cfb_adjusted_epa rate different teams"
        )
    worst = (joined["total"] - joined["expected"]).abs().max()
    if worst is None or worst > SUM_TOLERANCE:
        raise ValueError(
            f"adjustment_ledger {season}: contributions miss net_adj_epa by {worst}"
        )
    out = ledger.with_columns(
        *(pl.col(c).cast(pl.Int64, strict=True) for c in _INT_IDS),
        season=pl.lit(season, dtype=pl.Int64),
    )
    missing = [c for c in OUTPUT_SCHEMA if c not in out.columns]
    if missing:  # a renamed sdv-py column must fail the season, not publish as nulls
        raise ValueError(f"adjustment_ledger {season}: the ledger lacks {missing}")
    return out.select(pl.col(c).cast(t) for c, t in OUTPUT_SCHEMA.items())


def build_adjustment_ledger(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """``cfb_team_adjustment_ledger`` for one season, from the same release inputs as ``team_summaries``.

    ``base`` is accepted for the derived-builder signature; the inputs are the released pbp
    and schedule, as in ``build_summaries_season``. Returns the empty :data:`OUTPUT_SCHEMA`
    frame for a season with no plays yet.
    """
    del base  # same inputs as team_summaries: the releases, not a local tree
    from sportsdataverse.cfb import load_cfb_pbp, load_cfb_schedule

    pbp = load_cfb_pbp(seasons=[season])
    if pbp is None or pbp.height == 0:
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    schedule = load_cfb_schedule(seasons=[season])
    plays = add_derived_metrics(prepare_plays_input(pbp, schedule, season))
    return ledger_table(plays, season)
