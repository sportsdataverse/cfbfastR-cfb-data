"""Phase 2 feature engineering on the team-week spine.

The 384-column weekly substrate is season-to-date: in week 3 a team's rating
rests on two games. Phase 1 showed how costly that is -- the optimal rating
multiplier at 0-3 games played is 10.8 versus 55.4 once eight games are in, so
early-season ratings are mostly noise and the model is right to nearly ignore
them.

Ignoring them is not the only option. Two standard repairs, both of which use
information the substrate already has:

1. PRIOR-SEASON CARRYOVER. A team's rating at the end of last season is a far
   better week-3 estimate than two games of this season. Blending the two with
   a shrinkage weight ``n/(n+k)`` is what SP+ and FPI do, and it is the
   principled answer to the attenuation problem rather than a workaround.

2. RECENCY WEIGHTING. A season-to-date mean weights week 1 and week 10 equally
   even though teams change. An exponentially-weighted view tracks the current
   team.

Both are applied at the TEAM-WEEK level, before the home/away join, so they
inherit the as-of boundary from :func:`data.build_game_frame` rather than
introducing a second place where leakage could enter.
"""

from __future__ import annotations

import polars as pl

#: Rating columns worth carrying across a season boundary. Volume stats
#: (plays, yards) do not carry -- roster turnover makes last year's snap count
#: uninformative -- but efficiency and opponent-adjusted strength do.
CARRY_COLS = (
    "rt_adj_net",
    "rt_adj_off_epa",
    "rt_adj_def_epa",
    "net_adj_epa",
    "adj_off_epa",
    "adj_def_epa",
    "EPAplay_off",
    "EPAplay_def",
    "success_off",
    "success_def",
    "explosive_off",
    "explosive_def",
    "havoc_off",
    "havoc_def",
)


def add_prior_season(weekly: pl.DataFrame, cols=CARRY_COLS) -> pl.DataFrame:
    """Attach each team's FINAL rating from the previous season as ``prior_*``.

    "Final" = the largest ``through_week`` present for that (team, season).
    Teams new to FBS have no prior row and get null, which the tree models
    handle and the linear ones see as 0 after ``nan_to_num``.
    """
    have = [c for c in cols if c in weekly.columns]
    if not have:
        return weekly
    final = (
        weekly.sort("through_week")
        .group_by(["team_id", "season"])
        .last()
        .select(
            pl.col("team_id"),
            (pl.col("season") + 1).alias("season"),  # shift forward one year
            *[pl.col(c).alias(f"prior_{c}") for c in have],
        )
    )
    return weekly.join(final, on=["team_id", "season"], how="left")


def blend_prior(
    weekly: pl.DataFrame, cols=CARRY_COLS, *, k: float = 4.0
) -> pl.DataFrame:
    """Shrinkage blend of this season's as-of rating toward last season's final.

    ``blend = (n/(n+k)) * current + (k/(n+k)) * prior`` where ``n`` is games
    played. ``k`` is "how many games of this season it takes before the current
    number outweighs last year's" -- k=4 means a team is judged half on prior
    form until about a month in. Emitted as ``blend_*`` alongside the raw
    columns so a model can use either; k is a knob to fit, not a truth.
    """
    have = [c for c in cols if c in weekly.columns and f"prior_{c}" in weekly.columns]
    if not have:
        return weekly
    n = (
        pl.col("valid_games")
        if "valid_games" in weekly.columns
        else pl.col("through_week")
    )
    w = n.cast(pl.Float64) / (n.cast(pl.Float64) + k)
    return weekly.with_columns(
        [
            (
                w * pl.col(c).fill_null(0.0)
                + (1 - w) * pl.col(f"prior_{c}").fill_null(pl.col(c)).fill_null(0.0)
            ).alias(f"blend_{c}")
            for c in have
        ]
    )


def enrich_weekly(weekly: pl.DataFrame, *, k: float = 4.0) -> pl.DataFrame:
    """The full Phase-2 team-week enrichment, in dependency order."""
    return blend_prior(add_prior_season(weekly), k=k)
