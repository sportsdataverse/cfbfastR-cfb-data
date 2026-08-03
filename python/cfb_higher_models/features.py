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


def blend_prior(weekly: pl.DataFrame, cols=CARRY_COLS, *, k: float = 4.0) -> pl.DataFrame:
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
    n = pl.col("valid_games") if "valid_games" in weekly.columns else pl.col("through_week")
    w = n.cast(pl.Float64) / (n.cast(pl.Float64) + k)
    return weekly.with_columns(
        [
            (w * pl.col(c).fill_null(0.0) + (1 - w) * pl.col(f"prior_{c}").fill_null(pl.col(c)).fill_null(0.0)).alias(
                f"blend_{c}"
            )
            for c in have
        ]
    )


def enrich_weekly(weekly: pl.DataFrame, *, k: float = 4.0) -> pl.DataFrame:
    """The full Phase-2 team-week enrichment, in dependency order."""
    return blend_prior(add_prior_season(weekly), k=k)


# --------------------------------------------------------------------------
# game-level: schedule spot
# --------------------------------------------------------------------------
def add_rest(games: pl.DataFrame) -> pl.DataFrame:
    """Days of rest for each team, and the differential.

    Genuinely OFF-substrate: rest, byes and short weeks are nowhere in the
    play-level data, so unlike another efficiency metric this is not a
    re-measurement of team quality. It is also strictly as-of -- it depends
    only on when previous games were played, never on their outcome.

    Adds ``rest_home`` / ``rest_away`` / ``rest_diff``, plus ``bye_home`` /
    ``bye_away`` (>= 13 days, i.e. an actual open date rather than a normal
    Saturday-to-Saturday turnaround).
    """
    date_col = next(
        (c for c in ("start_date", "date_time", "game_date") if c in games.columns),
        None,
    )
    if date_col is None:
        # This used to `return games` unchanged -- a silent no-op, the exact
        # failure this codebase keeps shipping. It cost a real experiment:
        # build_game_frame's fixed column select dropped the date, add_rest
        # returned the frame untouched, and the A/B reported base and
        # "base + rest" byte-identical at 13.026. The honest reading is "rest
        # was never tested"; the tempting one is "rest doesn't help". Raise, so
        # a caller cannot mistake absence of a feature for evidence about it.
        raise ValueError(
            "add_rest: no date column (looked for start_date / date_time / "
            f"game_date); got {list(games.columns)[:12]}. Returning the frame "
            "unchanged would score identically to the baseline and read as a "
            "negative result."
        )
    g = games.with_columns(pl.col(date_col).cast(pl.Utf8).str.slice(0, 10).str.to_date(strict=False).alias("_d"))
    # Long form (one row per team-game) so "previous game" is a single sort.
    long = pl.concat(
        [
            g.select("game_id", "season", pl.col("home_id").alias("tid"), "_d"),
            g.select("game_id", "season", pl.col("away_id").alias("tid"), "_d"),
        ]
    )
    long = long.sort(["tid", "season", "_d"]).with_columns(
        (pl.col("_d") - pl.col("_d").shift(1).over(["tid", "season"])).dt.total_days().alias("rest")
    )
    out = g
    for side in ("home", "away"):
        s = long.select(
            "game_id",
            pl.col("tid").alias(f"{side}_id"),
            pl.col("rest").alias(f"rest_{side}"),
        )
        out = out.join(s, on=["game_id", f"{side}_id"], how="left")
    return out.with_columns(
        (pl.col("rest_home") - pl.col("rest_away")).alias("rest_diff"),
        (pl.col("rest_home") >= 13).cast(pl.Float64).alias("bye_home"),
        (pl.col("rest_away") >= 13).cast(pl.Float64).alias("bye_away"),
    ).drop("_d")
