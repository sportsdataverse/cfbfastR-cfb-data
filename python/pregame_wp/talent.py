"""Roster talent + returning production helpers.

Port of win-prob.ipynb cells 47–55.
"""
from __future__ import annotations

import polars as pl

from .constants import TALENT_FCS_PERCENTILE


def calculate_roster_talent(
    recruiting_df: pl.DataFrame,
    year: int,
    window: int = 4,
) -> pl.DataFrame:
    """Rolling 4-year mean recruiting composite per team, with FCS floor.

    Args:
        recruiting_df: DataFrame with columns ['team', 'year', 'rating'].
        year: Target year (inclusive upper bound).
        window: Number of prior years to average (default 4).

    Returns:
        DataFrame with columns ['team', 'talent'].
    """
    sub = recruiting_df.filter(pl.col("year") <= year).sort("year")
    # Keep the most recent `window` years per team, then mean the rating.
    talent = (
        sub.group_by("team", maintain_order=True)
        .agg(pl.col("rating").tail(window).mean().alias("talent"))
        .sort("team")
    )
    # FCS floor: clamp to 2nd percentile of the FBS distribution
    # pandas Series.quantile defaults to linear interpolation; match it
    floor = talent["talent"].quantile(TALENT_FCS_PERCENTILE, interpolation="linear")
    talent = talent.with_columns(pl.col("talent").clip(lower_bound=floor))
    return talent


def calculate_returning_production(
    returning_df: pl.DataFrame,
) -> pl.DataFrame:
    """Snap-share-weighted returning production per team.

    Args:
        returning_df: DataFrame with columns ['team', 'returning', 'snap_share'].

    Returns:
        DataFrame with columns ['team', 'returning_production'].
    """
    result = (
        returning_df.group_by("team", maintain_order=True)
        .agg(
            (
                (pl.col("returning") * pl.col("snap_share")).sum()
                / pl.col("snap_share").sum()
            ).alias("returning_production")
        )
        .sort("team")
    )
    return result
