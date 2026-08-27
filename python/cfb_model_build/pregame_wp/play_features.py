"""Per-play derived columns: EqPPP, play_successful, play_explosive.

Faithful port of win-prob.ipynb cells 20 and 22.

Note on play_successful (OQ-2): 3rd-down plays default to False regardless of
yards gained. Conversions appear as 1st-down plays in the subsequent sequence.
This matches the notebook's np.select conditions exactly.
"""
from __future__ import annotations

import polars as pl

from .constants import EXPLOSIVE_THRESHOLD, SR_DOWN1, SR_DOWN2, SR_DOWN4


def add_play_features(
    df: pl.DataFrame,
    ep_data: list[float],
    st_types: list[str],
    bad_types: list[str],
) -> pl.DataFrame:
    """Add play_explosive, play_successful, and (optionally) EqPPP columns."""
    is_bad = pl.col("play_type").is_in(bad_types)
    is_st = pl.col("play_type").is_in(st_types)

    # --- play_explosive (first-match-wins, mirrors np.select order) ---
    play_explosive = (
        pl.when(is_bad).then(False)
        .when(is_st).then(False)
        .when(pl.col("yards_gained") >= EXPLOSIVE_THRESHOLD).then(True)
        .otherwise(False)
        .alias("play_explosive")
    )

    # --- play_successful (3rd down intentionally absent → default False) ---
    play_successful = (
        pl.when(is_bad).then(False)
        .when(is_st).then(False)
        .when((pl.col("down") == 1) & (pl.col("yards_gained") >= SR_DOWN1 * pl.col("distance")))
        .then(True)
        .when((pl.col("down") == 2) & (pl.col("yards_gained") >= SR_DOWN2 * pl.col("distance")))
        .then(True)
        .when((pl.col("down") >= 4) & (pl.col("yards_gained") >= SR_DOWN4 * pl.col("distance")))
        .then(True)
        .otherwise(False)
        .alias("play_successful")
    )

    df = df.with_columns([play_explosive, play_successful])

    # --- EqPPP (zero for ST plays; skipped when ep_data is empty) ---
    if ep_data:
        last = len(ep_data) - 1  # EP curve is indexed by clamped yardline [0, 100]
        ep_list = pl.lit(pl.Series(ep_data, dtype=pl.Float64).implode())
        src_idx = pl.col("yard_line").clip(0, last).cast(pl.Int64)
        dst_idx = (pl.col("yard_line") + pl.col("yards_gained")).clip(0, last).cast(pl.Int64)
        eqppp_expr = ep_list.list.get(dst_idx) - ep_list.list.get(src_idx)
        df = df.with_columns(
            pl.when(is_st)
            .then(0.0)
            .otherwise(eqppp_expr)
            .alias("EqPPP")
        )

    return df
