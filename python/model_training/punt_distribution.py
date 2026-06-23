"""Build the CFB punt end-yardline distribution — the nfl4th ``punt_df`` analog.

This is the one nfl4th model artifact whose CFB build was missing from the training
repo: ``sportsdataverse/cfb/models/punt_distribution.parquet`` shipped in sdv-py but
had no reproducible builder here. ``get_punt_wp`` (the punt branch of the 4th-down
decision surface) consumes it.

Schema (matches the shipped consumer exactly)::

    yards_to_goal      Int64   punting team's pre-snap distance to the opponent end zone (31..99)
    yards_to_goal_end  Int64   post-punt spot from the punting team's frame; the receiving
                               team then starts at ``100 - yards_to_goal_end``. 1 = pinned at
                               the goal line, 100 = punt-return touchdown.
    pct                Float64 smoothed probability mass, normalized to sum to 1 per yards_to_goal.

Construction (faithful to nfl4th ``data-raw/_punt_and_fg_models.R``):

* ``net = yds_punted - yds_punt_return`` (kick distance minus return yards).
* ``yards_to_goal_end = clip(yards_to_goal - net, 1, 100)``; touchbacks are placed at
  the receiving team's own 25 (``yards_to_goal_end = 25``).
* The raw (yards_to_goal, yards_to_goal_end) cloud is smoothed with a 2-D Gaussian
  KDE (``scipy.stats.gaussian_kde``, the SciPy analog of R ``MASS::kde2d``) evaluated
  AT each observed cell — this is nfl4th's ``get_density`` trick, which both smooths
  rare block/return-TD tails and preserves the observed support (so the table is
  gappy in ``yards_to_goal_end`` exactly like the shipped artifact). Mass is then
  renormalized within each ``yards_to_goal``.

Run::

    python -m model_training.punt_distribution --pbp artifacts/pbp_full.parquet \
        --out artifacts/punt_distribution.parquet \
        --validate ../../sdv-py/sportsdataverse/cfb/models/punt_distribution.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

# Punts realistically only happen from ~midfield back; closer in teams kick or go.
PUNT_YTG_MIN: int = 31
PUNT_YTG_MAX: int = 99
TOUCHBACK_YTG_END: int = 25  # receiving team starts at its own 25 (modern CFB rule)


def _punt_outcomes(pbp: pl.DataFrame) -> pl.DataFrame:
    """Filter to punts and derive (yards_to_goal, yards_to_goal_end) per play.

    Returns a 2-column frame; rows with missing inputs or out-of-range starting
    field position are dropped.
    """
    ytg = pl.col("start.yardsToEndzone")
    net = pl.col("yds_punted").cast(pl.Float64) - pl.col("yds_punt_return").fill_null(0).cast(pl.Float64)
    tb = pl.col("punt_tb").fill_null(False).cast(pl.Boolean) if "punt_tb" in pbp.columns else pl.lit(False)
    return (
        pbp.filter(
            (pl.col("punt").cast(pl.Boolean) == True)  # noqa: E712
            & pl.col("yds_punted").is_not_null()
            & ytg.is_between(PUNT_YTG_MIN, PUNT_YTG_MAX)
        )
        .with_columns(
            yards_to_goal=ytg.cast(pl.Int64),
            yards_to_goal_end=pl.when(tb)
            .then(pl.lit(TOUCHBACK_YTG_END))
            .otherwise((ytg - net).round().clip(1, 100))
            .cast(pl.Int64),
        )
        .select("yards_to_goal", "yards_to_goal_end")
        .drop_nulls()
    )


def build_punt_distribution(pbp: pl.DataFrame, *, bw_method: float | str | None = 0.25) -> pl.DataFrame:
    """Build the smoothed punt end-yardline distribution from a play-by-play frame.

    Args:
        pbp: ESPN final.json play frame (``pbp_full.parquet``) carrying ``punt``,
            ``yds_punted``, ``yds_punt_return``, ``start.yardsToEndzone`` (+ optional
            ``punt_tb``).
        bw_method: bandwidth passed to ``gaussian_kde`` (smaller = less smoothing).
            ``0.25`` tracks the granularity of the shipped artifact.

    Returns:
        polars DataFrame with columns ``yards_to_goal``, ``yards_to_goal_end``,
        ``pct`` (Float64, summing to 1.0 within each ``yards_to_goal``), sorted.

    Raises:
        ValueError: if no punt rows survive the filter.
    """
    from scipy.stats import gaussian_kde

    out = _punt_outcomes(pbp)
    if out.height == 0:
        raise ValueError("no punt rows survived the filter — check punt/yds_punted columns")

    xy = out.select("yards_to_goal", "yards_to_goal_end").to_numpy().astype(float).T  # (2, n)
    kde = gaussian_kde(xy, bw_method=bw_method)

    # Evaluate the smoothed density AT each unique observed cell (nfl4th get_density),
    # then renormalize within each starting yards_to_goal so pct sums to 1 per Y.
    cells = (
        out.group_by("yards_to_goal", "yards_to_goal_end")
        .agg(n=pl.len())
        .sort("yards_to_goal", "yards_to_goal_end")
    )
    pts = cells.select("yards_to_goal", "yards_to_goal_end").to_numpy().astype(float).T
    dens = kde(pts)
    cells = cells.with_columns(pl.Series("dens", dens))
    return (
        cells.with_columns(
            pct=(pl.col("dens") / pl.col("dens").sum().over("yards_to_goal")).cast(pl.Float64)
        )
        .select("yards_to_goal", "yards_to_goal_end", "pct")
        .sort("yards_to_goal", "yards_to_goal_end")
    )


def validate_against(built: pl.DataFrame, shipped_path: str | Path) -> dict:
    """Compare a freshly built distribution to the shipped artifact.

    Returns a dict of agreement diagnostics: per-``yards_to_goal`` mean end-yardline
    correlation, pooled mass overlap, and row/coverage counts. Never raises on a
    schema/content mismatch — it reports so the caller can judge reproducibility.
    """
    shipped = pl.read_parquet(shipped_path)

    def _mean_end(df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by("yards_to_goal").agg(
            mean_end=(pl.col("yards_to_goal_end") * pl.col("pct")).sum()
        )

    j = _mean_end(built).join(_mean_end(shipped), on="yards_to_goal", suffix="_shipped").drop_nulls()
    corr = float(np.corrcoef(j["mean_end"].to_numpy(), j["mean_end_shipped"].to_numpy())[0, 1]) if j.height > 1 else float("nan")
    mae = float((j["mean_end"] - j["mean_end_shipped"]).abs().mean()) if j.height else float("nan")
    return {
        "mean_end_corr": round(corr, 4),
        "mean_end_mae_yards": round(mae, 3),
        "built_rows": built.height,
        "shipped_rows": shipped.height,
        "ytg_overlap": j.height,
        "built_ytg_range": [int(built["yards_to_goal"].min()), int(built["yards_to_goal"].max())],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="model_training.punt_distribution")
    ap.add_argument("--pbp", default="artifacts/pbp_full.parquet")
    ap.add_argument("--out", default="artifacts/punt_distribution.parquet")
    ap.add_argument("--bw", default="0.25", help="gaussian_kde bandwidth (float or 'scott'/'silverman')")
    ap.add_argument("--validate", default=None, help="path to the shipped punt_distribution.parquet to compare")
    args = ap.parse_args(argv)

    bw: float | str = args.bw
    try:
        bw = float(args.bw)
    except ValueError:
        pass

    pbp = pl.read_parquet(args.pbp)
    dist = build_punt_distribution(pbp, bw_method=bw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dist.write_parquet(args.out)
    print(f"punt_distribution: wrote {dist.height} rows -> {args.out} "
          f"(yards_to_goal {dist['yards_to_goal'].min()}..{dist['yards_to_goal'].max()})")
    if args.validate:
        print("validation vs shipped:", validate_against(dist, args.validate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
