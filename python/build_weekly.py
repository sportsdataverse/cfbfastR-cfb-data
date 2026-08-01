"""P8: weekly LONG-FORMAT cumulative snapshots.

Owner decision (plan P8): publish weekly snapshots as ONE asset per season
carrying a ``through_week`` column with every week's cumulative state stacked --
not one asset per week. Asset count stays 22/dataset; a consumer filters
``through_week == W``.

Shipped for the opponent-adjusted TEAM datasets, where the cumulative state is
NOT derivable from the per-game gamelog (the ridge is refit on everything up to
week W, so you cannot reconstruct it by summing games):
  cfb_ratings_weekly           <- cfb_ratings(as_of_date=<end of week W>)
  cfb_team_summaries_weekly    <- build_summaries_season(through_week=W)

Player tables and percentiles are deliberately NOT shipped weekly: those are
derivable from espn_cfb_adv_team_gamelog, so a weekly asset would be redundant
bulk.

WEEK -> DATE. cfb_ratings takes ``as_of_date``, not a week, so each week's
cutoff is the LAST kickoff of that week from the schedule master. Regular season
only (``season_type == 2``): postseason restarts week numbering at 1, so
including it would collide week labels (see the gamelog notes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from cfb_data_build.io import write_dataset

SCHEDULE = Path(
    r"C:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfbfastR-cfb-raw\cfb\cfb_schedule_master.parquet"
)

SPECS = {
    "ratings": ("cfb_ratings_weekly", "cfb_ratings_weekly", "cfb_ratings_weekly"),
    "team_summaries": (
        "cfb_team_summaries_weekly",
        "cfb_team_summaries_weekly",
        "cfb_team_summaries_weekly",
    ),
}


def week_cutoffs(season: int, schedule_path: Path = SCHEDULE) -> list[tuple[int, str]]:
    """(week, last-kickoff-date) for the season's REGULAR-season weeks, ascending."""
    s = pl.read_parquet(schedule_path).filter(
        (pl.col("season") == season) & (pl.col("season_type") == 2)
    )
    if s.height == 0 or "start_date" not in s.columns:
        return []
    g = (
        s.select(["week", "start_date"])
        .drop_nulls()
        .group_by("week")
        .agg(pl.col("start_date").max().alias("cutoff"))
        .sort("week")
    )
    return [
        (int(w), str(c)[:10])
        for w, c in zip(g["week"].to_list(), g["cutoff"].to_list())
    ]


def build_ratings_weekly(season: int) -> pl.DataFrame:
    import datetime as dt

    from sportsdataverse.cfb import cfb_ratings

    frames = []
    for week, cutoff in week_cutoffs(season):
        try:
            d = cfb_ratings(season, as_of_date=dt.date.fromisoformat(cutoff))
        except Exception as exc:  # noqa: BLE001 - one week must not kill the season
            print(f"    week {week}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        if d.height == 0:
            continue
        frames.append(d.with_columns(through_week=pl.lit(week, dtype=pl.Int32)))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def build_summaries_weekly(season: int, base: str) -> pl.DataFrame:
    from cfb_data_build.config import SUMMARIES_REGISTRY
    from cfb_data_build.summaries_build import build_summaries_season

    frames = []
    for week, _cutoff in week_cutoffs(season):
        try:
            build_summaries_season(season, through_week=week, base=base, publish=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    week {week}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        # Read dataset/stem from the registry -- hardcoding them silently
        # produced an empty frame: the stem is "cfb_team_summaries", not
        # "team_summaries", so the path never existed and every week was skipped.
        spec = SUMMARIES_REGISTRY["team_summaries"]
        snap = (
            Path(base)
            / "snapshots"
            / f"through_wk{week:02d}"
            / spec.dataset
            / "parquet"
            / f"{spec.stem}_{season}.parquet"
        )
        if not snap.exists():
            continue
        frames.append(
            pl.read_parquet(snap).with_columns(
                through_week=pl.lit(week, dtype=pl.Int32)
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SPECS))
    ap.add_argument("-s", "--start-year", type=int, default=2004)
    ap.add_argument("-e", "--end-year", type=int, default=2025)
    ap.add_argument("--base", default="cfb")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset, stem, tag = SPECS[args.dataset]
    failures: list[tuple[int, str]] = []
    for season in range(args.start_year, args.end_year + 1):
        try:
            df = (
                build_ratings_weekly(season)
                if args.dataset == "ratings"
                else build_summaries_weekly(season, args.base)
            )
            if df.height == 0:
                print(f"  {dataset} {season}: 0 rows, skipped", flush=True)
                continue
            weeks = df["through_week"].n_unique()
            print(
                f"  {dataset} {season}: {df.height} rows, {df.width} cols, {weeks} weekly snapshots",
                flush=True,
            )
            if args.dry_run:
                continue
            write_dataset(df, dataset, season, stem, base=args.base)
            if args.publish:
                from cfb_data_build.config import DatasetSpec
                from cfb_data_build.publish import publish_dataset

                publish_dataset(DatasetSpec(dataset, stem, tag), season, base=args.base)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True
            )
            failures.append((season, type(exc).__name__))

    print(f"\n=== {dataset}: {len(failures)} failures ===")
    for season, kind in failures:
        print(f"  {season}: {kind}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
