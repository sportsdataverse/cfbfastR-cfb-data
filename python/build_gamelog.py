"""P8: espn_cfb_adv_team_gamelog -- one row per team-GAME, backtest-ready.

WHY THIS IS NOT A DUPLICATE OF adv_team. adv_team already has the right grain
(2 rows per game) and the 75 advanced box metrics, but carries NO game context:
no opponent, no home/away, no scores, no result, no date -- and its `pos_team`
column holds a team ID, not a name (a pre-existing leak: the published 2014
asset shows '2545','164' too, so this predates the P2 republish).

Without opponent/home/scores/date you cannot do the things this program is FOR:
opponent-adjustment, strength-of-schedule, home/away splits, win/margin
outcomes, or rolling in-season trends. This adds exactly that context and gives
the identity columns honest names.

JOIN DISCIPLINE. The join is by ID, never by name -- a name-namespace mismatch
already burned this project (NameAlt matched 80%, homeTeamName 100%). game_id
is Int64 in adv_team and Int32 in the schedule master, so both sides are cast
explicitly before joining, per the repo's ID-dtype rule.
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
DATASET = "adv_team_gamelog"
STEM = "adv_team_gamelog"
TAG = "espn_cfb_adv_team_gamelog"


def build_season(
    season: int, *, base: str = "cfb", schedule_path: Path = SCHEDULE
) -> pl.DataFrame:
    adv_path = Path(base) / "adv_team" / "parquet" / f"adv_team_{season}.parquet"
    if not adv_path.exists():
        print(f"  gamelog {season}: no adv_team parquet at {adv_path}")
        return pl.DataFrame()

    adv = pl.read_parquet(adv_path)
    if adv.height == 0:
        return pl.DataFrame()

    sched = pl.read_parquet(schedule_path).filter(pl.col("season") == season)

    ctx_cols = [
        "game_id",
        "home_id",
        "away_id",
        "home_score",
        "away_score",
        "start_date",
        "neutral_site",
        "season_type",
        "week",
        "home_display_name",
        "away_display_name",
    ]
    ctx = sched.select([c for c in ctx_cols if c in sched.columns]).unique(
        subset=["game_id"]
    )

    # Cast every join/id key explicitly -- adv is Int64, schedule Int32.
    adv = adv.with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col("pos_team").cast(pl.Int64).alias("team_id"),
    ).drop("pos_team")
    ctx = ctx.with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col("home_id").cast(pl.Int64),
        pl.col("away_id").cast(pl.Int64),
    )
    # adv already carries `week`; keep the adv one and drop the schedule copy.
    if "week" in ctx.columns and "week" in adv.columns:
        ctx = ctx.drop("week")

    out = adv.join(ctx, on="game_id", how="left")

    is_home = pl.col("team_id") == pl.col("home_id")
    out = out.with_columns(
        is_home=is_home,
        opponent_id=pl.when(is_home)
        .then(pl.col("away_id"))
        .otherwise(pl.col("home_id")),
        team=pl.when(is_home)
        .then(pl.col("home_display_name"))
        .otherwise(pl.col("away_display_name")),
        opponent=pl.when(is_home)
        .then(pl.col("away_display_name"))
        .otherwise(pl.col("home_display_name")),
        points_for=pl.when(is_home)
        .then(pl.col("home_score"))
        .otherwise(pl.col("away_score"))
        .cast(pl.Int64, strict=False),
        points_against=pl.when(is_home)
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .cast(pl.Int64, strict=False),
    ).with_columns(
        margin=pl.col("points_for") - pl.col("points_against"),
        win=pl.when(pl.col("points_for") > pl.col("points_against"))
        .then(True)
        .when(pl.col("points_for") < pl.col("points_against"))
        .then(False)
        .otherwise(None),
    )

    # Identity first, then context, then the metrics -- readable for a consumer.
    lead = [
        c
        for c in (
            "season",
            "week",
            "season_type",
            "game_id",
            "start_date",
            "team_id",
            "team",
            "opponent_id",
            "opponent",
            "is_home",
            "neutral_site",
            "points_for",
            "points_against",
            "margin",
            "win",
        )
        if c in out.columns
    ]
    rest = [
        c
        for c in out.columns
        if c not in lead
        and c
        not in (
            "home_id",
            "away_id",
            "home_score",
            "away_score",
            "home_display_name",
            "away_display_name",
        )
    ]
    return out.select(lead + rest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start-year", type=int, default=2004)
    ap.add_argument("-e", "--end-year", type=int, default=2025)
    ap.add_argument("--base", default="cfb")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    failures: list[tuple[int, str]] = []
    for season in range(args.start_year, args.end_year + 1):
        try:
            df = build_season(season, base=args.base)
            if df.height == 0:
                print(f"  gamelog {season}: 0 rows, skipped")
                continue
            unmatched = df.filter(pl.col("opponent_id").is_null()).height
            print(
                f"  gamelog {season}: {df.height} team-games, {df.width} cols, "
                f"{df['team_id'].n_unique()} teams, unmatched_context={unmatched}",
                flush=True,
            )
            if args.dry_run:
                continue
            write_dataset(df, DATASET, season, STEM, base=args.base)
            if args.publish:
                from cfb_data_build.config import DatasetSpec
                from cfb_data_build.publish import publish_dataset

                publish_dataset(DatasetSpec(DATASET, STEM, TAG), season, base=args.base)
        except Exception as exc:  # noqa: BLE001 - one season must not abort the sweep
            print(
                f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True
            )
            failures.append((season, type(exc).__name__))

    print(f"\n=== gamelog: {len(failures)} failures ===")
    for season, kind in failures:
        print(f"  {season}: {kind}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
