"""Datasets DERIVED from already-built artifacts, not from ``final.json``.

P6 folded these in from standalone scripts so ``--dataset`` is the single
surface for every CFB dataset:

  gamelog                 ``espn_cfb_adv_team_gamelog`` -- adv_team plus the game
                          context it lacks (opponent, home/away, scores, result,
                          date). One row per team-GAME.
  ratings_weekly          ``cfb_ratings_weekly`` -- cfb_ratings at each week's end.
  team_summaries_weekly   ``cfb_team_summaries_weekly`` -- summaries at each
                          week's end.

The two ``*_weekly`` products are LONG FORMAT: one asset per season with a
``through_week`` column stacking every week's cumulative state, so the asset
count stays 22/dataset and a consumer filters ``through_week == W``.

Only the opponent-adjusted TEAM datasets ship weekly. The ridge is refit on
everything through week W, so that state cannot be reconstructed by summing
per-game rows. Player tables and percentiles CAN be derived from the gamelog, so
weekly assets there would be redundant bulk.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cfb_data_build.config import SUMMARIES_REGISTRY, DatasetSpec
from cfb_data_build.io import write_dataset

SCHEDULE = Path(
    r"C:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfbfastR-cfb-raw\cfb\cfb_schedule_master.parquet"
)

SPECS: dict[str, DatasetSpec] = {
    "gamelog": DatasetSpec("adv_team_gamelog", "adv_team_gamelog", "espn_cfb_adv_team_gamelog"),
    "ratings_weekly": DatasetSpec("cfb_ratings_weekly", "cfb_ratings_weekly", "cfb_ratings_weekly"),
    "team_summaries_weekly": DatasetSpec(
        "cfb_team_summaries_weekly", "cfb_team_summaries_weekly", "cfb_team_summaries_weekly"
    ),
}


def week_cutoffs(season: int, schedule_path: Path = SCHEDULE) -> list[tuple[int, str]]:
    """(week, last-kickoff-date) for the season's REGULAR-season weeks, ascending.

    Regular season only: the postseason restarts week numbering at 1 (ESPN's own
    convention -- the schedule master agrees), so including it would collide the
    week labels.
    """
    s = pl.read_parquet(schedule_path).filter((pl.col("season") == season) & (pl.col("season_type") == 2))
    if s.height == 0 or "start_date" not in s.columns:
        return []
    g = (
        s.select(["week", "start_date"])
        .drop_nulls()
        .group_by("week")
        .agg(pl.col("start_date").max().alias("cutoff"))
        .sort("week")
    )
    return [(int(w), str(c)[:10]) for w, c in zip(g["week"].to_list(), g["cutoff"].to_list())]


def build_gamelog(season: int, *, base: str = "cfb", schedule_path: Path = SCHEDULE) -> pl.DataFrame:
    """adv_team + game context, one row per team-GAME.

    Joined by ID, never by name: a name-namespace mismatch already cost this
    project once (NameAlt 80% vs homeTeamName 100%). ``game_id`` is Int64 in
    adv_team and Int32 in the schedule master, so both sides are cast.

    ``adv_team.pos_team`` holds a team ID under a name-shaped column (a
    pre-existing leak -- the published 2014 asset shows '2545','164' too), so it
    is surfaced here as ``team_id`` and the readable name is joined in.
    """
    adv_path = Path(base) / "adv_team" / "parquet" / f"adv_team_{season}.parquet"
    if not adv_path.exists():
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
    ctx = sched.select([c for c in ctx_cols if c in sched.columns]).unique(subset=["game_id"])

    adv = adv.with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col("pos_team").cast(pl.Int64).alias("team_id"),
    ).drop("pos_team")
    ctx = ctx.with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col("home_id").cast(pl.Int64),
        pl.col("away_id").cast(pl.Int64),
    )
    if "week" in ctx.columns and "week" in adv.columns:
        ctx = ctx.drop("week")

    out = adv.join(ctx, on="game_id", how="left")
    is_home = pl.col("team_id") == pl.col("home_id")
    out = out.with_columns(
        is_home=is_home,
        opponent_id=pl.when(is_home).then(pl.col("away_id")).otherwise(pl.col("home_id")),
        team=pl.when(is_home).then(pl.col("home_display_name")).otherwise(pl.col("away_display_name")),
        opponent=pl.when(is_home).then(pl.col("away_display_name")).otherwise(pl.col("home_display_name")),
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
    drop = {"home_id", "away_id", "home_score", "away_score", "home_display_name", "away_display_name"}
    rest = [c for c in out.columns if c not in lead and c not in drop]
    return out.select(lead + rest)


def build_ratings_weekly(season: int, *, base: str = "cfb") -> pl.DataFrame:
    import datetime as dt

    from sportsdataverse.cfb import cfb_ratings

    frames = []
    for week, cutoff in week_cutoffs(season):
        try:
            d = cfb_ratings(season, as_of_date=dt.date.fromisoformat(cutoff))
        except Exception as exc:  # noqa: BLE001 - one week must not kill the season
            print(f"    week {week}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        if d.height:
            frames.append(d.with_columns(through_week=pl.lit(week, dtype=pl.Int32)))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def build_team_summaries_weekly(season: int, *, base: str = "cfb") -> pl.DataFrame:
    from cfb_data_build.summaries_build import build_summaries_season

    spec = SUMMARIES_REGISTRY["team_summaries"]
    frames = []
    for week, _cutoff in week_cutoffs(season):
        try:
            build_summaries_season(season, through_week=week, base=base, publish=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    week {week}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        # Path comes from the registry. Hardcoding it silently produced an EMPTY
        # frame -- the stem is "cfb_team_summaries", not "team_summaries", so the
        # path never existed, all weeks were skipped, and the build still
        # reported success.
        snap = (
            Path(base)
            / "snapshots"
            / f"through_wk{week:02d}"
            / spec.dataset
            / "parquet"
            / f"{spec.stem}_{season}.parquet"
        )
        if snap.exists():
            frames.append(pl.read_parquet(snap).with_columns(through_week=pl.lit(week, dtype=pl.Int32)))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


BUILDERS = {
    "gamelog": build_gamelog,
    "ratings_weekly": build_ratings_weekly,
    "team_summaries_weekly": build_team_summaries_weekly,
}


def build_derived(
    dataset: str,
    start_year: int,
    end_year: int,
    *,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Build (and optionally publish) a derived dataset across a season range.

    Every season is isolated -- a failure is recorded and the sweep continues.
    Returns the list of ``(season, error_type)`` failures.
    """
    spec = SPECS[dataset]
    build = BUILDERS[dataset]
    failures: list[tuple[int, str]] = []
    for season in range(start_year, end_year + 1):
        try:
            df = build(season, base=base)
            if df.height == 0:
                print(f"  {spec.dataset} {season}: 0 rows, skipped", flush=True)
                continue
            extra = f", {df['through_week'].n_unique()} weekly snapshots" if "through_week" in df.columns else ""
            print(f"  {spec.dataset} {season}: {df.height} rows, {df.width} cols{extra}", flush=True)
            if dry_run:
                continue
            write_dataset(df, spec.dataset, season, spec.stem, base=base)
            if publish:
                from cfb_data_build.publish import publish_dataset

                publish_dataset(spec, season, base=base)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            failures.append((season, type(exc).__name__))
    return failures
