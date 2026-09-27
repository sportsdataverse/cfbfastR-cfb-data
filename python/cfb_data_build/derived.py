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

import os
from pathlib import Path

import polars as pl

from cfb_data_build.config import SUMMARIES_REGISTRY, DatasetSpec
from cfb_data_build.io import write_dataset


def _schedule_master() -> Path:
    """Locate cfbfastR-cfb-raw's schedule master.

    This was a hardcoded absolute path to one developer's Windows checkout, so
    `ratings_weekly` and `team_summaries_weekly` raised FileNotFoundError on
    every other machine -- including CI, where they had never once built.

    scripts/daily_cfb_processor.sh already exports CFB_RAW_ROOT (it has to:
    run_py does `cd python`, so a relative path would not resolve). Prefer it,
    and fall back to the sibling checkout beside this repo.
    """
    root = os.environ.get("CFB_RAW_ROOT")
    if root:
        return Path(root) / "cfb" / "cfb_schedule_master.parquet"
    return (
        Path(__file__).resolve().parents[3]
        / "cfbfastR-cfb-raw"
        / "cfb"
        / "cfb_schedule_master.parquet"
    )


#: Import-time snapshot, kept for callers that referenced it. Everything in
#: this module resolves via `_schedule_master()` at CALL time instead -- a
#: default argument would freeze the path at import, so any caller that sets
#: CFB_RAW_ROOT afterwards would silently keep using the old location.
SCHEDULE = _schedule_master()

SPECS: dict[str, DatasetSpec] = {
    "gamelog": DatasetSpec(
        "adv_team_gamelog", "adv_team_gamelog", "espn_cfb_adv_team_gamelog"
    ),
    "ratings_weekly": DatasetSpec(
        "cfb_ratings_weekly", "cfb_ratings_weekly", "cfb_ratings_weekly"
    ),
    "team_summaries_weekly": DatasetSpec(
        "cfb_team_summaries_weekly",
        "cfb_team_summaries_weekly",
        "cfb_team_summaries_weekly",
    ),
    "matchup_features": DatasetSpec(
        "cfb_matchup_features", "cfb_matchup_features", "cfb_matchup_features"
    ),
    "matchup_line": DatasetSpec(
        "cfb_matchup_line", "cfb_matchup_line", "cfb_matchup_line"
    ),
    "rolling_windows": DatasetSpec(
        "rolling_windows", "rolling_windows", "cfb_rolling_windows"
    ),
}


def schedule_master_available(schedule_path: "Path | None" = None) -> bool:
    """Whether cfbfastR-cfb-raw's schedule master is reachable.

    `ratings_weekly` and `team_summaries_weekly` are the only datasets in this
    builder that read the raw store, and CI does not check that repo out -- so
    they raised FileNotFoundError on every scheduled run and turned an
    otherwise-clean preseason build RED. With CFB week 1 days away, a job that
    is permanently red cannot signal a real failure.

    A missing raw store is an ABSENT INPUT, not a defect: the recruiting
    datasets already treat it that way (`daily_cfb_processor.sh` skips them
    with a warning when CFB_RAW_ROOT is unset). This is the same contract for
    the weekly pair.
    """
    path = schedule_path if schedule_path is not None else _schedule_master()
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        # A truncated or half-written download is a file, and `is_file()` alone
        # would wave it through into `pl.read_parquet` -- which raises, which is
        # exactly the hard failure this guard exists to prevent. Read the
        # footer only; this is cheap and does not load the data.
        pl.read_parquet_schema(path)
    except Exception:
        return False
    return True


def schedule_weeks(season: int, schedule_path: "Path | None" = None) -> list[int]:
    """The season's REGULAR-season weeks, ascending, from the raw schedule master.

    The master lists a week before its matchups exist (2026 week 14's
    championship TBDs), so the weekly assets carry the whole calendar.

    Regular season only: the postseason restarts week numbering at 1 (ESPN's own
    convention -- the schedule master agrees), so including it would collide the
    week labels.
    """
    schedule_path = schedule_path if schedule_path is not None else _schedule_master()
    s = pl.read_parquet(schedule_path).filter(
        (pl.col("season") == season) & (pl.col("season_type") == 2)
    )
    if s.height == 0 or "start_date" not in s.columns:
        return []
    return sorted(
        s.select("week", "start_date").drop_nulls()["week"].unique().to_list()
    )


def week_cutoffs(
    season: int,
    schedule_path: "Path | None" = None,
    games: "pl.DataFrame | None" = None,
) -> list[tuple[int, str | None]]:
    """(week, as-of bound) for each of :func:`schedule_weeks`.

    The bound is EXCLUSIVE, like sdv-py's ``cfb_ratings(as_of_date=)``
    (``date < as_of_date``), and comes from ``games`` -- ``load_cfb_schedule``,
    the schedule ``cfb_ratings`` dates plays by and the as-of consumers label
    weeks by -- over the FBS-vs-FBS games ``cfb_ratings`` fits, cancelled and
    postponed ones dropped. Bound = the day after week W's last kickoff, capped
    at the first kickoff of week W+1 and of the FBS postseason. Dates are UTC on
    both sides.

    Why each piece:

    - the last kickoff date itself dropped every game on it (2026 through_week
      4 held 412 of 430 team-games);
    - the master's own labels leaked 2005 Ohio-Miami (OH) -- master week 12,
      week 13 to the consumer -- into through_week 12, and its cancelled Dec 6
      week-15 row cut 14 real 2020 week-14 games;
    - cancelled and postponed rows keep their original date, so one can cap a
      week early (a guard: no FBS-vs-FBS row does today, 2004-2026);
    - the first bowl sits on the final bound's exact date in 2020, 2024 and
      2025; the postseason cap keeps a same-day bowl out.

    A week with no FBS game of its own (2019 week 16, 2026 week 14) reuses the
    previous week's bound. One before any FBS game gets None and is not rated.
    """
    weeks = schedule_weeks(season, schedule_path)
    if not weeks:
        return []
    if games is None:
        from sportsdataverse.cfb import load_cfb_schedule

        games = load_cfb_schedule(seasons=[season])
    fit = (
        games.filter(
            (pl.col("season") == season)
            & (pl.col("home_division") == "fbs")
            & (pl.col("away_division") == "fbs")
            & ~pl.col("status")
            .is_in(["STATUS_CANCELED", "STATUS_POSTPONED"])
            .fill_null(False)
        )
        .select(
            "season_type_id",
            pl.col("week").cast(pl.Int64),
            pl.col("start_date")
            .cast(pl.Utf8)
            .str.slice(0, 10)
            .str.to_date()
            .alias("d"),
        )
        .drop_nulls()
    )
    post_first = fit.filter(pl.col("season_type_id") == 3)["d"].min()
    bounds = (
        fit.filter(pl.col("season_type_id") == 2)
        .group_by("week")
        .agg(pl.col("d").min().alias("first"), pl.col("d").max().alias("last"))
        .sort("week")
        .select(
            "week",
            pl.min_horizontal(
                pl.col("last") + pl.duration(days=1),
                pl.col("first").shift(-1),
                pl.lit(post_first, dtype=pl.Date),
            ).alias("cutoff"),
        )
    )
    g = pl.DataFrame({"week": weeks}, schema={"week": pl.Int64}).join_asof(
        bounds, on="week", strategy="backward"
    )
    return [(int(w), None if c is None else str(c)) for w, c in g.iter_rows()]


def build_gamelog(
    season: int, *, base: str = "cfb", schedule_path: "Path | None" = None
) -> pl.DataFrame:
    """adv_team + game context, one row per team-GAME.

    Joined by ID, never by name: a name-namespace mismatch already cost this
    project once (NameAlt 80% vs homeTeamName 100%). ``game_id`` is Int64 in
    adv_team and Int32 in the schedule master, so both sides are cast.

    ``adv_team`` now ships ``pos_team_id`` (the ESPN team id) alongside a
    ``pos_team`` that holds the readable name. Older assets predate that split
    and still carry the id in ``pos_team``, so the id column is resolved by
    preference and both spellings are accepted.
    """
    adv_path = Path(base) / "adv_team" / "parquet" / f"adv_team_{season}.parquet"
    if not adv_path.exists():
        return pl.DataFrame()
    if not schedule_master_available(schedule_path):
        # Same absent-input contract as the weekly builders. This one is NOT
        # reachable in preseason -- gamelog returns early when no adv_team
        # artifact exists yet -- so it only bites once week 1 produces one,
        # which is exactly when a red job is most expensive.
        print(
            f"  adv_team_gamelog {season}: skipped -- no cfbfastR-cfb-raw "
            f"schedule master at {_schedule_master()} (set CFB_RAW_ROOT)",
            flush=True,
        )
        return pl.DataFrame()
    adv = pl.read_parquet(adv_path)
    if adv.height == 0:
        return pl.DataFrame()

    schedule_path = schedule_path if schedule_path is not None else _schedule_master()
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

    id_col = "pos_team_id" if "pos_team_id" in adv.columns else "pos_team"
    adv = adv.with_columns(
        pl.col("game_id").cast(pl.Int64),
        pl.col(id_col).cast(pl.Int64, strict=False).alias("team_id"),
    ).drop([c for c in ("pos_team", "pos_team_id") if c in adv.columns])
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
    drop = {
        "home_id",
        "away_id",
        "home_score",
        "away_score",
        "home_display_name",
        "away_display_name",
    }
    rest = [c for c in out.columns if c not in lead and c not in drop]
    return out.select(lead + rest)


def build_ratings_weekly(season: int, *, base: str = "cfb") -> pl.DataFrame:
    if not schedule_master_available():
        print(
            f"  ratings_weekly {season}: skipped -- no cfbfastR-cfb-raw schedule "
            f"master at {_schedule_master()} (set CFB_RAW_ROOT)",
            flush=True,
        )
        return pl.DataFrame()
    import datetime as dt

    from sportsdataverse.cfb import cfb_ratings

    frames = []
    built: list[int] = []
    cuts = week_cutoffs(season)
    for week, cutoff in cuts:
        if cutoff is None:
            continue  # no FBS game yet; as_of_date=None would fit the whole season
        d = _retry(
            lambda c=cutoff: cfb_ratings(season, as_of_date=dt.date.fromisoformat(c)),
            what=f"ratings_weekly {season} week {week}",
        )
        if d is not None and d.height:
            frames.append(d.with_columns(through_week=pl.lit(week, dtype=pl.Int32)))
            built.append(week)
    _report_gaps(season, built, [w for w, _ in cuts], "ratings_weekly")
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def build_team_summaries_weekly(season: int, *, base: str = "cfb") -> pl.DataFrame:
    if not schedule_master_available():
        print(
            f"  team_summaries_weekly {season}: skipped -- no cfbfastR-cfb-raw schedule "
            f"master at {_schedule_master()} (set CFB_RAW_ROOT)",
            flush=True,
        )
        return pl.DataFrame()
    from cfb_data_build.summaries_build import build_summaries_season

    spec = SUMMARIES_REGISTRY["team_summaries"]
    frames = []
    built: list[int] = []
    weeks = schedule_weeks(season)
    for week in weeks:
        if (
            _retry(
                lambda w=week: (
                    build_summaries_season(
                        season, through_week=w, base=base, publish=False
                    )
                    or True
                ),
                what=f"team_summaries_weekly {season} week {week}",
            )
            is None
        ):
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
            frames.append(
                pl.read_parquet(snap).with_columns(
                    through_week=pl.lit(week, dtype=pl.Int32)
                )
            )
            built.append(week)
    _report_gaps(season, built, weeks, "team_summaries_weekly")
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


#: Weekly builds fetch per week, so a transient network blip costs a whole
#: snapshot. Observed live on the 2026-08-03 republish: three WinError 10060
#: timeouts (2009 wk5, 2019 wk11, 2020 wk1) each silently dropped that week
#: while the season still reported success. A missing through_week is not a
#: cosmetic gap -- an as-of consumer joining week W+1 to through_week W finds
#: no row and drops those games entirely.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 5.0


def _retry(fn, *, what: str, attempts: int = _RETRY_ATTEMPTS):
    """Run ``fn``, retrying transient failures. Returns None if all fail."""
    import time

    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - one week must not kill the season
            last = f"{type(exc).__name__}: {str(exc)[:90]}"
            if i < attempts:
                print(
                    f"    {what}: {last} (attempt {i}/{attempts}, retrying)", flush=True
                )
                time.sleep(_RETRY_BACKOFF_S * i)
            else:
                print(f"    {what}: {last} -- GAVE UP after {attempts}", flush=True)
    return None


def _report_gaps(
    season: int, built: list[int], expected: list[int], dataset: str
) -> None:
    """Say loudly which weeks are missing. Silence here reads as completeness."""
    missing = sorted(set(expected) - set(built))
    if missing:
        print(
            f"  !! {dataset} {season}: MISSING through_week {missing} "
            f"({len(built)}/{len(expected)} built). Consumers joining week W+1 "
            f"to through_week W will silently drop those games -- re-run this "
            f"season before relying on it.",
            flush=True,
        )


def _build_matchup_features(season: int, *, base: str = "cfb") -> pl.DataFrame:
    from cfb_data_build.matchup_build import build_matchup_features

    return build_matchup_features(season, base=base)


def _build_matchup_line(season: int, *, base: str = "cfb") -> pl.DataFrame:
    from cfb_data_build.matchup_build import build_matchup_line_season

    return build_matchup_line_season(season, base=base)


#: first season the espn_cfb_pbp release covers; the career baseline starts here
PBP_FLOOR = 2004


def build_rolling_windows(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """``cfb_rolling_windows``: every entity's last-N-events form through ``season``.

    Reads every committed pbp season from ``PBP_FLOOR`` to ``season`` (the career
    history the baselines need) plus the unified schedule for kickoff dates.

    Memory: `sportsdataverse.rolling_windows.rolling_windows` holds the whole
    events history resident (~5-7 GB peak for the full 2004-2026 CFB run,
    measured ~1.5 GB for 4 seasons). This builder keeps its own footprint down
    by projecting to `FOOTBALL_PBP_COLUMNS` at read time rather than reading
    each season's full pbp frame, so it doesn't add its own multiple on top.
    """
    from sportsdataverse.rolling_windows import (
        FOOTBALL_PBP_COLUMNS,
        football_events,
        rolling_windows,
    )

    pbp_dir = Path(base) / "pbp" / "parquet"
    seasons = [
        s
        for s in range(PBP_FLOOR, season + 1)
        if (pbp_dir / f"play_by_play_{s}.parquet").is_file()
    ]
    if season not in seasons:
        return pl.DataFrame()
    pbp = pl.concat(
        [
            pl.read_parquet(
                pbp_dir / f"play_by_play_{s}.parquet",
                columns=list(FOOTBALL_PBP_COLUMNS),
            )
            for s in seasons
        ],
        how="diagonal_relaxed",
    )
    dates = (
        pl.concat(
            [
                pl.read_parquet(
                    Path(base)
                    / "cfb_schedules"
                    / "parquet"
                    / f"cfb_schedules_{s}.parquet",
                    columns=["game_id", "start_date"],
                )
                for s in seasons
            ],
            how="vertical_relaxed",
        )
        # whole-row (not "game_id" alone): two rows for the same game_id with
        # DIFFERING start_date must stay duplicated so football_events's own
        # duplicate-game_id check raises, instead of silently picking one.
        .unique()
        .select(
            pl.col("game_id").cast(pl.Int64),
            # start_date is a UTC instant; a late CFB kickoff is still the prior
            # evening on the US East Coast, so convert before taking the date --
            # otherwise a handful of late bowl games land on the wrong calendar day.
            game_date=pl.col("start_date")
            .str.to_datetime(time_zone="UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.date(),
        )
    )
    return rolling_windows(football_events(pbp, dates), season)


BUILDERS = {
    "gamelog": build_gamelog,
    "ratings_weekly": build_ratings_weekly,
    "team_summaries_weekly": build_team_summaries_weekly,
    "matchup_features": _build_matchup_features,
    "matchup_line": _build_matchup_line,
    "rolling_windows": build_rolling_windows,
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
            extra = (
                f", {df['through_week'].n_unique()} weekly snapshots"
                if "through_week" in df.columns
                else ""
            )
            print(
                f"  {spec.dataset} {season}: {df.height} rows, {df.width} cols{extra}",
                flush=True,
            )
            if dry_run:
                continue
            write_dataset(df, spec.dataset, season, spec.stem, base=base)
            if publish:
                from cfb_data_build.publish import publish_dataset

                publish_dataset(spec, season, base=base)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True
            )
            failures.append((season, type(exc).__name__))
    return failures
