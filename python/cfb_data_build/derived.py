"""Datasets DERIVED from already-built artifacts, not from ``final.json``.

P6 folded these in from standalone scripts so ``--dataset`` is the single
surface for every CFB dataset:

  gamelog                 ``espn_cfb_adv_team_gamelog`` -- adv_team plus the game
                          context it lacks (opponent, home/away, scores, result,
                          date). One row per team-GAME.
  team_opponent_splits    ``cfb_team_opponent_splits`` -- the gamelog projected to
                          opponent, points, EPA/play + adv_situational's success
                          rate. One row per team-GAME; reads this run's gamelog.
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

import datetime as dt
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
    "metric_curves": DatasetSpec("metric_curves", "metric_curves", "cfb_metric_curves"),
    "defense_vs_position": DatasetSpec(
        "defense_vs_position", "cfb_defense_vs_position", "cfb_defense_vs_position"
    ),
    "paper_index_games": DatasetSpec(
        "paper_index_games", "cfb_paper_index_games", "cfb_paper_index_games"
    ),
    "team_opponent_splits": DatasetSpec(
        "team_opponent_splits", "cfb_team_opponent_splits", "cfb_team_opponent_splits"
    ),
}


def schedule_master_available(schedule_path: "Path | None" = None) -> bool:
    """Whether cfbfastR-cfb-raw's schedule master is reachable.

    Four datasets in this builder read the master -- `ratings_weekly`,
    `team_summaries_weekly`, `adv_team_gamelog` and `cfb_schedules` (its
    `home_rank` / `away_rank`; see `schedules_unified.load_master_ranks`) --
    and CI does not check that repo out. The weekly pair raised
    FileNotFoundError on every scheduled run and turned an otherwise-clean
    preseason build RED. With CFB week 1 days away, a job that
    is permanently red cannot signal a real failure.

    A missing raw store is an ABSENT INPUT, not a defect: the recruiting
    datasets already treat it that way (`daily_cfb_processor.sh` skips them
    with a warning when CFB_RAW_ROOT is unset). This is the same contract for
    the weekly pair and the gamelog, which skip; `cfb_schedules` instead
    carries its ranks forward from its last build.
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

    The master lists every week before it is played -- and before its matchups
    exist (2026 week 14's championship TBDs) -- so this is the season's whole
    calendar. The weekly builders emit only its played part: see
    :func:`played_cutoffs`.

    Regular season only: the postseason restarts week numbering at 1 (ESPN's own
    convention -- the schedule master agrees), so including it would collide the
    week labels.
    """
    return sorted(week_starts(season, schedule_path))


def week_starts(season: int, schedule_path: "Path | None" = None) -> dict[int, dt.date]:
    """Each of :func:`schedule_weeks`' first kickoff date (UTC) in the master.

    The master dates a week before ``load_cfb_schedule`` has an FBS game in it
    (2026 week 14's championship TBDs), so this says whether a week has begun
    while :func:`week_cutoffs` still lends it the previous week's bound.
    """
    schedule_path = schedule_path if schedule_path is not None else _schedule_master()
    s = pl.read_parquet(schedule_path).filter(
        (pl.col("season") == season) & (pl.col("season_type") == 2)
    )
    if s.height == 0 or "start_date" not in s.columns:
        return {}
    return dict(
        s.select(
            "week",
            pl.col("start_date").cast(pl.Utf8).str.slice(0, 10).str.to_date(),
        )
        .drop_nulls()
        .group_by("week")
        .agg(pl.col("start_date").min())
        .iter_rows()
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
    previous week's bound; :func:`played_cutoffs` holds it back until the season
    moves past it. One before any FBS game gets None and is not rated.
    """
    return [(w, c) for w, c, _ in _week_bounds(season, schedule_path, games)[0]]


def _week_bounds(
    season: int,
    schedule_path: "Path | None" = None,
    games: "pl.DataFrame | None" = None,
) -> tuple[list[tuple[int, str | None, bool]], dt.date | None]:
    """:func:`week_cutoffs` with whether each bound is the week's OWN (False: no
    FBS game of its own, so it borrows an earlier week's), plus the first FBS
    postseason kickoff (None until one is scheduled)."""
    weeks = schedule_weeks(season, schedule_path)
    if not weeks:
        return [], None
    if games is None:
        from sportsdataverse.cfb import load_cfb_schedule

        games = _retry(
            lambda: load_cfb_schedule(seasons=[season]), what=f"cfb_schedules {season}"
        )
    if games is None or "season" not in games.columns:
        # No published schedule yet (preseason) or the fetch gave up: no week
        # can be bounded, so none is built -- played_cutoffs drops them all.
        return [(w, None, False) for w in weeks], None
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
                # earliest kickoff of ANY later week, not just the next one
                pl.col("first").reverse().cum_min().reverse().shift(-1),
                pl.lit(post_first, dtype=pl.Date),
            ).alias("cutoff"),
        )
    )
    g = pl.DataFrame({"week": weeks}, schema={"week": pl.Int64}).join_asof(
        bounds.with_columns(own=pl.col("week")), on="week", strategy="backward"
    )
    rows = [
        (int(w), None if c is None else str(c), o == w) for w, c, o in g.iter_rows()
    ]
    return rows, post_first


def played_cutoffs(
    season: int, today: dt.date | None = None
) -> list[tuple[int, str | None]]:
    """:func:`week_cutoffs` through the last week whose bound is not after ``today``.

    A later week has nothing new to fit: its snapshot refits the games already
    played and ships them relabelled. The 2026 weekly assets did exactly that --
    through_week 5-15 were copies of week 4 (review of 2026-09-28).

    The boundary is inclusive. The bound is EXCLUSIVE (games dated before it)
    and sits the day after the week's last kickoff, so on the bound's own date
    every game the snapshot can hold has kicked off. Waiting a day more would
    hold back a finished week: 2026 week 4 ended on the 27th (UTC) and its
    bound is the 28th. ``today`` is a UTC date, like the bounds; ``None`` means
    now.

    Bounds never decrease week to week (each is capped at every later week's
    first kickoff), so the first future week ends the season's played part.

    A week the master has not seen kick off (:func:`week_starts`) ends it too.
    2026 week 14 has no FBS game until its championship matchups are set, so it
    borrows week 13's bound (11-30) and, on the bound alone, shipped week 13
    relabelled from then until ~12-06.

    Kicking off is not enough for such a BORROWED bound either: on 12-04
    ``load_cfb_schedule`` can still lack every week-14 game, and the snapshot is
    week 13's. It waits until the season has moved past the week: a later week
    is bounded by its own games, or the FBS postseason has started. The latter
    keeps a completed season's borrowed final week (2019 week 16, Army-Navy).

    A week without a bound (``None``) stays when it leads the season, for each
    builder to treat as before. Trailing, it is every week: the schedule is not
    published yet or its fetch gave up, and team_summaries_weekly, which builds
    by week number, republished the whole season padded. Those are dropped.
    """
    today = today or dt.datetime.now(dt.UTC).date()
    starts = week_starts(season)
    rows, post_first = _week_bounds(season)
    started = post_first is not None and post_first <= today
    own_played = [
        w for w, c, own in rows if own and c and dt.date.fromisoformat(c) <= today
    ]
    played: list[tuple[int, str | None]] = []
    for week, cutoff, own in rows:
        if starts[week] > today or (
            cutoff is not None and dt.date.fromisoformat(cutoff) > today
        ):
            break
        if cutoff is not None and not own:
            if not (started or any(w > week for w in own_played)):
                break
        played.append((week, cutoff))
    while played and played[-1][1] is None:
        played.pop()
    return played


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


#: ``cfb_team_opponent_splits`` columns and dtypes; an empty season carries them too.
TEAM_OPPONENT_SPLITS_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64,
    "season_type": pl.Int64,
    "week": pl.Int64,
    "game_id": pl.Int64,
    "team_id": pl.Int64,
    "opponent_id": pl.Int64,
    "opponent": pl.Utf8,
    "is_home": pl.Boolean,
    "points_for": pl.Int64,
    "points_against": pl.Int64,
    "plays": pl.Int64,
    "epa_per_play": pl.Float64,
    "success_rate": pl.Float64,
}


def build_team_opponent_splits(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """One row per team-GAME: opponent, EPA/play, success rate and points.

    A projection, not a new aggregation: ``adv_team_gamelog`` gives the game
    context, points, ``EPA_per_play`` and its ``scrimmage_plays``; ``adv_situational``
    gives ``EPA_success_rate`` on the same ``(game_id, pos_team_id)`` grain.
    Every game is kept -- FCS opponents and bowls included; "FBS only" is a
    consumer filter. A left join, so a team-game with no situational row keeps
    a null ``success_rate`` rather than disappearing. A missing situational
    parquet raises (the season fails loudly) instead of publishing an all-null
    column, and so does a game with no situational rows at all: that is a stale
    ``adv_situational`` (its stage failed this run, leaving the last run's copy),
    which would publish the newest games with null success rates.

    The situational id column resolves the way :func:`build_gamelog` resolves
    adv_team's: ``pos_team_id`` when present (``pos_team`` is then the name),
    else the id itself in ``pos_team`` (older assets). Unlike there, the cast
    is strict: an id that will not parse raises rather than nulling a join key.
    """
    root = Path(base)
    gamelog = (
        root / "adv_team_gamelog" / "parquet" / f"adv_team_gamelog_{season}.parquet"
    )
    if not gamelog.exists():
        return pl.DataFrame(schema=TEAM_OPPONENT_SPLITS_SCHEMA)
    gl = pl.read_parquet(gamelog).select(
        *(
            pl.col(c).cast(pl.Int64)
            for c in (
                "season",
                "season_type",
                "week",
                "game_id",
                "team_id",
                "opponent_id",
            )
        ),
        "opponent",
        "is_home",
        "points_for",
        "points_against",
        # the scrimmage plays EPA_per_play averages over (E10); EPA_plays also
        # counts special teams (2025 median 77 vs 66)
        pl.col("scrimmage_plays").alias("plays"),
        pl.col("EPA_per_play").alias("epa_per_play"),
    )
    sit = pl.read_parquet(
        root / "adv_situational" / "parquet" / f"adv_situational_{season}.parquet"
    )
    id_col = "pos_team_id" if "pos_team_id" in sit.columns else "pos_team"
    sit = sit.select(
        pl.col("game_id").cast(pl.Int64),
        pl.col(id_col).cast(pl.Int64).alias("team_id"),
        pl.col("EPA_success_rate").alias("success_rate"),
    )
    for k in ("game_id", "team_id"):
        assert gl.schema[k] == sit.schema[k] == pl.Int64, (
            k,
            gl.schema[k],
            sit.schema[k],
        )
    stale = gl.join(sit, on="game_id", how="anti")["game_id"].n_unique()
    if stale:
        raise ValueError(
            f"team_opponent_splits {season}: {stale} games have no situational "
            f"rows (stale adv_situational?)"
        )
    return (
        gl.join(sit, on=["game_id", "team_id"], how="left", validate="m:1")
        .select(list(TEAM_OPPONENT_SPLITS_SCHEMA))
        .cast(TEAM_OPPONENT_SPLITS_SCHEMA)
    )


def build_ratings_weekly(
    season: int, *, base: str = "cfb", today: dt.date | None = None
) -> pl.DataFrame:
    if not schedule_master_available():
        print(
            f"  ratings_weekly {season}: skipped -- no cfbfastR-cfb-raw schedule "
            f"master at {_schedule_master()} (set CFB_RAW_ROOT)",
            flush=True,
        )
        return pl.DataFrame()
    from sportsdataverse.cfb import cfb_ratings

    frames = []
    built: list[int] = []
    cuts = played_cutoffs(season, today)
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


def build_team_summaries_weekly(
    season: int, *, base: str = "cfb", today: dt.date | None = None
) -> pl.DataFrame:
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
    # the same weeks as ratings_weekly; a week without a bound is still built,
    # since summaries count every game with week <= W, not dates
    weeks = [w for w, _ in played_cutoffs(season, today)]
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


#: first season the released pbp carries ``air_yards`` on a usable share of pass
#: attempts (41 % in 2025, 96 % in 2026). Every earlier season has 0-32 stray
#: air-yards plays (2014, 2021, 2024: one each; 2023: 32), which would publish as
#: a one-attempt "curve", so the two air-yards metrics are dropped before it.
AIR_YARDS_FLOOR = 2025
_AIR_YARDS_METRICS = ("cmp_pct_by_air_yards", "epa_by_air_yards")


def build_metric_curves(season: int, *, base: str = "cfb") -> pl.DataFrame:
    """``cfb_metric_curves``: league / team / player rate curves along a continuous axis.

    FG% by kick distance, completion% and EPA by air-yards bucket, 4th-down
    conversion by yards to go and success by down x distance, each bucket with
    attempts, successes, rate and EPA/attempt (``sportsdataverse.metric_curves``).
    Curves are per SEASON, so unlike :func:`build_rolling_windows` this reads only
    ``season``'s pbp, projected to ``FOOTBALL_ATTEMPT_COLUMNS``. The sdv-py
    function emits all three ``entity_type`` rows (league, team, player) from one
    call; the air-yards metrics are kept from ``AIR_YARDS_FLOOR`` on only.
    Returns the empty ``OUTPUT_SCHEMA`` frame when the season's pbp is missing.
    """
    from sportsdataverse.metric_curves import (
        FOOTBALL_ATTEMPT_COLUMNS,
        OUTPUT_SCHEMA,
        football_attempts,
        metric_curves,
    )

    path = Path(base) / "pbp" / "parquet" / f"play_by_play_{season}.parquet"
    if not path.is_file():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    pbp = pl.read_parquet(path, columns=list(FOOTBALL_ATTEMPT_COLUMNS))
    curves = metric_curves(football_attempts(pbp), "cfb")
    if season < AIR_YARDS_FLOOR:
        curves = curves.filter(~pl.col("metric").is_in(_AIR_YARDS_METRICS))
    return curves


def _build_defense_vs_position(season: int, *, base: str = "cfb") -> pl.DataFrame:
    from cfb_data_build.defense_vs_position import build_defense_vs_position

    return build_defense_vs_position(season, base=base)


def _build_paper_index_games(season: int, *, base: str = "cfb") -> pl.DataFrame:
    from cfb_data_build.paper_index import build_paper_index_games

    return build_paper_index_games(season, base=base)


BUILDERS = {
    "gamelog": build_gamelog,
    "ratings_weekly": build_ratings_weekly,
    "team_summaries_weekly": build_team_summaries_weekly,
    "matchup_features": _build_matchup_features,
    "matchup_line": _build_matchup_line,
    "rolling_windows": build_rolling_windows,
    "metric_curves": build_metric_curves,
    "defense_vs_position": _build_defense_vs_position,
    "paper_index_games": _build_paper_index_games,
    "team_opponent_splits": build_team_opponent_splits,
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
