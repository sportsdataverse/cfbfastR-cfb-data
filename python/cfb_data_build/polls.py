"""Weekly AP / Coaches / CFP poll history with movement analytics (F8).

Two datasets from one capture, both long over ``(season_type, week)``:

``cfb_poll_analytics``     season x season_type x week x poll x team
``cfb_poll_week_summary``  season x season_type x week x poll

Source: core-v2 ``/seasons/{s}/types/{t}/weeks/{w}/rankings`` (sdv-py wraps it
as ``espn_cfb_week_rankings``), which lists one ``$ref`` per poll published that
week; each ref resolves to the poll with its ``ranks``. Polls kept: 1 (AP ->
``ap``), 2 (AFCA Coaches -> ``coaches``) and 21 (CFP committee -> ``cfp``), the
ids GOP's ``rankings.ts`` orders on. The FCS / D-II / D-III coaches polls (20,
11, 12) ride the same listing and are dropped.

Probe findings (2026-10-01, from this repo's venv; ``tmp/f8/probe.log``):

* Listing shape ``{count, pageIndex, pageSize, pageCount, items[].$ref}``. An
  unpublished slot answers HTTP 200 with ``items: []`` (2024 t2 w17-20, t3 w2+),
  never a 404, so every slot up to ``_MAX_WEEK`` is probed and an empty one
  costs one cheap request. 2024's regular season runs to week 16; the
  postseason's week 1 holds the final poll (AP dated 2025-01-21, after the
  title game). 2004 carries AP + Coaches only. CFP (21) is in the 2024 listing
  at week 14 and absent at week 10 (the committee's first ranking falls between).
* Poll shape ``{$ref, id, name, shortName, type, date, lastUpdated, headline,
  shortHeadline, season, occurrence, availability, ranks[], others[],
  droppedOut[]}``. ``ranks`` is exactly the 25 ranked teams (``current`` 1..25);
  "others receiving votes" sit in ``others`` and are not captured. Each rank
  carries ``current, previous, points, firstPlaceVotes, trend, record,
  team.$ref`` (plus ``date`` / ``lastUpdated`` in recent seasons; 2004 lacks
  them). ``team.$ref`` is ``.../seasons/{s}/teams/{id}``; the id is parsed from
  it as an integer, never through float.
* Retention: 2016 t2 w5 AP still returns 25 ranks (#1 Alabama 333).
* Stability: the 2024 regular-season final (t2 w16) AP #1 is Oregon 2483 with 62
  first-place votes, dated 2024-12-08 -- the poll as published, not rewritten
  after the title game (the post-title final sits in t3 w1 with Ohio State 194).
  ESPN does NOT rewrite old poll weeks the way it rewrites FPI week 1, so this
  capture is a plain idempotent refetch per season, not append-only: every run
  rebuilds the season from the API and the backfill floor is 2004, not "first
  capture". A fetch failure raises and the season is left untouched rather than
  a partial season overwriting a complete one. In a local build a season whose
  rebuilt frame equals the committed parquet is not rewritten; with ``publish``
  it is always written (byte-identical parquet, so the tree stays clean) and
  uploaded, because local equality is no proof the release has the files.
* ESPN's "week W" poll is the one released ENTERING week W (the 2024 "Week 5"
  AP is dated 2024-09-22, after the week-4 games). Week 1 is the preseason poll.

Definitions. The analytics are pure functions over the long ranks frame; weeks
are sequenced per ``(season, poll)`` by ``(season_type, week)``, so the
postseason's week 1 follows the regular season's last published week.

* ``prev_rank``: the team's rank in that poll's previous published week, or null.
* ``move = prev_rank - rank``, so up is positive. Null unless both are set.
* ``entered``: ranked now and not in the previous week. Never true in a poll's
  first week of the season.
* ``exited``: a row is emitted for a team ranked last week and not this week,
  with ``rank = null`` (``points`` / ``first_place_votes`` null too).
* ``weeks_ranked``: cumulative count of weeks ranked in that poll and season,
  carried on exit rows too.
* Week summary: an unranked team counts as rank 26, over the union of teams
  ranked in either week (exactly that week's analytics rows).
  ``chaos = sum |prev26 - rank26|``; ``volatility`` = population sd of
  ``prev26 - rank26``. Both null in the poll's first week. ``entries`` /
  ``exits`` count the flags.

Request budget: 40 listings plus one request per kept poll per published week,
roughly 100-120 a season, sequential with a short pause (core-v2 403s under
aggressive rate). Backfill: ``--dataset poll_analytics -s 2004 -e 2025 --publish``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import polars as pl

from cfb_data_build.config import DatasetSpec
from cfb_data_build.fpi import (
    _CORE_HOSTS,
    _MAX_WEEK,
    _TEAM_RE,
    CORE,
    SEASON_TYPES,
    _get,
)
from cfb_data_build.io import write_dataset

SPECS: dict[str, DatasetSpec] = {
    "poll_analytics": DatasetSpec(
        "poll_analytics", "cfb_poll_analytics", "cfb_poll_analytics"
    ),
    "poll_week_summary": DatasetSpec(
        "poll_week_summary", "cfb_poll_week_summary", "cfb_poll_week_summary"
    ),
}

#: ESPN poll id -> published ``poll`` value. Anything else in the listing is dropped.
POLLS: dict[int, str] = {1: "ap", 2: "coaches", 21: "cfp"}
_POLL_RE = re.compile(r"/rankings/(\d+)")
#: Seconds between requests; ~120 a season at this pace is about a minute.
_PAUSE = 0.5
#: Rank assigned to a team absent from a week in the summary distances.
_UNRANKED = 26

RANKS_SCHEMA = {
    "season": pl.Int64,
    "season_type": pl.Int64,
    "week": pl.Int64,
    "poll": pl.Utf8,
    "team_id": pl.Int64,
    "rank": pl.Int64,
    "points": pl.Float64,
    "first_place_votes": pl.Int64,
}
ANALYTICS_SCHEMA = {
    "season": pl.Int64,
    "season_type": pl.Int64,
    "week": pl.Int64,
    "poll": pl.Utf8,
    "team_id": pl.Int64,
    "rank": pl.Int64,
    "prev_rank": pl.Int64,
    "move": pl.Int64,
    "entered": pl.Boolean,
    "exited": pl.Boolean,
    "weeks_ranked": pl.Int64,
    "points": pl.Float64,
    "first_place_votes": pl.Int64,
}
SUMMARY_SCHEMA = {
    "season": pl.Int64,
    "season_type": pl.Int64,
    "week": pl.Int64,
    "poll": pl.Utf8,
    "volatility": pl.Float64,
    "chaos": pl.Int64,
    "entries": pl.Int64,
    "exits": pl.Int64,
}


# --- capture ------------------------------------------------------------------


def poll_rows(
    season: int, season_type: int, week: int, poll: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per ranked team from one poll payload (``ranks`` only, never ``others``)."""
    rows: list[dict[str, Any]] = []
    for r in payload.get("ranks") or []:
        rank = r.get("current")
        if not rank:
            continue
        ref = (r.get("team") or {}).get("$ref") or ""
        m = _TEAM_RE.search(ref)
        if not m:
            raise ValueError(
                f"{season} type{season_type} wk{week} {poll}: no team id in {ref!r}"
            )
        fpv = r.get("firstPlaceVotes")
        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "week": week,
                "poll": poll,
                "team_id": int(m.group(1)),
                "rank": int(rank),
                "points": None if r.get("points") is None else float(r["points"]),
                "first_place_votes": None if fpv is None else int(fpv),
            }
        )
    return rows


def _week_rows(season: int, season_type: int, week: int) -> list[dict[str, Any]]:
    """Every kept poll published in one week slot; a fetch failure propagates."""
    listing = _get(f"{CORE}/seasons/{season}/types/{season_type}/weeks/{week}/rankings")
    items = listing.get("items") or []
    if (listing.get("count") or 0) > len(items):
        raise ValueError(
            f"{season} type{season_type} wk{week}: listing paginated "
            f"({listing.get('count')} polls, {len(items)} listed)"
        )
    rows: list[dict[str, Any]] = []
    for item in items:
        ref = (item or {}).get("$ref") or ""
        m = _POLL_RE.search(ref)
        if not m or int(m.group(1)) not in POLLS:
            continue
        if not ref.startswith(_CORE_HOSTS):
            print(
                f"  polls {season} type{season_type} wk{week}: skipped off-host ref {ref[:60]}",
                flush=True,
            )
            continue
        time.sleep(_PAUSE)
        payload = _get(ref.replace("http://", "https://", 1))
        got = poll_rows(season, season_type, week, POLLS[int(m.group(1))], payload)
        if not got:
            # A listed poll that resolves to no ranks (or to a 200 error
            # envelope) is a hole; writing it would publish and commit a gap.
            raise ValueError(
                f"{season} type{season_type} wk{week} {POLLS[int(m.group(1))]}: "
                "listed but returned no ranks"
            )
        rows.extend(got)
    return rows


def fetch_ranks(season: int) -> pl.DataFrame:
    """The season's kept polls, long over ``(season_type, week)``; sequential, polite."""
    rows: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        for week in range(1, _MAX_WEEK + 1):
            time.sleep(_PAUSE)
            rows.extend(_week_rows(season, season_type, week))
    return pl.DataFrame(rows, schema=RANKS_SCHEMA)


# --- analytics ----------------------------------------------------------------

_TEAM_KEY = ["season", "poll", "team_id"]


def _seq() -> pl.Expr:
    """1-based position of a week within its ``(season, poll)`` publication order."""
    order = pl.col("season_type") * 100 + pl.col("week")
    return order.rank("dense").over(["season", "poll"]).cast(pl.Int64)


def poll_analytics(ranks: pl.DataFrame) -> pl.DataFrame:
    """Movement, entries, exits and weeks ranked per team-week (see module docstring)."""
    if ranks.height == 0:
        return pl.DataFrame(schema=ANALYTICS_SCHEMA)
    cur = ranks.with_columns(seq=_seq())
    weeks = cur.select("season", "poll", "seq", "season_type", "week").unique()
    prev = cur.select(*_TEAM_KEY, seq=pl.col("seq") + 1, prev_rank=pl.col("rank"))
    df = (
        cur.drop("season_type", "week")
        .join(prev, on=[*_TEAM_KEY, "seq"], how="full", coalesce=True)
        # inner: a "next week" past the last published one is not a week, so the
        # exit rows it would carry are dropped here
        .join(weeks, on=["season", "poll", "seq"], how="inner")
        .sort([*_TEAM_KEY, "seq"])
        .with_columns(
            move=pl.col("prev_rank") - pl.col("rank"),
            entered=pl.col("rank").is_not_null()
            & pl.col("prev_rank").is_null()
            & (pl.col("seq") > 1),
            exited=pl.col("rank").is_null(),
            weeks_ranked=pl.col("rank")
            .is_not_null()
            .cast(pl.Int64)
            .cum_sum()
            .over(_TEAM_KEY),
        )
    )
    return (
        df.select(list(ANALYTICS_SCHEMA))
        .cast(ANALYTICS_SCHEMA)
        .sort(
            ["season", "poll", "season_type", "week", "rank", "team_id"],
            nulls_last=True,
        )
    )


def poll_week_summary(analytics: pl.DataFrame) -> pl.DataFrame:
    """Per poll-week volatility, chaos, entries and exits over the analytics rows."""
    if analytics.height == 0:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    d = pl.col("prev_rank").fill_null(_UNRANKED) - pl.col("rank").fill_null(_UNRANKED)
    first = pl.col("seq") == 1
    return (
        analytics.with_columns(seq=_seq())
        .group_by(["season", "season_type", "week", "poll", "seq"])
        .agg(
            volatility=d.std(ddof=0),
            chaos=d.abs().sum(),
            entries=pl.col("entered").sum(),
            exits=pl.col("exited").sum(),
        )
        .with_columns(
            volatility=pl.when(first).then(None).otherwise(pl.col("volatility")),
            chaos=pl.when(first).then(None).otherwise(pl.col("chaos")),
        )
        .select(list(SUMMARY_SCHEMA))
        .cast(SUMMARY_SCHEMA)
        .sort(["season", "poll", "season_type", "week"])
    )


# --- build --------------------------------------------------------------------


def _existing(spec: DatasetSpec, season: int, base: str) -> pl.DataFrame | None:
    path = Path(base) / spec.dataset / "parquet" / f"{spec.stem}_{season}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def build_polls(
    season: int, *, base: str = "cfb", publish: bool = False, dry_run: bool = False
) -> dict[str, int]:
    """Fetch one season, derive both tables, write + publish. Returns rows written per dataset.

    In a local build a table whose rebuilt frame equals the committed parquet
    is not rewritten (delete the season's parquet to force one). With
    ``publish`` the season is always written and uploaded, see below.
    """
    ranks = fetch_ranks(season)
    if ranks.height == 0:
        print(f"  polls {season}: 0 ranks, skipped", flush=True)
        return {}
    analytics = poll_analytics(ranks)
    tables = {
        "poll_analytics": analytics,
        "poll_week_summary": poll_week_summary(analytics),
    }
    written: dict[str, int] = {}
    for key, df in tables.items():
        spec = SPECS[key]
        weeks = df.select("season_type", "week").n_unique()
        print(
            f"  {spec.dataset} {season}: {df.height} rows, {df.width} cols, {weeks} weeks",
            flush=True,
        )
        if dry_run:
            continue
        # The equality skip governs the WRITE only, and only in a local build.
        # With --publish the season is always rewritten and uploaded: the local
        # tree cannot know whether the release has these files -- an upload
        # that failed after the write, or a build-only run followed by
        # --publish, would otherwise leave the release missing or stale while
        # the retry reports success. The parquet is byte-deterministic (an
        # unchanged season leaves the tree clean) and the upload is an
        # idempotent --clobber, so repeating both is cheap.
        existing = _existing(spec, season, base)
        if not publish and existing is not None and existing.equals(df):
            print(f"  {spec.dataset} {season}: unchanged, skipped", flush=True)
            continue
        write_dataset(df, spec.dataset, season, spec.stem, base=base)
        written[spec.dataset] = df.height
        if publish:
            from cfb_data_build.publish import publish_dataset

            publish_dataset(spec, season, base=base)
    return written


def build_polls_range(
    start_year: int,
    end_year: int,
    *,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Build every season in the range; one failure does not cost the rest."""
    failures: list[tuple[int, str]] = []
    for season in range(start_year, end_year + 1):
        try:
            build_polls(season, base=base, publish=publish, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - one season must not kill the backfill
            print(
                f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True
            )
            failures.append((season, type(exc).__name__))
    return failures
