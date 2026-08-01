"""ESPN Football Power Index datasets: weekly team FPI + per-game matchup FPI.

Two datasets, deliberately separate because they are different grains and come
from different endpoints:

``cfb_fpi_weekly``   team x season x season_type x week
    From core-v2 ``/seasons/{s}/types/{t}/weeks/{w}/powerindex``. These are genuine
    POINT-IN-TIME snapshots -- each row carries ESPN's own ``lastUpdated`` /
    ``run_date_time_key`` and the values move across weeks (Ohio State 2024:
    26.200 in wk3 -> 27.471 in wk8 -> 24.823 in wk15).

    Do NOT build this from the fitt-v3 season endpoint. That endpoint ACCEPTS a
    ``week`` query parameter and silently ignores it, returning the season-final
    figure for every week -- which would look correct and leak end-of-season
    information into early-week rows.

``cfb_power_index``  game x team
    From the per-event core-v2 ``powerindex`` refs, which carry the matchup-level
    predictions (``gameprojection``, ``teampredptdiff``, ``matchupquality``,
    ``teamadjgamescore``). The published asset previously shipped only the
    unresolved ``$ref`` links, so it was a link list rather than data.

The two are complementary: the weekly table has no per-matchup win probability,
and the per-game refs have no season FPI rating.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
import urllib.request
from typing import Any

import polars as pl

from cfb_data_build.config import DatasetSpec
from cfb_data_build.io import write_dataset

CORE = "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; sportsdataverse/cfb-data)"}
_TEAM_RE = re.compile(r"/teams/(\d+)")

SPECS: dict[str, DatasetSpec] = {
    "fpi_weekly": DatasetSpec("fpi_weekly", "cfb_fpi_weekly", "cfb_fpi_weekly"),
    "power_index": DatasetSpec("power_index", "power_index", "espn_cfb_power_index"),
}

# Regular season then postseason. Weeks are probed until a request returns zero
# rows rather than assumed, because the count varies by season (2024 regular runs
# to week 16; 2007 is shorter) and a hardcoded ceiling would silently truncate.
SEASON_TYPES = (2, 3)
_MAX_WEEK = 20


def _get(url: str, *, timeout: int = 45) -> dict[str, Any]:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=timeout
    ) as r:
        return json.load(r)


def _week_rows(season: int, season_type: int, week: int) -> list[dict[str, Any]]:
    """One row per team for a single (season, season_type, week) snapshot."""
    url = (
        f"{CORE}/seasons/{season}/types/{season_type}/weeks/{week}/powerindex?limit=400"
    )
    try:
        payload = _get(url)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        ref = ((item or {}).get("team") or {}).get("$ref", "") or ""
        m = _TEAM_RE.search(ref)
        row: dict[str, Any] = {
            "season": season,
            "season_type": season_type,
            "week": week,
            "team_id": int(m.group(1)) if m else None,
            "last_updated": item.get("lastUpdated"),
            "run_date_time_key": item.get("runDateTimeKey"),
        }
        for block in ("predictives", "efficiencies"):
            for f in item.get(block) or []:
                name = f.get("name")
                if not name:
                    continue
                key = str(name)
                if key in row:
                    key = f"{key}_{block[:3]}"
                row[key] = f.get("value")
        rows.append(row)
    return rows


def build_fpi_weekly(
    season: int, *, base: str = "cfb", workers: int = 6
) -> pl.DataFrame:
    """Weekly FPI snapshots for one season, long over (season_type, week).

    Weeks are walked until one comes back empty, then the walk stops for that
    season type -- ESPN publishes a different number of weeks per season, so a
    fixed range would either truncate a long season or waste requests on a short
    one. Requests are modest in number (about 17 per season), so concurrency is
    kept low to stay friendly to the core API.
    """
    rows: list[dict[str, Any]] = []
    for st in SEASON_TYPES:
        weeks = list(range(1, _MAX_WEEK + 1))
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            got = list(ex.map(lambda w: (w, _week_rows(season, st, w)), weeks))
        got.sort(key=lambda t: t[0])
        for _week, r in got:
            rows.extend(r)
    if not rows:
        return pl.DataFrame()
    df = pl.from_dicts(rows, infer_schema_length=None)

    # ESPN's WEEK 1 slot is NOT a week-1 snapshot: it is consistently overwritten
    # with a late/final computation. Verified across seasons --
    #   2019 wk1 = 2020-04-07   2022 wk1 = 2022-12-22
    #   2023 wk1 = 2023-12-11   2024 wk1 = 2024-12-15
    # each LATER than that season's week-15/16 stamp. Treating it as an as-of-week-1
    # rating would leak the whole season into it, so the condition is flagged IN THE
    # DATA rather than only in prose: a row is out of sequence when its snapshot was
    # computed after one belonging to a later week of the same season type.
    df = df.sort(["season_type", "week", "team_id"])
    # Ordered on run_date_time_key (an Int like 20240929040000), not last_updated:
    # the latter is a Utf8 timestamp and polars cannot cum_min a string column.
    key = pl.col("run_date_time_key").cast(pl.Int64, strict=False)
    later_min = (
        key.reverse().cum_min().reverse().shift(-1).over(["season", "season_type"])
    )

    # SECOND, INDEPENDENT trap: seasons before 2015 were computed RETROSPECTIVELY.
    # Every week of 2005-2014 carries one identical backfill stamp (2014-08-14 or
    # 2015-08-29), so those rows are a reconstruction, not a contemporaneous
    # snapshot, and are NOT valid as an as-of-week rating. The out-of-sequence
    # check cannot catch this -- when every stamp is equal, none is later than a
    # subsequent one. Flagged separately: a snapshot is contemporaneous when it
    # was computed inside the season's own window (Aug of the season year through
    # Feb of the next), which is exactly where the live weekly runs fall.
    # Keyed on last_updated, NOT run_date_time_key. The two are different things:
    # run_date_time_key is the AS-OF date the snapshot represents (2012 rows all
    # carry an in-season 20120903...), while last_updated is when ESPN actually
    # COMPUTED it (those same rows say 2015-08-29). The gap between them IS the
    # retrospection, so keying the flag off run_date_time_key made every
    # backfilled season look contemporaneous.
    yr = pl.col("last_updated").str.slice(0, 4).cast(pl.Int64, strict=False)
    month = pl.col("last_updated").str.slice(5, 2).cast(pl.Int64, strict=False)
    in_window = ((yr == pl.col("season")) & (month >= 8)) | (
        (yr == pl.col("season") + 1) & (month <= 2)
    )
    df = df.with_columns(
        snapshot_out_of_sequence=(key > later_min).fill_null(False),
        snapshot_is_contemporaneous=in_window.fill_null(False),
    )
    lead = [
        "season",
        "season_type",
        "week",
        "team_id",
        "last_updated",
        "run_date_time_key",
        "snapshot_out_of_sequence",
    ]
    rest = [c for c in df.columns if c not in lead]
    return df.select([c for c in lead if c in df.columns] + rest)


def _game_rows(game_id: int) -> list[dict[str, Any]]:
    """Matchup-level FPI for one game: one row per team."""
    url = f"{CORE}/events/{game_id}/competitions/{game_id}/powerindex?limit=10"
    try:
        payload = _get(url, timeout=30)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        ref = item.get("$ref") or ""
        try:
            entry = _get(ref, timeout=30) if "stats" not in item else item
        except Exception:
            continue
        tref = ((entry or {}).get("team") or {}).get("$ref", "") or ref
        m = _TEAM_RE.search(tref)
        row: dict[str, Any] = {
            "game_id": game_id,
            "team_id": int(m.group(1)) if m else None,
        }
        for s in entry.get("stats") or []:
            if s.get("name"):
                row[str(s["name"])] = s.get("value")
        rows.append(row)
    return rows


def build_power_index(
    season: int, *, base: str = "cfb", schedule: str | None = None, workers: int = 8
) -> pl.DataFrame:
    """Per-game matchup FPI for one season, resolving the refs the asset only linked.

    Two rows per game (one per team). Games with no FPI entry -- ESPN only
    publishes these for recent seasons -- simply contribute no rows rather than
    failing the build.
    """
    from cfb_data_ingest.schedule import season_game_ids

    ids = season_game_ids(schedule, [season])
    if not ids:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_game_rows, ids):
            rows.extend(r)
    if not rows:
        return pl.DataFrame()
    df = pl.from_dicts(rows, infer_schema_length=None).with_columns(
        season=pl.lit(int(season), dtype=pl.Int64)
    )
    lead = ["season", "game_id", "team_id"]
    return df.select(lead + [c for c in df.columns if c not in lead]).sort(
        ["game_id", "team_id"]
    )


BUILDERS = {"fpi_weekly": build_fpi_weekly, "power_index": build_power_index}


def build_fpi(
    dataset: str,
    start_year: int,
    end_year: int,
    *,
    base: str = "cfb",
    publish: bool = False,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Build (and optionally publish) an FPI dataset across a season range.

    Every season is isolated: a failure is recorded and the sweep continues, so
    one bad season cannot abort a backfill. Returns the ``(season, error)`` list.
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
            extra = ""
            if "week" in df.columns:
                extra = f", {df['week'].n_unique()} weeks x {df['team_id'].n_unique()} teams"
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
        except Exception as exc:  # noqa: BLE001 - one season must not kill the sweep
            print(
                f"  FAILED {season}: {type(exc).__name__}: {str(exc)[:150]}", flush=True
            )
            failures.append((season, type(exc).__name__))
    return failures
