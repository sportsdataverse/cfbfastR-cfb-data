"""Builder: ESPN team depth charts, daily append snapshot (NFL / NBA / MLB).

ESPN's ``teams/{team_id}/depthcharts`` endpoint reports **CURRENT STATE ONLY**:
who is listed where on the chart today. There is no history behind it, so the
value of the dataset is the ``as_of_date`` series -- a starter who slid to WR3
in October is only visible if both days were snapshotted. Every run therefore
**appends** today's rows to the prior release asset rather than replacing it,
exactly like ``espn_injuries_daily_snapshot.py``, whose append / publish /
empty-skip machinery this stage reuses instead of re-implementing.

Three leagues, measured live 2026-09-02 (and re-probed before this was wired):

===========  ======  ============  ==============
league       teams   groups/team   requests/day
===========  ======  ============  ==============
nfl          32      3             32
nba          30      1             30
mlb          30      1             30
nhl          --      **none**      0 (excluded)
wnba         --      **none**      0 (excluded)
cfb          --      **none**      0 (excluded)
===========  ======  ============  ==============

**92 requests a day, not the ~400 the stocktake costed.** NHL, WNBA and college
football answer HTTP 200 with the ``depthchart`` key absent entirely -- ESPN
publishes no depth chart for them, so polling them is 122 wasted requests a day
for zero rows, not an empty dataset waiting to fill. ESPN's ``transactions``
endpoint (the third item costed alongside this one) is **dead**: ``{}`` on 10 of
10 probes across six leagues, most recently 2026-09-02. It is not wired here and
should not be -- and if it ever returns, transactions are timestamped *events*,
so the right shape is an append-on-new-id event log, not a daily state snapshot.

The flatten lives in ``sportsdataverse.espn_snapshots.parse_depthchart_snapshot``
so there is one implementation; this stage owns only the producer-specific parts
(the Int64 id convention, the append, the empty-league skip, the publish).

Output: one row per ``(as_of_date, league, team, group, position slot, depth)``,
published as ``espn_{league}_depthcharts`` / ``depthcharts_{season}.parquet``.

Example:
    Build all three leagues into ``out/depthcharts`` without publishing::

        uv run python python/espn_depthcharts_daily_snapshot.py

    One league, then publish::

        uv run python python/espn_depthcharts_daily_snapshot.py -l nfl --publish
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
from sportsdataverse.espn_snapshots import (
    DEPTHCHART_SNAPSHOT_SCHEMA,
    espn_depthcharts_snapshot,
)

from espn_injuries_daily_snapshot import (
    REPO,
    _publish,
    append_snapshot,
    read_prior,
)

logger = logging.getLogger(__name__)

LEAGUES: tuple[str, ...] = ("nfl", "nba", "mlb")

#: ESPN answers aggressive polling with 403. One request per team, sequential,
#: 1.5s apart -- the same pace the injuries stage runs at. 92 requests is ~2.5
#: minutes; there is nothing to gain from racing it.
REQUEST_DELAY = 1.5

#: See ``espn_injuries_daily_snapshot.ID_COLUMNS``: the library parser emits
#: ESPN's own Utf8 ids, this repo publishes Int64 because every other published
#: ESPN asset does (``espn_cfb_rosters`` carries ``team_id`` / ``athlete_id`` as
#: Int64), and the cast happens once, here, at the producer's boundary.
ID_COLUMNS = ("team_id", "group_id", "position_id", "athlete_id")

SCHEMA: dict[str, pl.DataType] = {
    name: (pl.Int64 if name in ID_COLUMNS else dtype) for name, dtype in DEPTHCHART_SNAPSHOT_SCHEMA.items()
}

SORT_KEYS = [
    "as_of_date",
    "league",
    "team_id",
    "group_id",
    "position_slot",
    "depth_rank",
]


def fetch_league(league: str, as_of: Any) -> pl.DataFrame:
    """Every team's depth chart for one league, paced (one request per team)."""
    return espn_depthcharts_snapshot(league, as_of_date=as_of, request_delay=REQUEST_DELAY)


def cast_ids(frame: pl.DataFrame) -> pl.DataFrame:
    """Re-pin the parser's Utf8 ids to this repo's published Int64.

    ``Utf8 -> Int64`` is an integer parse, so the float route that turns ``123``
    into ``"123.0"`` is never taken. A value that is not an integer becomes null
    under ``strict=False`` -- a silently unjoinable key -- so that loss is
    asserted against rather than trusted.
    """
    cast = frame.with_columns(pl.col(list(ID_COLUMNS)).cast(pl.Int64, strict=False))
    for column in ID_COLUMNS:
        lost = cast[column].null_count() - frame[column].null_count()
        if lost:
            raise ValueError(f"{column}: {lost} id(s) did not survive the Int64 cast")
    return cast.select(list(SCHEMA))


def athlete_id_coverage(frame: pl.DataFrame) -> float:
    """Share of rows carrying an athlete_id. 1.0 on all three leagues 2026-09-02.

    Unlike the injuries endpoint, this one ships ``athlete.id`` directly -- but
    the measure is kept because a null id makes a row unjoinable to a roster
    while leaving it looking perfectly well-formed.
    """
    if not frame.height or "athlete_id" not in frame.columns:
        return 1.0
    return 1.0 - frame["athlete_id"].null_count() / frame.height


def build(
    leagues: list[str],
    out: Path,
    *,
    as_of: Optional[Any] = None,
    fetch: Optional[Callable[[str, Any], pl.DataFrame]] = None,
    prior_reader: Optional[Callable[..., Optional[pl.DataFrame]]] = None,
    repo: str = REPO,
) -> dict[str, int]:
    """Snapshot each league and append it to that league's release asset.

    Returns ``{"<tag>/<asset>": total_rows}`` for the leagues that produced rows.
    A league that errors, or that returns zero slots, is skipped and logged --
    never written, so an ESPN outage cannot publish a hole in the time series.
    """
    fetch = fetch or fetch_league
    prior_reader = prior_reader or read_prior
    as_of = as_of or datetime.now(timezone.utc).date()
    written: dict[str, int] = {}
    for league in leagues:
        tag = f"espn_{league}_depthcharts"
        try:
            today = cast_ids(fetch(league, as_of))
            coverage = athlete_id_coverage(today)
            if coverage < 1.0:
                logger.warning(
                    "depthcharts_athlete_id_gap league=%s coverage=%.3f rows=%s",
                    league,
                    coverage,
                    today.height,
                )
        except Exception as exc:  # noqa: BLE001 - one bad league never sinks the run
            logger.warning("depthcharts_skip league=%s error=%s", league, str(exc)[:160])
            continue
        if today.is_empty():
            logger.info("depthcharts_empty league=%s as_of=%s -- not written", league, as_of)
            continue
        season = today["season"][0]
        asset = f"depthcharts_{season}.parquet"
        merged = append_snapshot(prior_reader(tag, asset, repo=repo), today, SORT_KEYS)
        dest = out / tag / asset
        dest.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(dest)
        written[f"{tag}/{asset}"] = merged.height
        logger.info(
            "depthcharts_write league=%s as_of=%s teams=%s new=%s total=%s days=%s",
            league,
            as_of,
            today["team_id"].n_unique(),
            today.height,
            merged.height,
            merged["as_of_date"].n_unique(),
        )
    return written


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "-l",
        "--leagues",
        nargs="+",
        choices=LEAGUES,
        default=list(LEAGUES),
        help="leagues to snapshot (default: all 3 ESPN publishes depth charts for)",
    )
    ap.add_argument("--out", default="out/depthcharts", help="output directory")
    ap.add_argument("--repo", default=REPO, help="release repo")
    ap.add_argument("--publish", action="store_true", help="upload the built assets")
    ap.add_argument("--dry-run", action="store_true", help="plan publish without uploading")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    out = Path(args.out)
    written = build(args.leagues, out, repo=args.repo)
    for key, rows in sorted(written.items()):
        print(f"{key}: {rows} rows")
    if not written:
        print("no league produced rows -- nothing written, nothing published")
        return 0
    if args.publish or args.dry_run:
        return 1 if _publish(out, written, args.repo, dry_run=args.dry_run) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
