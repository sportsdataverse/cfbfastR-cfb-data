"""Builder: ESPN league-wide injury report, daily append snapshot (8 leagues).

ESPN's ``/injuries`` endpoint is **league-level, takes no arguments, and reports
CURRENT STATE ONLY** -- there is no history behind it. The whole value of this
dataset is therefore the ``as_of_date`` time series, so every run **appends**
today's snapshot to the prior release asset rather than replacing it.

Why this cross-league stage lives in the CFB producer: ``espn_cfb_injuries`` is
the only injuries tag in the ecosystem, and its per-game producer
(``espn_cfb_14_injuries_creation.py``) is *structurally incapable* of emitting
rows -- ESPN's game summary always ships ``injuries: []`` (verified across the
12 most-recent raw finals; ClaudeCowork ledger L53). Rather than spread eight
near-identical stages across eight ``-data`` repos, injuries keep one owner here
and the league endpoint feeds all eight tags. See CLAUDE.md.

One call per league per day -- ~8 requests total, not a per-team fan-out.

The flatten (and the athlete-id recovery ESPN forces on it) lives ONCE, in
``sportsdataverse.espn_snapshots.parse_injuries_snapshot``; this stage owns only
what is producer-specific: the season that names the asset, the Int64 id
convention, the append, the empty-league skip and the publish.

Output: one row per ``(as_of_date, league, team, athlete, injury)``, published as
``espn_{league}_injuries`` / ``injuries_{season}.parquet`` (the ``{season}``
asset template every other release uses, so a YAML loader row works later).

A league that returns zero athlete rows is **skipped and logged** -- a zero-row
asset is never written and never published (84 schema-only ``ncaa_baseball``
assets reached a release that way; ledger L54). Offseason leagues (mbb/wbb in
September) hit this every day, by design.

Example:
    Build every league into ``out/injuries`` without publishing::

        uv run python python/espn_injuries_daily_snapshot.py

    One league, then publish::

        uv run python python/espn_injuries_daily_snapshot.py -l nfl --publish
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
import requests
from sportsdataverse.espn_snapshots import (
    INJURY_SNAPSHOT_SCHEMA,
    parse_injuries_snapshot,
)

logger = logging.getLogger(__name__)

REPO = "sportsdataverse/sportsdataverse-data"
LEAGUES: tuple[str, ...] = ("nfl", "nba", "wnba", "cfb", "mbb", "wbb", "nhl", "mlb")

#: Every id ESPN ships on this endpoint. The library parser emits them as Utf8
#: (ESPN's own wire form); this producer publishes Int64 because every other
#: published ESPN asset does -- ``espn_cfb_rosters/cfb_rosters_2025.parquet``
#: carries ``team_id``/``athlete_id``/``position_id`` as Int64, and an injuries
#: asset keyed Utf8 would not join to it. The cast happens exactly once, here at
#: the producer boundary, and never inside a second parser.
ID_COLUMNS = ("team_id", "athlete_id", "injury_id", "type_id", "source_id")

#: The published frame contract: the library's
#: :data:`~sportsdataverse.espn_snapshots.INJURY_SNAPSHOT_SCHEMA` with the ids
#: re-pinned to Int64 and ``season`` added (it names the release asset).
#: Deriving it from the library's schema is what keeps the two in step -- a
#: column added upstream arrives here instead of silently going missing.
SCHEMA: dict[str, pl.DataType] = {
    "as_of_date": pl.Date,
    "league": pl.Utf8,
    "season": pl.Int64,
    **{
        name: (pl.Int64 if name in ID_COLUMNS else dtype)
        for name, dtype in INJURY_SNAPSHOT_SCHEMA.items()
        if name not in ("as_of_date", "league")
    },
}

SORT_KEYS = ["as_of_date", "league", "team_id", "athlete_id", "injury_id"]


def fetch_league(league: str) -> dict[str, Any]:
    """Raw ``espn_{league}_injuries()`` payload (the parsed frame stringifies
    the nested ``injuries`` list, which is exactly what we need to explode)."""
    mod = importlib.import_module(f"sportsdataverse.{league}")
    fn = getattr(mod, f"espn_{league}_injuries")
    return fn(return_parsed=False)


def _int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def athlete_id_coverage(frame: pl.DataFrame) -> float:
    """Share of rows carrying an athlete_id. 1.0 on every league on 2026-09-02.

    ESPN omits ``athlete.id`` on this endpoint (0 of 1,291 records across all 8
    leagues carried one on 2026-09-02), so the id is RECOVERED from the
    player-card link by the library parser. That makes it fragile in a specific
    way: if ESPN reshapes those links, every id silently becomes null, the rows
    stay joinable-looking, and nothing fails -- the snapshot just quietly stops
    being usable as a player time series. Measure it so a drop is visible.
    """
    if not frame.height or "athlete_id" not in frame.columns:
        return 1.0
    return 1.0 - frame["athlete_id"].null_count() / frame.height


def _cast_ids(frame: pl.DataFrame) -> pl.DataFrame:
    """Re-pin the parser's Utf8 ids to this repo's published Int64.

    polars parses ``Utf8 -> Int64`` as an integer, so the float route that turns
    ``123`` into ``"123.0"`` is never taken. A value that is not an integer
    becomes null under ``strict=False`` -- and that silent loss is exactly what a
    join key must not do, so it is asserted against rather than trusted.
    """
    cast = frame.with_columns(pl.col(list(ID_COLUMNS)).cast(pl.Int64, strict=False))
    for column in ID_COLUMNS:
        lost = cast[column].null_count() - frame[column].null_count()
        if lost:
            raise ValueError(f"{column}: {lost} id(s) did not survive the Int64 cast")
    return cast


def explode(payload: dict[str, Any], league: str, as_of: Any) -> pl.DataFrame:
    """Flatten the nested payload to one row per (team, athlete, injury).

    The flatten itself -- including recovering ``athlete_id`` from the player-card
    link, which ESPN omits from every record on this endpoint -- belongs to
    :func:`sportsdataverse.espn_snapshots.parse_injuries_snapshot`; this producer
    only adds what is its own: the ``season`` that names the release asset, and
    the Int64 id convention its other published assets join on.
    """
    season = _int((payload.get("season") or {}).get("year"))
    parsed = parse_injuries_snapshot(payload, league=league, as_of_date=as_of)
    return (
        _cast_ids(parsed)
        .with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))
        .select(list(SCHEMA))
    )


def read_prior(tag: str, asset: str, *, repo: str = REPO) -> Optional[pl.DataFrame]:
    """Prior release asset, or ``None`` when the tag/asset does not exist yet.

    A 404 is "nothing published yet"; any other failure is unknown and raises,
    so a rate limit can never be mistaken for an empty history and silently
    truncate the time series.
    """
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return pl.read_parquet(io.BytesIO(resp.content))


def append_snapshot(prior: Optional[pl.DataFrame], today: pl.DataFrame) -> pl.DataFrame:
    """Append today's rows, replacing any existing rows for the same date.

    Idempotent: re-running on the same day replaces that day's rows instead of
    duplicating them.

    The merged frame is normalized back to TODAY's contract. In an append
    dataset the prior asset is by construction older than the current schema, so
    drift is the normal case, not a hypothesis: ``diagonal_relaxed`` would
    otherwise carry a retired column into every future write (all-null from here
    on) and could widen a join key's dtype to the prior asset's. Columns that
    are genuinely gone are logged as they are dropped -- never silently.
    """
    if prior is None or prior.is_empty():
        return today.sort(SORT_KEYS)
    as_of = today["as_of_date"][0]
    keep = prior.filter(pl.col("as_of_date") != as_of)
    merged = pl.concat([keep, today], how="diagonal_relaxed")
    retired = [c for c in merged.columns if c not in today.columns]
    if retired:
        logger.warning(
            "append_dropped_retired_columns %s -- present in the prior asset, not in"
            " the current schema",
            retired,
        )
    # sort AFTER the cast, never before: `diagonal_relaxed` promotes a prior Utf8
    # id column to Utf8, and sorting there is lexical -- "10" lands before "2" and
    # the cast to Int64 preserves that wrong order.
    return merged.select(today.columns).cast(dict(today.schema)).sort(SORT_KEYS)


def build(
    leagues: list[str],
    out: Path,
    *,
    as_of: Optional[Any] = None,
    fetch: Optional[Callable[[str], dict[str, Any]]] = None,
    prior_reader: Optional[Callable[..., Optional[pl.DataFrame]]] = None,
    repo: str = REPO,
) -> dict[str, int]:
    """Snapshot each league and append it to that league's release asset.

    Returns ``{"<tag>/<asset>": total_rows}`` for the leagues that produced
    rows. A league that errors, or that returns zero athlete rows, is skipped
    and logged -- never written.
    """
    # resolved here, not bound as default args, so a caller (or a test) can
    # swap either seam by patching the module attribute
    fetch = fetch or fetch_league
    prior_reader = prior_reader or read_prior
    as_of = as_of or datetime.now(timezone.utc).date()
    written: dict[str, int] = {}
    for league in leagues:
        tag = f"espn_{league}_injuries"
        try:
            today = explode(fetch(league), league, as_of)
            coverage = athlete_id_coverage(today)
            if coverage < 1.0:
                logger.warning(
                    "injuries_athlete_id_gap league=%s coverage=%.3f rows=%s -- the id is"
                    " recovered from ESPN player links; a drop here means that shape changed",
                    league,
                    coverage,
                    today.height,
                )
        except Exception as exc:  # noqa: BLE001 - best-effort: one bad league never sinks the run
            logger.warning("injuries_skip league=%s error=%s", league, str(exc)[:160])
            continue
        if today.is_empty():
            logger.info(
                "injuries_empty league=%s as_of=%s -- not written", league, as_of
            )
            continue
        season = today["season"][0]
        asset = f"injuries_{season}.parquet"
        merged = append_snapshot(prior_reader(tag, asset, repo=repo), today)
        dest = out / tag / asset
        dest.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(dest)
        written[f"{tag}/{asset}"] = merged.height
        logger.info(
            "injuries_write league=%s as_of=%s new=%s total=%s days=%s",
            league,
            as_of,
            today.height,
            merged.height,
            merged["as_of_date"].n_unique(),
        )
    return written


def _publish(out: Path, written: dict[str, int], repo: str, *, dry_run: bool) -> int:
    """Upload ONLY the assets this run wrote, one tag at a time.

    Two things this deliberately does not do:

    * It does not glob the tag directory. ``sportsdataverse_upload`` defaults to
      ``overwrite=True``, so a stale local parquet left by an earlier run would
      replace a good release asset -- the empty-league skip stops us WRITING a bad
      asset, not uploading an old one. ``written`` is keyed ``"<tag>/<asset>"`` and
      is the exact set this invocation produced.
    * It does not let one tag end the run. ``sportsdataverse_upload`` raises
      ``RuntimeError`` when its retries are exhausted, which would abandon every
      later league -- so a single rate-limited tag would silently cost seven
      others. Each failure is counted and the loop continues.
    """
    from sportsdataverse.release import sportsdataverse_upload

    failed = 0
    by_tag: dict[str, list[Path]] = {}
    for key in written:
        tag, _, asset = key.partition("/")
        # `written` is keyed "<tag>/<asset>" and <asset> already carries its
        # extension -- appending ".parquet" here made every path miss, so the
        # loop uploaded nothing and still returned 0 (a silent green publish).
        path = out / tag / asset
        if path.is_file():
            by_tag.setdefault(tag, []).append(path)

    for tag, files in sorted(by_tag.items()):
        if dry_run:
            print(f"[dry-run] would upload to {tag}: {[f.name for f in sorted(files)]}")
            continue
        try:
            if not sportsdataverse_upload(sorted(files), tag, repo=repo):
                print(f"WARNING: upload failed for {tag}")
                failed += 1
        except RuntimeError as exc:  # retries exhausted for THIS tag only
            print(f"WARNING: upload raised for {tag}: {str(exc)[:160]}")
            failed += 1
    return failed


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "-l",
        "--leagues",
        nargs="+",
        choices=LEAGUES,
        default=list(LEAGUES),
        help="leagues to snapshot (default: all 8)",
    )
    ap.add_argument("--out", default="out/injuries", help="output directory")
    ap.add_argument("--repo", default=REPO, help="release repo")
    ap.add_argument("--publish", action="store_true", help="upload the built assets")
    ap.add_argument(
        "--dry-run", action="store_true", help="plan publish without uploading"
    )
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
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
