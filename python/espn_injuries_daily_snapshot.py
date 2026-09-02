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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
import requests

logger = logging.getLogger(__name__)

REPO = "sportsdataverse/sportsdataverse-data"
LEAGUES: tuple[str, ...] = ("nfl", "nba", "wnba", "cfb", "mbb", "wbb", "nhl", "mlb")

#: Explicit so an empty or partly-null league still carries the documented
#: schema, and so the two join keys are pinned to Int64 at the boundary rather
#: than inferred per run (ESPN ships every id as a numeric *string*).
SCHEMA: dict[str, pl.DataType] = {
    "as_of_date": pl.Date,
    "league": pl.Utf8,
    "season": pl.Int64,
    "team_id": pl.Int64,
    "team_display_name": pl.Utf8,
    "team_abbreviation": pl.Utf8,
    "athlete_id": pl.Int64,
    "athlete_display_name": pl.Utf8,
    "athlete_short_name": pl.Utf8,
    "position_abbreviation": pl.Utf8,
    "position_name": pl.Utf8,
    "injury_id": pl.Int64,
    "status": pl.Utf8,
    "type_name": pl.Utf8,
    "type_abbreviation": pl.Utf8,
    "type_description": pl.Utf8,
    "injury_date": pl.Utf8,
    "detail_type": pl.Utf8,
    "detail_location": pl.Utf8,
    "detail_side": pl.Utf8,
    "detail_return_date": pl.Utf8,
    "fantasy_status": pl.Utf8,
    "source": pl.Utf8,
    "short_comment": pl.Utf8,
    "long_comment": pl.Utf8,
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


def _athlete_id(athlete: dict[str, Any]) -> Optional[int]:
    """ESPN omits ``athlete.id`` on this endpoint; recover it from the player
    links (100% coverage on all 8 leagues, 2026-09-02) then the headshot."""
    for link in athlete.get("links") or []:
        found = re.search(r"/id/(\d+)", str(link.get("href") or ""))
        if found:
            return int(found.group(1))
    found = re.search(
        r"/(\d+)\.png", str((athlete.get("headshot") or {}).get("href") or "")
    )
    return int(found.group(1)) if found else None


def athlete_id_coverage(frame: pl.DataFrame) -> float:
    """Share of rows carrying an athlete_id. 1.0 on every league on 2026-09-02.

    ESPN omits ``athlete.id`` on this endpoint, so the id is RECOVERED from the
    player links (and a headshot fallback). That makes it fragile in a specific
    way: if ESPN reshapes those links, every id silently becomes null, the rows
    stay joinable-looking, and nothing fails -- the snapshot just quietly stops
    being usable as a player time series. Measure it so a drop is visible.
    """
    if not frame.height or "athlete_id" not in frame.columns:
        return 1.0
    return 1.0 - frame["athlete_id"].null_count() / frame.height


def explode(payload: dict[str, Any], league: str, as_of: Any) -> pl.DataFrame:
    """Flatten the nested payload to one row per (team, athlete, injury)."""
    season = _int((payload.get("season") or {}).get("year"))
    rows: list[dict[str, Any]] = []
    for team in payload.get("injuries") or []:
        team_id = _int(team.get("id"))
        for inj in team.get("injuries") or []:
            athlete = inj.get("athlete") or {}
            position = athlete.get("position") or {}
            details = inj.get("details") or {}
            itype = inj.get("type") or {}
            rows.append(
                {
                    "as_of_date": as_of,
                    "league": league,
                    "season": season,
                    "team_id": team_id,
                    "team_display_name": team.get("displayName"),
                    "team_abbreviation": (athlete.get("team") or {}).get(
                        "abbreviation"
                    ),
                    "athlete_id": _athlete_id(athlete),
                    "athlete_display_name": athlete.get("displayName"),
                    "athlete_short_name": athlete.get("shortName"),
                    "position_abbreviation": position.get("abbreviation"),
                    "position_name": position.get("displayName"),
                    "injury_id": _int(inj.get("id")),
                    "status": inj.get("status"),
                    "type_name": itype.get("name"),
                    "type_abbreviation": itype.get("abbreviation"),
                    "type_description": itype.get("description"),
                    "injury_date": inj.get("date"),
                    "detail_type": details.get("type"),
                    "detail_location": details.get("location"),
                    "detail_side": details.get("side"),
                    "detail_return_date": details.get("returnDate"),
                    "fantasy_status": (details.get("fantasyStatus") or {}).get(
                        "description"
                    ),
                    "source": (inj.get("source") or {}).get("description"),
                    "short_comment": inj.get("shortComment"),
                    "long_comment": inj.get("longComment"),
                }
            )
    return pl.DataFrame(rows, schema=SCHEMA)


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
    """
    if prior is None or prior.is_empty():
        return today.sort(SORT_KEYS)
    as_of = today["as_of_date"][0]
    keep = prior.filter(pl.col("as_of_date") != as_of)
    return pl.concat([keep, today], how="diagonal_relaxed").sort(SORT_KEYS)


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


def _publish(out: Path, tags: list[str], repo: str, *, dry_run: bool) -> int:
    from sportsdataverse.release import sportsdataverse_upload

    failed = 0
    for tag in tags:
        files = sorted((out / tag).glob("*.parquet"))
        if not files:
            continue
        if dry_run:
            print(f"[dry-run] would upload to {tag}: {[f.name for f in files]}")
            continue
        if not sportsdataverse_upload(files, tag, repo=repo):
            print(f"WARNING: upload failed for {tag}")
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
        tags = sorted({key.split("/")[0] for key in written})
        return 1 if _publish(out, tags, args.repo, dry_run=args.dry_run) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
