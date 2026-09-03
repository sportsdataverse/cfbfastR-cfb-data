"""Build ``espn_{league}_coaches`` — one row per coach per season, across eight leagues.

Item 7b of the data stocktake. The backlog assumed transactions/coaches/depth
charts all "need a per-team fan-out"; coaches does not work that way. The league
endpoint ``espn_{lg}_season_coaches(season)`` is a **single call** returning a
Core v2 ``{items: [{$ref}]}`` list — 265 refs for CFB, 32 for NFL, 30 NBA — and
each ``$ref`` resolves to the actual coach record. So the fan-out is **per coach,
not per team**, and it is bounded and cheap: roughly 400 resolutions for a whole
season across every league.

Unlike the injuries and depth-chart stages next to this one, coaches is **not an
append-with-``as_of_date`` snapshot**. A coaching staff is a property of a season,
not of a day: re-running mid-season should correct the season's row, not stack a
second one. So this writes one asset per league-season and replaces it.

Two properties of the published frame that a consumer must know, both measured on
the full CFB 2025 build (265 coaches, 241 distinct teams):

* **``coach_id`` and ``person_id`` are identical on every row.** ESPN ships both --
  the coach resource id and the ``person`` ``$ref`` id -- and today they never
  disagree. Both are kept because they are genuinely different fields upstream and
  could diverge, but they are **not two independent join keys**: joining on both
  buys nothing over joining on one.
* **``experience`` is sparse: null on 190 of 265 CFB rows (72%).** ESPN populates
  it for a minority of coaches. Averaging the column without conditioning on
  non-null describes the 28% who happen to have it, not the coaching population.

Example:
    Build every league for one season without publishing::

        uv run python python/espn_coaches_season_build.py --season 2025

    One league, then publish::

        uv run python python/espn_coaches_season_build.py -l nfl --season 2025 --publish
"""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from sportsdataverse.dl_utils import download

logger = logging.getLogger(__name__)

REPO = "sportsdataverse/sportsdataverse-data"
LEAGUES: tuple[str, ...] = ("nfl", "nba", "wnba", "cfb", "mbb", "wbb", "nhl", "mlb")

#: Which module each league's wrappers live in (the league slug is not always the
#: module name -- cfb wrappers live in ``sportsdataverse.cfb``, mbb in ``.mbb``).
_MODULES = {lg: lg for lg in LEAGUES}

#: Ids published as Int64, matching every other published ESPN asset. ESPN ships
#: them as strings on the wire; an asset keyed Utf8 would not join to
#: ``espn_cfb_rosters``. The cast happens once, here at the producer boundary.
ID_COLUMNS = ("coach_id", "team_id", "person_id")

SCHEMA: Dict[str, pl.DataType] = {
    "league": pl.Utf8,
    "season": pl.Int64,
    "coach_id": pl.Int64,
    "first_name": pl.Utf8,
    "last_name": pl.Utf8,
    "team_id": pl.Int64,
    "person_id": pl.Int64,
    "experience": pl.Int64,
}


def _ref_id(ref: Optional[str]) -> Optional[int]:
    """The trailing numeric id out of a Core v2 ``$ref`` URL.

    ESPN nests ``team`` and ``person`` as ``$ref`` pointers rather than ids, so the
    id has to come off the URL. Returns ``None`` rather than raising when the shape
    is not what we expect -- one odd record must not end a league's build.
    """
    if not ref:
        return None
    tail = str(ref).split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _coach_row(league: str, season: int, record: Dict[str, Any]) -> Dict[str, Any]:
    """One resolved coach payload -> one published row."""
    experience = record.get("experience")
    if isinstance(experience, dict):  # ESPN sometimes nests it as {"years": n}
        experience = experience.get("years")
    return {
        "league": league,
        "season": season,
        "coach_id": _ref_id(record.get("id"))
        if str(record.get("id", "")).isdigit()
        else None,
        "first_name": record.get("firstName"),
        "last_name": record.get("lastName"),
        "team_id": _ref_id((record.get("team") or {}).get("$ref")),
        "person_id": _ref_id((record.get("person") or {}).get("$ref")),
        "experience": experience if isinstance(experience, int) else None,
    }


def fetch_league(
    league: str, season: int, *, limit: Optional[int] = None
) -> pl.DataFrame:
    """Every coach for one league-season, as the published frame.

    Returns a zero-row frame carrying :data:`SCHEMA` when the league has no
    coaches for that season -- an offseason league is a normal state, not an
    error, and an empty frame with the right columns is what lets the caller skip
    it without a null-check.
    """
    module = importlib.import_module(f"sportsdataverse.{_MODULES[league]}")
    fn = getattr(module, f"espn_{league}_season_coaches", None)
    if fn is None:
        logger.warning("no_wrapper league=%s", league)
        return pl.DataFrame(schema=SCHEMA)

    index = fn(season)
    if index.height == 0 or "$ref" not in index.columns:
        logger.warning("no_coaches league=%s season=%s", league, season)
        return pl.DataFrame(schema=SCHEMA)

    refs: List[str] = index["$ref"].to_list()
    if limit is not None:
        refs = refs[:limit]

    rows: List[Dict[str, Any]] = []
    for ref in refs:
        try:
            rows.append(_coach_row(league, season, download(ref).json()))
        except Exception as exc:  # one bad ref must not lose the league
            logger.warning(
                "ref_failed league=%s ref=%s err=%s",
                league,
                str(ref)[:70],
                type(exc).__name__,
            )
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    return pl.DataFrame(rows, schema_overrides=SCHEMA).select(list(SCHEMA))


def build(
    season: int,
    out: Path,
    leagues: tuple[str, ...] = LEAGUES,
    *,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """Build one asset per league for *season*; return ``{"<tag>/<asset>": rows}``.

    A league with zero coaches is **skipped, not written**. Publishing a
    schema-only asset makes a tag advertise coverage it does not have -- the same
    defect that put 84 empty ``ncaa_baseball`` files on a release (ledger L54).
    """
    written: Dict[str, int] = {}
    for league in leagues:
        frame = fetch_league(league, season, limit=limit)
        if frame.height == 0:
            logger.info("skip_empty league=%s season=%s", league, season)
            continue
        tag = f"espn_{league}_coaches"
        directory = out / tag
        directory.mkdir(parents=True, exist_ok=True)
        asset = f"{tag}_{season}.parquet"
        frame.write_parquet(directory / asset)
        written[f"{tag}/{asset}"] = frame.height
        logger.info("wrote league=%s season=%s rows=%s", league, season, frame.height)
    return written


def _publish(out: Path, written: Dict[str, int], repo: str, *, dry_run: bool) -> int:
    """Upload exactly what this run produced. Returns the number of failed tags.

    Deliberately uploads only the paths in ``written`` rather than globbing the tag
    directory: the empty-league skip stops us writing a bad asset, not uploading a
    stale one left by an earlier run.
    """
    from sportsdataverse.release import sportsdataverse_upload

    failures = 0
    by_tag: Dict[str, List[Path]] = {}
    for key in written:
        tag, asset = key.split("/", 1)
        by_tag.setdefault(tag, []).append(out / tag / asset)
    for tag, files in sorted(by_tag.items()):
        if dry_run:
            print(f"[dry-run] would upload to {tag}: {[f.name for f in sorted(files)]}")
            continue
        try:
            sportsdataverse_upload(files, tag=tag, repo=repo)
        except RuntimeError as exc:  # one tag failing must not end the run
            logger.error("publish_failed tag=%s err=%s", tag, exc)
            failures += 1
    return failures


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "-l", "--leagues", nargs="+", default=list(LEAGUES), choices=list(LEAGUES)
    )
    ap.add_argument("--season", type=int, required=True, help="Season (starting year).")
    ap.add_argument("--out", default="out/coaches", help="output directory")
    ap.add_argument("--repo", default=REPO, help="release repo")
    ap.add_argument(
        "--limit", type=int, default=None, help="cap refs per league (smoke tests)"
    )
    ap.add_argument("--publish", action="store_true", help="upload the built assets")
    ap.add_argument(
        "--dry-run", action="store_true", help="plan publish without uploading"
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parser().parse_args(argv)
    out = Path(args.out)
    written = build(args.season, out, tuple(args.leagues), limit=args.limit)
    for key, rows in sorted(written.items()):
        print(f"{key}: {rows} rows")
    if not written:
        print("nothing built -- no league produced rows; not publishing")
        return 0
    if args.publish or args.dry_run:
        return 1 if _publish(out, written, args.repo, dry_run=args.dry_run) else 0
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
