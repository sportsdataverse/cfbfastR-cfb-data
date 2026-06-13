#!/usr/bin/env python
"""Build + (optionally) publish the CFB cross-source crosswalk artifacts.

Python sibling of this repo's R ``espn_cfb_0N_*_creation.R`` dataset builders.
Unlike those (which reshape ``final`` JSON), the crosswalk is built live by
**sportsdataverse-py** -- the ESPN x Fox x Yahoo matching logic is Python-only,
so this dataset is produced here in Python rather than R. It rebuilds the
crosswalks season-by-season, writes each to parquet, and -- with ``--upload`` --
pushes them to the ``cfb_crosswalk`` release tag on ``sportsdataverse-data``
(this repo's only publish target), which is exactly where the generated
``sportsdataverse.cfb.load_cfb_*_crosswalk()`` loaders read from.

Three datasets per season:

* **teams**    -> ``cfb_teams_crosswalk_{season}.parquet``
  (``sportsdataverse.cfb.cfb_teams_crosswalk``)
* **schedule** -> ``cfb_schedule_crosswalk_{season}.parquet``
  (``sportsdataverse.cfb.cfb_schedule_crosswalk``, full-season mode)
* **rosters**  -> ``cfb_rosters_crosswalk.parquet`` (single *current* snapshot)
  (``sportsdataverse.cfb.cfb_rosters_crosswalk`` is *per-team* and ESPN/Fox only
  expose the **current** roster -- ``season`` doesn't reach those endpoints -- so
  a per-season series would be the same current data relabeled by year. Instead
  this fans the builder out over the *current* season's ESPN<->Fox team-id pairs,
  tags each frame with ``espn_team_id`` / ``fox_team_id``, and writes one
  season-less table. The ``-s`` / ``-e`` range applies only to teams + schedule.)

Requirements:
    A Python env with ``sportsdataverse`` (>= the crosswalk release) + ``polars``
    installed, and the GitHub CLI (``gh``) authenticated for ``--upload``. The
    simplest path is the sibling ``cfbfastR-cfb-raw`` uv env::

        uv run --project ../cfbfastR-cfb-raw python scripts/build_cfb_crosswalk.py ...

    or any environment where ``pip install sportsdataverse`` has been run.

Usage::

    # build the current season locally, no upload (default)
    python scripts/build_cfb_crosswalk.py

    # build 2022-2024 and publish to the cfb_crosswalk release tag
    python scripts/build_cfb_crosswalk.py -s 2022 -e 2024 --upload

    # just the teams + schedule datasets for one season
    python scripts/build_cfb_crosswalk.py -s 2024 -e 2024 --datasets teams schedule

Publishing creates the release tag on first use (piggyback/gh uploads do NOT
create a missing tag -- it would otherwise retry "release not found" until it
gives up). Assets upload with ``--clobber`` so re-runs overwrite in place
(idempotent), consistent with this repo's daily dataset refresh.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import polars as pl

from sportsdataverse.cfb import (
    cfb_rosters_crosswalk,
    cfb_schedule_crosswalk,
    cfb_teams_crosswalk,
)
from sportsdataverse.cfb.cfb_schedule import most_recent_cfb_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_cfb_crosswalk")

DATASETS = ("teams", "schedule", "rosters")
DEFAULT_TAG = "cfb_crosswalk"
DEFAULT_REPO = "sportsdataverse/sportsdataverse-data"


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------
def build_teams(season: int) -> pl.DataFrame:
    """The ESPN x Fox x Yahoo team-id crosswalk for ``season``."""
    return cfb_teams_crosswalk(season=season)


def build_schedule(season: int) -> pl.DataFrame:
    """The full-season ESPN x Fox x Yahoo game-id crosswalk for ``season``."""
    return cfb_schedule_crosswalk(season)  # week omitted -> whole season


def build_rosters(season: int, teams_xwalk: pl.DataFrame, workers: int) -> pl.DataFrame:
    """Fan the per-team roster crosswalk over every ESPN<->Fox team-id pair.

    ``cfb_rosters_crosswalk`` is keyed on a single (espn_team_id, fox_team_id)
    pair, so a per-season artifact requires walking the season's team crosswalk
    and concatenating each team's frame (tagged with the team ids it came from).
    Teams missing either id are skipped -- the roster join needs both providers.
    """
    pairs = (
        teams_xwalk.filter(pl.col("espn_team_id").is_not_null() & pl.col("fox_team_id").is_not_null())
        .select("espn_team_id", "fox_team_id")
        .unique()
    )
    if pairs.height == 0:
        logger.warning("season %s: no espn<->fox team pairs; rosters crosswalk empty", season)
        return pl.DataFrame()

    def one(espn_id: int, fox_id: str) -> Optional[pl.DataFrame]:
        try:
            df = cfb_rosters_crosswalk(espn_id, fox_id, season=season, providers=("espn", "fox"))
        except Exception as exc:  # noqa: BLE001 -- one team must not sink the season
            logger.warning("season %s team espn=%s fox=%s: roster crosswalk failed: %s", season, espn_id, fox_id, exc)
            return None
        if df.height == 0:
            return None
        # Prepend the team-id provenance so a row can be traced back to its team.
        return df.select(
            pl.lit(espn_id).cast(pl.Int64).alias("espn_team_id"),
            pl.lit(str(fox_id)).alias("fox_team_id"),
            pl.all(),
        )

    rows = list(pairs.iter_rows(named=True))
    frames: List[pl.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, r["espn_team_id"], r["fox_team_id"]): r for r in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            frame = fut.result()
            if frame is not None:
                frames.append(frame)
            if done % 25 == 0 or done == len(rows):
                logger.info(
                    "season %s rosters: %s/%s teams fetched (%s with players)", season, done, len(rows), len(frames)
                )
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


# ---------------------------------------------------------------------------
# IO + publish
# ---------------------------------------------------------------------------
def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # zstd to match the R artifacts' on-disk format.
    df.write_parquet(path, compression="zstd")
    logger.info("wrote %s (%s rows, %s cols)", path.name, df.height, df.width)


def ensure_release(tag: str, repo: str) -> None:
    """Create the release tag if it doesn't exist yet (uploads can't create it)."""
    exists = subprocess.run(
        ["gh", "release", "view", tag, "-R", repo],
        capture_output=True, text=True,
    )
    if exists.returncode == 0:
        return
    logger.info("creating release tag %s on %s", tag, repo)
    subprocess.run(
        ["gh", "release", "create", tag, "-R", repo,
         "--title", "CFB Cross-Source Crosswalk",
         "--notes", "ESPN x Fox x Yahoo CFB identity crosswalks (teams / schedule / rosters), "
                    "per season. Built by cfbfastR-cfb-data scripts/build_cfb_crosswalk.py "
                    "(sportsdataverse-py); read by sportsdataverse.cfb.load_cfb_*_crosswalk()."],
        check=True,
    )


def upload_assets(paths: List[Path], tag: str, repo: str) -> None:
    if not paths:
        return
    ensure_release(tag, repo)
    logger.info("uploading %s asset(s) to %s@%s (one at a time)", len(paths), repo, tag)
    # Upload per-file: a single batched `gh release upload` of many assets can
    # silently drop some (observed: the larger files dropped while gh still exited
    # 0), so upload individually with --clobber and let check=True surface any
    # failure. Slower, but each asset is verified.
    for p in paths:
        subprocess.run(
            ["gh", "release", "upload", tag, str(p), "--clobber", "-R", repo],
            check=True,
        )
        logger.info("  uploaded %s", p.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    recent = most_recent_cfb_season()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--start-year", type=int, default=recent, help="first season to build (default: most recent)")
    parser.add_argument("-e", "--end-year", type=int, default=recent, help="last season to build (default: most recent)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS), help="which datasets to build")
    parser.add_argument("--out-dir", type=Path, default=Path("cfb/crosswalk/parquet"), help="local parquet workspace")
    parser.add_argument("--roster-workers", type=int, default=4, help="parallel teams when building the rosters crosswalk")
    parser.add_argument("--upload", action="store_true", help="publish artifacts to the release tag (outward-facing)")
    parser.add_argument("--release-tag", default=DEFAULT_TAG, help=f"release tag to upload to (default: {DEFAULT_TAG})")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo hosting the release (default: {DEFAULT_REPO})")
    args = parser.parse_args(argv)

    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")

    seasons = range(args.start_year, args.end_year + 1)
    manifest: List[dict] = []
    produced: List[Path] = []

    # teams + schedule are genuinely per-season and loop over the -s/-e range.
    for season in seasons:
        logger.info("=== season %s ===", season)

        if "teams" in args.datasets:
            try:
                teams_df = build_teams(season)
                if teams_df.height:
                    p = args.out_dir / f"cfb_teams_crosswalk_{season}.parquet"
                    write_parquet(teams_df, p)
                    produced.append(p)
                    manifest.append(_row("teams", season, teams_df, "cfb_teams_crosswalk()"))
            except Exception as exc:  # noqa: BLE001
                logger.error("season %s: teams crosswalk failed: %s", season, exc)

        if "schedule" in args.datasets:
            try:
                sched = build_schedule(season)
                if sched.height:
                    p = args.out_dir / f"cfb_schedule_crosswalk_{season}.parquet"
                    write_parquet(sched, p)
                    produced.append(p)
                    manifest.append(_row("schedule", season, sched, "cfb_schedule_crosswalk()"))
                else:
                    logger.warning("season %s: schedule crosswalk empty; skipped", season)
            except Exception as exc:  # noqa: BLE001
                logger.error("season %s: schedule crosswalk failed: %s", season, exc)

    # Rosters are CURRENT-ONLY: ESPN/Fox roster endpoints ignore season, so a
    # per-season series would just relabel today's roster by year. Build one
    # season-less snapshot from the current season's ESPN<->Fox team-id pairs.
    if "rosters" in args.datasets:
        current = most_recent_cfb_season()
        logger.info("=== rosters (current snapshot, season %s) ===", current)
        try:
            cur_teams = build_teams(current)
            rosters = build_rosters(current, cur_teams, args.roster_workers) if cur_teams.height else pl.DataFrame()
            if rosters.height:
                p = args.out_dir / "cfb_rosters_crosswalk.parquet"
                write_parquet(rosters, p)
                produced.append(p)
                manifest.append(_row("rosters", current, rosters, "cfb_rosters_crosswalk() [current]"))
            else:
                logger.warning("rosters crosswalk empty; skipped")
        except Exception as exc:  # noqa: BLE001
            logger.error("rosters crosswalk failed: %s", exc)

    if not produced:
        logger.error("no artifacts produced; nothing to write or upload")
        return 1

    manifest_path = args.out_dir / "cfb_crosswalk_in_data_repo.csv"
    _write_manifest(manifest, manifest_path)
    produced.append(manifest_path)

    logger.info("built %s artifact(s) under %s", len(produced), args.out_dir)
    if args.upload:
        upload_assets(produced, args.release_tag, args.repo)
        logger.info("published to %s@%s", args.repo, args.release_tag)
    else:
        logger.info("build-only (pass --upload to publish to %s@%s)", args.repo, args.release_tag)
    return 0


def _row(dataset: str, season: int, df: pl.DataFrame, source_fn: str) -> dict:
    return {
        "dataset": dataset,
        "season": season,
        "row_count": df.height,
        "col_count": df.width,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_fn": source_fn,
    }


def _write_manifest(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "season", "row_count", "col_count", "generated_at_utc", "source_fn"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote manifest %s (%s rows)", path.name, len(rows))


if __name__ == "__main__":
    sys.exit(main())
