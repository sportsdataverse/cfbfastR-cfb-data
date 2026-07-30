"""P2: publish espn_cfb_pbp 2004-2025 season-by-season, verifying between each.

Per the program plan, everything downstream (cfb_ratings, the summaries family,
adv_*) is rebuilt from this pbp -- so a bad season must be caught BEFORE it
propagates, not after. Each season is built, gated, and only then published.

GATES (a season that fails is reported and NOT published; the sweep continues):
  1. non-empty
  2. sack_vec present -- team_summaries/summaries_input consume it; publishing
     without it breaks the very fan-out this feeds
  3. no column regression vs the current release beyond the 7 known
     python-port-vs-R differences
  4. game count within tolerance of the release (the id-union should keep this
     tight; a large drop means the scope widening regressed)

Usage:
  python publish_pbp_p2.py --cache-dir <cfb-raw>/cfb/json/final [-s 2004 -e 2025]
                           [--publish] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from cfb_data_build.build import build_season
from cfb_data_build.config import REGISTRY

# Known python-port vs R-pipeline differences -- NOT regressions from the
# rebuild. Documented in dev/session-notes/2026-07-30-cfb-p2-pbp-rebuild.md.
KNOWN_MISSING = {
    "date",
    "game_date",
    "game_date_time",
    "end.def_team.id",
    "playType",
    "kickoff_return_player_name",
    "punt_return_player_name",
}
GAME_DROP_TOLERANCE = 0.05  # >5% fewer games than the release fails the gate


def gate(df: pl.DataFrame, season: int) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if df.height == 0:
        return False, ["empty frame"]
    if "sack_vec" not in df.columns:
        problems.append("sack_vec MISSING -- would break the summaries fan-out")

    try:
        import sportsdataverse.cfb as cfb

        rel = cfb.load_cfb_pbp([season])
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not load release for comparison ({type(exc).__name__})")
        return not problems, problems

    unexpected = sorted((set(rel.columns) - set(df.columns)) - KNOWN_MISSING)
    if unexpected:
        problems.append(f"unexpected column regression: {unexpected}")

    ng = df["game_id"].cast(pl.Int64).n_unique()
    og = rel["game_id"].cast(pl.Int64).n_unique()
    if og and (og - ng) / og > GAME_DROP_TOLERANCE:
        problems.append(
            f"game shortfall {ng} vs released {og} (>{GAME_DROP_TOLERANCE:.0%})"
        )
    return not problems, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("-s", "--start-year", type=int, default=2004)
    ap.add_argument("-e", "--end-year", type=int, default=2025)
    ap.add_argument("--base", default="cfb")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="build + gate, never publish"
    )
    args = ap.parse_args()

    spec = REGISTRY["pbp"]
    passed, failed = [], []
    for season in range(args.start_year, args.end_year + 1):
        print(f"\n===== {season} =====", flush=True)
        df = build_season(
            spec,
            season,
            cache_dir=Path(args.cache_dir),
            fetch=False,
            publish=False,  # gate first, publish second
            base=args.base,
            output="full",
            include_release_ids=True,
        )
        ok, problems = gate(df, season)
        if not ok:
            print(f"  GATE FAILED {season}: {problems}", flush=True)
            failed.append((season, problems))
            continue
        print(
            f"  GATE OK {season}: {df.height:,} rows, {df.width} cols, "
            f"{df['game_id'].cast(pl.Int64).n_unique()} games",
            flush=True,
        )
        if args.publish and not args.dry_run:
            from cfb_data_build.publish import publish_dataset

            publish_dataset(spec, season, base=args.base)
            print(f"  PUBLISHED {season}", flush=True)
        passed.append(season)

    print(f"\n=== P2 pbp sweep: {len(passed)} gated OK, {len(failed)} failed ===")
    for season, problems in failed:
        print(f"  {season}: {problems}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
