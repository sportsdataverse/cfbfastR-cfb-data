"""P2 fan-out: rebuild everything downstream of the republished espn_cfb_pbp, ONCE.

Per the program plan, P3-P5 publish exactly one time on the rebuilt pbp rather
than once per phase. Three families:

  adv        10 adv_* datasets -- generic advBoxScore flatten from the local
             final.json tree. Also closes the P7 format gap: those tags carry
             csv.gz + parquet but only 1 rds, so this publish adds plain csv +
             rds per season.
  summaries   5 tables (percentiles/team_summaries/passing/rushing/receiving)
             built from the RELEASED pbp. Now that P2 republished pbp with
             player ids for every season, the player tables should build for
             all 22 rather than the 5 id-seasons they were limited to.
  ratings     cfb_ratings over the released pbp.

Every unit is isolated: one failing dataset-season is recorded and the sweep
continues. The first P2 publish attempt died on a single gh timeout and
abandoned 20 seasons -- that must not repeat.

Usage:
  python fanout_p2.py --phase adv --cache-dir <cfb-raw>/cfb/json/final [--publish]
  python fanout_p2.py --phase summaries [--publish]
  python fanout_p2.py --phase ratings [--publish]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ADV_DATASETS = [
    "adv_defensive",
    "adv_defensive_players",
    "adv_drives",
    "adv_passing",
    "adv_receiving",
    "adv_rushing",
    "adv_situational",
    "adv_specialists",
    "adv_team",
    "adv_turnover",
]


def run_adv(args) -> list[tuple[str, str]]:
    from cfb_data_build.build import build_season
    from cfb_data_build.config import REGISTRY

    failures: list[tuple[str, str]] = []
    for dataset in ADV_DATASETS:
        spec = REGISTRY[dataset]
        for season in range(args.start_year, args.end_year + 1):
            unit = f"{dataset}/{season}"
            try:
                df = build_season(
                    spec,
                    season,
                    cache_dir=Path(args.cache_dir),
                    fetch=False,
                    publish=args.publish and not args.dry_run,
                    base=args.base,
                    include_release_ids=True,
                )
                if df.height == 0:
                    print(f"  {unit}: 0 rows (skipped write)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  FAILED {unit}: {type(exc).__name__}: {str(exc)[:140]}",
                    flush=True,
                )
                failures.append((unit, type(exc).__name__))
    return failures


def run_summaries(args) -> list[tuple[str, str]]:
    from cfb_data_build.summaries_build import build_summaries_season

    failures: list[tuple[str, str]] = []
    for season in range(args.start_year, args.end_year + 1):
        unit = f"summaries/{season}"
        try:
            counts = build_summaries_season(
                season,
                base=args.base,
                publish=args.publish and not args.dry_run,
                dry_run=args.dry_run,
            )
            print(f"  {unit}: {counts}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAILED {unit}: {type(exc).__name__}: {str(exc)[:140]}", flush=True
            )
            if args.traceback:
                traceback.print_exc()
            failures.append((unit, type(exc).__name__))
    return failures


def run_ratings(args) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    try:
        from sportsdataverse.cfb import cfb_ratings
    except Exception as exc:  # noqa: BLE001
        print(f"  cfb_ratings unavailable: {type(exc).__name__}: {exc}")
        return [("ratings/import", type(exc).__name__)]

    for season in range(args.start_year, args.end_year + 1):
        unit = f"ratings/{season}"
        try:
            df = cfb_ratings(season)
            n = getattr(df, "height", len(df))
            print(f"  {unit}: {n} teams", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  FAILED {unit}: {type(exc).__name__}: {str(exc)[:140]}", flush=True
            )
            failures.append((unit, type(exc).__name__))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("adv", "summaries", "ratings"))
    ap.add_argument("--cache-dir", default=None, help="final.json tree (adv phase)")
    ap.add_argument("-s", "--start-year", type=int, default=2004)
    ap.add_argument("-e", "--end-year", type=int, default=2025)
    ap.add_argument("--base", default="cfb")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    if args.phase == "adv" and not args.cache_dir:
        ap.error("--cache-dir is required for --phase adv")

    print(
        f"===== fan-out phase={args.phase} {args.start_year}-{args.end_year} "
        f"publish={args.publish and not args.dry_run} =====",
        flush=True,
    )
    failures = {"adv": run_adv, "summaries": run_summaries, "ratings": run_ratings}[
        args.phase
    ](args)

    print(f"\n=== fan-out {args.phase}: {len(failures)} failures ===")
    for unit, kind in failures:
        print(f"  {unit}: {kind}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
