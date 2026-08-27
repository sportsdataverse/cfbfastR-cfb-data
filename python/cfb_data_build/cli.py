"""CLI for cfb_data_build -- mirrors the R creation scripts' ``-s/-e`` driver."""

from __future__ import annotations

import os

import argparse

from cfb_data_build.build import build_dataset
from cfb_data_build.config import REGISTRY

# Datasets DERIVED from already-built artifacts rather than from final.json.
# They were standalone scripts (build_gamelog.py / build_weekly.py) until P6
# folded them in, so `--dataset` is now the single surface for every CFB
# dataset. Adding one is an entry here plus its builder.
#
#   gamelog                  adv_team + schedule context, one row per team-GAME
#   ratings_weekly           cfb_ratings at each week's end, long format
#   team_summaries_weekly    summaries at each week's end, long format
DERIVED = ("gamelog", "ratings_weekly", "team_summaries_weekly")

# ESPN Football Power Index. Separate from DERIVED because these are fetched from
# the core-v2 API rather than derived from an already-built artifact.
#   fpi_weekly    team x season_type x week, point-in-time snapshots
#   power_index   game x team, the matchup predictions the old asset only linked
FPI = ("fpi_weekly", "power_index")

#: Roster-continuity datasets. `recruits`/`team_talent` come from the raw 247
#: store; `returning_production` comes from the ESPN player box + rosters and
#: needs no raw store at all.
RECRUITING = ("recruits", "team_talent", "returning_production")

#: Per-season ESPN team + conference reference, compiled from the season bundles
#: cfbfastR-cfb-raw commits at cfb/teams/json/{season}.json (read over HTTP).
TEAMS = ("teams",)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_data_build")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=sorted(REGISTRY) + ["summaries", *DERIVED, *FPI, *RECRUITING, *TEAMS],
    )
    ap.add_argument("-s", "--start-year", type=int, required=True)
    ap.add_argument("-e", "--end-year", type=int, required=True)
    ap.add_argument(
        "--through-week",
        type=int,
        default=None,
        help="summaries only: cumulative snapshot through week W (default: full season)",
    )
    ap.add_argument("--cache-dir", default=".cache/cfb_final")
    ap.add_argument(
        "--schedule", default=None, help="schedule master path/URL (default: raw URL)"
    )
    ap.add_argument(
        "--no-fetch", action="store_true", help="use cached final.json only"
    )
    ap.add_argument(
        "--publish", action="store_true", help="upload to the espn_cfb_* release"
    )
    ap.add_argument("--base", default="cfb", help="output root directory")
    ap.add_argument(
        "--raw-root",
        default=os.environ.get("CFB_RAW_ROOT", "../cfbfastR-cfb-raw"),
        help=(
            "recruiting datasets: cfbfastR-cfb-raw checkout holding "
            "cfb/recruits/json (env CFB_RAW_ROOT)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="derived datasets: build + report, never write or publish",
    )
    ap.add_argument(
        "--include-release-ids",
        action="store_true",
        help=(
            "also build games the CURRENT release has that the schedule master "
            "omits, so a republish never drops a game consumers can query today"
        ),
    )
    ap.add_argument(
        "--output",
        choices=("default", "lean", "full"),
        default="default",
        help=(
            "pbp column tier. Publish espn_cfb_pbp with --output full: the "
            "default tier drops sack_vec, which team_summaries consumes."
        ),
    )
    return ap


def _run_derived(args) -> int:
    """Dispatch a DERIVED dataset (gamelog / *_weekly) through the shared driver."""
    from cfb_data_build.derived import build_derived

    failures = build_derived(
        args.dataset,
        args.start_year,
        args.end_year,
        base=args.base,
        publish=args.publish,
        dry_run=args.dry_run,
    )
    print(f"\n=== {args.dataset}: {len(failures)} failures ===")
    for season, kind in failures:
        print(f"  {season}: {kind}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.through_week is not None:
        if args.dataset != "summaries":
            ap.error("--through-week only applies to --dataset summaries")
        if args.publish:
            ap.error(
                "--through-week snapshots cannot be published; canonical tags hold season-final builds"
            )
    if args.dataset == "summaries":
        from cfb_data_build.summaries_build import build_summaries_season

        for season in range(args.start_year, args.end_year + 1):
            build_summaries_season(
                season,
                through_week=args.through_week,
                base=args.base,
                publish=args.publish,
            )
        return 0

    if args.dataset in FPI:
        from cfb_data_build.fpi import build_fpi

        failures = build_fpi(
            args.dataset,
            args.start_year,
            args.end_year,
            base=args.base,
            publish=args.publish,
            dry_run=args.dry_run,
        )
        print(f"\n=== {args.dataset}: {len(failures)} failures ===")
        for season, kind in failures:
            print(f"  {season}: {kind}")
        return 1 if failures else 0

    if args.dataset in RECRUITING:
        from cfb_data_build.recruiting import build_recruiting

        failures = build_recruiting(
            args.dataset,
            args.start_year,
            args.end_year,
            raw_root=args.raw_root,
            base=args.base,
            publish=args.publish,
            dry_run=args.dry_run,
        )
        print(f"\n=== {args.dataset}: {len(failures)} failures ===")
        for season, kind in failures:
            print(f"  {season}: {kind}")
        return 1 if failures else 0

    if args.dataset in TEAMS:
        from cfb_data_build.teams import build as build_teams_range

        failures = build_teams_range(
            args.start_year,
            args.end_year,
            base=args.base,
            publish=args.publish,
            dry_run=args.dry_run,
        )
        print()
        print(f"=== {args.dataset}: {len(failures)} failures ===")
        for season, kind in failures:
            print(f"  {season}: {kind}")
        return 1 if failures else 0

    if args.dataset in DERIVED:
        return _run_derived(args)
    build_dataset(
        args.dataset,
        args.start_year,
        args.end_year,
        cache_dir=args.cache_dir,
        schedule=args.schedule,
        fetch=not args.no_fetch,
        publish=args.publish,
        base=args.base,
        output=args.output,
        include_release_ids=args.include_release_ids,
    )
    return 0
