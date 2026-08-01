"""P7: delete legacy .csv.gz assets once their plain-csv replacement is published.

Owner directive: "Every release ships .csv (remove .csv.gz), plus parquet + rds
in every release." The replacements were published by the P2 fan-out; this
removes the superseded compressed copies.

DELETE-AFTER-UPLOAD, PER ASSET. Every .csv.gz is only removed when the exact
matching <stem>.csv exists on the SAME tag. A blanket delete would open the
404 window the program plan warns about (a missing asset reds out PR live-loader
tests), and a tag whose csv publish partially failed would silently lose data.

Consumer audit (2026-07-30) found no reader of .csv.gz:
  sdv-py cfb loaders  parquet only (32 refs, 0 csv.gz)
  cfbfastR            docstring mention only; loaders read rds/parquet
  game-on-paper / sdv-web / sdv-db   no references

Usage:
  python p7_drop_csvgz.py --dry-run      # report what would go
  python p7_drop_csvgz.py --apply
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

REPO = "sportsdataverse/sportsdataverse-data"
TAGS = [
    "espn_cfb_pbp",
    "espn_cfb_adv_defensive",
    "espn_cfb_adv_defensive_players",
    "espn_cfb_adv_drives",
    "espn_cfb_adv_passing",
    "espn_cfb_adv_receiving",
    "espn_cfb_adv_rushing",
    "espn_cfb_adv_situational",
    "espn_cfb_adv_specialists",
    "espn_cfb_adv_team",
    "espn_cfb_adv_turnover",
    "espn_cfb_passing",
    "espn_cfb_rushing",
    "espn_cfb_receiving",
    "espn_cfb_team_summaries",
    "espn_cfb_percentiles",
    "cfb_ratings",
]


def assets(tag: str) -> list[str]:
    out = subprocess.run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "assets"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"{tag}: gh release view failed: {out.stderr[:160]}")
    return [a["name"] for a in json.loads(out.stdout)["assets"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("pass --dry-run or --apply")

    total_del = total_skip = 0
    failures: list[tuple[str, str]] = []
    for tag in TAGS:
        try:
            names = set(assets(tag))
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag}: SKIPPED ({type(exc).__name__}: {str(exc)[:100]})")
            failures.append((tag, "view"))
            continue

        gz = sorted(n for n in names if n.endswith(".csv.gz"))
        deletable, orphaned = [], []
        for g in gz:
            if g[: -len(".csv.gz")] + ".csv" in names:
                deletable.append(g)
            else:
                orphaned.append(g)

        if orphaned:
            print(f"  {tag}: {len(orphaned)} csv.gz have NO csv replacement -- KEEPING: {orphaned[:3]}")
            total_skip += len(orphaned)

        for g in deletable:
            if args.dry_run:
                continue
            r = subprocess.run(
                ["gh", "release", "delete-asset", tag, g, "--repo", REPO, "--yes"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode != 0:
                print(f"    FAILED delete {tag}/{g}: {r.stderr[:120]}")
                failures.append((tag, g))
                continue
        total_del += len(deletable)
        print(f"  {tag}: {'would delete' if args.dry_run else 'deleted'} {len(deletable)} csv.gz")

    verb = "would delete" if args.dry_run else "deleted"
    print(f"\n=== P7 csv.gz sweep: {verb} {total_del}, kept {total_skip} (no csv), {len(failures)} failures ===")
    for tag, what in failures:
        print(f"  {tag}: {what}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
