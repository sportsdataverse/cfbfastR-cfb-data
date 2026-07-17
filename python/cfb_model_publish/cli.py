from __future__ import annotations

import argparse

from .artifacts import upload_artifacts
from .builders import (
    build_ratings,
    build_recruiting,
    write_ratings_card,
    write_recruiting_card,
)


def _seasons(spec: str) -> list[int]:
    """Parse ``2004:2024`` or ``2023`` into a season list."""
    if ":" in spec:
        start, end = spec.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_publish")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("artifacts")
    a.add_argument("--artifacts", required=True)
    a.add_argument("--tag", default="espn_cfb_model_artifacts")
    a.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    a.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("ratings", help="build + publish opponent-adjusted team ratings")
    r.add_argument(
        "--seasons",
        required=True,
        help="a season (2023) or an inclusive range (2004:2024)",
    )
    r.add_argument("--out", default="out/cfb_ratings")
    r.add_argument("--tag", default="cfb_ratings")
    r.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument(
        "--build-only",
        action="store_true",
        help="write parquet + card, skip the upload",
    )

    p = sub.add_parser(
        "recruiting", help="build + publish preseason recruiting-based projections"
    )
    p.add_argument(
        "--seasons",
        required=True,
        help="a target season (2025) or an inclusive range (2016:2025)",
    )
    p.add_argument("--out", default="out/cfb_recruiting_proj")
    p.add_argument("--tag", default="cfb_recruiting_proj")
    p.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--build-only",
        action="store_true",
        help="write parquet + card, skip the upload",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "artifacts":
        res = upload_artifacts(
            args.artifacts, args.tag, args.repo, dry_run=args.dry_run
        )
        created = " (created release)" if res.get("created_release") else ""
        print(
            f"publish: uploaded={res['uploaded']} files={len(res['files'])} -> {args.repo}:{res['tag']}"
            + created
            + (" (dry-run)" if args.dry_run else "")
        )
    elif args.cmd == "ratings":
        results = build_ratings(_seasons(args.seasons), args.out)
        write_ratings_card(results, args.out)
        total = sum(r["rows"] for r in results)
        if args.build_only:
            print(
                f"ratings: built seasons={len(results)} rows={total} -> {args.out} (build-only)"
            )
            return 0
        res = upload_artifacts(
            args.out,
            args.tag,
            args.repo,
            pattern="cfb_ratings_*.*",
            dry_run=args.dry_run,
        )
        created = " (created release)" if res.get("created_release") else ""
        print(
            f"publish: seasons={len(results)} rows={total} uploaded={res['uploaded']} "
            f"-> {args.repo}:{res['tag']}"
            + created
            + (" (dry-run)" if args.dry_run else "")
        )
    elif args.cmd == "recruiting":
        results = build_recruiting(_seasons(args.seasons), args.out)
        write_recruiting_card(results, args.out)
        total = sum(r["rows"] for r in results)
        if args.build_only:
            print(
                f"recruiting: built seasons={len(results)} rows={total} -> {args.out} (build-only)"
            )
            return 0
        res = upload_artifacts(
            args.out,
            args.tag,
            args.repo,
            pattern="cfb_recruiting_proj_*.*",
            dry_run=args.dry_run,
        )
        created = " (created release)" if res.get("created_release") else ""
        print(
            f"publish: seasons={len(results)} rows={total} uploaded={res['uploaded']} "
            f"-> {args.repo}:{res['tag']}"
            + created
            + (" (dry-run)" if args.dry_run else "")
        )
    return 0
