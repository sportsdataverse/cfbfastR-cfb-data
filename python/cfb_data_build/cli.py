"""CLI for cfb_data_build -- mirrors the R creation scripts' ``-s/-e`` driver."""

from __future__ import annotations

import argparse

from cfb_data_build.build import build_dataset
from cfb_data_build.config import REGISTRY


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_data_build")
    ap.add_argument("--dataset", required=True, choices=sorted(REGISTRY))
    ap.add_argument("-s", "--start-year", type=int, required=True)
    ap.add_argument("-e", "--end-year", type=int, required=True)
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
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_dataset(
        args.dataset,
        args.start_year,
        args.end_year,
        cache_dir=args.cache_dir,
        schedule=args.schedule,
        fetch=not args.no_fetch,
        publish=args.publish,
        base=args.base,
    )
    return 0
