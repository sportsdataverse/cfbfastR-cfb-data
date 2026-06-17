from __future__ import annotations

import argparse

from .fetch import fetch_final


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_data_ingest")
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    ap.add_argument("--cache-dir", default=".cache/cfb_final")
    ap.add_argument("--schedule", default=None, help="local schedule master override; default RAW_BASE URL")
    ap.add_argument("--refresh", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stats = fetch_final(args.seasons, args.cache_dir, schedule=args.schedule, refresh=args.refresh)
    print(f"ingest: fetched={stats['fetched']} skipped={stats['skipped']} "
          f"missing={stats['missing']} total={stats['total']} -> {args.cache_dir}")
    return 0
