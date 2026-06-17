from __future__ import annotations

import argparse

from .artifacts import upload_artifacts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_publish")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("artifacts")
    a.add_argument("--artifacts", required=True)
    a.add_argument("--tag", default="espn_cfb_model_artifacts")
    a.add_argument("--repo", default="sportsdataverse/sportsdataverse-data")
    a.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "artifacts":
        res = upload_artifacts(args.artifacts, args.tag, args.repo, dry_run=args.dry_run)
        print(f"publish: uploaded={res['uploaded']} files={len(res['files'])} -> {args.repo}:{res['tag']}"
              + (" (dry-run)" if args.dry_run else ""))
    return 0
