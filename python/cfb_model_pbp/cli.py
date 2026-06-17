from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from . import __version__
from .build import build_carry_frame, last_completeness, score_cpoe


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cfb_model_pbp")
    ap.add_argument("--final-dir", default=".cache/cfb_final")
    ap.add_argument("--cp-model", required=True, help="trained CP booster (.ubj)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from model_training.ingest import _read_final_plays
    plays = _read_final_plays(args.final_dir, args.seasons)
    carry = build_carry_frame(args.final_dir, args.seasons)
    scored = score_cpoe(carry, plays, args.cp_model)
    scored = scored.with_columns(
        model_pbp_version=pl.lit(__version__),
        cp_model_version=pl.lit(Path(args.cp_model).name),
        ep_model_version=pl.lit("carried:final_json"),
        wp_model_version=pl.lit("carried:final_json"),
        scored_date=pl.lit(date.today().isoformat()),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(args.out)
    comp = last_completeness()
    print(f"model_pbp: rows={scored.height} kept={comp['kept']} dropped={comp['dropped']} -> {args.out}")
    return 0
