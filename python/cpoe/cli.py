"""CLI entry-point for the CFB CPOE training pipeline (Track 5).

Usage
-----
Train on 2021–2023, run LOSO CV, save model::

    uv run --env-file .env python -m cpoe \\
        --final-dir .cache/cfb_final \\
        --out-dir artifacts/cpoe \\
        --seasons 2021 2022 2023 \\
        --loso

Args
----
--final-dir  Directory of per-game final.json files produced by the
             cfbfastR-cfb-raw scraper (layout: <final-dir>/<game_id>.json).
             Default: .cache/cfb_final
--out-dir    Output directory for the trained model (.ubj) and CV results (.json).
--seasons    One or more integer seasons to include in training.
--loso       If set, run LOSO cross-validation before full-data training.
--nrounds    XGBoost boosting rounds (default: 560).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cpoe",
        description="Train the CFB CP model and compute CPOE.",
    )
    p.add_argument(
        "--final-dir",
        default=".cache/cfb_final",
        help="Directory of per-game final.json files (default: .cache/cfb_final).",
    )
    # Keep --raw-dir as a hidden alias so existing callers don't hard-break.
    p.add_argument("--raw-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-dir", required=True, help="Output directory for model + CV results.")
    p.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[],
        metavar="YEAR",
        help="Seasons to include (e.g. 2021 2022 2023).",
    )
    p.add_argument(
        "--loso",
        action="store_true",
        default=False,
        help="Run leave-one-season-out cross-validation.",
    )
    p.add_argument(
        "--nrounds",
        type=int,
        default=None,
        help="XGBoost boosting rounds (default: 560 from constants).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CPOE training pipeline.

    Returns:
        Exit code (0 = success, non-zero = failure).
    """
    from .constants import XGB_NROUNDS
    from .ingest import load_season_pass_plays

    parser = build_parser()
    args = parser.parse_args(argv)

    # --raw-dir is a deprecated alias for --final-dir; prefer --final-dir.
    final_dir = pathlib.Path(args.raw_dir if args.raw_dir is not None else args.final_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nrounds = args.nrounds if args.nrounds is not None else XGB_NROUNDS

    seasons = args.seasons
    if not seasons:
        print("ERROR: --seasons must list at least one season.", file=sys.stderr)
        return 1

    if not final_dir.exists():
        print(f"ERROR: final-dir not found: {final_dir}", file=sys.stderr)
        return 1

    print(f"Loading pass plays from {final_dir} (seasons: {seasons}) ...")
    import pandas as pd
    all_df = load_season_pass_plays(final_dir, seasons=seasons)
    if isinstance(all_df, pd.DataFrame) and all_df.empty:
        print("ERROR: no data loaded -- check --final-dir and --seasons.", file=sys.stderr)
        return 1

    print(f"Total pass plays loaded: {len(all_df):,}")

    # --- optional LOSO CV ---
    if args.loso:
        from .loso import run_loso_cv
        print("Running LOSO cross-validation ...")
        cv_result = run_loso_cv(all_df, nrounds=nrounds)
        cv_path = out_dir / "loso_cv.json"
        cv_path.write_text(json.dumps(cv_result, indent=2), encoding="utf-8")
        print(f"  mean log-loss: {cv_result['summary']['mean_log_loss']:.4f}")
        print(f"  mean Brier:    {cv_result['summary']['mean_brier_score']:.4f}")
        print(f"  CV results -> {cv_path}")

    # --- full-data training ---
    from .constants import FEATURE_COLS, TARGET_COL
    from .train_cp import save_cp_model, train_cp_model

    print(f"Training on full dataset ({len(all_df):,} plays, nrounds={nrounds}) ...")
    booster = train_cp_model(all_df[FEATURE_COLS], all_df[TARGET_COL], nrounds=nrounds)
    model_path = out_dir / "cfb_cp_model.ubj"
    save_cp_model(booster, model_path)
    print(f"  Model saved -> {model_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
