"""CLI entry-point for the CFB CPOE training pipeline (Track 5).

Usage
-----
Train on 2021–2023, run LOSO CV, save model::

    uv run --env-file .env python -m cfb_model_build.cpoe \\
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
--variant    game_state | air_yards | both (default: both). The two CP
             models are complements, not alternatives -- see AIR_YARDS.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cfb_model_build.cpoe",
        description="Train the CFB CP model and compute CPOE.",
    )
    p.add_argument(
        "--final-dir",
        default=".cache/cfb_final",
        help="Directory of per-game final.json files (default: .cache/cfb_final).",
    )
    # Keep --raw-dir as a hidden alias so existing callers don't hard-break.
    p.add_argument("--raw-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--out-dir", required=True, help="Output directory for model + CV results."
    )
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
    p.add_argument(
        "--variant",
        choices=("game_state", "air_yards", "both"),
        default="both",
        help=(
            "Which CP model(s) to train. 'game_state' is the 2004+ 8-feature "
            "model; 'air_yards' adds throw depth and trains on the 2025+ rows "
            "that have it; 'both' (default) trains each and is what the "
            "published artifacts carry. The two are complements, not "
            "alternatives -- see AIR_YARDS.md."
        ),
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
    final_dir = pathlib.Path(
        args.raw_dir if args.raw_dir is not None else args.final_dir
    )
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
        print(
            "ERROR: no data loaded -- check --final-dir and --seasons.", file=sys.stderr
        )
        return 1

    print(f"Total pass plays loaded: {len(all_df):,}")

    from .constants import (
        AIR_YARDS_FEATURE_COLS,
        AIR_YARDS_MODEL_FILENAME,
        FEATURE_COLS,
        MIN_SEASON_AIR_YARDS,
        MODEL_FILENAME,
        TARGET_COL,
    )
    from .train_cp import save_cp_model, train_cp_model

    wanted = ["game_state", "air_yards"] if args.variant == "both" else [args.variant]
    trained = 0

    for variant in wanted:
        if variant == "air_yards":
            missing = [c for c in AIR_YARDS_FEATURE_COLS if c not in all_df.columns]
            if missing:
                msg = f"air-yards variant needs {missing}, absent from the loaded plays"
                if args.variant == "air_yards":
                    print(f"ERROR: {msg}.", file=sys.stderr)
                    return 1
                # 'both' over a pre-2025 corpus is the normal case, not a failure.
                print(f"Skipping air-yards variant: {msg}.")
                continue
            # Season floor as well as non-null air yards. ESPN emitted these
            # spots only from 2025; a stray pre-2025 play carrying air_yards is
            # a parse artifact of an era that did not report them, not a sample
            # from the same regime, and MIN_SEASON_AIR_YARDS would otherwise be
            # a constant documenting an intent nothing enforces.
            eligible = all_df["air_yards"].notna()
            if "season" in all_df.columns:
                eligible &= all_df["season"] >= MIN_SEASON_AIR_YARDS
            df = all_df[eligible].reset_index(drop=True)
            feats, fname, mtype = (
                AIR_YARDS_FEATURE_COLS,
                AIR_YARDS_MODEL_FILENAME,
                "cpoe_air_yards",
            )
            if df.empty:
                msg = f"no loaded play has air_yards in {MIN_SEASON_AIR_YARDS}+"
                if args.variant == "air_yards":
                    print(f"ERROR: {msg}.", file=sys.stderr)
                    return 1
                print(f"Skipping air-yards variant: {msg}.")
                continue
        else:
            df, feats, fname, mtype = all_df, FEATURE_COLS, MODEL_FILENAME, "cpoe"

        print()
        print(f"=== {variant} ({len(df):,} plays, {len(feats)} features) ===")

        if args.loso:
            cv, cv_name = _cross_validate(df, variant, nrounds, feats)
            cv_path = out_dir / cv_name
            cv_path.write_text(json.dumps(cv, indent=2), encoding="utf-8")
            print(f"  mean log-loss: {cv['summary']['mean_log_loss']:.4f}")
            print(f"  mean Brier:    {cv['summary']['mean_brier_score']:.4f}")
            print(f"  CV results -> {cv_path}")

        print(f"  Training on full dataset (nrounds={nrounds}) ...")
        booster = train_cp_model(
            df[feats], df[TARGET_COL], nrounds=nrounds, features=list(feats)
        )
        model_path = out_dir / fname
        save_cp_model(booster, model_path, features=list(feats), model_type=mtype)
        print(f"  Model saved -> {model_path}")
        trained += 1

    if not trained:
        print("ERROR: no variant could be trained.", file=sys.stderr)
        return 1
    return 0


def _cross_validate(df, variant: str, nrounds: int, feats: list):
    """Pick the CV design the variant's season coverage can actually support.

    The air-yards model exists only for 2025+, so LOSO would be two folds, one
    of them a partial season -- a number too noisy to gate on. Grouping by game
    keeps five folds without letting two passes from the same drive land on
    opposite sides of the split.
    """
    from .loso import run_grouped_cv, run_loso_cv

    # The game-state model keeps writing loso_cv.json under its original name:
    # it is the pre-existing artifact and other things read that path. Only
    # the new variant takes a suffixed name.
    stem = "loso_cv" if variant == "game_state" else f"cv_{variant}"

    n_seasons = df["season"].nunique() if "season" in df.columns else 0
    if variant == "air_yards" or n_seasons < 3:
        print(
            f"  Cross-validating: GroupKFold by game ({n_seasons} season(s) present) ..."
        )
        return run_grouped_cv(
            df, nrounds=nrounds, features=list(feats)
        ), f"{stem}.json"
    print("  Cross-validating: LOSO ...")
    return run_loso_cv(
        df, nrounds=nrounds, features=list(feats)
    ), f"{stem}.json"


if __name__ == "__main__":
    sys.exit(main())
