"""CLI for model family 35 (``cfb_matchup``).

    python -m cfb_model_build.cfb_matchup train-scoring-opp --seasons 2014-2024 --out-dir python/.cache/matchup/candidate
    python -m cfb_model_build.cfb_matchup train-rush-expect --seasons 2014-2024 --out-dir python/.cache/matchup/candidate

Training rows come from the ``cfbfastR_cfb_pbp`` release (sdv-py
``load_cfb_pbp_r``), one season at a time, or from ``--frame`` (a parquet
already in the trainer's shape) for offline / oracle runs. Each command writes
``<stem>_coef.json`` (the applier's input) and ``<stem>_meta.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from cfb_data_build.matchup_features import (
    RUSH_EXPECT_FEATURES,
    SCORING_OPP_FEATURES,
    build_drive_frame,
    dedupe_plays,
)
from cfb_model_build.cfb_matchup.glm import (
    fit_rush_expect,
    fit_scoring_opp,
    rush_expect_training_frame,
    write_fit,
)

SCORING_OPP_FORMULA = "scoring_opp ~ " + " + ".join(SCORING_OPP_FEATURES)
RUSH_EXPECT_FORMULA = "rush ~ " + " + ".join(RUSH_EXPECT_FEATURES)


def _seasons(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def _load_pbp(seasons: list[int]) -> pl.DataFrame:
    from sportsdataverse.cfb import load_cfb_pbp_r

    frames = [dedupe_plays(load_cfb_pbp_r([s])) for s in seasons]
    return pl.concat(frames, how="diagonal_relaxed")


def _drive_frame_for_training(pbp: pl.DataFrame) -> pl.DataFrame:
    """The drive frame WITHOUT an expectation column (the model is what we fit).

    ``build_drive_frame`` also scores drives with a coefficient table; for
    training we only need the snapshot + label, so score with a zero table.
    """
    from cfb_data_build.matchup_features import GlmCoefficients

    zero = GlmCoefficients(0.0, {}, list(SCORING_OPP_FEATURES), [], "binomial", "logit")
    return build_drive_frame(pbp, zero).drop("scoring_opp_prediction", "scoring_opp_oe")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfb_matchup", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train-scoring-opp", "train-rush-expect"):
        sp = sub.add_parser(name)
        sp.add_argument(
            "--seasons", default="2014-2024", help="e.g. 2014-2024 or 2019,2021"
        )
        sp.add_argument(
            "--frame",
            type=Path,
            default=None,
            help="pre-built training frame parquet (skips the release load)",
        )
        sp.add_argument(
            "--out-dir",
            type=Path,
            required=True,
            help="where to write <stem>_coef.json + _meta.json (a candidate dir; the bundle needs --promote)",
        )
        sp.add_argument(
            "--promote",
            action="store_true",
            help="allow --out-dir to be the bundled artifacts dir (only after the parity + level gates pass)",
        )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seasons = _seasons(args.seasons)
    bundle = Path(__file__).resolve().parent / "artifacts"
    if args.out_dir.resolve() == bundle and not args.promote:
        raise SystemExit(
            "refusing to overwrite the bundled artifacts without --promote: fit into a "
            "candidate dir, run tests/cfb_matchup (parity + level gates), then promote"
        )
    if args.command == "train-scoring-opp":
        frame = (
            pl.read_parquet(args.frame)
            if args.frame
            else _drive_frame_for_training(_load_pbp(seasons))
        )
        result = fit_scoring_opp(frame)
        paths = write_fit(
            result,
            args.out_dir,
            "scoring_opp",
            formula=SCORING_OPP_FORMULA,
            seasons=seasons,
        )
    else:
        frame = (
            pl.read_parquet(args.frame)
            if args.frame
            else rush_expect_training_frame(_load_pbp(seasons))
        )
        result = fit_rush_expect(frame)
        paths = write_fit(
            result,
            args.out_dir,
            "rush_expect",
            formula=RUSH_EXPECT_FORMULA,
            seasons=seasons,
        )
    print(
        f"{args.command}: n={result.n_obs:,} log_loss={result.log_loss:.4f} "
        f"accuracy={result.accuracy:.4f} -> {paths[0]}"
    )
    return 0
