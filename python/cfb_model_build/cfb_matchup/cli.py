"""CLI for model family 35 (``cfb_matchup``).

    python -m cfb_model_build.cfb_matchup train-scoring-opp --seasons 2014-2024 --out-dir python/.cache/matchup/candidate
    python -m cfb_model_build.cfb_matchup train-rush-expect --seasons 2014-2024 --out-dir python/.cache/matchup/candidate
    python -m cfb_model_build.cfb_matchup train-wepa-weights --seasons 2014-2024 --holdout 2025 --n 500 --seed 4 --out-dir python/.cache/matchup/candidate
    python -m cfb_model_build.cfb_matchup score-wepa --seasons 2025 --weights python/cfb_model_build/cfb_matchup/artifacts/wepa_weights.json

Training rows come from the ``cfbfastR_cfb_pbp`` release (sdv-py
``load_cfb_pbp_r``), one season at a time, or from ``--frame`` (a parquet
already in the trainer's shape) for offline / oracle runs. The glm commands
write ``<stem>_coef.json`` (the applier's input) and ``<stem>_meta.json``.
``train-wepa-weights`` runs the random weight search (scored by the adjusted
R² of home margin on the four as-of WEPA means, as the source did) over the
given seasons, reports every candidate's held-out score on ``--holdout``, and
writes ``wepa_weights.json`` + meta for the best in-sample candidate; the
games come from CFBD ``/games`` (needs ``CFBD_API_KEY``).
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
from cfb_model_build.cfb_matchup.wepa import (
    load_search_table,
    random_candidates,
    score_weights,
    search,
    tag_for_search,
    weights_of,
    write_weights,
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
    Population: every drive of the release, as the source's trainer
    (``tools/scoring_opp_rate.R``) builds it -- it applies no ``ppa`` filter;
    the per-team APPLY path filters null ``ppa`` because it aggregates EPA,
    which the fit does not read.
    """
    from cfb_data_build.matchup_features import GlmCoefficients

    zero = GlmCoefficients(0.0, {}, list(SCORING_OPP_FEATURES), [], "binomial", "logit")
    return build_drive_frame(pbp, zero).drop("scoring_opp_prediction", "scoring_opp_oe")


def _games_for_scoring(seasons: list[int]) -> pl.DataFrame:
    """CFBD games with final scores for the scorer (display spellings, Date)."""
    from cfb_data_build.matchup_build import fetch_cfbd_games, tidy_cfbd_games
    from cfb_data_build.matchup_features import to_display_names

    frames = [tidy_cfbd_games(fetch_cfbd_games(s)) for s in seasons]
    games = to_display_names(pl.concat(frames), "home_team", "away_team")
    return games.with_columns(
        start_date=pl.col("start_date").str.slice(0, 10).str.to_date()
    ).filter(pl.col("home_points").is_not_null() & pl.col("away_points").is_not_null())


def _games_with_plays(games: pl.DataFrame, tagged: pl.DataFrame) -> pl.DataFrame:
    """The scorer's corpus: games that have plays in the release (the oracle's definition)."""
    return games.filter(pl.col("game_id").is_in(tagged["game_id"].unique().to_list()))


def _holdout_inputs(seasons: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    tagged = _tagged_for_scoring(seasons)
    return tagged, _games_with_plays(_games_for_scoring(seasons), tagged)


def _tagged_for_scoring(seasons: list[int]) -> pl.DataFrame:
    from cfb_data_build.matchup_features import TEAM_COLUMNS, to_display_names

    pbp = _load_pbp(seasons)
    return tag_for_search(
        to_display_names(pbp, *[c for c in TEAM_COLUMNS if c in pbp.columns])
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfb_matchup", description=__doc__)
    # the generic stage driver (scripts/cfb_models.sh) runs every shim with no
    # arguments; that must not exit 2 here, so the default is a read-only
    # "status" of the bundle -- training stays an explicit subcommand
    sub = p.add_subparsers(dest="command", required=False)
    sub.add_parser("status", help="verify + print the bundled artifacts (default)")
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
    sw = sub.add_parser("train-wepa-weights")
    sw.add_argument("--seasons", default="2014-2024", help="search corpus seasons")
    sw.add_argument(
        "--holdout", default="2025", help="seasons scored out of sample, never searched"
    )
    sw.add_argument(
        "--n", type=int, default=500, help="random candidates (the source used 500)"
    )
    sw.add_argument("--seed", type=int, default=4, help="the source's set.seed(4)")
    sw.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="score THIS table of weight vectors instead of drawing",
    )
    sw.add_argument("--out-dir", type=Path, required=True)
    sw.add_argument("--promote", action="store_true")
    sc = sub.add_parser("score-wepa")
    sc.add_argument("--seasons", default="2025")
    sc.add_argument(
        "--weights", type=Path, required=True, help="a wepa_weights.json to score"
    )
    return p


def _status() -> int:
    """Load every bundled artifact through its applier and print the meta (exit 0)."""
    import json

    from cfb_data_build.matchup_features import load_coefficients, load_wepa_weights

    bundle = Path(__file__).resolve().parent / "artifacts"
    for stem in ("scoring_opp", "rush_expect"):
        coef = load_coefficients(bundle / f"{stem}_coef.json")
        meta = json.loads((bundle / f"{stem}_meta.json").read_text(encoding="utf-8"))
        print(
            f"{stem}: {len(coef.coefficients)} active terms, aliased={coef.aliased}, n_obs={meta.get('n_obs')}"
        )
    weights = load_wepa_weights(bundle / "wepa_weights.json")
    meta = json.loads((bundle / "wepa_weights_meta.json").read_text(encoding="utf-8"))
    print(
        f"wepa_weights: {len(weights)} weights, holdout_2025 adj_r2={meta.get('adj_r2_holdout_2025')}"
    )
    return 0


def _run_wepa(args, seasons: list[int]) -> int:
    from cfb_data_build.matchup_features import load_wepa_weights

    if args.command == "score-wepa":
        tagged = _tagged_for_scoring(seasons)
        s = score_weights(
            tagged,
            _games_with_plays(_games_for_scoring(seasons), tagged),
            load_wepa_weights(args.weights),
        )
        print(
            f"score-wepa {args.seasons}: adj_r2={s.adj_r2:.4f} sd_err={s.sigma:.3f} n_games={s.n_games}"
        )
        return 0
    holdout_seasons = _seasons(args.holdout) if args.holdout else []
    if set(holdout_seasons) & set(seasons):
        raise SystemExit(
            "--holdout overlaps --seasons; the held-out score would be in-sample"
        )
    candidates = (
        load_search_table(args.candidates)
        if args.candidates
        else random_candidates(args.n, seed=args.seed)
    )
    if "candidate" not in candidates.columns:
        candidates = candidates.with_row_index("candidate")
    tagged = _tagged_for_scoring(seasons)
    games = _games_with_plays(_games_for_scoring(seasons), tagged)
    holdout = _holdout_inputs(holdout_seasons) if holdout_seasons else None
    table = search(tagged, games, candidates, holdout=holdout)
    best = table.sort("adj_r2", descending=True).row(0, named=True)
    meta = {
        "model": "wepa_weights",
        "search_seasons": seasons,
        "holdout_seasons": holdout_seasons,
        "n_candidates": table.height,
        "seed": args.seed,
        "selected_by": "max in-sample adj_r2 (the source's rule)",
        "adj_r2_in_sample": best["adj_r2"],
        "sd_err_in_sample": best["sd_err"],
        "adj_r2_holdout": best.get("holdout_adj_r2"),
        "fitted_by": "cfb_matchup train-wepa-weights",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out_dir / "wepa_search.csv")
    wp, _ = write_weights(weights_of(best), args.out_dir, meta=meta)
    print(
        f"train-wepa-weights: {table.height} candidates on {args.seasons}; best adj_r2={best['adj_r2']:.4f}"
        f" holdout={best.get('holdout_adj_r2')} -> {wp}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "status"):
        return _status()
    seasons = _seasons(args.seasons)
    bundle = Path(__file__).resolve().parent / "artifacts"
    out_dir = getattr(args, "out_dir", None)
    if out_dir is not None and out_dir.resolve() == bundle and not args.promote:
        raise SystemExit(
            "refusing to overwrite the bundled artifacts without --promote: fit into a "
            "candidate dir, run tests/cfb_matchup (parity + level gates), then promote"
        )
    if args.command in ("train-wepa-weights", "score-wepa"):
        return _run_wepa(args, seasons)
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
            frame=frame,
            source=str(args.frame)
            if args.frame
            else f"cfbfastR_cfb_pbp {args.seasons}",
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
            frame=frame,
            source=str(args.frame)
            if args.frame
            else f"cfbfastR_cfb_pbp {args.seasons}",
        )
    print(
        f"{args.command}: n={result.n_obs:,} log_loss={result.log_loss:.4f} "
        f"accuracy={result.accuracy:.4f} -> {paths[0]}"
    )
    return 0
