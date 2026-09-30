"""CLI: ingest | train-ep | train-wp | train-qbr | validate | figures | export-analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import constants as C
from .ingest import add_winner, build_training_frame, write_training_frame  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="model_training")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--final-dir", default=".cache/cfb_final")
    i.add_argument("--out", default="pbp_full.parquet")
    i.add_argument("--seasons", nargs="*", type=int)
    i.add_argument(
        "--odds",
        default=None,
        help="cfb_line_odds parquet; applies the consensus spread backfill to the frame",
    )
    for name in (
        "train-ep",
        "train-wp",
        "train-fg",
        "train-xpass",
        "train-two-pt",
    ):
        s = sub.add_parser(name)
        s.add_argument("--pbp", default="pbp_full.parquet")
        s.add_argument("--out", required=True)
        if name == "train-wp":
            s.add_argument("--variant", choices=["spread", "naive"], default="spread")
    q = sub.add_parser(
        "train-qbr",
        help="fit xQBR on the served adv_passing features, gate it vs the incumbent (exit 3 = no arm passed)",
    )
    q.add_argument("--data-dir", default="../cfb", help="dataset tree holding adv_passing/ and pbp/ parquet")
    q.add_argument("--labels", default=None, help="ESPN QBR labels (default: the committed models/qbr file)")
    q.add_argument("--incumbent", default=None, help="model to beat (default: sdv-py's bundled qbr_model.ubj)")
    q.add_argument("--out", required=True)
    cq = sub.add_parser("capture-qbr-labels", help="re-capture ESPN game QBR for seasons into the committed labels file")
    cq.add_argument("--seasons", nargs="+", type=int, required=True)
    cq.add_argument("--out", default=None)
    v = sub.add_parser(
        "validate", help="prediction-parity of a retrained model vs a shipped reference"
    )
    v.add_argument("--model", required=True, help="path to the retrained .ubj")
    v.add_argument("--ref", required=True, help="path to the shipped reference .ubj")
    v.add_argument(
        "--type",
        required=True,
        choices=["ep", "wp", "wp_naive"],
        help="feature family used to build the comparison matrix",
    )
    v.add_argument("--pbp", default="pbp_full.parquet", help="feature source frame")
    v.add_argument(
        "--tol", type=float, default=1e-3, help="max abs prediction diff to pass"
    )
    v.add_argument(
        "--sample",
        type=int,
        default=0,
        help="optional row cap for a quick check (0 = all)",
    )
    lo = sub.add_parser(
        "loso", help="leave-one-season-out CV (pooled + per-season metrics)"
    )
    lo.add_argument("--pbp", default="pbp_full.parquet")
    lo.add_argument(
        "--model", required=True, choices=["ep", "wp", "fg", "xpass", "two_pt"]
    )
    lo.add_argument(
        "--oof-out", help="optional path to write the out-of-fold predictions parquet"
    )
    h = sub.add_parser(
        "hpo", help="hyperparameter search (TPE) gated on the shipped metric"
    )
    h.add_argument("--pbp", default="pbp_full.parquet")
    h.add_argument(
        "--model",
        required=True,
        choices=["ep", "wp_spread", "wp_naive", "fg", "xpass", "two_pt"],
    )
    h.add_argument("--trials", type=int, default=60)
    h.add_argument("--folds", type=int, default=4)
    h.add_argument("--seed", type=int, default=17)
    h.add_argument("--out-dir", default="artifacts/hpo")
    f = sub.add_parser("figures")
    f.add_argument("--table", required=True)
    f.add_argument("--out", required=True)
    ea = sub.add_parser(
        "export-analysis",
        help="per-model analysis frames (ids + the exact engineered feature matrix) for docs/models/deepdive.qmd",
    )
    ea.add_argument("--pbp", default="pbp_full.parquet")
    ea.add_argument("--out-dir", default="artifacts/analysis")
    ea.add_argument("--models", nargs="*", default=None, help="subset of: ep wp xpass cp (default all)")
    rc = sub.add_parser("rebuild-cards", help="rewrite model cards from saved boosters")
    rc.add_argument("--artifacts-dir", required=True, help="directory holding the .ubj files")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        n = write_training_frame(
            args.final_dir, args.out, args.seasons, odds_path=args.odds
        )
        print(f"wrote {n} rows -> {args.out}")
    elif args.cmd == "capture-qbr-labels":
        from .qbr_labels import LABELS_PATH, update_labels

        df = update_labels(args.seasons, out=args.out or LABELS_PATH)
        print(f"labels: {df.height} rows -> {args.out or LABELS_PATH}")
    elif args.cmd == "train-qbr":
        from .qbr_labels import LABELS_PATH
        from .train_qbr import train_qbr

        incumbent = args.incumbent
        if incumbent is None:
            import sportsdataverse

            incumbent = Path(sportsdataverse.__file__).parent / "cfb" / "models" / "qbr_model.ubj"
        return train_qbr(args.data_dir, args.labels or LABELS_PATH, incumbent, args.out)
    elif args.cmd in (
        "train-ep",
        "train-wp",
        "train-fg",
        "train-xpass",
        "train-two-pt",
    ):
        import polars as pl

        df = add_winner(pl.read_parquet(args.pbp))
        if args.cmd == "train-ep":
            from .train_ep import train_ep

            model = train_ep(df)
        elif args.cmd == "train-wp":
            from .train_wp import train_wp

            model = train_wp(df, variant=args.variant, stage=args.stage)
        elif args.cmd == "train-fg":
            from .train_fg import train_fg

            model = train_fg(df)
        elif args.cmd == "train-xpass":
            from .train_xpass import train_xpass

            model = train_xpass(df)
        elif args.cmd == "train-two-pt":
            from .train_two_pt import train_two_pt

            model = train_two_pt(df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(args.out)
        from .model_card import write_xgb_model_card

        # Resolve per-cmd metadata without eagerly evaluating args.variant
        # (only train-wp defines --variant).
        if args.cmd == "train-ep":
            _mtype, _label = "ep", "next_score_label"
        elif args.cmd == "train-wp":
            _mtype, _label = f"wp_{args.variant}", "label"
        elif args.cmd == "train-fg":
            _mtype, _label = "fg", "fg_made"
        elif args.cmd == "train-xpass":
            _mtype, _label = "xpass", "is_pass"
        else:
            _mtype, _label = "two_pt", "two_point_success"
        _n_rows = df.height
        # Hyperparameters and training seasons were being dropped, leaving cards
        # with `hyperparameters: None` / `training_seasons: None` -- i.e. no way to
        # answer "what was this trained on" without reading the code. Record the
        # recipe, the season span, and the corpus the frame came from.
        _params = {
            "ep": C.EP_PARAMS, "fg": C.FG_PARAMS, "xpass": C.XPASS_PARAMS,
            "two_pt": C.TWO_PT_PARAMS,
            "wp_spread": C.WP_SPREAD_PARAMS, "wp_naive": C.WP_NAIVE_PARAMS,
        }.get(_mtype)
        _rounds = {
            "ep": C.EP_NROUNDS, "fg": C.FG_NROUNDS, "xpass": C.XPASS_NROUNDS,
            "two_pt": C.TWO_PT_NROUNDS,
            "wp_spread": C.WP_SPREAD_NROUNDS, "wp_naive": C.WP_NAIVE_NROUNDS,
        }.get(_mtype)
        _seasons = (
            sorted(int(s) for s in df["season"].unique().to_list())
            if "season" in df.columns else None
        )
        write_xgb_model_card(
            args.out, model_type=_mtype, label=_label, model=model, n_rows=_n_rows,
            hyperparams=dict(_params or {}),
            seasons=_seasons,
            extra={"num_boost_round": _rounds, "training_frame": str(args.pbp)},
        )
        print(f"saved -> {args.out} (+ model_card.json)")
    elif args.cmd == "hpo":
        import polars as pl

        from .hpo import search

        # add_winner is what supplies the WP label; the loso path applies it too,
        # so search and confirm see identical frames.
        df = add_winner(pl.read_parquet(args.pbp))
        search(
            df,
            args.model,
            n_trials=args.trials,
            k_folds=args.folds,
            seed=args.seed,
            out_dir=Path(args.out_dir),
        )
    elif args.cmd == "loso":
        import polars as pl

        from .validate import loso_cv

        df = add_winner(pl.read_parquet(args.pbp))
        res = loso_cv(df, args.model)
        pooled = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res["pooled"].items()
        )
        print(f"LOSO {args.model} POOLED: {pooled}")
        if args.oof_out and res["oof"].height:
            Path(args.oof_out).parent.mkdir(parents=True, exist_ok=True)
            res["oof"].write_parquet(args.oof_out)
            print(f"wrote out-of-fold predictions -> {args.oof_out}")
    elif args.cmd == "validate":
        import polars as pl
        import xgboost as xgb

        from .features import ep_matrix, wp_matrix
        from .validate import prediction_parity

        df = add_winner(pl.read_parquet(args.pbp))
        if args.sample:
            df = df.head(args.sample)
        if args.type == "ep":
            X, _, _ = ep_matrix(df)
        else:
            X, _, _ = wp_matrix(df, "naive" if args.type == "wp_naive" else "spread")
        new = xgb.Booster()
        new.load_model(args.model)
        ref = xgb.Booster()
        ref.load_model(args.ref)
        rep = prediction_parity(new, ref, X, tol=args.tol)
        verdict = "PASS" if rep["within_tol"] else "OUT-OF-TOL"
        print(
            f"validate {args.type}: max_abs_diff={rep['max_abs_diff']:.6f} "
            f"tol={rep['tol']:g} n={len(X)} -> {verdict}",
        )
        return 0 if rep["within_tol"] else 1
    elif args.cmd == "figures":
        print(
            "figures: CLI wiring not yet implemented — "
            "use the model_training.figures library API directly.",
            file=sys.stderr,
        )
        return 2
    elif args.cmd == "export-analysis":
        from .analysis import MODELS, export_analysis_frames

        models = tuple(args.models) if args.models else MODELS
        unknown = set(models) - set(MODELS)
        if unknown:
            print(f"export-analysis: unknown models {sorted(unknown)}", file=sys.stderr)
            return 2
        rows = export_analysis_frames(args.pbp, args.out_dir, models)
        for name, n in rows.items():
            print(f"analysis_{name}.parquet: {n} rows")
        print(f"wrote {len(rows)} frames + analysis_manifest.json -> {args.out_dir}")
    elif args.cmd == "rebuild-cards":
        from .rebuild_cards import rebuild_all

        for p in rebuild_all(args.artifacts_dir):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
