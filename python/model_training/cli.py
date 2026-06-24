"""CLI: ingest | train-ep | train-wp | train-qbr | validate | figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ingest import add_winner, build_training_frame, write_training_frame  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="model_training")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--final-dir", default=".cache/cfb_final")
    i.add_argument("--out", default="pbp_full.parquet")
    i.add_argument("--seasons", nargs="*", type=int)
    i.add_argument("--odds", default=None,
                   help="cfb_line_odds parquet; applies the consensus spread backfill to the frame")
    for name in ("train-ep", "train-wp", "train-qbr", "train-fg", "train-xpass", "train-two-pt"):
        s = sub.add_parser(name)
        s.add_argument("--pbp", default="pbp_full.parquet")
        s.add_argument("--out", required=True)
        if name == "train-wp":
            s.add_argument("--variant", choices=["spread", "naive"], default="spread")
        if name == "train-qbr":
            s.add_argument("--espn-qbr", required=True)
    v = sub.add_parser("validate", help="prediction-parity of a retrained model vs a shipped reference")
    v.add_argument("--model", required=True, help="path to the retrained .ubj")
    v.add_argument("--ref", required=True, help="path to the shipped reference .ubj")
    v.add_argument("--type", required=True, choices=["ep", "wp", "wp_naive", "qbr"],
                   help="feature family used to build the comparison matrix")
    v.add_argument("--pbp", default="pbp_full.parquet", help="feature source frame")
    v.add_argument("--tol", type=float, default=1e-3, help="max abs prediction diff to pass")
    v.add_argument("--sample", type=int, default=0, help="optional row cap for a quick check (0 = all)")
    lo = sub.add_parser("loso", help="leave-one-season-out CV (pooled + per-season metrics)")
    lo.add_argument("--pbp", default="pbp_full.parquet")
    lo.add_argument("--model", required=True,
                    choices=["ep", "wp", "qbr", "fg", "xpass", "two_pt"])
    lo.add_argument("--espn-qbr", help="ESPN QBR reference parquet (required for --model qbr)")
    lo.add_argument("--oof-out", help="optional path to write the out-of-fold predictions parquet")
    f = sub.add_parser("figures")
    f.add_argument("--table", required=True)
    f.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        n = write_training_frame(args.final_dir, args.out, args.seasons, odds_path=args.odds)
        print(f"wrote {n} rows -> {args.out}")
    elif args.cmd in ("train-ep", "train-wp", "train-qbr", "train-fg", "train-xpass",
                      "train-two-pt"):
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
        else:
            from .train_qbr import train_qbr

            model = train_qbr(df, pl.read_parquet(args.espn_qbr))
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
        elif args.cmd == "train-two-pt":
            _mtype, _label = "two_pt", "two_point_success"
        else:
            _mtype, _label = "qbr", "qbr"
        # train-qbr aggregates to per-QB-game rows and inner-joins ESPN QBR,
        # so df.height (raw PBP rows) is misleading for that branch.
        _n_rows = None if args.cmd == "train-qbr" else df.height
        write_xgb_model_card(args.out, model_type=_mtype, label=_label, model=model,
                             n_rows=_n_rows)
        print(f"saved -> {args.out} (+ model_card.json)")
    elif args.cmd == "loso":
        import polars as pl

        from .validate import loso_cv

        if args.model == "qbr" and not args.espn_qbr:
            print("loso --model qbr requires --espn-qbr <reference.parquet>", file=sys.stderr)
            return 2
        df = add_winner(pl.read_parquet(args.pbp))
        espn = None
        if args.espn_qbr:
            espn = pl.read_parquet(args.espn_qbr).select(
                pl.col("game_id").cast(pl.Int64),
                pl.col("passer_player_name"),
                pl.col("raw_qbr").cast(pl.Float64, strict=False),
            ).drop_nulls()
        res = loso_cv(df, args.model, espn_qbr=espn)
        pooled = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in res["pooled"].items())
        print(f"LOSO {args.model} POOLED: {pooled}")
        if args.oof_out and res["oof"].height:
            Path(args.oof_out).parent.mkdir(parents=True, exist_ok=True)
            res["oof"].write_parquet(args.oof_out)
            print(f"wrote out-of-fold predictions -> {args.oof_out}")
    elif args.cmd == "validate":
        import polars as pl
        import xgboost as xgb

        from .features import ep_matrix, qbr_matrix, wp_matrix
        from .validate import prediction_parity

        df = add_winner(pl.read_parquet(args.pbp))
        if args.sample:
            df = df.head(args.sample)
        if args.type == "ep":
            X, _, _ = ep_matrix(df)
        elif args.type in ("wp", "wp_naive"):
            X, _, _ = wp_matrix(df, "naive" if args.type == "wp_naive" else "spread")
        else:  # qbr
            X, _, _ = qbr_matrix(df)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
