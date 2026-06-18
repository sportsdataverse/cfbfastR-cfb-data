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
    for name in ("train-ep", "train-wp", "train-qbr"):
        s = sub.add_parser(name)
        s.add_argument("--pbp", default="pbp_full.parquet")
        s.add_argument("--out", required=True)
        if name == "train-wp":
            s.add_argument("--variant", choices=["spread", "naive"], default="spread")
        if name == "train-qbr":
            s.add_argument("--espn-qbr", required=True)
    v = sub.add_parser("validate", help="prediction-parity of a candidate model vs a reference .ubj")
    v.add_argument("--model-type", required=True, choices=["ep", "wp", "qbr"])
    v.add_argument("--model", required=True, help="candidate model .ubj")
    v.add_argument("--ref", required=True, help="reference model .ubj")
    v.add_argument("--pbp", default="pbp_full.parquet")
    v.add_argument("--tol", type=float, default=1e-3)
    lo = sub.add_parser("loso", help="leave-one-season-out CV (pooled + per-season metrics)")
    lo.add_argument("--pbp", default="pbp_full.parquet")
    lo.add_argument("--model", required=True, choices=["ep", "wp", "qbr"])
    lo.add_argument("--espn-qbr", help="ESPN QBR reference parquet (required for --model qbr)")
    lo.add_argument("--oof-out", help="optional path to write the out-of-fold predictions parquet")
    f = sub.add_parser("figures", help="render a calibration plot from a loso OOF parquet")
    f.add_argument("--oof", required=True, help="out-of-fold predictions parquet (from `loso --oof-out`)")
    f.add_argument("--out", required=True, help="output path stem (writes <stem>.png/.csv/.parquet)")
    f.add_argument("--pred-col", default=None, help="prediction column (default: auto-detect wp_pred)")
    f.add_argument("--title", default="Calibration")
    f.add_argument("--subtitle", default="LOSO")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        n = write_training_frame(args.final_dir, args.out, args.seasons)
        print(f"wrote {n} rows -> {args.out}")
    elif args.cmd in ("train-ep", "train-wp", "train-qbr"):
        import polars as pl

        df = add_winner(pl.read_parquet(args.pbp))
        if args.cmd == "train-ep":
            from .train_ep import train_ep

            model = train_ep(df)
        elif args.cmd == "train-wp":
            from .train_wp import train_wp

            model = train_wp(df, variant=args.variant, stage=args.stage)
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
        if args.model_type == "ep":
            X, _, _ = ep_matrix(df)
        elif args.model_type == "wp":
            X, _, _ = wp_matrix(df, "spread")
        else:
            X, _, _ = qbr_matrix(df)
        cand, ref = xgb.Booster(), xgb.Booster()
        cand.load_model(args.model)
        ref.load_model(args.ref)
        res = prediction_parity(cand, ref, X, tol=args.tol)
        print(f"validate {args.model_type}: max_abs_diff={res['max_abs_diff']:.3e} "
              f"within_tol={res['within_tol']} (tol={res['tol']})")
        return 0 if res["within_tol"] else 1
    elif args.cmd == "figures":
        import polars as pl

        from .figures import write_calibration
        from .validate import calibration_table, weighted_cal_error

        oof = pl.read_parquet(args.oof)
        pred_col = args.pred_col or next((c for c in ("wp_pred", "pred") if c in oof.columns), None)
        if pred_col is None or "y" not in oof.columns:
            print("figures: OOF parquet needs a 'y' column and a prediction column "
                  "(wp_pred/pred); EP-value calibration is not a probability plot.", file=sys.stderr)
            return 2
        by = oof["season"].to_list() if "season" in oof.columns else ["all"] * oof.height
        tab = calibration_table(oof[pred_col].to_list(), oof["y"].to_list(), by, bin_size=0.05)
        png, csv = write_calibration(tab, args.out, title=args.title,
                                     subtitle=args.subtitle, cal_error=weighted_cal_error(tab))
        print(f"wrote {png} (+ {csv} + .parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
