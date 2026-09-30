"""Fit + save the side-by-side artifacts for the era-experiment / spread-backfill keepers.

These are written ALONGSIDE the shipped canonical ``.ubj`` files (never overwriting
them) for review before any promotion/publish. Each is a single full-data fit with the
shipped XGBoost params, on the authoritative frame.

Keepers (from ``era_report.md`` — material out-of-fold gains only):

  (qbr_era.ubj was fitted here too; the QBR model now trains on the served box score
   via ``train-qbr`` -- see ``train_qbr.py``.)
  fg_era.ubj               one-hot era, canonical frame           (LOSO logloss 0.5258 -> 0.5240)
  fd_model_era.ubj         one-hot era (replaces ordinal), backfilled frame
                                                                  (1st-down cal-MAE 0.0035 -> 0.0027)
  wp_spread_backfilled.ubj shipped 13-feat recipe, backfilled frame
                                                                  (LOSO logloss 0.3616 -> 0.3518; era neutral here)

Run::

    python -m cfb_model_build.model_training.era_artifacts --artifacts artifacts \
        --backfilled artifacts/pbp_full_spreadfilled.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import fg_matrix, wp_matrix
from .ingest import add_winner


def _save(model: xgb.Booster, path: Path, *, model_type: str, label: str, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    try:
        from .model_card import write_xgb_model_card
        write_xgb_model_card(path, model_type=model_type, label=label, model=model,
                             features=features)
    except Exception:  # noqa: BLE001 — card is best-effort
        pass
    print(f"  wrote {path.name} ({len(features)} feats: {', '.join(features)})")


def fit_fg_era(canonical: pl.DataFrame, out: Path) -> None:
    X, y, _ = fg_matrix(canonical, era_onehot=True)
    m = xgb.train(C.FG_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=C.FG_NROUNDS)
    _save(m, out / "fg_era.ubj", model_type="fg_era", label="fg_made", features=list(X.columns))


def fit_fd_era(backfilled: pl.DataFrame, out: Path) -> None:
    from .fourth_down.constants import FD_NROUNDS, FD_PARAMS
    from .fourth_down.features import fd_features
    X, y = fd_features(backfilled, era_onehot=True)
    m = xgb.train(FD_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=FD_NROUNDS)
    _save(m, out / "fd_model_era.ubj", model_type="fourth_down_era", label="yards_gained_class",
          features=list(X.columns))


def fit_wp_spread_backfilled(backfilled: pl.DataFrame, out: Path) -> None:
    df = add_winner(backfilled)
    X, y, _ = wp_matrix(df, variant="spread", era_onehot=False)  # era neutral; shipped 13-feat
    m = xgb.train(C.WP_SPREAD_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=C.WP_SPREAD_NROUNDS)
    _save(m, out / "wp_spread_backfilled.ubj", model_type="wp_spread_backfilled", label="win_indicator",
          features=list(X.columns))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cfb_model_build.model_training.era_artifacts")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--canonical", default="artifacts/pbp_full.parquet")
    ap.add_argument("--backfilled", default="artifacts/pbp_full_spreadfilled.parquet")
    ap.add_argument("--only", default="", help="comma list: fg,fourth_down,wp_spread")
    args = ap.parse_args(argv)
    out = Path(args.artifacts)
    targets = {t.strip() for t in args.only.split(",") if t.strip()} or {"fg", "fourth_down", "wp_spread"}

    # Only fourth_down/wp_spread consume the backfilled frame; lazy-load it so
    # `--only fg` (which uses --canonical) doesn't require the backfill parquet.
    backfilled = (
        pl.read_parquet(args.backfilled)
        if targets & {"fourth_down", "wp_spread"}
        else None
    )
    print("fitting side-by-side artifacts (no canonical overwrite):")
    if "fg" in targets:
        fit_fg_era(pl.read_parquet(args.canonical), out)
    if "fourth_down" in targets:
        fit_fd_era(backfilled, out)
    if "wp_spread" in targets:
        fit_wp_spread_backfilled(backfilled, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
