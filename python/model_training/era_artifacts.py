"""Fit + save the side-by-side artifacts for the era-experiment / spread-backfill keepers.

These are written ALONGSIDE the shipped canonical ``.ubj`` files (never overwriting
them) for review before any promotion/publish. Each is a single full-data fit with the
shipped XGBoost params, on the authoritative frame.

Keepers (from ``era_report.md`` — material out-of-fold gains only):

  qbr_era.ubj              one-hot era, spread-backfilled frame   (LOSO RMSE 17.89 -> 17.42)
  fg_era.ubj               one-hot era, canonical frame           (LOSO logloss 0.5258 -> 0.5240)
  fd_model_era.ubj         one-hot era (replaces ordinal), backfilled frame
                                                                  (1st-down cal-MAE 0.0035 -> 0.0027)
  wp_spread_backfilled.ubj shipped 13-feat recipe, backfilled frame
                                                                  (LOSO logloss 0.3616 -> 0.3518; era neutral here)

Run::

    python -m model_training.era_artifacts --artifacts artifacts \
        --backfilled artifacts/pbp_full_spreadfilled.parquet \
        --espn-qbr ../../cfbfastR-cfb-raw/cfb/qbr/espn_qbr.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import fg_matrix, qbr_matrix, wp_matrix
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


def fit_qbr_era(backfilled: pl.DataFrame, espn_qbr: pl.DataFrame, out: Path) -> None:
    X, _, keys = qbr_matrix(backfilled, era_onehot=True)
    j = pl.from_pandas(keys).hstack(pl.from_pandas(X)).join(
        espn_qbr, on=["game_id", "passer_player_name"], how="inner").drop_nulls("raw_qbr")
    feats = [c for c in j.columns if c not in ("game_id", "season", "passer_player_name", "raw_qbr")]
    m = xgb.train(C.QBR_PARAMS, xgb.DMatrix(j.select(feats).to_pandas(), label=j["raw_qbr"].to_numpy()),
                  num_boost_round=C.QBR_NROUNDS)
    _save(m, out / "qbr_era.ubj", model_type="qbr_era", label="raw_qbr", features=feats)


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
    ap = argparse.ArgumentParser(prog="model_training.era_artifacts")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--canonical", default="artifacts/pbp_full.parquet")
    ap.add_argument("--backfilled", default="artifacts/pbp_full_spreadfilled.parquet")
    ap.add_argument("--espn-qbr", default="../../cfbfastR-cfb-raw/cfb/qbr/espn_qbr.parquet")
    ap.add_argument("--only", default="", help="comma list: qbr,fg,fourth_down,wp_spread")
    args = ap.parse_args(argv)
    out = Path(args.artifacts)
    targets = {t.strip() for t in args.only.split(",") if t.strip()} or {"qbr", "fg", "fourth_down", "wp_spread"}

    backfilled = pl.read_parquet(args.backfilled)
    print("fitting side-by-side artifacts (no canonical overwrite):")
    if "fg" in targets:
        fit_fg_era(pl.read_parquet(args.canonical), out)
    if "qbr" in targets:
        espn = pl.read_parquet(args.espn_qbr).select(
            pl.col("game_id").cast(pl.Int64), pl.col("passer_player_name"),
            pl.col("raw_qbr").cast(pl.Float64, strict=False)).drop_nulls()
        fit_qbr_era(backfilled, espn, out)
    if "fourth_down" in targets:
        fit_fd_era(backfilled, out)
    if "wp_spread" in targets:
        fit_wp_spread_backfilled(backfilled, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
