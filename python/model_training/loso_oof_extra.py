"""Augmented leave-one-season-out (LOSO) out-of-fold writers for calibration plots.

The shipped ``loso_*_oof.parquet`` artifacts carry only the scalars the pooled
metrics need (EP: ``ep_pred``/``realized``; WP: ``wp_pred``). The cfbscrapR /
nflfastR signature calibration figures need *more* per-play columns:

* **EP** — the cfbscrapR ``02-EPA-Model.R`` calibration facets by
  ``next_score_type`` (the 7 score classes) and plots, **per class**, the binned
  predicted class probability against the empirical rate of that class. That
  needs the full 7-vector of softprob predictions per play, which the scalar
  ``ep_pred`` collapses away. We re-run EP LOSO and persist ``p0..p6`` + the
  actual class ``y`` + ``season``.
* **WP (spread + naive)** — cfbscrapR ``03-WPA-Model.R`` facets by ``period``
  (quarter). We persist ``wp_pred`` + win label ``y`` + ``period`` per play so
  the figure can facet honestly out-of-sample (the naive variant previously had
  no LOSO parquet at all — only an in-sample fallback).

This is the honest out-of-sample path: for each held-out season we train on
every *other* season with the shipped params/rounds unchanged, then predict the
held-out season. Outputs are cached under ``artifacts/`` so the report figures
are cheap to regenerate.

Run::

    python -m model_training.loso_oof_extra --artifacts artifacts \
        --targets ep,wp_spread,wp_naive
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import ep_matrix, wp_matrix
from .train_ep import train_ep
from .train_wp import train_wp


def _period_array(te: pl.DataFrame) -> np.ndarray:
    src = C.WP_SOURCE.get("period", "period")
    col = src if src in te.columns else ("period" if "period" in te.columns else None)
    if col is None:
        return np.zeros(te.height, dtype=np.int32)
    return te[col].to_numpy()


def write_ep_class_oof(df: pl.DataFrame, out: Path, log=print) -> Path:
    """LOSO EP with the full 7-class softprob vector persisted per play.

    Output columns: ``season``, ``y`` (actual class 0..6), ``p0``..``p6``.
    """
    seasons = sorted(df["season"].unique().to_list())
    frames: list[pl.DataFrame] = []
    for s in seasons:
        t0 = time.time()
        tr = df.filter(pl.col("season") != s)
        te = df.filter(pl.col("season") == s)
        m = train_ep(tr)
        X, y, _ = ep_matrix(te)
        y = y.astype(int)
        P = m.predict(xgb.DMatrix(X))  # (n, 7)
        d = {"season": np.full(len(y), s, dtype=np.int32), "y": y}
        for k in range(7):
            d[f"p{k}"] = P[:, k]
        frames.append(pl.DataFrame(d))
        log(f"[ep-class] fold {s}: n={len(y)} ({time.time() - t0:.0f}s)")
    oof = pl.concat(frames)
    oof.write_parquet(out)
    log(f"[ep-class] wrote {out} ({oof.height} rows)")
    return out


def write_wp_period_oof(df: pl.DataFrame, out: Path, *, variant: str, log=print) -> Path:
    """LOSO WP (``spread`` or ``naive``) with ``period`` persisted per play.

    Output columns: ``season``, ``y`` (win indicator), ``wp_pred``, ``period``.
    """
    seasons = sorted(df["season"].unique().to_list())
    frames: list[pl.DataFrame] = []
    for s in seasons:
        t0 = time.time()
        tr = df.filter(pl.col("season") != s)
        te = df.filter(pl.col("season") == s)
        m = train_wp(tr, variant=variant, stage=2)
        X, y, _ = wp_matrix(te, variant)
        y = y.astype(int)
        p = m.predict(xgb.DMatrix(X))
        frames.append(
            pl.DataFrame(
                {
                    "season": np.full(len(y), s, dtype=np.int32),
                    "y": y,
                    "wp_pred": p,
                    "period": _period_array(te).astype(np.int32),
                }
            )
        )
        log(f"[wp-{variant}] fold {s}: n={len(y)} ({time.time() - t0:.0f}s)")
    oof = pl.concat(frames)
    oof.write_parquet(out)
    log(f"[wp-{variant}] wrote {out} ({oof.height} rows)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="model_training.loso_oof_extra")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--targets", default="ep,wp_spread,wp_naive",
                    help="comma list of {ep,wp_spread,wp_naive}")
    args = ap.parse_args(argv)
    art = Path(args.artifacts)
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    df = pl.read_parquet(art / "pbp_full.parquet")
    if "ep" in targets:
        write_ep_class_oof(df, art / "loso_ep_class_oof.parquet")
    if {"wp_spread", "wp_naive"} & targets:
        # The WP label is the game winner — add_winner() must run before train_wp /
        # wp_matrix (the EP path uses the pre-computed next_score label, so it
        # doesn't need this).
        from .ingest import add_winner

        df = add_winner(df)
    if "wp_spread" in targets:
        write_wp_period_oof(df, art / "loso_wp_spread_period_oof.parquet", variant="spread")
    if "wp_naive" in targets:
        write_wp_period_oof(df, art / "loso_wp_naive_oof.parquet", variant="naive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
