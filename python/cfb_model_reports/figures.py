"""Figure generation for model reports — derives calibration frames from LOSO OOF
parquets and renders PNGs via the bespoke cfbfastR plotnine styling in
``model_training.figures`` / ``model_training.fourth_down.figures``.

Every builder is defensive: a missing parquet / model / optional dependency
(plotnine) returns ``[]`` so the report still renders (with a note) instead of
crashing the CLI. Only figures that were actually written are returned, as
repo-relative paths (``figures/<name>.png``) for embedding in the Markdown /
Quarto reports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# WP win-prob facet labels by quarter (period column on the OOF source).
_QUARTER_LABEL = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


def _rel(png: Path, out: Path) -> str:
    """Return a repo-relative ``figures/<name>.png`` path for embedding."""
    try:
        return png.relative_to(out).as_posix()
    except ValueError:
        return f"figures/{png.name}"


def derive_quarter_aligned(pbp_parquet, oof_parquet):
    """Per-row quarter array aligned to the season-sorted WP OOF row order.

    The WP OOF was built by ``pl.concat`` over ``sorted(seasons)``; reproducing
    that ordering over ``pbp_full`` recovers the per-play period in the same
    order. Returns ``None`` if files/columns/lengths don't line up.
    """
    pp, op = Path(pbp_parquet), Path(oof_parquet)
    if not (pp.exists() and op.exists()):
        return None
    try:
        import model_training.constants as C
    except Exception:
        return None
    df = pl.read_parquet(pp)
    oof = pl.read_parquet(op)
    period_src = C.WP_SOURCE.get("period", "period")
    if "season" not in df.columns or period_src not in df.columns:
        return None
    seasons = sorted(df["season"].unique().to_list())
    df_ord = pl.concat([df.filter(pl.col("season") == s) for s in seasons])
    if df_ord.height != oof.height:
        return None
    return df_ord[period_src].to_numpy()


def build_wp_calibration(oof_parquet, fig_dir: Path, out: Path, *, stem: str,
                         title: str, subtitle: str, by_quarter=None) -> list[str]:
    """Render a WP calibration figure (predicted bin vs observed rate, faceted).

    Args:
        oof_parquet: parquet with columns ``season``, ``y``, ``wp_pred``.
        fig_dir: ``<out>/figures`` directory to write into.
        out: report output root (for repo-relative pathing).
        stem: figure file stem (no extension).
        title / subtitle: plot text.
        by_quarter: optional per-row quarter array (same length as the OOF) used
            to facet the calibration by quarter; falls back to a single panel.

    Returns:
        ``["figures/<stem>.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
        from model_training.validate import calibration_table, weighted_cal_error
    except Exception:
        return []
    df = pl.read_parquet(p)
    pred = df["wp_pred"].to_numpy()
    y = df["y"].to_numpy().astype(int)
    if by_quarter is not None and len(by_quarter) == len(pred):
        by = [_QUARTER_LABEL.get(int(q), "other") for q in by_quarter]
    else:
        by = ["all"] * len(pred)
    tab = calibration_table(pred.tolist(), y.tolist(), by, bin_size=0.05)
    cal_err = round(weighted_cal_error(tab), 4)
    png, _ = write_calibration(tab, fig_dir / stem, title=title,
                               subtitle=subtitle, cal_error=cal_err)
    return [_rel(Path(png), out)]


def build_ep_calibration(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render an EP calibration figure: predicted EP bin vs realized EP.

    EP is a regression-style summary of a multiclass head (sum of class probs x
    point values). We bin the predicted EP and plot mean realized next-score
    value per bin against the bin centre (y=x is perfect calibration). A single
    facet (``by="EP"``) is used.

    Args:
        oof_parquet: parquet with columns ``ep_pred`` (predicted EP) and
            ``realized`` (realized next-score points).

    Returns:
        ``["figures/ep_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
    except Exception:
        return []
    df = pl.read_parquet(p)
    ep_pred = df["ep_pred"].to_numpy()
    realized = df["realized"].to_numpy()
    bin_size = 0.5
    b = np.round(ep_pred / bin_size) * bin_size
    tab = (
        pl.DataFrame({"bin": b, "realized": realized})
        .group_by("bin")
        .agg(n_plays=pl.len(), actual=pl.col("realized").mean())
        .with_columns(by=pl.lit("EP"))
        .sort("bin")
    )
    # weighted calibration MAE in points (matches the loso eval recipe scale)
    cal_err = float(
        (tab["bin"] - tab["actual"]).abs().dot(tab["n_plays"]) / tab["n_plays"].sum()
    )
    png, _ = write_calibration(
        tab.select(["by", "bin", "n_plays", "actual"]),
        fig_dir / "ep_calibration",
        title="Expected Points — LOSO Calibration",
        subtitle="Predicted EP (points) vs Realized next-score value",
        cal_error=round(cal_err, 4),
    )
    return [_rel(Path(png), out)]


def build_qbr_scatter(oof_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render a QBR predicted-vs-actual hex/scatter calibration figure.

    Args:
        oof_parquet: parquet with columns ``y`` (ESPN raw QBR) and ``qbr_pred``.

    Returns:
        ``["figures/qbr_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(oof_parquet)
    if not p.exists():
        return []
    try:
        from plotnine import (
            aes,
            coord_fixed,
            element_rect,
            element_text,
            geom_abline,
            geom_bin2d,
            geom_smooth,
            ggplot,
            labs,
            theme,
            theme_bw,
        )
    except Exception:
        return []
    df = pl.read_parquet(p).to_pandas()
    garnet, grey95, grey99 = "#500f1b", "#f2f2f2", "#fcfcfc"
    font = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]
    smoother = "lm"
    try:
        import skmisc  # noqa: F401

        smoother = "loess"
    except ModuleNotFoundError:
        pass
    gg = (
        ggplot(df, aes("qbr_pred", "y"))
        + geom_bin2d(bins=40)
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_smooth(method=smoother, se=False, color=garnet, size=0.6)
        + coord_fixed(xlim=[0, 100], ylim=[0, 100])
        + labs(
            title="QBR — LOSO Predicted vs ESPN QBR",
            subtitle="Out-of-fold pooled across 2005-2025",
            x="Predicted QBR",
            y="ESPN raw QBR",
        )
        + theme_bw()
        + theme(
            text=element_text(family=font),
            plot_background=element_rect(fill=grey99, color="black"),
            panel_background=element_rect(fill=grey95),
            legend_position="bottom",
        )
    )
    png = fig_dir / "qbr_calibration.png"
    gg.save(str(png), width=6, height=5, dpi=200, verbose=False)
    return [_rel(png, out)]


def build_rb_calibration(loso_parquet, fig_dir: Path, out: Path) -> list[str]:
    """Render the xREPA RB-eval calibration figure from the LOSO parquet.

    Args:
        loso_parquet: ``xrepa_loso.parquet`` with columns ``exp_rb_epa`` and
            ``target``.

    Returns:
        ``["figures/rb_eval_calibration.png"]`` if written, else ``[]``.
    """
    p = Path(loso_parquet)
    if not p.exists():
        return []
    try:
        from model_training.figures import write_calibration
        from rb_eval.validate import calibration_table, weighted_cal_error
    except Exception:
        return []
    cv = pl.read_parquet(p)
    tbl = calibration_table(cv)
    cal_err = round(weighted_cal_error(tbl), 4)
    # remap rb_eval's column names to the write_calibration contract
    frame = tbl.select(
        by=pl.lit("xREPA"),
        bin=pl.col("bin_pred_epa"),
        n_plays=pl.col("total_instances"),
        actual=pl.col("bin_actual_epa"),
    ).sort("bin")
    png, _ = write_calibration(
        frame,
        fig_dir / "rb_eval_calibration",
        title="RB Evaluation (xREPA) — LOSO Calibration",
        subtitle="Predicted EPA/play vs Realized unadjusted EPA",
        cal_error=cal_err,
    )
    return [_rel(Path(png), out)]


def build_fd_figures(model_path, pbp_parquet, fig_dir: Path, out: Path,
                     *, sample_rows: int = 200_000) -> list[str]:
    """Render fourth-down first-down calibration + feature-importance figures.

    Uses ``model_training.fourth_down.validate.calibration_fd`` over a sample of
    ``pbp_full.parquet`` (full 2.2M is unnecessary for a calibration figure).

    Args:
        model_path: ``fd_model.ubj`` path.
        pbp_parquet: ``pbp_full.parquet`` path (must carry the 6 FD features +
            ``statYardage``/yards-gained label).
        sample_rows: cap on plays sampled for the figure.

    Returns:
        ``["figures/fd_calibration.png", "figures/fd_feature_importance.png"]``
        for whichever were written, else ``[]``.
    """
    mp, pp = Path(model_path), Path(pbp_parquet)
    if not mp.exists() or not pp.exists():
        return []
    try:
        import pandas as pd
        import xgboost as xgb

        from model_training.fourth_down.constants import FD_FEATURES, FD_LABEL_OFFSET
        from model_training.fourth_down.features import fd_features
        from model_training.fourth_down.figures import write_fd_figures
        from model_training.fourth_down.validate import calibration_fd
    except Exception:
        return []
    booster = xgb.Booster()
    booster.load_model(str(mp))
    df = pl.read_parquet(pp)
    try:
        X, y_label = fd_features(df)
    except Exception:
        return []
    if len(X) == 0:
        return []
    Xp = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FD_FEATURES)
    # calibration_fd wants raw yards gained; fd_features emits class labels (yards + offset).
    yv = np.asarray(y_label).astype(int) - FD_LABEL_OFFSET
    if len(Xp) > sample_rows:
        rng = np.random.default_rng(20260622)
        idx = rng.choice(len(Xp), size=sample_rows, replace=False)
        Xp = Xp.iloc[idx].reset_index(drop=True)
        yv = yv[idx]
    cal = calibration_fd(booster, Xp, yv, n_bins=10)
    cal_err = float(
        (cal["pred_fd_prob"] - cal["empirical_fd_rate"]).abs().mul(cal["n_plays"]).sum()
        / cal["n_plays"].sum()
    )
    score = booster.get_score(importance_type="gain")
    imp = pd.DataFrame(
        {"Feature": list(score.keys()), "Gain": list(score.values())}
    ).sort_values("Gain", ascending=False)
    cal_png, imp_png = write_fd_figures(cal, imp, fig_dir, cal_error=cal_err)
    return [_rel(Path(cal_png), out), _rel(Path(imp_png), out)]
